# cs336_systems/flash_attention/trition_versions/forward_backward_version2.py
import math

import torch
import triton
import triton.language as tl


def _pick_block_d(D: int) -> int:
    if D <= 16:
        return 16
    if D <= 32:
        return 32
    if D <= 64:
        return 64
    if D <= 128:
        return 128
    raise ValueError(f"Unsupported head dim D={D}")


# ============================================================
# Forward (same algorithm, Triton)
# ============================================================
@triton.jit
def _flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,  # LSE
    stride_qb: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lm: tl.constexpr,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # batch*head
    pid_m = tl.program_id(1)  # query block

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BM,)
    offs_n = tl.arange(0, BLOCK_N)  # (BN,)
    offs_d = tl.arange(0, BLOCK_D)  # (BD,)

    # Q: (BM, D)
    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    # online softmax
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in tl.static_range(0, N_K, BLOCK_N):
        k_ids = start_n + offs_n  # (BN,)

        # K: (BN, D)
        k_ptrs = K_ptr + pid_bh * stride_kb + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # scores: (BM, BN)
        scores = tl.dot(q, tl.trans(k)) * SCALE

        # mask invalid keys
        scores = tl.where(k_ids[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            q_ids = offs_m
            causal = q_ids[:, None] >= k_ids[None, :]
            scores = tl.where(causal, scores, -float("inf"))

            # Triton no-break: if key-block is entirely future, mask it out
            active = start_n < (pid_m + 1) * BLOCK_M
            scores = tl.where(active, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])

        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p, axis=1)

        # V: (BN, D)
        v_ptrs = V_ptr + pid_bh * stride_vb + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_ij
        l_i = l_ij

    o = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    o_ptrs = O_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D))

    l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
    tl.store(l_ptrs, lse, mask=(offs_m < N_Q))


def _flash_forward_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3

    B_eff, N_q, D = q.shape
    N_k = k.shape[1]

    O = torch.empty((B_eff, N_q, D), device=q.device, dtype=torch.float32)
    L = torch.empty((B_eff, N_q), device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = _pick_block_d(D)
    scale = 1.0 / math.sqrt(D)

    grid = (B_eff, triton.cdiv(N_q, BLOCK_M))
    _flash_fwd_kernel[grid](
        q,
        k,
        v,
        O,
        L,
        stride_qb=q.stride(0),
        stride_qm=q.stride(1),
        stride_qd=q.stride(2),
        stride_kb=k.stride(0),
        stride_kn=k.stride(1),
        stride_kd=k.stride(2),
        stride_vb=v.stride(0),
        stride_vn=v.stride(1),
        stride_vd=v.stride(2),
        stride_ob=O.stride(0),
        stride_om=O.stride(1),
        stride_od=O.stride(2),
        stride_lb=L.stride(0),
        stride_lm=L.stride(1),
        N_Q=N_q,
        N_K=N_k,
        D=D,
        SCALE=scale,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return O, L


# ============================================================
# Backward (BIG CHANGE): Fused tile backward (Q-block primary)
# - Computes dQ (no atomic)
# - Atomically accumulates dK and dV
# - Each tile computes QK^T once and reuses it for dQ/dK/dV
# ============================================================


@triton.jit
def _flash_bwd_drow_kernel(
    dO_ptr,
    O_ptr,
    Drow_ptr,  # Drow = sum(dO * O) over D
    stride_dob: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dm: tl.constexpr,
    N_Q: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    dO_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    O_ptrs = O_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    dO = tl.load(dO_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    O = tl.load(O_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    drow = tl.sum(dO * O, axis=1)  # (BM,)
    Drow_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
    tl.store(Drow_ptrs, drow, mask=(offs_m < N_Q))


@triton.jit
def _flash_bwd_fused_q_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    Drow_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # program: (bh, q_block)
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BM,)
    offs_n = tl.arange(0, BLOCK_N)  # (BN,)
    offs_d = tl.arange(0, BLOCK_D)  # (BD,)

    # Load Q (BM, D)
    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    Q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    # Load dO (BM, D)
    dO_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    dO = tl.load(dO_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    # Load LSE (BM,)
    l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
    LSE = tl.load(l_ptrs, mask=(offs_m < N_Q), other=-float("inf")).to(tl.float32)

    # Load Drow (BM,)
    drow_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
    Drow = tl.load(drow_ptrs, mask=(offs_m < N_Q), other=0.0).to(tl.float32)

    # Accumulate dQ in registers (BM, D)
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Iterate over K/V blocks
    for start_n in tl.static_range(0, N_K, BLOCK_N):
        k_ids = start_n + offs_n  # (BN,)

        # Load K/V (BN, D)
        k_ptrs = K_ptr + pid_bh * stride_kb + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + pid_bh * stride_vb + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vd
        K = tl.load(k_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        V = tl.load(v_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # S = QK^T * scale  -> (BM, BN)
        scores = tl.dot(Q, tl.trans(K)) * SCALE
        scores = tl.where(k_ids[None, :] < N_K, scores, -float("inf"))

        # causal masking
        if IS_CAUSAL:
            q_ids = offs_m
            causal = q_ids[:, None] >= k_ids[None, :]
            scores = tl.where(causal, scores, -float("inf"))
            # no-break: future blocks masked out
            active = start_n < (pid_m + 1) * BLOCK_M
            scores = tl.where(active, scores, -float("inf"))

        # P = exp(scores - LSE)
        P = tl.exp(scores - LSE[:, None])  # (BM, BN)

        # ---- dV tile: P^T @ dO  -> (BN, D)
        dV_tile = tl.dot(tl.trans(P), dO)

        # ---- dP = dO @ V^T -> (BM, BN)
        dP = tl.dot(dO, tl.trans(V))

        # ---- dS = P * (dP - Drow)
        dS = P * (dP - Drow[:, None])

        # ---- dK tile: dS^T @ Q * scale -> (BN, D)
        dK_tile = tl.dot(tl.trans(dS), Q) * SCALE

        # ---- dQ accumulate: dS @ K * scale -> (BM, D)
        dQ_acc += tl.dot(dS, K) * SCALE

        # Atomic add dK/dV into global buffers
        dK_ptrs = dK_ptr + pid_bh * stride_dkb + k_ids[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        dV_ptrs = dV_ptr + pid_bh * stride_dvb + k_ids[:, None] * stride_dvn + offs_d[None, :] * stride_dvd

        mask_kd = (k_ids[:, None] < N_K) & (offs_d[None, :] < D)

        tl.atomic_add(dK_ptrs, dK_tile, mask=mask_kd)
        tl.atomic_add(dV_ptrs, dV_tile, mask=mask_kd)

    # store dQ (no atomic needed: unique writer per q-block)
    dQ_ptrs = dQ_ptr + pid_bh * stride_dqb + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dQ_ptrs, dQ_acc, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D))


def _flash_backward_triton(
    q_f: torch.Tensor,
    k_f: torch.Tensor,
    v_f: torch.Tensor,
    O: torch.Tensor,
    L: torch.Tensor,
    dO: torch.Tensor,
    *,
    is_causal: bool,
):
    """
    BIG-CHANGE backward:
    - compute Drow (sum(dO * O))
    - fused kernel per q-block updates:
        dQ (store), dK/dV (atomic_add)
    All tensors expected fp32 for stability.
    """
    assert q_f.is_cuda and k_f.is_cuda and v_f.is_cuda and O.is_cuda and L.is_cuda and dO.is_cuda
    assert q_f.dtype == torch.float32 and k_f.dtype == torch.float32 and v_f.dtype == torch.float32
    assert dO.dtype == torch.float32 and O.dtype == torch.float32 and L.dtype == torch.float32

    B_eff, N_q, D = q_f.shape
    N_k = k_f.shape[1]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = _pick_block_d(D)
    scale = 1.0 / math.sqrt(D)

    # 1) Drow
    Drow = torch.empty((B_eff, N_q), device=q_f.device, dtype=torch.float32)
    grid_m = (B_eff, triton.cdiv(N_q, BLOCK_M))
    _flash_bwd_drow_kernel[grid_m](
        dO,
        O,
        Drow,
        stride_dob=dO.stride(0),
        stride_dom=dO.stride(1),
        stride_dod=dO.stride(2),
        stride_ob=O.stride(0),
        stride_om=O.stride(1),
        stride_od=O.stride(2),
        stride_db=Drow.stride(0),
        stride_dm=Drow.stride(1),
        N_Q=N_q,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    # 2) fused backward (q-block primary)
    dQ = torch.empty((B_eff, N_q, D), device=q_f.device, dtype=torch.float32)
    dK = torch.zeros((B_eff, N_k, D), device=q_f.device, dtype=torch.float32)
    dV = torch.zeros((B_eff, N_k, D), device=q_f.device, dtype=torch.float32)

    grid = (B_eff, triton.cdiv(N_q, BLOCK_M))
    _flash_bwd_fused_q_kernel[grid](
        q_f,
        k_f,
        v_f,
        dO,
        L,
        Drow,
        dQ,
        dK,
        dV,
        stride_qb=q_f.stride(0),
        stride_qm=q_f.stride(1),
        stride_qd=q_f.stride(2),
        stride_kb=k_f.stride(0),
        stride_kn=k_f.stride(1),
        stride_kd=k_f.stride(2),
        stride_vb=v_f.stride(0),
        stride_vn=v_f.stride(1),
        stride_vd=v_f.stride(2),
        stride_dob=dO.stride(0),
        stride_dom=dO.stride(1),
        stride_dod=dO.stride(2),
        stride_lb=L.stride(0),
        stride_lm=L.stride(1),
        stride_db=Drow.stride(0),
        stride_dm=Drow.stride(1),
        stride_dqb=dQ.stride(0),
        stride_dqm=dQ.stride(1),
        stride_dqd=dQ.stride(2),
        stride_dkb=dK.stride(0),
        stride_dkn=dK.stride(1),
        stride_dkd=dK.stride(2),
        stride_dvb=dV.stride(0),
        stride_dvn=dV.stride(1),
        stride_dvd=dV.stride(2),
        N_Q=N_q,
        N_K=N_k,
        D=D,
        SCALE=scale,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    return dQ, dK, dV
