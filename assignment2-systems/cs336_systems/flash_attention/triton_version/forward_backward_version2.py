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
    raise ValueError(f"Unsupported head dim D={D} for this implementation")


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
    # batch*head index: current batch * head index (from 0 to B_eff-1)
    pid_bh = tl.program_id(0)
    # query block index: which block of queries are we processing (from 0 to ceil(N_Q / BLOCK_M)-1)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BM,) # query offsets
    offs_n = tl.arange(0, BLOCK_N)  # (BN,) # key/value offsets
    offs_d = tl.arange(0, BLOCK_D)  # (BD,) # feature dimension offsets

    # Load Q tile: (BM, D)
    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    # Online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Iterate over K/V blocks
    for start_n in range(0, N_K, BLOCK_N):
        k_ids = start_n + offs_n  # (BN,)

        # Load K tile: (BN, D)
        k_ptrs = K_ptr + pid_bh * stride_kb + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # Scores: (BM, BN)
        scores = tl.dot(q, tl.trans(k)) * SCALE

        # Mask invalid keys
        scores = tl.where(k_ids[None, :] < N_K, scores, -float("inf"))

        # Causal mask
        if IS_CAUSAL:
            q_ids = offs_m
            causal = q_ids[:, None] >= k_ids[None, :]
            scores = tl.where(causal, scores, -float("inf"))

            # No `break` in Triton: mask-out blocks that are entirely in the future
            # For q-block [pid_m*BM, (pid_m+1)*BM), any key block starting at >= (pid_m+1)*BM is future.
            # active = start_n < (pid_m + 1) * BLOCK_M
            # scores = tl.where(active, scores, -float("inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))  # (BM,)
        p = tl.exp(scores - m_ij[:, None])  # (BM, BN)

        alpha = tl.exp(m_i - m_ij)  # (BM,)
        l_ij = alpha * l_i + tl.sum(p, axis=1)  # (BM,)

        # Load V tile: (BN, D)
        v_ptrs = V_ptr + pid_bh * stride_vb + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # acc = alpha*acc + p@v
        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_ij
        l_i = l_ij

    # Write output: O = acc / l_i, LSE = m_i + log(l_i)
    o = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    o_ptrs = O_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D))

    l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
    tl.store(l_ptrs, lse, mask=(offs_m < N_Q))


def _flash_forward_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
    """
    q/k/v: (B_eff, N, D) CUDA tensors
    Returns:
      O: (B_eff, N_q, D) fp32
      L: (B_eff, N_q) fp32 (logsumexp per row)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton FlashAttention requires CUDA tensors"
    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3

    B_eff, N_q, D = q.shape
    N_k = k.shape[1]

    O = torch.empty((B_eff, N_q, D), device=q.device, dtype=torch.float32)
    L = torch.empty((B_eff, N_q), device=q.device, dtype=torch.float32)

    # Define block sizes
    BLOCK_M = 64  # Number of queries per block
    BLOCK_N = 64  # Number of keys, values per block
    BLOCK_D = _pick_block_d(D)  # Dimension per block

    grid = (B_eff, triton.cdiv(N_q, BLOCK_M))  # Start (B_eff, Num query blocks)
    scale = 1.0 / math.sqrt(D)

    _flash_fwd_kernel[grid](
        # Pointers to Q, K, V, O, L
        q,
        k,
        v,
        O,
        L,
        # Strides tell the kernel how to index into the tensors
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
        # Other parameters
        N_Q=N_q,
        N_K=N_k,
        D=D,
        SCALE=scale,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return O, L


# ----------------------------
# Backward helper: compute D_i = sum_d (dO_i,d * O_i,d)
# ----------------------------


@triton.jit
def _flash_bwd_d_kernel(
    dO_ptr,
    O_ptr,
    D_ptr,
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

    d = tl.sum(dO * O, axis=1)  # (BM,)

    D_ptrs = D_ptr + pid_bh * stride_db + offs_m * stride_dm
    tl.store(D_ptrs, d, mask=(offs_m < N_Q))


@triton.jit
def _flash_bwd_dkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    Drow_ptr,
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
    # program: (bh, key_block)
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BN,)
    offs_d = tl.arange(0, BLOCK_D)  # (BD,)

    # ---- load K/V tile once per (bh, key_block) ----
    k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    # ---- iterate over query blocks ----
    # NOTE: static_range helps unrolling/pipelining
    for start_m in tl.static_range(0, N_Q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)  # (BM,)

        # load Q (BM, D)
        q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # scores (BM, BN)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(offs_n[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            q_ids = offs_m
            k_ids = offs_n
            causal = q_ids[:, None] >= k_ids[None, :]
            scores = tl.where(causal, scores, -float("inf"))
            # block-level "break" emulation:
            # if k_block_start >= q_end => no contribution
            # active = (pid_n * BLOCK_N) < (start_m + BLOCK_M)
            # scores = tl.where(active, scores, -float("inf"))

        # load LSE (BM,)
        l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
        LSE = tl.load(l_ptrs, mask=(offs_m < N_Q), other=-float("inf")).to(tl.float32)

        # P = exp(scores - LSE)
        P = tl.exp(scores - LSE[:, None])  # (BM, BN)

        # load dO (BM, D)
        dO_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        dO = tl.load(dO_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # -------- dV accumulation --------
        # dV += P^T @ dO
        dv_acc += tl.dot(tl.trans(P), dO)

        # -------- dK accumulation --------
        # dP = dO @ V^T  -> (BM, BN)
        dP = tl.dot(dO, tl.trans(v))

        # Drow (BM,)
        drow_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
        Drow = tl.load(drow_ptrs, mask=(offs_m < N_Q), other=0.0).to(tl.float32)

        # dS = P * (dP - Drow)
        dS = P * (dP - Drow[:, None])

        # dK += dS^T @ Q * SCALE
        dk_acc += tl.dot(tl.trans(dS), q) * SCALE

    # ---- store dK, dV ----
    dK_ptrs = dK_ptr + pid_bh * stride_dkb + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    tl.store(dK_ptrs, dk_acc, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D))

    dV_ptrs = dV_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dV_ptrs, dv_acc, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D))


# ----------------------------
# Backward: dQ kernel (per query-block, loops over all key blocks)
# dQ += dS @ K * scale, with dS = P * (dP - Drow)
# ----------------------------
@triton.jit
def _flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dQ_ptr,
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
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q (BM, D)
    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    # Load dO (BM, D)
    do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    do = tl.load(do_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    # Load LSE (BM,) and Drow (BM,)
    l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
    lse = tl.load(l_ptrs, mask=(offs_m < N_Q), other=float("inf")).to(tl.float32)
    d_ptrs = D_ptr + pid_bh * stride_db + offs_m * stride_dm
    Drow = tl.load(d_ptrs, mask=(offs_m < N_Q), other=0.0).to(tl.float32)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K and V for this block (BN, D)
        k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # scores (BM, BN)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(offs_n[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal, scores, -float("inf"))

            # mask-out future blocks
            active = start_n < (pid_m + 1) * BLOCK_M
            scores = tl.where(active, scores, -float("inf"))

        P = tl.exp(scores - lse[:, None])

        dP = tl.dot(do, tl.trans(v))  # (BM, BN)
        dS = P * (dP - Drow[:, None])  # (BM, BN)

        acc_dq += tl.dot(dS, k) * SCALE

    dq_ptrs = dQ_ptr + pid_bh * stride_dqb + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, acc_dq, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D))


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
    assert q_f.is_cuda and k_f.is_cuda and v_f.is_cuda and dO.is_cuda
    B_eff, N_q, D = q_f.shape
    N_k = k_f.shape[1]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = _pick_block_d(D)
    scale = 1.0 / math.sqrt(D)

    # 1) Drow = sum(dO * O) over D
    Drow = torch.empty((B_eff, N_q), device=q_f.device, dtype=torch.float32)
    grid_m = (B_eff, triton.cdiv(N_q, BLOCK_M))
    _flash_bwd_d_kernel[grid_m](
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
    )

    # 2) dK and dV in ONE kernel (medium-change optimization)
    dK = torch.zeros((B_eff, N_k, D), device=q_f.device, dtype=torch.float32)
    dV = torch.zeros((B_eff, N_k, D), device=q_f.device, dtype=torch.float32)

    grid_n = (B_eff, triton.cdiv(N_k, BLOCK_N))
    _flash_bwd_dkv_kernel[grid_n](
        q_f,
        k_f,
        v_f,
        dO,
        L,
        Drow,
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
        num_stages=3,  # 可以试 2/3/4
    )

    # 3) dQ (keep your existing dQ kernel)
    dQ = torch.zeros((B_eff, N_q, D), device=q_f.device, dtype=torch.float32)
    _flash_bwd_dq_kernel[grid_m](
        q_f,
        k_f,
        v_f,
        dO,
        L,
        Drow,
        dQ,
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
