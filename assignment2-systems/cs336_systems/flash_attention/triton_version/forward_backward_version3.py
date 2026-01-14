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
# Mixed-precision variant:
# - Q/K/V/dO/O are LOADED as fp16 or bf16
# - all accumulators / logits / exp / reductions stay fp32
#
# Key trick: when one operand is fp32 (P, dS), cast it to half just for tl.dot
# so the dot uses tensor cores and accumulates fp32 via out_dtype=tl.float32.


@triton.jit
def _flash_bwd_drow_kernel(
    dO_ptr,
    O_ptr,
    Drow_ptr,  # (B, N_Q)
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
    USE_BF16: tl.constexpr,  # True -> bf16, False -> fp16
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    dO_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    O_ptrs = O_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    in_dtype = tl.bfloat16 if USE_BF16 else tl.float16

    # load as half/bf16, multiply+sum in fp32
    dO_h = tl.load(dO_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(in_dtype)
    O_h = tl.load(O_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(in_dtype)

    prod = dO_h.to(tl.float32) * O_h.to(tl.float32)
    drow = tl.sum(prod, axis=1)  # (BM,) fp32

    Drow_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
    tl.store(Drow_ptrs, drow, mask=(offs_m < N_Q))


@triton.jit
def _flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    Drow_ptr,
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
    USE_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    in_dtype = tl.bfloat16 if USE_BF16 else tl.float16

    # Load Q, dO in half/bf16
    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    Q_h = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(in_dtype)
    dO_h = tl.load(do_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(in_dtype)

    # Load LSE, Drow in fp32
    l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
    dr_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
    LSE = tl.load(l_ptrs, mask=(offs_m < N_Q), other=-float("inf")).to(tl.float32)
    Drow = tl.load(dr_ptrs, mask=(offs_m < N_Q), other=0.0).to(tl.float32)

    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in tl.static_range(0, N_K, BLOCK_N):
        k_ids = start_n + offs_n  # (BN,)

        # Load K, V in half/bf16
        k_ptrs = K_ptr + pid_bh * stride_kb + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + pid_bh * stride_vb + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vd
        K_h = tl.load(k_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(in_dtype)
        V_h = tl.load(v_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(in_dtype)

        # scores = (Q K^T) * scale   [tensor core], fp32 output
        scores = tl.dot(Q_h, tl.trans(K_h), out_dtype=tl.float32) * SCALE
        scores = tl.where(k_ids[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            q_ids = offs_m
            causal = q_ids[:, None] >= k_ids[None, :]
            scores = tl.where(causal, scores, -float("inf"))

        # P fp32
        P = tl.exp(scores - LSE[:, None])  # (BM, BN) fp32

        # dP = dO V^T  [tensor core], fp32
        dP = tl.dot(dO_h, tl.trans(V_h), out_dtype=tl.float32)

        # dS fp32
        dS = P * (dP - Drow[:, None])

        # dQ += (dS K) * scale
        # Cast dS to half/bf16 just for dot (keeps accum fp32)
        dS_h = dS.to(in_dtype)
        dQ_acc += tl.dot(dS_h, K_h, out_dtype=tl.float32) * SCALE

    dq_ptrs = dQ_ptr + pid_bh * stride_dqb + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dQ_acc, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D))


@triton.jit
def _flash_bwd_dkdv_kernel(
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
    USE_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    in_dtype = tl.bfloat16 if USE_BF16 else tl.float16

    # Load K/V once per k-block (half/bf16)
    k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    K_h = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(in_dtype)
    V_h = tl.load(v_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(in_dtype)

    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for start_m in tl.static_range(0, N_Q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)

        # Load Q, dO in half/bf16
        q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        Q_h = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(in_dtype)
        dO_h = tl.load(do_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(in_dtype)

        # Load LSE, Drow (fp32)
        l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
        dr_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
        LSE = tl.load(l_ptrs, mask=(offs_m < N_Q), other=-float("inf")).to(tl.float32)
        Drow = tl.load(dr_ptrs, mask=(offs_m < N_Q), other=0.0).to(tl.float32)

        # scores = Q K^T * scale  [tensor core], fp32
        scores = tl.dot(Q_h, tl.trans(K_h), out_dtype=tl.float32) * SCALE
        scores = tl.where(offs_n[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            q_ids = offs_m
            causal = q_ids[:, None] >= offs_n[None, :]
            scores = tl.where(causal, scores, -float("inf"))

        # P fp32
        P = tl.exp(scores - LSE[:, None])  # fp32

        # dV += P^T dO  (cast P to half/bf16 for dot)
        P_h = P.to(in_dtype)
        dV_acc += tl.dot(tl.trans(P_h), dO_h, out_dtype=tl.float32)

        # dP = dO V^T   [tensor core], fp32
        dP = tl.dot(dO_h, tl.trans(V_h), out_dtype=tl.float32)

        # dS fp32
        dS = P * (dP - Drow[:, None])

        # dK += dS^T Q * scale  (cast dS to half/bf16 for dot)
        dS_h = dS.to(in_dtype)
        dK_acc += tl.dot(tl.trans(dS_h), Q_h, out_dtype=tl.float32) * SCALE

    dk_ptrs = dK_ptr + pid_bh * stride_dkb + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = dV_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    mask_nd = (offs_n[:, None] < N_K) & (offs_d[None, :] < D)
    tl.store(dk_ptrs, dK_acc, mask=mask_nd)
    tl.store(dv_ptrs, dV_acc, mask=mask_nd)


def flash_backward_triton(
    q: torch.Tensor,  # fp16/bf16 recommended
    k: torch.Tensor,
    v: torch.Tensor,
    O: torch.Tensor,  # fp16/bf16 ok
    L: torch.Tensor,  # fp32 recommended (LSE)
    dO: torch.Tensor,  # fp16/bf16 ok
    *,
    is_causal: bool,
):
    """
    Returns dQ, dK, dV in fp32.
    q/k/v/dO/O are loaded as fp16 or bf16 inside kernels, accum stays fp32.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and O.is_cuda and L.is_cuda and dO.is_cuda
    assert q.shape == O.shape == dO.shape
    B_eff, N_Q, D = q.shape
    N_K = k.shape[1]
    assert k.shape == v.shape == (B_eff, N_K, D)
    assert L.shape == (B_eff, N_Q)

    # Choose load dtype
    use_bf16 = q.dtype == torch.bfloat16
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k.dtype == q.dtype and v.dtype == q.dtype and dO.dtype == q.dtype and O.dtype == q.dtype
    assert L.dtype == torch.float32, "Recommend L (LSE) be fp32."

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = _pick_block_d(D)
    scale = 1.0 / math.sqrt(D)

    # 1) Drow (fp32)
    Drow = torch.empty((B_eff, N_Q), device=q.device, dtype=torch.float32)
    grid_m = (B_eff, triton.cdiv(N_Q, BLOCK_M))
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
        N_Q=N_Q,
        D=D,
        USE_BF16=use_bf16,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    # 2) dQ (fp32 out)
    dQ = torch.empty((B_eff, N_Q, D), device=q.device, dtype=torch.float32)
    grid_q = (B_eff, triton.cdiv(N_Q, BLOCK_M))
    _flash_bwd_dq_kernel[grid_q](
        q,
        k,
        v,
        dO,
        L,
        Drow,
        dQ,
        stride_qb=q.stride(0),
        stride_qm=q.stride(1),
        stride_qd=q.stride(2),
        stride_kb=k.stride(0),
        stride_kn=k.stride(1),
        stride_kd=k.stride(2),
        stride_vb=v.stride(0),
        stride_vn=v.stride(1),
        stride_vd=v.stride(2),
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
        N_Q=N_Q,
        N_K=N_K,
        D=D,
        SCALE=scale,
        IS_CAUSAL=is_causal,
        USE_BF16=use_bf16,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    # 3) dK/dV (fp32 out)
    dK = torch.empty((B_eff, N_K, D), device=q.device, dtype=torch.float32)
    dV = torch.empty((B_eff, N_K, D), device=q.device, dtype=torch.float32)
    grid_k = (B_eff, triton.cdiv(N_K, BLOCK_N))
    _flash_bwd_dkdv_kernel[grid_k](
        q,
        k,
        v,
        dO,
        L,
        Drow,
        dK,
        dV,
        stride_qb=q.stride(0),
        stride_qm=q.stride(1),
        stride_qd=q.stride(2),
        stride_kb=k.stride(0),
        stride_kn=k.stride(1),
        stride_kd=k.stride(2),
        stride_vb=v.stride(0),
        stride_vn=v.stride(1),
        stride_vd=v.stride(2),
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
        N_Q=N_Q,
        N_K=N_K,
        D=D,
        SCALE=scale,
        IS_CAUSAL=is_causal,
        USE_BF16=use_bf16,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    return dQ, dK, dV
