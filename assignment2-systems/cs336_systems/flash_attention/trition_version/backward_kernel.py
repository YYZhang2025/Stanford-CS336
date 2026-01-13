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


# ----------------------------
# Backward: dV kernel (per key-block, loops over all query blocks)
# dV[n] = sum_m P(m,n)^T @ dO[m]
# ----------------------------
@triton.jit
def _flash_bwd_dv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
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
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)  # key block

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BN,)
    offs_d = tl.arange(0, BLOCK_D)  # (BD,)

    # Load K tile once (BN, D)
    k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for start_m in range(0, N_Q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)  # (BM,)

        # Load Q (BM, D)
        q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # scores (BM, BN)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(offs_n[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal, scores, -float("inf"))

            # mask-out totally future blocks
            active = (pid_n * BLOCK_N) < (start_m + BLOCK_M)
            scores = tl.where(active, scores, -float("inf"))

        # Load LSE for queries (BM,)
        l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
        lse = tl.load(l_ptrs, mask=(offs_m < N_Q), other=float("inf")).to(tl.float32)

        # P (BM, BN)
        p = tl.exp(scores - lse[:, None])

        # Load dO (BM, D)
        do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        do = tl.load(do_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # dV += P^T @ dO  -> (BN, D)
        acc_dv += tl.dot(tl.trans(p), do)

    dv_ptrs = dV_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, acc_dv, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D))


# ----------------------------
# Backward: dK kernel (per key-block, loops over all query blocks)
# dS = P * (dP - Drow), where dP = dO @ V^T, Drow = sum(dO*O)
# dK += dS^T @ Q * scale
# ----------------------------
@triton.jit
def _flash_bwd_dk_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,  # (B_eff, N_Q) float32
    dK_ptr,
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
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BN,)
    offs_d = tl.arange(0, BLOCK_D)  # (BD,)

    # Load K and V for this key-block
    k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for start_m in range(0, N_Q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)  # (BM,)

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

        # scores (BM, BN)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(offs_n[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal, scores, -float("inf"))

            active = (pid_n * BLOCK_N) < (start_m + BLOCK_M)
            scores = tl.where(active, scores, -float("inf"))

        # P
        P = tl.exp(scores - lse[:, None])

        # dP = dO @ V^T
        dP = tl.dot(do, tl.trans(v))  # (BM, BN)

        # dS
        dS = P * (dP - Drow[:, None])  # (BM, BN)

        # dK += dS^T @ Q * scale
        acc_dk += tl.dot(tl.trans(dS), q) * SCALE

    dk_ptrs = dK_ptr + pid_bh * stride_dkb + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, acc_dk, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D))


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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    O: torch.Tensor,
    L: torch.Tensor,
    dO: torch.Tensor,
    is_causal: bool,
):
    """
    All tensors flattened:
      q,k,v : (B_eff, N, D)  (input dtype)
      O     : (B_eff, N_q, D) fp32
      L     : (B_eff, N_q) fp32
      dO    : (B_eff, N_q, D) (same dtype as O output gradient, typically fp32)
    Returns grads (fp32), caller will cast to input dtype.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and O.is_cuda and L.is_cuda and dO.is_cuda
    B_eff, N_q, D = q.shape
    N_k = k.shape[1]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = _pick_block_d(D)
    scale = 1.0 / math.sqrt(D)

    # Compute D_buf = sum(dO * O) per row in Triton (fp32)
    D_buf = torch.empty((B_eff, N_q), device=q.device, dtype=torch.float32)
    grid_d = (B_eff, triton.cdiv(N_q, BLOCK_M))
    _flash_bwd_d_kernel[grid_d](
        dO,
        O,
        D_buf,
        stride_dob=dO.stride(0),
        stride_dom=dO.stride(1),
        stride_dod=dO.stride(2),
        stride_ob=O.stride(0),
        stride_om=O.stride(1),
        stride_od=O.stride(2),
        stride_db=D_buf.stride(0),
        stride_dm=D_buf.stride(1),
        N_Q=N_q,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    # dV (fp32)
    dV = torch.empty((B_eff, N_k, D), device=q.device, dtype=torch.float32)
    grid_dv = (B_eff, triton.cdiv(N_k, BLOCK_N))
    _flash_bwd_dv_kernel[grid_dv](
        q,
        k,
        v,
        dO,
        L,
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
    )

    # dK (fp32)
    dK = torch.empty((B_eff, N_k, D), device=q.device, dtype=torch.float32)
    grid_dk = (B_eff, triton.cdiv(N_k, BLOCK_N))
    _flash_bwd_dk_kernel[grid_dk](
        q,
        k,
        v,
        dO,
        L,
        D_buf,
        dK,
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
        stride_db=D_buf.stride(0),
        stride_dm=D_buf.stride(1),
        stride_dkb=dK.stride(0),
        stride_dkn=dK.stride(1),
        stride_dkd=dK.stride(2),
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

    # dQ (fp32)
    dQ = torch.empty((B_eff, N_q, D), device=q.device, dtype=torch.float32)
    grid_dq = (B_eff, triton.cdiv(N_q, BLOCK_M))
    _flash_bwd_dq_kernel[grid_dq](
        q,
        k,
        v,
        dO,
        L,
        D_buf,
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
        stride_db=D_buf.stride(0),
        stride_dm=D_buf.stride(1),
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
    )

    return dQ, dK, dV
