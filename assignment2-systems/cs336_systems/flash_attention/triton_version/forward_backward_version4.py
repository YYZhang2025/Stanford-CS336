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


@triton.jit
def drow_kernel(
    dO_ptr,
    O_ptr,
    Drow_ptr,
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

    drow = tl.sum(dO * O, axis=1)  # (BM,) fp32

    out_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
    tl.store(out_ptrs, drow, mask=(offs_m < N_Q))


# ----------------------------
# Kernel A: (bh, q_block) 计算 dQ
# 这里会“重算 P”
# ----------------------------
@triton.jit
def bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    LSE_ptr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # q indices
    offs_n = tl.arange(0, BLOCK_N)  # k indices within block
    offs_d = tl.arange(0, BLOCK_D)

    # load Q, dO
    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    Q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    dO = tl.load(do_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    # load LSE, Drow
    l_ptrs = LSE_ptr + pid_bh * stride_lb + offs_m * stride_lm
    dr_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
    LSE = tl.load(l_ptrs, mask=(offs_m < N_Q), other=-float("inf")).to(tl.float32)
    Drow = tl.load(dr_ptrs, mask=(offs_m < N_Q), other=0.0).to(tl.float32)

    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in tl.static_range(0, N_K, BLOCK_N):
        k_ids = start_n + offs_n

        # load K, V (只用于 dP / dQ 的这次 pass)
        k_ptrs = K_ptr + pid_bh * stride_kb + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + pid_bh * stride_vb + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vd
        K = tl.load(k_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        V = tl.load(v_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # scores = QK^T * scale
        scores = tl.dot(Q, tl.trans(K)) * SCALE
        scores = tl.where(k_ids[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            q_ids = offs_m
            scores = tl.where(q_ids[:, None] >= k_ids[None, :], scores, -float("inf"))

        # -------- 重算 P（第一次）--------
        P = tl.exp(scores - LSE[:, None])  # (BM, BN)

        # dP = dO V^T
        dP = tl.dot(dO, tl.trans(V))

        # dS = P * (dP - Drow)
        dS = P * (dP - Drow[:, None])

        # dQ += dS K * scale
        dQ_acc += tl.dot(dS, K) * SCALE

    out_ptrs = dQ_ptr + pid_bh * stride_dqb + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(out_ptrs, dQ_acc, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D))


# ----------------------------
# Kernel B: (bh, k_block) 计算 dK, dV
# 这里会“再重算一次 P”
# ----------------------------
@triton.jit
def bwd_dkdv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    LSE_ptr,
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
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # load K, V once for this k-block
    k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    K = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    V = tl.load(v_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for start_m in tl.static_range(0, N_Q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)

        # load Q, dO
        q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        Q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        dO = tl.load(do_ptrs, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # load LSE, Drow for these q
        l_ptrs = LSE_ptr + pid_bh * stride_lb + offs_m * stride_lm
        dr_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm
        LSE = tl.load(l_ptrs, mask=(offs_m < N_Q), other=-float("inf")).to(tl.float32)
        Drow = tl.load(dr_ptrs, mask=(offs_m < N_Q), other=0.0).to(tl.float32)

        # scores = QK^T * scale (注意：这里 K 是当前 k-block 的 K)
        scores = tl.dot(Q, tl.trans(K)) * SCALE
        scores = tl.where(offs_n[None, :] < N_K, scores, -float("inf"))

        if IS_CAUSAL:
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -float("inf"))

        # -------- 重算 P（第二次）--------
        P = tl.exp(scores - LSE[:, None])  # (BM, BN)

        # dV += P^T dO
        dV_acc += tl.dot(tl.trans(P), dO)

        # dP = dO V^T (V 是当前 k-block 的 V)
        dP = tl.dot(dO, tl.trans(V))

        # dS = P * (dP - Drow)
        dS = P * (dP - Drow[:, None])

        # dK += dS^T Q * scale
        dK_acc += tl.dot(tl.trans(dS), Q) * SCALE

    dk_ptrs = dK_ptr + pid_bh * stride_dkb + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = dV_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    mask_nd = (offs_n[:, None] < N_K) & (offs_d[None, :] < D)
    tl.store(dk_ptrs, dK_acc, mask=mask_nd)
    tl.store(dv_ptrs, dV_acc, mask=mask_nd)


def flash_bwd_triton(q, k, v, O, LSE, dO, *, is_causal: bool):
    """
    q,k,v,O,dO: (B_eff, N, D) fp32 (先用 fp32 跑通；要混精后面再改)
    LSE: (B_eff, N) fp32
    Return: dQ,dK,dV fp32
    """

    B_eff, N_Q, D = q.shape
    N_K = k.shape[1]
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = _pick_block_d(D)
    scale = 1.0 / math.sqrt(D)

    # 1) Drow
    Drow = torch.empty((B_eff, N_Q), device=q.device, dtype=torch.float32)
    grid_m = (B_eff, triton.cdiv(N_Q, BLOCK_M))
    drow_kernel[grid_m](
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
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    # 2) dQ pass (bh, q_block) —— 重算 P #1
    dQ = torch.empty((B_eff, N_Q, D), device=q.device, dtype=torch.float32)
    grid_q = (B_eff, triton.cdiv(N_Q, BLOCK_M))
    bwd_dq_kernel[grid_q](
        q,
        k,
        v,
        dO,
        LSE,
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
        stride_lb=LSE.stride(0),
        stride_lm=LSE.stride(1),
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    # 3) dK/dV pass (bh, k_block) —— 重算 P #2（没有 atomic）
    dK = torch.empty((B_eff, N_K, D), device=q.device, dtype=torch.float32)
    dV = torch.empty((B_eff, N_K, D), device=q.device, dtype=torch.float32)
    grid_k = (B_eff, triton.cdiv(N_K, BLOCK_N))
    bwd_dkdv_kernel[grid_k](
        q,
        k,
        v,
        dO,
        LSE,
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
        stride_lb=LSE.stride(0),
        stride_lm=LSE.stride(1),
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    return dQ, dK, dV
