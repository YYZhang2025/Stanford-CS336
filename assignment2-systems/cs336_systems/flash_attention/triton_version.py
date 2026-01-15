import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_kv_kernel(  # inlined device function (NOT launched with [...])
    q_block,  # (BLOCK_Q, BLOCK_D) already loaded and scaled
    K_ptr,
    V_ptr,
    pid_bh: tl.constexpr,
    offs_m,  # (BLOCK_Q,) absolute q indices
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    N_K: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Accumulators (keep in fp32 for stability)
    m_i = tl.full([BLOCK_Q], -float("inf"), tl.float32)  # running row max
    l_i = tl.zeros([BLOCK_Q], tl.float32)  # running sum exp
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)  # running numerator

    # Loop over K/V blocks
    for start_n in range(0, N_K, BLOCK_K):
        offs_n = start_n + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
        mask_n = offs_n < N_K

        offs_d = tl.arange(0, BLOCK_D)

        # ---- load K as (BLOCK_D, BLOCK_K) for tl.dot(q, k) ----
        k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[None, :] & (offs_d[:, None] < BLOCK_D), other=0.0).to(tl.float16)

        # scores: (BLOCK_Q, BLOCK_K)
        scores = tl.dot(q_block, k).to(tl.float32)

        # Apply causal mask if needed: disallow k positions > q positions
        if IS_CAUSAL:
            # offs_m: (BLOCK_Q,), offs_n: (BLOCK_K,)
            causal = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal, scores, -float("inf"))

        # Also mask out invalid K columns (past N_K)
        scores = tl.where(mask_n[None, :], scores, -float("inf"))

        # Online softmax update
        m_ij = tl.max(scores, axis=1)  # (BLOCK_Q,)
        m_new = tl.maximum(m_i, m_ij)  # (BLOCK_Q,)

        p = tl.exp(scores - m_new[:, None])  # (BLOCK_Q, BLOCK_K) in fp32

        # rescale old accumulators
        alpha = tl.exp(m_i - m_new)  # (BLOCK_Q,)
        l_new = alpha * l_i + tl.sum(p, axis=1)  # (BLOCK_Q,)

        # ---- load V as (BLOCK_K, BLOCK_D) for tl.dot(p, v) ----
        v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_D), other=0.0).to(tl.float16)

        acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v).to(tl.float32)

        # commit running stats
        m_i = m_new
        l_i = l_new

    # Final normalize
    o = acc / l_i[:, None]  # (BLOCK_Q, BLOCK_D)
    lse = m_i + tl.log(l_i)  # (BLOCK_Q,)

    return o, lse


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
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # 0..B_eff-1
    pid_q = tl.program_id(1)  # q-block index

    offs_m = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # (BLOCK_Q,)
    offs_d = tl.arange(0, BLOCK_D)  # (BLOCK_D,)

    # Load Q block (BLOCK_Q, BLOCK_D)
    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < N_Q) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float16)
    q = (q * SCALE).to(tl.float16)  # match your PyTorch: q = q * scale

    # Compute O_block and LSE for this Q block by streaming over K/V blocks
    o_block, lse = _flash_kv_kernel(
        q,
        K_ptr,
        V_ptr,
        pid_bh,
        offs_m,
        stride_kb,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vn,
        stride_vd,
        N_K,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=IS_CAUSAL,
    )

    # Store O
    o_ptrs = O_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < N_Q) & (offs_d[None, :] < D)
    tl.store(o_ptrs, o_block.to(tl.float16), mask=o_mask)

    # Store LSE
    l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
    tl.store(l_ptrs, lse.to(tl.float32), mask=(offs_m < N_Q))


def flash_fwd_triton(q, k, v, is_causal: bool):
    # q,k,v: (B_eff, N, D) contiguous preferred

    B_eff, N_Q, D = q.shape
    N_K = k.shape[1]
    scale = 1.0 / math.sqrt(D)

    O = torch.empty((B_eff, N_Q, D), device=q.device, dtype=q.dtype)
    L = torch.empty((B_eff, N_Q), device=q.device, dtype=torch.float32)

    BLOCK_Q = 64
    BLOCK_K = 64
    # choose BLOCK_D >= D (power of 2 is common); simplest:
    BLOCK_D = 128 if D > 64 else (64 if D > 32 else (32 if D > 16 else 16))

    grid = (B_eff, triton.cdiv(N_Q, BLOCK_Q))

    _flash_fwd_kernel[grid](
        q,
        k,
        v,
        O,
        L,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        L.stride(0),
        L.stride(1),
        N_Q=N_Q,
        N_K=N_K,
        D=D,
        SCALE=scale,
        IS_CAUSAL=is_causal,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return O, L


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


# ============================================================
# Inner: dK/dV for ONE key-block (official-style loop order)
# ============================================================
@triton.jit
def _attn_bwd_dkdv(
    dk_acc,
    dv_acc,
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    Drow_ptr,
    pid_bh: tl.constexpr,
    start_n: tl.constexpr,
    offs_n,
    offs_d,
    k,
    v,
    stride_qb: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dm: tl.constexpr,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
):
    # We keep k/v in fp16 for tensor core, accum in fp32
    k16 = k.to(tl.float16)
    v16 = v.to(tl.float16)

    # ---- 1) diagonal band (masked) ----
    start_m0 = 0
    if IS_CAUSAL:
        MASK_BM: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
        tl.static_assert(BLOCK_N % MASK_BM == 0)
        num_steps: tl.constexpr = BLOCK_N // MASK_BM

        # m in [start_n, start_n + BLOCK_N) with MASK_BM rows each
        for step in tl.static_range(0, num_steps):
            start_m = start_n + step * MASK_BM
            offs_m_mask = start_m + tl.arange(0, MASK_BM)  # shape [MASK_BM], NEVER reused as [BLOCK_M]

            q_ptrs = (
                Q_ptr + pid_bh * stride_qb + offs_m_mask[:, None] * stride_qm + offs_d[None, :] * stride_qd
            )
            q16 = tl.load(
                q_ptrs,
                mask=(offs_m_mask[:, None] < N_Q) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float16)

            # scores: (MASK_BM, BN)
            scores = tl.dot(q16, tl.trans(k16)).to(tl.float32) * SCALE
            scores = tl.where(offs_n[None, :] < N_K, scores, -float("inf"))

            causal = offs_m_mask[:, None] >= offs_n[None, :]
            scores = tl.where(causal, scores, -float("inf"))

            l_ptrs = L_ptr + pid_bh * stride_lb + offs_m_mask * stride_lm
            lse = tl.load(l_ptrs, mask=(offs_m_mask < N_Q), other=-float("inf")).to(tl.float32)

            # P = exp(scores - lse)
            P = tl.exp(scores - lse[:, None]).to(tl.float16)  # keep P in fp16 for dot

            do_ptrs = (
                dO_ptr
                + pid_bh * stride_dob
                + offs_m_mask[:, None] * stride_dom
                + offs_d[None, :] * stride_dod
            )
            do16 = tl.load(
                do_ptrs,
                mask=(offs_m_mask[:, None] < N_Q) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float16)

            # dv += P^T @ dO
            dv_acc += tl.dot(tl.trans(P), do16).to(tl.float32)

            # dp = dO @ V^T
            dp = tl.dot(do16, tl.trans(v16)).to(tl.float32)

            drow_ptrs = Drow_ptr + pid_bh * stride_db + offs_m_mask * stride_dm
            drow = tl.load(drow_ptrs, mask=(offs_m_mask < N_Q), other=0.0).to(tl.float32)

            ds = (P.to(tl.float32)) * (dp - drow[:, None])  # fp32
            ds16 = ds.to(tl.float16)

            # dk += dS^T @ Q
            dk_acc += tl.dot(tl.trans(ds16), q16).to(tl.float32) * SCALE

        start_m0 = start_n + BLOCK_N  # after diagonal band

    # ---- 2) remaining blocks (unmasked) ----
    # Iterate over full query blocks; skip those before start_m0 with a mask (no shape changes!)
    for qbid in tl.static_range(0, NUM_Q_BLOCKS):
        start_m = qbid * BLOCK_M
        active_blk = True
        if IS_CAUSAL:
            active_blk = start_m >= start_m0

        offs_m_full = start_m + tl.arange(0, BLOCK_M)  # [BM]
        row_valid = active_blk & (offs_m_full < N_Q)  # [BM]

        # load Q / dO only for valid rows
        q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m_full[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q16 = tl.load(
            q_ptrs,
            mask=row_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float16)

        do_ptrs = (
            dO_ptr + pid_bh * stride_dob + offs_m_full[:, None] * stride_dom + offs_d[None, :] * stride_dod
        )
        do16 = tl.load(
            do_ptrs,
            mask=row_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float16)

        # scores: make invalid rows -inf
        scores = tl.dot(q16, tl.trans(k16)).to(tl.float32) * SCALE
        scores = tl.where(row_valid[:, None] & (offs_n[None, :] < N_K), scores, -float("inf"))

        # lse: invalid rows use 0.0 (NOT -inf)
        l_ptrs = L_ptr + pid_bh * stride_lb + offs_m_full * stride_lm
        lse = tl.load(l_ptrs, mask=row_valid, other=0.0).to(tl.float32)

        # P: invalid rows become exp(-inf)=0 safely
        P = tl.exp(scores - lse[:, None]).to(tl.float16)

        dv_acc += tl.dot(tl.trans(P), do16).to(tl.float32)

        dp = tl.dot(do16, tl.trans(v16)).to(tl.float32)

        drow_ptrs = Drow_ptr + pid_bh * stride_db + offs_m_full * stride_dm
        drow = tl.load(drow_ptrs, mask=row_valid, other=0.0).to(tl.float32)

        ds = P.to(tl.float32) * (dp - drow[:, None])
        dk_acc += tl.dot(tl.trans(ds.to(tl.float16)), q16).to(tl.float32) * SCALE

    return dk_acc, dv_acc


@triton.jit
def _attn_bwd_dq(
    acc_dq,
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    Drow_ptr,
    pid_bh: tl.constexpr,
    start_m: tl.constexpr,
    offs_m,
    offs_d,
    q16,
    do16,
    lse,
    drow,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # keep as your math: dq += dS @ K
    for start_n in tl.static_range(0, N_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k16 = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float16)
        v16 = tl.load(v_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float16)

        scores = tl.dot(q16, tl.trans(k16)).to(tl.float32) * SCALE
        scores = tl.where(offs_n[None, :] < N_K, scores, -float("inf"))
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal, scores, -float("inf"))

        P = tl.exp(scores - lse[:, None]).to(tl.float16)

        dp = tl.dot(do16, tl.trans(v16)).to(tl.float32)
        ds = (P.to(tl.float32)) * (dp - drow[:, None])

        acc_dq += tl.dot(ds.to(tl.float16), k16).to(tl.float32) * SCALE

    return acc_dq


@triton.jit
def _attn_bwd(
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
    BLK_SLICE_FACTOR: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_blk = tl.program_id(1)

    # ---- dK/dV for key-block pid_blk ----
    start_n = pid_blk * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    dk_acc, dv_acc = _attn_bwd_dkdv(
        dk_acc,
        dv_acc,
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        L_ptr,
        Drow_ptr,
        pid_bh,
        start_n,
        offs_n,
        offs_d,
        k,
        v,
        stride_qb,
        stride_qm,
        stride_qd,
        stride_dob,
        stride_dom,
        stride_dod,
        stride_lb,
        stride_lm,
        stride_db,
        stride_dm,
        N_Q,
        N_K,
        D,
        SCALE,
        IS_CAUSAL,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        BLK_SLICE_FACTOR,
        NUM_Q_BLOCKS,
    )

    dK_ptrs = dK_ptr + pid_bh * stride_dkb + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dV_ptrs = dV_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dK_ptrs, dk_acc, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D))
    tl.store(dV_ptrs, dv_acc, mask=(offs_n[:, None] < N_K) & (offs_d[None, :] < D))

    # ---- dQ for query-block pid_blk ----
    start_m = pid_blk * BLOCK_M
    in_range_q = start_m < N_Q
    offs_m = start_m + tl.arange(0, BLOCK_M)

    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
    d_ptrs = Drow_ptr + pid_bh * stride_db + offs_m * stride_dm

    q16 = tl.load(q_ptrs, mask=in_range_q & (offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(
        tl.float16
    )
    do16 = tl.load(do_ptrs, mask=in_range_q & (offs_m[:, None] < N_Q) & (offs_d[None, :] < D), other=0.0).to(
        tl.float16
    )
    lse = tl.load(l_ptrs, mask=in_range_q & (offs_m < N_Q), other=-float("inf")).to(tl.float32)
    drow = tl.load(d_ptrs, mask=in_range_q & (offs_m < N_Q), other=0.0).to(tl.float32)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    acc_dq = _attn_bwd_dq(
        acc_dq,
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        L_ptr,
        Drow_ptr,
        pid_bh,
        start_m,
        offs_m,
        offs_d,
        q16,
        do16,
        lse,
        drow,
        stride_kb,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vn,
        stride_vd,
        N_Q,
        N_K,
        D,
        SCALE,
        IS_CAUSAL,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
    )

    dQ_ptrs = dQ_ptr + pid_bh * stride_dqb + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dQ_ptrs, acc_dq, mask=in_range_q & (offs_m[:, None] < N_Q) & (offs_d[None, :] < D))


def flash_backward_triton(q_f, k_f, v_f, O, L, dO, *, is_causal: bool):
    assert q_f.is_cuda and k_f.is_cuda and v_f.is_cuda and dO.is_cuda
    B_eff, N_q, D = q_f.shape
    N_k = k_f.shape[1]
    assert N_q == N_k

    BLOCK_M = 64
    BLOCK_N = 64
    assert BLOCK_M == BLOCK_N
    BLOCK_D = _pick_block_d(D)
    scale = 1.0 / math.sqrt(D)

    # 1) Drow
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

    dQ = torch.zeros((B_eff, N_q, D), device=q_f.device, dtype=torch.float32)
    dK = torch.zeros((B_eff, N_k, D), device=q_f.device, dtype=torch.float32)
    dV = torch.zeros((B_eff, N_k, D), device=q_f.device, dtype=torch.float32)

    n_blocks = triton.cdiv(N_k, BLOCK_N)
    grid = (B_eff, n_blocks)

    NUM_Q_BLOCKS = triton.cdiv(N_q, BLOCK_M)
    BLK_SLICE_FACTOR = 2

    _attn_bwd[grid](
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
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        NUM_Q_BLOCKS=NUM_Q_BLOCKS,
        num_warps=4,
        num_stages=3,
    )

    return dQ, dK, dV


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        # Triton implementation here
        had_heads = q.dim() == 4
        if had_heads:
            B, H, N_q, D = q.shape
            _, _, N_k, _ = k.shape
            # Fold heads into batch for the block-wise implementation
            q = q.reshape(B * H, N_q, D)
            k = k.reshape(B * H, N_k, D)
            v = v.reshape(B * H, N_k, D)
        else:
            B, N_q, D = q.shape
            _, N_k, _ = k.shape

        # Triton forward produces fp32 O and fp32 LSE
        q = q.contiguous().float()
        k = k.contiguous().float()
        v = v.contiguous().float()

        O, L = flash_fwd_triton(q, k, v, is_causal)

        ctx.save_for_backward(q, k, v, O, L)

        ctx.is_causal = is_causal
        ctx.had_heads = had_heads
        if had_heads:
            (
                ctx.B,
                ctx.H,
            ) = B, H

        if had_heads:
            O = O.reshape(B, H, N_q, D)
            L = L.reshape(B, H, N_q)
        return O

    @staticmethod
    def backward(ctx, grad_O: torch.Tensor):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        had_heads = ctx.had_heads

        if had_heads:
            B, H = ctx.B, ctx.H
            B_eff = B * H
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]
            grad_O_ = grad_O.reshape(B_eff, N_q, D).contiguous()
        else:
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]
            grad_O_ = grad_O

        # Triton backward expects CUDA
        if not (q.is_cuda and grad_O_.is_cuda):
            raise RuntimeError("This Triton backward requires CUDA tensors.")

        # Ensure fp32 dO for stable math
        # dO = grad_O_.to(torch.float32)
        dQ, dK, dV = flash_backward_triton(q, k, v, O, L, grad_O_, is_causal=is_causal)

        # Cast back to input dtype (match PyTorch autograd expectations)
        grad_q = dQ.to(dtype=torch.float32)
        grad_k = dK.to(dtype=torch.float32)
        grad_v = dV.to(dtype=torch.float32)

        if had_heads:
            grad_q = grad_q.reshape(B, H, N_q, D)
            grad_k = grad_k.reshape(B, H, N_k, D)
            grad_v = grad_v.reshape(B, H, N_k, D)

        return grad_q, grad_k, grad_v, None
