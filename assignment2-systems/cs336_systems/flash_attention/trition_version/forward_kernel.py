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
