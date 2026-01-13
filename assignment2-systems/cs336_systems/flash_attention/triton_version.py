# cs336_systems/flash_attention/triton_version.py
import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


@triton.jit
def _flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
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

    # Online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N_K, BLOCK_N):
        k_ids = start_n + offs_n  # (BN,)

        # K: (BN, D)
        k_ptrs = K_ptr + pid_bh * stride_kb + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # scores: (BM, BN)
        scores = tl.dot(q, tl.trans(k)) * SCALE

        # causal mask
        if IS_CAUSAL:
            q_ids = offs_m
            causal = q_ids[:, None] >= k_ids[None, :]
            scores = tl.where(causal, scores, -float("inf"))

        # mask invalid keys
        scores = tl.where(k_ids[None, :] < N_K, scores, -float("inf"))

        # online softmax update
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])

        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p, axis=1)

        # V: (BN, D)
        v_ptrs = V_ptr + pid_bh * stride_vb + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(k_ids[:, None] < N_K) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # acc = alpha*acc + p@v
        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_ij
        l_i = l_ij

        # early break for causal
        if IS_CAUSAL:
            if start_n >= (pid_m + 1) * BLOCK_M:
                break

    o = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    o_ptrs = O_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o, mask=(offs_m[:, None] < N_Q) & (offs_d[None, :] < D))

    l_ptrs = L_ptr + pid_bh * stride_lb + offs_m * stride_lm
    tl.store(l_ptrs, lse, mask=(offs_m < N_Q))


def _flash_forward_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
    """q/k/v: (B_eff, N, D) -> returns (O, L) in fp32."""
    if triton is None:
        raise RuntimeError("Triton is not available in this environment")

    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton FlashAttention requires CUDA tensors"

    B_eff, N_q, D = q.shape
    N_k = k.shape[1]

    O = torch.empty((B_eff, N_q, D), device=q.device, dtype=torch.float32)
    L = torch.empty((B_eff, N_q), device=q.device, dtype=torch.float32)

    stride_qb, stride_qm, stride_qd = q.stride()
    stride_kb, stride_kn, stride_kd = k.stride()
    stride_vb, stride_vn, stride_vd = v.stride()
    stride_ob, stride_om, stride_od = O.stride()
    stride_lb, stride_lm = L.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    if D <= 16:
        BLOCK_D = 16
    elif D <= 32:
        BLOCK_D = 32
    elif D <= 64:
        BLOCK_D = 64
    else:
        BLOCK_D = 128

    grid = (B_eff, triton.cdiv(N_q, BLOCK_M))
    scale = 1.0 / math.sqrt(D)

    _flash_fwd_kernel[grid](
        q,
        k,
        v,
        O,
        L,
        stride_qb=stride_qb,
        stride_qm=stride_qm,
        stride_qd=stride_qd,
        stride_kb=stride_kb,
        stride_kn=stride_kn,
        stride_kd=stride_kd,
        stride_vb=stride_vb,
        stride_vn=stride_vn,
        stride_vd=stride_vd,
        stride_ob=stride_ob,
        stride_om=stride_om,
        stride_od=stride_od,
        stride_lb=stride_lb,
        stride_lm=stride_lm,
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


class FlashAttention(torch.autograd.Function):
    """Triton FlashAttention forward; backward uses PyTorch math for correctness."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        had_heads = q.dim() == 4
        if had_heads:
            B, H, N_q, D = q.shape
            _, _, N_k, _ = k.shape
            q_ = q.reshape(B * H, N_q, D)
            k_ = k.reshape(B * H, N_k, D)
            v_ = v.reshape(B * H, N_k, D)
        else:
            B, N_q, D = q.shape
            N_k = k.shape[1]
            q_, k_, v_ = q, k, v

        O, L = _flash_forward_triton(q_, k_, v_, is_causal=is_causal)

        ctx.save_for_backward(q_, k_, v_, O, L)
        ctx.is_causal = is_causal
        ctx.had_heads = had_heads
        if had_heads:
            ctx.B, ctx.H = B, H

        if had_heads:
            O = O.reshape(B, H, N_q, D)
        return O

    @staticmethod
    def backward(ctx, grad_O: torch.Tensor):
        # Backward matches your PyTorch reference math (correctness first).
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        had_heads = ctx.had_heads

        if had_heads:
            B, H = ctx.B, ctx.H
            B_eff = B * H
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]
            grad_O = grad_O.reshape(B_eff, N_q, D)
        else:
            B_eff, N_q, D = q.shape
            N_k = k.shape[1]

        device = q.device
        scale = 1.0 / math.sqrt(D)

        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        q_pos = torch.arange(N_q, device=device)
        k_pos = torch.arange(N_k, device=device)

        D_blk = torch.sum(grad_O * O, dim=-1)  # (B_eff, N_q)

        for i in range(T_q):
            q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)

            Q_blk_i = q[:, q_start:q_end, :]
            grad_O_blk_i = grad_O[:, q_start:q_end, :]
            L_blk_i = L[:, q_start:q_end]
            D_blk_i = D_blk[:, q_start:q_end]

            q_idx = q_pos[q_start:q_end]

            for j in range(T_k):
                k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)

                if is_causal and k_start >= q_end:
                    break

                k_blk_j = k[:, k_start:k_end, :]
                v_blk_j = v[:, k_start:k_end, :]

                scores_ij = (Q_blk_i.float() @ k_blk_j.float().transpose(-1, -2)) * scale
                if is_causal:
                    k_idx = k_pos[k_start:k_end]
                    causal_mask = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
                    scores_ij = scores_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                P_ij = torch.exp(scores_ij - L_blk_i.unsqueeze(-1))

                grad_v[:, k_start:k_end, :] += torch.bmm(
                    P_ij.transpose(1, 2),
                    grad_O_blk_i.float(),
                ).to(grad_v.dtype)

                grad_P = torch.bmm(
                    grad_O_blk_i.float(),
                    v_blk_j.float().transpose(1, 2),
                )

                grad_S = P_ij * (grad_P - D_blk_i.unsqueeze(-1))

                grad_q[:, q_start:q_end, :] += (torch.bmm(grad_S, k_blk_j.float()) * scale).to(grad_q.dtype)
                grad_k[:, k_start:k_end, :] += (
                    torch.bmm(grad_S.transpose(1, 2), Q_blk_i.float()) * scale
                ).to(grad_k.dtype)

        if had_heads:
            grad_q = grad_q.reshape(B, H, N_q, D)
            grad_k = grad_k.reshape(B, H, N_k, D)
            grad_v = grad_v.reshape(B, H, N_k, D)

        return grad_q, grad_k, grad_v, None
