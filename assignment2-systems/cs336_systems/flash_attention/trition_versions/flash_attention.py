import torch

from .forward_backward_version1 import _flash_backward_triton, _flash_forward_triton


# ----------------------------
# Autograd wrapper
# ----------------------------
class FlashAttention(torch.autograd.Function):
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

        # Triton forward produces fp32 O and fp32 LSE
        q_ = q_.contiguous()
        k_ = k_.contiguous()
        v_ = v_.contiguous()

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
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        had_heads = ctx.had_heads

        if had_heads:
            B, H = ctx.B, ctx.H
            B_eff = B * H
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]
            grad_O_ = grad_O.reshape(B_eff, N_q, D)
        else:
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]
            grad_O_ = grad_O

        # Triton backward expects CUDA
        if not (q.is_cuda and grad_O_.is_cuda):
            raise RuntimeError("This Triton backward requires CUDA tensors.")

        # Ensure fp32 dO for stable math
        dO = grad_O_.to(torch.float32)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        O = O.contiguous()
        L = L.contiguous()
        grad_O_ = grad_O_.contiguous()

        dO = grad_O_.to(torch.float32)
        q_f = q.to(torch.float32)
        k_f = k.to(torch.float32)
        v_f = v.to(torch.float32)
        dQ, dK, dV = _flash_backward_triton(q_f, k_f, v_f, O, L, dO, is_causal=is_causal)

        # Cast back to input dtype (match PyTorch autograd expectations)
        grad_q = dQ.to(dtype=torch.float32)
        grad_k = dK.to(dtype=torch.float32)
        grad_v = dV.to(dtype=torch.float32)

        if had_heads:
            grad_q = grad_q.reshape(B, H, N_q, D)
            grad_k = grad_k.reshape(B, H, N_k, D)
            grad_v = grad_v.reshape(B, H, N_k, D)

        return grad_q, grad_k, grad_v, None
