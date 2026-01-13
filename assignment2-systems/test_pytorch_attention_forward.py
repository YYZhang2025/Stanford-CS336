import rich
import torch

from cs336_systems.flash_attention.pytorch_version import FlashAttention

if __name__ == "__main__":
    import rich

    # If you put this in triton_version.py, FlashAttention here is the Triton one.
    # If you put this in a separate script, do:
    # from cs336_systems.flash_attention.triton_version import FlashAttention as TritonFlashAttention
    # and replace FlashAttention.apply -> TritonFlashAttention.apply below.

    # if not torch.cuda.is_available():
    #     raise RuntimeError("This Triton test requires CUDA.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    B = 8
    H = 10
    N_q = 128
    N_k = 128
    D = 64
    is_causal = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16  # bfloat16 also OK if your GPU supports it

    q = torch.randn(B, H, N_q, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N_k, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N_k, D, device=device, dtype=dtype, requires_grad=True)

    # -------- Forward --------
    out_impl = FlashAttention.apply(q, k, v, is_causal)

    # Reference (PyTorch SDPA)
    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)

    # Triton forward in this implementation returns fp32 O, while ref returns dtype.
    # Compare in fp32 for fairness.
    rich.print("[blue]Test Forward pass:[/blue]")
    assert torch.allclose(out_impl.float(), out_ref.float(), atol=1e-2, rtol=1e-2)
    rich.print("[green] Forward pass test passed![/green]")

    # -------- Backward --------
    loss_impl = out_impl.sum()
    loss_ref = out_ref.sum()

    loss_impl.backward()
    q_grad_impl = q.grad.detach().clone()
    k_grad_impl = k.grad.detach().clone()
    v_grad_impl = v.grad.detach().clone()

    # Clear grads
    q.grad = None
    k.grad = None
    v.grad = None

    loss_ref.backward()
    q_grad_ref = q.grad.detach().clone()
    k_grad_ref = k.grad.detach().clone()
    v_grad_ref = v.grad.detach().clone()

    rich.print("[blue]Test Backward pass:[/blue]")
    assert torch.allclose(q_grad_impl.float(), q_grad_ref.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(k_grad_impl.float(), k_grad_ref.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(v_grad_impl.float(), v_grad_ref.float(), atol=1e-2, rtol=1e-2)
    rich.print("[green] Backward pass test passed![/green]")
