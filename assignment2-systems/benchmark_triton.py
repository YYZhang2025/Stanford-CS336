import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from cs336_systems.flash_attention.pytorch_version import FlashAttention as PyTorchFA
from cs336_systems.flash_attention.triton_version import get_flash_attention_triton


def benchmark(description: str, fn, num_warmup: int, num_iters: int) -> float:
    # warmup
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    avg = float(np.mean(times))
    print(f"{description}: {avg * 1000:.3f} ms/run ({num_iters} iters)")
    return avg


def attention_fn_wrapper(attn_apply, q, k, v, is_causal: bool):
    def wrapped():
        # Triton versions usually have signature (q, k, v, is_causal)
        return attn_apply(q, k, v, is_causal)

    return wrapped


def sdpa_wrapper(q, k, v, is_causal: bool):
    # PyTorch official SDPA
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This Triton benchmark requires CUDA."
    device = torch.device("cuda")

    # -----------------------------
    # Benchmark shapes
    # -----------------------------
    B = 32
    H = 16
    D = 64
    n_qs = [512, 1024, 2048, 4096]
    is_causal = True

    # 你要求：float32
    dtype = torch.float32

    N_WARMUP = 10
    N_ITERS = 50

    TritonFA_V1 = get_flash_attention_triton(version=1)
    TritonFA_V2 = get_flash_attention_triton(version=2)
    TritonFA_V3 = get_flash_attention_triton(version=3)

    # -----------------------------
    # Implementations to compare
    # -----------------------------
    impls = [
        ("Triton FA v1", TritonFA_V1.apply),
        ("Triton FA v2", TritonFA_V2.apply),
        ("Triton FA v3", TritonFA_V3.apply),
        ("PyTorch FlashAttention(ref)", PyTorchFA.apply),
        ("PyTorch SDPA", sdpa_wrapper),
    ]

    # records for plotting
    rows = []

    for n_q in n_qs:
        q = torch.randn(B, H, n_q, D, device=device, dtype=dtype)
        k = torch.randn(B, H, n_q, D, device=device, dtype=dtype)
        v = torch.randn(B, H, n_q, D, device=device, dtype=dtype)

        # 可选：确保 contiguous（对 Triton 很重要）
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        for impl_name, fn in impls:
            t = benchmark(
                f"{impl_name} (N_q={n_q})",
                attention_fn_wrapper(fn, q, k, v, is_causal),
                num_warmup=N_WARMUP,
                num_iters=N_ITERS,
            )
            rows.append({"N_q": str(n_q), "impl": impl_name, "ms": t * 1000.0})

        print("-" * 80)

    # -----------------------------
    # Plot
    # -----------------------------
    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 5))

    sns.barplot(
        data=df,
        x="N_q",
        y="ms",
        hue="impl",
        dodge=True,
        edgecolor="0.2",
        linewidth=0.8,
        errorbar=None,
        ax=ax,
    )

    ax.set_xlabel("N_q")
    ax.set_ylabel("Time (ms / run)")
    ax.set_title(f"Attention benchmark [dtype={dtype}]")

    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=10)

    sns.despine(ax=ax)
    fig.tight_layout()

    out = f"attention_benchmark_triton_versions_{str(dtype).split('.')[-1]}.png"
    plt.savefig(out, dpi=200)
    plt.show()
    plt.close()
    print(f"Saved plot to: {out}")
