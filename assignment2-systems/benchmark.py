import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from cs336_systems.flash_attention.pytorch_version import FlashAttention
from cs336_systems.flash_attention.triton_version import FlashAttention as TritonFlashAttention


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool | None = None,
) -> torch.Tensor:
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)
    device = query.device

    if is_causal:
        q_len, k_len = query.size(-2), key.size(-2)
        causal_mask = torch.arange(q_len).unsqueeze(1) >= torch.arange(k_len).unsqueeze(0)
        causal_mask = causal_mask.to(device)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


def benchmark(
    description: str,
    attention_fn,
    num_warmsup: int,
    num_iters: int,
):
    # Warm-up
    for _ in range(num_warmsup):
        attention_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        start_time = time.time()
        attention_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"{description}: {avg_time * 1000:.3f} ms per run over {num_iters} runs")
    return avg_time


def attention_fn_wrapper(attention_fn, *args, **kwargs):
    def wrapped():
        return attention_fn(*args, **kwargs)

    return wrapped


if __name__ == "__main__":
    if torch.cuda.is_available():
        B = 32  # batch size
        H = 16  # number of heads
        D = 64  # head dimension
        n_qs = [512, 1024, 2048, 4096]

    else:
        B = 8  # batch size
        H = 4  # number of heads
        D = 10  # head dimension
        n_qs = [64, 128, 256, 512]

    is_causal = True
    N_WARMUP = 10
    N_ITERS = 50

    pytorch_flash_attention_times = []
    naive_pytorch_attention_times = []
    triton_flash_attention_times = []
    pytorch_sdpa_times = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    for n_q in n_qs:
        q = torch.randn(B, H, n_q, D, device=device, dtype=dtype)
        k = torch.randn(B, H, n_q, D, device=device, dtype=dtype)
        v = torch.randn(B, H, n_q, D, device=device, dtype=dtype)

        naive_pytorch_attention_time = benchmark(
            f"Naive SDPA (N_q={n_q})",
            attention_fn_wrapper(scaled_dot_product_attention, q, k, v, is_causal),
            num_warmsup=N_WARMUP,
            num_iters=N_ITERS,
        )

        pytorch_flash_attention_time = benchmark(
            f"PyTorch Flash Attention (N_q={n_q})",
            attention_fn_wrapper(FlashAttention.apply, q, k, v, is_causal),
            num_warmsup=N_WARMUP,
            num_iters=N_ITERS,
        )

        triton_flash_attention_time = benchmark(
            f"Triton Flash Attention (N_q={n_q})",
            attention_fn_wrapper(TritonFlashAttention.apply, q, k, v, is_causal),
            num_warmsup=N_WARMUP,
            num_iters=N_ITERS,
        )

        # official_flash_attention_time = benchmark(
        #     f"Official Flash Attention (N_q={n_q})",
        #     attention_fn_wrapper(scaled_dot_product_attention, q, k, v, is_causal),
        #     num_warmsup=N_WARMUP,
        #     num_iters=N_ITERS,
        # )

        pytorch_sdpa_time = benchmark(
            f"PyTorch SDPA Attention (N_q={n_q})",
            attention_fn_wrapper(
                torch.nn.functional.scaled_dot_product_attention, q, k, v, is_causal=is_causal
            ),
            num_warmsup=N_WARMUP,
            num_iters=N_ITERS,
        )

        pytorch_flash_attention_times.append(pytorch_flash_attention_time)
        naive_pytorch_attention_times.append(naive_pytorch_attention_time)
        triton_flash_attention_times.append(triton_flash_attention_time)
        # official_flash_attention_times.append(official_flash_attention_time)
        pytorch_sdpa_times.append(pytorch_sdpa_time)

        print("-" * 80)

    # -------- Plot Results --------
    t_pytorch_flash_ms = (np.array(pytorch_flash_attention_times) * 1000.0).tolist()
    t_sdpa_ms = (np.array(pytorch_sdpa_times) * 1000.0).tolist()
    # t_official_flash_ms = (np.array(official_flash_attention_times) * 1000.0).tolist()
    t_triton_flash_ms = (np.array(triton_flash_attention_times) * 1000.0).tolist()
    t_naive_pytorch_ms = (np.array(naive_pytorch_attention_times) * 1000.0).tolist()

    impls = [
        ("Naive SDPA", t_naive_pytorch_ms),
        ("PyTorch FlashAttention", t_pytorch_flash_ms),
        ("Triton Flash Attention", t_triton_flash_ms),
        # ("Official Flash Attention", t_official_flash_ms),
        ("PyTorch SDPA", t_sdpa_ms),
    ]

    rows = []
    for n_q, i in zip(n_qs, range(len(n_qs))):
        for impl_name, ys in impls:
            rows.append({"N_q": str(n_q), "impl": impl_name, "ms": float(ys[i])})

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
    ax.set_title(f"Attention benchmark [{str(dtype).split('.')[-1]}]")

    # Put legend outside for cleanliness
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Annotate bars with values
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=10)

    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    plt.savefig(f"attention_benchmark_{str(dtype).split('.')[-1]}.png")
    plt.close()

torch.int8
