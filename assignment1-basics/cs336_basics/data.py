from dataclasses import dataclass

import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    x_t = torch.as_tensor(x)

    # Valid start indices t satisfy: t + context_length < n  =>  t in [0, n-context_length-1]
    # torch.randint uses an exclusive high bound.
    starts = torch.randint(0, n - context_length, (batch_size,), dtype=torch.long)

    offsets = torch.arange(context_length, dtype=torch.long).unsqueeze(0)  # (1, m)
    idx = starts.unsqueeze(1) + offsets  # (B, m)

    inputs = x_t[idx]
    targets = x_t[idx + 1]

    # Move to requested device
    inputs = inputs.to(device)
    targets = targets.to(device)

    return inputs, targets


def data_loading(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(x, batch_size, context_length, device)


# --- Sequential/traversal batching ---


@dataclass
class BatchState:
    pos: int = 0


def get_batch_sequential(
    x_t: torch.Tensor | np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    state: BatchState,
    *,
    stride: int | None = None,
):
    if stride is None:
        stride = context_length

    n = x_t.numel()
    max_start = n - context_length - 1
    if max_start < 0:
        raise ValueError(f"Sequence too short: n={n}, context_length={context_length}")

    # Avoid per-sample modulo wrap. If we would run off the end, reset cursor.
    last_start = state.pos + (batch_size - 1) * stride
    end = last_start + context_length + 1
    if end > n:
        state.pos = 0
        last_start = (batch_size - 1) * stride
        end = last_start + context_length + 1

    base = x_t[state.pos : end]  # 1D contiguous slice

    # 2D views: (B, T). Strides are in *elements* for PyTorch tensors.
    inputs = base.as_strided(size=(batch_size, context_length), stride=(stride, 1))
    targets = base[1:].as_strided(size=(batch_size, context_length), stride=(stride, 1))

    state.pos += batch_size * stride

    # Transfer + cast (cast happens AFTER transfer => cheaper for CPU)
    if (isinstance(device, torch.device) and device.type == "cuda") or (
        isinstance(device, str) and "cuda" in device.lower()
    ):
        inputs = inputs.to(device, non_blocking=True).long()
        targets = targets.to(device, non_blocking=True).long()
    else:
        inputs = inputs.long().to(device)
        targets = targets.long().to(device)

    return inputs, targets


def data_loading_sequential(
    x: np.ndarray | torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    state: BatchState,
    *,
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch_sequential(x, batch_size, context_length, device, state, stride=stride)
