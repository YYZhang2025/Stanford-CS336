from __future__ import annotations

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
    x_t = torch.as_tensor(x, dtype=torch.long)

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
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    state: BatchState,
    *,
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    if stride is None:
        stride = context_length

    # Max valid start t satisfies: t + context_length < n
    max_start = n - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"Sequence too short: n={n}, context_length={context_length}. Need n >= context_length+1."
        )

    x_t = torch.as_tensor(x, dtype=torch.long)

    # Deterministic starts based on current cursor.
    starts = state.pos + torch.arange(batch_size, dtype=torch.long) * int(stride)
    starts = starts % (max_start + 1)

    offsets = torch.arange(context_length, dtype=torch.long).unsqueeze(0)  # (1, m)
    idx = starts.unsqueeze(1) + offsets  # (B, m)

    inputs = x_t[idx]
    targets = x_t[idx + 1]

    # Advance cursor by a whole batch.
    state.pos = int((state.pos + batch_size * int(stride)) % (max_start + 1))

    return inputs.pin_memory().to(device, non_blocking=True), targets.pin_memory().to(
        device, non_blocking=True
    )


def data_loading_sequential(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    state: BatchState,
    *,
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch_sequential(x, batch_size, context_length, device, state, stride=stride)
