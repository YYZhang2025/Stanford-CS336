from __future__ import annotations

import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (inputs, targets) from a 1D token stream.

    Given a single tokenized sequence x = (x1, ..., xn) as a 1D numpy integer array,
    we sample `batch_size` starting positions and return:
      - inputs:  x[t : t+context_length]
      - targets: x[t+1 : t+context_length+1]

    Both outputs have shape (batch_size, context_length), dtype torch.long, and are
    moved to `device`.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D numpy array of token ids, got shape {x.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    n = int(x.shape[0])
    # Need at least one next-token target for each position in the context window.
    if n <= context_length:
        raise ValueError(f"Token stream too short: len(x)={n} must be > context_length={context_length}")

    # Convert once on CPU; indexing happens on CPU and then we move to device.
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


# Backwards-compatible alias (some scaffolding/tests may look for this name).
# If your assignment expects a different function name, you can still call get_batch.


def data_loading(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(x, batch_size, context_length, device)
