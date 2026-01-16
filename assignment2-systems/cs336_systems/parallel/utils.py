import os

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_process_group(rank: int, world_size: int) -> None:
    """Initialize a local multi-process process group."""
    # Use a fixed localhost init for simplicity (works for single-node training)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    # Pick backend based on availability
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def broadcast_model_from_rank0(model: nn.Module) -> None:
    """Broadcast parameters (and buffers) from rank 0 to all other ranks."""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)
