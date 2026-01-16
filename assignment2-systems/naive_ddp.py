import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# -----------------------------
# Config
# -----------------------------
WORLD_SIZE = 4
SINGLE_MATCH_BATCH_SIZE = 8  # per-rank batch size; global batch = WORLD_SIZE * SINGLE_MATCH_BATCH_SIZE


@dataclass
class TrainConfig:
    steps: int = 1
    lr: float = 1e-3
    seed: int = 0


# -----------------------------
# Model
# -----------------------------
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 100)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.norm(out)
        out = self.linear2(out)
        return out


# -----------------------------
# Distributed helpers
# -----------------------------


@torch.no_grad()
def broadcast_model_from_rank0(model: nn.Module) -> None:
    """Broadcast parameters (and buffers) from rank 0 to all other ranks."""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)


def average_gradients(model: nn.Module, world_size: int) -> None:
    """All-reduce gradients across ranks and average them in-place."""
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


# -----------------------------
# Naive DDP training loop
# -----------------------------
def naive_ddp_worker(rank: int, world_size: int, cfg: TrainConfig) -> None:
    setup_process_group(rank, world_size)

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 0) Each rank constructs a randomly-initialized model (intentionally different)
    torch.manual_seed(1234 + rank)
    model = ToyModel().to(device)

    # 0.5) Broadcast params from rank 0 so everybody starts identical
    broadcast_model_from_rank0(model)

    # Save the identical initial weights for a single-process reference run
    # (all ranks have the same weights after broadcast; only rank0 needs the copy)
    init_state = None
    if rank == 0:
        init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Create optimizer AFTER broadcast so all ranks start from identical params
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()  # reduction='mean' by default

    # ------------------------------------------------------------
    # Build ONE global batch (size = SINGLE_MATCH_BATCH_SIZE * world_size) on rank0,
    # shard it across ranks using scatter, do ONE DDP update.
    # ------------------------------------------------------------
    torch.manual_seed(cfg.seed)
    global_bs = SINGLE_MATCH_BATCH_SIZE * world_size

    if rank == 0:
        x_global = torch.randn(global_bs, 10, device=device)
        y_global = torch.randn(global_bs, 10, device=device)
        x_chunks = list(x_global.chunk(world_size, dim=0))
        y_chunks = list(y_global.chunk(world_size, dim=0))
    else:
        x_global = None
        y_global = None
        x_chunks = None
        y_chunks = None

    # Each rank receives exactly its shard (n/d examples)
    x_local = torch.empty((SINGLE_MATCH_BATCH_SIZE, 10), device=device)
    y_local = torch.empty((SINGLE_MATCH_BATCH_SIZE, 10), device=device)
    dist.scatter(x_local, scatter_list=x_chunks, src=0)
    dist.scatter(y_local, scatter_list=y_chunks, src=0)

    # 2) Local forward + backward on the shard
    optim.zero_grad(set_to_none=True)
    out = model(x_local)
    loss_local = loss_fn(out, y_local)
    loss_local.backward()

    # 3) All-reduce to average gradients across ranks
    #    With per-rank MSELoss(mean), averaging grads across ranks equals the
    #    gradient of MSELoss(mean) on the full global batch.
    average_gradients(model, world_size)

    # 4) Optimizer step (every rank applies the same averaged grads)
    optim.step()

    # ------------------------------------------------------------
    # Reference: single-process update on rank0 using the FULL global batch,
    # then verify DDP params match the reference params.
    # ------------------------------------------------------------
    # Create a reference model on ALL ranks so we can broadcast the reference weights.
    ref_model = ToyModel().to(device)
    broadcast_model_from_rank0(ref_model)  # temp sync so shapes/buffers are consistent

    if rank == 0:
        assert init_state is not None
        ref_model.load_state_dict(init_state)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=cfg.lr)

        # full-batch forward/backward/step
        ref_optim.zero_grad(set_to_none=True)
        out_full = ref_model(x_global)  # type: ignore[arg-type]
        loss_ref = loss_fn(out_full, y_global)  # type: ignore[arg-type]
        loss_ref.backward()
        ref_optim.step()

    # Broadcast reference parameters from rank0 so all ranks can compare locally
    broadcast_model_from_rank0(ref_model)

    # Compute max absolute parameter difference between DDP model and reference model
    with torch.no_grad():
        max_diff = torch.tensor(0.0, device=device)
        for p, rp in zip(model.parameters(), ref_model.parameters()):
            max_diff = torch.maximum(max_diff, (p - rp).abs().max())

        # Also compute a cheap checksum to ensure all ranks match each other
        checksum = torch.tensor(0.0, device=device)
        for p in model.parameters():
            checksum += p.data.float().sum()

    # Reduce max diff across ranks (should be ~0)
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)

    # Gather checksums to rank0
    checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
    dist.all_gather(checksums, checksum)

    if rank == 0:
        checksums_list = [c.item() for c in checksums]
        print(f"local_loss(rank0)={loss_local.item():.6f}")
        print(f"max_param_abs_diff_vs_reference={max_diff.item():.6e}")
        print(f"param_checksum_per_rank={['%.6f' % v for v in checksums_list]}")

        assert max_diff.item() < 1e-6, "DDP parameters do not match reference!"
        print("PASS: DDP parameters match reference implementation.")

    cleanup_process_group()


def main() -> None:
    cfg = TrainConfig()

    # On macOS, spawn is the safest start method.
    mp.set_start_method("spawn", force=True)

    mp.spawn(
        naive_ddp_worker,
        args=(WORLD_SIZE, cfg),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
