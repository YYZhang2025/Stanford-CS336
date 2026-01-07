import gc
import json
import os
import random
import typing
from dataclasses import asdict

import numpy as np
import torch
from rich import print


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_color(content: str, color: str = "green"):
    print(f"[{color}]{content}[/{color}]")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    iteration,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    verbose: bool = True,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state, out)

    if verbose:
        print_color(f"Checkpoint saved to {out}", "blue")


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model, optimizer, verbose: bool = True
) -> int:
    state = torch.load(src, map_location=get_device())

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    if verbose:
        print_color(f"Checkpoint loaded from {src}", "blue")

    return state["iteration"]
