import gc
import os
import random
import typing
from contextlib import nullcontext

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


def get_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        if verbose:
            print_color("Using CUDA device", "blue")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        if verbose:
            print_color("Using MPS device", "blue")
        return torch.device("mps")
    else:
        if verbose:
            print_color("Using CPU device", "blue")
        return torch.device("cpu")


def print_color(content: str, color: str = "green"):
    print(f"[{color}]{content}[/{color}]")


def get_ctx(use_mixed: bool, device: torch.device, amp_mode: str = "auto", verbose: bool = False):
    if not use_mixed or amp_mode == "off":
        if verbose:
            print("Not using autocast context")
        return nullcontext()

    dev = device.type

    if amp_mode == "fp16":
        dtype = torch.float16
    elif amp_mode == "bf16":
        dtype = torch.bfloat16
    else:
        if dev == "cuda":
            dtype = torch.bfloat16
        elif dev == "mps":
            dtype = torch.float16
        elif dev == "cpu":
            dtype = torch.float16
        else:
            return nullcontext()

    if verbose:
        print(f"Using autocast with dtype={dtype} on device={dev}")
    return torch.autocast(device_type=dev, dtype=dtype)


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
