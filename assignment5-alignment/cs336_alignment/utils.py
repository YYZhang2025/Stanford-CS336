from contextlib import nullcontext

import rich
import torch


def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.float().item()
    elif isinstance(x, str):
        return float(x.strip())

    return float(x)


def cycle_dataloader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def print_color(text: str, color: str = "red"):
    rich.print(f"[{color}]{text}[/{color}]")


def get_ctx(use_mixed: bool, device: torch.device, verbose: bool = True):
    if use_mixed and device.type == "cuda":
        if verbose:
            print_color("Using mixed precision on CUDA with BFloat16", "blue")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        if verbose:
            print_color("Not using mixed precision", "blue")
        return nullcontext()


def get_device(verbose: bool = True, use_mps: bool = False) -> torch.device:
    if torch.cuda.is_available():
        if verbose:
            print_color("Using CUDA device", "blue")
        return torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        if verbose:
            print_color("Using MPS device", "blue")
        return torch.device("mps")
    else:
        if verbose:
            print_color("Using CPU device", "blue")
        return torch.device("cpu")
