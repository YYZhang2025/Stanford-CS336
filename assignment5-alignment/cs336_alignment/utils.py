import gc
import json
from contextlib import nullcontext

import rich
import torch


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


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


def print_rich_dict(data: dict) -> None:
    from rich.pretty import pprint

    """Pretty print dictionary with colors using rich."""
    pprint(data, expand_all=True)


def get_ctx(use_mixed: bool, device: torch.device, verbose: bool = True):
    if use_mixed and device.type == "cuda":
        if verbose:
            print_color("Using mixed precision on CUDA with BFloat16", "blue")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif use_mixed and device.type == "mps":
        if verbose:
            print_color("Using mixed precision on MPS with Float16", "blue")
        return torch.autocast(device_type="mps", dtype=torch.float16)
    elif use_mixed and device.type == "cpu":
        if verbose:
            print_color("Using mixed precision on CPU with Float16", "blue")
        return torch.autocast(device_type="cpu", dtype=torch.float16)
    else:
        if verbose:
            print_color("Not using mixed precision", "blue")
        return nullcontext()


def get_device(verbose: bool = True, rank: int = 0, use_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        if verbose:
            print_color(f"Using CUDA device cuda:{rank}", "blue")
        return torch.device(f"cuda:{rank}")
    elif use_mps and torch.backends.mps.is_available():
        if verbose:
            print_color("Using MPS device", "blue")
        return torch.device("mps")
    else:
        if verbose:
            print_color("Using CPU device", "blue")
        return torch.device("cpu")


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cur_step: int | None,
    checkpoint_path: str,
):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if cur_step is not None:
        state["cur_step"] = cur_step

    torch.save(state, checkpoint_path)
    print_color(f"Saved model checkpoint to {checkpoint_path}", "cyan")


def seed_everything(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_cot_with_answer(cot: str, answer: str) -> str:
    return f"{cot}\n</think> <answer>{str(answer)}</answer>"


def load_dataset(path: str, prompt_template_path: str = ""):
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    prompts = []
    cots = []
    answers = []
    for row in rows:
        prompts.append(prompt_template.format(question=row["question"]))
        cots.append(row["cot"])
        answers.append(row["answer"])

    return prompts, cots, answers
