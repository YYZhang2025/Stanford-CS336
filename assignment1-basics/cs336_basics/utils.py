import gc
import json
import random
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
