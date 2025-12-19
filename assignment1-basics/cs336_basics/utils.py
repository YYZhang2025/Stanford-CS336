import gc
import random

import numpy as np
import torch

import json
from dataclasses import asdict


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



def save_config(config: object, filepath: str | ) -> None:
    with open(filepath, "w") as f:
        json.dump(asdict(config), f, indent=4)
        
    