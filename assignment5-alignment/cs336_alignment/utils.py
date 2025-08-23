import gc
import json
import time
from pathlib import Path

import regex as re
import torch
from rich import print


def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def safe_slug(s: str) -> str:
    # Replace path separators and any weird chars with '-'
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s.replace("/", "-").replace("\\", "-"))


def get_run_name(prefix: str, config):
    date = time.strftime("%m%d-%H%M%S")
    return f"{prefix}-{safe_slug(config.model_name)}-{config.num_example}-{config.data_path.split('/')[2]}-{date}"


def save_model_and_tokenizer(model, tokenizer, config):
    out_dir = Path(f"./{config.experiment_name_base}/{config.experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Model and tokenizer saved to {out_dir}")


def print_formatted_dict(data: dict) -> None:
    """Pretty print a dictionary with indentation and color-friendly formatting."""
    print(json.dumps(data, indent=4, ensure_ascii=False))


def print_rich_dict(data: dict) -> None:
    """Pretty print dictionary with colors using rich."""
    from rich.pretty import pprint

    pprint(data, expand_all=True)


def print_color(text: str, color: str = "red"):
    print(f"[{color}]{text}[/{color}]")


def cycle_dataloader(dataloader):
    """
    Creates a cycling iterator for a PyTorch DataLoader.
    """
    while True:
        for batch in dataloader:
            yield batch
