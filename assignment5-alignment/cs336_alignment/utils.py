import time
from pathlib import Path

import regex as re


def safe_slug(s: str) -> str:
    # Replace path separators and any weird chars with '-'
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s.replace("/", "-").replace("\\", "-"))


def get_run_name(prefix: str, config):
    date = time.strftime("%m%d-%H%M%S")
    return f"{prefix}-{safe_slug(config.model_name)}-{config.num_example}-{config.data_path.split('/')[2]}-{date}"


def save_model_and_tokenizer(model, tokenizer, experiment_name):
    out_dir = Path(f"./experiments/{experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Model and tokenizer saved to {out_dir}")
