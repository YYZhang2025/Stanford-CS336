import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DATASET_NAME_TO_PATH = {
    "math": "./data/math/processed",
    "gsm8k": "./data/gsm8k",
    "alpaca_eval": "./data/alpaca_eval",
    "mmlu": "./data/mmlu",
}


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    # Choices: "math", "gsm8k", "alpaca_eval", "mmlu"
    dataset_name: str = "math"
    dataset_path: str = ""  # will be set in __post_init__

    # WanDB logging
    wandb_logging: bool = True
    project_name: str = "assignment05-alignment"
    run_name: str = ""

    # Training hyperparameters
    batch_size: int = 8
    total_training_steps: int = 10
    gradient_accumulation_steps: int = 2
    betas: tuple = field(default=(0.9, 0.98))
    weight_decay: float = 1e-5
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    mixed_precision_training: bool = True

    seed: int = 42

    def __post_init__(self):
        if self.dataset_name not in DATASET_NAME_TO_PATH:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")
        self.dataset_path: str = DATASET_NAME_TO_PATH[self.dataset_name]

        self.run_name = f"{self.model_name.split('/')[-1]}_dataset({self.dataset_name})"

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        allowed = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
