import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    model_name: str = "./models/Qwen/Qwen2.5-Math-1.5B"
    prompt_template_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    # Choices: "math", "gsm8k",  "mmlu"
    dataset_base_path: str = "./data/pre-processed"
    dataset_name: str = "math"
    dataset_path: str = ""  # will be set in __post_init__

    # WanDB logging
    wandb_logging: bool = True
    project_name: str = "assignment05-alignment"
    run_name: str = ""

    # Training hyperparameters
    batch_size: int = 8
    total_training_steps: int = 2000
    gradient_accumulation_steps: int = 10
    betas: tuple = field(default=(0.9, 0.98))
    weight_decay: float = 1e-5
    max_lr: float = 1e-5
    min_lr: float = 1e-6
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    mixed_precision_training: bool = True

    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 2
    eval_batch_size: int = 4
    eval_steps: int = 5
    seed: int = 42
    sample_interval: int = 20

    def __post_init__(self):
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
