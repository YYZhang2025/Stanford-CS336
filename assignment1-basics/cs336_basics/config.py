import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from cs336_basics.utils import get_device


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    max_seq_len: int = 256

    d_model: int = 512
    d_ff: int = 1344

    num_heads: int = 16
    num_layers: int = 4

    dropout: float = 0.1

    use_rms_norm: bool = True
    pre_norm: bool = True

    # Special token IDs
    eos_token_id: int = 256

    # RoPE
    use_rope: bool = True
    rope_theta: float = 10000.0

    # MoE specific parameters
    use_moe: bool = False
    num_experts: int = 4
    top_k: int = 1
    router_jitter: float = 0.1
    z_loss_coef: float = 1e-3
    lb_loss_coef: float = 1e-1

    # Others
    tie_weights: bool = False
    use_final_norm: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        allowed = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)


@dataclass
class TrainingConfig:
    batch_size: int = 256
    num_steps: int = 10_000
    dataset_dir: str = "datasets/tiny_stories"
    train_data_path: str = "datasets/tiny_stories/train.bin"
    eval_data_path: str = "datasets/tiny_stories/eval.bin"

    # Optimizer related parameters
    betas: tuple = field(default=(0.9, 0.98))
    weight_decay: float = 1e-5
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Logging & checkpointing
    wandb_logging: bool = True
    eval_log_interval: int = 500
    sampling_log_interval: int = 200

    # Others:
    model_name: str = "tiny_stories_transformer"
    save_checkpoint_dir: str = "checkpoints"
    device: str = get_device().type
    debug_mode: bool = False
    use_mixed_precision: bool = True
    log_moe_every: int = 500
    seed: int = 2025

    def __post_init__(self):
        # Validate lr_scheduler_type
        if self.debug_mode:
            self.num_steps = 100
            self.batch_size = 8
            self.log_moe_every = 10
            self.train_data_path = "datasets/tiny_stories/eval.bin"

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        data = dict(data)  # shallow copy，避免修改外部传入的 dict

        # ---- fix betas: JSON list -> tuple ----
        if "betas" in data:
            b = data["betas"]
            if isinstance(b, list):
                b = tuple(b)
            if not (isinstance(b, tuple) and len(b) == 2):
                raise ValueError(f"betas must be a tuple of length 2, got: {data['betas']}")
            data["betas"] = (float(b[0]), float(b[1]))

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
