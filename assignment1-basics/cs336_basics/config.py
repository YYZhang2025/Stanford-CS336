import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Union


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    max_seq_len: int = 512

    d_model: int = 512
    d_ff: int = 2048

    num_heads: int = 8
    num_layers: int = 6

    dropout: float = 0.1

    use_rms_norm: bool = True
    pre_norm: bool = True

    # Special token IDs
    eos_token_id: int = 2
    pad_token_id: int = 0

    # RoPE
    use_rope: bool = True
    rope_theta: float = 10000.0

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """Load a ModelConfig from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        """Create a ModelConfig from a mapping, ignoring unknown keys."""
        allowed = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this config to a plain Python dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write this config to a JSON file."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)


@dataclass
class TrainingConfig:
    batch_size: int = 64

    # Learning rate scheduler parameters
    lr_scheduler_type: str = "linear"  # Options: "linear", "cos
    learning_rate: float = 0.001
    warmup_steps: int = 500

    # AdamW related parameters
    betas: tuple = field(default=(0.9, 0.98))
    weight_decay: float = 1e-5

    # WandB logging flag
    wandb_logging: bool = False
    log_interval: int = 100

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        """Load a TrainingConfig from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        """Create a TrainingConfig from a mapping, ignoring unknown keys."""
        allowed = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this config to a plain Python dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write this config to a JSON file."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
