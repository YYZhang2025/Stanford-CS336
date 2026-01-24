import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Reusable config base class.

    Subclasses should be `@dataclass`es. This base provides:
      - `from_json(path)` / `to_json(path)`
      - `from_dict(mapping)` / `to_dict()`

    By default, unknown keys in the JSON/dict are ignored.
    Set `strict=True` to raise on unknown keys.
    """

    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    prompt_template_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    # Dataset Choices: "math", "gsm8k",  "mmlu"
    dataset_base_path: str = "./data/pre-processed"
    dataset_name: str = "gsm8k"
    dataset_path: str = ""  # will be set in __post_init__

    # WanDB logging
    wandb_logging: bool = True
    project_name: str = "assignment05-alignment"
    run_name: str = ""  # will be set in __post_init__

    @classmethod
    def from_json(cls: type[T], path: str | Path, *, strict: bool = False) -> T:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected a JSON object at {path}, got {type(data).__name__}")
        return cls.from_dict(data, strict=strict)

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, Any], *, strict: bool = False) -> T:
        allowed = {f.name for f in fields(cls)}
        unknown = [k for k in data.keys() if k not in allowed]
        if strict and unknown:
            raise KeyError(f"Unknown config keys for {cls.__name__}: {unknown}")
        filtered: dict[str, Any] = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
