from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    max_seq_length: int = 512

    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 2048
    ffn_act_fn: str = "gelu"

    dropout: float = 0.1

    use_rms_norm: bool = True
    pre_norm: bool = True

    # Special token IDs
    eos_token_id: int = 2
    pad_token_id: int = 0


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
