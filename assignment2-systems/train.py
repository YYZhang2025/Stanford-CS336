import os
import time

import dotenv
import fire

from cs336_basics.config import ModelConfig, TrainingConfig
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.train_engine import train
from cs336_basics.utils import get_device, print_color, seed_everything


def main(
    train_config_json: str | None = "./configs/pytorch_flash_attn/train_config.json",
    model_config_json: str | None = "./configs/pytorch_flash_attn/model_config.json",
):
    # Load configs
    train_config = TrainingConfig.from_json(train_config_json) if train_config_json else TrainingConfig()
    model_config = ModelConfig.from_json(model_config_json) if model_config_json else ModelConfig()
    # Save configs
    out_dir = os.path.join(
        train_config.save_checkpoint_dir,
        train_config.model_name,
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_config.to_json(os.path.join(out_dir, "model_config.json"))
    train_config.to_json(os.path.join(out_dir, "train_config.json"))

    train_config.wandb_logging = True

    train_config.device = get_device()

    # Load environment and set WanDB config
    dotenv.load_dotenv()
    wandb_api = os.getenv("WANDB_API_KEY")
    if train_config.wandb_logging and wandb_api is None:
        raise ValueError("WANDB_API_KEY not found in environment variables.")
    if train_config.wandb_logging:
        import wandb

        wandb.login(key=wandb_api)
        wandb.init(
            project="cs336-basics-assignment2",
            name=f"{train_config.model_name}_batch{train_config.batch_size}_steps{train_config.num_steps}",
            config={
                "model_config": model_config.to_dict(),
                "train_config": train_config.to_dict(),
            },
        )

    seed_everything(train_config.seed)

    # Initialize model
    model = TransformerLM(model_config)
    model = model.to(train_config.device)
    model.train()

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.min_lr,
        betas=train_config.betas,
        weight_decay=train_config.weight_decay,
    )

    # Start training
    print_color("Starting training...", "blue")
    print_color(f"[info] Total steps: {train_config.num_steps}", "blue")
    START_TIME = time.perf_counter()
    train(model=model, optimizer=optimizer, train_config=train_config)
    ELAPSED_TIME_S = time.perf_counter() - START_TIME
    print_color(f"Training completed.", "blue")
    print_color(f"Elapsed time: {ELAPSED_TIME_S:.2f}s", "blue")

    # Finalize WandB run
    if train_config.wandb_logging:
        wandb.log(
            {
                "time/elapsed_s": ELAPSED_TIME_S,
                "time/elapsed_min": ELAPSED_TIME_S / 60.0,
                "speed/steps_per_s": train_config.num_steps / max(ELAPSED_TIME_S, 1e-9),
            }
        )
        wandb.summary["time/elapsed_s"] = ELAPSED_TIME_S  # 放到 Summary 里更适合对比
        wandb.summary["speed/steps_per_s"] = train_config.num_steps / max(ELAPSED_TIME_S, 1e-9)

        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
