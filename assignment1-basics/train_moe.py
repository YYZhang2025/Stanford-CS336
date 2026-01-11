import os

import dotenv
import fire

from cs336_basics.config import ModelConfig, TrainingConfig
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.train_engine import train
from cs336_basics.utils import get_device, print_color, seed_everything


def main(
    train_config_json: str | None = None,
    model_config_json: str | None = None,
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
            project="cs336-basics-assignment1",
            name=train_config.model_name + f"_batch-{train_config.batch_size}_steps-{train_config.num_steps}",
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
    train(model=model, optimizer=optimizer, train_config=train_config)

    print_color("Training completed.", "blue")

    # Finalize WandB run
    if train_config.wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    main(
        train_config_json="./configs/tiny_moe_transformer_small_dff/train_config.json",
        model_config_json="./configs/tiny_moe_transformer_small_dff/model_config.json",
    )
