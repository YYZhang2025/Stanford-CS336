import logging
import os

import dotenv
import fire
import torch
from transformers import AutoModelForCausalLM

from cs336_alignment.algs import EITrainConfig, EITrainer
from cs336_alignment.utils import get_device, print_color, seed_everything
from cs336_alignment.vllm_utils import init_vllm


def main(
    train_config_path: str = "configs/ei/train_config.json",
    dataset_name: str = "gsm8k",
):
    logging.getLogger("vllm").setLevel(logging.WARNING)
    dotenv.load_dotenv()

    train_config = EITrainConfig.from_json(train_config_path)
    train_config.dataset_name = dataset_name
    seed_everything(train_config.seed)

    # init vllm
    vllm_device = get_device(rank=1, verbose=False)
    vllm = init_vllm(
        model_id=train_config.model_name,
        device=str(vllm_device),
        gpu_memory_utilization=0.85,
        seed=train_config.seed,
    )
    print_color(f"Initialized VLLM on {str(vllm_device)}", color="cyan")

    model_device = get_device(rank=0, verbose=False)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    model.to(model_device)
    print_color(f"Loaded model to {str(model_device)}", color="cyan")

    if train_config.wandb_logging:
        import wandb

        wandb_api = os.getenv("WANDB_API_KEY")
        if wandb_api is None:
            raise ValueError("WANDB_API_KEY not found in environment variables.")
        wandb.login(key=wandb_api)
        wandb.init(
            project=train_config.project_name,
            name=train_config.run_name,
            config={
                "train_config": train_config.to_dict(),
            },
        )

    ei_trainer = EITrainer(
        model=model,
        train_config=train_config,
        device=model_device,
    )
    ei_trainer.train(vllm=vllm)

    print_color("Training completed. Saving final model checkpoint...", color="green")
    # checkpoint_file = os.path.join(ei_trainer.checkpoint_path, "checkpoint_final.pt")

    # Cleanup
    if train_config.wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
