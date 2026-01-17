import os

import dotenv
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.algs.sft import SFTTrainer
from cs336_alignment.config import TrainConfig
from cs336_alignment.utils import get_device, print_color

# from cs336_alignment.vllm_utils import init_vllm


def main(
    train_config_path: str = "configs/sft/train_config.json",
):
    dotenv.load_dotenv()
    train_config = TrainConfig.from_json(train_config_path)

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

    # init vllm
    # vllm = init_vllm(
    #     model_id=train_config.model_name,
    #     device="cuda",
    #     seed=train_config.seed,
    #     gpu_memory_utilization=0.85,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.float16,
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    device = get_device()
    model.to(device)
    print_color(f"Loaded model and tokenizer: {train_config.model_name}", color="cyan")

    sft_trainer = SFTTrainer(
        model=model,
        train_config=train_config,
        device=device,
    )
    sft_trainer.train(vllm=None)

    # Cleanup
    if train_config.wandb_logging:
        wandb.finish()
    vllm.shutdown()


if __name__ == "__main__":
    fire.Fire(main)
