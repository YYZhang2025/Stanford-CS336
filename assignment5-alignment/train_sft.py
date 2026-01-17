import os
from contextlib import nullcontext

import dotenv
import fire
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.config import TrainConfig
from cs336_alignment.dataset_utils.utils import load_and_format_prompts
from cs336_alignment.sft_utils import (
    SFTDataset,
    collate_fn,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.utils import cycle_dataloader, get_ctx, get_device, print_color, to_float

# from cs336_alignment.vllm_utils import init_vllm


def train_sft_model(
    model: torch.nn.Module,
    tokenizer,
    train_config: TrainConfig,
    prompts: list[str],
    cots: list[str],
    answers: list[str],
    vllm=None,
):
    dataset = SFTDataset(prompts, cots, answers)
    dataloader = cycle_dataloader(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )
    )

    optmizer = torch.optim.AdamW(
        model.parameters(),
        betas=train_config.betas,
        lr=train_config.min_lr,
        weight_decay=train_config.weight_decay,
    )

    model.train()
    device = model.device

    ctx = get_ctx(train_config.mixed_precision_training, device)
    batch_loss = 0.0
    for step in range(train_config.total_training_steps):
        print_color(f"Training step {step + 1}/{train_config.total_training_steps}", "yellow")
        batch = next(dataloader)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)
        answers_batch = batch["answers"]

        print(answers_batch)
        with ctx:
            log_prob = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)
            log_prob = log_prob["log_probs"]
            loss, metadata = sft_microbatch_train_step(
                log_prob, response_mask, train_config.gradient_accumulation_steps
            )

        batch_loss += to_float(loss)

        if (step + 1) % train_config.gradient_accumulation_steps == 0:
            batch_loss /= train_config.gradient_accumulation_steps
            optmizer.step()
            optmizer.zero_grad()
            batch_loss = 0.0
            if train_config.wandb_logging:
                wandb.log({"train/loss": batch_loss}, step=step)
            print_color(f"Step {step + 1}/{train_config.total_training_steps}, Loss: {batch_loss}", "green")


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

    prompts, cots, answers = load_and_format_prompts(
        data_path=train_config.dataset_path,
        prompt_template_path=train_config.prompt_path,
    )
    print_color(
        f"Loaded {len(prompts)} training examples from dataset {train_config.dataset_name}",
        color="green",
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.float16,
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    model.to(get_device())
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        use_fast=True,
    )
    print_color(f"Loaded model and tokenizer: {train_config.model_name}", color="cyan")

    print_color(
        f"[-info] Starting SFT training total {train_config.total_training_steps} steps", color="green"
    )
    train_sft_model(
        model,
        tokenizer,
        train_config,
        prompts,
        cots,
        answers,
    )

    # Cleanup
    if train_config.wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
