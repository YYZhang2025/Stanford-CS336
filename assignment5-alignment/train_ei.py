import gc
import logging
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Callable, List

import dotenv
import fire
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.data_utils import load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.utils import get_run_name, print_color, print_rich_dict, save_model_and_tokenizer
from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance
from train_sft import SFTDataset, evaluate_sft_model, log_generate, sft_collate_fn

logging.getLogger("vllm").setLevel(logging.WARNING)


def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def cycle_dataloader(dataloader):
    """
    Creates a cycling iterator for a PyTorch DataLoader.
    """
    while True:
        for batch in dataloader:
            yield batch


def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    # pure cosine decay
    decay_ratio = it / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@dataclass
class TrainConfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "ei-qwen2.5"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    num_example: int = 128

    # Optimization
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    training_steps: int = 8
    mixed_precision_training: bool = True
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    # Expert Iteration hyper-parameters
    n_ei_steps: int = 4  # outer EI steps (replaces training_steps)
    question_per_step: int = 256
    samples_per_prompt: int = 4  # G in Algorithm 2

    # Logging / eval
    log_print_steps: int = 12
    eval_device: str = "cuda:1"
    eval_interval_steps: int = 12

    # Learning Rate adjust
    lr_mode: str = "global"  # one of: "global", "per_outer", "constant"
    warmup_ratio: float = 0.03  # 0 = no warmup
    min_lr_factor: float = 0.1  # min_lr = lr * min_lr_factor
    scale_lr_by_pairs: bool = True  # scale lr by 1/sqrt(num_kept_pairs) per EI step

    # For Expert Iteration sampling
    ei_temperature: float = 1.0
    ei_top_p: float = 1.0
    ei_max_tokens: int = 1024
    ei_stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    ei_include_stop_str_in_output: bool = True
    ei_min_tokens: int = 4


@dataclass
class EvaluateConfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    max_tokens: int = 1024
    include_stop_str_in_output: bool = True


def train_sft_model(
    model,
    tokenizer,
    optimizer,
    train_config: TrainConfig,
    train_prompts,
    train_cot,
    train_answers,
    global_step: int = 0,
    ei_steps: int = 0,
    pairs_this_ei: int = 0,
):
    # Data Preparation
    # ---------------------
    dataset = SFTDataset(train_prompts, train_cot, train_answers)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: sft_collate_fn(batch, tokenizer),
    )
    print(f"[train | ei step-{ei_steps}] Dataloader initialized with batch size {train_config.batch_size}")

    # ---------------------
    # Mixed Precision Context
    # ---------------------
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if train_config.mixed_precision_training
        else nullcontext()
    )

    # ---------------------
    # Training Process
    # ---------------------
    cur_step = 0  # Initialize current step
    batch_loss = 0
    total_loss = 0
    total_micro_steps = 0
    global_step_ = global_step
    while True:
        for i, data in enumerate(dataloader):
            total_micro_steps += 1
            input_ids = data["input_ids"].to(train_config.train_device)
            labels = data["labels"].to(train_config.train_device)
            response_mask = data["response_mask"].to(train_config.train_device)

            with ctx:
                log_prob = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)
                log_prob = log_prob["log_probs"]
                loss, _ = sft_microbatch_train_step(
                    log_prob, response_mask, train_config.gradient_accumulation_steps
                )

            batch_loss += loss

            del input_ids
            del labels
            del response_mask
            clear()

            if total_micro_steps % train_config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # adj_lr = compute_adjusted_lr(
                #     train_config=train_config,
                #     pairs_this_ei=pairs_this_ei,
                #     cur_step=cur_step,
                #     global_step=global_step_,
                # )
                adj_lr = get_lr(ei_steps, train_config.learning_rate, train_config.n_ei_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = adj_lr

                optimizer.step()
                optimizer.zero_grad()
                total_loss += batch_loss / train_config.gradient_accumulation_steps

                print(
                    f"[train | ei step-{ei_steps}] Step {global_step_} | Loss: {batch_loss / train_config.gradient_accumulation_steps:.4f} | LR: {adj_lr:.6f}"
                )
                wandb.log(
                    {
                        "train/loss": batch_loss / train_config.gradient_accumulation_steps,
                        "train/lr": adj_lr,
                        "train/step": global_step_,
                    }
                )

                batch_loss = 0
                cur_step += 1
                global_step_ += 1

                if cur_step >= train_config.training_steps:
                    break

        if cur_step >= train_config.training_steps:
            break

    return total_loss / cur_step, global_step_


class EIDataset(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        prompt = self.train_prompts[idx]
        cot = self.train_cot[idx]
        answer = self.train_answers[idx].strip()

        return prompt, cot, answer


@torch.no_grad()
def ei_collect_correct_pairs(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    train_config: TrainConfig,
) -> tuple[list[str], list[str]]:
    """Return (kept_prompts, kept_outputs) where reward==1."""
    kept_prompts: list[str] = []
    kept_outputs: list[str] = []

    # Ensure we sample multiple outputs per prompt.
    ei_sp = SamplingParams(
        temperature=train_config.ei_temperature,
        top_p=train_config.ei_top_p,
        max_tokens=train_config.ei_max_tokens,
        stop=train_config.ei_stop_tokens,
        include_stop_str_in_output=train_config.ei_include_stop_str_in_output,
        min_tokens=train_config.ei_min_tokens,
        n=train_config.samples_per_prompt,
    )

    all_gens = vllm_model.generate(prompts, ei_sp)
    for q, a, gens in zip(prompts, answers, all_gens):
        for i, o in enumerate(gens.outputs):
            # compute reward
            r = reward_fn(o.text, str(a))
            if r.get("reward", 0) == 1:
                kept_prompts.append(q)
                kept_outputs.append(o.text)

    return kept_prompts, kept_outputs


def train_ei_model(
    train_config: TrainConfig,
    eval_config: EvaluateConfig,
    train_prompts,
    train_cot,
    train_answers,
    vllm: LLM,
):
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-alignment-ei",
        config={"train": asdict(train_config), "eval": asdict(eval_config)},
        name=get_run_name("ei", train_config),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    eval_sp = SamplingParams(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        stop=eval_config.stop_tokens,
        include_stop_str_in_output=eval_config.include_stop_str_in_output,
    )

    # Load model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    ).to(train_config.train_device)
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
    print(f"[ei train] Tokenizer {train_config.model_name} loaded")
    print(f"[ei train] Model {train_config.model_name} loaded on {train_config.train_device}")
    print("[ei train] Optimizer loaded")

    # Base dataloader that provides (prompt, cot, answer); used only to sample questions/answers
    base_ds = EIDataset(train_prompts, train_cot, train_answers)
    base_dl = DataLoader(
        dataset=base_ds,
        batch_size=train_config.question_per_step,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    cycled_dataloader = cycle_dataloader(base_dl)

    # -------------------
    # Training Loop
    # -------------------
    global_step = 0
    for step in range(train_config.n_ei_steps):
        # (3) Sample a batch of questions Db from D
        # This batch will contain (prompt, cot, answer) tuples
        batch = next(cycled_dataloader)
        question_batch = batch[0]  # Convert to list
        answer_batch = batch[2]  # Convert to list

        # (5-6-7) Sample G outputs per question, compute rewards, filter to correct pairs
        kept_prompts, kept_outputs = ei_collect_correct_pairs(
            vllm_model=vllm,
            reward_fn=r1_zero_reward_fn,
            prompts=question_batch,
            answers=answer_batch,
            train_config=train_config,
        )

        if len(kept_prompts) == 0:
            print(f"[EI] Step {step}: no correct generations; skipping SFT update.")
            continue

        print_color(f"[EI] Step {step} | Pairs: {len(kept_prompts)}")
        print("[EI] Example kept pair:")
        assert len(kept_prompts) == len(kept_outputs)
        print("Kept prompts: ", kept_prompts[0])
        print("Kept outputs: ", kept_outputs[0])
        # (8) pi_theta <- SFT(pi_theta, D_sft)
        loss, global_step = train_sft_model(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            train_config=train_config,
            train_prompts=kept_prompts,
            train_cot=kept_outputs,
            train_answers=[0] * len(kept_prompts),  # Dummy answers, not used in SFT
            global_step=global_step,
            ei_steps=step,
            pairs_this_ei=len(kept_prompts),
        )
        print_color(f"[EI] Step {step} | Pairs: {len(kept_prompts)} | Loss: {loss:.4f}", color="green")

        # wandb.log({"train/pairs": len(kept_prompts), "train_step": step, "train/loss": loss})

        print(f"Loaded weights to vllm at step {step}")
        load_model_into_vllm_instance(model, vllm)
        # Periodic qualitative logging and eval
        if (step + 1) % train_config.log_print_steps == 0:
            log_generate(
                vllm,
                reward_fn=r1_zero_reward_fn,
                prompts=base_ds.train_prompts,
                cot=base_ds.train_cot,
                answers=[str(x) for x in base_ds.train_answers],
                eval_sampling_params=eval_sp,
                cur_step=step,
                num_example=2,
            )

        if (step + 1) % train_config.eval_interval_steps == 0:
            evaluate_sft_model(eval_config, vllm, global_step)
            save_model_and_tokenizer(model, tokenizer, train_config)

    save_model_and_tokenizer(model, tokenizer, train_config)
    wandb.finish()

    return model


def main(
    *,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "./data/gsm8k/train.jsonl",
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seed: int = 123,
):
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/workspace/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf/models"
    os.environ["HF_HUB_CACHE"] = "/workspace/hf/hub"

    train_config = TrainConfig()
    eval_config = EvaluateConfig()

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    vllm = init_vllm(model_id=model_name, device=train_config.eval_device, seed=seed)
    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    for num_samples in [len(prompts)]:
        train_config.num_example = num_samples
        train_config.experiment_name = f"experiment_{num_samples}"
        train_config.n_ei_steps = math.ceil(num_samples / train_config.question_per_step)

        print_rich_dict({"train config": asdict(train_config), "eval config": asdict(eval_config)})

        train_prompts = prompts[:num_samples]
        train_cot = cot[:num_samples]
        train_answers = answers[:num_samples]

        train_ei_model(
            train_config,
            eval_config=eval_config,
            train_prompts=train_prompts,
            train_cot=train_cot,
            train_answers=train_answers,
            vllm=vllm,
        )

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
