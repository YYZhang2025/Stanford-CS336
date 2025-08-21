import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, List

import dotenv
import fire
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.data_utils import extract_reference_answer, load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate import get_response
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.utils import (
    get_run_name,
    print_rich_dict,
    save_model_and_tokenizer,
)
from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance

logging.getLogger("vllm").setLevel(logging.WARNING)


@dataclass
class TrainConfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "sft-qwen2.5"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    training_steps: int = 512
    mixed_precision_training: bool = True
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    num_example: int = 128

    # Log print:
    log_print_steps = 12

    # For evaluation
    eval_device: str = "cuda:1"
    eval_interval_steps: int = 32


@dataclass
class EvaluateConfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024


def sft_collate_fn(batch, tokenizer):
    prompts, cot, answers = zip(*batch)  # each is a tuple of strings
    prompts = list(prompts)
    cot = list(cot)

    batch_enc = tokenize_prompt_and_output(prompts, cot, tokenizer)

    return {**batch_enc, "answers": answers}


class SFTDataset(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        prompt = self.train_prompts[idx]
        cot = self.train_cot[idx]
        answer = int(self.train_answers[idx].strip())

        return prompt, cot, answer


def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    # pure cosine decay
    decay_ratio = it / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def log_generate(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    cot: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    cur_step: int,
    num_example=2,
):
    random_indices = random.sample(range(len(prompts)), k=num_example)
    sampled_prompts = [prompts[i] for i in random_indices]
    sampled_cot = [cot[i] for i in random_indices]
    sampled_answers = [answers[i] for i in random_indices]

    responses = get_response(vllm_model, sampled_prompts, eval_sampling_params)

    for i in range(num_example):
        response = responses[i]
        answer = sampled_answers[i]
        prompt = sampled_prompts[i]
        extracted_answer = extract_reference_answer(response)
        true_label = sampled_cot[i]

        reward_dict = reward_fn(response, answer)

        info_dict = {
            "prompt": prompt,
            "true_label": true_label,
            "response": response,
            "true_answer": answer,
            "extracted_answer": extracted_answer,
            **reward_dict,
        }

        print(f"======= Step: {cur_step}; Example {i} =======")
        # print_formatted_dict(info_dict)
        print_rich_dict(info_dict)
        print("==============================================\n")


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
):
    responses = get_response(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    for response, answer, prompt in zip(responses, answers, prompts):
        # extracted_answer = extract_reference_answer(response)
        reward_dict = reward_fn(response, answer)
        allinfo_dict_list.append(reward_dict)

    overview = {"correct": 0, "format_wrong": 0, "answer_wrong": 0, "count": 0}
    for reward in allinfo_dict_list:
        overview["count"] += 1
        if reward["reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1

    return overview


def evaluate_sft_model(config: EvaluateConfig, vllm: LLM, eval_step: int):
    prompts, cot, answers = load_and_format_prompts(config.data_path, config.prompt_path)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, sampling_params)

    wandb.log(
        {
            "eval/correct": results["correct"],
            "eval/answer_wrong": results["answer_wrong"],
            "eval/format_wrong": results["format_wrong"],
            "eval_step": eval_step,
        }
    )


def train_sft_model(
    model,
    tokenizer,
    train_config: TrainConfig,
    eval_config: EvaluateConfig,
    train_prompts,
    train_cot,
    train_answers,
    vllm: LLM,
    evaluate: bool = True,
):
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-alignment-sft",
        config={
            "train": asdict(train_config),
            "eval": asdict(eval_config),
        },
        name=get_run_name("sft", train_config),
        reinit=True,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

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
    print(f"[train] Dataloader initialized with batch size {train_config.batch_size}")

    # ---------------------
    # Mixed Precision Context
    # ---------------------
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if train_config.mixed_precision_training
        else nullcontext()
    )

    # ---------------------
    # Optimizer
    # ---------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
    print("[train] Optimizer initialized")

    # ---------------------
    # Training Process
    # ---------------------
    cur_step = 0  # Initialize current step
    batch_loss = 0
    total_micro_steps = 0
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
            if total_micro_steps % train_config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adj_lr = get_lr(cur_step, train_config.learning_rate, train_config.training_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = adj_lr

                optimizer.step()
                optimizer.zero_grad()

                print(
                    f"[train] Step {cur_step} | Loss: {batch_loss / train_config.gradient_accumulation_steps:.4f} | LR: {adj_lr:.6f}"
                )

                wandb.log(
                    {
                        "train/loss": batch_loss / train_config.gradient_accumulation_steps,
                        "train/lr": adj_lr,
                        "train_step": cur_step,
                    }
                )

                batch_loss = 0
                cur_step += 1

                if (cur_step + 1) % train_config.log_print_steps == 0 and evaluate:
                    load_model_into_vllm_instance(model, vllm)
                    log_generate(
                        vllm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=dataset.train_prompts,
                        cot=dataset.train_cot,
                        answers=dataset.train_answers,
                        eval_sampling_params=SamplingParams(
                            temperature=eval_config.temperature,
                            top_p=eval_config.top_p,
                            max_tokens=eval_config.max_tokens,
                            stop=["</answer>"],
                            include_stop_str_in_output=True,
                        ),
                        cur_step=cur_step,
                        num_example=3,
                    )

                if (cur_step + 1) % train_config.eval_interval_steps == 0 and evaluate:
                    print(
                        f"[train] Step {cur_step}: saving model at {train_config.experiment_name}_{train_config.num_example}"
                    )
                    save_model_and_tokenizer(model, tokenizer, train_config)

                    # Run evaluatoin
                    print(f"[eval] at step {cur_step}")
                    load_model_into_vllm_instance(model, vllm)
                    evaluate_sft_model(eval_config, vllm, eval_step=cur_step)
                    print(f"[eval] Evaluation completed for step {cur_step}")

                if cur_step >= train_config.training_steps:
                    break

        if cur_step >= train_config.training_steps:
            break

    save_model_and_tokenizer(model, tokenizer, train_config)
    print(f"[train] Training finished at step {cur_step}")

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
    seed: int = 42,
):
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_config = TrainConfig()
    eval_config = EvaluateConfig()

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    vllm = init_vllm(model_id=model_name, device=train_config.eval_device, seed=seed)

    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    for num_samples in [len(prompts)]:
        # ---------------------
        # Load Model and tokenizer
        # ---------------------
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=train_config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        ).to(train_config.train_device)
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
        print(f"[train] Tokenizer {train_config.model_name} loaded")
        print(f"[train] Model {train_config.model_name} loaded on {train_config.train_device}")

        train_config.num_example = num_samples

        train_prompts = prompts[:num_samples]
        train_cot = cot[:num_samples]
        train_answers = answers[:num_samples]

        train_sft_model(
            model,
            tokenizer,
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
