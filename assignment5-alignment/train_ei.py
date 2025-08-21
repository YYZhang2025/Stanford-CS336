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
    experiment_name: str = "ei-qwen2.5"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    # Optimization
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    mixed_precision_training: bool = True
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    # Expert Iteration hyper-parameters
    n_ei_steps: int = 512  # outer EI steps (replaces training_steps)
    samples_per_prompt: int = 4  # G in Algorithm 2

    # Logging / eval
    num_example: int = 128
    log_print_steps: int = 12
    eval_device: str = "cuda:1"
    eval_interval_steps: int = 32


@dataclass
class EvaluateConfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024


def make_sft_collate_fn(tokenizer):
    def _collate(batch):
        prompts, outputs, answers = zip(*batch)  # outputs can be ground-truth CoT or model generations
        prompts = list(prompts)
        outputs = list(outputs)
        batch_enc = tokenize_prompt_and_output(prompts, outputs, tokenizer)
        return {**batch_enc, "answers": answers}

    return _collate


class CustomDataset(Dataset):
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


@torch.no_grad()
def ei_collect_correct_pairs(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    sampling_params: SamplingParams,
    samples_per_prompt: int,
) -> tuple[list[str], list[str]]:
    """Return (kept_prompts, kept_outputs) where reward==1."""

    kept_prompts: list[str] = []
    kept_outputs: list[str] = []

    # Ensure we sample multiple outputs per prompt.
    sp = SamplingParams(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=sampling_params.max_tokens,
        stop=sampling_params.stop,
        include_stop_str_in_output=sampling_params.include_stop_str_in_output,
        min_tokens=4,
        n=samples_per_prompt,
    )

    # vLLM returns a list of generations per prompt when n>1 via our utility.
    for q, a in zip(prompts, answers):
        gens = get_response(vllm_model, [q], sp)
        # get_response returns a flat list for a single prompt with n samples
        # => length == samples_per_prompt
        for o in gens:
            r = reward_fn(o, a)
            if r.get("reward", 0) == 1:
                kept_prompts.append(q)
                kept_outputs.append(o)

    return kept_prompts, kept_outputs


def run_sft_on_pairs(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    outputs: List[str],
    *,
    device: str,
    grad_accum_steps: int,
    optimizer: torch.optim.Optimizer,
    mixed_precision: bool,
) -> float:
    """One pass of SFT over provided (prompt, output) pairs. Returns average loss."""
    ds = CustomDataset(prompts, outputs, [0] * len(prompts))
    dl = DataLoader(
        dataset=ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=make_sft_collate_fn(tokenizer),
    )
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()
    total_loss = 0.0
    micro = 0
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)
        with ctx:
            out = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)
            log_prob = out["log_probs"]
            loss, _ = sft_microbatch_train_step(log_prob, response_mask, grad_accum_steps)
        total_loss += float(loss)
        micro += 1
        if micro % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(1, micro)


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


def cycle_dataloader(dataloader):
    """
    Creates a cycling iterator for a PyTorch DataLoader.
    """
    while True:
        for batch in dataloader:
            yield batch


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
        reinit=True,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Load model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    ).to(train_config.train_device)
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    print(f"[train] Tokenizer {train_config.model_name} loaded")
    print(f"[train] Model {train_config.model_name} loaded on {train_config.train_device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
    print("[train] Optimizer initialized")

    # Base dataloader that provides (prompt, cot, answer); used only to sample questions/answers
    base_ds = CustomDataset(train_prompts, train_cot, train_answers)
    base_dl = DataLoader(
        dataset=base_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    cycled_dataloader = cycle_dataloader(base_dl)

    sampling_params = SamplingParams(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    cur_step = 0
    for step in range(train_config.n_ei_steps):
        # (3) Sample a batch of questions Db from D
        # This batch will contain (prompt, cot, answer) tuples
        batch = next(cycled_dataloader)
        q_batch = batch[0]
        a_batch = batch[2]

        # (4) Set old policy model pi_old <- pi_theta
        load_model_into_vllm_instance(model, vllm)

        # (5-6-7) Sample G outputs per question, compute rewards, filter to correct pairs
        kept_prompts, kept_outputs = ei_collect_correct_pairs(
            vllm_model=vllm,
            reward_fn=r1_zero_reward_fn,
            prompts=q_batch,
            answers=a_batch,
            sampling_params=sampling_params,
            samples_per_prompt=train_config.samples_per_prompt,
        )

        if len(kept_prompts) == 0:
            print(f"[EI] Step {step}: no correct generations; skipping SFT update.")
            continue

        # (8) pi_theta <- SFT(pi_theta, D_sft)
        loss = run_sft_on_pairs(
            model,
            tokenizer,
            kept_prompts,
            kept_outputs,
            device=train_config.train_device,
            grad_accum_steps=train_config.gradient_accumulation_steps,
            optimizer=optimizer,
            mixed_precision=train_config.mixed_precision_training,
        )

        # Cosine LR schedule based on outer step count (optional)
        adj_lr = get_lr(cur_step, train_config.learning_rate, train_config.n_ei_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = adj_lr

        print(f"[EI] Step {step} | Pairs: {len(kept_prompts)} | Loss: {loss:.4f} | LR: {adj_lr:.6f}")
        wandb.log(
            {"train/loss": loss, "train/pairs": len(kept_prompts), "train/lr": adj_lr, "train_step": cur_step}
        )
        cur_step += 1

        # Periodic qualitative logging and eval
        if (step + 1) % train_config.log_print_steps == 0:
            load_model_into_vllm_instance(model, vllm)
            log_generate(
                vllm,
                reward_fn=r1_zero_reward_fn,
                prompts=base_ds.train_prompts,
                cot=base_ds.train_cot,
                answers=[str(x) for x in base_ds.train_answers],
                eval_sampling_params=sampling_params,
                cur_step=cur_step,
                num_example=3,
            )

        if (step + 1) % train_config.eval_interval_steps == 0:
            print(
                f"[EI] Step {step}: saving model at {train_config.experiment_name}_{train_config.num_example}"
            )
            save_model_and_tokenizer(model, tokenizer, train_config)
            print(f"[eval] at step {step}")
            load_model_into_vllm_instance(model, vllm)
            evaluate_sft_model(eval_config, vllm, eval_step=cur_step)
            print(f"[eval] Evaluation completed for step {step}")

    save_model_and_tokenizer(model, tokenizer, train_config)
    print(f"[train] EI finished at step {cur_step}")
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
        train_config.num_example = num_samples
        train_config.experiment_name = f"experiment_{num_samples}"

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
