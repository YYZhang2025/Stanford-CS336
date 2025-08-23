import logging
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import dotenv
import fire
import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.data_utils import load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.sft_utils import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.utils import cycle_dataloader, get_run_name, print_rich_dict
from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance

logging.getLogger("vllm").setLevel(logging.WARNING)


@dataclass
class TrainConfig:
    # Basic
    experiment_name_base: str = "experiments"
    experiment_name: str = "grpo-qwen2.5"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    advantage_eps: float = 1e-6

    # GRPO
    question_per_grpo_step: int = 10
    n_grpo_steps: int = 100
    n_train_steps_per_rollout_batch: int = 100
    group_size: int = 4
    advantage_eps: float = 1e-6
    use_std_normalization: bool = True

    eval_device: str = "cuda:1"
    mixed_precision_training: bool = True
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    # For VLLM sampling
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True
    min_tokens: int = 4
    vllm_seed: int = 42


@dataclass
class EvaluateConfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    max_tokens: int = 1024
    include_stop_str_in_output: bool = True


class GRPODataset(Dataset):
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


def get_old_log_probs(model, input_ids, labels, train_config: TrainConfig) -> tuple[list[float], list[float]]:
    model.eval()

    log_probs = []
    token_entropy = []

    for train_step in range(train_config.question_per_grpo_step):
        log_probs_step, token_entropy_step = get_response_log_probs(
            model=model,
            input_ids=input_ids[train_step : train_step + train_config.group_size],
            labels=labels[train_step : train_step + train_config.group_size],
            return_token_entropy=True,
        )
        log_probs.append(log_probs_step)
        token_entropy.append(token_entropy_step)

    assert len(log_probs) == len(input_ids)
    assert len(token_entropy) == len(input_ids)

    return log_probs, token_entropy


class GRPORolloutDataset(Dataset):
    """The rollout dataset for GRPO training.
    input: (prompt, response, raw_reward, advantage)
    """

    def __init__(
        self, model, prompts, responses, raw_rewards, advantages, train_config: TrainConfig, tokenizer
    ):
        self.prompts = prompts
        self.responses = responses
        self.raw_rewards = raw_rewards
        self.advantages = advantages
        self.old_log_probs = get_old_log_probs(model, prompts, responses, train_config)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple:
        prompt = self.prompts[idx]
        response = self.responses[idx]
        raw_reward = self.raw_rewards[idx]
        advantage = self.advantages[idx]
        old_log_probs = self.old_log_probs[idx]

        encoded = tokenize_prompt_and_output([prompt], [response], self.tokenizer)
        input_ids = encoded["input_ids"][0]
        labels = encoded["labels"][0]
        response_mask = encoded["response_mask"][0]

        return input_ids, labels, response_mask, raw_reward, advantage, old_log_probs[0]


def update_policy(
    model,
    optimizer,
    train_config: TrainConfig,
    prompts,
    responses,
    raw_rewards,
    advantages,
    tokenizer,
    global_step,
):
    model.eval()
    with torch.no_grad():
        dataset = GRPORolloutDataset(
            model,
            prompts,
            responses,
            raw_rewards,
            advantages,
            train_config,
            tokenizer=tokenizer,
        )
    dataloader = DataLoader(dataset=dataset, batch_size=train_config.micro_batch_size, shuffle=True)
    cycled_dataloader = cycle_dataloader(dataloader)
    model.train()

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if train_config.mixed_precision_training
        else nullcontext()
    )

    cur_step = 0
    global_step_ = global_step
    total_steps = 0
    while True:
        batch = next(cycled_dataloader)
        input_ids, labels, response_mask, raw_rewards, advantages, old_log_probs = batch

        with ctx:
            outputs = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)
            log_prob = outputs["log_probs"]
            loss = grpo_microbatch_train_step(
                log_prob,
                response_mask,
                train_config.gradient_accumulation_steps,
                loss_type="grpo_clip",
                raw_rewards=raw_rewards,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=0.2,
            )

        print(f"Step {global_step_ + 1}: Loss {loss}")
        total_steps += 1
        if (total_steps + 1) % train_config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            cur_step += 1
            global_step_ += 1

        if cur_step >= train_config.n_grpo_steps:
            break

    return global_step_


def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    # pure cosine decay
    decay_ratio = it / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_grpo(
    train_config: TrainConfig,
    eval_config: EvaluateConfig,
    train_prompts,
    train_cot,
    train_answers,
    vllm: LLM,
):
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-alignment-grpo",
        config={"train": asdict(train_config), "eval": asdict(eval_config)},
        name=get_run_name("ei", train_config),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    grpo_sp = SamplingParams(
        temperature=train_config.temperature,
        top_p=train_config.top_p,
        max_tokens=train_config.max_tokens,
        min_tokens=train_config.min_tokens,
        stop=train_config.stop_tokens,
        include_stop_str_in_output=train_config.include_stop_str_in_output,
        n=train_config.group_size,
        seed=train_config.vllm_seed,
    )
    eval_sp = SamplingParams(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        stop=eval_config.stop_tokens,
        include_stop_str_in_output=eval_config.include_stop_str_in_output,
    )

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

    # This will return the batch for micro-step which used for gradient accumulation
    base_ds = GRPODataset(train_prompts, train_cot, train_answers)
    base_dl = DataLoader(
        dataset=base_ds,
        batch_size=train_config.question_per_grpo_step,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    cycled_dataloader = cycle_dataloader(base_dl)

    global_step = 0
    for grpo_step in range(train_config.n_grpo_steps):
        # (3): Sample a batch of questions from dataset
        sample_batch = next(cycled_dataloader)
        sample_prompts, sample_cots, sample_answers = zip(*sample_batch)
        sample_prompts = list(sample_prompts)
        sample_cots = list(sample_cots)
        sample_answers = list(sample_answers)

        # (4): Set the old policy
        load_model_into_vllm_instance(model, vllm)

        # (5): Sample G outputs per question.
        all_gens = vllm.generate(sample_prompts, grpo_sp)
        all_prompts = []
        all_responses = []
        all_answers = []
        for q, a, gens in zip(sample_prompts, sample_answers, all_gens):
            for i, o in enumerate(gens.outputs):
                print(f"[ei train] Generated output {i} for question '{q}': {o.text}")
                all_prompts.append(q)
                all_responses.append(o.text)
                all_answers.append(a)

        # (6) / (7): Compute rewards for each sampled output
        advantages, rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=all_answers,
            group_size=train_config.group_size,
            advantage_eps=train_config.advantage_eps,
            normalized_by_std=train_config.use_std_normalization,
        )

        print(f"[ei train] Advantages: {advantages}, Rewards: {rewards}")

        advantages_train = advantages.to(train_config.train_device)
        raw_rewards_train = rewards.to(train_config.train_device)

        global_step = update_policy(
            model,
            optimizer,
            train_config,
            advantages_train,
            raw_rewards_train,
            sample_prompts,
            sample_answers,
            tokenizer,
            global_step,
        )

        # Evaluate


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

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    train_config = TrainConfig()
    eval_config = EvaluateConfig()
    print_rich_dict({"train config": asdict(train_config), "eval config": asdict(eval_config)})

    vllm = init_vllm(model_id=model_name, device=train_config.eval_device, seed=seed)
    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    train_grpo(
        train_config=train_config,
        eval_config=eval_config,
        train_prompts=prompts,
        train_cot=cot,
        train_answers=answers,
        vllm=vllm,
    )

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
