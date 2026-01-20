import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn as nn
import wandb
from tqdm import trange
from transformers import AutoTokenizer, PreTrainedModel
from vllm import SamplingParams

from cs336_alignment.algs.sft import (
    get_response_log_probs,
    tokenize_prompt_and_output,
)
from cs336_alignment.base_config import BaseConfig
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval import evaluate_responses
from cs336_alignment.utils import (
    clear_memory,
    get_ctx,
    print_color,
    print_rich_dict,
)
from cs336_alignment.vllm_utils import generate_responses, load_policy_into_vllm_instance
from tests.conftest import response_mask

REWARD_FN_MAP = {
    "r1_zero_reward_fn": r1_zero_reward_fn,
}


@dataclass
class GRPOTrainConfig(BaseConfig):
    n_grpo_steps: int = 200
    rollout_batch_size: int = 256
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    group_size: int = 8

    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128

    reward_fn: Literal["r1_zero_reward_fn"] = "r1_zero_reward_fn"
    cliprange: float = 0.2

    # Optimizer hyperparameters
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "grpo_clip"
    betas: tuple = field(default=(0.9, 0.95))
    weight_decay: float = 0.0
    max_lr: float = 5e-6
    max_grad_norm: float = 1.0

    # Sampling hyperparameters
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 1024
    sampling_min_tokens: int = 4
    sampling_top_p: float = 1.0
    sampling_stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])

    # Others
    mixed_precision_training: bool = True
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42

    def __post_init__(self):
        self.run_name = f"grpo_dataset({self.dataset_name})_reward({self.reward_fn})_prompt({self.prompt_template_path.split('/')[-1]})_loss_type({self.loss_type})"


def load_dataset(path: str, prompt_template: str = ""):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    prompts = []
    cots = []
    answers = []
    for row in rows:
        prompts.append(prompt_template.format(question=row["question"]))
        cots.append(row["cot"])
        answers.append(row["answer"])

    return prompts, cots, answers


def sample_g_outputs_per_prompt(vllm, sampling_params, prompts: list[str]):
    rollout_sampling_params = SamplingParams(
        temperature=sampling_params.temperature,
        max_tokens=sampling_params.max_tokens,
        min_tokens=sampling_params.min_tokens,
        top_p=sampling_params.top_p,
        stop=sampling_params.stop,
        include_stop_str_in_output=sampling_params.include_stop_str_in_output,
        logprobs=1,
    )

    all_responses, gen_ids, old_log_probs = generate_responses(
        vllm,
        prompts,
        rollout_sampling_params,
    )
    return all_responses, gen_ids, old_log_probs


def compute_rewards_from_responses(
    responses: list[str],
    answers: list[str],
    reward_fn: Callable,
) -> list[dict]:
    rewards = []
    for response, answer in zip(responses, answers):
        reward = reward_fn(response, answer)
        rewards.append(reward)
    return rewards


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalized_by_std: bool = True,
):
    formatted_rewards = []
    answer_correct_rewards = []
    rewards = []
    for response, true_answer in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, true_answer)
        rewards.append(reward_info["reward"])
        formatted_rewards.append(reward_info["format_reward"])
        answer_correct_rewards.append(reward_info["answer_reward"])

    advs = []
    for i in range(0, len(rewards), group_size):
        group_rewards = rewards[i : i + group_size]
        group_rewards_tensor = torch.tensor(group_rewards)
        group_mean = torch.mean(group_rewards_tensor)
        if normalized_by_std:
            group_std = torch.std(group_rewards_tensor) + advantage_eps
            normalized_rewards = (group_rewards_tensor - group_mean) / group_std
        else:
            normalized_rewards = group_rewards_tensor - group_mean
        advs.extend(normalized_rewards.tolist())

    advs = torch.tensor(advs)
    meta_info = {}

    return advs, rewards, meta_info


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    """
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)

    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float = 0.2,
):
    """
    Compute the GRPO-CLIP loss at every token.
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped_loss = -advantages * ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_loss = -advantages * clipped_ratio

    loss = torch.max(unclipped_loss, clipped_loss)
    metadata = {}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor | None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float = 0.2,
):
    """
    Compute the policy-gradient loss at every token.

    Args:
        policy_log_probs: Log probabilities of the current policy at each token.
        loss_type: The type of policy gradient loss to compute.
        raw_rewards: Raw rewards for each sequence.
        advantages: Pre-computed advantages for each sequence (if applicable).
        old_log_probs: Log probabilities from the old policy (if applicable).
        cliprange: Clipping range for GRPO-CLIP loss.
    Returns:

        A tuple of (loss tensor, metadata dictionary).
    """

    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
        return loss, metadata
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("Advantages must be provided for reinforce_with_baseline loss.")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
        return loss, metadata

    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None:
            raise ValueError("Advantages and old_log_probs must be provided for grpo_clip loss.")
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        return loss, metadata

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def sample_batch_questions(
    prompts: list[str],
    answers: list[str],
    batch_size: int,
    group_size: int = 8,
) -> tuple[list[str], list[str]]:
    index = random.sample(range(len(prompts)), k=batch_size)
    sampled_prompts = [prompts[i] for i in index]
    sampled_answers = [answers[i] for i in index]

    batch_prompts = []
    batch_answers = []
    for p, a in zip(sampled_prompts, sampled_answers):
        batch_prompts.extend([p] * group_size)
        batch_answers.extend([a] * group_size)

    return batch_prompts, batch_answers


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    sum_masked = torch.sum(masked_tensor, dim=dim)
    count_nonzero = torch.sum(mask, dim=dim).clamp(min=1e-8)
    mean = sum_masked / count_nonzero
    return mean


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the GRPO loss over microbatches for training.

    Args:
        policy_log_probs: Log probabilities of the current policy at each token.
        response_mask: Mask indicating valid tokens in the responses.
        gradient_accumulation_steps: Number of microbatches to accumulate gradients over.
        loss_type: The type of policy gradient loss to compute.
        raw_rewards: Raw rewards for each sequence.
        advantages: Pre-computed advantages for each sequence (if applicable).
        old_log_probs: Log probabilities from the old policy (if applicable).
        cliprange: Clipping range for GRPO-CLIP loss.
    Returns:
        A tuple of (mean loss tensor, metadata dictionary).
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    masked_loss = masked_mean(
        tensor=loss,
        mask=response_mask,
        dim=-1,
    )

    masked_loss = masked_loss / gradient_accumulation_steps
    masked_loss.backward()

    return masked_loss, metadata


def pad_logprobs_to_T(old_lp_list: list[list[float]], T: int, pad_value: float = 0.0) -> torch.Tensor:
    # old_lp_list: length B, each is length Li (variable)
    out = []
    for lp in old_lp_list:
        lp = lp[:T]  # truncate if longer than T
        if len(lp) < T:
            lp = lp + [pad_value] * (T - len(lp))
        out.append(lp)
    return torch.tensor(out, dtype=torch.float32)  # [B, T]


class GRPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_config: GRPOTrainConfig,
        device: torch.device,
        dataset_dir_base: str = "./data/pre-processed",
    ):
        self.model = model
        self.train_config = train_config
        self.device = device
        self.dataset_dir_base = dataset_dir_base

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=train_config.betas,
            lr=self.train_config.max_lr,
            weight_decay=self.train_config.weight_decay,
        )
        self.ctx = get_ctx(
            use_mixed=self.train_config.mixed_precision_training,
            device=device,
        )

        with open(train_config.prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read().strip()
        dataset_dir = os.path.join(dataset_dir_base, train_config.dataset_name)
        train_prompts, train_cots, train_answers = load_dataset(
            os.path.join(dataset_dir, "train.jsonl"), prompt_template=self.prompt_template
        )
        self.train_prompts = train_prompts
        self.train_cots = train_cots
        self.train_answers = train_answers

        test_prompts, test_cots, test_answers = load_dataset(
            os.path.join(dataset_dir, "test.jsonl"), prompt_template=self.prompt_template
        )
        self.test_prompts = test_prompts
        self.test_cots = test_cots
        self.test_answers = test_answers

        self.checkpoint_path = os.path.join(
            train_config.checkpoint_dir,
            f"grpo_{train_config.model_name.split('/')[-1]}_{train_config.dataset_name}",
        )
        os.makedirs(self.checkpoint_path, exist_ok=True)
        train_config.to_json(
            os.path.join(self.checkpoint_path, "train_config.json"),
        )
        self.sampling_params = SamplingParams(
            temperature=self.train_config.sampling_temperature,
            max_tokens=self.train_config.sampling_max_tokens,
            top_p=self.train_config.sampling_top_p,
            min_tokens=self.train_config.sampling_min_tokens,
            include_stop_str_in_output=True,
            stop=self.train_config.sampling_stop_tokens,
        )

        self.reward_fn = REWARD_FN_MAP[self.train_config.reward_fn]

        self.grpo_step = 0

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=train_config.model_name,
            use_fast=True,
        )

    def _load_into_vllm(self, vllm):
        load_policy_into_vllm_instance(self.model, vllm)
        print_color(
            f"Loaded SFT model weights at step {self.grpo_step} into VLLM instance for evaluation.",
            color="magenta",
        )

    @torch.no_grad()
    def evaluate(self, vllm=None):
        print_color(f"Evaluating EI model at {self.grpo_step} on test dataset ", color="magenta")
        self._load_into_vllm(vllm)

        overview = evaluate_responses(
            vllm=vllm,
            prompts=self.test_prompts,
            answers=self.test_answers,
            sampling_params=self.sampling_params,
        )

        print_color("Evaluation Overview:", color="magenta")
        print_rich_dict(overview)
        return overview

    @torch.no_grad()
    def sample_responses(
        self,
        vllm=None,
        num_samples: int = 5,
    ) -> list[str]:
        print_color(
            f"Sampling {num_samples} responses from EI model at step {self.grpo_step}...", color="cyan"
        )
        self._load_into_vllm(vllm)

        index = random.sample(range(len(self.test_prompts)), k=num_samples)
        prompts = [self.test_prompts[i] for i in index]
        true_answers = [self.test_answers[i] for i in index]

        responses = generate_responses(
            vllm,
            prompts,
            self.sampling_params,
        )

        print_color("Sampled Responses:", color="cyan")
        for i, (prompt, response, true_answer) in enumerate(zip(prompts, responses, true_answers)):
            print_color(f"\n=== Example {i + 1} ===", color="cyan")
            print_color("Prompt: ", color="cyan")
            print(prompt)
            print_color("Response: ", color="cyan")
            print(response)
            print_color("True Answer: ", color="cyan")
            print(true_answer)

        return responses

    def grpo_train_step(
        self,
        vllm,
    ):
        print_color(f"Sampling batch of {self.train_config.rollout_batch_size} questions...", color="green")
        sample_prompts, sample_answers = sample_batch_questions(
            self.train_prompts,
            self.train_answers,
            self.train_config.rollout_batch_size,
            self.train_config.group_size,
        )

        print_color("Load into old policy VLLM instance...", color="green")
        load_policy_into_vllm_instance(
            self.model,
            vllm,
        )

        print_color("Generating rollout responses...", color="green")

        (all_responses, gen_ids, old_log_probs) = sample_g_outputs_per_prompt(
            vllm,
            self.sampling_params,
            sample_prompts,
        )

        print_color("Computing rewards...", color="green")
        repeated_ground_truths = sample_answers
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=self.reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=self.train_config.group_size,
            advantage_eps=self.train_config.advantage_eps,
            normalized_by_std=True,
        )

        n_train_steps = self.train_config.epochs_per_rollout_batch * (
            self.train_config.rollout_batch_size
            * self.train_config.group_size
            // self.train_config.train_batch_size
        )
        print_color(f"Performing {n_train_steps} training steps...", color="green")
        for train_step in range(n_train_steps):
            clear_memory()
            batch_loss = 0.0

            for micro_step in trange(
                self.train_config.train_batch_size // self.train_config.gradient_accumulation_steps,
                desc="Microbatches",
            ):
                start_index = (
                    micro_step
                    * (self.train_config.train_batch_size // self.train_config.gradient_accumulation_steps)
                    + train_step * self.train_config.train_batch_size
                )
                end_index = start_index + (
                    self.train_config.train_batch_size // self.train_config.gradient_accumulation_steps
                )
                micro_prompts = sample_prompts[start_index:end_index]
                micro_responses = all_responses[start_index:end_index]
                micro_advantages = torch.tensor(advantages[start_index:end_index]).to(self.device)
                micro_raw_rewards = torch.tensor(raw_rewards[start_index:end_index]).to(self.device)
                # micro_old_log_probs = old_log_probs[start_index:end_index]

                tokenized = tokenize_prompt_and_output(micro_prompts, micro_responses, self.tokenizer)
                input_ids = tokenized["input_ids"].to(self.device, non_blocking=True)
                labels = tokenized["labels"].to(self.device, non_blocking=True)
                response_mask = tokenized["response_attention_mask"].to(self.device, non_blocking=True)

                T = response_mask.shape[1]
                micro_old_log_probs = pad_logprobs_to_T(
                    old_lp_list=old_log_probs[start_index:end_index],
                    T=T,
                    pad_value=0.0,
                ).to(self.device, non_blocking=True)

                with self.ctx:
                    policy_outputs = get_response_log_probs(
                        self.model,
                        input_ids=input_ids,
                        labels=labels,
                        return_token_entropy=False,
                    )
                    policy_log_probs = policy_outputs["log_probs"]

                micro_loss, micro_metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
                    loss_type=self.train_config.loss_type,
                    raw_rewards=micro_raw_rewards,
                    advantages=micro_advantages,
                    old_log_probs=micro_old_log_probs,
                    cliprange=self.train_config.cliprange,
                )

                batch_loss += micro_loss.item()

            print_color(
                f"GRPO Step {self.grpo_step} | Train Step {train_step + 1}/{n_train_steps} | "
                f"Batch Loss: {batch_loss:.4f}",
                color="green",
            )
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.grpo_step += 1
        return raw_rewards, advantages, metadata

    def train(
        self,
        vllm,
    ):
        for _ in trange(
            self.train_config.n_grpo_steps,
            desc="GRPO Training Steps",
        ):
            self.grpo_train_step(
                vllm,
            )

            if self.grpo_step % 5 == 0:
                load_policy_into_vllm_instance(
                    self.model,
                    vllm,
                )
                out = self.evaluate(vllm)
                log_dict = {}
                log_dict["eval/answer_accuracy"] = out["answer_accuracy"]
                log_dict["eval/answer_correct"] = out["answer_correct"]
                log_dict["eval/format_correct"] = out["format_correct"]
                log_dict["eval/formatted_but_answer_wrong"] = out["formatted_but_answer_wrong"]
                log_dict["eval/reward_1"] = out["reward_1"]
                wandb.log(log_dict, step=self.grpo_step)

                self.sample_responses(vllm=vllm, num_samples=3)

            # Save checkpoint
            checkpoint_file = os.path.join(
                self.checkpoint_path,
                f"grpo_step_{self.grpo_step}.pt",
            )
            torch.save(self.model.state_dict(), checkpoint_file)
            print_color(f"Saved GRPO checkpoint at {checkpoint_file}", color="magenta")
