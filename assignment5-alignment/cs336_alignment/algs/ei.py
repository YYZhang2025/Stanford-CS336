import os
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoTokenizer, PreTrainedModel
from vllm import SamplingParams

from cs336_alignment.algs.sft import SFTDataset, sft_collate_fn, sft_microbatch_train_step
from cs336_alignment.algs.utils import (
    REWARD_FN_MAP,
    compute_rewards_from_responses,
    get_response_log_probs,
    log_generation,
    sample_g_outputs_per_prompt,
)
from cs336_alignment.base_config import BaseConfig
from cs336_alignment.eval import evaluate_responses
from cs336_alignment.lr import update_learning_rate
from cs336_alignment.utils import (
    clear_memory,
    cycle_dataloader,
    get_ctx,
    load_dataset,
    print_color,
    print_rich_dict,
    to_float,
)
from cs336_alignment.vllm_utils import load_policy_into_vllm_instance


@dataclass
class EITrainConfig(BaseConfig):
    # EI training config
    ei_steps: int = 5
    ei_batch_size: int = 512
    reward_fn: str = "r1_zero_reward_fn"
    num_responses_per_prompt: int = 4

    # SFT hyperparameters
    sft_steps_per_ei_step: int = 100
    sft_batch_size: int = 128
    sft_gradient_accumulation_steps: int = 64

    # Optimizer hyperparameters
    betas: tuple = field(default=(0.9, 0.98))
    weight_decay: float = 1e-5
    max_lr: float = 5e-6
    max_grad_norm: float = 1.0

    # Other training options
    mixed_precision_training: bool = True

    save_interval: int = 100
    checkpoint_dir: str = "./checkpoints"

    # Evaluation and sampling
    eval_steps: int = 50
    seed: int = 42

    # For VLLM sampling during evaluation and response sampling
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 1024
    sampling_top_p: float = 1.0
    sampling_stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])

    def __post_init__(self):
        self.run_name = f"ei_dataset({self.dataset_name})_prompt({self.prompt_template_path.split('/')[-1]})_reward({self.reward_fn})"

        self.sft_micro_batch_size = self.sft_batch_size // self.sft_gradient_accumulation_steps
        self.total_training_steps = self.ei_steps * self.sft_steps_per_ei_step


# ========== dataset loading functions ==========#
def get_ei_batch(
    prompts: list[str],
    answers: list[str],
    batch_size: int = 512,
    num_responses_per_prompt: int = 4,
):
    random_index = random.sample(range(len(prompts)), k=batch_size)
    random_prompts = [prompts[i] for i in random_index]
    random_answers = [answers[i] for i in random_index]

    all_prompts = []
    for prompt in random_prompts:
        all_prompts.extend([prompt] * num_responses_per_prompt)
    all_true_answers = []
    for answer in random_answers:
        all_true_answers.extend([answer] * num_responses_per_prompt)

    return {
        "prompts": list(all_prompts),
        "true_answers": list(all_true_answers),
    }


# ========== end dataset loading functions ==========#


# ========== EI Alg helper functions ==========#


def filter_by_reward(
    prompts: list[str],
    responses: list[str],
    answers: list[str],
    rewards: list[dict],
):
    filtered_prompts = []
    filtered_responses = []
    filtered_answers = []
    for prompt, response, answer, reward in zip(prompts, responses, answers, rewards):
        if reward["reward"] == 1:
            filtered_prompts.append(prompt)
            filtered_responses.append(response)
            filtered_answers.append(answer)

    return filtered_prompts, filtered_responses, filtered_answers


# ========== end EI Alg helper functions ==========#


class EITrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_config: EITrainConfig,
        device: torch.device,
        dataset_dir_base: str = "./data/pre-processed",
    ):
        self.model = model
        self.device = device
        self.train_config = train_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=train_config.model_name,
            use_fast=True,
        )

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

        # ======= Load Dataset =======#
        dataset_dir = os.path.join(dataset_dir_base, train_config.dataset_name)
        train_prompts, train_cots, train_answers = load_dataset(
            os.path.join(dataset_dir, "train.jsonl"), train_config.prompt_template_path
        )
        self.train_prompts = train_prompts
        self.train_answers = train_answers

        test_prompts, test_cots, test_answer = load_dataset(
            os.path.join(dataset_dir, "test.jsonl"), train_config.prompt_template_path
        )
        self.test_prompts = test_prompts
        self.test_true_answers = test_answer
        # ============================#

        # ======= Other Initializations =======#
        self.reward_fn = REWARD_FN_MAP[train_config.reward_fn]

        self.checkpoint_path = os.path.join(
            train_config.checkpoint_dir,
            f"sft_{train_config.model_name.split('/')[-1]}_{train_config.dataset_name}",
        )
        os.makedirs(self.checkpoint_path, exist_ok=True)
        train_config.to_json(
            os.path.join(self.checkpoint_path, "train_config.json"),
        )

        self.sampling_params = SamplingParams(
            temperature=self.train_config.sampling_temperature,
            max_tokens=self.train_config.sampling_max_tokens,
            top_p=self.train_config.sampling_top_p,
            include_stop_str_in_output=True,
            stop=self.train_config.sampling_stop_tokens,
        )

        # ======= Other state variables =======#
        self.ei_start_step = 0
        self.sft_start_step = 0
        self.ei_cur_step = 0
        self.sft_cur_step = 0

    @torch.no_grad()
    def evaluate(self, vllm=None):
        print_color(f"Evaluating EI model on test dataset at step {self.ei_cur_step}", color="magenta")

        overview = evaluate_responses(
            vllm=vllm,
            prompts=self.test_prompts,
            answers=self.test_true_answers,
            sampling_params=self.sampling_params,
        )

        print_color("Evaluation Overview", color="magenta")
        print_rich_dict(overview)
        return overview

    @torch.no_grad()
    def sample_responses(
        self,
        vllm=None,
        num_samples: int = 5,
    ):
        print_color(f"Sampling {num_samples} responses from EI model...", color="cyan")

        index = random.sample(range(len(self.test_prompts)), k=num_samples)
        prompts = [self.test_prompts[i] for i in index]
        true_answers = [self.test_true_answers[i] for i in index]

        out = log_generation(
            prompts=prompts,
            true_answers=true_answers,
            reward_fn=self.reward_fn,
            model=self.model,
            tokenizer=self.tokenizer,
            vllm=vllm,
            sampling_params=self.sampling_params,
        )

        print_rich_dict(out)

    def sft_train_step(
        self,
        prompts: list[str],
        responses: list[str],
        answers: list[str],
    ):
        print_color(
            f"ei step {self.ei_cur_step} | Performing SFT training step on {len(prompts)} examples",
            color="green",
        )

        train_dataset = SFTDataset(prompts, responses, answers)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_config.sft_micro_batch_size,
            shuffle=True,
            collate_fn=lambda batch: sft_collate_fn(batch, self.tokenizer),
            drop_last=False,
        )
        train_dataloader = cycle_dataloader(train_dataloader)

        # Perform SFT steps
        # Here we assume for each EI step, we do a fixed number of SFT steps
        # It is also possible to do SFT for one epoch over the filtered dataset
        for _ in range(self.train_config.sft_steps_per_ei_step):
            batch_loss = 0.0
            token_entropy_avg = 0.0
            self.sft_cur_step += 1

            # Accumulate gradients over micro-batches
            for _ in trange(
                self.train_config.sft_gradient_accumulation_steps,
                desc="micro-batches",
                leave=False,
            ):
                micro_batch = next(train_dataloader)
                input_ids = micro_batch["input_ids"].to(self.device, non_blocking=True)
                labels = micro_batch["labels"].to(self.device, non_blocking=True)
                response_mask = micro_batch["response_mask"].to(self.device, non_blocking=True)

                with self.ctx:
                    policy_outputs = get_response_log_probs(
                        self.model,
                        input_ids=input_ids,
                        labels=labels,
                        return_token_entropy=True,
                    )

                    policy_log_probs = policy_outputs["log_probs"]
                    token_entropy = policy_outputs["token_entropy"]

                    loss_scaled, metadata = sft_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=self.train_config.sft_gradient_accumulation_steps,
                        normalize_constant=1,
                    )

                del input_ids, labels, response_mask
                batch_loss += to_float(loss_scaled)
                token_entropy_avg += (
                    to_float(token_entropy.mean()) / self.train_config.sft_gradient_accumulation_steps
                )

            nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
            update_learning_rate(
                optimizer=self.optimizer,
                step=self.sft_cur_step,
                total_steps=self.train_config.total_training_steps,
                max_lr=self.train_config.max_lr,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            clear_memory()

            print_color(
                f"ei step {self.ei_cur_step} | sft step {self.sft_cur_step} | loss: {batch_loss:.4f} | token_entropy: {token_entropy_avg:.4f}",
                color="green",
            )

            if self.train_config.wandb_logging:
                wandb.log(
                    {
                        "train/loss": batch_loss,
                        "train/token_entropy": token_entropy_avg,
                    },
                    step=self.sft_cur_step,
                )

    def train(self, vllm=None):
        print_color("||" + "=" * 80, color="green")
        print_color("||Starting SFT training...", color="green")
        print_color("||Training on dataset: " + self.train_config.dataset_name, color="green")
        print_color("||" + "=" * 80, color="green")

        load_policy_into_vllm_instance(self.model, vllm)
        for step in range(self.ei_start_step, self.train_config.ei_steps):
            self.model.train()
            self.ei_cur_step = step + 1

            # 3. Sample a batch of questions from the training dataset
            ei_batch = get_ei_batch(
                prompts=self.train_prompts,
                answers=self.train_answers,
                batch_size=self.train_config.ei_batch_size,
                num_responses_per_prompt=self.train_config.num_responses_per_prompt,
            )
            prompts = ei_batch["prompts"]
            true_answers = ei_batch["true_answers"]

            # 4. Set the old policy model
            # (already set at the end of last EI step)

            # 5. Sample responses from the current policy model
            print_color(
                f"Sampling EI batch at EI step {self.ei_cur_step} on {self.train_config.ei_batch_size} samples with {self.train_config.num_responses_per_prompt} responses per prompt",
                color="green",
            )
            sampled_prompts, sampled_responses = sample_g_outputs_per_prompt(
                vllm,
                self.sampling_params,
                prompts,
            )

            # 6. Compute rewards from the sampled responses
            rewards_dict = compute_rewards_from_responses(
                sampled_responses,
                true_answers,
                reward_fn=REWARD_FN_MAP[self.train_config.reward_fn],
            )

            # 7. Filter responses by reward
            filtered_prompts, filtered_responses, filtered_answers = filter_by_reward(
                sampled_prompts,
                sampled_responses,
                true_answers,
                rewards_dict,
            )

            # 8. Perform SFT training step on the filtered responses
            if len(filtered_prompts) > 0:
                self.sft_train_step(filtered_prompts, filtered_responses, filtered_answers)

            # Evaluation and sampling
            clear_memory()
            log_dict = {}
            load_policy_into_vllm_instance(self.model, vllm)  # load updated model into vllm

            self.sample_responses(
                vllm=vllm,
            )

            out = self.evaluate(vllm)
            log_dict["eval/answer_accuracy"] = out["answer_accuracy"]
            log_dict["eval/answer_correct"] = out["answer_correct"]
            log_dict["eval/format_correct"] = out["format_correct"]
            log_dict["eval/formatted_but_answer_wrong"] = out["formatted_but_answer_wrong"]
            log_dict["eval/reward_1"] = out["reward_1"]

            if self.train_config.wandb_logging:
                wandb.log(log_dict, step=self.sft_cur_step)
