import os
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from transformers import AutoTokenizer, PreTrainedModel
from vllm import SamplingParams

from cs336_alignment.algs.utils import (
    REWARD_FN_MAP,
    get_response_log_probs,
    log_generation,
    masked_normalize,
    tokenize_prompt_and_output,
)
from cs336_alignment.base_config import BaseConfig
from cs336_alignment.eval import evaluate_responses
from cs336_alignment.lr import get_lr, update_learning_rate
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
class SFTTrainingConfig(BaseConfig):
    # Training hyperparameters
    total_training_steps: int = 500  # total number of training steps
    batch_size: int = 4  # the batch size of mini-batch
    # total effective batch size = batch_size * gradient_accumulation_steps
    gradient_accumulation_steps: int = 64

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
    eval_steps: int = 5
    seed: int = 42

    reward_fn: str = "r1_zero_reward_fn"

    # For VLLM sampling during evaluation and response sampling
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 1024
    sampling_top_p: float = 1.0
    sampling_stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])

    def __post_init__(self):
        self.run_name = f"sft_dataset({self.dataset_name})_prompt({self.prompt_template_path.split('/')[-1]})_reward({self.reward_fn})"


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    policy_log_probs: (batch_size, seq_len) - log probabilities from the policy model
    response_mask: (batch_size, seq_len) - boolean mask indicating response tokens 1 for normalization
    gradient_accumulation_steps: number of microbatches to accumulate gradients over
    normalize_constant: constant to normalize the loss
    """

    loss_unscaled = masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )

    loss_unscaled = -loss_unscaled.mean()  # negative log likelihood per token

    loss_scaled = loss_unscaled / gradient_accumulation_steps
    loss_scaled.backward()

    metadata = {
        "loss_unscaled": loss_unscaled.detach(),
    }
    return loss_scaled.detach(), metadata


class SFTDataset(Dataset):
    def __init__(
        self,
        questions: list[str],
        cots: list[str],
        answers: list[str],
        prompt_template_path: str | None = None,
    ):
        self.questions = questions
        self.cots = cots
        self.answers = answers

        self.prompt_template = None
        if prompt_template_path is not None:
            with open(prompt_template_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()

        self.prompts = (
            [self.prompt_template.format(question=q) for q in self.questions]
            if self.prompt_template is not None
            else self.questions
        )

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        cot = self.cots[idx]
        answer = self.answers[idx]

        return prompt, cot, answer

    @classmethod
    def load_from_disk(cls, path: str, prompt_template_path: str):
        prompts, cots, answers = load_dataset(path, prompt_template_path=prompt_template_path)
        return cls(prompts, cots, answers)


def sft_collate_fn(batch, tokenizer):
    """
    return:
        {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask,
        }
    """
    prompts, cots, answers = zip(*batch)
    tokenized = tokenize_prompt_and_output(
        prompt_strs=list(prompts),
        output_strs=list(cots),
        tokenizer=tokenizer,
    )

    return tokenized


class SFTTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_config: SFTTrainingConfig,
        device: torch.device,
        dataset_dir_base: str = "./data/pre-processed",
    ):
        # ------- Model and Optimizer Setup -------#
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

        # ------- Dataset Loading -------#
        dataset_dir = os.path.join(dataset_dir_base, train_config.dataset_name)
        train_dataset = SFTDataset.load_from_disk(
            os.path.join(dataset_dir, "train.jsonl"), train_config.prompt_template_path
        )
        test_dataset = SFTDataset.load_from_disk(
            os.path.join(dataset_dir, "test.jsonl"), train_config.prompt_template_path
        )
        self.test_prompts = test_dataset.prompts
        self.test_true_answers = test_dataset.answers

        self.train_dataloader = cycle_dataloader(
            DataLoader(
                train_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                collate_fn=lambda batch: sft_collate_fn(batch, self.tokenizer),
                drop_last=False,
            )
        )

        # ------ Checkpointing and other setups -------#
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

        self.start_step = 0
        self.cur_step = 0
        self.ctx = get_ctx(
            use_mixed=self.train_config.mixed_precision_training,
            device=device,
        )

        self.reward_fn = REWARD_FN_MAP[self.train_config.reward_fn]

    @classmethod
    def load_from_checkpoint(cls, model, checkpoint_path: str, device: torch.device) -> "SFTTrainer":
        state = torch.load(os.path.join(checkpoint_path), map_location=device)
        train_config = SFTTrainingConfig.from_json(
            os.path.join(os.path.dirname(checkpoint_path), "train_config.json")
        )
        model.load_state_dict(state["model_state_dict"])
        trainer = cls(
            model=model,
            train_config=train_config,
            device=device,
        )
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        trainer.start_step = state["step"]

        print_color(
            f"Loaded SFTTrainer from checkpoint: {checkpoint_path}, starting from step {trainer.start_step}",
            color="green",
        )
        return trainer

    @torch.no_grad()
    def evaluate(self, vllm=None):
        print_color(f"Evaluating SFT model on test dataset at step {self.cur_step}", color="magenta")

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
        print_color(f"Sampling {num_samples} responses from SFT model...", color="cyan")

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

    def train_step(
        self,
    ) -> tuple[float, float]:
        batch_loss = 0.0
        token_entropy_avg = 0.0

        for _ in trange(
            self.train_config.gradient_accumulation_steps,
            desc="micro-batches",
            leave=False,
        ):
            micro_batch = next(self.train_dataloader)
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
                    gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
                    normalize_constant=1,
                )

            del input_ids, labels, response_mask
            batch_loss += to_float(loss_scaled)
            token_entropy_avg += (
                to_float(token_entropy.mean()) / self.train_config.gradient_accumulation_steps
            )

        nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
        update_learning_rate(
            optimizer=self.optimizer,
            step=self.cur_step,
            total_steps=self.train_config.total_training_steps,
            max_lr=self.train_config.max_lr,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        clear_memory()
        return batch_loss, token_entropy_avg

    def train(self, vllm=None):
        print_color("||" + "=" * 80, color="green")
        print_color("||Starting SFT training...", color="green")
        print_color("||Training on dataset: " + self.train_config.dataset_name, color="green")
        print_color(
            "||Total training steps: "
            + str(self.train_config.total_training_steps)
            + " | Batch size: "
            + str(self.train_config.batch_size)
            + " | Gradient accumulation steps: "
            + str(self.train_config.gradient_accumulation_steps),
            color="green",
        )
        print_color("||" + "=" * 80, color="green")

        for step in range(self.start_step, self.train_config.total_training_steps):
            self.model.train()
            log_dict = {}

            self.cur_step = step + 1
            print_color(
                f"Starting training step {self.cur_step}/{self.train_config.total_training_steps}",
                color="yellow",
            )

            # Main Training Step
            loss, token_entropy_avg = self.train_step()

            # Logging & Evaluation
            print_color(
                f"Step {self.cur_step}/{self.train_config.total_training_steps}, Loss: {loss:.4f}, Lr: {get_lr(self.optimizer):.7f}\n"
            )

            log_dict["train/loss"] = loss
            log_dict["train/token_entropy_avg"] = token_entropy_avg

            if self.cur_step % self.train_config.eval_steps == 0:
                clear_memory()
                load_policy_into_vllm_instance(self.model, vllm)

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
                wandb.log(log_dict, step=self.cur_step)
