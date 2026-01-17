import json
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from cs336_alignment.config import TrainConfig
from cs336_alignment.lr import get_lr, update_learning_rate
from cs336_alignment.utils import (
    clear_memory,
    cycle_dataloader,
    get_ctx,
    print_color,
    save_model_checkpoint,
    to_float,
)

# from cs336_alignment.vllm_utils import load_vllm_model


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, torch.Tensor]:
    prompt_tokens = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    output_tokens = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    input_ids = []
    response_mask = []

    for p_ids, o_ids in zip(prompt_tokens["input_ids"], output_tokens["input_ids"]):
        combined_ids = p_ids + o_ids
        input_ids.append(combined_ids)

        mask = ([False] * len(p_ids)) + ([True] * len(o_ids))
        response_mask.append(mask)

    max_len = max(len(ids) for ids in input_ids)
    pad_id = tokenizer.pad_token_id

    def pad_to(x, value):
        return x + [value] * (max_len - len(x))

    full = torch.tensor([pad_to(x, pad_id) for x in input_ids], dtype=torch.long)
    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()
    response_mask = torch.tensor([pad_to(x, False) for x in response_mask], dtype=torch.bool)[
        :, 1:
    ].contiguous()

    assert input_ids.shape == labels.shape == response_mask.shape, (
        "Shapes of input_ids, labels, and response_mask must match"
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(
    model, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    out = model(input_ids=input_ids)
    logits = out.logits

    logp = nn.functional.log_softmax(logits, dim=-1)
    log_probs = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    res = {
        "log_probs": log_probs,
    }
    if return_token_entropy:
        entropy = compute_entropy(logits)
        res["token_entropy"] = entropy
    return res


def masked_normalize(
    tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: int = 1, dim: int | None = None
) -> torch.Tensor:
    assert tensor.shape == mask.shape, "Tensor and mask must have the same shape"

    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return torch.sum(masked_tensor, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int = 1,
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

    loss_unscaled = -loss_unscaled.mean()
    loss_scaled = loss_unscaled / gradient_accumulation_steps
    loss_scaled.backward()

    # For logging purposes
    # per_token_loss = -policy_log_probs
    # with torch.no_grad():
    #     m = response_mask.to(dtype=policy_log_probs.dtype)
    #     denom = m.sum().clamp_min(1.0)
    #     mean_logp = (policy_log_probs * m).sum() / denom
    #     mean_ce = (per_token_loss * m).sum() / denom

    metadata = {
        "loss_unscaled": loss_unscaled.detach(),
        # "mean_ce": mean_ce.detach(),
        # "mean_logp": mean_logp.detach(),
        # "response_tokens": denom.detach(),
        # "grad_accum_steps": torch.tensor(float(gradient_accumulation_steps), device=policy_log_probs.device),
    }
    return loss_scaled.detach(), metadata


class SFTDataset(Dataset):
    def __init__(self, questions: list[str], cots: list[str], answers: list[str], prompt_template_path: str):
        self.questions = questions
        self.cots = cots
        self.answers = answers

        with open(prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        prompt = self.prompt_template.format(question=question)
        cot = self.cots[idx]
        answer = self.answers[idx]

        return prompt, cot, answer

    @classmethod
    def load_from_disk(cls, path: str, prompt_template_path: str):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        questions = []
        cots = []
        answers = []
        for row in rows:
            questions.append(row["question"])
            cots.append(row["cot"])
            answers.append(row["answer"])

        return cls(questions, cots, answers, prompt_template_path=prompt_template_path)


def collate_fn(batch, tokenizer):
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

    tokenized["answers"] = list(answers)
    return tokenized


class SFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        device: torch.device,
        dataset_dir_base: str = "./data/pre-processed",
    ):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=train_config.model_name,
            use_fast=True,
        )
        self.device = device
        self.train_config = train_config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=train_config.betas,
            lr=self.train_config.min_lr,
            weight_decay=self.train_config.weight_decay,
        )
        self.ctx = get_ctx(
            use_mixed=self.train_config.mixed_precision_training,
            device=device,
        )

        dataset_dir = os.path.join(dataset_dir_base, train_config.dataset_name)
        train_dataset = SFTDataset.load_from_disk(
            os.path.join(dataset_dir, "train.jsonl"), train_config.prompt_template_path
        )

        self.sample_dataset = [train_dataset[i] for i in range(2)]

        test_dataset = SFTDataset.load_from_disk(
            os.path.join(dataset_dir, "test.jsonl"), train_config.prompt_template_path
        )

        self.train_dataloader = cycle_dataloader(
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
                drop_last=False,
            )
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.train_config.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
        )

        self.checkpoint_path = os.path.join(
            train_config.checkpoint_dir,
            f"sft_{train_config.model_name.split('/')[-1]}_{train_config.dataset_name}",
        )
        os.makedirs(self.checkpoint_path, exist_ok=True)
        train_config.to_json(
            os.path.join(self.checkpoint_path, "train_config.json"),
        )
        self.start_step = 0

    @classmethod
    def load_from_checkpoint(cls, model, checkpoint_path: str, device: torch.device) -> "SFTTrainer":
        state = torch.load(os.path.join(checkpoint_path), map_location=device)
        train_config = TrainConfig.from_json(
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
        print_color("Evaluating SFT model on test dataset...", color="magenta")
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.test_dataloader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                response_mask = batch["response_mask"].to(self.device, non_blocking=True)

                with self.ctx:
                    policy_outputs = get_response_log_probs(
                        self.model,
                        input_ids=input_ids,
                        labels=labels,
                        return_token_entropy=False,
                    )
                    policy_log_probs = policy_outputs["log_probs"]

                    loss_scaled, metadata = sft_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=1,
                        normalize_constant=1,
                    )

                total_loss += to_float(loss_scaled)
                total_batches += 1

        avg_loss = total_loss / total_batches
        print_color(f"Evaluation completed. Average Loss: {avg_loss:.4f}\n", color="magenta")
        self.model.train()
        return avg_loss

    @torch.no_grad()
    def sample_responses(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        vllm=None,
    ) -> list[str]:
        print_color("Sampling responses from SFT model...", color="cyan")
        # print(self.sample_dataset)
        prompts, cots, true_answers = zip(*self.sample_dataset)
        prompts = list(prompts)
        cots = list(cots)
        true_answers = list(true_answers)

        if vllm is not None:
            pass
            # load
        else:
            self.model.eval()
            input_tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
                return_attention_mask=True,
            ).to(self.device)
            # Quick sanity check to fail fast if the model has NaNs/Infs
            with torch.no_grad():
                out = self.model(
                    input_ids=input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"]
                )
                if not torch.isfinite(out.logits).all():
                    bad = (~torch.isfinite(out.logits)).sum().item()
                    raise RuntimeError(
                        f"Model produced non-finite logits during sampling (count={bad}). "
                        "This usually indicates NaNs/Infs in weights or unstable training."
                    )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_tokens["input_ids"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    attention_mask=input_tokens["attention_mask"],
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            responses = []
            for i, gen_ids in enumerate(generated_ids):
                gen_text = self.tokenizer.decode(
                    gen_ids[input_tokens["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                responses.append(gen_text.strip())

            self.model.train()
        print_color("Sampled Responses:", color="cyan")
        for i, (prompt, response, true_answer) in enumerate(zip(prompts, responses, true_answers)):
            print_color(f"=== Example {i + 1} ===", color="cyan")
            print_color(f"[green]Prompt[green]: {prompt}", color="cyan")
            print_color(f"[green]Response[green]: {response}", color="cyan")
            print_color(f"[green]True Answer[green]: {true_answer}\n", color="cyan")

        return responses

    def train_step(
        self,
    ) -> float:
        batch_loss = 0.0
        for micro_step in range(self.train_config.gradient_accumulation_steps):
            print_color(
                f"    Microbatch step {micro_step + 1}/{self.train_config.gradient_accumulation_steps}",
                color="blue",
            )
            batch = next(self.train_dataloader)

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            response_mask = batch["response_mask"].to(self.device, non_blocking=True)

            with self.ctx:
                policy_outputs = get_response_log_probs(
                    self.model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                policy_log_probs = policy_outputs["log_probs"]

                loss_scaled, metadata = sft_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
                    normalize_constant=1,
                )

            del input_ids, labels, response_mask
            batch_loss += to_float(loss_scaled)

        nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return batch_loss / self.train_config.gradient_accumulation_steps

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

        self.sample_responses(
            vllm=vllm,
        )

        for step in range(self.start_step, self.train_config.total_training_steps):
            self.model.train()
            print_color(
                f"Starting training step {step + 1}/{self.train_config.total_training_steps}", color="yellow"
            )
            log_dict = {}

            update_learning_rate(
                optimizer=self.optimizer,
                step=step,
                train_config=self.train_config,
            )
            loss = self.train_step()
            log_dict["train/loss"] = loss

            print_color(
                f"Step {step + 1}/{self.train_config.total_training_steps}, Loss: {loss:.4f}, Lr: {get_lr(self.optimizer):.6f}\n"
            )

            if (step + 1) % self.train_config.sample_interval == 0:
                # clear_memory()
                self.sample_responses(
                    vllm=vllm,
                )

            # if (step + 1) % self.train_config.eval_steps == 0:
            #     clear_memory()
            #     self.evaluate(vllm)

            # wandb.log(log_dict, step=(step + 1))
            if self.train_config.wandb_logging:
                wandb.log(log_dict, step=(step + 1))

            # if (step + 1) % self.train_config.save_interval == 0:
            #     checkpoint_file = os.path.join(self.checkpoint_path, f"checkpoint_step_{step + 1}.pt")
            #     save_model_checkpoint(
            #         model=self.model,
            #         optimizer=self.optimizer,
            #         cur_step=step + 1,
            #         checkpoint_path=checkpoint_file,
            #     )
