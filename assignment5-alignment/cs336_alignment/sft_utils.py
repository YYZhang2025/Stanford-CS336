import torch
import torch.nn as nn
from torch.utils.data import Dataset

from cs336_alignment.utils import to_float


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
    tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float = 1.0, dim: int | None = None
) -> torch.Tensor:
    assert tensor.shape == mask.shape, "Tensor and mask must have the same shape"

    masked_tensor = tensor * mask
    sum_masked = masked_tensor.sum(dim=dim, keepdim=True)
    sum_mask = mask.sum(dim=dim, keepdim=True)

    normalized = sum_masked / (normalize_constant * (sum_mask + 1e-8))
    return normalized


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

    per_token_loss = -policy_log_probs

    loss_unscaled = masked_normalize(
        per_token_loss,
        response_mask,
        normalize_constant=normalize_constant,
        dim=None,
    )

    loss_scaled = loss_unscaled / gradient_accumulation_steps
    loss_scaled.backward()

    # For logging purposes
    with torch.no_grad():
        m = response_mask.to(dtype=policy_log_probs.dtype)
        denom = m.sum().clamp_min(1.0)
        mean_logp = (policy_log_probs * m).sum() / denom
        mean_ce = (per_token_loss * m).sum() / denom

    metadata = {
        "loss_unscaled": loss_unscaled.detach(),
        "mean_ce": mean_ce.detach(),
        "mean_logp": mean_logp.detach(),
        "response_tokens": denom.detach(),
        "grad_accum_steps": torch.tensor(float(gradient_accumulation_steps), device=policy_log_probs.device),
    }
    return loss_scaled.detach(), metadata


class SFTDataset(Dataset):
    def __init__(self, prompts: list[str], cots: list[str], answers: list[str]):
        self.prompts = prompts
        self.cots = cots
        self.answers = answers

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        cot = self.cots[idx]
        answer = self.answers[idx]

        return prompt, cot, answer


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
