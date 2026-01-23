from typing import Callable

import torch
import torch.nn.functional as F

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import generate_responses

REWARD_FN_MAP = {
    "r1_zero_reward_fn": r1_zero_reward_fn,
    "question_only_reward_fn": question_only_reward_fn,
}


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict:
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

    MAX_LEN = max(len(ids) for ids in input_ids)
    # 151643 for Qwen/Qwen2.5-Math-1.5B
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad_to(x, value):
        return x + [value] * (MAX_LEN - len(x))

    full = torch.tensor([pad_to(x, pad_id) for x in input_ids], dtype=torch.long)
    response_mask = torch.tensor([pad_to(x, False) for x in response_mask], dtype=torch.bool)

    assert full.shape == response_mask.shape, "Shapes of full and response_mask must match"

    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()
    response_mask = response_mask[:, 1:].contiguous()

    assert input_ids.shape == labels.shape == response_mask.shape, (
        "Shapes of input_ids, labels, and response_mask must match"
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of the probability distribution defined by the logits.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy


def get_response_log_probs(
    model, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    logits = model(input_ids=input_ids).logits

    logp = F.log_softmax(logits, dim=-1)
    log_probs = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    res = {
        "log_probs": log_probs,
    }
    if return_token_entropy:
        entropy = compute_entropy(logits)
        res["token_entropy"] = entropy
    return res


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    mask_f = mask.type_as(tensor)
    masked_tensor = tensor * mask_f
    sum_masked = torch.sum(masked_tensor, dim=dim)
    count_nonzero = torch.sum(mask, dim=dim).clamp(min=1e-8)
    mean = sum_masked / count_nonzero
    return mean


def masked_normalize(
    tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float = 1.0, dim: int | None = None
) -> torch.Tensor:
    assert tensor.shape == mask.shape, "Tensor and mask must have the same shape"

    masked_f = mask.type_as(tensor)
    masked_tensor = tensor * masked_f
    masked_sum = torch.sum(masked_tensor, dim=dim) if dim is not None else torch.sum(masked_tensor)

    return masked_sum / normalize_constant


@torch.no_grad()
def log_generation(
    prompts: list[str],
    true_answers: list[str],
    reward_fn: Callable,
    model,
    tokenizer,
    vllm,
    sampling_params,
):
    device = next(model.parameters()).device
    responses = generate_responses(
        vllm,
        prompts,
        sampling_params,
    )

    reward_dicts = [reward_fn(resp, gt) for resp, gt in zip(responses, true_answers)]

    total_rewards = torch.tensor([float(d["reward"]) for d in reward_dicts])
    fmt_rewards = torch.tensor([float(d["format_reward"]) for d in reward_dicts])
    ans_rewards = torch.tensor([float(d["answer_reward"]) for d in reward_dicts])
    correct = total_rewards == 1.0

    tok = tokenize_prompt_and_output(
        prompts,
        responses,
        tokenizer,
    )
    input_ids, labels, response_mask = tok["input_ids"], tok["labels"], tok["response_mask"]

    model.eval()
    out = get_response_log_probs(
        model,
        input_ids=input_ids.to(device),
        labels=labels.to(device),
        return_token_entropy=True,
    )
    ent = out["token_entropy"].cpu()

    res_len = response_mask.sum(dim=1).type_as(total_rewards)  # Number of response tokens per sample
    avg_ent = (ent * response_mask.type_as(ent)).sum(dim=1) / res_len

    rows = [
        {
            "prompt": p,
            "response": r,
            "true_answer": gt,
            "total_reward": float(tr.item()),
            "format_reward": float(fr.item()),
            "answer_reward": float(ar.item()),
            "is_correct": bool(c.item()),
            "response_length": int(rl.item()),
            "avg_token_entropy": float(ae.item()),
        }
        for p, r, gt, tr, fr, ar, c, rl, ae in zip(
            prompts,
            responses,
            true_answers,
            total_rewards,
            fmt_rewards,
            ans_rewards,
            correct,
            res_len,
            avg_ent,
        )
    ]

    summary = {
        "avg_reward": float(total_rewards.float().mean().item()),
        "avg_token_entropy": float(avg_ent.mean().detach().cpu().item()),
        "avg_resp_len": float(res_len.float().mean().item()),
        "avg_len_correct": float(res_len[correct].float().mean().item()) if correct.any() else 0.0,
        "avg_len_wrong": float(res_len[~correct].float().mean().item()) if (~correct).any() else 0.0,
        "n_examples": len(prompts),
    }

    model.train()
    return {"summary": summary, "rows": rows}


def compute_rewards_from_responses(
    responses: list[str],
    true_answers: list[str],
    reward_fn: Callable,
) -> list[dict]:
    rewards = []
    for response, answer in zip(responses, true_answers):
        reward = reward_fn(response, answer)
        rewards.append(reward)
    return rewards
