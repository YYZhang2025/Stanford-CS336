import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer: PreTrainedTokenizerBase):
    assert len(prompt_strs) == len(output_strs), "Prompt and output lists must have the same length"

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # Tokenize separately (no padding yet)
    prompt_enc = tokenizer(prompt_strs, padding=False, add_special_tokens=False)["input_ids"]
    output_enc = tokenizer(output_strs, padding=False, add_special_tokens=False)["input_ids"]

    # Concatenate and record lengths
    concat_ids = []
    prompt_lens = []
    for p_ids, o_ids in zip(prompt_enc, output_enc):
        concat_ids.append(p_ids + o_ids)  # EOS at the end of each output
        prompt_lens.append(len(p_ids))

    # Find max length, then subtract 1 for shift
    T = max(len(seq) for seq in concat_ids) - 1
    B = len(concat_ids)

    # Prepare tensors
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), pad_id, dtype=torch.long)
    response_mask = torch.zeros((B, T), dtype=torch.long)

    for i, (seq, p_len) in enumerate(zip(concat_ids, prompt_lens)):
        inp = seq[:-1] if len(seq) == (T + 1) else seq[:]  # Input
        n = len(inp)
        input_ids[i, :n] = torch.tensor(inp, dtype=torch.long)

        lab = seq[1:]  # Label
        n = len(lab)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        response_mask[i, max(p_len - 1, 0) : n] = 1  # mark only response tokens

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


from transformers import PreTrainedModel


def get_response_log_probs(
    model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)

    cond_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs": cond_log_probs, "token_entropy": token_entropy}

    return {"log_probs": cond_log_probs}


def masked_normalize(
    tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float = 1.0, dim: int | None = None
) -> torch.Tensor:
    assert normalize_constant != 0, "Normalization constant must not be zero"
    masked_tensor = tensor * mask

    summed = masked_tensor.sum(dim=dim) if dim is not None else masked_tensor.sum()
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single micro-batch SFT update.

    Args:
        policy_log_probs: (B, T) per-token log-probs for the *ground-truth* tokens.
        response_mask:   (B, T) 1 for response tokens, 0 for prompt/pad.
        gradient_accumulation_steps: number of micro-batches per optimizer step.
        normalize_constant: divide the masked sum by this value (e.g., total response tokens in microbatch).
    Returns:
        loss (scalar tensor) and metadata dict.
    """
    assert gradient_accumulation_steps > 0, "Gradient accumulation steps must be positive"
    assert policy_log_probs.shape == response_mask.shape, (
        "policy_log_probs and response_mask must have same shape"
    )

    # Ensure proper dtypes
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant)

    # Adjust for gradient accumulation (simulate splitting the full batch)
    loss = -loss / policy_log_probs.shape[0] / gradient_accumulation_steps

    # Backprop for this microbatch
    loss.backward()

    # Metadata for logging (detach to avoid holding graph)
    # num_resp_tokens = mask.sum()
    # total_nll = -masked_sum / normalize_constant
    # avg_nll_per_token = total_nll / (num_resp_tokens)

    metadata: dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        # "num_response_tokens": num_resp_tokens.detach(),
        # "total_nll": total_nll.detach(),
        # "avg_nll_per_token": avg_nll_per_token.detach(),
    }

    return loss, metadata
