import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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

    for i, (seq, prompt) in enumerate(zip(concat_ids, prompt_lens)):
        inp = seq[:-1] if len(seq) == (T + 1) else seq[:]  # Input
        n = len(inp)
        input_ids[i, :n] = torch.tensor(inp, dtype=torch.long)

        lab = seq[1:]  # Label
        n = len(lab)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        response_mask[i, max(prompt - 1, 0) : n] = 1  # mark only response tokens

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)  # p(x)
    log_probs = F.log_softmax(logits, dim=-1)  # log p(x)
    return -torch.sum(probs * log_probs, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits

    # First way
    # log_probs = F.log_softmax(logits, dim=-1)
    # cond_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Second way
    nll = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (B*T, V)
        labels.view(-1),  # (B*T,)
        reduction="none",
        ignore_index=-100,  # optional, if you mark pads as -100
    ).view(labels.size())  # reshape back to (B, T)

    cond_log_probs = -nll  # convert NLL to log probs

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
    assert gradient_accumulation_steps > 0, "Gradient accumulation steps must be positive"
    assert policy_log_probs.shape == response_mask.shape, (
        "policy_log_probs and response_mask must have same shape"
    )

    # Create loss
    # One thing to notice is that, the loss return by the masked_normalize function is
    # NOT averaged over the batch
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant)

    # average of the loss and scale by gradient accumulation steps
    loss = -loss / policy_log_probs.shape[0] / gradient_accumulation_steps

    # Backprop for this microbatch
    loss.backward()

    # Metadata for logging (detach to avoid holding graph)
    num_resp_tokens = response_mask.sum()  # The total number of tokens in the response

    metadata: dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "num_response_tokens": num_resp_tokens.detach(),
    }

    return loss, metadata
