import torch
import torch.nn.functional as F  

from transformers import PretrainedTokenizer 

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer: PreTrainedTokenizerBase):
    assert len(prompt_strs) == len(output_strs)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # Tokenize separately (no padding yet)
    prompt_enc = tokenizer(prompt_strs, padding=False, return_tensors=None)["input_ids"]
    output_enc = tokenizer(output_strs, padding=False, return_tensors=None)["input_ids"]

    # Concatenate and record lengths
    concat_ids = []
    prompt_lens = []
    for p_ids, o_ids in zip(prompt_enc, output_enc):
        concat_ids.append(p_ids + o_ids)
        prompt_lens.append(len(p_ids))

    # Find max length, then subtract 1 for shift
    L_max = max(len(seq) for seq in concat_ids)
    T = L_max - 1

    # Prepare tensors
    input_ids = torch.full((len(concat_ids), T), pad_id, dtype=torch.long)
    labels = torch.full((len(concat_ids), T), pad_id, dtype=torch.long)
    response_mask = torch.zeros((len(concat_ids), T), dtype=torch.long)

    for i, (seq, p_len) in enumerate(zip(concat_ids, prompt_lens)):
        inp = seq[:-1]
        lab = seq[1:]
        n = len(inp)
        input_ids[i, :n] = torch.tensor(inp, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        response_mask[i, max(p_len - 1, 0):n] = 1  # mark only response tokens

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)



from transformers import PreTrainedModel 

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor, 
    labels: torch.Tensor, 
    return_token_entropy: bool = False 
) -> dict[str, torch.Tensor]:


    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    
    cond_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs": cond_log_probs, "token_entropy": token_entropy}

    return {"log_probs": cond_log_probs}


def masked_normalize(
    tensor: torch.Tensor, 
    mask: torch.Tensor,
    normalize_constant: float, 
    dim: int | None = None 
) -> torch.Tensor:
    mask_bool = mask.bool()
    masked_tensor = tensor.masked_fill(~mask_bool, 0.0)
    
    summed = masked_tensor.sum(dim = dim ) if dim is not None else masked_tensor.sum()
    return tensor / (summed + normalize_constant)



def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
