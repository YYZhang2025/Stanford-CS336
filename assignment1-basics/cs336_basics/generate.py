from cs336_basics.tokenizer.tokenizer import BPETokenizer
import torch


def top_p_sampling(
    logits: torch.Tensor,
    top_p: float,
):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Create a mask for tokens to keep
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter back to original indexing
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        1, sorted_indices, sorted_indices_to_remove
    )

    # Set logits of tokens to remove to -inf
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def top_k_sampling(
    logits: torch.Tensor,
    top_k: int,
):
    if top_k <= 0:
        return logits  # No filtering

    # Get the top_k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

    # Create a mask for tokens to keep
    indices_to_remove = torch.ones_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(1, top_k_indices, 0)

    # Set logits of tokens to remove to -inf
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor | str,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str] | None = None,
    max_new_tokens: int = 100,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
) -> dict:
    # Load Tokenizer
    tokenizer = BPETokenizer.from_file(vocab_filepath, merges_filepath, special_tokens)

    if isinstance(prompt, str):
        input_ids = tokenizer.encode(prompt).unsqueeze(0)  # Add batch dimension
    else:
        input_ids = prompt.unsqueeze(0)  # Add batch dimension

    input_ids = input_ids.to(next(model.parameters()).device)
    input_len = input_ids.shape[1]

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]  # Get logits for the last token

        # Sample from the distribution
        assert temperature > 0.0, "Temperature must be positive."
        assert top_p == 0.0 or top_k == 0, "Only one of top_p or top_k should be set."
        next_token_logits = next_token_logits / temperature
        if top_k > 0:
            next_token_id = top_k_sampling(next_token_logits, top_k)
        elif top_p > 0.0:
            next_token_id = top_p_sampling(next_token_logits, top_p)
        else:
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)  # Greedy if no sampling

        if next_token_id.item() == tokenizer.eos_token_id:
            break  # Stop if EOS token is generated
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)  # Append to input_ids

    input_ids = input_ids.squeeze(0)  # Remove batch dimension
    all_text = tokenizer.decode(input_ids.tolist())
    generated_ids = input_ids[input_len:]
    generated_text = tokenizer.decode(input_ids.tolist()[generated_ids:])

    return {
        "all_text": all_text,
        "generated_text": generated_text,
        "generated_ids": generated_ids,
    }
