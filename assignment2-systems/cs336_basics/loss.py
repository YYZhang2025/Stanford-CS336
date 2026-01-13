import torch
import torch.nn as nn


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor):
    # Subtract the largest element for numerical stability.
    logits = logits - torch.max(logits, dim=1, keepdim=True).values

    # Cancel out log and exp whenever possible.
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))

    labels = labels.unsqueeze(1)
    loss = log_probs.gather(1, labels).squeeze(1)
    loss = -loss.mean()
    return loss


def perplexity(loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(loss)
