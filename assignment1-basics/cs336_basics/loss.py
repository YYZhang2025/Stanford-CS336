import torch
import torch.nn as nn


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor):
    # Subtract the largest element for numerical stability.
    logits = logits - torch.max(logits, dim=1, keepdim=True).values

    # Cancel out log and exp whenever possible.
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))

    # Handle Labels
    # if labels.dim() == 1:
    #     labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1]).float()
    # else:
    #     labels = labels.float()

    # loss = -torch.mean(torch.sum(labels * log_probs, dim=1))
    # return loss

    labels = labels.unsqueeze(1)
    loss = log_probs.gather(1, labels).squeeze(1)
    loss = -loss.mean()
    return loss


def perplexity(loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(loss)
