import torch
import numpy as np


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy loss.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Extract logits corresponding to the target class
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))

    # log-sum-exp trick for numerical stability by subtracting the largest element
    logsumexp = torch.logsumexp(inputs, -1, keepdim=True)

    # Cancel out log and exp after softmax when calculating loss
    loss_matrix = -target_logits + logsumexp

    # Average loss
    loss = torch.mean(loss_matrix)
    return loss