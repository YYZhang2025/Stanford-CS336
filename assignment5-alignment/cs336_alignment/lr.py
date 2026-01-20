import math

import torch


def adjust_learning_rate(cur_step: int, max_lr: float, max_steps: int):
    min_lr = max_lr * 0.1
    if cur_step >= max_steps:
        return min_lr
    # pure cosine decay
    decay_ratio = cur_step / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def update_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    max_lr: float,
    total_steps: int,
):
    lr = adjust_learning_rate(
        cur_step=step,
        max_lr=max_lr,
        max_steps=total_steps,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_lr(
    optimizer: torch.optim.Optimizer,
) -> float:
    return optimizer.param_groups[0]["lr"]
