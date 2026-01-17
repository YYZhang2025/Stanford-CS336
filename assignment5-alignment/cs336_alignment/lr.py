import math

import torch

from cs336_alignment.config import TrainConfig


def cosine_annealing_lr(
    t: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int,
) -> float:
    # Warm-up
    if Tw > 0 and t < Tw:
        return (t / Tw) * alpha_max

    # Cosine annealing (including the exact boundary t==Tw)
    if t <= Tc:
        # If Tc == Tw, there is no annealing window; at t==Tw return alpha_max.
        if Tc == Tw:
            return alpha_max

        progress = (t - Tw) / (Tc - Tw)  # in [0, 1]
        return alpha_min + 0.5 * (1.0 + math.cos(math.pi * progress)) * (alpha_max - alpha_min)

    # Post-annealing
    return alpha_min


def update_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    train_config: TrainConfig,
):
    lr = cosine_annealing_lr(
        t=step,
        alpha_max=train_config.max_lr,
        alpha_min=train_config.min_lr,
        Tw=train_config.warmup_steps,
        Tc=train_config.total_training_steps,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_lr(
    optimizer: torch.optim.Optimizer,
) -> float:
    return optimizer.param_groups[0]["lr"]
