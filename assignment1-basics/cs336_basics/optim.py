import math
from typing import Iterable, Optional, Tuple

import torch


# Cosine annealing learning rate schedule with linear warmup (LLaMA-style)
def cosine_annealing_lr(
    t: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int,
) -> float:
    """Cosine annealing LR schedule with linear warmup (LLaMA-style).

    Args:
        t: current iteration (0-based).
        alpha_max: maximum learning rate (peak after warmup).
        alpha_min: minimum/final learning rate.
        Tw: number of warmup iterations.
        Tc: final iteration for cosine annealing (inclusive). After Tc, LR is alpha_min.

    Returns:
        alpha_t according to:
          - Warmup:        t < Tw  => (t / Tw) * alpha_max
          - Cosine:  Tw <= t <= Tc => alpha_min + 0.5*(1+cos((t-Tw)/(Tc-Tw)*pi))*(alpha_max-alpha_min)
          - Post:          t > Tc  => alpha_min

    Notes:
        Handles edge cases Tw==0 and/or Tc==Tw.
    """
    if Tw < 0 or Tc < 0:
        raise ValueError(f"Tw and Tc must be non-negative, got Tw={Tw}, Tc={Tc}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got t={t}")

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


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple of length 2, got: {betas}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2 value: {beta2}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

                # Bias correction
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                # Compute step size
                step_size = lr / bias_correction1

                # Denominator: sqrt(v_hat) + eps
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


import math
from typing import Iterable, Optional, Tuple

import torch


# Cosine annealing learning rate schedule with linear warmup (LLaMA-style)
def cosine_annealing_lr(
    t: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int,
) -> float:
    """Cosine annealing LR schedule with linear warmup (LLaMA-style).

    Args:
        t: current iteration (0-based).
        alpha_max: maximum learning rate (peak after warmup).
        alpha_min: minimum/final learning rate.
        Tw: number of warmup iterations.
        Tc: final iteration for cosine annealing (inclusive). After Tc, LR is alpha_min.

    Returns:
        alpha_t according to:
          - Warmup:        t < Tw  => (t / Tw) * alpha_max
          - Cosine:  Tw <= t <= Tc => alpha_min + 0.5*(1+cos((t-Tw)/(Tc-Tw)*pi))*(alpha_max-alpha_min)
          - Post:          t > Tc  => alpha_min

    Notes:
        Handles edge cases Tw==0 and/or Tc==Tw.
    """
    if Tw < 0 or Tc < 0:
        raise ValueError(f"Tw and Tc must be non-negative, got Tw={Tw}, Tc={Tc}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got t={t}")

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


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple of length 2, got: {betas}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2 value: {beta2}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

                # Bias correction
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                # Compute step size
                step_size = lr / bias_correction1

                # Denominator: sqrt(v_hat) + eps
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


@torch.no_grad()
def gradient_clip(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """Clips gradients of the given parameters to have a maximum L2 norm.

    Args:
        parameters: Iterable of model parameters to clip.
        max_l2_norm: Maximum allowed L2 norm for the gradients.

    Returns:
        None. The gradients are modified in place.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
