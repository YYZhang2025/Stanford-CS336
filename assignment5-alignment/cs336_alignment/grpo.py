# New function for policy gradient loss dispatch
from typing import Callable, Literal

import einops
import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalized_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards = []
    for rollout_response, gt_response in zip(rollout_responses, repeated_ground_truths):
        curr_reward = reward_fn(rollout_response, gt_response)["reward"]
        raw_rewards.append(curr_reward)

    raw_rewards = torch.tensor(raw_rewards)  # prompts * group_size, 1
    rewards_per_group = raw_rewards.reshape((-1, group_size))  # prompts * group_size
    mean_reward_per_group = torch.mean(rewards_per_group, dim=-1, keepdim=True)  # prompts * 1
    advantage = rewards_per_group - mean_reward_per_group  # prompts * group_size

    if normalized_by_std:
        std_reward_per_group = torch.std(rewards_per_group, dim=-1, keepdim=True)
        advantage /= std_reward_per_group + advantage_eps  # prompts
    advantage = advantage.flatten()  # prompts * group_size, 1

    metadata = {
        "mean": torch.mean(raw_rewards).item(),
        "std": torch.std(raw_rewards).item(),
        "max": torch.max(raw_rewards).item(),
        "min": torch.min(raw_rewards).item(),
    }

    return advantage, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    _, seq_len = policy_log_probs.shape
    raw_rewards_or_advantages = einops.repeat(raw_rewards_or_advantages, "b 1->b s", s=seq_len)
    loss = -raw_rewards_or_advantages * policy_log_probs

    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor, policy_log_probs: torch.Tensor, old_log_probs: torch.Tensor, cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    _, seq_len = policy_log_probs.shape
    advantages = einops.repeat(advantages, "b 1 -> b s", s=seq_len)
    v = pi_ratio * advantages
    v_clip = torch.clip(pi_ratio, min=1 - cliprange, max=1 + cliprange) * advantages

    meta = {}
    return -torch.min(v, v_clip), meta


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the requested policy-gradient loss and collect metadata.

    Returns per-token loss with the same shape as `policy_log_probs` and a metadata dict.
    """
    assert loss_type in {"no_baseline", "reinforce_with_baseline", "grpo_clip"}, (
        f"Unknown loss_type: {loss_type}"
    )
    B, T = policy_log_probs.shape

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for no_baseline"
        assert raw_rewards.shape == (B, 1), (
            f"raw_rewards must have shape (B, 1); got {tuple(raw_rewards.shape)}"
        )
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata: dict[str, torch.Tensor] = {"mean_raw_reward": raw_rewards.mean()}
        return loss, metadata

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for reinforce_with_baseline"
        assert advantages.shape == (B, 1), f"advantages must have shape (B, 1); got {tuple(advantages.shape)}"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"mean_advantage": advantages.mean()}
        return loss, metadata

    # GRPO-Clip
    assert advantages is not None, "advantages must be provided for grpo_clip"
    assert old_log_probs is not None, "old_log_probs must be provided for grpo_clip"
    assert cliprange is not None, "cliprange must be provided for grpo_clip"
    assert advantages.shape == (B, 1), f"advantages must have shape (B, 1); got {tuple(advantages.shape)}"
    assert old_log_probs.shape == (B, T), (
        f"old_log_probs must have shape (B, T); got {tuple(old_log_probs.shape)}"
    )
    assert cliprange >= 0.0, "cliprange should be non-negative"

    loss, meta = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=float(cliprange),
    )

    # Add clip fraction statistic (fraction of tokens where clipping was active)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - float(cliprange), 1 + float(cliprange))
    was_clipped = (clipped_ratio != ratio).to(policy_log_probs.dtype)
    clip_fraction = masked_mean(was_clipped, torch.ones_like(policy_log_probs))

    meta_out = dict(meta)
    meta_out.update(
        {
            "mean_advantage": advantages.mean(),
            "clip_fraction": clip_fraction,
        }
    )
    return loss, meta_out


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[float, dict[str, torch.Tensor]]:
    # Compute loss and metadata
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss = masked_mean(loss, response_mask)
    loss = loss / gradient_accumulation_steps
    # Backpropagate the loss
    loss.backward()

    return loss.item(), metadata
