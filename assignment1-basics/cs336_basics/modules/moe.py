import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.modules.linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.linear = Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits


class Expert(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.gate = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(silu(self.up(x)) * self.gate(x))


class MoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 1,
        router_jitter: float = 0.0,
        z_loss_coef: float = 1e-3,
        lb_loss_coef: float = 1e-1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.z_loss_coef = z_loss_coef
        self.lb_loss_coef = lb_loss_coef

    @staticmethod
    def _z_loss(logits: torch.Tensor) -> torch.Tensor:
        log_sum_exp = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(log_sum_exp**2)
        return z_loss

    @staticmethod
    def _load_balance_loss(
        router_probs: torch.Tensor,  # (B, S, E) softmax(logits)
        topk_indices: torch.Tensor,  # (B, S, K)
        num_experts: int,
    ) -> torch.Tensor:
        # p_i: mean probability per expert
        p = router_probs.mean(dim=(0, 1))  # (E,)

        # f_i: fraction of tokens dispatched to each expert (averaged over K)
        # one_hot: (B, S, K, E) -> mean over (B,S,K) => (E,)
        dispatch = F.one_hot(topk_indices, num_classes=num_experts).to(router_probs.dtype)
        f = dispatch.mean(dim=(0, 1, 2))  # (E,)

        # Switch auxiliary loss
        return num_experts * torch.sum(p * f)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, d_model = x.size()
        logits = self.router(x)  # (batch_size, seq_len, num_experts)

        if self.router_jitter > 0.0 and self.training:
            noise = torch.randn_like(logits) * self.router_jitter
            logits = logits + noise

        z_loss = self._z_loss(logits)
        router_probs = torch.softmax(logits, dim=-1)  # (B, S, E)

        topk_logits, topk_indices = torch.topk(
            logits, self.top_k, dim=-1
        )  # both (batch_size, seq_len, top_k)
        topk_gates = torch.softmax(topk_logits, dim=-1)  # (batch_size, seq_len, top_k)
        lb_loss = self._load_balance_loss(router_probs, topk_indices, self.num_experts)

        expert_outputs = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_index = topk_indices[:, :, k]  # (batch_size, seq_len)
            gate_values = topk_gates[:, :, k].unsqueeze(-1)  # (batch_size, seq_len, 1)

            for expert_id in range(self.num_experts):
                mask = (expert_index == expert_id).unsqueeze(-1)  # (batch_size, seq_len, 1)
                if mask.sum() == 0:
                    continue
                expert_input = x * mask.float()  # Zero out non-selected tokens
                expert_output = self.experts[expert_id](expert_input)  # (batch_size, seq_len, d_model)
                expert_outputs += expert_output * gate_values * mask.float()

        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for expert_id in range(self.num_experts):
            tokens_per_expert[expert_id] = (
                (topk_indices == expert_id).sum() / self.top_k / (batch_size * seq_len)
            )

        return {
            "output": expert_outputs,
            "tokens_per_expert": tokens_per_expert,
            "z_loss": z_loss,
            "z_loss_scaled": z_loss * self.z_loss_coef,
            "lb_loss": lb_loss,
            "lb_loss_scaled": lb_loss * self.lb_loss_coef,
        }
