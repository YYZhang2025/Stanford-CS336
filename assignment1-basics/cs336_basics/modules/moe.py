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
        if self.top_k == 1:
            topk_gates = router_probs.gather(-1, topk_indices)
        else:
            topk_gates = torch.softmax(topk_logits, dim=-1)  # (batch_size, seq_len, top_k)
        lb_loss = self._load_balance_loss(router_probs, topk_indices, self.num_experts)

        x_flat = x.reshape(batch_size * seq_len, d_model)  # (N, D)
        out_flat = x_flat.new_zeros((batch_size * seq_len, d_model))  # (N, D)

        if self.top_k == 1:
            expert_ids = topk_indices.reshape(batch_size * seq_len)  # (N,)
            gate_flat = topk_gates.reshape(batch_size * seq_len)  # (N,)

            for e in range(self.num_experts):
                pos = (expert_ids == e).nonzero(as_tuple=False).squeeze(1)  # token positions for expert e
                if pos.numel() == 0:
                    continue

                x_e = x_flat.index_select(0, pos)  # (n_e, D)
                y_e = self.experts[e](x_e)  # (n_e, D)
                y_e = y_e * gate_flat.index_select(0, pos).unsqueeze(1)  # gate

                out_flat.index_add_(0, pos, y_e)  # scatter back

            # tokens per expert（fraction）
            counts = torch.bincount(expert_ids, minlength=self.num_experts).to(x.dtype)
            tokens_per_expert = counts / (batch_size * seq_len)

        else:
            # 把每个 token 复制 K 次：token_ids 对应 topk 的每一项
            token_ids = (
                torch.arange(batch_size * seq_len, device=x.device)
                .unsqueeze(1)
                .expand(batch_size * seq_len, self.top_k)
                .reshape(-1)
            )  # (N*K,)
            expert_ids = topk_indices.reshape(-1)  # (N*K,)
            gate_flat = topk_gates.reshape(-1)  # (N*K,)

            for e in range(self.num_experts):
                sel = (expert_ids == e).nonzero(as_tuple=False).squeeze(1)  # positions in (N*K,)
                if sel.numel() == 0:
                    continue

                tok = token_ids.index_select(0, sel)  # token indices in [0..N)
                x_e = x_flat.index_select(0, tok)  # (n_e, D)
                y_e = self.experts[e](x_e)  # (n_e, D)
                y_e = y_e * gate_flat.index_select(0, sel).unsqueeze(1)

                out_flat.index_add_(0, tok, y_e)  # 同一个 token 可能被多个 topk 命中 -> index_add_ 自动累加

            counts = torch.bincount(expert_ids, minlength=self.num_experts).to(x.dtype)
            tokens_per_expert = counts / (batch_size * seq_len * self.top_k)

        expert_outputs = out_flat.view(batch_size, seq_len, d_model)

        return {
            "output": expert_outputs,
            "tokens_per_expert": tokens_per_expert,
            "z_loss": z_loss,
            "z_loss_scaled": z_loss * self.z_loss_coef,
            "lb_loss": lb_loss,
            "lb_loss_scaled": lb_loss * self.lb_loss_coef,
        }
