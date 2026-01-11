import torch
import torch.nn as nn

from cs336_basics.config import ModelConfig
from cs336_basics.modules import FFN, MHA, Linear, RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.mha = MHA(
            d_model=config.d_model,
            num_heads=config.num_heads,
            use_rope=config.use_rope,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
        )
        self.use_moe = config.use_moe
        if self.use_moe:
            from cs336_basics.modules import MoE

            self.ffn = MoE(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                top_k=config.top_k,
                router_jitter=config.router_jitter,
                z_loss_coef=config.z_loss_coef,
                lb_loss_coef=config.lb_loss_coef,
            )
        else:
            self.ffn = FFN(
                d_model=config.d_model,
                d_ff=config.d_ff,
            )
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor | tuple:
        aux = {
            "z_loss": x.new_zeros(()),
            "z_loss_scaled": x.new_zeros(()),
            "tokens_per_expert": None,  # 可选：debug/monitor
            "lb_loss": x.new_zeros(()),  # 可选：debug/monitor
            "lb_loss_scaled": x.new_zeros(()),  # 可选：debug/monitor
        }

        x = x + self.mha(self.norm1(x), token_positions=token_positions)

        if self.use_moe:
            out = self.ffn(self.norm2(x))
            x = x + out["output"]

            aux["tokens_per_expert"] = out.get("tokens_per_expert", None)
            aux["z_loss"] = out.get("z_loss", x.new_zeros(()))
            aux["z_loss_scaled"] = out.get("z_loss_scaled", x.new_zeros(()))
            aux["lb_loss"] = out.get("lb_loss", x.new_zeros(()))
            aux["lb_loss_scaled"] = out.get("lb_loss_scaled", x.new_zeros(()))
        else:
            x = x + self.ffn(self.norm2(x))

        return x, aux


class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size, use_norm: bool = False):
        super().__init__()
        self.linear = Linear(d_model, vocab_size)
        self.norm = RMSNorm(d_model) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.linear(x)
        return logits


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.output_layer = OutputLayer(config.d_model, config.vocab_size, use_norm=config.use_final_norm)

        if config.tie_weights:
            self._tie_weights()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> tuple:
        x = self.token_embedding(x)
        total_z_scaled = x.new_zeros(())
        tokens_per_expert_all = []  # list[Tensor] or []
        total_lb_loss_scaled = x.new_zeros(())
        moe_layers = 0

        for layer in self.layers:
            x, aux = layer(x, token_positions=token_positions)
            if self.config.use_moe:
                total_z_scaled = total_z_scaled + aux["z_loss_scaled"]
                total_lb_loss_scaled = total_lb_loss_scaled + aux["lb_loss_scaled"]
                tokens_per_expert_all.append(aux["tokens_per_expert"])
                moe_layers += 1
        x = self.final_norm(x)
        logits = self.output_layer(x)

        aux_out = {
            "z_loss_scaled": total_z_scaled,
            "moe_layers": moe_layers,
            "tokens_per_expert": tokens_per_expert_all,  # list[Tensor] or []
            "lb_loss_scaled": total_lb_loss_scaled,
        }
        return logits, aux_out

    def _tie_weights(self):
        self.output_layer.linear.weight = self.token_embedding.weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def _generate_core(self):
        self.eval()
        pass

    def generate(self, x: torch.Tensor, max_length: int) -> torch.Tensor:
        pass

    def generate_streaming(self, x: torch.Tensor, max_length: int) -> torch.Tensor:
        pass
