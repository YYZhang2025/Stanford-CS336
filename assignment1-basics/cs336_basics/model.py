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
        self.ffn = FFN(
            d_model=config.d_model,
            d_ff=config.d_ff,
        )
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.mha(self.norm1(x), token_positions=token_positions)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.output_linear = Linear(config.d_model, config.vocab_size)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embedding(x)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.final_norm(x)
        logits = self.output_linear(x)
        return logits

    @torch.no_grad()
    def _generate_core(self):
        self.eval()
        pass

    def generate(self, x: torch.Tensor, max_length: int) -> torch.Tensor:
        pass

    def generate_streaming(self, x: torch.Tensor, max_length: int) -> torch.Tensor:
        pass
