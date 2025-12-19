import torch
import torch.nn as nn

from cs336_basics.config import ModelConfig


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def _generate_core(self):
        self.eval()
        pass

    def generate(self, x: torch.Tensor, max_length: int) -> torch.Tensor:
        pass

    def generate_streaming(self, x: torch.Tensor, max_length: int) -> torch.Tensor:
        pass
