import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape  # x: (B, L)
        out = x.reshape(-1)  # (B*L,)
        out = self.weight.index_select(0, out)  # (B*L, D)
        out = out.reshape(B, L, self.embedding_dim)  # (B, L, D)

        return out

    def _init_weight(self):
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)
