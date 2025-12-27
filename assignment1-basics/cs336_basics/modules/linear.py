import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self._init_weight()

    def forward(self, x):
        o = x @ self.weight

        if self.bias is not None:
            o = o + self.bias

        return o

    def _init_weight(self):
        mean = 0.0
        std = 1.0 / (2 * (self.in_features + self.out_features) ** 0.5)
        torch.nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3 * std, b=3 * std)
