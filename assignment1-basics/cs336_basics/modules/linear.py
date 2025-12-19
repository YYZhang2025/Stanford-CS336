import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

    def forward(self, x):
        o = x @ self.weight
        if self.bias is not None:
            o = o + self.bias
        return o
