import torch 
import torch.nn as nn 

import einops

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: float, max_seq_len: int, device = None):
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        self.device = device if device else torch.device("cpu")
        
        
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        
        return einops.rearrange(x, "... d r -> ... (d r)")
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [..., seq_len, d_k]
        token_positions: tensor of shape [..., seq_len], typically just arange(seq_len)
        """
        seq_len = x.size(-2)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0).expand(x.size(0), -1)

        # (B, S), (d_k // 2) => (B, S, d_k // 2)
        # Equal to token_positions[..., :, None] * self.inv_freq[None, None, :]
        theta = torch.einsum("... n, d ->  ... n d", token_positions, self.inv_freq)
        
        # get cos and sine 
        cos = theta.cos().repeat_interleave(2, dim=-1)
        sin = theta.sin().repeat_interleave(2, dim=-1)
        
        x = x * cos + self._rotate_half(x) * sin

        return x
