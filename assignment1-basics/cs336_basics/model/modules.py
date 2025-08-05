import torch 
import torch.nn as nn 

from cs336_basics.model.rope import RotaryPositionalEmbedding 


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device= None, dtype = None 
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(in_features, out_features, device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight
    

    
    

class Embedding(nn.Module):
    def __init__(self, num_embedding: int, embedding_dim: int, device = None, dtype = None):
        
        super().__init__()
        
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty(num_embedding, embedding_dim, device=device, dtype=dtype)
        )
        self._init_weight()
    
    def _init_weight(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """ 
        token_ids: (B, S) 
        return: (B, S, D) where D is the embedding dimension
        """
        
        return self.weight[token_ids]
        

# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None ):
#         super().__init__()
        
#         self.d_model = d_model
#         self.eps = eps
        
#         self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         dtype = x.dtype
#         x = x.to(torch.float32)
        
#         div_term = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
#         x = x / div_term
#         x = x * self.weight
        
#         return x.to(dtype)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()  # Ensure x is float for numerical stability
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normalized_x = x / norm

        return normalized_x * self.weight


def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Swish activation function.
    """
    return x * torch.sigmoid(x)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self.w2(SwiGLU(self.w1(x)) * self.w3(x))
        return output



def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    q: (B, S_q, D)
    k: (B, S_k, D)
    v: (B, S_v, D)
    mask: (B, S_q, S_k) or None
    """
    
    d_k = k.size(-1)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # (B, S_q, S_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = softmax(scores, dim=-1)  # (B, S_q, S_k)
    
    output = torch.matmul(attn_weights, v)  # (B, S_q, D)
    
    return output

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.head_dim = d_model // num_heads

        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions

        if use_rope and (max_seq_len is not None and theta is not None):
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        
        self.out_proj = Linear(d_model, d_model)
    
    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        mask =  mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, in_features: torch.Tensor):
        """
        in_features: (B, S, D)
        """
        B, S, D = in_features.size()
        
        q = self.q_proj(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)
        
        mask = self._causal_mask(S) 
        mask = mask.to(q.device)
        
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_output)
        return output
    
