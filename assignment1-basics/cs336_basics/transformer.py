import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        self._init_weights()

    def _init_weights(self):
        mean = 0.0
        std = np.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=mean,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        self._init_weights()

    def _init_weights(self):
        mean = 0.0
        std = 1.0
        nn.init.trunc_normal_(
            self.weight,
            mean=mean,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of shape (B, T) containing token indices.

        Returns:
            Tensor of shape (B, T, embedding_dim) containing embeddings.
        """
        return self.weight[token_ids]


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


def silu(in_features: torch.Tensor) -> torch.Tensor:
    return in_features * torch.sigmoid(in_features)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu
            w1 (Float[Tensor, "d_ff d_model"]): Stored weights for W1
            w2 (Float[Tensor, "d_model d_ff"]): Stored weights for W2
            w3 (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self.w2(silu(self.w1(x)) * self.w3(x))
        return output


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta (float): Theta value for the RoPE
            d_k (int): Dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device): | None = None Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.rotation_matrix_table = self.generate_rotation_matrix(
            theta, d_k, max_seq_len
        )
        self.register_buffer(
            "rotation_matrix", self.rotation_matrix_table, persistent=False
        )

    def generate_rotation_block(
        self, theta: float, block_index: int, seq_pos: int, d_k: int
    ) -> torch.Tensor:
        angle = torch.tensor(seq_pos / (theta ** (2 * block_index / d_k)))
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        r_matrix = torch.Tensor([[cos, -sin], [sin, cos]])
        return r_matrix

    def generate_rotation_matrix(self, theta: float, d_k: int, max_seq_len: int):
        """
        Generate the rotation matrix
        """
        rotation_matrix_table = torch.zeros(max_seq_len, d_k, d_k)
        for i in range(max_seq_len):
            blocks = [
                self.generate_rotation_block(theta, k, i, d_k) for k in range(d_k // 2)
            ]
            rotation_matrix_table[i, :, :] = torch.block_diag(*blocks)
        return rotation_matrix_table

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Run RoPE for a given input tensor.
        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor(Query or Key) to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        *prefix_dims, seq_len, d_k = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        rotation_matrix = self.rotation_matrix_table[
            token_positions
        ]  # (batch_size, seq_len, d_k, d_k)
        x_rotated = rotation_matrix @ x.unsqueeze(-1)
        x_rotated = x_rotated.squeeze(-1)
        return x_rotated


def softmax(
    in_features: torch.Tensor, dim: int = -1, dtype: torch.dtype | None = None
) -> torch.Tensor:
    if dtype is not None:
        in_features = in_features.to(dtype)
    max_in_features = torch.max(in_features, dim=dim, keepdim=True).values
    exp_in_features = torch.exp(
        in_features - max_in_features
    )  # Softmax stability trick
    sum_exp_in_features = torch.sum(exp_in_features, dim=dim, keepdim=True)
    softmax_out = exp_in_features / sum_exp_in_features
    return softmax_out


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)

    if mask is not None:
        scores.masked_fill_(mask == 0, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output


from einops import einsum, rearrange


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            use_rope (bool): Whether to use RoPE
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = (
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            if use_rope
            else None
        )
        self.token_positions = token_positions
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, in_features: torch.Tensor):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """
        seq_len = in_features.shape[-2]
        qkv_proj = torch.cat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]
        )
        qkv = in_features @ qkv_proj.T
        q, k, v = qkv.chunk(3, -1)

        q = rearrange(
            q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)

        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :]
        output = scaled_dot_product_attention(q, k, v, ~casual_mask)
        output = rearrange(output, "... h seq_len d_head ->  ... seq_len (h d_head)")
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            d_ff (int): Dimensionality of the feedforward layer.
            use_rope (bool): Whether to use RoPE.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
        """
        super().__init__()
        self.mha = MultiHeadSelfAttention(
            d_model, num_heads, use_rope, max_seq_len, theta
        )
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            d_ff (int): Dimensionality of the feedforward layer.
            num_layers (int): Number of transformer layers.
            use_rope (bool): Whether to use RoPE.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
        """
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size,  # Example size, adjust as needed
            embedding_dim=d_model,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    use_rope=True,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )

        self.rms_norm = RMSNorm(d_model=d_model)
        self.output_embeddings = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.rms_norm(x)
        x = self.output_embeddings(x)

        return x
