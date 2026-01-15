import math

import torch

# class FlashAttention(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
#         # Support q/k/v shaped (B, S, D) or (B, H, S, D)
#         had_heads = q.dim() == 4
#         if had_heads:
#             B, H, N_q, D = q.shape
#             _, _, N_k, _ = k.shape
#             # Fold heads into batch for the block-wise implementation
#             q = q.reshape(B * H, N_q, D)
#             k = k.reshape(B * H, N_k, D)
#             v = v.reshape(B * H, N_k, D)
#             B_eff = B * H
#         else:
#             B, N_q, D = q.shape
#             _, N_k, _ = k.shape
#             B_eff = B
#         device = q.device
#         scale = 1.0 / math.sqrt(D)

#         B_q, B_k = 64, 64  # block sizes
#         T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)  # number of blocks

#         # Accumulate in fp32 (matches FA-2 algorithm stability)
#         O = torch.empty((B_eff, N_q, D), device=device, dtype=torch.float32)
#         # Store per-token logsumexp (L_i = m_i + log(l_i))
#         L = torch.empty((B_eff, N_q), device=device, dtype=torch.float32)

#         # Precompute indices for causal masking once
#         q_pos = torch.arange(N_q, device=device)
#         k_pos = torch.arange(N_k, device=device)

#         # 📘 Outer loop over query blocks
#         for i in range(T_q):  # For each query block
#             q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
#             q_len = q_end - q_start

#             # 📘 Load Q Tile
#             Q_blk_i = q[:, q_start:q_end, :]  # (B_eff, q_len, D)

#             # 📘 Online softmax states and output accumulator
#             m_i = torch.full(
#                 (B_eff, q_len), -float("inf"), device=device, dtype=torch.float32
#             )  # (B_eff, q_len)
#             l_i = q.new_zeros((B_eff, q_len), dtype=torch.float32)  # (B_eff, q_len)
#             O_i = q.new_zeros((B_eff, q_len, D), dtype=torch.float32)  # (B_eff, q_len, D)

#             # For causal mask(this is same across all batches)
#             q_idx = q_pos[q_start:q_end]

#             # 📘  Inner loop over key, value blocks
#             for j in range(T_k):
#                 k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)

#                 if is_causal and k_start >= q_end:
#                     # All subsequent blocks will be masked
#                     break

#                 # 📘 Load K, V Tiles
#                 k_blk_j = k[:, k_start:k_end, :]  # (B_eff, k_len, D)
#                 v_blk_j = v[:, k_start:k_end, :]  # (B_eff, k_len, D)

#                 # 📘 Compute tile of pre-softmax attention scores
#                 scores_ij = (
#                     Q_blk_i.float() @ k_blk_j.float().transpose(-1, -2)
#                 ) * scale  # (B_eff, q_len, k_len)
#                 if is_causal:
#                     k_idx = k_pos[k_start:k_end]
#                     causal_mask = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)  # (q_len, k_len)
#                     scores_ij = scores_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

#                 # 📘 Compute m_ij
#                 m_ij = torch.maximum(m_i, scores_ij.max(dim=-1).values)  # (B_eff, q_len)

#                 # 📘 Compute p~_ij = exp(scores_ij - m_ij)
#                 p_tilde_ij = torch.exp(scores_ij - m_ij.unsqueeze(-1))  # (B_eff, q_len, k_len)

#                 # 📘 Compute l_ij
#                 alpha = torch.exp(m_i - m_ij)  # (B_eff, q_len)
#                 l_ij = alpha * l_i + p_tilde_ij.sum(dim=-1).to(l_i.dtype)  # (B_eff, q_len)

#                 # 📘 Compute O_ij
#                 O_ij = alpha.unsqueeze(-1) * O_i + (p_tilde_ij @ v_blk_j.float())  # (B_eff, q_len, D)

#                 # 📘 Update online softmax states
#                 m_i = m_ij
#                 l_i = l_ij
#                 O_i = O_ij

#             # End of inner loop over key, value blocks
#             L[:, q_start:q_end] = m_i + torch.log(l_i)  # (B_eff, q_len)
#             O[:, q_start:q_end, :] = O_i / l_i.unsqueeze(-1)  # (B_eff, q_len, D)

#         # End of outer loop over query blocks
#         if had_heads:
#             O = O.reshape(B, H, N_q, D)
#             L = L.reshape(B, H, N_q)
#         ctx.save_for_backward(q, k, v, O, L)
#         ctx.is_causal = is_causal
#         ctx.had_heads = had_heads
#         if had_heads:
#             ctx.B, ctx.H = B, H

#         return O

#     @staticmethod
#     def backward(ctx, grad_O: torch.Tensor):
#         q, k, v, O, L = ctx.saved_tensors
#         is_causal = ctx.is_causal
#         had_heads = ctx.had_heads
#         if had_heads:
#             # q/k/v were saved in flattened form: (B_eff, S, D)
#             B, H = ctx.B, ctx.H
#             B_eff = B * H
#             N_q, D = q.shape[1], q.shape[2]
#             N_k = k.shape[1]

#             # grad_O arrives as (B, H, S, D); O as (B, H, S, D); L as (B, H, S)
#             grad_O = grad_O.reshape(B_eff, N_q, D)
#             O = O.reshape(B_eff, N_q, D)
#             L = L.reshape(B_eff, N_q)
#         else:
#             B_eff, N_q, D = q.shape
#             N_k = k.shape[1]

#         device = q.device
#         scale = 1.0 / math.sqrt(D)

#         B_q, B_k = 64, 64  # block sizes
#         T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)  # number of blocks

#         grad_q = torch.zeros_like(q)
#         grad_k = torch.zeros_like(k)
#         grad_v = torch.zeros_like(v)

#         # Precompute indices for causal masking once
#         q_pos = torch.arange(N_q, device=device)
#         k_pos = torch.arange(N_k, device=device)

#         D_blk = torch.sum(grad_O * O, dim=-1)  # (B_eff, N_q)

#         # Outer loop over query blocks
#         for i in range(T_q):  # For each query block
#             q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)

#             # Load Q Tile
#             Q_blk_i = q[:, q_start:q_end, :]  # (B_eff, q_len, D)
#             grad_O_blk_i = grad_O[:, q_start:q_end, :]  # (B_eff, q_len, D)
#             L_blk_i = L[:, q_start:q_end]  # (B_eff, q_len)
#             D_blk_i = D_blk[:, q_start:q_end]  # (B_eff, q_len)

#             # For causal mask(this is same across all batches)
#             q_idx = q_pos[q_start:q_end]

#             # Inner loop over key, value blocks
#             for j in range(T_k):
#                 k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)

#                 if is_causal and k_start >= q_end:
#                     # All subsequent blocks will be masked
#                     break

#                 # Load K, V Tiles
#                 k_blk_j = k[:, k_start:k_end, :]  # (B_eff, k_len, D)
#                 v_blk_j = v[:, k_start:k_end, :]  # (B_eff, k_len, D)

#                 # Re-Compute score tile for q_i, k_j
#                 scores_ij = (
#                     Q_blk_i.float() @ k_blk_j.float().transpose(-1, -2)
#                 ) * scale  # (B_eff, q_len, k_len)
#                 if is_causal:
#                     k_idx = k_pos[k_start:k_end]
#                     causal_mask = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)  # (q_len, k_len)
#                     scores_ij = scores_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

#                 # Softmax probabilities for this tile: P = exp(S - logsumexp)
#                 P_ij = torch.exp(scores_ij - L_blk_i.unsqueeze(-1))  # (B_eff, q_len, k_len)

#                 # dV += P^T @ dO
#                 grad_v[:, k_start:k_end, :] += torch.bmm(
#                     P_ij.transpose(1, 2),
#                     grad_O_blk_i.float(),
#                 ).to(grad_v.dtype)

#                 # dP = dO @ V^T
#                 grad_P = torch.bmm(
#                     grad_O_blk_i.float(),
#                     v_blk_j.float().transpose(1, 2),
#                 )  # (B_eff, q_len, k_len)

#                 grad_S = P_ij * (grad_P - D_blk_i.unsqueeze(-1))  # (B_eff, q_len, k_len)

#                 # dQ += dS @ K ;  dK += dS^T @ Q
#                 grad_q[:, q_start:q_end, :] += (torch.bmm(grad_S, k_blk_j.float()) * scale).to(grad_q.dtype)
#                 grad_k[:, k_start:k_end, :] += (
#                     torch.bmm(grad_S.transpose(1, 2), Q_blk_i.float()) * scale
#                 ).to(grad_k.dtype)
#             # End of inner loop over key, value blocks
#         # End of outer loop over query blocks

#         if had_heads:
#             grad_q = grad_q.reshape(B, H, N_q, D)
#             grad_k = grad_k.reshape(B, H, N_k, D)
#             grad_v = grad_v.reshape(B, H, N_k, D)
#         return grad_q, grad_k, grad_v, None


class PyTorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        had_heads = q.ndim == 4
        if had_heads:
            B, H, N_q, D = q.shape
            _, _, N_k, _ = k.shape
            # Fold heads into batch for the block-wise implementation
            q = q.reshape(B * H, N_q, D)
            k = k.reshape(B * H, N_k, D)
            v = v.reshape(B * H, N_k, D)
            B_eff = B * H
        else:
            B, N_q, D = q.shape
            _, N_k, _ = k.shape
            B_eff = B

        scale = 1.0 / math.sqrt(D)
        device = q.device
        q = q * scale

        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)

        # Flash Attention implementation
        O = torch.empty((B_eff, N_q, D), device=device, dtype=q.dtype)
        L = torch.empty((B_eff, N_q), device=device, dtype=q.dtype)

        for i in range(T_q):
            q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
            q_len = q_end - q_start

            # Load Q_i block
            Q_i = q[:, q_start:q_end, :]  # (B_eff, q_len, D)

            # Initialize accumulators for the output and log-sum-exp
            O_i = torch.zeros((B_eff, q_len, D), device=device, dtype=q.dtype)
            L_i = torch.zeros((B_eff, q_len), device=device, dtype=q.dtype)
            m_i = torch.full((B_eff, q_len), float("-inf"), device=device, dtype=q.dtype)

            # Inner loop over K, V blocks
            for t_k in range(T_k):
                # Load K_j, V_j block
                k_start, k_end = t_k * B_k, min((t_k + 1) * B_k, N_k)
                K_j = k[:, k_start:k_end, :]  # (B_eff, k_len, D)
                V_j = v[:, k_start:k_end, :]  # (B_eff, k_len, D)

                # Compute attention scores
                S_ij = torch.bmm(Q_i, K_j.transpose(1, 2))

                if is_causal:
                    k_idx = torch.arange(k_start, k_end, device=device)
                    q_idx = torch.arange(q_start, q_end, device=device)

                    causal_mask = (q_idx.unsqueeze(-1) >= k_idx.unsqueeze(0)).to(torch.bool)
                    S_ij = S_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                # Online softmax with numerical stability
                m_new = torch.maximum(m_i, torch.max(S_ij, dim=-1).values)
                P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))

                L_ij = torch.exp(m_i - m_new) * L_i + torch.sum(P_ij, dim=-1)
                O_ij = torch.exp(m_i - m_new).unsqueeze(-1) * O_i + torch.bmm(P_ij, V_j)

                m_i = m_new
                L_i = L_ij
                O_i = O_ij
            # End of inner loop over K, V blocks
            # Compute final output for Q_i block
            O[:, q_start:q_end, :] = O_i / L_i.unsqueeze(-1)
            L[:, q_start:q_end] = m_i + torch.log(L_i)

        # End of outer loop over Q blocks
        if had_heads:
            O = O.reshape(B, H, N_q, D)
            L = L.reshape(B, H, N_q)

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        ctx.had_heads = had_heads
        if had_heads:
            (
                ctx.B,
                ctx.H,
            ) = B, H
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        had_heads = ctx.had_heads
        if had_heads:
            B, H = ctx.B, ctx.H
            B_eff = int(B * H)
            N_q, d_k = q.shape[1], q.shape[2]
            N_k = k.shape[1]

            dO = dO.reshape(B_eff, N_q, d_k)
            O = O.reshape(B_eff, N_q, d_k)
            L = L.reshape(B_eff, N_q)
        else:
            B_eff, N_q, D = q.shape
            N_k = k.shape[1]

        scale = 1.0 / math.sqrt(d_k)

        device = q.device
        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)

        # step3: Initialize gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # step4: Compute D=rowsum(dO*O)
        D = torch.sum(dO * O, dim=-1)  # (B_eff, N_q)

        # step5: Outer loop over Key, Value blocks
        for j in range(T_k):
            # step6: Load K_j, V_j block
            k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
            K_j = k[:, k_start:k_end, :]  # (B_eff, k_len, D)
            V_j = v[:, k_start:k_end, :]  # (B_eff, k

            # step7: Initialize dK_j, dV_j
            dK_j = k.new_zeros((B_eff, int(k_end - k_start), d_k))
            dV_j = v.new_zeros((B_eff, int(k_end - k_start), d_k))

            # step8: Inner loop over Query blocks
            for i in range(T_q):
                # step9: Load Q_i, O_i, dO_i, L_i, D_i block
                q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
                Q_i = q[:, q_start:q_end, :]  # (B_eff, q_len, D)
                O_i = O[:, q_start:q_end, :]  # (B_eff, q_len, D)
                dO_i = dO[:, q_start:q_end, :]  # (B_eff, q_len, D)
                L_i = L[:, q_start:q_end]  # (B_eff, q_len)
                D_i = D[:, q_start:q_end]  # (B_eff, q_len)

                # step10: Compute S_ij
                S_ij = torch.bmm(Q_i, k[:, k_start:k_end, :].transpose(1, 2))  # (B_eff, q_len, k_len)

                # step10.1 : Apply causal mask if needed
                if is_causal:
                    k_idx = torch.arange(k_start, k_end, device=device)
                    q_idx = torch.arange(q_start, q_end, device=device)

                    causal_mask = (q_idx.unsqueeze(-1) >= k_idx.unsqueeze(0)).to(torch.bool)
                    S_ij = S_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                # step11: Compute P_ij
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # (B_eff, q_len, k_len)

                # step12: Update dV_j
                dV_j = dV_j + torch.bmm(P_ij.transpose(1, 2), dO_i)  # (B_eff, k_len, D)

                # step13: Compute dP_ij
                dP_ij = torch.bmm(dO_i, V_j.transpose(1, 2))  # (B_eff, q_len, k_len)

                # step14: Compute dS_ij
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # (B_eff, q_len, k_len)

                # step15: Load dQ_i, update dQ_i
                dq_i = dq[:, q_start:q_end, :]  # (B_eff, q_len, D)
                dq_i = dq_i + torch.bmm(dS_ij, k[:, k_start:k_end, :])  # (B_eff, q_len, D)
                dq[:, q_start:q_end, :] = dq_i

                # step16: Update dK_j
                dK_j = dK_j + torch.bmm(dS_ij.transpose(1, 2), Q_i)  # (B_eff, k_len, D)

            # End of inner loop over Query blocks

            # step17: Store dK_j, dV_j
            dk[:, k_start:k_end, :] = dk[:, k_start:k_end, :] + dK_j
            dv[:, k_start:k_end, :] = dv[:, k_start:k_end, :] + dV_j

        # End of outer loop over Key, Value blocks

        if had_heads:
            dq = dq.reshape(B, H, N_q, d_k)
            dk = dk.reshape(B, H, N_k, d_k)
            dv = dv.reshape(B, H, N_k, d_k)

        return dq * scale, dk, dv, None
