from einops import rearrange
import numpy
import torch
import torch.nn.functional as F

from .adapters import (
    run_multihead_self_attention_with_rope,
    run_rope,
    run_silu,
    run_multihead_self_attention,
    run_swiglu,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
    run_linear, 
    run_embedding,
)


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0][f"layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        output
    )


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0][f"token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(
        output
    )


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    # reference_weights = torch.load(FIXTURES_PATH / "positionwise_feedforward_weights.pt")
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # torch.manual_seed(42)
    # Take the first batch item, so we test the 3D case
    # (input shape (batch_size, seq_len, d_k)) for scaled dot-product attention.
    # K = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_K.pt")[0]
    # Q = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_Q.pt")[0]
    # V = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_V.pt")[0]
    # mask = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_mask.pt")
    # expected_output = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_expected_output.pt")[0]
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    # numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    # reference_weights = torch.load(FIXTURES_PATH / "unbatched_multihead_self_attention_weights.pt")
    # expected_output = torch.load(FIXTURES_PATH / "unbatched_multihead_self_attention_expected_output.pt")
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    # numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6)
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_multihead_self_attention_with_rope(
    numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    # reference_weights = torch.load(FIXTURES_PATH / "unbatched_multihead_self_attention_weights.pt")
    # expected_output = torch.load(FIXTURES_PATH / "unbatched_multihead_self_attention_expected_output.pt")
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    # numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6)
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_transformer_lm(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    # reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    # in_indices = torch.load(FIXTURES_PATH / "in_indices.pt")
    # expected_output = torch.load(FIXTURES_PATH / "transformer_lm_expected_output.pt")
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    # numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-4)
    numpy_snapshot.assert_match(
        actual_output, 
        atol=1e-4,
        rtol=1e-2
    )


def test_transformer_lm_truncated_input(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    # reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    # in_indices_truncated = torch.load(FIXTURES_PATH / "in_indices_truncated.pt")
    # truncated_expected_output = torch.load(FIXTURES_PATH / "transformer_lm_truncated_expected_output.pt")
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
        in_indices=in_indices,
    )

    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta):
    # reference_weights = torch.load(FIXTURES_PATH / "transformer_block_weights.pt")
    # in_features = torch.load(FIXTURES_PATH / "in_features.pt")

    block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]
    # reference_weights = torch.load(FIXTURES_PATH / "rmsnorm_weights.pt")
    # in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    # expected_output = torch.load(FIXTURES_PATH / "rmsnorm_expected_output.pt")
    # actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_features)

    # in_features = torch.randn()

    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings)

    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
    output = run_rope(
        d_model, theta=theta, max_seq_len=n_queries, in_query_or_key=in_embeddings, token_positions=pos_ids
    )
    numpy_snapshot.assert_match(output, atol=1e-6)


def test_silu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6)
