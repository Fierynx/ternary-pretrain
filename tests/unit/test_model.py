from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from ternary_pretrain.config import ModelConfig
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.model.layers import Attention, apply_rope, rope_frequencies


def test_model_shapes_weight_tying_and_float32_loss(small_model_config: ModelConfig) -> None:
    model = DecoderLM(small_model_config)
    inputs = torch.randint(0, small_model_config.vocab_size, (2, 8))
    output = model(inputs, labels=inputs)
    assert output.logits.shape == (2, 8, small_model_config.vocab_size)
    assert output.loss is not None and output.loss.dtype == torch.float32
    assert model.lm_head.weight is model.token_embedding.weight


def test_causal_prefix_is_invariant_to_future_tokens(small_model_config: ModelConfig) -> None:
    torch.manual_seed(4)
    model = DecoderLM(small_model_config).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 9, 10]])
    with torch.no_grad():
        first_logits = model(first).logits[:, :3]
        second_logits = model(second).logits[:, :3]
    torch.testing.assert_close(first_logits, second_logits)


def test_sdpa_matches_reference_attention(small_model_config: ModelConfig) -> None:
    torch.manual_seed(5)
    attention = Attention(small_model_config).eval()
    inputs = torch.randn(2, 6, small_model_config.d_model)
    actual = attention(inputs)
    batch, sequence, _ = inputs.shape
    query = attention.q_proj(inputs).view(batch, sequence, 2, 8).transpose(1, 2)
    key = attention.k_proj(inputs).view(batch, sequence, 1, 8).transpose(1, 2)
    value = attention.v_proj(inputs).view(batch, sequence, 1, 8).transpose(1, 2)
    cosine, sine = rope_frequencies(sequence, 8, 10_000.0, device=inputs.device)
    query = apply_rope(query, cosine, sine)
    key = apply_rope(key, cosine, sine).repeat_interleave(2, dim=1)
    value = value.repeat_interleave(2, dim=1)
    mask = torch.triu(torch.full((sequence, sequence), -math.inf), diagonal=1)
    probabilities = F.softmax(query @ key.transpose(-1, -2) / math.sqrt(8) + mask, dim=-1)
    reference = attention.o_proj(
        (probabilities @ value).transpose(1, 2).contiguous().view(batch, sequence, -1)
    )
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_residual_projections_use_depth_scaled_initialization() -> None:
    config = ModelConfig(64, 128, 8, 4, 2, 256, 16, initializer_std=0.08)
    torch.manual_seed(8)
    model = DecoderLM(config)
    expected = 0.08 / math.sqrt(16)
    actual = float(model.layers[0].attention.o_proj.weight.detach().std())
    assert actual == pytest.approx(expected, rel=0.08)
