from __future__ import annotations

import torch

from ternary_pretrain.config import ModelConfig
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.quantization import (
    TernaryLinear,
    quantization_diagnostics,
    ternary_codes,
    ternary_weight,
)
from ternary_pretrain.quantization.ternary import centered_scale


def test_ternary_codes_centering_scaling_and_ste() -> None:
    weight = torch.tensor([[-2.0, -0.1, 0.4], [0.9, 1.3, 4.0]], requires_grad=True)
    codes = ternary_codes(weight)
    assert set(codes.tolist()[0] + codes.tolist()[1]) <= {-1.0, 0.0, 1.0}
    quantized = ternary_weight(weight)
    quantized.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))
    centered, scale = centered_scale(weight.detach())
    torch.testing.assert_close(centered.mean(), torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(scale, centered.abs().mean())


def test_qat_changes_only_hidden_matrix_forward_weights(small_model_config: ModelConfig) -> None:
    model = DecoderLM(small_model_config)
    embedding_identity = id(model.token_embedding.weight)
    latent = [parameter.detach().clone() for parameter in model.hidden_matrix_parameters()]
    inputs = torch.randint(0, small_model_config.vocab_size, (1, 5))
    float_logits = model(inputs).logits
    model.set_qat_enabled(True)
    qat_logits = model(inputs).logits
    assert id(model.token_embedding.weight) == embedding_identity
    for before, after in zip(latent, model.hidden_matrix_parameters(), strict=True):
        torch.testing.assert_close(before, after)
    assert not torch.equal(float_logits, qat_logits)


def test_ternary_linear_toggle_preserves_activations_as_float() -> None:
    layer = TernaryLinear(4, 3)
    inputs = torch.randn(2, 4, dtype=torch.float32)
    layer.set_qat_enabled(True)
    assert layer(inputs).dtype == torch.float32


def test_quantization_diagnostics_match_codes() -> None:
    weight = torch.tensor([[-2.0, -0.1, 0.4], [0.9, 1.3, 4.0]])
    metrics = quantization_diagnostics((weight,))
    codes = ternary_codes(weight)
    assert metrics["quantization/zero_fraction"] == float((codes == 0).sum()) / weight.numel()
    assert metrics["quantization/relative_squared_error"] >= 0
    assert metrics["quantization/mean_scale"] > 0
