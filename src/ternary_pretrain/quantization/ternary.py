from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def centered_scale(weight: Tensor) -> tuple[Tensor, Tensor]:
    # Use one scale for the whole matrix.
    centered = weight - weight.mean()
    minimum = torch.finfo(weight.dtype).eps
    scale = centered.abs().mean().clamp_min(minimum)
    return centered, scale


def ternary_codes(weight: Tensor) -> Tensor:
    centered, scale = centered_scale(weight)
    return torch.round(centered / scale).clamp_(-1, 1)


def ternary_weight(weight: Tensor) -> Tensor:
    centered, scale = centered_scale(weight)
    quantized = torch.round(centered / scale).clamp_(-1, 1) * scale
    # Use quantized weights in forward, but keep gradients on the full weights.
    return weight + (quantized - weight).detach()


@torch.no_grad()
def quantization_diagnostics(weights: Iterable[Tensor]) -> dict[str, float]:
    total_values = 0
    zero_values = 0
    squared_error = torch.tensor(0.0)
    squared_weight = torch.tensor(0.0)
    scales = []
    for weight in weights:
        centered, scale = centered_scale(weight.float())
        codes = torch.round(centered / scale).clamp_(-1, 1)
        quantized = codes * scale
        total_values += weight.numel()
        zero_values += int((codes == 0).sum())
        squared_error = (
            squared_error.to(weight.device) + (quantized - weight.float()).square().sum()
        )
        squared_weight = squared_weight.to(weight.device) + weight.float().square().sum()
        scales.append(scale)
    if total_values == 0:
        return {}
    mean_scale = torch.stack(scales).mean()
    relative_error = squared_error / squared_weight.clamp_min(torch.finfo(torch.float32).eps)
    return {
        "quantization/zero_fraction": zero_values / total_values,
        "quantization/relative_squared_error": float(relative_error),
        "quantization/mean_scale": float(mean_scale),
    }


class TernaryLinear(nn.Module):
    """Bias-free linear layer with an optional ternary QAT forward path."""

    in_features: int
    out_features: int
    _qat_enabled: Tensor

    def __init__(self, in_features: int, out_features: int, *, qat_enabled: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer(
            "_qat_enabled", torch.tensor(qat_enabled, dtype=torch.bool), persistent=True
        )
        self.reset_parameters()

    @property
    def qat_enabled(self) -> bool:
        return bool(self._qat_enabled.item())

    def set_qat_enabled(self, enabled: bool) -> None:
        self._qat_enabled.fill_(enabled)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        weight = ternary_weight(self.weight) if self.qat_enabled else self.weight
        return F.linear(inputs, weight)
