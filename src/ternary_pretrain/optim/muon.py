from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from ternary_pretrain.optim._matrix import MatrixOptimizer
from ternary_pretrain.optim.polar import newton_schulz_polar


class Muon(MatrixOptimizer):
    """Apply Nesterov momentum followed by a Newton-Schulz polar update."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        momentum: float,
        newton_schulz_steps: int,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "newton_schulz_steps": newton_schulz_steps,
            },
        )

    def _update_parameter(
        self, parameter: nn.Parameter, gradient: Tensor, parameter_group: dict[str, Any]
    ) -> None:
        # Keep optimizer state in float32.
        gradient_fp32 = gradient.float()
        state = self.state[parameter]
        momentum_buffer = state.setdefault("momentum_buffer", torch.zeros_like(gradient_fp32))
        momentum = parameter_group["momentum"]
        momentum_buffer.lerp_(gradient_fp32, 1 - momentum)
        nesterov_gradient = gradient_fp32.lerp(momentum_buffer, momentum)
        orthogonal_update = newton_schulz_polar(
            nesterov_gradient, parameter_group["newton_schulz_steps"]
        )
        # Keep update sizes comparable across matrix shapes.
        aspect_ratio_scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
        orthogonal_update.mul_(aspect_ratio_scale)
        parameter.add_(
            orthogonal_update.to(parameter.dtype),
            alpha=-parameter_group["lr"],
        )
