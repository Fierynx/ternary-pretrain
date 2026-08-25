from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor, nn

from ternary_pretrain.optim._matrix import MatrixOptimizer
from ternary_pretrain.optim.polar import newton_schulz_polar, row_normalize


class Muown(MatrixOptimizer):
    """Combine Muon direction updates with Adam-style row-magnitude updates."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        momentum: float,
        newton_schulz_steps: int,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "newton_schulz_steps": newton_schulz_steps,
                "betas": betas,
                "eps": eps,
            },
        )

    def _direction_gradient(
        self, direction: Tensor, row_magnitude: Tensor, gradient: Tensor
    ) -> Tensor:
        del direction, row_magnitude
        return gradient

    def _update_row_magnitude(
        self,
        parameter: nn.Parameter,
        radial_gradient: Tensor,
        row_magnitude: Tensor,
        parameter_group: dict[str, Any],
    ) -> Tensor:
        state = self.state[parameter]
        state["step"] = int(state.get("step", 0)) + 1
        first_moment = state.setdefault("row_exp_avg", torch.zeros_like(radial_gradient))
        second_moment = state.setdefault("row_exp_avg_sq", torch.zeros_like(radial_gradient))
        beta1, beta2 = parameter_group["betas"]
        first_moment.mul_(beta1).add_(radial_gradient, alpha=1 - beta1)
        second_moment.mul_(beta2).addcmul_(radial_gradient, radial_gradient, value=1 - beta2)
        step = state["step"]
        corrected_first_moment = first_moment / (1 - beta1**step)
        corrected_second_moment = second_moment / (1 - beta2**step)
        update = corrected_first_moment / (corrected_second_moment.sqrt() + parameter_group["eps"])
        return cast(
            Tensor,
            (row_magnitude - parameter_group["lr"] * update).clamp_min(parameter_group["eps"]),
        )

    def _update_parameter(
        self, parameter: nn.Parameter, gradient: Tensor, parameter_group: dict[str, Any]
    ) -> None:
        parameter_fp32 = parameter.float()
        gradient_fp32 = gradient.float()
        # Split each row into its length and direction.
        direction = row_normalize(parameter_fp32)
        row_magnitude = parameter_fp32.norm(dim=1, keepdim=True).clamp_min(parameter_group["eps"])
        radial_gradient = (gradient_fp32 * direction).sum(dim=1, keepdim=True)
        next_row_magnitude = self._update_row_magnitude(
            parameter, radial_gradient, row_magnitude, parameter_group
        )

        direction_gradient = self._direction_gradient(direction, row_magnitude, gradient_fp32)
        state = self.state[parameter]
        direction_momentum = state.setdefault(
            "direction_momentum", torch.zeros_like(direction_gradient)
        )
        direction_momentum.mul_(parameter_group["momentum"]).add_(direction_gradient)
        nesterov_gradient = direction_gradient.add(
            direction_momentum, alpha=parameter_group["momentum"]
        )
        orthogonal_update = newton_schulz_polar(
            nesterov_gradient, parameter_group["newton_schulz_steps"]
        )
        aspect_ratio_scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
        next_direction = row_normalize(
            direction - parameter_group["lr"] * aspect_ratio_scale * orthogonal_update
        )
        # Normalize directions before restoring row lengths.
        parameter.copy_((next_row_magnitude * next_direction).to(parameter.dtype))
        state["row_magnitude"] = next_row_magnitude.clone()
