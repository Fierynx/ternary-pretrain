from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor, nn

from ternary_pretrain.optim._matrix import MatrixOptimizer
from ternary_pretrain.optim.polar import muown_newton_schulz


class Muown(MatrixOptimizer):
    """Apply Muon to implicit directions and Adam to their row magnitudes."""

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

    def _initialize_state(self, parameter: nn.Parameter) -> dict[str, Any]:
        state = self.state[parameter]
        row_norm = parameter.float().norm(dim=1, keepdim=True)
        state["row_magnitude"] = row_norm.clone()
        state["direction_norm"] = row_norm.clone()
        state["direction_momentum"] = torch.zeros_like(parameter, dtype=torch.float32)
        state["row_exp_avg"] = torch.zeros_like(row_norm)
        state["row_exp_avg_sq"] = torch.zeros_like(row_norm)
        state["step"] = 0
        return cast(dict[str, Any], state)

    def _update_parameter(
        self, parameter: nn.Parameter, gradient: Tensor, parameter_group: dict[str, Any]
    ) -> None:
        parameter_fp32 = parameter.float()
        gradient_fp32 = gradient.float()
        state = self.state[parameter]
        if not state:
            state = self._initialize_state(parameter)

        state["step"] += 1
        step = state["step"]
        eps = parameter_group["eps"]
        row_magnitude = state["row_magnitude"]
        direction_norm = state["direction_norm"]
        safe_magnitude = torch.copysign(row_magnitude.abs().clamp_min(eps), row_magnitude)
        safe_direction_norm = direction_norm.clamp_min(eps)

        unit_direction = parameter_fp32 / safe_magnitude
        direction = unit_direction * safe_direction_norm
        radial_gradient = (gradient_fp32 * unit_direction).sum(dim=1, keepdim=True)
        direction_gradient = (row_magnitude / safe_direction_norm) * (
            gradient_fp32 - unit_direction * radial_gradient
        )

        momentum = parameter_group["momentum"]
        direction_momentum = state["direction_momentum"]
        direction_momentum.mul_(momentum).add_(direction_gradient)
        nesterov_gradient = direction_gradient.add(direction_momentum, alpha=momentum)
        orthogonal_update = muown_newton_schulz(
            nesterov_gradient, parameter_group["newton_schulz_steps"]
        )
        direction_scale = 0.2 * max(parameter.shape) ** 0.5
        next_direction = direction.add(
            orthogonal_update, alpha=-parameter_group["lr"] * direction_scale
        )

        beta1, beta2 = parameter_group["betas"]
        first_moment = state["row_exp_avg"]
        second_moment = state["row_exp_avg_sq"]
        first_moment.mul_(beta1).add_(radial_gradient, alpha=1 - beta1)
        second_moment.mul_(beta2).addcmul_(radial_gradient, radial_gradient, value=1 - beta2)
        corrected_first = first_moment / (1 - beta1**step)
        corrected_second = second_moment / (1 - beta2**step)
        row_magnitude.addcdiv_(
            corrected_first,
            corrected_second.sqrt().add_(eps),
            value=-parameter_group["lr"],
        )

        next_direction_norm = next_direction.norm(dim=1, keepdim=True).clamp_min(eps)
        parameter.copy_((row_magnitude * next_direction / next_direction_norm).to(parameter.dtype))
        state["direction_norm"] = next_direction_norm

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        magnitudes = [
            state["row_magnitude"].abs().reshape(-1)
            for state in self.state.values()
            if "row_magnitude" in state
        ]
        if not magnitudes:
            return {}
        values = torch.cat(magnitudes)
        return {
            "optimizer/row_magnitude_mean": float(values.mean()),
            "optimizer/row_magnitude_max": float(values.max()),
        }
