from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, cast, overload

import torch
from torch import Tensor, nn

from ternary_pretrain.optim._matrix import MatrixOptimizer
from ternary_pretrain.optim.polar import polar_express

_MINIMUM_ROW_MAGNITUDE = 1e-7
_ZERO_ROW_MAGNITUDE = 0.33**0.5


def angular_lr_multiplier(
    step: int, *, warmup_steps: int, decay_scale: float, decay_degree: float
) -> float:
    steps_after_warmup = max(0, step - warmup_steps)
    if decay_degree == 0:
        return 1.0
    return float((1 + decay_scale * steps_after_warmup) ** (-decay_degree))


class AngularMuown(MatrixOptimizer):
    """Optimize row magnitudes and normalized directions independently."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        momentum: float,
        newton_schulz_steps: int,
        betas: tuple[float, float],
        eps: float,
        warmup_steps: int,
        decay_scale: float,
        decay_degree: float,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("AngularMuown warmup cannot be negative")
        if not math.isfinite(decay_scale) or decay_scale <= 0:
            raise ValueError("AngularMuown decay scale must be positive")
        if not math.isfinite(decay_degree) or decay_degree < 0:
            raise ValueError("AngularMuown decay degree cannot be negative")
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "newton_schulz_steps": newton_schulz_steps,
                "betas": betas,
                "eps": eps,
                "warmup_steps": warmup_steps,
                "decay_scale": decay_scale,
                "decay_degree": decay_degree,
                "angular_step": 0,
            },
        )

    def _initialize_state(self, parameter: nn.Parameter) -> dict[str, Any]:
        state = self.state[parameter]
        row_magnitude = parameter.float().norm(dim=1, keepdim=True)
        row_magnitude = torch.where(
            row_magnitude <= _MINIMUM_ROW_MAGNITUDE,
            row_magnitude.new_full(row_magnitude.shape, _ZERO_ROW_MAGNITUDE),
            row_magnitude,
        )
        state["row_magnitude"] = row_magnitude.clone()
        state["direction_momentum"] = torch.zeros_like(parameter, dtype=torch.float32)
        state["row_exp_avg"] = torch.zeros_like(row_magnitude)
        state["row_exp_avg_sq"] = torch.zeros_like(row_magnitude)
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
        row_magnitude = state["row_magnitude"]
        safe_magnitude = torch.copysign(
            row_magnitude.abs().clamp_min(_MINIMUM_ROW_MAGNITUDE), row_magnitude
        )
        direction = parameter_fp32 / safe_magnitude
        radial_gradient = (gradient_fp32 * direction).sum(dim=1, keepdim=True)
        direction_gradient = row_magnitude * (gradient_fp32 - direction * radial_gradient)

        momentum = parameter_group["momentum"]
        direction_momentum = state["direction_momentum"]
        direction_momentum.mul_(momentum).add_(direction_gradient)
        nesterov_gradient = direction_gradient.add(direction_momentum, alpha=momentum)
        orthogonal_update = polar_express(nesterov_gradient, parameter_group["newton_schulz_steps"])
        direction_scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
        multiplier = angular_lr_multiplier(
            parameter_group["angular_step"],
            warmup_steps=parameter_group["warmup_steps"],
            decay_scale=parameter_group["decay_scale"],
            decay_degree=parameter_group["decay_degree"],
        )
        next_direction = direction.add(
            orthogonal_update,
            alpha=-parameter_group["lr"] * direction_scale * multiplier,
        )

        beta1, beta2 = parameter_group["betas"]
        first_moment = state["row_exp_avg"]
        second_moment = state["row_exp_avg_sq"]
        first_moment.mul_(beta1).add_(radial_gradient, alpha=1 - beta1)
        second_moment.mul_(beta2).addcmul_(radial_gradient, radial_gradient, value=1 - beta2)
        corrected_first = first_moment / (1 - beta1**step)
        corrected_second = second_moment / (1 - beta2**step)
        row_update = corrected_first / (corrected_second.sqrt() + parameter_group["eps"])
        candidate = row_magnitude - parameter_group["lr"] * row_update
        row_magnitude.copy_(
            torch.copysign(candidate.abs().clamp_min(_MINIMUM_ROW_MAGNITUDE), candidate)
        )

        direction_norm = next_direction.norm(dim=1, keepdim=True).clamp_min(_MINIMUM_ROW_MAGNITUDE)
        parameter.copy_((row_magnitude * next_direction / direction_norm).to(parameter.dtype))

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = super().step(closure)
        for parameter_group in self.param_groups:
            parameter_group["angular_step"] += 1
        return loss

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        parameter_group = self.param_groups[0]
        last_step = max(0, int(parameter_group["angular_step"]) - 1)
        metrics = {
            "optimizer/angular_lr_multiplier": angular_lr_multiplier(
                last_step,
                warmup_steps=parameter_group["warmup_steps"],
                decay_scale=parameter_group["decay_scale"],
                decay_degree=parameter_group["decay_degree"],
            )
        }
        magnitudes = [
            state["row_magnitude"].abs().reshape(-1)
            for state in self.state.values()
            if "row_magnitude" in state
        ]
        if magnitudes:
            values = torch.cat(magnitudes)
            metrics["optimizer/row_magnitude_mean"] = float(values.mean())
            metrics["optimizer/row_magnitude_max"] = float(values.max())
        return metrics
