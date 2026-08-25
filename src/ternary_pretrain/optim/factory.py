from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ternary_pretrain.config import OptimizerConfig, ScheduleConfig
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.optim.angular_muown import AngularMuown
from ternary_pretrain.optim.muon import Muon
from ternary_pretrain.optim.muown import Muown


@dataclass(frozen=True, slots=True)
class ParameterPartition:
    matrices: tuple[tuple[str, nn.Parameter], ...]
    auxiliary: tuple[tuple[str, nn.Parameter], ...]

    @classmethod
    def from_model(cls, model: DecoderLM) -> ParameterPartition:
        # Use object identity so tied weights are assigned only once.
        hidden_ids = {id(parameter) for parameter in model.hidden_matrix_parameters()}
        matrices = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if id(parameter) in hidden_ids and parameter.requires_grad
        )
        auxiliary = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if id(parameter) not in hidden_ids and parameter.requires_grad
        )
        owned = [id(parameter) for _, parameter in (*matrices, *auxiliary)]
        trainable = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
        if len(owned) != len(set(owned)) or set(owned) != set(trainable):
            raise RuntimeError("optimizer parameter groups are not complete and disjoint")
        if any(parameter.ndim != 2 for _, parameter in matrices):
            raise RuntimeError("matrix optimizer partition contains a non-matrix parameter")
        return cls(matrices=matrices, auxiliary=auxiliary)

    def audit(self) -> dict[str, list[str]]:
        return {
            "matrix_optimizer": [name for name, _ in self.matrices],
            "auxiliary_optimizer": [name for name, _ in self.auxiliary],
        }


class CompositeOptimizer:
    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("at least one optimizer is required")
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if set(state_dict) != set(self.optimizers):
            raise ValueError("composite optimizer members do not match checkpoint")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict[name])

    def diagnostics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for optimizer in self.optimizers.values():
            diagnostics = getattr(optimizer, "diagnostics", None)
            if diagnostics is not None:
                metrics.update(diagnostics())
        return metrics


def build_optimizer(
    model: DecoderLM, config: OptimizerConfig, schedule: ScheduleConfig | None = None
) -> tuple[CompositeOptimizer, ParameterPartition]:
    partition = ParameterPartition.from_model(model)
    if config.kind == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        return CompositeOptimizer({"adamw": optimizer}), partition
    # Hidden matrices use the selected optimizer; everything else uses AdamW.
    matrix_parameters = [parameter for _, parameter in partition.matrices]
    if config.kind == "muon":
        matrix: torch.optim.Optimizer = Muon(
            matrix_parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            newton_schulz_steps=config.newton_schulz_steps,
        )
    elif config.kind == "muown":
        matrix = Muown(
            matrix_parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            newton_schulz_steps=config.newton_schulz_steps,
            betas=config.betas,
            eps=config.eps,
        )
    else:
        if schedule is None:
            raise ValueError("AngularMuown requires the run schedule")
        matrix = AngularMuown(
            matrix_parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            newton_schulz_steps=config.newton_schulz_steps,
            betas=config.betas,
            eps=config.eps,
            warmup_steps=schedule.warmup_steps,
            decay_scale=config.angular_decay_scale,
            decay_degree=config.angular_decay_degree,
        )
    auxiliary = torch.optim.AdamW(
        (parameter for _, parameter in partition.auxiliary),
        lr=config.auxiliary_learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    return CompositeOptimizer({"matrix": matrix, "adamw": auxiliary}), partition
