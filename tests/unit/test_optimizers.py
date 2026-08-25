from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ternary_pretrain.config import ModelConfig, OptimizerConfig, OptimizerKind, ScheduleConfig
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.optim import build_optimizer
from ternary_pretrain.optim.angular_muown import AngularMuown, angular_lr_multiplier
from ternary_pretrain.optim.muon import Muon
from ternary_pretrain.optim.muown import Muown
from ternary_pretrain.optim.polar import (
    newton_schulz_polar,
    polar_express,
    row_normalize,
    tangent_projection,
)


def optimizer_config(kind: OptimizerKind) -> OptimizerConfig:
    return OptimizerConfig(
        kind=kind,
        learning_rate=0.02,
        auxiliary_learning_rate=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        momentum=0.9,
        newton_schulz_steps=5,
    )


def optimizer_schedule() -> ScheduleConfig:
    return ScheduleConfig(2, 6, 2, 0.1)


def test_polar_and_oblique_fixed_vectors() -> None:
    matrix = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    polar = newton_schulz_polar(matrix, steps=10)
    expected_polar = torch.tensor([[0.78125, 0.0], [0.0, 1.140625]]).bfloat16()
    torch.testing.assert_close(polar, expected_polar, atol=0, rtol=0)
    expected_polar_express = torch.tensor([[1.1328125, 0.0], [0.0, 0.89453125]]).bfloat16()
    torch.testing.assert_close(polar_express(matrix), expected_polar_express, atol=0, rtol=0)
    direction = row_normalize(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    torch.testing.assert_close(direction.norm(dim=1), torch.ones(2))
    projected = tangent_projection(direction, torch.ones_like(direction))
    torch.testing.assert_close(
        (projected * direction).sum(dim=1), torch.zeros(2), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("mode", ["muon", "muown", "angular_muown"])
def test_matrix_optimizer_reduces_same_toy_loss_and_serializes(mode: OptimizerKind) -> None:
    parameter = nn.Parameter(torch.tensor([[1.0, -0.5], [0.4, -1.2]]))
    optimizer = _matrix_optimizer(mode, parameter)
    initial = float(parameter.detach().square().sum())
    for _ in range(4):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    assert float(parameter.detach().square().sum()) < initial
    state = copy.deepcopy(optimizer.state_dict())
    clone = nn.Parameter(parameter.detach().clone())
    restored = _matrix_optimizer(mode, clone)
    restored.load_state_dict(state)
    assert restored.state_dict()["state"]
    if mode == "angular_muown":
        assert restored.param_groups[0]["angular_step"] == 4


@pytest.mark.parametrize("kind", ["adamw", "muon", "muown", "angular_muown"])
def test_parameter_partition_is_complete_disjoint_and_auditable(kind: OptimizerKind) -> None:
    model = DecoderLM(ModelConfig(64, 16, 1, 2, 1, 32, 16))
    optimizer, partition = build_optimizer(model, optimizer_config(kind), optimizer_schedule())
    matrix_ids = {id(parameter) for _, parameter in partition.matrices}
    auxiliary_ids = {id(parameter) for _, parameter in partition.auxiliary}
    assert matrix_ids.isdisjoint(auxiliary_ids)
    assert matrix_ids | auxiliary_ids == {id(parameter) for parameter in model.parameters()}
    assert optimizer.state_dict().keys() == optimizer.optimizers.keys()
    if kind != "adamw":
        assert all(
            group.get("weight_decay", 0.0) == 0.0
            for group in optimizer.optimizers["matrix"].param_groups
        )


@pytest.mark.parametrize(
    ("kind", "expected"),
    # Reference commits: Muon f98f1ca, Muown 3bd0c05, AngularMuown b9050b6.
    [
        ("muon", [[0.98417968, -0.46562499], [0.36015627, -1.21152353]]),
        ("muown", [[0.95899427, -0.47013471], [0.37479380, -1.15565515]]),
        ("angular_muown", [[0.97617066, -0.43334460], [0.32976168, -1.16930163]]),
    ],
)
def test_matrix_optimizer_matches_reference_step(
    kind: OptimizerKind, expected: list[list[float]]
) -> None:
    parameter = nn.Parameter(torch.tensor([[1.0, -0.5], [0.4, -1.2]]))
    parameter.grad = torch.tensor([[0.2, -0.3], [0.5, 0.1]])
    optimizer = _matrix_optimizer(kind, parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor(expected), atol=1e-7, rtol=1e-7)


def test_angular_multiplier_starts_after_warmup() -> None:
    values = [
        angular_lr_multiplier(step, warmup_steps=2, decay_scale=0.001, decay_degree=1.0)
        for step in range(4)
    ]
    assert values[:3] == [1.0, 1.0, 1.0]
    assert values[3] == pytest.approx(1 / 1.001)


def _matrix_optimizer(kind: OptimizerKind, parameter: nn.Parameter) -> Muon | Muown | AngularMuown:
    common = {"lr": 0.05, "momentum": 0.5, "newton_schulz_steps": 5}
    if kind == "muon":
        return Muon([parameter], **common)
    if kind == "muown":
        return Muown([parameter], betas=(0.9, 0.99), eps=1e-8, **common)
    if kind == "angular_muown":
        return AngularMuown(
            [parameter],
            betas=(0.9, 0.99),
            eps=1e-8,
            warmup_steps=0,
            decay_scale=0.001,
            decay_degree=1.0,
            **common,
        )
    raise AssertionError(f"unexpected matrix optimizer: {kind}")
