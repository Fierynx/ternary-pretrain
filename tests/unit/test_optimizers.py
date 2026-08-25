from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ternary_pretrain.config import ModelConfig, OptimizerConfig, OptimizerKind
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.optim import build_optimizer
from ternary_pretrain.optim.angular_muown import AngularMuown
from ternary_pretrain.optim.muon import Muon
from ternary_pretrain.optim.muown import Muown
from ternary_pretrain.optim.polar import newton_schulz_polar, row_normalize, tangent_projection


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


def test_polar_and_oblique_fixed_vectors() -> None:
    matrix = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    polar = newton_schulz_polar(matrix, steps=10)
    expected_polar = torch.tensor([[0.734499, 0.0], [0.0, 1.134264]])
    torch.testing.assert_close(polar, expected_polar, atol=3e-5, rtol=3e-5)
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


@pytest.mark.parametrize("kind", ["adamw", "muon", "muown", "angular_muown"])
def test_parameter_partition_is_complete_disjoint_and_auditable(kind: OptimizerKind) -> None:
    model = DecoderLM(ModelConfig(64, 16, 1, 2, 1, 32, 16))
    optimizer, partition = build_optimizer(model, optimizer_config(kind))
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


def _matrix_optimizer(kind: OptimizerKind, parameter: nn.Parameter) -> Muon | Muown | AngularMuown:
    common = {"lr": 0.05, "momentum": 0.5, "newton_schulz_steps": 5}
    if kind == "muon":
        return Muon([parameter], **common)
    if kind == "muown":
        return Muown([parameter], betas=(0.9, 0.99), eps=1e-8, **common)
    if kind == "angular_muown":
        return AngularMuown([parameter], betas=(0.9, 0.99), eps=1e-8, **common)
    raise AssertionError(f"unexpected matrix optimizer: {kind}")
