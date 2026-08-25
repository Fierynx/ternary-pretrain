from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

_POLAR_EXPRESS_COEFFICIENTS = (
    (8.237312490495555, -23.157747414558198, 16.680568411445915),
    (4.082441999064836, -2.8930477353325887, 0.5252849256975651),
    (3.9263479922546556, -2.8547468034765293, 0.5318022422894989),
    (3.2982187133085143, -2.4245419810267062, 0.48632008358844075),
    (2.320007312889811, -1.6862169729967622, 0.42068027340235137),
)


def _working_copy(matrix: Tensor) -> Tensor:
    return matrix.bfloat16()


def newton_schulz_polar(matrix: Tensor, steps: int = 5) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("polar update requires a matrix")
    if steps < 0:
        raise ValueError("polar step count cannot be negative")
    work = _working_copy(matrix)
    # Work on the smaller side of tall matrices.
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    work = work / (work.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Coefficients used by the quintic Newton-Schulz update.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = work @ work.T
        polynomial = b * gram + (c * gram) @ gram
        work = a * work + polynomial @ work
    if transposed:
        work = work.T
    return cast(Tensor, work)


def muown_newton_schulz(matrix: Tensor, steps: int = 5) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("Muown polar update requires a matrix")
    if steps < 0:
        raise ValueError("Muown polar step count cannot be negative")
    original_dtype = matrix.dtype
    work = _working_copy(matrix) / (matrix.norm() + 1e-7)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = work @ work.T
        cubic = gram @ work
        work = a * work + b * cubic + c * gram @ cubic
    if transposed:
        work = work.T
    return cast(Tensor, work.to(original_dtype))


def polar_express(matrix: Tensor, steps: int = 5) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("Polar Express requires a matrix")
    if steps < 0:
        raise ValueError("Polar Express step count cannot be negative")
    work = _working_copy(matrix)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    work = work / (work.norm() * 1.01 + 1e-7)
    for step in range(steps):
        a, b, c = _POLAR_EXPRESS_COEFFICIENTS[min(step, len(_POLAR_EXPRESS_COEFFICIENTS) - 1)]
        gram = work @ work.T
        polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        work = torch.addmm(work, polynomial, work, beta=a)
    if transposed:
        work = work.T
    return work


def row_normalize(matrix: Tensor, eps: float = 1e-12) -> Tensor:
    return cast(Tensor, matrix / matrix.norm(dim=1, keepdim=True).clamp_min(eps))


def tangent_projection(direction: Tensor, gradient: Tensor) -> Tensor:
    # Remove the part of the gradient that changes row length.
    return gradient - (gradient * direction).sum(dim=1, keepdim=True) * direction
