from __future__ import annotations

from typing import cast

from torch import Tensor


def newton_schulz_polar(matrix: Tensor, steps: int = 5) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("polar update requires a matrix")
    original_dtype = matrix.dtype
    # Always run this math in float32.
    work = matrix.float()
    # Work on the smaller side of tall matrices.
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    work = work / (work.norm() + 1e-7)
    # Coefficients used by the quintic Newton-Schulz update.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = work @ work.T
        work = a * work + (b * gram + c * (gram @ gram)) @ work
    if transposed:
        work = work.T
    return cast(Tensor, work.to(original_dtype))


def row_normalize(matrix: Tensor, eps: float = 1e-12) -> Tensor:
    return cast(Tensor, matrix / matrix.norm(dim=1, keepdim=True).clamp_min(eps))


def tangent_projection(direction: Tensor, gradient: Tensor) -> Tensor:
    # Remove the part of the gradient that changes row length.
    return gradient - (gradient * direction).sum(dim=1, keepdim=True) * direction
