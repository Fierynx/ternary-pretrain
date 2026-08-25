from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch import Tensor, nn


class MatrixOptimizer(torch.optim.Optimizer, ABC):
    def __init__(self, params: Iterable[nn.Parameter], defaults: dict[str, Any]) -> None:
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for parameter_group in self.param_groups:
            for parameter in parameter_group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise RuntimeError("matrix optimizer received a non-matrix parameter")
                self._update_parameter(parameter, parameter.grad, parameter_group)
        return loss

    @abstractmethod
    def _update_parameter(
        self, parameter: nn.Parameter, gradient: Tensor, parameter_group: dict[str, Any]
    ) -> None: ...
