from __future__ import annotations

from typing import Any

from ternary_pretrain.config import ScheduleConfig
from ternary_pretrain.optim import CompositeOptimizer


class WarmupStableCooldown:
    """Piecewise-linear warmup, stable, and cooldown schedule."""

    def __init__(self, optimizer: CompositeOptimizer, config: ScheduleConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.completed_steps = 0

    def multiplier(self, completed_steps: int) -> float:
        if self.config.warmup_steps and completed_steps < self.config.warmup_steps:
            return (completed_steps + 1) / self.config.warmup_steps
        stable_end = self.config.warmup_steps + self.config.stable_steps
        if completed_steps < stable_end or self.config.cooldown_steps == 0:
            return 1.0
        progress = min(1.0, (completed_steps - stable_end + 1) / self.config.cooldown_steps)
        return 1.0 + progress * (self.config.final_learning_rate_ratio - 1.0)

    def step(self) -> None:
        factor = self.multiplier(self.completed_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor
        self.completed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {"base_lrs": self.base_lrs, "completed_steps": self.completed_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        base_lrs = [float(value) for value in state["base_lrs"]]
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler optimizer groups do not match checkpoint")
        self.base_lrs = base_lrs
        self.completed_steps = int(state["completed_steps"])
