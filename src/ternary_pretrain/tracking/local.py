from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class LocalTracker:
    """Append scalar records locally and mirror them to TensorBoard."""

    def __init__(self, run_dir: Path, *, wandb_project: str | None = None) -> None:
        self.metrics_path = run_dir / "metrics.jsonl"
        self.stream = self.metrics_path.open("a", encoding="utf-8", buffering=1)
        self.writer = SummaryWriter(log_dir=str(run_dir / "events"))
        self.wandb_run: Any = None
        if wandb_project is not None:
            # W&B stays optional for local runs.
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError("W&B tracking requires the 'wandb' extra") from error
            self.wandb_run = wandb.init(project=wandb_project, dir=str(run_dir))

    def log(self, step: int, metrics: Mapping[str, float | int | str]) -> None:
        record: dict[str, float | int | str] = {"step": step, **metrics}
        self.stream.write(json.dumps(record, sort_keys=True) + "\n")
        scalar_metrics = {
            key: value for key, value in metrics.items() if isinstance(value, (float, int))
        }
        for name, value in scalar_metrics.items():
            self.writer.add_scalar(name, value, step)
        if self.wandb_run is not None:
            self.wandb_run.log(scalar_metrics, step=step)

    def close(self) -> None:
        self.stream.close()
        self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def __enter__(self) -> LocalTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
