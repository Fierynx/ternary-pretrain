from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ternary_pretrain.config import ModelConfig, OptimizerConfig, ScheduleConfig
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.optim import build_optimizer
from ternary_pretrain.optim.schedule import WarmupStableCooldown
from ternary_pretrain.training.checkpoint import save_checkpoint


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path: Path) -> None:
    model = DecoderLM(ModelConfig(64, 16, 1, 2, 1, 32, 8))
    optimizer, _ = build_optimizer(
        model,
        OptimizerConfig("adamw", 0.01, 0.01, 0.0, (0.9, 0.95), 1e-8, 0.9, 3),
    )
    scheduler = WarmupStableCooldown(optimizer, ScheduleConfig(0, 1, 0, 1.0))
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous-valid-checkpoint")
    with (
        patch("torch.save", side_effect=OSError("simulated write failure")),
        pytest.raises(OSError, match="simulated"),
    ):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_steps=0,
            consumed_tokens=0,
            compatibility={},
            status="running",
        )
    assert path.read_bytes() == b"previous-valid-checkpoint"
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()
