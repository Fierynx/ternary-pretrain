from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from ternary_pretrain.optim import CompositeOptimizer
from ternary_pretrain.optim.schedule import WarmupStableCooldown


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: CompositeOptimizer,
    scheduler: WarmupStableCooldown,
    completed_steps: int,
    consumed_tokens: int,
    compatibility: dict[str, str],
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "completed_steps": completed_steps,
        "consumed_tokens": consumed_tokens,
        "data_cursor": {"completed_steps": completed_steps},
        "rng": capture_rng_state(),
        "compatibility": compatibility,
        "status": status,
    }
    try:
        # Finish the temporary file before replacing the checkpoint.
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: CompositeOptimizer | None,
    scheduler: WarmupStableCooldown | None,
    expected: dict[str, str],
    restore_rng: bool = True,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    # Check compatibility before changing model state.
    if payload.get("compatibility") != expected:
        raise ValueError("checkpoint configuration or artifact hashes are incompatible")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return payload
