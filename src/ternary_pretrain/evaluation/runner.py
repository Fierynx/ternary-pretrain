from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ternary_pretrain.config import RunConfig, load_model_config
from ternary_pretrain.data import MMapTokenStream
from ternary_pretrain.evaluation.metrics import evaluate_model
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.training.checkpoint import load_checkpoint
from ternary_pretrain.training.identity import checkpoint_identity
from ternary_pretrain.training.manifest import write_json_atomic


def load_native_checkpoint(config: RunConfig, checkpoint: Path) -> tuple[DecoderLM, dict[str, Any]]:
    model = DecoderLM(load_model_config(config.model_config))
    payload = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        expected=checkpoint_identity(config),
        restore_rng=False,
    )
    return model, payload


def evaluate_checkpoint(
    config: RunConfig, checkpoint: Path, *, max_batches: int = 8
) -> dict[str, float | int]:
    """Evaluate a checkpoint at deterministic validation offsets and persist the result."""
    model, payload = load_native_checkpoint(config, checkpoint)
    device = torch.device(config.runtime.device)
    model.to(device)
    stream = MMapTokenStream(config.validation_manifest, config.runtime.sequence_length, 0)
    metrics = evaluate_model(
        model,
        stream,
        device=device,
        batch_size=config.runtime.micro_batch_size,
        max_batches=max_batches,
    )
    output = (
        checkpoint.parent.parent / f"evaluation-step-{int(payload['completed_steps']):08d}.json"
    )
    write_json_atomic(output, metrics)
    return metrics
