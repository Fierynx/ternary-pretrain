from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ternary_pretrain.config import RunConfig, config_hash, load_model_config
from ternary_pretrain.data import MMapTokenStream
from ternary_pretrain.evaluation import evaluate_model
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.optim import build_optimizer
from ternary_pretrain.optim.schedule import WarmupStableCooldown
from ternary_pretrain.tracking import LocalTracker
from ternary_pretrain.training.checkpoint import load_checkpoint, save_checkpoint
from ternary_pretrain.training.distributed import DistributedContext, finalize, initialize
from ternary_pretrain.training.identity import checkpoint_identity
from ternary_pretrain.training.manifest import (
    build_run_manifest,
    write_json_atomic,
    write_json_exclusive,
)


@dataclass(frozen=True, slots=True)
class TrainResult:
    run_dir: Path
    checkpoint: Path
    completed_steps: int
    consumed_tokens: int
    status: str


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_run_dir(config: RunConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = config.output_dir / f"{config.name}-{timestamp}-{config_hash(config)[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    for child in ("events", "checkpoints", "exports"):
        (path / child).mkdir()
    return path


def _broadcast_run_dir(path: Path | None, context: DistributedContext) -> Path:
    if context.world_size == 1:
        if path is None:
            raise RuntimeError("primary process did not create a run directory")
        return path
    values = [str(path) if context.is_primary and path is not None else None]
    dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], str):
        raise RuntimeError("failed to broadcast the run directory")
    return Path(values[0])


def _base_model(model: nn.Module) -> DecoderLM:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(module, DecoderLM):
        raise TypeError("training model is not a DecoderLM")
    return module


def _set_transition_state(
    model: DecoderLM, config: RunConfig, *, consumed_before_micro_batch: int
) -> None:
    if config.quantization.mode == "disabled":
        enabled = False
    elif config.quantization.mode == "from_init":
        enabled = True
    else:
        boundary = config.quantization.transition_tokens
        if boundary is None:
            raise RuntimeError("transition boundary was not validated")
        enabled = consumed_before_micro_batch >= boundary
    if model.qat_enabled != enabled:
        # Keep the same weights and optimizer state when QAT starts.
        model.set_qat_enabled(enabled)


def _reduced_mean(value: float, context: DistributedContext) -> float:
    if context.world_size == 1:
        return value
    tensor = torch.tensor(value, device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor / context.world_size)


def train(
    config: RunConfig,
    *,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
) -> TrainResult:
    """Train or explicitly resume a run; checkpoint only completed optimizer steps."""
    context = initialize(config.runtime.device, config.runtime.distributed_backend)
    tracker: LocalTracker | None = None
    run_dir: Path | None = None
    try:
        if config.runtime.micro_batch_size % context.world_size:
            raise ValueError("runtime.micro_batch_size must be divisible by world size")
        # micro_batch_size is shared across all ranks.
        local_batch_size = config.runtime.micro_batch_size // context.world_size
        tokens_per_micro = config.runtime.micro_batch_size * config.runtime.sequence_length
        tokens_per_step = tokens_per_micro * config.runtime.gradient_accumulation_steps
        if (
            config.quantization.mode == "transition"
            and config.quantization.transition_tokens is not None
            and config.quantization.transition_tokens % tokens_per_micro
        ):
            raise ValueError("transition_tokens must align with a logical micro-batch boundary")

        _seed_all(config.seed)
        model_config = load_model_config(config.model_config)
        if config.runtime.sequence_length > model_config.max_seq_len:
            raise ValueError("runtime sequence length exceeds the model maximum")
        tokenizer_metadata = json.loads(
            config.tokenizer_file.with_suffix(".metadata.json").read_text(encoding="utf-8")
        )
        if int(tokenizer_metadata["vocab_size"]) != model_config.vocab_size:
            raise ValueError("tokenizer and model vocabulary sizes differ")
        train_stream = MMapTokenStream(
            config.train_manifest, config.runtime.sequence_length, config.seed
        )
        validation_stream = MMapTokenStream(
            config.validation_manifest, config.runtime.sequence_length, 0
        )
        tokenizer_hash = str(tokenizer_metadata["tokenizer_sha256"])
        if (
            train_stream.manifest["tokenizer_sha256"] != tokenizer_hash
            or validation_stream.manifest["tokenizer_sha256"] != tokenizer_hash
        ):
            raise ValueError("dataset and tokenizer hashes differ")

        native_model = DecoderLM(model_config).to(context.device)
        _set_transition_state(native_model, config, consumed_before_micro_batch=0)
        optimizer, partition = build_optimizer(native_model, config.optimizer)
        scheduler = WarmupStableCooldown(optimizer, config.schedule)
        compatibility = checkpoint_identity(config)

        completed_steps = 0
        consumed_tokens = 0
        if resume is not None:
            # Resume inside the original run directory.
            inferred_run_dir = resume.resolve().parent.parent
            if not (inferred_run_dir / "run.json").is_file():
                raise ValueError("resume checkpoint is not inside a valid run directory")
            manifest = json.loads((inferred_run_dir / "run.json").read_text(encoding="utf-8"))
            if manifest["config_sha256"] != compatibility["config_sha256"]:
                raise ValueError("resume run manifest has a different configuration hash")
            payload = load_checkpoint(
                resume,
                model=native_model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected=compatibility,
            )
            completed_steps = int(payload["completed_steps"])
            consumed_tokens = int(payload["consumed_tokens"])
            if consumed_tokens != completed_steps * tokens_per_step:
                raise ValueError("checkpoint token count does not match its completed step")
            run_dir = inferred_run_dir if context.is_primary else None
        elif context.is_primary:
            run_dir = _new_run_dir(config)
            repo_root = Path(__file__).resolve().parents[3]
            manifest = build_run_manifest(
                config,
                repo_root=repo_root,
                world_size=context.world_size,
                tokenizer_hash=tokenizer_hash,
                train_data_hash=train_stream.identity,
                validation_data_hash=validation_stream.identity,
            )
            manifest["parameter_partition"] = partition.audit()
            write_json_exclusive(run_dir / "run.json", manifest)
        run_dir = _broadcast_run_dir(run_dir, context)
        if context.world_size > 1:
            ddp_device_ids = [context.local_rank] if context.device.type == "cuda" else None
            model: nn.Module = DistributedDataParallel(native_model, device_ids=ddp_device_ids)
        else:
            model = native_model
        if context.is_primary:
            tracker = LocalTracker(
                run_dir,
                wandb_project=config.tracking.wandb_project if config.tracking.wandb else None,
            )

        requested_stop = config.runtime.max_steps if stop_after_steps is None else stop_after_steps
        target_step = min(config.runtime.max_steps, requested_stop)
        if target_step < completed_steps:
            raise ValueError("stop_after_steps precedes the resumed checkpoint")
        final_metrics: dict[str, float | int] = {}
        model.train()
        while completed_steps < target_step:
            optimizer.zero_grad()
            accumulated_loss = 0.0
            for micro_step in range(config.runtime.gradient_accumulation_steps):
                consumed_before = consumed_tokens + micro_step * tokens_per_micro
                _set_transition_state(
                    _base_model(model), config, consumed_before_micro_batch=consumed_before
                )
                inputs, labels = train_stream.batch(
                    completed_step=completed_steps,
                    micro_step=micro_step,
                    batch_size=local_batch_size,
                    gradient_accumulation_steps=config.runtime.gradient_accumulation_steps,
                    rank=context.rank,
                    world_size=context.world_size,
                )
                inputs, labels = inputs.to(context.device), labels.to(context.device)
                synchronize = micro_step + 1 == config.runtime.gradient_accumulation_steps
                # Sync gradients only on the last accumulated micro-batch.
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel) and not synchronize
                    else nullcontext()
                )
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if config.runtime.precision == "bfloat16"
                    else nullcontext()
                )
                with sync_context, autocast_context:
                    output = model(inputs, labels=labels)
                    if output.loss is None:
                        raise RuntimeError("training model did not return loss")
                    loss = output.loss / config.runtime.gradient_accumulation_steps
                    loss.backward()
                    accumulated_loss += float(output.loss.detach())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.runtime.gradient_clip_norm
            )
            scheduler.step()
            optimizer.step()
            # Move the data cursor after a complete optimizer step.
            completed_steps += 1
            consumed_tokens += tokens_per_step
            _set_transition_state(native_model, config, consumed_before_micro_batch=consumed_tokens)
            mean_loss = _reduced_mean(
                accumulated_loss / config.runtime.gradient_accumulation_steps, context
            )
            if (
                context.is_primary
                and tracker is not None
                and (completed_steps % config.runtime.log_interval == 0)
            ):
                tracker.log(
                    completed_steps,
                    {
                        "train/loss": mean_loss,
                        "train/gradient_norm": float(gradient_norm),
                        "train/consumed_tokens": consumed_tokens,
                        "train/qat_enabled": int(native_model.qat_enabled),
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    },
                )
            if completed_steps % config.runtime.evaluation_interval == 0:
                if context.is_primary:
                    final_metrics = evaluate_model(
                        native_model,
                        validation_stream,
                        device=context.device,
                        batch_size=local_batch_size,
                        max_batches=2,
                    )
                    if tracker is not None:
                        tracker.log(completed_steps, final_metrics)
                if context.world_size > 1:
                    dist.barrier()
                model.train()
            if completed_steps % config.runtime.checkpoint_interval == 0 and context.is_primary:
                save_checkpoint(
                    run_dir / "checkpoints" / f"step-{completed_steps:08d}.pt",
                    model=native_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    completed_steps=completed_steps,
                    consumed_tokens=consumed_tokens,
                    compatibility=compatibility,
                    status="running",
                )

        status = "completed" if completed_steps == config.runtime.max_steps else "stopped"
        checkpoint_path = run_dir / "checkpoints" / f"step-{completed_steps:08d}.pt"
        if context.is_primary:
            save_checkpoint(
                checkpoint_path,
                model=native_model,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_steps=completed_steps,
                consumed_tokens=consumed_tokens,
                compatibility=compatibility,
                status=status,
            )
            write_json_atomic(
                run_dir / "summary.json",
                {
                    "status": status,
                    "completed_steps": completed_steps,
                    "successful_tokens": consumed_tokens,
                    "final_metrics": final_metrics,
                    "checkpoint": checkpoint_path.name,
                },
            )
        if context.world_size > 1:
            dist.barrier()
        return TrainResult(run_dir, checkpoint_path, completed_steps, consumed_tokens, status)
    finally:
        if tracker is not None:
            tracker.close()
        finalize(context)
