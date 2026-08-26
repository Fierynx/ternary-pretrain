from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_model

from ternary_pretrain.config import OptimizerKind, QuantizationMode
from ternary_pretrain.evaluation.runner import evaluate_checkpoint, load_native_checkpoint
from ternary_pretrain.export import export_checkpoint
from ternary_pretrain.model import DecoderLM
from ternary_pretrain.training import train
from ternary_pretrain.training.manifest import _git_state
from tests.testing import ExperimentFixture


def test_container_build_revision_is_recorded(tmp_path: Path) -> None:
    revision = "a" * 40
    (tmp_path / ".build-revision").write_text(f"{revision}\n", encoding="utf-8")
    assert _git_state(tmp_path) == {"revision": revision, "dirty": None}


def checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model"]


def test_uninterrupted_and_resumed_training_are_identical(
    prepared_experiment: ExperimentFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = prepared_experiment.run_config
    monkeypatch.setenv("VASTAI_API_KEY", "fake-vast-credential-that-must-not-be-recorded")
    monkeypatch.setenv("HF_TOKEN", "fake-huggingface-credential-that-must-not-be-recorded")
    uninterrupted = train(config)
    interrupted = train(config, stop_after_steps=1)
    resumed = train(config, resume=interrupted.checkpoint)
    assert resumed.run_dir == interrupted.run_dir
    for name, expected in checkpoint_model(uninterrupted.checkpoint).items():
        torch.testing.assert_close(
            expected, checkpoint_model(resumed.checkpoint)[name], rtol=0, atol=0
        )
    uninterrupted_metrics = (uninterrupted.run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    resumed_metrics = (resumed.run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert uninterrupted_metrics == resumed_metrics
    run_manifest = json.loads((resumed.run_dir / "run.json").read_text(encoding="utf-8"))
    serialized_manifest = json.dumps(run_manifest)
    assert "environment" not in run_manifest
    assert "fake-vast-credential" not in serialized_manifest
    assert "fake-huggingface-credential" not in serialized_manifest
    assert run_manifest["training"]["world_size"] == 1
    assert run_manifest["platform"]["accelerator"] == {"device_type": "cpu"}
    assert "wandb" not in sys.modules


@pytest.mark.parametrize("kind", ["adamw", "muon", "muown", "angular_muown"])
def test_minimal_training_run_for_every_optimizer(
    prepared_experiment: ExperimentFixture, kind: OptimizerKind
) -> None:
    config = prepared_experiment.run_config
    one_step = replace(
        config,
        name=f"optimizer-{kind}",
        optimizer=replace(config.optimizer, kind=kind),
        runtime=replace(config.runtime, max_steps=1),
        schedule=replace(config.schedule, stable_steps=1),
    )
    result = train(one_step)
    assert result.status == "completed"


@pytest.mark.parametrize(
    ("mode", "transition_tokens"),
    [("disabled", None), ("from_init", None), ("transition", 16)],
)
def test_minimal_training_run_for_every_qat_mode(
    prepared_experiment: ExperimentFixture,
    mode: QuantizationMode,
    transition_tokens: int | None,
) -> None:
    config = prepared_experiment.run_config
    one_step = replace(
        config,
        name=f"qat-{mode}",
        runtime=replace(config.runtime, max_steps=1),
        schedule=replace(config.schedule, stable_steps=1),
        quantization=replace(
            config.quantization,
            mode=mode,
            transition_tokens=transition_tokens,
        ),
    )
    result = train(one_step)
    model, _ = load_native_checkpoint(one_step, result.checkpoint)
    assert model.qat_enabled is (mode != "disabled")


def test_evaluation_native_and_huggingface_exports_match_logits(
    prepared_experiment: ExperimentFixture,
) -> None:
    from ternary_pretrain.integrations.transformers import TernaryPreTrainedModel

    config = prepared_experiment.run_config
    result = train(config)
    metrics = evaluate_checkpoint(config, result.checkpoint, max_batches=1)
    assert metrics["validation_nll"] > 0
    native_dir = export_checkpoint(config, result.checkpoint, export_format="native")
    huggingface_dir = export_checkpoint(config, result.checkpoint, export_format="huggingface")
    expected, _ = load_native_checkpoint(config, result.checkpoint)
    native = DecoderLM(expected.config)
    load_model(native, str(native_dir / "model.safetensors"), strict=True)
    huggingface = TernaryPreTrainedModel.from_pretrained(huggingface_dir)
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected_logits = expected(inputs).logits
        torch.testing.assert_close(native(inputs).logits, expected_logits)
        torch.testing.assert_close(huggingface(input_ids=inputs).logits, expected_logits)
