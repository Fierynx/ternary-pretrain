from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from ternary_pretrain.config import (
    ConfigError,
    OptimizerConfig,
    OptimizerKind,
    config_hash,
    load_data_config,
    load_model_config,
    load_run_config,
)


def test_model_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "model.toml"
    path.write_text(
        """vocab_size=64
d_model=16
n_layers=1
n_heads=2
n_kv_heads=1
ffn_dim=32
max_seq_len=16
unknown=true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_model_config(path)


def test_config_hash_is_stable(small_model_config: object) -> None:
    assert config_hash(small_model_config) == config_hash(small_model_config)
    assert len(config_hash(small_model_config)) == 64


def test_model_config_rejects_wrong_toml_types(tmp_path: Path) -> None:
    path = tmp_path / "model.toml"
    path.write_text(
        """vocab_size=64
d_model="16"
n_layers=1
n_heads=2
n_kv_heads=1
ffn_dim=32
max_seq_len=16
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="dimensions"):
        load_model_config(path)


def test_optimizer_config_rejects_unknown_condition() -> None:
    config = OptimizerConfig(
        kind=cast(OptimizerKind, "unknown"),
        learning_rate=0.01,
        auxiliary_learning_rate=0.01,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        momentum=0.9,
        newton_schulz_steps=5,
    )
    with pytest.raises(ConfigError, match="unsupported optimizer"):
        config.validate()


def test_25m_config_has_expected_parameter_count() -> None:
    from ternary_pretrain.model import DecoderLM

    root = Path(__file__).resolve().parents[2]
    model = DecoderLM(load_model_config(root / "configs/models/25m.toml"))
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 25_000_000 <= count <= 26_000_000


def test_gpu_configs_are_explicit_and_frozen() -> None:
    root = Path(__file__).resolve().parents[2]
    data = load_data_config(root / "configs/data/fineweb_edu.toml")
    assert len(data.revision or "") == 40
    assert data.allow_patterns == (
        "sample/10BT/000_00000.parquet",
        "sample/10BT/001_00000.parquet",
    )
    for name in ("gpu_canary.toml", "25m_pilot.toml"):
        run = load_run_config(root / "configs/runs" / name, require_artifacts=False)
        assert run.runtime.device == "cuda"
        assert run.runtime.precision == "bfloat16"
        assert run.runtime.deterministic is True
        assert run.runtime.allow_tf32 is False


def test_lr_calibration_matrix_is_qat_disabled() -> None:
    root = Path(__file__).resolve().parents[2]
    calibration_dir = root / "configs/runs/calibration"
    expected = {
        "adamw-3e-4.toml": ("adamw", 3e-4),
        "adamw-6e-4.toml": ("adamw", 6e-4),
        "adamw-1p2e-3.toml": ("adamw", 1.2e-3),
        "muon-1e-2.toml": ("muon", 1e-2),
        "muon-2e-2.toml": ("muon", 2e-2),
        "muon-4e-2.toml": ("muon", 4e-2),
        "muown-1e-3.toml": ("muown", 1e-3),
        "muown-2e-3.toml": ("muown", 2e-3),
        "muown-4e-3.toml": ("muown", 4e-3),
        "angular-muown-1e-2.toml": ("angular_muown", 1e-2),
        "angular-muown-2e-2.toml": ("angular_muown", 2e-2),
        "angular-muown-4e-2.toml": ("angular_muown", 4e-2),
    }
    assert {path.name for path in calibration_dir.glob("*.toml")} == set(expected)
    for name, (optimizer, learning_rate) in expected.items():
        run = load_run_config(calibration_dir / name, require_artifacts=False)
        assert run.optimizer.kind == optimizer
        assert run.optimizer.learning_rate == learning_rate
        assert run.quantization.mode == "disabled"
        assert run.seed == 1729
        assert run.runtime.max_steps == 1000
        assert run.runtime.checkpoint_interval == run.runtime.max_steps
