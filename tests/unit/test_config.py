from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from ternary_pretrain.config import (
    ConfigError,
    OptimizerConfig,
    OptimizerKind,
    config_hash,
    load_model_config,
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
