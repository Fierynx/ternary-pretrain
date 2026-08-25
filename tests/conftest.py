from __future__ import annotations

import json
from pathlib import Path

import pytest

from ternary_pretrain.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    QuantizationConfig,
    RunConfig,
    RuntimeConfig,
    ScheduleConfig,
    TokenizerConfig,
    TrackingConfig,
)
from ternary_pretrain.data.prepare import prepare_data
from ternary_pretrain.data.tokenize import tokenize_data
from ternary_pretrain.data.tokenizer import train_tokenizer
from tests.testing import ExperimentFixture


@pytest.fixture
def prepared_experiment(tmp_path: Path) -> ExperimentFixture:
    corpus = tmp_path / "corpus.jsonl"
    documents = [
        {"text": f"Document {index} contains deterministic byte-level training text."}
        for index in range(24)
    ]
    corpus.write_text(
        "".join(json.dumps(document) + "\n" for document in documents), encoding="utf-8"
    )
    prepared = tmp_path / "prepared"
    tokenizer_path = tmp_path / "tokenizer" / "tokenizer.json"
    token_dir = tmp_path / "tokens"
    data_config = DataConfig(
        mode="local",
        source_files=(corpus,),
        output_dir=prepared,
        tokenizer_file=tokenizer_path,
        token_output_dir=token_dir,
        validation_fraction=0.25,
        seed=17,
        shard_token_limit=80,
    )
    prepare_data(data_config)
    tokenizer_config = TokenizerConfig(
        input_files=(prepared / "train.jsonl",),
        output_file=tokenizer_path,
        vocab_size=320,
        min_frequency=1,
        eod_token="<|endofdocument|>",
    )
    train_tokenizer(tokenizer_config)
    train_manifest, validation_manifest = tokenize_data(data_config)
    tokenizer_metadata = json.loads(
        tokenizer_path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    model_path = tmp_path / "model.toml"
    model_path.write_text(
        "\n".join(
            (
                f"vocab_size = {tokenizer_metadata['vocab_size']}",
                "d_model = 16",
                "n_layers = 1",
                "n_heads = 2",
                "n_kv_heads = 1",
                "ffn_dim = 32",
                "max_seq_len = 16",
                "rope_theta = 10000.0",
                "rms_norm_eps = 1e-5",
                "initializer_std = 0.02",
                "",
            )
        ),
        encoding="utf-8",
    )
    run_config = RunConfig(
        name="test",
        output_dir=tmp_path / "runs",
        model_config=model_path,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        tokenizer_file=tokenizer_path,
        seed=23,
        runtime=RuntimeConfig(
            device="cpu",
            distributed_backend="gloo",
            precision="float32",
            max_steps=2,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            sequence_length=8,
            gradient_clip_norm=1.0,
            log_interval=1,
            checkpoint_interval=1,
            evaluation_interval=1,
        ),
        optimizer=OptimizerConfig(
            kind="adamw",
            learning_rate=0.003,
            auxiliary_learning_rate=0.003,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            momentum=0.9,
            newton_schulz_steps=3,
        ),
        schedule=ScheduleConfig(
            warmup_steps=0,
            stable_steps=2,
            cooldown_steps=0,
            final_learning_rate_ratio=1.0,
        ),
        quantization=QuantizationConfig(mode="disabled"),
        tracking=TrackingConfig(tensorboard=True, wandb=False),
    )
    return ExperimentFixture(run_config=run_config, data_config=data_config)


@pytest.fixture
def small_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=32,
        max_seq_len=16,
    )
