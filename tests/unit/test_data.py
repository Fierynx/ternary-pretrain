from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ternary_pretrain.config import ConfigError, DataConfig, TokenizerConfig, file_sha256
from ternary_pretrain.data import MMapTokenStream
from ternary_pretrain.data.prepare import prepare_data
from ternary_pretrain.data.tokenizer import train_tokenizer
from tests.testing import ExperimentFixture


def test_split_isolation_shards_checksums_and_cross_shard_reads(
    prepared_experiment: ExperimentFixture,
) -> None:
    config = prepared_experiment.data_config
    train_documents = {
        json.loads(line)["id"]
        for line in (config.output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    }
    validation_documents = {
        json.loads(line)["id"]
        for line in (config.output_dir / "validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert train_documents.isdisjoint(validation_documents)
    manifest_path = config.token_output_dir / "train.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["shards"]) > 1
    for shard in manifest["shards"]:
        path = manifest_path.parent / shard["file"]
        assert np.load(path, mmap_mode="r").dtype.str == "<u2"
        assert file_sha256(path) == shard["sha256"]
    stream = MMapTokenStream(manifest_path, sequence_length=8, seed=3)
    width = stream.sequence_length + 1
    valid_starts = stream.token_count - width + 1
    sample_index = next(
        index
        for index in range(valid_starts)
        if any(
            (3 % valid_starts + index * width) % valid_starts
            < shard_end
            < (3 % valid_starts + index * width) % valid_starts + width
            for shard_end in stream.shard_ends[:-1]
        )
    )
    inputs, labels = stream.sample(sample_index)
    assert inputs.shape == labels.shape == (8,)
    assert inputs[1:].equal(labels[:-1])


def test_ddp_ranks_receive_disjoint_logical_samples(
    prepared_experiment: ExperimentFixture,
) -> None:
    config = prepared_experiment.data_config
    stream = MMapTokenStream(config.token_output_dir / "train.manifest.json", 8, seed=7)
    rank_zero = stream.batch(
        completed_step=0,
        micro_step=0,
        batch_size=1,
        gradient_accumulation_steps=1,
        rank=0,
        world_size=2,
    )[0]
    rank_one = stream.batch(
        completed_step=0,
        micro_step=0,
        batch_size=1,
        gradient_accumulation_steps=1,
        rank=1,
        world_size=2,
    )[0]
    assert not rank_zero.equal(rank_one)


def test_shard_hash_mismatch_is_rejected(
    prepared_experiment: ExperimentFixture,
) -> None:
    config = prepared_experiment.data_config
    manifest_path = config.token_output_dir / "train.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = manifest_path.parent / manifest["shards"][0]["file"]
    with shard_path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        MMapTokenStream(manifest_path, 8, seed=0)


def test_prepare_never_overwrites_existing_splits(
    prepared_experiment: ExperimentFixture,
) -> None:
    config = prepared_experiment.data_config
    with pytest.raises(FileExistsError):
        prepare_data(config)


def test_tokenizer_training_is_deterministic(
    prepared_experiment: ExperimentFixture, tmp_path: Path
) -> None:
    config = prepared_experiment.data_config
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name / "tokenizer.json"
        train_tokenizer(
            TokenizerConfig(
                input_files=(config.output_dir / "train.jsonl",),
                output_file=output,
                vocab_size=320,
                min_frequency=1,
                eod_token="<|endofdocument|>",
            )
        )
        outputs.append(file_sha256(output))
    assert outputs[0] == outputs[1]


def test_huggingface_mode_requires_frozen_capped_fineweb_edu(tmp_path: Path) -> None:
    config = DataConfig(
        mode="huggingface",
        source_files=(),
        output_dir=tmp_path / "data",
        tokenizer_file=tmp_path / "tokenizer.json",
        token_output_dir=tmp_path / "tokens",
        validation_fraction=0.1,
        seed=1,
        shard_token_limit=100,
        repo_id="another/dataset",
        revision="main",
        allow_patterns=("data/*.parquet",),
    )
    with pytest.raises(ConfigError, match="FineWeb-Edu"):
        config.validate()
