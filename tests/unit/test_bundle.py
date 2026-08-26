from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ternary_pretrain.config import file_sha256
from ternary_pretrain.data.bundle import verify_training_data


def _write_bundle_fixture(root: Path) -> dict[str, object]:
    token_dir = root / "artifacts/data/fineweb_edu/tokens"
    tokenizer_dir = root / "artifacts/tokenizers/fineweb_edu_32k"
    token_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)

    tokenizer = tokenizer_dir / "tokenizer.json"
    tokenizer.write_text("{}\n", encoding="utf-8")
    tokenizer_hash = file_sha256(tokenizer)
    (tokenizer_dir / "tokenizer.metadata.json").write_text(
        json.dumps({"tokenizer_sha256": tokenizer_hash}), encoding="utf-8"
    )

    manifest_hashes: dict[str, str] = {}
    for split, values in (("train", [1, 2, 3]), ("validation", [4, 5, 6])):
        shard = token_dir / f"{split}-00000.npy"
        np.save(shard, np.asarray(values, dtype="<u2"), allow_pickle=False)
        manifest = {
            "source_revision": "source-revision",
            "tokenizer_sha256": tokenizer_hash,
            "token_count": len(values),
            "shards": [{"file": shard.name, "sha256": file_sha256(shard)}],
        }
        manifest_path = token_dir / f"{split}.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_hashes[split] = file_sha256(manifest_path)

    return {
        "source_revision": "source-revision",
        "tokenizer_sha256": tokenizer_hash,
        "train_manifest_sha256": manifest_hashes["train"],
        "validation_manifest_sha256": manifest_hashes["validation"],
        "train_tokens": 3,
        "validation_tokens": 3,
    }


def test_training_data_bundle_identity(tmp_path: Path) -> None:
    expected = _write_bundle_fixture(tmp_path)
    assert verify_training_data(tmp_path, expected) == expected


def test_training_data_bundle_rejects_changed_shard(tmp_path: Path) -> None:
    expected = _write_bundle_fixture(tmp_path)
    shard = tmp_path / "artifacts/data/fineweb_edu/tokens/train-00000.npy"
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="token shard hash mismatch"):
        verify_training_data(tmp_path, expected)
