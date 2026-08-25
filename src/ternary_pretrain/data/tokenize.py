from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from tokenizers import Tokenizer

from ternary_pretrain.config import DataConfig, file_sha256
from ternary_pretrain.data.tokenizer import load_tokenizer


def _encoded_documents(
    stream: TextIO, tokenizer: Tokenizer, eod_id: int, batch_size: int = 256
) -> Iterator[tuple[str, list[int]]]:
    texts: list[str] = []
    for line in stream:
        document = json.loads(line)
        text = document.get("text")
        if not isinstance(text, str):
            raise ValueError("prepared document has no text field")
        texts.append(text)
        if len(texts) < batch_size:
            continue
        for batch_text, encoding in zip(
            texts,
            tokenizer.encode_batch(texts, add_special_tokens=False),
            strict=True,
        ):
            yield batch_text, [*encoding.ids, eod_id]
        texts.clear()
    if texts:
        for batch_text, encoding in zip(
            texts,
            tokenizer.encode_batch(texts, add_special_tokens=False),
            strict=True,
        ):
            yield batch_text, [*encoding.ids, eod_id]


def _write_shard(
    output_dir: Path,
    split: str,
    index: int,
    token_ids: list[int],
    document_count: int,
    byte_count: int,
    tokenizer_hash: str,
    source_hash: str,
    source_revision: str | None,
) -> dict[str, Any]:
    path = output_dir / f"{split}-{index:05d}.npy"
    if max(token_ids, default=0) > np.iinfo(np.uint16).max:
        raise ValueError("token id exceeds uint16 range")
    # Always write the same byte order on every platform.
    array = np.asarray(token_ids, dtype=np.dtype("<u2"))
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    sidecar = {
        "format_version": 1,
        "file": path.name,
        "sha256": file_sha256(path),
        "tokenizer_sha256": tokenizer_hash,
        "source_sha256": source_hash,
        "source_revision": source_revision,
        "token_count": len(token_ids),
        "document_count": document_count,
        "byte_count": byte_count,
        "dtype": "<u2",
    }
    sidecar_path = path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def tokenize_data(config: DataConfig) -> tuple[Path, Path]:
    """Encode prepared splits into deterministic memory-mappable uint16 shards."""
    tokenizer = load_tokenizer(config.tokenizer_file)
    metadata = json.loads(
        config.tokenizer_file.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    tokenizer_hash = str(metadata["tokenizer_sha256"])
    eod_id = int(metadata["eod_token_id"])
    if config.token_output_dir.exists():
        raise FileExistsError(f"token output already exists: {config.token_output_dir}")
    config.token_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{config.token_output_dir.name}-",
            dir=config.token_output_dir.parent,
        )
    )
    manifests: list[Path] = []
    try:
        for split in ("train", "validation"):
            source_path = config.output_dir / f"{split}.jsonl"
            source_hash = file_sha256(source_path)
            shard_manifests: list[dict[str, Any]] = []
            shard_token_ids: list[int] = []
            shard_document_count = 0
            shard_byte_count = 0
            total_documents = 0
            total_bytes = 0
            with source_path.open("r", encoding="utf-8") as stream:
                for text, document_token_ids in _encoded_documents(stream, tokenizer, eod_id):
                    # Keep each document and its EOD token in the same shard.
                    if (
                        shard_token_ids
                        and len(shard_token_ids) + len(document_token_ids)
                        > config.shard_token_limit
                    ):
                        shard_manifests.append(
                            _write_shard(
                                staging_dir,
                                split,
                                len(shard_manifests),
                                shard_token_ids,
                                shard_document_count,
                                shard_byte_count,
                                tokenizer_hash,
                                source_hash,
                                config.revision,
                            )
                        )
                        shard_token_ids, shard_document_count, shard_byte_count = [], 0, 0
                    shard_token_ids.extend(document_token_ids)
                    shard_document_count += 1
                    shard_byte_count += len(text.encode("utf-8"))
                    total_documents += 1
                    total_bytes += len(text.encode("utf-8"))
            if shard_token_ids:
                shard_manifests.append(
                    _write_shard(
                        staging_dir,
                        split,
                        len(shard_manifests),
                        shard_token_ids,
                        shard_document_count,
                        shard_byte_count,
                        tokenizer_hash,
                        source_hash,
                        config.revision,
                    )
                )
            if not shard_manifests:
                raise ValueError(f"prepared {split} split is empty")
            manifest_payload = {
                "format_version": 1,
                "split": split,
                "source_sha256": source_hash,
                "source_revision": config.revision,
                "tokenizer_sha256": tokenizer_hash,
                "token_count": sum(int(shard["token_count"]) for shard in shard_manifests),
                "document_count": total_documents,
                "byte_count": total_bytes,
                "shards": shard_manifests,
            }
            manifest_path = staging_dir / f"{split}.manifest.json"
            manifest_path.write_text(
                json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            manifests.append(manifest_path)
        staging_dir.replace(config.token_output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return (
        config.token_output_dir / manifests[0].name,
        config.token_output_dir / manifests[1].name,
    )
