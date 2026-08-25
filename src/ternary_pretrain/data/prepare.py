from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from itertools import chain, islice
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
from huggingface_hub import snapshot_download

from ternary_pretrain.config import DataConfig, file_sha256


def _read_jsonl(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path}:{line_number} has no non-empty text field")
            yield text


def _read_parquet(path: Path) -> Iterator[str]:
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=["text"], batch_size=1024):
        for text in batch.column(0).to_pylist():
            if isinstance(text, str) and text:
                yield text


def _source_documents(config: DataConfig) -> tuple[list[str], dict[str, Any]]:
    source: dict[str, Any]
    if config.mode == "local":
        document_iterator = chain.from_iterable(_read_jsonl(path) for path in config.source_files)
        source = {
            "mode": "local",
            "files": [
                {"path": path.as_posix(), "sha256": file_sha256(path)}
                for path in config.source_files
            ],
        }
    else:
        # Download only the pinned files listed in the config.
        snapshot = Path(
            snapshot_download(
                repo_id=config.repo_id or "",
                repo_type="dataset",
                revision=config.revision,
                allow_patterns=list(config.allow_patterns),
            )
        )
        files = sorted(
            {
                path
                for pattern in config.allow_patterns
                for path in snapshot.glob(pattern)
                if path.suffix == ".parquet"
            }
        )
        if not files:
            raise ValueError("the allow-list resolved no Parquet files")
        document_iterator = chain.from_iterable(_read_parquet(path) for path in files)
        source = {
            "mode": "huggingface",
            "repo_id": config.repo_id,
            "revision": config.revision,
            "allow_patterns": list(config.allow_patterns),
            "files": [path.relative_to(snapshot).as_posix() for path in files],
        }
    documents = list(islice(document_iterator, config.max_documents))
    if len(documents) < 2:
        raise ValueError("at least two documents are required")
    return documents, source


def prepare_data(config: DataConfig) -> Path:
    """Prepare immutable train and validation JSONL splits plus a source manifest."""
    documents, source = _source_documents(config)
    # Include the source position so duplicate documents still get unique IDs.
    identities = [
        hashlib.sha256(f"{config.seed}\0{index}\0{text}".encode()).hexdigest()
        for index, text in enumerate(documents)
    ]
    validation_count = max(
        1, min(len(documents) - 1, round(len(documents) * config.validation_fraction))
    )
    validation_ids = set(sorted(identities)[:validation_count])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    split_hashes: dict[str, str] = {}
    for split in ("train", "validation"):
        selected = [
            (identity, text)
            for identity, text in zip(identities, documents, strict=True)
            if (identity in validation_ids) == (split == "validation")
        ]
        output = config.output_dir / f"{split}.jsonl"
        # Do not overwrite data that another artifact may already use.
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            for identity, text in selected:
                stream.write(json.dumps({"id": identity, "text": text}, ensure_ascii=False) + "\n")
        split_counts[split] = len(selected)
        split_hashes[split] = file_sha256(output)
    manifest = config.output_dir / "prepare.manifest.json"
    payload = {
        "format_version": 1,
        "source": source,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "document_counts": split_counts,
        "split_sha256": split_hashes,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
