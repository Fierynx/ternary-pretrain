from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from itertools import chain, islice
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

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


def _source_documents(config: DataConfig) -> tuple[Iterator[str], dict[str, Any]]:
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
        # Use the declared HTTP dependency path on machines without optional Xet or symlinks.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        from huggingface_hub import snapshot_download

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
            "files": [
                {
                    "path": path.relative_to(snapshot).as_posix(),
                    "sha256": file_sha256(path),
                }
                for path in files
            ],
        }
    return islice(document_iterator, config.max_documents), source


def prepare_data(config: DataConfig) -> Path:
    """Prepare immutable train and validation JSONL splits plus a source manifest."""
    documents, source = _source_documents(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: config.output_dir / f"{split}.jsonl" for split in ("train", "validation")}
    existing = [
        str(path)
        for path in (*outputs.values(), config.output_dir / "prepare.manifest.json")
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"prepared data already exists: {', '.join(existing)}")
    split_counts: dict[str, int] = {}
    split_hashes: dict[str, str] = {}
    split_counts = {"train": 0, "validation": 0}
    threshold = int(config.validation_fraction * (1 << 64))
    try:
        with (
            outputs["train"].open("x", encoding="utf-8", newline="\n") as train_stream,
            outputs["validation"].open("x", encoding="utf-8", newline="\n") as validation_stream,
        ):
            streams = {"train": train_stream, "validation": validation_stream}
            for index, text in enumerate(documents):
                # The source position keeps duplicate documents distinct.
                identity = hashlib.sha256(f"{config.seed}\0{index}\0{text}".encode()).hexdigest()
                split = "validation" if int(identity[:16], 16) < threshold else "train"
                streams[split].write(
                    json.dumps({"id": identity, "text": text}, ensure_ascii=False) + "\n"
                )
                split_counts[split] += 1
        if min(split_counts.values()) == 0:
            raise ValueError("prepared data requires at least one document in each split")
    except BaseException:
        for output in outputs.values():
            output.unlink(missing_ok=True)
        raise
    for split, output in outputs.items():
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
