from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ternary_pretrain.config import RunConfig, canonical_data, config_hash, file_sha256


def _git_state(repo: Path) -> dict[str, str | bool | None]:
    def command(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    revision = command("rev-parse", "HEAD")
    status = command("status", "--porcelain", "--untracked-files=normal")
    return {"revision": revision, "dirty": None if status is None else bool(status)}


def build_run_manifest(
    config: RunConfig,
    *,
    repo_root: Path,
    world_size: int,
    tokenizer_hash: str,
    train_data_hash: str,
    validation_data_hash: str,
) -> dict[str, Any]:
    # Record known safe values only. Never copy environment variables here.
    lock = repo_root / "uv.lock"
    train_manifest = json.loads(config.train_manifest.read_text(encoding="utf-8"))
    validation_manifest = json.loads(config.validation_manifest.read_text(encoding="utf-8"))
    dependencies: dict[str, str | None] = {}
    for package in (
        "torch",
        "numpy",
        "huggingface-hub",
        "pyarrow",
        "tokenizers",
        "safetensors",
        "tensorboard",
    ):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": canonical_data(config),
        "config_sha256": config_hash(config),
        "dependency_lock_sha256": file_sha256(lock) if lock.is_file() else None,
        "git": _git_state(repo_root),
        "artifacts": {
            "tokenizer_sha256": tokenizer_hash,
            "train_data_sha256": train_data_hash,
            "validation_data_sha256": validation_data_hash,
        },
        "training": {
            "seed": config.seed,
            "precision": config.runtime.precision,
            "optimizer": config.optimizer.kind,
            "world_size": world_size,
            "train_tokens_available": int(train_manifest["token_count"]),
            "validation_tokens_available": int(validation_manifest["token_count"]),
        },
        "platform": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": dependencies,
    }


def write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    # Replace the file only after the new JSON is complete.
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)
