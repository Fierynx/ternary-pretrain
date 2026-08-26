from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ternary_pretrain.config import file_sha256

_TOKEN_PATH = Path("artifacts/data/fineweb_edu/tokens")
_TOKENIZER_PATH = Path("artifacts/tokenizers/fineweb_edu_32k")


def _manifest_path(archive: Path) -> Path:
    return archive.with_name(f"{archive.name}.manifest.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def _add_tree(archive: tarfile.TarFile, repo_root: Path, relative_root: Path) -> None:
    root = repo_root / relative_root
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = [root, *(path for path in root.rglob("*") if path.is_file())]
    for path in paths:
        archive.add(
            path,
            arcname=path.relative_to(repo_root).as_posix(),
            recursive=False,
            filter=_normalized_tar_info,
        )


def _dataset_identity(repo_root: Path) -> dict[str, Any]:
    token_dir = repo_root / _TOKEN_PATH
    tokenizer_dir = repo_root / _TOKENIZER_PATH
    train_path = token_dir / "train.manifest.json"
    validation_path = token_dir / "validation.manifest.json"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    metadata_path = tokenizer_dir / "tokenizer.metadata.json"
    train = _read_json(train_path)
    validation = _read_json(validation_path)
    metadata = _read_json(metadata_path)
    tokenizer_hash = file_sha256(tokenizer_path)
    if metadata.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("tokenizer metadata hash does not match tokenizer.json")
    for split, manifest in (("train", train), ("validation", validation)):
        if manifest.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError(f"{split} manifest uses a different tokenizer")
    return {
        "source_revision": train.get("source_revision"),
        "tokenizer_sha256": tokenizer_hash,
        "train_manifest_sha256": file_sha256(train_path),
        "validation_manifest_sha256": file_sha256(validation_path),
        "train_tokens": int(train["token_count"]),
        "validation_tokens": int(validation["token_count"]),
    }


def verify_training_data(repo_root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    actual = _dataset_identity(repo_root)
    for field, value in actual.items():
        if expected.get(field) != value:
            raise ValueError(f"training data identity mismatch: {field}")

    token_dir = repo_root / _TOKEN_PATH
    for manifest_name in ("train.manifest.json", "validation.manifest.json"):
        manifest = _read_json(token_dir / manifest_name)
        for shard in manifest["shards"]:
            shard_path = token_dir / str(shard["file"])
            if file_sha256(shard_path) != shard["sha256"]:
                raise ValueError(f"token shard hash mismatch: {shard_path}")
    return actual


def create_bundle(repo_root: Path, archive_path: Path) -> Path:
    repo_root = repo_root.resolve()
    archive_path = archive_path.resolve()
    manifest_path = _manifest_path(archive_path)
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError("bundle output already exists")
    zstd = shutil.which("zstd")
    if zstd is None:
        raise FileNotFoundError("zstd executable not found")

    identity = _dataset_identity(repo_root)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with archive_path.open("xb") as output:
            process = subprocess.Popen(
                [zstd, "-3", "--threads=0", "--stdout"],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None or process.stderr is None:
                raise RuntimeError("failed to open zstd streams")
            try:
                with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
                    _add_tree(archive, repo_root, _TOKEN_PATH)
                    _add_tree(archive, repo_root, _TOKENIZER_PATH)
            finally:
                process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"zstd failed: {stderr.strip()}")
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise

    manifest = {
        "format_version": 1,
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": file_sha256(archive_path),
        **identity,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _unpack_archive(archive_path: Path, destination: Path) -> None:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise FileNotFoundError("zstd executable not found")
    process = subprocess.Popen(
        [zstd, "--decompress", "--stdout", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("failed to open zstd streams")
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(destination, filter="data")
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"zstd failed: {stderr.strip()}")


def restore_bundle(repo_root: Path, archive_path: Path, manifest_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    archive_path = archive_path.resolve()
    manifest = _read_json(manifest_path.resolve())
    if manifest.get("archive") != archive_path.name:
        raise ValueError("bundle manifest names a different archive")
    if manifest.get("archive_bytes") != archive_path.stat().st_size:
        raise ValueError("bundle archive size mismatch")
    if manifest.get("archive_sha256") != file_sha256(archive_path):
        raise ValueError("bundle archive hash mismatch")

    destinations = [repo_root / _TOKEN_PATH, repo_root / _TOKENIZER_PATH]
    if any(path.exists() for path in destinations):
        raise FileExistsError("training data destination already exists")
    staging = repo_root / "artifacts" / f".bundle-staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        _unpack_archive(archive_path, staging)
        identity = verify_training_data(staging, manifest)
        for relative_path, destination in zip(
            (_TOKEN_PATH, _TOKENIZER_PATH), destinations, strict=True
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative_path).replace(destination)
        return identity
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack or restore frozen training data.")
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--repo-root", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    unpack = commands.add_parser("unpack")
    unpack.add_argument("--repo-root", type=Path, required=True)
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--manifest", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    handlers: dict[str, Callable[[], object]] = {
        "pack": lambda: create_bundle(args.repo_root, args.archive),
        "unpack": lambda: restore_bundle(args.repo_root, args.archive, args.manifest),
        "verify": lambda: verify_training_data(args.repo_root, _read_json(args.manifest)),
    }
    result = handlers[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
