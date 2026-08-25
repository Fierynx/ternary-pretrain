from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ternary_pretrain.config import TokenizerConfig, file_sha256


def _texts(paths: tuple[Path, ...]) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    text = value.get("text")
                    if not isinstance(text, str):
                        raise ValueError(f"document in {path} has no text field")
                    yield text


def train_tokenizer(config: TokenizerConfig) -> Path:
    """Train and save a deterministic byte-level BPE tokenizer."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=[config.eod_token],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_texts(config.input_files), trainer=trainer)
    if tokenizer.token_to_id(config.eod_token) is None:
        raise RuntimeError("trained tokenizer does not contain the end-of-document token")
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    if config.output_file.exists():
        raise FileExistsError(f"tokenizer output already exists: {config.output_file}")
    tokenizer.save(str(config.output_file), pretty=True)
    metadata = {
        "format_version": 1,
        "tokenizer_sha256": file_sha256(config.output_file),
        "vocab_size": tokenizer.get_vocab_size(),
        "eod_token": config.eod_token,
        "eod_token_id": tokenizer.token_to_id(config.eod_token),
        "input_sha256": [file_sha256(path) for path in config.input_files],
    }
    config.output_file.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config.output_file


def load_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"tokenizer metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["tokenizer_sha256"] != file_sha256(path):
        raise ValueError("tokenizer content does not match its metadata hash")
    return tokenizer
