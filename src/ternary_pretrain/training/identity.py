from __future__ import annotations

from ternary_pretrain.config import RunConfig, config_hash, file_sha256


def checkpoint_identity(config: RunConfig) -> dict[str, str]:
    return {
        "config_sha256": config_hash(config),
        "model_config_sha256": file_sha256(config.model_config),
        "tokenizer_sha256": file_sha256(config.tokenizer_file),
        "train_manifest_sha256": file_sha256(config.train_manifest),
        "validation_manifest_sha256": file_sha256(config.validation_manifest),
    }
