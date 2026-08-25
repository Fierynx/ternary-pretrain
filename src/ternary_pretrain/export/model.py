from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from safetensors.torch import save_model

from ternary_pretrain.config import RunConfig, canonical_data, file_sha256
from ternary_pretrain.evaluation.runner import load_native_checkpoint


def _copy_tokenizer(config: RunConfig, output: Path) -> None:
    shutil.copy2(config.tokenizer_file, output / "tokenizer.json")
    metadata = config.tokenizer_file.with_suffix(".metadata.json")
    shutil.copy2(metadata, output / "tokenizer.metadata.json")


def export_checkpoint(
    config: RunConfig,
    checkpoint: Path,
    *,
    export_format: Literal["native", "huggingface"],
) -> Path:
    """Export latent model weights after strict checkpoint validation."""
    model, payload = load_native_checkpoint(config, checkpoint)
    step = int(payload["completed_steps"])
    output = checkpoint.parent.parent / "exports" / f"{export_format}-step-{step:08d}"
    output.mkdir(parents=True, exist_ok=False)
    _copy_tokenizer(config, output)
    if export_format == "native":
        # Export full weights so the model can continue training later.
        save_model(model, str(output / "model.safetensors"))
        metadata = {
            "format_version": 1,
            "format": "ternary-pretrain-native",
            "model": canonical_data(model.config),
            "quantization": {
                "scheme": "w1.58a16",
                "latent_weights": True,
                "qat_enabled": model.qat_enabled,
            },
            "checkpoint_step": step,
            "weights_sha256": file_sha256(output / "model.safetensors"),
        }
        (output / "model.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        try:
            from ternary_pretrain.integrations.transformers import TernaryPreTrainedModel
        except ImportError as error:
            raise RuntimeError("Hugging Face export requires the 'transformers' extra") from error
        wrapper = TernaryPreTrainedModel.from_native(model)
        wrapper.save_pretrained(output, safe_serialization=True)
    return output
