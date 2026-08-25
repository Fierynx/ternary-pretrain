from __future__ import annotations

import math

import torch
from torch import nn

from ternary_pretrain.data import MMapTokenStream


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    stream: MMapTokenStream,
    *,
    device: torch.device,
    batch_size: int,
    max_batches: int,
) -> dict[str, float | int]:
    """Compute fixed-offset validation NLL, perplexity, and corpus-normalized BPB."""
    model.eval()
    total_nll = 0.0
    token_count = 0
    for batch_index in range(max_batches):
        samples = [stream.sample(batch_index * batch_size + offset) for offset in range(batch_size)]
        inputs = torch.stack([sample[0] for sample in samples]).to(device)
        labels = torch.stack([sample[1] for sample in samples]).to(device)
        output = model(inputs, labels=labels)
        if output.loss is None:
            raise RuntimeError("evaluation model did not return loss")
        count = labels.numel()
        total_nll += float(output.loss) * count
        token_count += count
    mean_nll = total_nll / token_count
    byte_count = int(stream.manifest.get("byte_count", 0))
    source_tokens = int(stream.manifest["token_count"])
    bits_per_byte = math.nan
    if byte_count:
        # Use full-corpus token and byte counts for BPB.
        bits_per_byte = mean_nll * source_tokens / (math.log(2.0) * byte_count)
    # Training may continue after this evaluation.
    model.train()
    return {
        "validation_nll": mean_nll,
        "validation_perplexity": math.exp(min(mean_nll, 80.0)),
        "validation_bits_per_byte": bits_per_byte,
        "validation_tokens": token_count,
    }
