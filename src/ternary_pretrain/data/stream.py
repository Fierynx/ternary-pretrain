from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ternary_pretrain.config import file_sha256


class MMapTokenStream:
    """Read fixed-length samples from an immutable logical concatenation of shards."""

    def __init__(self, manifest_path: Path, sequence_length: int, seed: int) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.sequence_length = sequence_length
        self.seed = seed
        self._shards: list[np.ndarray] = []
        self._shard_ends: list[int] = []
        # Read shards as one stream without loading them into RAM.
        total = 0
        for shard in self.manifest["shards"]:
            path = self.manifest_path.parent / shard["file"]
            if file_sha256(path) != shard["sha256"]:
                raise ValueError(f"token shard hash mismatch: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype.str != "<u2" or array.ndim != 1:
                raise ValueError(f"invalid token shard dtype or shape: {path}")
            self._shards.append(array)
            total += len(array)
            self._shard_ends.append(total)
        self.token_count = total
        if total <= sequence_length:
            raise ValueError("token stream is shorter than one sample")

    def _read_tokens(self, start: int, count: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        position = start
        remaining = count
        while remaining:
            shard_index = bisect_right(self._shard_ends, position)
            shard_start = 0 if shard_index == 0 else self._shard_ends[shard_index - 1]
            local_start = position - shard_start
            available = len(self._shards[shard_index]) - local_start
            take = min(available, remaining)
            pieces.append(
                np.asarray(
                    self._shards[shard_index][local_start : local_start + take],
                    dtype=np.int64,
                )
            )
            position += take
            remaining -= take
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)

    def sample(self, sample_index: int) -> tuple[Tensor, Tensor]:
        width = self.sequence_length + 1
        valid_starts = self.token_count - width + 1
        seed_offset = self.seed % valid_starts
        # The same sample index always points to the same tokens.
        start = (seed_offset + sample_index * width) % valid_starts
        values = torch.from_numpy(self._read_tokens(start, width))
        return values[:-1], values[1:]

    def batch(
        self,
        *,
        completed_step: int,
        micro_step: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        rank: int,
        world_size: int,
    ) -> tuple[Tensor, Tensor]:
        # Give every rank a different part of the global batch.
        logical_micro_step = completed_step * gradient_accumulation_steps + micro_step
        first = (logical_micro_step * world_size + rank) * batch_size
        samples = [self.sample(first + offset) for offset in range(batch_size)]
        inputs, labels = zip(*samples, strict=True)
        return torch.stack(inputs), torch.stack(labels)

    @property
    def identity(self) -> str:
        return file_sha256(self.manifest_path)

    @property
    def shard_ends(self) -> tuple[int, ...]:
        return tuple(self._shard_ends)
