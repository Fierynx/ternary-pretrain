from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize(device_kind: str, backend: str) -> DistributedContext:
    # torchrun provides these values. Missing values mean a local one-process run.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if device_kind == "cuda":
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in {None, ":4096:8"}:
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8' for CUDA runs")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def finalize(context: DistributedContext) -> None:
    if context.initialized_here:
        # Do not add a final barrier; one failed rank must not hold the others open.
        dist.destroy_process_group()
