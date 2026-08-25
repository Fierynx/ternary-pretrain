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
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    if device_kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def finalize(context: DistributedContext) -> None:
    if context.initialized_here:
        # Only close a process group created here.
        dist.barrier()
        dist.destroy_process_group()
