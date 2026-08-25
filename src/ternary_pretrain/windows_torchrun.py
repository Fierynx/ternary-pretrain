from __future__ import annotations

import importlib
from typing import Any


def configure_tcp_store() -> None:
    import torch.distributed as distributed

    # Windows wheels need TCPStore without libuv.
    original_store = distributed.__dict__.setdefault(
        "_ternary_pretrain_original_tcp_store", distributed.__dict__["TCPStore"]
    )

    def windows_tcp_store(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("use_libuv", False)
        return original_store(*args, **kwargs)

    modules = (
        distributed,
        importlib.import_module("torch.distributed.rendezvous"),
        importlib.import_module("torch.distributed.elastic.rendezvous.c10d_rendezvous_backend"),
        importlib.import_module("torch.distributed.elastic.rendezvous.static_tcp_rendezvous"),
    )
    for module in modules:
        # These modules keep their own reference to TCPStore.
        module.__dict__["TCPStore"] = windows_tcp_store
