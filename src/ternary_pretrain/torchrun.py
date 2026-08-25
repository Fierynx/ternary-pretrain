from __future__ import annotations

import os
import sys


def main() -> None:
    if sys.platform == "win32":
        # Use only signals available on Windows.
        sys.argv.insert(1, "--signals-to-handle=SIGTERM,SIGINT")

        from ternary_pretrain.windows_torchrun import configure_tcp_store

        configure_tcp_store()
        os.environ["TERNARY_PRETRAIN_WINDOWS_TORCHRUN"] = "1"

    from torch.distributed.run import main as torchrun_main

    torchrun_main()


if __name__ == "__main__":
    main()
