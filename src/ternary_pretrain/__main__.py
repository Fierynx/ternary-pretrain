import os
import sys

from ternary_pretrain.cli import main

if __name__ == "__main__":
    if sys.platform == "win32" and os.environ.get("TERNARY_PRETRAIN_WINDOWS_TORCHRUN") == "1":
        from ternary_pretrain.windows_torchrun import configure_tcp_store

        configure_tcp_store()
    main()
