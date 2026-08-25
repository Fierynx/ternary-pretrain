from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ternary_pretrain.training import train
from tests.testing import ExperimentFixture, write_run_config


@pytest.mark.ddp
@pytest.mark.skipif(sys.platform == "win32", reason="the CI DDP acceptance test runs on Ubuntu")
def test_two_process_cpu_ddp_matches_single_process(
    prepared_experiment: ExperimentFixture, tmp_path: Path
) -> None:
    config = prepared_experiment.run_config
    config = replace(
        config,
        runtime=replace(config.runtime, max_steps=1),
        schedule=replace(config.schedule, stable_steps=1),
    )
    single = train(config)
    ddp_config = tmp_path / "ddp.toml"
    write_run_config(ddp_config, replace(config, name="ddp"))
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "ternary_pretrain",
            "train",
            "--config",
            str(ddp_config),
        ],
        check=True,
        env=environment,
    )
    ddp_run = next(path for path in config.output_dir.iterdir() if path.name.startswith("ddp-"))
    single_state = torch.load(single.checkpoint, map_location="cpu", weights_only=False)["model"]
    ddp_checkpoint = next((ddp_run / "checkpoints").glob("*.pt"))
    ddp_state = torch.load(ddp_checkpoint, map_location="cpu", weights_only=False)["model"]
    for name, expected in single_state.items():
        torch.testing.assert_close(ddp_state[name], expected, rtol=1e-5, atol=1e-6)
