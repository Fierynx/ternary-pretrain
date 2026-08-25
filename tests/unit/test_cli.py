from __future__ import annotations

import subprocess
import sys

import pytest

from ternary_pretrain.cli import build_parser


def test_cli_help_and_invalid_command(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["--help"])
    assert help_exit.value.code == 0
    assert "data" in capsys.readouterr().out
    with pytest.raises(SystemExit) as invalid_exit:
        parser.parse_args(["invalid"])
    assert invalid_exit.value.code == 2


def test_building_cli_help_does_not_import_training_dependencies() -> None:
    command = (
        "import sys; from ternary_pretrain.cli import build_parser; "
        "build_parser(); assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


@pytest.mark.parametrize(
    "arguments",
    [
        ["data", "prepare"],
        ["data", "tokenize"],
        ["tokenizer", "train"],
        ["train"],
        ["evaluate"],
        ["export"],
        ["inspect"],
        ["cost", "report"],
    ],
)
def test_every_command_requires_an_explicit_config(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)
    assert error.value.code == 2
