from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path

from ternary_pretrain.config import ConfigError


def _absolute_path(value: str) -> Path:
    return Path(value).resolve()


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True, type=_absolute_path, help="explicit TOML configuration"
    )


def _run_data_prepare(args: argparse.Namespace) -> object:
    # Import command dependencies only when that command runs.
    from ternary_pretrain.config import load_data_config
    from ternary_pretrain.data.prepare import prepare_data

    return prepare_data(load_data_config(args.config, operation="prepare"))


def _run_data_tokenize(args: argparse.Namespace) -> object:
    from ternary_pretrain.config import load_data_config
    from ternary_pretrain.data.tokenize import tokenize_data

    return tokenize_data(load_data_config(args.config, operation="tokenize"))


def _run_tokenizer_train(args: argparse.Namespace) -> object:
    from ternary_pretrain.config import load_tokenizer_config
    from ternary_pretrain.data.tokenizer import train_tokenizer

    return train_tokenizer(load_tokenizer_config(args.config))


def _run_training(args: argparse.Namespace) -> object:
    from ternary_pretrain.config import load_run_config
    from ternary_pretrain.training import train

    return train(load_run_config(args.config), resume=args.resume)


def _run_evaluation(args: argparse.Namespace) -> object:
    from ternary_pretrain.config import load_run_config
    from ternary_pretrain.evaluation.runner import evaluate_checkpoint

    return evaluate_checkpoint(
        load_run_config(args.config), args.checkpoint, max_batches=args.max_batches
    )


def _run_export(args: argparse.Namespace) -> object:
    from ternary_pretrain.config import load_run_config
    from ternary_pretrain.export import export_checkpoint

    return export_checkpoint(
        load_run_config(args.config),
        args.checkpoint,
        export_format=args.export_format,
    )


def _run_inspection(args: argparse.Namespace) -> object:
    from ternary_pretrain.config import load_model_config
    from ternary_pretrain.model import DecoderLM

    model = DecoderLM(load_model_config(args.config))
    parameters = {id(parameter): parameter for parameter in model.parameters()}
    return {
        "config": str(args.config),
        "parameters": sum(parameter.numel() for parameter in parameters.values()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters.values() if parameter.requires_grad
        ),
        "hidden_matrix_parameters": sum(
            parameter.numel() for parameter in model.hidden_matrix_parameters()
        ),
        "weight_tied": model.lm_head.weight is model.token_embedding.weight,
    }


def _run_cost_report(args: argparse.Namespace) -> object:
    from ternary_pretrain.cost import cost_report

    return cost_report(args.config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ternary-pretrain",
        description="Prepare data and run deterministic language-model pre-training.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data_parser = commands.add_parser("data", help="prepare or tokenize corpus data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    prepare_parser = data_commands.add_parser("prepare", help="prepare document splits")
    _add_config_argument(prepare_parser)
    prepare_parser.set_defaults(handler=_run_data_prepare)
    tokenize_parser = data_commands.add_parser("tokenize", help="write uint16 token shards")
    _add_config_argument(tokenize_parser)
    tokenize_parser.set_defaults(handler=_run_data_tokenize)

    tokenizer_parser = commands.add_parser("tokenizer", help="tokenizer operations")
    tokenizer_commands = tokenizer_parser.add_subparsers(dest="tokenizer_command", required=True)
    tokenizer_train_parser = tokenizer_commands.add_parser("train", help="train byte-level BPE")
    _add_config_argument(tokenizer_train_parser)
    tokenizer_train_parser.set_defaults(handler=_run_tokenizer_train)

    train_parser = commands.add_parser("train", help="run or resume pre-training")
    _add_config_argument(train_parser)
    train_parser.add_argument("--resume", type=_absolute_path, help="explicit checkpoint to resume")
    train_parser.set_defaults(handler=_run_training)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate a checkpoint")
    _add_config_argument(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=_absolute_path, required=True)
    evaluate_parser.add_argument("--max-batches", type=int, default=8)
    evaluate_parser.set_defaults(handler=_run_evaluation)

    export_parser = commands.add_parser("export", help="export a validated checkpoint")
    _add_config_argument(export_parser)
    export_parser.add_argument("--checkpoint", type=_absolute_path, required=True)
    export_parser.add_argument(
        "--format",
        dest="export_format",
        choices=("native", "huggingface"),
        required=True,
    )
    export_parser.set_defaults(handler=_run_export)

    inspect_parser = commands.add_parser("inspect", help="inspect a model configuration")
    _add_config_argument(inspect_parser)
    inspect_parser.set_defaults(handler=_run_inspection)

    cost_parser = commands.add_parser("cost", help="billing-based cost operations")
    cost_commands = cost_parser.add_subparsers(dest="cost_command", required=True)
    cost_report_parser = cost_commands.add_parser(
        "report", help="report actual billed cost efficiency"
    )
    _add_config_argument(cost_report_parser)
    cost_report_parser.set_defaults(handler=_run_cost_report)
    return parser


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")


def _print_result(result: object) -> None:
    print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], object] = args.handler
    try:
        result = handler(args)
        if int(os.environ.get("RANK", "0")) == 0:
            _print_result(result)
    except (ConfigError, FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
