from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ternary_pretrain.config import DataConfig, RunConfig, canonical_data


@dataclass(frozen=True, slots=True)
class ExperimentFixture:
    run_config: RunConfig
    data_config: DataConfig


def write_run_config(path: Path, config: RunConfig) -> None:
    document = canonical_data(config)
    if not isinstance(document, dict):
        raise TypeError("run configuration did not serialize to a table")
    section_names = ("runtime", "optimizer", "schedule", "quantization", "tracking")
    lines = [
        f"{key} = {_toml_value(value)}"
        for key, value in document.items()
        if key not in section_names
    ]
    for section_name in section_names:
        section = document[section_name]
        if not isinstance(section, dict):
            raise TypeError(f"{section_name} did not serialize to a table")
        lines.append(f"\n[{section_name}]")
        lines.extend(
            f"{key} = {_toml_value(value)}" for key, value in section.items() if value is not None
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"unsupported TOML value: {value!r}")
