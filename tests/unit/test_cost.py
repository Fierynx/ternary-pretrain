from __future__ import annotations

import json
from pathlib import Path

import pytest

from ternary_pretrain.cost import cost_report


def test_cost_report_uses_provided_bills_and_recorded_tokens_exactly(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "successful_tokens": 1_000_000_000,
                "final_metrics": {"validation_nll": 1.0},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "cost.toml"
    config.write_text(
        """summary_file = "summary.json"
output_file = "report.json"
compute_usd = 1.0
storage_usd = 1.0
other_usd = 1.0
target_loss = 2.0
""",
        encoding="utf-8",
    )
    report = cost_report(config)
    assert report["total_usd"] == 3.0
    assert report["usd_per_billion_successful_tokens"] == 3.0
    assert report["usd_to_target_loss"] == 3.0
    assert (tmp_path / "report.json").is_file()


def test_cost_report_rejects_negative_billing(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text('{"successful_tokens": 1}', encoding="utf-8")
    config = tmp_path / "cost.toml"
    config.write_text(
        'summary_file="summary.json"\noutput_file="out.json"\ncompute_usd=-1\n'
        "storage_usd=0\nother_usd=0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="negative"):
        cost_report(config)
