from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from ternary_pretrain.training.manifest import write_json_atomic

CostReportValue = float | int | bool | str | None


def cost_report(config_path: Path) -> dict[str, CostReportValue]:
    """Calculate cost efficiency without deriving any marketplace hourly estimate."""
    with config_path.open("rb") as stream:
        billing_config: dict[str, Any] = tomllib.load(stream)
    allowed_keys = {
        "summary_file",
        "output_file",
        "compute_usd",
        "storage_usd",
        "other_usd",
        "target_loss",
    }
    unknown_keys = set(billing_config) - allowed_keys
    missing_keys = allowed_keys - {"target_loss"} - set(billing_config)
    if unknown_keys or missing_keys:
        errors = []
        if unknown_keys:
            errors.append(f"unknown: {', '.join(sorted(unknown_keys))}")
        if missing_keys:
            errors.append(f"missing: {', '.join(sorted(missing_keys))}")
        raise ValueError("invalid cost config (" + "; ".join(errors) + ")")
    config_dir = config_path.resolve().parent
    summary_path = (config_dir / str(billing_config["summary_file"])).resolve()
    output_path = (config_dir / str(billing_config["output_file"])).resolve()
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    successful_tokens = int(run_summary["successful_tokens"])
    if successful_tokens <= 0:
        raise ValueError("summary has no successfully processed tokens")
    # Use the final billed amounts, not estimates from hourly prices.
    billed_usd = {
        "compute_usd": float(billing_config["compute_usd"]),
        "storage_usd": float(billing_config["storage_usd"]),
        "other_usd": float(billing_config["other_usd"]),
    }
    if any(value < 0 for value in billed_usd.values()):
        raise ValueError("billing values cannot be negative")
    total_usd = sum(billed_usd.values())
    target_loss = billing_config.get("target_loss")
    target_reached: bool | None = None
    usd_to_target_loss: float | None = None
    if target_loss is not None:
        final_loss = run_summary.get("final_metrics", {}).get("validation_nll")
        target_reached = final_loss is not None and float(final_loss) <= float(target_loss)
        usd_to_target_loss = total_usd if target_reached else None
    report: dict[str, CostReportValue] = {
        **billed_usd,
        "total_usd": total_usd,
        "successful_tokens": successful_tokens,
        "usd_per_billion_successful_tokens": total_usd * 1_000_000_000 / successful_tokens,
        "target_loss": None if target_loss is None else float(target_loss),
        "target_loss_reached": target_reached,
        "usd_to_target_loss": usd_to_target_loss,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, report)
    return report
