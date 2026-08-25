from __future__ import annotations

import statistics

import torch


class CudaStepTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.pending: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def start(self) -> torch.cuda.Event | None:
        if self.device.type != "cuda":
            return None
        event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        event.record()
        return event

    def stop(self, start: torch.cuda.Event | None) -> None:
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end.record()
        self.pending.append((start, end))

    def metrics(self, tokens_per_step: int) -> dict[str, float | int]:
        if not self.pending:
            return {}
        self.pending[-1][1].synchronize()
        durations = [start.elapsed_time(end) / 1000 for start, end in self.pending]
        total_seconds = sum(durations)
        sorted_durations = sorted(durations)
        p95_index = min(len(sorted_durations) - 1, int(0.95 * len(sorted_durations)))
        metrics: dict[str, float | int] = {
            "performance/step_seconds_median": statistics.median(durations),
            "performance/step_seconds_p95": sorted_durations[p95_index],
            "performance/tokens_per_second": tokens_per_step * len(durations) / total_seconds,
            "performance/peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "performance/peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }
        self.pending.clear()
        return metrics
