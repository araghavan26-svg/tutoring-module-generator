from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator


def _format_log_fields(fields: Dict[str, object]) -> str:
    return " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))


@dataclass
class StageTimingLogger:
    logger: logging.Logger
    request_id: str
    request_started_at: float = field(default_factory=time.perf_counter)
    stage_starts: Dict[str, float] = field(default_factory=dict)
    stage_durations: Dict[str, float] = field(default_factory=dict)

    def log_event(self, event: str, **fields: object) -> None:
        payload = {"request_id": self.request_id, **fields}
        self.logger.info("%s %s", event, _format_log_fields(payload))

    def start(self, stage: str, **fields: object) -> None:
        self.stage_starts[stage] = time.perf_counter()
        self.log_event(f"{stage}.start", stage=stage, **fields)

    def end(self, stage: str, **fields: object) -> float:
        started_at = self.stage_starts.pop(stage, None)
        duration = max(0.0, time.perf_counter() - started_at) if started_at is not None else 0.0
        self.stage_durations[stage] = duration
        self.log_event(f"{stage}.end", stage=stage, duration_ms=round(duration * 1000, 1), **fields)
        return duration

    @contextmanager
    def measure(self, stage: str, **fields: object) -> Iterator[None]:
        self.start(stage, **fields)
        try:
            yield
        finally:
            self.end(stage, **fields)

    def total_duration_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.request_started_at)

    def finish(self, **fields: object) -> float:
        total_duration = self.total_duration_seconds()
        self.log_event("generation.total", total_duration_ms=round(total_duration * 1000, 1), **fields)
        return total_duration
