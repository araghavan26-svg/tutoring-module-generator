from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Iterator

from .logging_utils import StructuredLogger


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _emit_log(logger: Any, event: str, **fields: Any) -> None:
    if isinstance(logger, StructuredLogger):
        logger.info(event, **fields)
        return
    payload = {"event": event, **{key: _serialize(value) for key, value in fields.items()}}
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str))


@dataclass
class StageTimingLogger:
    logger: Any
    request_id: str
    request_started_at: float = field(default_factory=time.perf_counter)
    stage_starts: Dict[str, float] = field(default_factory=dict)
    stage_durations: Dict[str, float] = field(default_factory=dict)

    def log_event(self, event: str, **fields: object) -> None:
        payload = {"request_id": self.request_id, **fields}
        _emit_log(self.logger, event, **payload)

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


def log_stage(stage: str, *, fields_factory: Callable[..., Dict[str, Any]] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            stage_timer = kwargs.get("stage_timer")
            if not isinstance(stage_timer, StageTimingLogger):
                return func(*args, **kwargs)
            fields = fields_factory(*args, **kwargs) if fields_factory is not None else {}
            with stage_timer.measure(stage, **fields):
                return func(*args, **kwargs)

        return wrapper

    return decorator
