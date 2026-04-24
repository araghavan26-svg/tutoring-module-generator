from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import get_settings


class StructuredLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def info(self, event: str, **fields: Any) -> None:
        self._logger.info(_build_log_message(event, **fields))

    def warning(self, event: str, **fields: Any) -> None:
        self._logger.warning(_build_log_message(event, **fields))

    def error(self, event: str, **fields: Any) -> None:
        self._logger.error(_build_log_message(event, **fields))

    def exception(self, event: str, **fields: Any) -> None:
        self._logger.exception(_build_log_message(event, **fields))


_logging_configured = False


def configure_logging() -> None:
    global _logging_configured
    if _logging_configured:
        return
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO), format="%(message)s")
    _logging_configured = True


def get_logger(name: str) -> StructuredLogger:
    configure_logging()
    return StructuredLogger(logging.getLogger(name))


def _serialize(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _build_log_message(event: str, **fields: Any) -> str:
    payload = {"event": event, **{key: _serialize(value) for key, value in fields.items()}}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
