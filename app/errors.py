from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging_utils import StructuredLogger, get_logger


@dataclass
class AppError(Exception):
    detail: str
    status_code: int = 400
    error_code: str = "app_error"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {"detail": self.detail, "error_code": self.error_code}
        if self.extra:
            payload["meta"] = self.extra
        return payload


class NotFoundError(AppError):
    def __init__(self, detail: str = "Resource not found.", *, error_code: str = "not_found") -> None:
        super().__init__(detail=detail, status_code=404, error_code=error_code)


class ValidationAppError(AppError):
    def __init__(self, detail: str, *, error_code: str = "validation_error") -> None:
        super().__init__(detail=detail, status_code=422, error_code=error_code)


class ConfigurationError(AppError):
    def __init__(self, detail: str, *, error_code: str = "configuration_error") -> None:
        super().__init__(detail=detail, status_code=503, error_code=error_code)


class UpstreamTimeoutError(AppError):
    def __init__(self, detail: str, *, error_code: str = "upstream_timeout") -> None:
        super().__init__(detail=detail, status_code=504, error_code=error_code)


class UpstreamServiceError(AppError):
    def __init__(self, detail: str, *, error_code: str = "upstream_service_error") -> None:
        super().__init__(detail=detail, status_code=502, error_code=error_code)


class ConflictError(AppError):
    def __init__(self, detail: str, *, error_code: str = "conflict") -> None:
        super().__init__(detail=detail, status_code=409, error_code=error_code)


class UnauthorizedAppError(AppError):
    def __init__(self, detail: str, *, error_code: str = "unauthorized") -> None:
        super().__init__(detail=detail, status_code=401, error_code=error_code)


def register_exception_handlers(app: FastAPI, *, logger: StructuredLogger | None = None) -> None:
    active_logger = logger or get_logger("tutoring_module_api")

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        active_logger.warning(
            "request.app_error",
            status_code=exc.status_code,
            error_code=exc.error_code,
            detail=exc.detail,
            meta=exc.extra,
        )
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        active_logger.exception("request.unhandled_exception", error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong. Please try again.", "error_code": "internal_error"},
        )
