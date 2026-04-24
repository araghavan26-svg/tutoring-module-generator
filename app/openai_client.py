from __future__ import annotations

from threading import Lock

from openai import OpenAI

from .config import ensure_openai_api_key, get_settings


class OpenAIClientService:
    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._lock = Lock()

    def get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                settings = get_settings()
                self._client = OpenAI(
                    api_key=ensure_openai_api_key(),
                    timeout=settings.openai_timeout_seconds,
                    max_retries=0,
                )
        return self._client

    def reset(self) -> None:
        with self._lock:
            self._client = None


openai_client_service = OpenAIClientService()


def get_openai_client() -> OpenAI:
    return openai_client_service.get_client()
