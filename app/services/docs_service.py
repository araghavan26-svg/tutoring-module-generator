from __future__ import annotations

import time
from pathlib import Path
from typing import Awaitable, Callable, List

from fastapi import UploadFile
from openai import APITimeoutError

from ..config import get_settings
from ..errors import ConfigurationError, UpstreamServiceError, UpstreamTimeoutError, ValidationAppError
from ..models import DocsUploadItem, DocsUploadResponse, utc_now
from ..openai_service import MODULE_TIMEOUT_MESSAGE, poll_vector_file_status
from ..openai_client import get_openai_client


async def upload_documents(
    files: List[UploadFile],
    *,
    client_provider: Callable[[], object] = get_openai_client,
) -> DocsUploadResponse:
    if not files:
        raise ValidationAppError("At least one file must be uploaded.")

    try:
        client = client_provider()
    except RuntimeError as exc:
        raise ConfigurationError(str(exc)) from exc

    settings = get_settings()
    try:
        vector_store = client.vector_stores.create(
            name=f"tutoring-modules-{int(time.time())}",
            timeout=settings.upload_timeout_seconds,
        )
    except APITimeoutError as exc:
        raise UpstreamTimeoutError(MODULE_TIMEOUT_MESSAGE) from exc

    vector_store_id = vector_store.id
    allowed_suffixes = {".pdf", ".txt"}
    items: List[DocsUploadItem] = []
    successful_count = 0

    for upload in files:
        filename = upload.filename or "uploaded-document"
        suffix = Path(filename).suffix.lower()
        raw = await upload.read()

        if suffix not in allowed_suffixes:
            items.append(
                DocsUploadItem(
                    file_id="",
                    vector_store_file_id=None,
                    filename=filename,
                    bytes=len(raw),
                    status="failed-unsupported-type",
                    indexed_at=utc_now(),
                )
            )
            continue

        if not raw:
            items.append(
                DocsUploadItem(
                    file_id="",
                    vector_store_file_id=None,
                    filename=filename,
                    bytes=0,
                    status="failed-empty-file",
                    indexed_at=utc_now(),
                )
            )
            continue

        try:
            file_obj = client.files.create(
                file=(filename, raw, upload.content_type or "application/octet-stream"),
                purpose="assistants",
                timeout=settings.upload_timeout_seconds,
            )
            vector_file = client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_obj.id,
                timeout=settings.upload_timeout_seconds,
            )
            vector_store_file_id = getattr(vector_file, "id", None)
            status = str(getattr(vector_file, "status", "in_progress"))
            if vector_store_file_id:
                status = poll_vector_file_status(client, vector_store_id, vector_store_file_id)
            if status not in {"failed", "cancelled"}:
                successful_count += 1
            items.append(
                DocsUploadItem(
                    file_id=file_obj.id,
                    vector_store_file_id=vector_store_file_id,
                    filename=filename,
                    bytes=len(raw),
                    status=status,
                    indexed_at=utc_now(),
                )
            )
        except APITimeoutError:
            items.append(
                DocsUploadItem(
                    file_id="",
                    vector_store_file_id=None,
                    filename=filename,
                    bytes=len(raw),
                    status="failed-timeout",
                    indexed_at=utc_now(),
                )
            )
        except Exception as exc:  # pragma: no cover - network/runtime branch
            items.append(
                DocsUploadItem(
                    file_id="",
                    vector_store_file_id=None,
                    filename=filename,
                    bytes=len(raw),
                    status=f"failed-{type(exc).__name__}",
                    indexed_at=utc_now(),
                )
            )

    if successful_count == 0:
        raise UpstreamServiceError("No files were uploaded/indexed successfully.")

    return DocsUploadResponse(vector_store_id=vector_store_id, docs=items)
