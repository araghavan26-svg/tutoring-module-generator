from __future__ import annotations

import time
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import ensure_openai_api_key
from .models import (
    DocsUploadItem,
    DocsUploadResponse,
    ModuleGenerateRequest,
    ModuleGenerateResponse,
    SectionRegenerateRequest,
    SectionRegenerateResponse,
    utc_now,
)
from .openai_service import (
    build_evidence_pack,
    build_unverified_module,
    enforce_quality_gate,
    generate_module_from_evidence,
    generate_section_from_evidence,
    get_openai_client,
    merge_evidence_pack,
    next_evidence_index,
    objective_for_section,
    poll_vector_file_status,
    resolve_section_index,
)
from .store import module_store


app = FastAPI(
    title="Grounded Tutoring Module API",
    version="1.0.0",
    description=(
        "Two-step tutoring-module generator: (1) retrieve evidence with tools, "
        "(2) generate module from evidence with tools disabled."
    ),
)


@app.on_event("startup")
def enforce_byok_startup_check() -> None:
    ensure_openai_api_key()


@app.get("/health")
def health() -> dict:
    return {"status": "OK"}


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/docs/upload", response_model=DocsUploadResponse)
async def upload_docs(files: List[UploadFile] = File(...)) -> DocsUploadResponse:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file must be uploaded.")

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    vector_store = client.vector_stores.create(name=f"tutoring-modules-{int(time.time())}")
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
            )
            vector_file = client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_obj.id,
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
        except Exception as exc:  # pragma: no cover - runtime network branch
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
        raise HTTPException(status_code=502, detail="No files were uploaded/indexed successfully.")

    return DocsUploadResponse(vector_store_id=vector_store_id, docs=items)


@app.post("/v1/modules/generate", response_model=ModuleGenerateResponse)
def generate_module(payload: ModuleGenerateRequest) -> ModuleGenerateResponse:
    module_id = str(uuid4())

    if not payload.vector_store_id and not payload.allow_web:
        module = build_unverified_module(
            module_id=module_id,
            request=payload,
            reason="No retrieval source configured (vector_store_id missing and allow_web=false).",
        )
        module = enforce_quality_gate(module, evidence_pack=[])
        module_store.save(module_id, payload, module, [])
        return ModuleGenerateResponse(module=module, evidence_pack=[])

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        evidence_pack = build_evidence_pack(
            client,
            topic=payload.topic,
            audience_level=payload.audience_level,
            learning_objectives=payload.learning_objectives,
            allow_web=payload.allow_web,
            vector_store_id=payload.vector_store_id,
            start_index=1,
        )
    except Exception as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=502, detail=f"Evidence retrieval failed: {type(exc).__name__}: {exc}") from exc

    try:
        module = generate_module_from_evidence(
            client,
            request=payload,
            evidence_pack=evidence_pack,
            module_id=module_id,
        )
    except Exception as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=502, detail=f"Module generation failed: {type(exc).__name__}: {exc}") from exc

    try:
        module = enforce_quality_gate(module, evidence_pack=evidence_pack)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    module_store.save(module_id, payload, module, evidence_pack)
    return ModuleGenerateResponse(module=module, evidence_pack=evidence_pack)


@app.post("/v1/modules/{module_id}/regenerate", response_model=SectionRegenerateResponse)
def regenerate_section(module_id: str, payload: SectionRegenerateRequest) -> SectionRegenerateResponse:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")

    try:
        section_index = resolve_section_index(record.module, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target_section = record.module.sections[section_index]
    effective_allow_web = record.request.allow_web if payload.allow_web is None else payload.allow_web
    effective_vector_store_id = payload.vector_store_id or record.request.vector_store_id
    effective_objective = (
        payload.learning_objective
        or objective_for_section(record.request, section_index, fallback_heading=target_section.heading)
    )

    new_evidence = []
    if effective_vector_store_id or effective_allow_web:
        try:
            client = get_openai_client()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            new_evidence = build_evidence_pack(
                client,
                topic=record.request.topic,
                audience_level=record.request.audience_level,
                learning_objectives=[effective_objective],
                allow_web=effective_allow_web,
                vector_store_id=effective_vector_store_id,
                start_index=next_evidence_index(record.evidence_pack),
            )
        except Exception as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(
                status_code=502,
                detail=f"Evidence retrieval failed during regeneration: {type(exc).__name__}: {exc}",
            ) from exc

        try:
            regenerated = generate_section_from_evidence(
                client,
                request=record.request,
                module=record.module,
                target_section=target_section,
                evidence_pack=new_evidence,
                instructions=payload.instructions,
            )
        except Exception as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(
                status_code=502,
                detail=f"Section regeneration failed: {type(exc).__name__}: {exc}",
            ) from exc
    else:
        regenerated = target_section.model_copy(
            update={
                "content": "No retrievable evidence source was configured for this regeneration request.",
                "citations": [],
                "unverified": True,
                "unverified_reason": "No retrieval source configured.",
            }
        )

    if not regenerated.citations and not regenerated.unverified:
        raise HTTPException(
            status_code=422,
            detail=(
                "Quality check failed for regenerated section: "
                "section must include citations or be marked unverified=true."
            ),
        )

    updated_sections = [section.model_copy(deep=True) for section in record.module.sections]
    updated_sections[section_index] = regenerated

    merged_evidence = merge_evidence_pack(record.evidence_pack, new_evidence)
    updated_module = record.module.model_copy(update={"sections": updated_sections})

    try:
        updated_module = enforce_quality_gate(updated_module, evidence_pack=merged_evidence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    updated_request = record.request.model_copy(
        update={
            "allow_web": effective_allow_web,
            "vector_store_id": effective_vector_store_id,
        }
    )
    module_store.save(module_id, updated_request, updated_module, merged_evidence)

    return SectionRegenerateResponse(
        module=updated_module,
        regenerated_section_index=section_index,
        evidence_pack=new_evidence,
    )
