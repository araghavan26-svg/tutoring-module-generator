from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Response, UploadFile

from .config import ensure_openai_api_key
from .models import (
    DocsUploadItem,
    DocsUploadResponse,
    EvidenceItem,
    ModuleGenerateRequest,
    ModuleGenerateResponse,
    RefreshSourcesRequest,
    RefreshSourcesResponse,
    SectionRegenerateByIdRequest,
    SectionRegenerateRequest,
    SectionRegenerateResponse,
    SourcePolicy,
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


def _build_module_footnotes(module_data: Dict[str, Any], evidence_pack: List[Any]) -> List[Dict[str, Any]]:
    evidence_by_id: Dict[str, Any] = {}
    for evidence in evidence_pack:
        evidence_id = getattr(evidence, "evidence_id", "")
        if evidence_id:
            evidence_by_id[evidence_id] = evidence

    footnotes: List[Dict[str, Any]] = []
    seen = set()
    for section in module_data.get("sections", []):
        citations = section.get("citations", []) if isinstance(section, dict) else []
        if not isinstance(citations, list):
            continue
        for citation_id in citations:
            citation_key = str(citation_id).strip()
            if not citation_key or citation_key in seen:
                continue
            seen.add(citation_key)
            evidence = evidence_by_id.get(citation_key)
            if not evidence:
                continue
            footnotes.append(
                {
                    "footnote_id": citation_key,
                    "source_type": evidence.source_type,
                    "domain": evidence.domain,
                    "title": evidence.title,
                    "url": evidence.url,
                    "doc_name": evidence.doc_name,
                    "location": evidence.location,
                    "snippet": evidence.snippet,
                    "retrieved_at": evidence.retrieved_at,
                }
            )
    return footnotes


def _module_export_payload(module_data: Dict[str, Any], evidence_pack: List[Any]) -> Dict[str, Any]:
    return {
        "module_id": module_data.get("module_id"),
        "title": module_data.get("title"),
        "overview": module_data.get("overview", ""),
        "sections": module_data.get("sections", []),
        "glossary": module_data.get("glossary", []),
        "mcqs": module_data.get("mcqs", []),
        "footnotes": _build_module_footnotes(module_data, evidence_pack),
    }


def _module_export_markdown(module_data: Dict[str, Any], footnotes: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    title = str(module_data.get("title", "Untitled module")).strip() or "Untitled module"
    overview = str(module_data.get("overview", "")).strip() or "No overview provided."
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overview")
    lines.append(overview)
    lines.append("")

    lines.append("## Sections")
    sections = module_data.get("sections", [])
    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading", f"Section {idx + 1}")).strip() or f"Section {idx + 1}"
        learning_goal = str(section.get("learning_goal", "")).strip() or "No learning goal provided."
        content = str(section.get("content", "")).strip() or "No content provided."
        lines.append(f"### {idx + 1}. {heading}")
        lines.append(f"Learning Goal: {learning_goal}")
        lines.append("")
        lines.append(content)
        citations = section.get("citations", [])
        if isinstance(citations, list) and citations:
            refs = ", ".join(f"[^{str(item).strip()}]" for item in citations if str(item).strip())
            lines.append("")
            lines.append(f"Citations: {refs}")
        lines.append("")

    lines.append("## Glossary")
    glossary = module_data.get("glossary", [])
    if isinstance(glossary, list) and glossary:
        for item in glossary:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            definition = str(item.get("definition", "")).strip()
            if term and definition:
                lines.append(f"- **{term}**: {definition}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## MCQs")
    mcqs = module_data.get("mcqs", [])
    if isinstance(mcqs, list) and mcqs:
        for idx, mcq in enumerate(mcqs):
            if not isinstance(mcq, dict):
                continue
            question = str(mcq.get("question", f"Question {idx + 1}")).strip()
            lines.append(f"{idx + 1}. {question}")
            options = mcq.get("options", [])
            if isinstance(options, list):
                for option_idx, option in enumerate(options):
                    letter = chr(ord("A") + option_idx)
                    lines.append(f"   - {letter}. {str(option).strip()}")
            answer_index = mcq.get("answer_index")
            explanation = str(mcq.get("explanation", "")).strip()
            lines.append(f"   - Answer index: {answer_index}")
            if explanation:
                lines.append(f"   - Explanation: {explanation}")
            lines.append("")
    else:
        lines.append("- (none)")
        lines.append("")

    lines.append("## Footnotes")
    if footnotes:
        for note in footnotes:
            footnote_id = str(note.get("footnote_id", "")).strip()
            title = str(note.get("title", "Untitled source")).strip() or "Untitled source"
            url = str(note.get("url", "") or "").strip()
            doc_name = str(note.get("doc_name", "") or "").strip()
            location = str(note.get("location", "") or "").strip()
            snippet = str(note.get("snippet", "") or "").strip()
            source_ref = url or " / ".join(part for part in [doc_name, location] if part) or "source"
            lines.append(f"[^{footnote_id}]: {title} ({source_ref}) - {snippet}")
    else:
        lines.append("- (none)")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def _current_source_policy(record: Any) -> SourcePolicy:
    policy = getattr(record, "source_policy", None)
    if policy is not None:
        return policy
    if record.module.source_policy is not None:
        return record.module.source_policy
    return record.request.effective_source_policy()


def _current_evidence_pack(record: Any) -> List[EvidenceItem]:
    module_cached = list(record.module.evidence_pack or [])
    if module_cached:
        return module_cached
    return list(record.evidence_pack or [])


def _refresh_record_evidence_pack(
    *,
    client: Any,
    record: Any,
    source_policy: SourcePolicy,
) -> List[EvidenceItem]:
    vector_store_id = record.request.vector_store_id
    if not vector_store_id and not source_policy.allow_web:
        return []

    evidence_result = build_evidence_pack(
        client,
        topic=record.request.topic,
        audience_level=record.request.audience_level,
        learning_objectives=record.request.learning_objectives,
        allow_web=source_policy.allow_web,
        vector_store_id=vector_store_id,
        source_policy=source_policy,
        start_index=1,
    )
    return evidence_result.evidence_pack


def _evidence_source_counts(evidence_pack: List[EvidenceItem]) -> Dict[str, int]:
    doc_count = sum(1 for item in evidence_pack if item.source_type == "doc")
    web_count = sum(1 for item in evidence_pack if item.source_type == "web")
    return {
        "evidence_count": len(evidence_pack),
        "doc_count": doc_count,
        "web_count": web_count,
    }


def _normalize_regenerated_section(section: Any, evidence_pack: List[EvidenceItem]) -> Any:
    evidence_ids = {item.evidence_id for item in evidence_pack}
    citations = [str(item).strip() for item in section.citations if str(item).strip()]
    citations = list(dict.fromkeys(citations))
    invalid = [item for item in citations if item not in evidence_ids]
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Regenerated section references unknown evidence ids: {joined}.")
    if not citations and not section.unverified:
        raise ValueError("Regenerated section must include citations or set unverified=true.")

    reason = str(section.unverified_reason or "").strip()
    if section.unverified and not reason:
        reason = "Insufficient retrieved evidence for this section."
    if not section.unverified:
        reason = ""

    return section.model_copy(update={"citations": citations, "unverified_reason": reason})


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
    source_policy = payload.effective_source_policy()
    effective_allow_web = source_policy.allow_web

    if not payload.vector_store_id and not effective_allow_web:
        module = build_unverified_module(
            module_id=module_id,
            request=payload,
            reason="No retrieval source configured (vector_store_id missing and source_policy.allow_web=false).",
        )
        module = enforce_quality_gate(module, evidence_pack=[])
        module_store.save(module_id, payload, module, [], source_policy=source_policy)
        module = module.model_copy(update={"source_policy": source_policy, "evidence_pack": []})
        return ModuleGenerateResponse(module=module, evidence_pack=[])

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        evidence_result = build_evidence_pack(
            client,
            topic=payload.topic,
            audience_level=payload.audience_level,
            learning_objectives=payload.learning_objectives,
            allow_web=effective_allow_web,
            vector_store_id=payload.vector_store_id,
            source_policy=source_policy,
            start_index=1,
        )
        evidence_pack = evidence_result.evidence_pack
    except Exception as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=502, detail=f"Evidence retrieval failed: {type(exc).__name__}: {exc}") from exc

    try:
        module = generate_module_from_evidence(
            client,
            request=payload,
            evidence_pack=evidence_pack,
            module_id=module_id,
            web_unavailable_objectives=evidence_result.web_unavailable_objectives,
            objectives_without_evidence=evidence_result.objectives_without_evidence,
        )
    except Exception as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=502, detail=f"Module generation failed: {type(exc).__name__}: {exc}") from exc

    try:
        module = enforce_quality_gate(module, evidence_pack=evidence_pack)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    module_store.save(module_id, payload, module, evidence_pack, source_policy=source_policy)
    module = module.model_copy(update={"source_policy": source_policy, "evidence_pack": evidence_pack})
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
    base_source_policy = _current_source_policy(record)
    effective_allow_web = base_source_policy.allow_web if payload.allow_web is None else payload.allow_web
    effective_source_policy = base_source_policy.model_copy(update={"allow_web": effective_allow_web})
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
            evidence_result = build_evidence_pack(
                client,
                topic=record.request.topic,
                audience_level=record.request.audience_level,
                learning_objectives=[effective_objective],
                allow_web=effective_allow_web,
                vector_store_id=effective_vector_store_id,
                source_policy=effective_source_policy,
                start_index=next_evidence_index(record.evidence_pack),
            )
            new_evidence = evidence_result.evidence_pack
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
            "source_policy": effective_source_policy,
        }
    )
    module_store.save(
        module_id,
        updated_request,
        updated_module,
        merged_evidence,
        source_policy=effective_source_policy,
    )
    updated_module = updated_module.model_copy(
        update={"source_policy": effective_source_policy, "evidence_pack": merged_evidence}
    )

    return SectionRegenerateResponse(
        module=updated_module,
        regenerated_section_index=section_index,
        evidence_pack=new_evidence,
    )


@app.post("/v1/modules/{module_id}/refresh_sources", response_model=RefreshSourcesResponse)
def refresh_sources(module_id: str, payload: RefreshSourcesRequest) -> RefreshSourcesResponse:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")

    source_policy = payload.source_policy or _current_source_policy(record)
    evidence_pack: List[EvidenceItem] = []

    if record.request.vector_store_id or source_policy.allow_web:
        try:
            client = get_openai_client()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            evidence_pack = _refresh_record_evidence_pack(
                client=client,
                record=record,
                source_policy=source_policy,
            )
        except Exception as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(status_code=502, detail=f"Evidence refresh failed: {type(exc).__name__}: {exc}") from exc

    updated_request = record.request.model_copy(
        update={
            "allow_web": source_policy.allow_web,
            "source_policy": source_policy,
        }
    )
    updated_module = record.module.model_copy(update={"source_policy": source_policy, "evidence_pack": evidence_pack})
    module_store.save(
        module_id,
        updated_request,
        updated_module,
        evidence_pack,
        source_policy=source_policy,
    )
    counts = _evidence_source_counts(evidence_pack)
    return RefreshSourcesResponse(
        module_id=module_id,
        source_policy=source_policy,
        vector_store_id=record.request.vector_store_id,
        **counts,
    )


@app.post("/v1/modules/{module_id}/sections/{section_id}/regenerate", response_model=SectionRegenerateResponse)
def regenerate_section_by_id(
    module_id: str,
    section_id: str,
    payload: SectionRegenerateByIdRequest,
) -> SectionRegenerateResponse:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")

    section_index = next((idx for idx, section in enumerate(record.module.sections) if section.section_id == section_id), -1)
    if section_index < 0:
        raise HTTPException(status_code=404, detail="Section not found.")

    target_section = record.module.sections[section_index]
    source_policy = _current_source_policy(record)
    evidence_pack = _current_evidence_pack(record)
    client: Any = None

    if payload.refresh_sources:
        if record.request.vector_store_id or source_policy.allow_web:
            try:
                client = get_openai_client()
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            try:
                evidence_pack = _refresh_record_evidence_pack(
                    client=client,
                    record=record,
                    source_policy=source_policy,
                )
            except Exception as exc:  # pragma: no cover - runtime network branch
                raise HTTPException(
                    status_code=502,
                    detail=f"Evidence refresh failed during regeneration: {type(exc).__name__}: {exc}",
                ) from exc
        else:
            evidence_pack = []

    if evidence_pack:
        if client is None:
            try:
                client = get_openai_client()
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            regenerated = generate_section_from_evidence(
                client,
                request=record.request,
                module=record.module,
                target_section=target_section,
                evidence_pack=evidence_pack,
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
                "content": "No cached evidence is available for this module.",
                "citations": [],
                "unverified": True,
                "unverified_reason": "No cached evidence available.",
            }
        )

    try:
        regenerated = _normalize_regenerated_section(regenerated, evidence_pack)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    updated_sections = [section.model_copy(deep=True) for section in record.module.sections]
    updated_sections[section_index] = regenerated

    updated_module = record.module.model_copy(
        update={
            "sections": updated_sections,
            "source_policy": source_policy,
            "evidence_pack": evidence_pack,
        }
    )
    updated_request = record.request.model_copy(
        update={
            "allow_web": source_policy.allow_web,
            "source_policy": source_policy,
        }
    )
    module_store.save(
        module_id,
        updated_request,
        updated_module,
        evidence_pack,
        source_policy=source_policy,
    )

    return SectionRegenerateResponse(
        module=updated_module,
        regenerated_section_index=section_index,
        evidence_pack=evidence_pack,
    )


@app.get("/v1/modules/{module_id}/export/json")
def export_module_json(module_id: str) -> Dict[str, Any]:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")
    module_data = record.module.model_dump(mode="json")
    return _module_export_payload(module_data, _current_evidence_pack(record))


@app.get("/v1/modules/{module_id}/export/markdown")
def export_module_markdown(module_id: str) -> Response:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")
    module_data = record.module.model_dump(mode="json")
    payload = _module_export_payload(module_data, _current_evidence_pack(record))
    markdown = _module_export_markdown(module_data, payload["footnotes"])
    return Response(content=markdown, media_type="text/markdown")
