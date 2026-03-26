from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from openai import APITimeoutError
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import ensure_openai_api_key, settings
from .models import (
    DocsUploadItem,
    DocsUploadResponse,
    EvidenceItem,
    ModuleAssignmentResponse,
    ModuleAskRequest,
    ModuleAskResponse,
    ModuleDeleteResponse,
    ModuleGradeRequest,
    ModuleGradeResponse,
    ModuleLibraryItem,
    Module,
    ModuleGenerateRequest,
    ModuleGenerateResponse,
    ModuleShareRequest,
    ModuleShareResponse,
    ModuleVersionSummary,
    RefreshSourcesRequest,
    RefreshSourcesResponse,
    SectionRegenerateByIdRequest,
    SectionRegenerateRequest,
    SectionRegenerateResponse,
    SourcePolicy,
    TopicContinuityContext,
    utc_now,
)
from .openai_service import (
    MODULE_TIMEOUT_MESSAGE,
    OpenAIOperationTimeoutError,
    answer_question_from_module,
    build_topic_bridge_context,
    build_evidence_pack,
    build_unverified_module,
    detect_topic_relatedness,
    enforce_quality_gate,
    generate_assignment_from_module,
    generate_module_from_evidence,
    generate_section_from_evidence,
    get_openai_client,
    grade_assignment_from_module,
    human_readable_snippet,
    merge_evidence_pack,
    next_evidence_index,
    objective_for_section,
    poll_vector_file_status,
    resolve_section_index,
)
from .stage_timing import StageTimingLogger
from .store import module_store


logger = logging.getLogger("tutoring_module_api")
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


app = FastAPI(
    title="Grounded Tutoring Module API",
    version="1.0.0",
    description=(
        "Two-step tutoring-module generator: (1) retrieve evidence with tools, "
        "(2) generate module from evidence with tools disabled."
    ),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
SAMPLES_DIR = PROJECT_ROOT / "samples"
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


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
                    "snippet": human_readable_snippet(evidence.snippet),
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


def _share_url(request: Request, share_id: str | None) -> str | None:
    share_key = str(share_id or "").strip()
    if not share_key:
        return None
    return str(request.url_for("shared_module_page", share_id=share_key))


def _footnote_markdown(note: Dict[str, Any]) -> str:
    footnote_id = str(note.get("footnote_id", "")).strip()
    title = str(note.get("title", "Untitled source")).strip() or "Untitled source"
    url = str(note.get("url", "") or "").strip()
    domain = str(note.get("domain", "") or "").strip()
    doc_name = str(note.get("doc_name", "") or "").strip()
    location = str(note.get("location", "") or "").strip()
    snippet = str(note.get("snippet", "") or "").strip()

    if url:
        label = f"[{title}]({url})"
        meta = f"Web source: {domain or 'external source'}"
    else:
        label = f"**{doc_name or title}**"
        meta_parts = ["Document source"]
        if location:
            meta_parts.append(location)
        meta = " | ".join(meta_parts)

    lines = [f"[^{footnote_id}]: {label}"]
    lines.append(f"    - {meta}")
    if snippet:
        lines.append(f"    - Snippet: {snippet}")
    return "\n".join(lines)


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
            lines.append(_footnote_markdown(note))
    else:
        lines.append("- (none)")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def _assignment_export_markdown(assignment: ModuleAssignmentResponse) -> str:
    lines: List[str] = []
    lines.append("# Assignment")
    lines.append("")
    lines.append("## Prompt")
    lines.append(assignment.prompt)
    lines.append("")
    lines.append("## Rubric")
    for index, criterion in enumerate(assignment.rubric, start=1):
        lines.append(f"### {index}. {criterion.criteria}")
        for level in criterion.levels:
            lines.append(f"- {level.score}: {level.description}")
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


def _current_continuity_context(record: Any) -> TopicContinuityContext:
    return TopicContinuityContext(
        topic=record.request.topic,
        learning_objectives=list(record.request.learning_objectives or []),
        module_summary=str(record.module.overview or "").strip(),
    )


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
        subject=record.request.subject,
        topic=record.request.topic,
        audience_level=record.request.audience_level,
        learning_objectives=record.request.learning_objectives,
        allow_web=source_policy.allow_web,
        vector_store_id=vector_store_id,
        source_policy=source_policy,
        start_index=1,
        fast_mode=record.request.fast_mode,
        source_preference=record.request.source_preference or "",
        prefer_high_trust_sources=record.request.prefer_high_trust_sources,
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


def _format_timestamp_for_ui(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%b %d, %Y %I:%M %p")
    raw = str(value or "").strip()
    return raw or "Unknown time"


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
    module_store.load_from_disk()


@app.get("/")
def landing_page(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "ModuleForge",
        },
    )


@app.get("/create")
def create_page(request: Request) -> Response:
    sample_doc_path = SAMPLES_DIR / "sample.txt"
    latest_record = module_store.latest()
    return templates.TemplateResponse(
        request,
        "create.html",
        {
            "title": "Create Module",
            "sample_doc_exists": sample_doc_path.exists(),
            "has_previous_module": latest_record is not None,
            "previous_topic": latest_record.request.topic if latest_record is not None else "",
        },
    )


@app.get("/modules")
def modules_dashboard_page(request: Request) -> Response:
    modules = module_store.list_modules()
    module_rows = []
    for item in modules:
        row = item.model_dump(mode="json")
        row["created_label"] = _format_timestamp_for_ui(item.created_at)
        row["updated_label"] = _format_timestamp_for_ui(item.updated_at)
        row["share_url"] = _share_url(request, item.share_id) if item.share_enabled else None
        module_rows.append(row)
    return templates.TemplateResponse(
        request,
        "modules_dashboard.html",
        {
            "title": "Saved Modules",
            "modules": module_rows,
        },
    )


@app.get("/modules/{module_id}")
def module_editor_page(request: Request, module_id: str) -> Response:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")

    module_data = record.module.model_dump(mode="json")
    module_data["evidence_pack"] = [item.model_dump(mode="json") for item in _current_evidence_pack(record)]
    module_data["source_policy"] = _current_source_policy(record).model_dump(mode="json")
    module_data["share_url"] = _share_url(request, record.module.share_id) if record.module.share_enabled else None
    return templates.TemplateResponse(
        request,
        "module_editor.html",
        {
            "title": "Module Editor",
            "module_id": module_id,
            "initial_module": module_data,
            "initial_history": [item.model_dump(mode="json") for item in module_store.history(module_id)],
        },
    )


@app.get("/shared/{share_id}")
def shared_module_page(request: Request, share_id: str) -> Response:
    record = module_store.get_by_share_id(share_id)
    if not record:
        return templates.TemplateResponse(
            request,
            "shared_unavailable.html",
            {
                "title": "Shared Module Unavailable",
            },
            status_code=404,
        )

    module_data = record.module.model_dump(mode="json")
    footnotes = _build_module_footnotes(module_data, _current_evidence_pack(record))
    footnotes_by_id = {item["footnote_id"]: item for item in footnotes}
    return templates.TemplateResponse(
        request,
        "shared_module.html",
        {
            "title": record.module.title,
            "module": module_data,
            "footnotes": footnotes,
            "footnotes_by_id": footnotes_by_id,
        },
    )


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

    try:
        vector_store = client.vector_stores.create(
            name=f"tutoring-modules-{int(time.time())}",
            timeout=settings.upload_timeout_seconds,
        )
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail=MODULE_TIMEOUT_MESSAGE) from exc
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
        except APITimeoutError:  # pragma: no cover - runtime network branch
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
    previous_record = module_store.latest()
    topic_bridge = None
    stage_timer = StageTimingLogger(logger=logger, request_id=module_id)
    stage_timer.log_event(
        "generation.request.start",
        topic=payload.topic,
        objective_count=len(payload.learning_objectives),
        fast_mode=payload.fast_mode,
    )

    try:
        if not payload.vector_store_id and not effective_allow_web:
            module = build_unverified_module(
                module_id=module_id,
                request=payload,
                reason="No retrieval source configured (vector_store_id missing and source_policy.allow_web=false).",
            )
            module = enforce_quality_gate(module, evidence_pack=[])
            module_store.save(
                module_id,
                payload,
                module,
                [],
                source_policy=source_policy,
                action="generated",
            )
            module = module.model_copy(update={"source_policy": source_policy, "evidence_pack": []})
            return ModuleGenerateResponse(module=module, evidence_pack=[])

        try:
            client = get_openai_client()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            if payload.related_to_previous and previous_record is not None:
                relatedness = detect_topic_relatedness(
                    client,
                    topic=payload.topic,
                    learning_objectives=payload.learning_objectives,
                    previous_context=_current_continuity_context(previous_record),
                )
                topic_bridge = build_topic_bridge_context(
                    relatedness,
                    _current_continuity_context(previous_record),
                )
        except OpenAIOperationTimeoutError:
            topic_bridge = None
        except Exception:
            topic_bridge = None

        try:
            with stage_timer.measure("evidence_retrieval", topic=payload.topic):
                evidence_result = build_evidence_pack(
                    client,
                    subject=payload.subject,
                    topic=payload.topic,
                    audience_level=payload.audience_level,
                    learning_objectives=payload.learning_objectives,
                    allow_web=effective_allow_web,
                    vector_store_id=payload.vector_store_id,
                    source_policy=source_policy,
                    start_index=1,
                    fast_mode=payload.fast_mode,
                    stage_timer=stage_timer,
                    source_preference=payload.source_preference or "",
                    prefer_high_trust_sources=payload.prefer_high_trust_sources,
                )
            evidence_pack = evidence_result.evidence_pack
        except OpenAIOperationTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(status_code=502, detail=f"Evidence retrieval failed: {type(exc).__name__}: {exc}") from exc

        try:
            with stage_timer.measure("module_generation", topic=payload.topic, evidence_count=len(evidence_pack)):
                module = generate_module_from_evidence(
                    client,
                    request=payload,
                    evidence_pack=evidence_pack,
                    module_id=module_id,
                    web_unavailable_objectives=evidence_result.web_unavailable_objectives,
                    objectives_without_evidence=evidence_result.objectives_without_evidence,
                    topic_bridge=topic_bridge,
                )
        except OpenAIOperationTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(status_code=502, detail=f"Module generation failed: {type(exc).__name__}: {exc}") from exc

        try:
            module = enforce_quality_gate(module, evidence_pack=evidence_pack)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        module_store.save(
            module_id,
            payload,
            module,
            evidence_pack,
            source_policy=source_policy,
            action="generated",
        )
        module = module.model_copy(update={"source_policy": source_policy, "evidence_pack": evidence_pack})
        return ModuleGenerateResponse(module=module, evidence_pack=evidence_pack)
    finally:
        stage_timer.finish(topic=payload.topic)


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
                subject=record.request.subject,
                topic=record.request.topic,
                audience_level=record.request.audience_level,
                learning_objectives=[effective_objective],
                allow_web=effective_allow_web,
                vector_store_id=effective_vector_store_id,
                source_policy=effective_source_policy,
                start_index=next_evidence_index(record.evidence_pack),
                fast_mode=record.request.fast_mode,
                source_preference=record.request.source_preference or "",
                prefer_high_trust_sources=record.request.prefer_high_trust_sources,
            )
            new_evidence = evidence_result.evidence_pack
        except OpenAIOperationTimeoutError as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(status_code=504, detail=str(exc)) from exc
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
        except OpenAIOperationTimeoutError as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(status_code=504, detail=str(exc)) from exc
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
        action="section_improved",
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
        except OpenAIOperationTimeoutError as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(status_code=504, detail=str(exc)) from exc
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
        action="sources_refreshed",
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
            except OpenAIOperationTimeoutError as exc:  # pragma: no cover - runtime network branch
                raise HTTPException(status_code=504, detail=str(exc)) from exc
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
        except OpenAIOperationTimeoutError as exc:  # pragma: no cover - runtime network branch
            raise HTTPException(status_code=504, detail=str(exc)) from exc
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
        action="section_improved",
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


@app.get("/v1/modules/{module_id}/history", response_model=list[ModuleVersionSummary])
def module_history(module_id: str) -> list[ModuleVersionSummary]:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")
    return module_store.history(module_id)


@app.post("/v1/modules/{module_id}/revert/{version_id}", response_model=Module)
def revert_module_version(module_id: str, version_id: str) -> Module:
    restored = module_store.revert(module_id, version_id)
    if restored is None:
        if module_store.get(module_id) is None:
            raise HTTPException(status_code=404, detail="Module not found.")
        raise HTTPException(status_code=404, detail="Version not found.")
    return restored


@app.post("/v1/modules/{module_id}/ask", response_model=ModuleAskResponse)
def ask_module_question(module_id: str, payload: ModuleAskRequest) -> ModuleAskResponse:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")

    evidence_pack = _current_evidence_pack(record)
    if not evidence_pack:
        return ModuleAskResponse(
            answer=(
                "I cannot answer that confidently from this module because no cached source evidence is available. "
                "Please refresh sources or ask about material already shown in the module."
            ),
            citations=[],
            unverified=True,
        )

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        return answer_question_from_module(
            client,
            module=record.module,
            question=payload.question,
            evidence_pack=evidence_pack,
            mode=payload.mode,
            quiz_prompt=payload.quiz_prompt,
        )
    except OpenAIOperationTimeoutError as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=502, detail=f"Tutor answer failed: {type(exc).__name__}: {exc}") from exc


@app.post("/v1/modules/{module_id}/assignment", response_model=ModuleAssignmentResponse)
def create_module_assignment(module_id: str) -> ModuleAssignmentResponse:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        return generate_assignment_from_module(
            client,
            module=record.module,
            evidence_pack=_current_evidence_pack(record),
        )
    except OpenAIOperationTimeoutError as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=502, detail=f"Assignment generation failed: {type(exc).__name__}: {exc}") from exc


@app.post("/v1/modules/{module_id}/assignment/export/json", response_model=ModuleAssignmentResponse)
def export_assignment_json(module_id: str, payload: ModuleAssignmentResponse) -> ModuleAssignmentResponse:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")
    return payload


@app.post("/v1/modules/{module_id}/assignment/export/markdown")
def export_assignment_markdown(module_id: str, payload: ModuleAssignmentResponse) -> Response:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")
    return Response(
        content=_assignment_export_markdown(payload),
        media_type="text/markdown",
    )


@app.post("/v1/modules/{module_id}/grade", response_model=ModuleGradeResponse)
def grade_module_assignment(module_id: str, payload: ModuleGradeRequest) -> ModuleGradeResponse:
    record = module_store.get(module_id)
    if not record:
        raise HTTPException(status_code=404, detail="Module not found.")

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        return grade_assignment_from_module(
            client,
            module=record.module,
            evidence_pack=_current_evidence_pack(record),
            student_response=payload.student_response,
            rubric=payload.rubric,
        )
    except OpenAIOperationTimeoutError as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime network branch
        raise HTTPException(status_code=502, detail=f"Assignment grading failed: {type(exc).__name__}: {exc}") from exc


@app.post("/v1/modules/{module_id}/share", response_model=ModuleShareResponse)
def toggle_module_share(request: Request, module_id: str, payload: ModuleShareRequest) -> ModuleShareResponse:
    updated_module = module_store.set_share_enabled(module_id, payload.enabled)
    if updated_module is None:
        raise HTTPException(status_code=404, detail="Module not found.")

    return ModuleShareResponse(
        module_id=module_id,
        share_enabled=updated_module.share_enabled,
        share_id=updated_module.share_id,
        share_url=_share_url(request, updated_module.share_id) if updated_module.share_enabled else None,
    )


@app.post("/v1/modules/{module_id}/delete", response_model=ModuleDeleteResponse)
def delete_module(module_id: str) -> ModuleDeleteResponse:
    deleted = module_store.delete(module_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Module not found.")
    return ModuleDeleteResponse(module_id=module_id, deleted=True)
