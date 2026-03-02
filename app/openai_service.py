from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse

from openai import OpenAI

from .config import ensure_openai_api_key, settings
from .models import (
    EvidenceItem,
    Module,
    ModuleGenerateRequest,
    ModuleSection,
    SectionRegenerateRequest,
    utc_now,
)


MODULE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["module_id", "title", "sections", "glossary", "mcqs"],
    "properties": {
        "module_id": {"type": "string"},
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "section_id",
                    "heading",
                    "content",
                    "citations",
                    "unverified",
                    "unverified_reason",
                ],
                "properties": {
                    "section_id": {"type": "string"},
                    "heading": {"type": "string"},
                    "content": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "unverified": {"type": "boolean"},
                    "unverified_reason": {"type": "string"},
                },
            },
        },
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["term", "definition"],
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                },
            },
        },
        "mcqs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "options", "answer_index", "explanation"],
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer_index": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
            },
        },
    },
}


SECTION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["section_id", "heading", "content", "citations", "unverified", "unverified_reason"],
    "properties": {
        "section_id": {"type": "string"},
        "heading": {"type": "string"},
        "content": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "unverified": {"type": "boolean"},
        "unverified_reason": {"type": "string"},
    },
}


WEB_EVIDENCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "url", "snippet"],
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                },
            },
        }
    },
}


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=ensure_openai_api_key())


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _model_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="python", exclude_none=True)
    return {}


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output_items = getattr(response, "output", None) or []
    parts: List[str] = []
    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) == "output_text":
                block_text = _clean_text(getattr(block, "text", ""))
                if block_text:
                    parts.append(block_text)
    return "\n".join(parts).strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty text.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON is not an object.")
    return parsed


def poll_vector_file_status(client: OpenAI, vector_store_id: str, vector_file_id: str) -> str:
    status = "in_progress"
    for _ in range(max(1, settings.vector_poll_attempts)):
        state = client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=vector_file_id,
        )
        status = str(getattr(state, "status", status))
        if status in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(max(0.05, settings.vector_poll_sleep_seconds))
    return status


def _location_from_attributes(attributes: Any) -> str | None:
    if not isinstance(attributes, dict):
        return None
    for key in ("location", "page", "page_number", "chunk", "section", "line"):
        value = _clean_text(attributes.get(key))
        if value:
            return f"{key}:{value}"
    return None


def _extract_doc_candidates(
    client: OpenAI,
    *,
    vector_store_id: str,
    topic: str,
    audience_level: str,
    objective: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    query = (
        f"Topic: {topic}\n"
        f"Audience level: {audience_level}\n"
        f"Learning objective: {objective}\n"
        "Retrieve the most relevant source chunks."
    )
    response = client.responses.create(
        model=settings.retrieval_model,
        input=query,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": max(1, max_results),
            }
        ],
        tool_choice="required",
        include=["file_search_call.results"],
        temperature=0,
        max_output_tokens=128,
    )
    retrieved_at = utc_now()
    candidates: List[Dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "file_search_call":
            continue
        call_dict = _model_to_dict(item)
        raw_results = call_dict.get("results") or []
        if not isinstance(raw_results, list):
            continue
        for result in raw_results:
            result_dict = _model_to_dict(result)
            snippet = _clean_text(result_dict.get("text"))
            if not snippet:
                continue
            filename = _clean_text(result_dict.get("filename")) or "Uploaded document"
            candidates.append(
                {
                    "source_type": "doc",
                    "title": filename,
                    "url": None,
                    "doc_name": filename,
                    "location": _location_from_attributes(result_dict.get("attributes")),
                    "snippet": snippet,
                    "retrieved_at": retrieved_at,
                }
            )
    return candidates[: max(1, max_results)]


def _extract_web_urls_from_response(response: Any) -> Dict[str, str]:
    urls: Dict[str, str] = {}
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "web_search_call":
            call = _model_to_dict(item)
            action = call.get("action") if isinstance(call.get("action"), dict) else {}
            sources = action.get("sources") if isinstance(action, dict) else None
            if isinstance(sources, list):
                for source in sources:
                    source_dict = _model_to_dict(source)
                    url = _clean_text(source_dict.get("url"))
                    if not url:
                        continue
                    urls[url] = urls.get(url) or _infer_title_from_url(url)
        if item_type != "message":
            continue
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) != "output_text":
                continue
            for annotation in getattr(block, "annotations", []) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = _clean_text(getattr(annotation, "url", ""))
                title = _clean_text(getattr(annotation, "title", ""))
                if url:
                    urls[url] = title or urls.get(url) or _infer_title_from_url(url)
    return urls


def _infer_title_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc or "Web source"
        return host
    except Exception:
        return "Web source"


def _extract_web_candidates(
    client: OpenAI,
    *,
    topic: str,
    audience_level: str,
    objective: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    prompt = (
        f"Topic: {topic}\n"
        f"Audience level: {audience_level}\n"
        f"Learning objective: {objective}\n\n"
        "Search the web and return evidence snippets for teaching this objective."
    )
    response = client.responses.create(
        model=settings.retrieval_model,
        input=prompt,
        tools=[{"type": "web_search", "search_context_size": "medium"}],
        tool_choice="required",
        include=["web_search_call.action.sources", "web_search_call.results"],
        text={
            "format": {
                "type": "json_schema",
                "name": "web_evidence_items",
                "schema": WEB_EVIDENCE_SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
        max_output_tokens=1200,
    )
    payload = parse_json_object(response_text(response))
    raw_items = payload.get("items", [])
    source_urls = _extract_web_urls_from_response(response)

    retrieved_at = utc_now()
    parsed: List[Dict[str, Any]] = []
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            title = _clean_text(raw.get("title")) or "Web source"
            url = _clean_text(raw.get("url"))
            snippet = _clean_text(raw.get("snippet"))
            if not url or not snippet:
                continue
            if source_urls and url not in source_urls:
                continue
            parsed.append(
                {
                    "source_type": "web",
                    "title": title,
                    "url": url,
                    "doc_name": None,
                    "location": None,
                    "snippet": snippet,
                    "retrieved_at": retrieved_at,
                }
            )

    if not parsed and source_urls:
        fallback_snippet = _clean_text(response_text(response))[:450] or "Relevant source retrieved via web search."
        for url, title in source_urls.items():
            parsed.append(
                {
                    "source_type": "web",
                    "title": title or _infer_title_from_url(url),
                    "url": url,
                    "doc_name": None,
                    "location": None,
                    "snippet": fallback_snippet,
                    "retrieved_at": retrieved_at,
                }
            )

    return parsed[: max(1, max_results)]


def build_evidence_pack(
    client: OpenAI,
    *,
    topic: str,
    audience_level: str,
    learning_objectives: Sequence[str],
    allow_web: bool,
    vector_store_id: str | None,
    start_index: int = 1,
) -> List[EvidenceItem]:
    if not vector_store_id and not allow_web:
        return []

    raw_candidates: List[Dict[str, Any]] = []
    retrieval_errors: List[str] = []
    for objective in learning_objectives:
        objective_text = _clean_text(objective)
        if not objective_text:
            continue
        if vector_store_id:
            try:
                raw_candidates.extend(
                    _extract_doc_candidates(
                        client,
                        vector_store_id=vector_store_id,
                        topic=topic,
                        audience_level=audience_level,
                        objective=objective_text,
                        max_results=settings.doc_results_per_objective,
                    )
                )
            except Exception as exc:
                retrieval_errors.append(f"doc({objective_text}): {type(exc).__name__}: {exc}")
        if allow_web:
            try:
                raw_candidates.extend(
                    _extract_web_candidates(
                        client,
                        topic=topic,
                        audience_level=audience_level,
                        objective=objective_text,
                        max_results=settings.web_results_per_objective,
                    )
                )
            except Exception as exc:
                retrieval_errors.append(f"web({objective_text}): {type(exc).__name__}: {exc}")

    if not raw_candidates and retrieval_errors:
        detail = " | ".join(retrieval_errors[:4])
        raise RuntimeError(f"Evidence retrieval failed: {detail}")

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in raw_candidates:
        key = (
            item.get("source_type"),
            item.get("url") or item.get("doc_name") or item.get("title"),
            item.get("snippet"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    evidence_pack: List[EvidenceItem] = []
    index = max(1, start_index)
    for item in deduped:
        evidence_pack.append(
            EvidenceItem(
                evidence_id=f"E{index:03d}",
                source_type=item["source_type"],
                title=item["title"],
                url=item.get("url"),
                doc_name=item.get("doc_name"),
                location=item.get("location"),
                snippet=item["snippet"],
                retrieved_at=item.get("retrieved_at") or utc_now(),
            )
        )
        index += 1
    return evidence_pack


def build_unverified_section(
    *,
    section_id: str,
    heading: str,
    reason: str,
) -> ModuleSection:
    return ModuleSection(
        section_id=section_id,
        heading=heading,
        content="No verifiable instructional content could be generated from retrieved evidence.",
        citations=[],
        unverified=True,
        unverified_reason=reason,
    )


def build_unverified_module(
    *,
    module_id: str,
    request: ModuleGenerateRequest,
    reason: str,
) -> Module:
    headings = request.learning_objectives or [request.topic]
    sections: List[ModuleSection] = []
    for idx, heading in enumerate(headings):
        sections.append(
            build_unverified_section(
                section_id=f"section-{idx + 1}",
                heading=heading or f"Section {idx + 1}",
                reason=reason,
            )
        )
    return Module(
        module_id=module_id,
        title=f"{request.topic} - Grounded Tutoring Module",
        sections=sections,
        glossary=[],
        mcqs=[],
    )


def generate_module_from_evidence(
    client: OpenAI,
    *,
    request: ModuleGenerateRequest,
    evidence_pack: Sequence[EvidenceItem],
    module_id: str,
) -> Module:
    if not evidence_pack:
        return build_unverified_module(
            module_id=module_id,
            request=request,
            reason="No evidence was retrieved from the configured sources.",
        )

    evidence_payload = [item.model_dump(mode="json") for item in evidence_pack]
    allowed_ids = [item.evidence_id for item in evidence_pack]
    prompt_payload = {
        "module_request": {
            "topic": request.topic,
            "audience_level": request.audience_level,
            "learning_objectives": request.learning_objectives,
        },
        "allowed_evidence_ids": allowed_ids,
        "evidence_pack": evidence_payload,
    }

    response = client.responses.create(
        model=settings.generation_model,
        instructions=(
            "You are a tutoring-module generator.\n"
            "Use only evidence from evidence_pack. Never invent facts.\n"
            "Every verified section must cite evidence_id values from allowed_evidence_ids.\n"
            "If evidence is insufficient for a section, set unverified=true and explain why."
        ),
        input=json.dumps(prompt_payload, ensure_ascii=True),
        text={
            "format": {
                "type": "json_schema",
                "name": "grounded_module",
                "schema": MODULE_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
        temperature=0.1,
        max_output_tokens=3500,
    )
    payload = parse_json_object(response_text(response))
    module = Module.model_validate(payload)
    return module.model_copy(update={"module_id": module_id})


def generate_section_from_evidence(
    client: OpenAI,
    *,
    request: ModuleGenerateRequest,
    module: Module,
    target_section: ModuleSection,
    evidence_pack: Sequence[EvidenceItem],
    instructions: str | None,
) -> ModuleSection:
    if not evidence_pack:
        return build_unverified_section(
            section_id=target_section.section_id,
            heading=target_section.heading,
            reason="No evidence was retrieved for this regeneration request.",
        )

    prompt_payload = {
        "module_context": {
            "module_id": module.module_id,
            "module_title": module.title,
            "topic": request.topic,
            "audience_level": request.audience_level,
            "learning_objectives": request.learning_objectives,
        },
        "target_section": {
            "section_id": target_section.section_id,
            "heading": target_section.heading,
            "existing_content": target_section.content,
        },
        "instructions": _clean_text(instructions),
        "allowed_evidence_ids": [item.evidence_id for item in evidence_pack],
        "evidence_pack": [item.model_dump(mode="json") for item in evidence_pack],
    }

    response = client.responses.create(
        model=settings.generation_model,
        instructions=(
            "Regenerate one section using only evidence_pack.\n"
            "For verified content, citations must contain valid evidence_id values.\n"
            "If evidence is insufficient, set unverified=true and provide unverified_reason."
        ),
        input=json.dumps(prompt_payload, ensure_ascii=True),
        text={
            "format": {
                "type": "json_schema",
                "name": "grounded_section",
                "schema": SECTION_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
        temperature=0.1,
        max_output_tokens=1600,
    )
    payload = parse_json_object(response_text(response))
    section = ModuleSection.model_validate(payload)
    return section.model_copy(
        update={
            "section_id": target_section.section_id,
            "heading": target_section.heading,
        }
    )


def enforce_quality_gate(module: Module, evidence_pack: Sequence[EvidenceItem]) -> Module:
    evidence_ids = {item.evidence_id for item in evidence_pack}
    normalized_sections: List[ModuleSection] = []

    for section in module.sections:
        citations = [item.strip() for item in section.citations if item and item.strip()]
        citations = list(dict.fromkeys(citations))

        if citations:
            invalid = [item for item in citations if item not in evidence_ids]
            if invalid:
                joined = ", ".join(invalid)
                raise ValueError(f"Section '{section.heading}' references unknown evidence ids: {joined}.")

        if not citations and not section.unverified:
            raise ValueError(
                f"Section '{section.heading}' failed quality check: no citations and unverified is false."
            )

        reason = _clean_text(section.unverified_reason)
        if section.unverified and not reason:
            reason = "Insufficient retrieved evidence for this section."
        if not section.unverified:
            reason = ""

        normalized_sections.append(
            section.model_copy(
                update={
                    "citations": citations,
                    "unverified_reason": reason,
                }
            )
        )

    return module.model_copy(update={"sections": normalized_sections})


def merge_evidence_pack(existing: Sequence[EvidenceItem], new_items: Sequence[EvidenceItem]) -> List[EvidenceItem]:
    seen = set()
    merged: List[EvidenceItem] = []
    for item in [*existing, *new_items]:
        if item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        merged.append(item)
    return merged


def next_evidence_index(evidence_pack: Sequence[EvidenceItem]) -> int:
    max_seen = 0
    for item in evidence_pack:
        match = re.match(r"^E(\d+)$", item.evidence_id)
        if not match:
            continue
        max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def resolve_section_index(module: Module, request: SectionRegenerateRequest) -> int:
    if request.section_index is not None:
        if request.section_index >= len(module.sections):
            raise ValueError("section_index is out of range.")
        return request.section_index

    if request.section_id:
        for idx, section in enumerate(module.sections):
            if section.section_id == request.section_id:
                return idx
        raise ValueError("section_id not found.")

    if request.section_heading:
        target = request.section_heading.strip().lower()
        for idx, section in enumerate(module.sections):
            if section.heading.strip().lower() == target:
                return idx
        raise ValueError("section_heading not found.")

    raise ValueError("No section selector provided.")


def objective_for_section(request: ModuleGenerateRequest, section_index: int, fallback_heading: str) -> str:
    if 0 <= section_index < len(request.learning_objectives):
        candidate = _clean_text(request.learning_objectives[section_index])
        if candidate:
            return candidate
    return _clean_text(fallback_heading) or "Regenerated section objective"
