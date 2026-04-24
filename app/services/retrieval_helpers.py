from __future__ import annotations

from typing import Any, Dict, List

from ..models import EvidenceItem, ModuleGenerateRequest, SourcePolicy, TopicContinuityContext
from ..openai_service import build_evidence_pack, next_evidence_index


def has_retrieval_source(vector_store_id: str | None, allow_web: bool) -> bool:
    return bool(vector_store_id or allow_web)


def current_source_policy(record: Any) -> SourcePolicy:
    policy = getattr(record, "source_policy", None)
    if policy is not None:
        return policy
    if record.module.source_policy is not None:
        return record.module.source_policy
    return record.request.effective_source_policy()


def current_evidence_pack(record: Any) -> List[EvidenceItem]:
    module_cached = list(record.module.evidence_pack or [])
    if module_cached:
        return module_cached
    return list(record.evidence_pack or [])


def current_continuity_context(record: Any) -> TopicContinuityContext:
    return TopicContinuityContext(
        topic=record.request.topic,
        learning_objectives=list(record.request.learning_objectives or []),
        module_summary=str(record.module.overview or "").strip(),
    )


def evidence_source_counts(evidence_pack: List[EvidenceItem]) -> Dict[str, int]:
    doc_count = sum(1 for item in evidence_pack if item.source_type == "doc")
    web_count = sum(1 for item in evidence_pack if item.source_type == "web")
    return {
        "evidence_count": len(evidence_pack),
        "doc_count": doc_count,
        "web_count": web_count,
    }


def format_timestamp_for_ui(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%b %d, %Y %I:%M %p")
    raw = str(value or "").strip()
    return raw or "Unknown time"


def normalize_regenerated_section(section: Any, evidence_pack: List[EvidenceItem]) -> Any:
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


def refresh_record_evidence_pack(
    *,
    client: Any,
    record: Any,
    source_policy: SourcePolicy,
) -> List[EvidenceItem]:
    vector_store_id = record.request.vector_store_id
    if not has_retrieval_source(vector_store_id, source_policy.allow_web):
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


def refresh_regeneration_evidence(
    *,
    client: Any,
    record: Any,
    source_policy: SourcePolicy,
    learning_objectives: List[str],
    vector_store_id: str | None,
) -> List[EvidenceItem]:
    if not has_retrieval_source(vector_store_id, source_policy.allow_web):
        return []
    evidence_result = build_evidence_pack(
        client,
        subject=record.request.subject,
        topic=record.request.topic,
        audience_level=record.request.audience_level,
        learning_objectives=learning_objectives,
        allow_web=source_policy.allow_web,
        vector_store_id=vector_store_id,
        source_policy=source_policy,
        start_index=next_evidence_index(record.evidence_pack),
        fast_mode=record.request.fast_mode,
        source_preference=record.request.source_preference or "",
        prefer_high_trust_sources=record.request.prefer_high_trust_sources,
    )
    return evidence_result.evidence_pack


def updated_request_with_source_policy(
    request: ModuleGenerateRequest,
    *,
    source_policy: SourcePolicy,
    vector_store_id: str | None = None,
) -> ModuleGenerateRequest:
    update_payload: Dict[str, Any] = {
        "allow_web": source_policy.allow_web,
        "source_policy": source_policy,
    }
    if vector_store_id is not None:
        update_payload["vector_store_id"] = vector_store_id
    return request.model_copy(update=update_payload)
