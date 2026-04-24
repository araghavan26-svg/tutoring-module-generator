from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence
from uuid import uuid4

from ..errors import ConfigurationError, NotFoundError, UpstreamServiceError, UpstreamTimeoutError, ValidationAppError
from ..logging_utils import get_logger
from ..models import (
    EvidenceItem,
    Module,
    ModuleAssignmentResponse,
    ModuleAskRequest,
    ModuleAskResponse,
    ModuleDeleteResponse,
    ModuleGenerateRequest,
    ModuleGenerateResponse,
    ModuleGradeRequest,
    ModuleGradeResponse,
    ModuleShareResponse,
    ModuleVersionSummary,
    RefreshSourcesRequest,
    RefreshSourcesResponse,
    SectionRegenerateByIdRequest,
    SectionRegenerateRequest,
    SectionRegenerateResponse,
    SourcePolicy,
)
from ..openai_client import get_openai_client
from ..openai_service import (
    MODULE_TIMEOUT_MESSAGE,
    OpenAIOperationTimeoutError,
    answer_question_from_module,
    build_topic_bridge_context,
    build_unverified_module,
    build_evidence_pack,
    detect_topic_relatedness,
    enforce_quality_gate,
    generate_assignment_from_module,
    generate_module_from_evidence,
    generate_section_from_evidence,
    grade_assignment_from_module,
    merge_evidence_pack,
    next_evidence_index,
    objective_for_section,
    resolve_section_index,
)
from ..services.export_service import assignment_export_markdown, module_export_markdown, module_export_payload, share_url
from ..services.retrieval_helpers import (
    current_continuity_context,
    current_evidence_pack,
    current_source_policy,
    evidence_source_counts,
    has_retrieval_source,
    normalize_regenerated_section,
    refresh_record_evidence_pack,
    updated_request_with_source_policy,
)
from ..stage_timing import StageTimingLogger, log_stage
from ..store import PersistentModuleStore, module_store


logger = get_logger("tutoring_module_api")


def _client_from(provider: Callable[[], Any]) -> Any:
    try:
        return provider()
    except RuntimeError as exc:
        raise ConfigurationError(str(exc)) from exc


def _module_record_or_raise(store: PersistentModuleStore, module_id: str) -> Any:
    record = store.get(module_id)
    if record is None:
        raise NotFoundError("Module not found.", error_code="module_not_found")
    return record


@log_stage("evidence_retrieval", fields_factory=lambda payload, **_: {"topic": payload.topic})
def _retrieve_evidence_for_module(
    payload: ModuleGenerateRequest,
    *,
    client: Any,
    source_policy: SourcePolicy,
    evidence_builder: Callable[..., Any] = build_evidence_pack,
    stage_timer: StageTimingLogger | None = None,
) -> Any:
    return evidence_builder(
        client,
        subject=payload.subject,
        topic=payload.topic,
        audience_level=payload.audience_level,
        learning_objectives=payload.learning_objectives,
        allow_web=source_policy.allow_web,
        vector_store_id=payload.vector_store_id,
        source_policy=source_policy,
        start_index=1,
        fast_mode=payload.fast_mode,
        stage_timer=stage_timer,
        source_preference=payload.source_preference or "",
        prefer_high_trust_sources=payload.prefer_high_trust_sources,
    )


@log_stage(
    "module_generation",
    fields_factory=lambda payload, evidence_pack, **_: {"topic": payload.topic, "evidence_count": len(evidence_pack)},
)
def _generate_grounded_module(
    payload: ModuleGenerateRequest,
    evidence_pack: Sequence[EvidenceItem],
    *,
    client: Any,
    module_id: str,
    evidence_result: Any,
    topic_bridge: Any,
    module_generator: Callable[..., Module] = generate_module_from_evidence,
    stage_timer: StageTimingLogger | None = None,
) -> Module:
    return module_generator(
        client,
        request=payload,
        evidence_pack=evidence_pack,
        module_id=module_id,
        web_unavailable_objectives=evidence_result.web_unavailable_objectives,
        objectives_without_evidence=evidence_result.objectives_without_evidence,
        topic_bridge=topic_bridge,
    )


def generate_module_response(
    payload: ModuleGenerateRequest,
    *,
    store: PersistentModuleStore = module_store,
    client_provider: Callable[[], Any] = get_openai_client,
    evidence_builder: Callable[..., Any] = build_evidence_pack,
    module_generator: Callable[..., Module] = generate_module_from_evidence,
    quality_gate: Callable[..., Module] = enforce_quality_gate,
    relatedness_detector: Callable[..., Any] = detect_topic_relatedness,
    topic_bridge_builder: Callable[..., Any] = build_topic_bridge_context,
    unverified_module_builder: Callable[..., Module] = build_unverified_module,
) -> ModuleGenerateResponse:
    module_id = str(uuid4())
    source_policy = payload.effective_source_policy()
    stage_timer = StageTimingLogger(logger=logger, request_id=module_id)
    stage_timer.log_event(
        "generation.request.start",
        topic=payload.topic,
        objective_count=len(payload.learning_objectives),
        fast_mode=payload.fast_mode,
    )

    try:
        if not has_retrieval_source(payload.vector_store_id, source_policy.allow_web):
            module = unverified_module_builder(
                module_id=module_id,
                request=payload,
                reason="No retrieval source configured (vector_store_id missing and source_policy.allow_web=false).",
            )
            module = quality_gate(module, evidence_pack=[])
            store.save(module_id, payload, module, [], source_policy=source_policy, action="generated")
            module = module.model_copy(update={"source_policy": source_policy, "evidence_pack": []})
            return ModuleGenerateResponse(module=module, evidence_pack=[])

        previous_record = store.latest()
        client = _client_from(client_provider)
        topic_bridge = None
        if payload.related_to_previous and previous_record is not None:
            try:
                relatedness = relatedness_detector(
                    client,
                    topic=payload.topic,
                    learning_objectives=payload.learning_objectives,
                    previous_context=current_continuity_context(previous_record),
                )
                topic_bridge = topic_bridge_builder(relatedness, current_continuity_context(previous_record))
            except OpenAIOperationTimeoutError:
                topic_bridge = None
            except Exception:
                topic_bridge = None

        try:
            evidence_result = _retrieve_evidence_for_module(
                payload,
                client=client,
                source_policy=source_policy,
                evidence_builder=evidence_builder,
                stage_timer=stage_timer,
            )
        except OpenAIOperationTimeoutError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise UpstreamServiceError(f"Evidence retrieval failed: {type(exc).__name__}: {exc}") from exc

        evidence_pack = evidence_result.evidence_pack
        try:
            module = _generate_grounded_module(
                payload,
                evidence_pack,
                client=client,
                module_id=module_id,
                evidence_result=evidence_result,
                topic_bridge=topic_bridge,
                module_generator=module_generator,
                stage_timer=stage_timer,
            )
        except OpenAIOperationTimeoutError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise UpstreamServiceError(f"Module generation failed: {type(exc).__name__}: {exc}") from exc

        try:
            module = quality_gate(module, evidence_pack=evidence_pack)
        except ValueError as exc:
            raise ValidationAppError(str(exc)) from exc

        store.save(module_id, payload, module, evidence_pack, source_policy=source_policy, action="generated")
        module = module.model_copy(update={"source_policy": source_policy, "evidence_pack": evidence_pack})
        return ModuleGenerateResponse(module=module, evidence_pack=evidence_pack)
    finally:
        stage_timer.finish(topic=payload.topic)


def regenerate_section_response(
    module_id: str,
    payload: SectionRegenerateRequest,
    *,
    store: PersistentModuleStore = module_store,
    client_provider: Callable[[], Any] = get_openai_client,
    evidence_builder: Callable[..., Any] = build_evidence_pack,
    section_generator: Callable[..., Any] = generate_section_from_evidence,
    quality_gate: Callable[..., Module] = enforce_quality_gate,
) -> SectionRegenerateResponse:
    record = _module_record_or_raise(store, module_id)
    try:
        section_index = resolve_section_index(record.module, payload)
    except ValueError as exc:
        raise ValidationAppError(str(exc)) from exc

    target_section = record.module.sections[section_index]
    base_source_policy = current_source_policy(record)
    effective_allow_web = base_source_policy.allow_web if payload.allow_web is None else payload.allow_web
    effective_source_policy = base_source_policy.model_copy(update={"allow_web": effective_allow_web})
    effective_vector_store_id = payload.vector_store_id or record.request.vector_store_id
    effective_objective = payload.learning_objective or objective_for_section(
        record.request,
        section_index,
        fallback_heading=target_section.heading,
    )

    new_evidence: List[EvidenceItem] = []
    if has_retrieval_source(effective_vector_store_id, effective_allow_web):
        client = _client_from(client_provider)
        try:
            evidence_result = evidence_builder(
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
        except OpenAIOperationTimeoutError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise UpstreamServiceError(
                f"Evidence retrieval failed during regeneration: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            regenerated = section_generator(
                client,
                request=record.request,
                module=record.module,
                target_section=target_section,
                evidence_pack=new_evidence,
                instructions=payload.instructions,
            )
        except OpenAIOperationTimeoutError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise UpstreamServiceError(f"Section regeneration failed: {type(exc).__name__}: {exc}") from exc
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
        raise ValidationAppError(
            "Quality check failed for regenerated section: section must include citations or be marked unverified=true."
        )

    updated_sections = [section.model_copy(deep=True) for section in record.module.sections]
    updated_sections[section_index] = regenerated
    merged_evidence = merge_evidence_pack(record.evidence_pack, new_evidence)
    updated_module = record.module.model_copy(update={"sections": updated_sections})

    try:
        updated_module = quality_gate(updated_module, evidence_pack=merged_evidence)
    except ValueError as exc:
        raise ValidationAppError(str(exc)) from exc

    updated_request = updated_request_with_source_policy(
        record.request,
        source_policy=effective_source_policy,
        vector_store_id=effective_vector_store_id,
    )
    store.save(
        module_id,
        updated_request,
        updated_module,
        merged_evidence,
        source_policy=effective_source_policy,
        action="section_improved",
    )
    updated_module = updated_module.model_copy(update={"source_policy": effective_source_policy, "evidence_pack": merged_evidence})
    return SectionRegenerateResponse(module=updated_module, regenerated_section_index=section_index, evidence_pack=new_evidence)


def refresh_sources_response(
    module_id: str,
    payload: RefreshSourcesRequest,
    *,
    store: PersistentModuleStore = module_store,
    client_provider: Callable[[], Any] = get_openai_client,
) -> RefreshSourcesResponse:
    record = _module_record_or_raise(store, module_id)
    source_policy = payload.source_policy or current_source_policy(record)
    evidence_pack: List[EvidenceItem] = []

    if has_retrieval_source(record.request.vector_store_id, source_policy.allow_web):
        client = _client_from(client_provider)
        try:
            evidence_pack = refresh_record_evidence_pack(client=client, record=record, source_policy=source_policy)
        except OpenAIOperationTimeoutError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise UpstreamServiceError(f"Evidence refresh failed: {type(exc).__name__}: {exc}") from exc

    updated_request = updated_request_with_source_policy(record.request, source_policy=source_policy)
    updated_module = record.module.model_copy(update={"source_policy": source_policy, "evidence_pack": evidence_pack})
    store.save(module_id, updated_request, updated_module, evidence_pack, source_policy=source_policy, action="sources_refreshed")
    counts = evidence_source_counts(evidence_pack)
    return RefreshSourcesResponse(module_id=module_id, source_policy=source_policy, vector_store_id=record.request.vector_store_id, **counts)


def regenerate_section_by_id_response(
    module_id: str,
    section_id: str,
    payload: SectionRegenerateByIdRequest,
    *,
    store: PersistentModuleStore = module_store,
    client_provider: Callable[[], Any] = get_openai_client,
    section_generator: Callable[..., Any] = generate_section_from_evidence,
) -> SectionRegenerateResponse:
    record = _module_record_or_raise(store, module_id)
    section_index = next((idx for idx, section in enumerate(record.module.sections) if section.section_id == section_id), -1)
    if section_index < 0:
        raise NotFoundError("Section not found.", error_code="section_not_found")

    target_section = record.module.sections[section_index]
    source_policy = current_source_policy(record)
    evidence_pack = current_evidence_pack(record)
    client: Any = None

    if payload.refresh_sources:
        if has_retrieval_source(record.request.vector_store_id, source_policy.allow_web):
            client = _client_from(client_provider)
            try:
                evidence_pack = refresh_record_evidence_pack(client=client, record=record, source_policy=source_policy)
            except OpenAIOperationTimeoutError as exc:
                raise UpstreamTimeoutError(str(exc)) from exc
            except Exception as exc:
                raise UpstreamServiceError(
                    f"Evidence refresh failed during regeneration: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            evidence_pack = []

    if evidence_pack:
        if client is None:
            client = _client_from(client_provider)
        try:
            regenerated = section_generator(
                client,
                request=record.request,
                module=record.module,
                target_section=target_section,
                evidence_pack=evidence_pack,
                instructions=payload.instructions,
            )
        except OpenAIOperationTimeoutError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise UpstreamServiceError(f"Section regeneration failed: {type(exc).__name__}: {exc}") from exc
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
        regenerated = normalize_regenerated_section(regenerated, evidence_pack)
    except ValueError as exc:
        raise ValidationAppError(str(exc)) from exc

    updated_sections = [section.model_copy(deep=True) for section in record.module.sections]
    updated_sections[section_index] = regenerated
    updated_module = record.module.model_copy(
        update={"sections": updated_sections, "source_policy": source_policy, "evidence_pack": evidence_pack}
    )
    updated_request = updated_request_with_source_policy(record.request, source_policy=source_policy)
    store.save(module_id, updated_request, updated_module, evidence_pack, source_policy=source_policy, action="section_improved")
    return SectionRegenerateResponse(module=updated_module, regenerated_section_index=section_index, evidence_pack=evidence_pack)


def export_module_json_payload(module_id: str, *, store: PersistentModuleStore = module_store) -> Dict[str, Any]:
    record = _module_record_or_raise(store, module_id)
    return module_export_payload(record.module.model_dump(mode="json"), current_evidence_pack(record))


def export_module_markdown_text(module_id: str, *, store: PersistentModuleStore = module_store) -> str:
    record = _module_record_or_raise(store, module_id)
    module_data = record.module.model_dump(mode="json")
    payload = module_export_payload(module_data, current_evidence_pack(record))
    return module_export_markdown(module_data, payload["footnotes"])


def module_history(module_id: str, *, store: PersistentModuleStore = module_store) -> List[ModuleVersionSummary]:
    _module_record_or_raise(store, module_id)
    return store.history(module_id)


def revert_module_version(module_id: str, version_id: str, *, store: PersistentModuleStore = module_store) -> Module:
    restored = store.revert(module_id, version_id)
    if restored is None:
        if store.get(module_id) is None:
            raise NotFoundError("Module not found.", error_code="module_not_found")
        raise NotFoundError("Version not found.", error_code="version_not_found")
    return restored


def ask_module_question_response(
    module_id: str,
    payload: ModuleAskRequest,
    *,
    store: PersistentModuleStore = module_store,
    client_provider: Callable[[], Any] = get_openai_client,
    answer_generator: Callable[..., ModuleAskResponse] = answer_question_from_module,
) -> ModuleAskResponse:
    record = _module_record_or_raise(store, module_id)
    evidence_pack = current_evidence_pack(record)
    if not evidence_pack:
        return ModuleAskResponse(
            answer=(
                "I cannot answer that confidently from this module because no cached source evidence is available. "
                "Please refresh sources or ask about material already shown in the module."
            ),
            citations=[],
            unverified=True,
        )
    client = _client_from(client_provider)
    try:
        return answer_generator(
            client,
            module=record.module,
            question=payload.question,
            evidence_pack=evidence_pack,
            mode=payload.mode,
            quiz_prompt=payload.quiz_prompt,
        )
    except OpenAIOperationTimeoutError as exc:
        raise UpstreamTimeoutError(str(exc)) from exc
    except Exception as exc:
        raise UpstreamServiceError(f"Tutor answer failed: {type(exc).__name__}: {exc}") from exc


def create_module_assignment_response(
    module_id: str,
    *,
    store: PersistentModuleStore = module_store,
    client_provider: Callable[[], Any] = get_openai_client,
    assignment_generator: Callable[..., ModuleAssignmentResponse] = generate_assignment_from_module,
) -> ModuleAssignmentResponse:
    record = _module_record_or_raise(store, module_id)
    client = _client_from(client_provider)
    try:
        return assignment_generator(client, module=record.module, evidence_pack=current_evidence_pack(record))
    except OpenAIOperationTimeoutError as exc:
        raise UpstreamTimeoutError(str(exc)) from exc
    except Exception as exc:
        raise UpstreamServiceError(f"Assignment generation failed: {type(exc).__name__}: {exc}") from exc


def export_assignment_json_payload(
    module_id: str,
    payload: ModuleAssignmentResponse,
    *,
    store: PersistentModuleStore = module_store,
) -> ModuleAssignmentResponse:
    _module_record_or_raise(store, module_id)
    return payload


def export_assignment_markdown_text(
    module_id: str,
    payload: ModuleAssignmentResponse,
    *,
    store: PersistentModuleStore = module_store,
) -> str:
    _module_record_or_raise(store, module_id)
    return assignment_export_markdown(payload)


def grade_assignment_response(
    module_id: str,
    payload: ModuleGradeRequest,
    *,
    store: PersistentModuleStore = module_store,
    client_provider: Callable[[], Any] = get_openai_client,
    grading_generator: Callable[..., ModuleGradeResponse] = grade_assignment_from_module,
) -> ModuleGradeResponse:
    record = _module_record_or_raise(store, module_id)
    client = _client_from(client_provider)
    try:
        return grading_generator(
            client,
            module=record.module,
            evidence_pack=current_evidence_pack(record),
            student_response=payload.student_response,
            rubric=payload.rubric,
        )
    except OpenAIOperationTimeoutError as exc:
        raise UpstreamTimeoutError(str(exc)) from exc
    except Exception as exc:
        raise UpstreamServiceError(f"Assignment grading failed: {type(exc).__name__}: {exc}") from exc


def toggle_module_share_response(
    request: Any,
    module_id: str,
    *,
    enabled: bool,
    store: PersistentModuleStore = module_store,
) -> ModuleShareResponse:
    updated_module = store.set_share_enabled(module_id, enabled)
    if updated_module is None:
        raise NotFoundError("Module not found.", error_code="module_not_found")
    return ModuleShareResponse(
        module_id=module_id,
        share_enabled=updated_module.share_enabled,
        share_id=updated_module.share_id,
        share_url=share_url(request, updated_module.share_id) if updated_module.share_enabled else None,
    )


def delete_module_response(module_id: str, *, store: PersistentModuleStore = module_store) -> ModuleDeleteResponse:
    deleted = store.delete(module_id)
    if not deleted:
        raise NotFoundError("Module not found.", error_code="module_not_found")
    return ModuleDeleteResponse(module_id=module_id, deleted=True)
