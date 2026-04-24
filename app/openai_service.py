from __future__ import annotations

import json
from pathlib import Path
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Sequence
from urllib.parse import urlparse

from openai import APITimeoutError, OpenAI

from .config import settings
from .logging_utils import get_logger
from .models import (
    EvidenceItem,
    ModuleAssignmentResponse,
    ModuleAskResponse,
    ModuleGradeBreakdownItem,
    ModuleGradeResponse,
    Module,
    ModuleGenerateRequest,
    ModuleSection,
    RubricCriterion,
    SectionRegenerateRequest,
    SourcePolicy,
    TopicContinuityContext,
    normalize_domain,
    utc_now,
)
from .openai_client import get_openai_client
from .stage_timing import StageTimingLogger


logger = get_logger("tutoring_module_api")
MODULE_TIMEOUT_MESSAGE = "Module generation took longer than expected. Please try again in a moment."


MODULE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["module_id", "title", "overview", "sections", "glossary", "mcqs"],
    "properties": {
        "module_id": {"type": "string"},
        "title": {"type": "string"},
        "overview": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "section_id",
                    "objective_index",
                    "learning_goal",
                    "heading",
                    "content",
                    "citations",
                    "unverified",
                    "unverified_reason",
                ],
                "properties": {
                    "section_id": {"type": "string"},
                    "objective_index": {"type": "integer"},
                    "learning_goal": {"type": "string"},
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
    "required": [
        "section_id",
        "objective_index",
        "learning_goal",
        "heading",
        "content",
        "citations",
        "unverified",
        "unverified_reason",
    ],
    "properties": {
        "section_id": {"type": "string"},
        "objective_index": {"type": "integer"},
        "learning_goal": {"type": "string"},
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


TOPIC_RELATEDNESS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relation", "reason"],
    "properties": {
        "relation": {
            "type": "string",
            "enum": ["closely related", "somewhat related", "unrelated"],
        },
        "reason": {"type": "string"},
    },
}


ASK_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "citations", "unverified"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "unverified": {"type": "boolean"},
    },
}


ASSIGNMENT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt", "rubric"],
    "properties": {
        "prompt": {"type": "string"},
        "rubric": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["criteria", "levels"],
                "properties": {
                    "criteria": {"type": "string"},
                    "levels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["score", "description"],
                            "properties": {
                                "score": {"type": "integer"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}


GRADE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "feedback", "breakdown", "unverified"],
    "properties": {
        "score": {"type": "integer"},
        "feedback": {"type": "string"},
        "breakdown": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["criteria", "score", "max_score", "feedback"],
                "properties": {
                    "criteria": {"type": "string"},
                    "score": {"type": "integer"},
                    "max_score": {"type": "integer"},
                    "feedback": {"type": "string"},
                },
            },
        },
        "unverified": {"type": "boolean"},
    },
}
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


def _call_openai(operation: str, func: Any) -> Any:
    try:
        return func()
    except APITimeoutError as exc:
        logger.warning("openai_timeout", operation=operation)
        raise OpenAIOperationTimeoutError() from exc


def _truncate_snippet(text: str, max_chars: int = 280) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[: max_chars - 1].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip(".,;:") + "..."


def _looks_like_serialized_json(text: str) -> bool:
    raw = _clean_text(text)
    if not raw:
        return False
    if raw.startswith("{") or raw.startswith("["):
        return True
    return '{"items":' in raw[:120] or '"items":' in raw[:120]


def human_readable_snippet(value: Any, *, fallback: str = "") -> str:
    def gather(node: Any) -> List[str]:
        if node is None:
            return []
        if isinstance(node, str):
            raw = _clean_text(node)
            if not raw:
                return []
            if _looks_like_serialized_json(raw):
                try:
                    return gather(json.loads(raw))
                except Exception:
                    return []
            return [raw]
        if isinstance(node, dict):
            preferred_keys = (
                "snippet",
                "text",
                "summary",
                "description",
                "content",
                "excerpt",
                "quote",
                "body",
            )
            candidates: List[str] = []
            for key in preferred_keys:
                if key in node:
                    candidates.extend(gather(node.get(key)))
            if candidates:
                return candidates
            for key in ("items", "results", "sources"):
                if key in node:
                    candidates.extend(gather(node.get(key)))
            if candidates:
                return candidates
            for item in node.values():
                candidates.extend(gather(item))
            return candidates
        if isinstance(node, list):
            candidates: List[str] = []
            for item in node[:5]:
                candidates.extend(gather(item))
                if candidates:
                    break
            return candidates
        return []

    for candidate in gather(value):
        if candidate and not _looks_like_serialized_json(candidate):
            return _truncate_snippet(candidate)

    fallback_text = _clean_text(fallback)
    if fallback_text and not _looks_like_serialized_json(fallback_text):
        return _truncate_snippet(fallback_text)
    return ""


def _is_placeholder_snippet(text: str) -> bool:
    normalized = _clean_text(text).lower()
    if not normalized:
        return True
    placeholders = {
        "relevant source retrieved via web search.",
        "relevant source retrieved via web search",
        "relevant source for this objective.",
        "relevant source for this objective",
    }
    return normalized in placeholders or normalized.startswith("relevant source")


WEB_SNIPPET_FALLBACK = "Source page retrieved successfully, but no descriptive snippet was available."


def _title_looks_like_domain(title: str, url: str = "") -> bool:
    cleaned = _clean_text(title).strip(" -|:")
    if not cleaned:
        return True
    title_domain = normalize_domain(cleaned)
    url_domain = normalize_domain(url)
    if title_domain and url_domain and title_domain == url_domain:
        return True
    if title_domain and not url_domain:
        return True
    normalized_words = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
    domain_words = re.sub(r"[^a-z0-9]+", " ", url_domain.lower()).strip()
    return bool(normalized_words and domain_words and normalized_words == domain_words)


def _title_candidates_from_metadata(metadata: Any) -> List[str]:
    if not isinstance(metadata, dict):
        return []
    keys = (
        "title",
        "name",
        "headline",
        "article_title",
        "page_title",
        "meta_title",
        "og_title",
    )
    values: List[str] = []
    for key in keys:
        value = _clean_text(metadata.get(key))
        if value:
            values.append(value)
    return values


def _snippet_candidates_from_metadata(metadata: Any) -> List[Any]:
    if not isinstance(metadata, dict):
        return []
    keys = (
        "snippet",
        "summary",
        "description",
        "excerpt",
        "text",
        "content",
        "body",
        "quote",
        "meta_description",
        "abstract",
        "dek",
    )
    values: List[Any] = []
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            values.append(value)
    return values


def _is_low_quality_web_snippet(snippet: str, *, domain: str, title: str) -> bool:
    normalized = _clean_text(snippet)
    if not normalized or _is_placeholder_snippet(normalized):
        return True

    normalized_domain = normalize_domain(domain)
    snippet_domain = normalize_domain(normalized)
    if normalized_domain and snippet_domain and snippet_domain == normalized_domain:
        return True

    clean_title = _clean_text(title)
    if clean_title and normalized.lower() == clean_title.lower():
        if _title_looks_like_domain(clean_title, domain):
            return True
        word_count = len(re.findall(r"[a-zA-Z0-9]+", normalized))
        if word_count < 4 and len(normalized) < 40:
            return True

    word_count = len(re.findall(r"[a-zA-Z0-9]+", normalized))
    if word_count < 4 and len(normalized) < 28:
        return True
    return False


def choose_web_title(*, url: str, candidates: Sequence[str]) -> str:
    cleaned_candidates = []
    for candidate in candidates:
        value = _clean_text(candidate)
        if value:
            cleaned_candidates.append(value)
    if cleaned_candidates:
        def score(candidate: str) -> tuple[int, int]:
            return (0 if _title_looks_like_domain(candidate, url) else 1, len(candidate))

        best_candidate = max(cleaned_candidates, key=score)
        if not _title_looks_like_domain(best_candidate, url):
            return best_candidate

    inferred = _infer_title_from_url(url)
    if inferred and not _title_looks_like_domain(inferred, url):
        return inferred
    return normalize_domain(url) or inferred or "Web source"


def choose_web_snippet(*, objective: str, title: str, domain: str, candidates: Sequence[Any]) -> str:
    objective_terms = {
        term
        for term in re.findall(r"[a-zA-Z]+", objective.lower())
        if len(term) >= 4
    }
    snippet_candidates: List[str] = []
    for candidate in candidates:
        snippet = human_readable_snippet(candidate)
        if snippet and not _is_low_quality_web_snippet(snippet, domain=domain, title=title):
            snippet_candidates.append(snippet)

    if snippet_candidates:
        def score(snippet: str) -> tuple[int, int]:
            lower = snippet.lower()
            overlap = sum(1 for term in objective_terms if term in lower)
            return (overlap, len(snippet))

        return max(snippet_candidates, key=score)

    clean_title = _clean_text(title)
    if clean_title and not _title_looks_like_domain(clean_title, domain):
        word_count = len(re.findall(r"[a-zA-Z0-9]+", clean_title))
        if word_count >= 4 or len(clean_title) >= 28:
            return _truncate_snippet(clean_title)
    return WEB_SNIPPET_FALLBACK


def _best_matching_doc_items(
    objective: str,
    doc_pool: Sequence[Dict[str, Any]],
    *,
    limit: int = 1,
) -> List[Dict[str, Any]]:
    objective_terms = {
        term
        for term in re.findall(r"[a-zA-Z]+", objective.lower())
        if len(term) >= 4
    }
    if not objective_terms:
        return []

    ranked: List[tuple[tuple[int, int], Dict[str, Any]]] = []
    for item in doc_pool:
        haystack = " ".join(
            [
                _clean_text(item.get("title")),
                _clean_text(item.get("doc_name")),
                _clean_text(item.get("snippet")),
            ]
        ).lower()
        overlap = sum(1 for term in objective_terms if term in haystack)
        if overlap <= 0:
            continue
        ranked.append(((overlap, len(haystack)), item))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: List[Dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for _score, item in ranked:
        key = (
            str(item.get("doc_name") or item.get("title") or ""),
            str(item.get("location") or ""),
            str(item.get("snippet") or ""),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(dict(item))
        if len(selected) >= limit:
            break
    return selected


def retrieval_plan(fast_mode: bool) -> RetrievalPlan:
    if fast_mode:
        return RetrievalPlan(
            section_limit=max(1, min(settings.fast_section_limit, 6)),
            doc_results_per_objective=max(1, settings.fast_doc_results_per_objective),
            web_results_per_objective=max(1, settings.fast_web_results_per_objective),
            max_evidence_items=max(1, settings.fast_max_evidence_items),
        )
    return RetrievalPlan(
        section_limit=6,
        doc_results_per_objective=max(1, settings.doc_results_per_objective),
        web_results_per_objective=max(1, settings.web_results_per_objective),
        max_evidence_items=max(1, settings.max_evidence_items),
    )


def retrieval_plan_for_request(request: ModuleGenerateRequest) -> RetrievalPlan:
    return retrieval_plan(request.fast_mode)


DIRECT_OBJECTIVE_VERBS = (
    "understand",
    "explain",
    "identify",
    "apply",
    "analyze",
    "interpret",
    "compare",
    "evaluate",
    "describe",
    "calculate",
    "use",
    "connect",
    "review",
    "practice",
    "prepare",
    "build",
    "trace",
    "summarize",
    "distinguish",
    "write",
)

FIRST_PERSON_GOAL_PREFIXES = (
    r"^(?:i|we)\s+(?:want|would\s+like|need|hope)\s+to\s+",
    r"^(?:i|we)\s+am\s+trying\s+to\s+",
    r"^(?:i'?m|we're)\s+trying\s+to\s+",
    r"^(?:help|teach|show)\s+me\s+",
    r"^can\s+you\s+(?:help\s+me\s+)?",
    r"^please\s+help\s+me\s+",
)

GOAL_CONTEXT_PATTERNS = (
    r"\bat an?\s+ap(?:\s+\w+)?\s+level\b",
    r"\bat an?\s+advanced(?:\s+\w+)?\s+level\b",
    r"\bat an?\s+high school\s+level\b",
    r"\bat an?\s+intro college\s+level\b",
    r"\bfor\s+ap(?:\s+\w+)?\b",
    r"\bfor\s+(?:my\s+)?class\b",
    r"\bfor\s+course\s+prep\b",
    r"\bfor\s+test\s+prep\b",
)

ACADEMIC_DISCIPLINE_LABELS = {
    "statistics": "Statistics",
    "probability": "Probability",
    "biology": "Biology",
    "chemistry": "Chemistry",
    "physics": "Physics",
    "science": "Science",
    "history": "History",
    "literature": "Literature",
    "english": "English Literature",
    "economics": "Economics",
    "psychology": "Psychology",
    "government": "Government",
    "geography": "Geography",
}


def _level_text(request: ModuleGenerateRequest | None) -> str:
    if request is None:
        return ""
    return _clean_text(
        " ".join(
            item
            for item in [
                request.learner_level or "",
                request.custom_level_description or "",
                request.audience_level or "",
            ]
            if _clean_text(item)
        )
    )


def _is_advanced_level_text(value: str) -> bool:
    lowered = _clean_text(value).lower()
    return any(marker in lowered for marker in ("ap", "advanced", "honors", "college board"))


def _is_course_style_level_text(value: str) -> bool:
    lowered = _clean_text(value).lower()
    return _is_advanced_level_text(lowered) or any(
        marker in lowered for marker in ("high school", "intro college", "college")
    )


def _contains_first_person(text: str) -> bool:
    lowered = f" {_clean_text(text).lower()} "
    return any(token in lowered for token in (" i ", " me ", " my ", " we ", " our "))


def _discipline_label(subject: str, topic: str, text: str) -> str:
    combined = " ".join(item for item in [text, subject, topic] if _clean_text(item)).lower()
    for keyword, label in ACADEMIC_DISCIPLINE_LABELS.items():
        if keyword in combined:
            return label
    fallback = _clean_text(subject) or _clean_text(topic)
    return fallback or "the topic"


def _course_style_label(request: ModuleGenerateRequest | None, text: str = "") -> str:
    subject_label = _discipline_label(
        getattr(request, "subject", "") or "",
        getattr(request, "topic", "") or "",
        text,
    )
    level_text = _level_text(request)
    lowered = level_text.lower()
    if "ap" in lowered and not subject_label.lower().startswith("ap "):
        return f"AP {subject_label}"
    if "honors" in lowered and not subject_label.lower().startswith("honors "):
        return f"Honors {subject_label}"
    if "advanced" in lowered and not subject_label.lower().startswith("advanced "):
        return f"Advanced {subject_label}"
    return subject_label


def _sentence_case(text: str) -> str:
    cleaned = _clean_text(text).strip(" .,:;")
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


def _headline_case(text: str) -> str:
    cleaned = _clean_text(text).strip(" .,:;")
    if not cleaned:
        return ""
    small_words = {"and", "or", "the", "a", "an", "of", "to", "for", "in", "on", "with"}
    acronyms = {"AP", "IB", "SAT", "ACT", "STEM", "USA", "US"}
    words = cleaned.split()
    titled: List[str] = []
    for index, word in enumerate(words):
        bare = word.strip()
        upper = bare.upper()
        lower = bare.lower()
        if upper in acronyms:
            titled.append(upper)
        elif index > 0 and lower in small_words:
            titled.append(lower)
        else:
            titled.append(lower.capitalize())
    return " ".join(titled)


def normalize_learning_goal_text(
    raw_goal: Any,
    *,
    request: ModuleGenerateRequest | None = None,
    subject: str = "",
    topic: str = "",
    audience_level: str = "",
) -> str:
    cleaned = _clean_text(str(raw_goal or "").replace("’", "'"))
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    for pattern in FIRST_PERSON_GOAL_PREFIXES:
        lowered = re.sub(pattern, "", lowered, flags=re.IGNORECASE)
    for pattern in GOAL_CONTEXT_PATTERNS:
        lowered = re.sub(pattern, "", lowered, flags=re.IGNORECASE)

    lowered = re.sub(r"^to\s+", "", lowered)
    lowered = re.sub(r"^learn\s+about\s+", "understand ", lowered)
    lowered = re.sub(r"^know\s+about\s+", "understand ", lowered)
    lowered = re.sub(r"^understand\s+what\s+(.+?)\s+is$", r"understand \1", lowered)
    lowered = re.sub(r"^explain\s+what\s+(.+?)\s+is$", r"explain \1", lowered)
    lowered = re.sub(r"\b(?:please|really|just)\b", "", lowered)
    lowered = _clean_text(lowered).strip(" .,:;")

    request_level = _level_text(request) or audience_level
    advanced_like = _is_advanced_level_text(f"{request_level} {cleaned}")
    subject_label = _discipline_label(subject or getattr(request, "subject", ""), topic or getattr(request, "topic", ""), lowered)
    course_label = _course_style_label(request, lowered) if request is not None else subject_label
    if request is None and advanced_like:
        lowered_level = _clean_text(audience_level).lower()
        if "ap" in lowered_level and not course_label.lower().startswith("ap "):
            course_label = f"AP {subject_label}"
        elif "honors" in lowered_level and not course_label.lower().startswith("honors "):
            course_label = f"Honors {subject_label}"
        elif "advanced" in lowered_level and not course_label.lower().startswith("advanced "):
            course_label = f"Advanced {subject_label}"

    broad_understanding = (
        lowered == subject_label.lower()
        or lowered == course_label.lower()
        or lowered in {
            f"understand {subject_label.lower()}",
            f"understand {course_label.lower()}",
            f"learn {subject_label.lower()}",
            f"learn {course_label.lower()}",
        }
    )
    if advanced_like and broad_understanding:
        return f"Understand core {course_label} concepts and expectations"

    if advanced_like and re.match(r"^understand\b", lowered) and subject_label.lower() in lowered and len(lowered.split()) <= 8:
        return f"Understand core {course_label} concepts and expectations"

    if not re.match(rf"^({'|'.join(DIRECT_OBJECTIVE_VERBS)})\b", lowered):
        lowered = f"understand {lowered}"

    return _sentence_case(lowered)


def selected_learning_objectives(request: ModuleGenerateRequest) -> List[str]:
    cleaned: List[str] = []
    for item in request.learning_objectives or []:
        normalized = normalize_learning_goal_text(
            item,
            request=request,
            subject=request.subject or "",
            topic=request.topic,
            audience_level=request.audience_level,
        )
        if normalized:
            cleaned.append(normalized)
    if not cleaned:
        cleaned = [
            normalize_learning_goal_text(
                request.topic,
                request=request,
                subject=request.subject or "",
                topic=request.topic,
                audience_level=request.audience_level,
            )
            or "Learning goal"
        ]
    return cleaned[: retrieval_plan_for_request(request).section_limit]


def build_personalization_context(request: ModuleGenerateRequest) -> Dict[str, Any]:
    familiarity = _clean_text(request.current_familiarity)
    purpose = _clean_text(request.learning_purpose)
    style = _clean_text(request.explanation_style)
    confusion = _clean_text(request.confusion_points)
    level_text = _level_text(request)
    advanced_like = _is_advanced_level_text(level_text)
    course_style = _is_course_style_level_text(level_text)

    starting_point = "Match the learner level and begin at an appropriate entry point."
    if familiarity == "Brand new":
        starting_point = "Start from the basics, define key terms, and assume no prior knowledge."
    elif familiarity == "I know a little":
        starting_point = "Briefly recap the basics, then build up carefully."
    elif familiarity == "I know the basics":
        starting_point = "Assume the core vocabulary is familiar, skip extended definitions, and move quickly into connections and application."
    elif familiarity == "I want a deeper review":
        starting_point = "Keep the basics brief and spend more time on nuance, depth, and synthesis."
    elif familiarity == "I'm advanced but want structured practice":
        starting_point = "Skip most basic exposition and focus on organized practice and challenge."
    if advanced_like:
        starting_point = "Assume course-level familiarity, skip broad beginner framing, and move quickly into analytical distinctions and expectations."

    pacing_guidance = "Use a steady pace that matches the learner level."
    if purpose == "Learn from scratch":
        pacing_guidance = "Build the lesson gradually from fundamentals to simple application."
    elif purpose == "Review for a test":
        pacing_guidance = "Keep the pace brisk and prioritize high-yield review, patterns, and application."
    elif purpose == "Understand a confusing concept":
        pacing_guidance = "Slow down around the hardest ideas and explicitly clear up likely misconceptions."
    elif purpose == "Practice applying ideas":
        pacing_guidance = "Move quickly into application and worked examples."
    elif purpose == "Prepare to write about it":
        pacing_guidance = "Balance explanation with clear organization and precise phrasing."
    elif purpose == "Build an assignment / lesson":
        pacing_guidance = "Organize the content so it can easily support teaching or lesson planning."
    if advanced_like and familiarity in {"I know the basics", "I want a deeper review", "I'm advanced but want structured practice"}:
        pacing_guidance = "Move briskly, assume the basics are in place, and focus on analysis, interpretation, and disciplined application."

    tone_guidance = "Use a clear, supportive tutoring tone."
    if style == "Simple and beginner-friendly":
        tone_guidance = "Use plain, beginner-friendly language and avoid jargon unless you explain it immediately."
    elif style == "Step-by-step":
        tone_guidance = "Explain ideas in a clear sequence with explicit transitions from one step to the next."
    elif style == "Example-heavy":
        tone_guidance = "Use concrete examples frequently so abstract ideas feel tangible."
    elif style == "Formal/academic":
        tone_guidance = "Use a more formal academic tone while staying readable."
    elif style == "Quiz-focused":
        tone_guidance = "Keep explanations crisp and orient the learner toward checks for understanding and practice."
    elif style == "Concise review":
        tone_guidance = "Keep explanations compact, high-signal, and review-oriented."
    if course_style:
        tone_guidance = f"{tone_guidance} Frame the material with course-style expectations and disciplinary precision."
    if advanced_like:
        tone_guidance = (
            "Use precise disciplinary vocabulary, minimize hand-holding, and write with course-style analytical expectations."
        )

    depth_guidance = "Match the depth to the learner level and learning goals."
    if familiarity in {"Brand new", "I know a little"}:
        depth_guidance = "Prioritize core ideas, definitions, and simple examples before deeper detail."
    elif familiarity in {"I want a deeper review", "I'm advanced but want structured practice"}:
        depth_guidance = "Assume the learner can handle denser detail, stronger comparisons, and more demanding practice."
    if advanced_like:
        depth_guidance = "Assume strong learner capacity, use rigorous distinctions, and emphasize formal reasoning, interpretation, and analytical expectations."

    section_emphasis = "Keep the module balanced across explanation, vocabulary, and application."
    if purpose == "Review for a test":
        section_emphasis = "Shape sections toward review and application rather than broad introductory exposition."
    elif purpose == "Practice applying ideas":
        section_emphasis = "Emphasize worked application, comparison, and transfer."
    elif purpose == "Prepare to write about it":
        section_emphasis = "Emphasize structure, evidence-based explanation, and language the learner can reuse in writing."
    elif purpose == "Build an assignment / lesson":
        section_emphasis = "Emphasize teachable structure, checkpoints, and assignment-ready framing."
    if advanced_like and purpose in {"Review for a test", "Prepare to write about it"}:
        section_emphasis = "Shape sections as course-style review and application, with analytical expectations instead of broad introduction."

    focus_guidance = (
        f"Give extra care to this confusion or focus area when the evidence supports it: {confusion}"
        if confusion
        else "No extra confusion point was provided."
    )

    return {
        "subject": _clean_text(request.subject),
        "learner_level": _clean_text(request.learner_level) or request.audience_level,
        "audience_level": request.audience_level,
        "current_familiarity": familiarity,
        "learning_purpose": purpose,
        "explanation_style": style,
        "confusion_points": confusion,
        "source_preference": _clean_text(request.source_preference),
        "prefer_high_trust_sources": request.prefer_high_trust_sources,
        "starting_point": starting_point,
        "pacing_guidance": pacing_guidance,
        "tone_guidance": tone_guidance,
        "depth_guidance": depth_guidance,
        "section_emphasis": section_emphasis,
        "focus_guidance": focus_guidance,
    }


ACADEMIC_SOURCE_PREFERRED_DOMAINS: Dict[str, List[str]] = {
    "statistics": ["collegeboard.org", "khanacademy.org", "amstat.org", "openstax.org", "britannica.com"],
    "probability": ["khanacademy.org", "openstax.org", "collegeboard.org", "britannica.com"],
    "science": ["khanacademy.org", "openstax.org", "britannica.com", "nih.gov", "ncbi.nlm.nih.gov"],
    "biology": ["khanacademy.org", "openstax.org", "britannica.com", "nih.gov", "ncbi.nlm.nih.gov"],
    "chemistry": ["khanacademy.org", "openstax.org", "britannica.com", "nist.gov"],
    "physics": ["khanacademy.org", "openstax.org", "britannica.com", "nasa.gov", "nist.gov"],
    "history": ["britannica.com", "loc.gov", "archives.gov", "history.state.gov", "si.edu"],
    "literature": ["britannica.com", "poetryfoundation.org", "folger.edu", "khanacademy.org"],
    "default": ["khanacademy.org", "britannica.com", "openstax.org", "vocabulary.com"],
}

BEGINNER_FRIENDLY_SOURCE_DOMAINS = [
    "kids.britannica.com",
    "khanacademy.org",
    "vocabulary.com",
    "britannica.com",
]

HIGH_TRUST_SOURCE_DOMAINS = [
    "khanacademy.org",
    "openstax.org",
    "britannica.com",
    "collegeboard.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "loc.gov",
    "archives.gov",
    "history.state.gov",
    "nasa.gov",
    "si.edu",
]

WEAK_ACADEMIC_SOURCE_PATTERNS = (
    "coursehero.com",
    "chegg.com",
    "studocu.com",
    "bartleby.com",
    "enotes.com",
    "gradesaver.com",
    "cliffsnotes.com",
    "bestcolleges.com",
    "bootcamp",
)


def _academic_source_defaults(subject: str, topic: str) -> List[str]:
    combined = f"{_clean_text(subject)} {_clean_text(topic)}".lower()
    for keyword, domains in ACADEMIC_SOURCE_PREFERRED_DOMAINS.items():
        if keyword == "default":
            continue
        if keyword in combined:
            return domains
    if any(keyword in combined for keyword in ("statistics", "probability", "history", "biology", "chemistry", "physics", "literature", "science")):
        return ACADEMIC_SOURCE_PREFERRED_DOMAINS["default"]
    return []


def _default_source_preferences(
    *,
    subject: str,
    topic: str,
    source_preference: str = "",
    prefer_high_trust_sources: bool = False,
) -> tuple[List[str], bool]:
    cleaned_preference = _clean_text(source_preference)
    preferred: List[str] = []
    academic_mode = False

    if cleaned_preference == "Academic / Educational":
        preferred.extend(_academic_source_defaults(subject, topic) or ACADEMIC_SOURCE_PREFERRED_DOMAINS["default"])
        academic_mode = True
    elif cleaned_preference == "Beginner-friendly":
        preferred.extend(BEGINNER_FRIENDLY_SOURCE_DOMAINS)
    elif cleaned_preference == "Mixed":
        preferred.extend(_academic_source_defaults(subject, topic) or ACADEMIC_SOURCE_PREFERRED_DOMAINS["default"])
        preferred.extend(BEGINNER_FRIENDLY_SOURCE_DOMAINS)
        academic_mode = True
    else:
        preferred.extend(_academic_source_defaults(subject, topic))
        academic_mode = bool(preferred)

    if prefer_high_trust_sources:
        preferred.extend(HIGH_TRUST_SOURCE_DOMAINS)
        academic_mode = True

    deduped: List[str] = []
    seen: set[str] = set()
    for item in preferred:
        normalized = normalize_domain(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped, academic_mode


def _source_quality_score(
    *,
    domain: str,
    url: str,
    preferred_domains: Sequence[str] | None,
    academic_mode: bool,
    prefer_high_trust_sources: bool = False,
) -> int:
    normalized_domain = normalize_domain(domain or url)
    if not normalized_domain:
        return 0

    score = 0
    for preferred in preferred_domains or []:
        normalized_preferred = normalize_domain(preferred)
        if normalized_preferred and (
            normalized_domain == normalized_preferred or normalized_domain.endswith(f".{normalized_preferred}")
        ):
            score += 4
            break

    if academic_mode and (normalized_domain.endswith(".edu") or normalized_domain.endswith(".gov")):
        score += 3

    if prefer_high_trust_sources and any(
        normalized_domain == trusted or normalized_domain.endswith(f".{trusted}")
        for trusted in HIGH_TRUST_SOURCE_DOMAINS
    ):
        score += 2

    if academic_mode and any(pattern in normalized_domain for pattern in WEAK_ACADEMIC_SOURCE_PATTERNS):
        score -= 4

    lowered_url = _clean_text(url).lower()
    if academic_mode and any(fragment in lowered_url for fragment in ("/courses/", "/bootcamp/", "/programs/", "/degrees/")):
        score -= 2
    return score


SECTION_TEMPLATE_OPENERS = (
    r"^\s*this section introduces\s+",
    r"^\s*this section explains\s+",
    r"^\s*here,?\s+you will learn(?: about)?\s+",
    r"^\s*in this section,?\s+you will learn(?: about)?\s+",
    r"^\s*in this section,?\s+",
)


def learning_goal_to_heading(goal: str) -> str:
    cleaned = _clean_text(goal)
    if not cleaned:
        return "Section"
    if re.search(r"\bkey vocabulary\b", cleaned, flags=re.IGNORECASE):
        return "Key Vocabulary"
    if re.search(r"\bexample\b", cleaned, flags=re.IGNORECASE):
        return "Worked Example"

    base = re.sub(rf"^({'|'.join(word.title() for word in DIRECT_OBJECTIVE_VERBS)})\s+", "", cleaned, flags=re.IGNORECASE)
    base = re.sub(r"^(?:what|how|why)\s+", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\bis\b$", "", base, flags=re.IGNORECASE).strip(" .,:;")
    base = re.sub(r"^(?:the|a|an)\s+", "", base, flags=re.IGNORECASE)
    if len(base.split()) > 8:
        base = " ".join(base.split()[:8])
    return _headline_case(base or cleaned)


def _normalize_section_heading(heading: str, learning_goal: str) -> str:
    cleaned = _clean_text(heading).strip(" .,:;")
    if not cleaned:
        return learning_goal_to_heading(learning_goal)
    if _contains_first_person(cleaned):
        return learning_goal_to_heading(learning_goal)
    if re.match(r"^(?:this section|here|in this section)\b", cleaned, flags=re.IGNORECASE):
        return learning_goal_to_heading(learning_goal)
    if re.match(rf"^({'|'.join(DIRECT_OBJECTIVE_VERBS)})\b", cleaned, flags=re.IGNORECASE):
        return learning_goal_to_heading(learning_goal)
    if len(cleaned.split()) > 9:
        return learning_goal_to_heading(learning_goal)
    return _headline_case(cleaned)


def _polish_section_content(content: str) -> str:
    cleaned = str(content or "").strip()
    if not cleaned:
        return ""
    for pattern in SECTION_TEMPLATE_OPENERS:
        updated = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        if updated != cleaned:
            cleaned = updated.lstrip()
            break
    cleaned = cleaned.strip()
    if cleaned:
        cleaned = cleaned[:1].upper() + cleaned[1:]
    return cleaned


def poll_vector_file_status(client: OpenAI, vector_store_id: str, vector_file_id: str) -> str:
    status = "in_progress"
    for _ in range(max(1, settings.vector_poll_attempts)):
        state = _call_openai(
            "vector_file_status",
            lambda: client.vector_stores.files.retrieve(
                vector_store_id=vector_store_id,
                file_id=vector_file_id,
                timeout=settings.upload_timeout_seconds,
            ),
        )
        status = str(getattr(state, "status", status))
        if status in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(max(0.05, settings.vector_poll_sleep_seconds))
    return status


def _location_from_attributes(attributes: Any, *, filename: str = "", result_index: int | None = None) -> str | None:
    data = attributes if isinstance(attributes, dict) else {}
    suffix = Path(filename).suffix.lower()

    for key in ("page", "page_number", "start_page"):
        value = _clean_text(data.get(key))
        if value:
            return f"page:{value}"

    line_start = _clean_text(data.get("line_start") or data.get("start_line"))
    line_end = _clean_text(data.get("line_end") or data.get("end_line"))
    if line_start and line_end:
        return f"lines:{line_start}-{line_end}"
    if line_start:
        return f"line:{line_start}"

    for key in ("location", "chunk", "section", "line"):
        value = _clean_text(data.get(key))
        if value:
            return f"{key}:{value}"

    if result_index is not None:
        if suffix == ".txt":
            return f"chunk:{result_index}"
        if suffix == ".pdf":
            return f"chunk:{result_index}"
        return f"chunk:{result_index}"
    return None


def _extract_doc_candidates(
    client: OpenAI,
    *,
    vector_store_id: str,
    subject: str = "",
    topic: str,
    audience_level: str,
    objective: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    subject_prefix = f"Subject: {subject}\n" if _clean_text(subject) else ""
    query = (
        subject_prefix
        + (
        f"Topic: {topic}\n"
        f"Audience level: {audience_level}\n"
        f"Learning objective: {objective}\n"
        "Retrieve the most relevant source chunks."
        )
    )
    response = _call_openai(
        "doc_retrieval",
        lambda: client.responses.create(
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
            timeout=settings.retrieval_timeout_seconds,
        ),
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
        for result_index, result in enumerate(raw_results, start=1):
            result_dict = _model_to_dict(result)
            snippet = human_readable_snippet(result_dict.get("text"))
            if not snippet:
                continue
            filename = _clean_text(result_dict.get("filename")) or "Uploaded document"
            candidates.append(
                {
                    "source_type": "doc",
                    "title": filename,
                    "url": None,
                    "doc_name": filename,
                    "location": _location_from_attributes(
                        result_dict.get("attributes"),
                        filename=filename,
                        result_index=result_index,
                    )
                    or f"chunk:{result_index}",
                    "snippet": snippet,
                    "retrieved_at": retrieved_at,
                }
            )
    return candidates[: max(1, max_results)]


def _extract_web_sources_from_response(response: Any) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    seen_urls: set[str] = set()

    def add_entry(url: str, *, title_candidates: Sequence[Any] = (), snippet_candidates: Sequence[Any] = ()) -> None:
        clean_url = _clean_text(url)
        if not clean_url or clean_url in seen_urls:
            return
        seen_urls.add(clean_url)
        title = choose_web_title(
            url=clean_url,
            candidates=[_clean_text(candidate) for candidate in title_candidates if _clean_text(candidate)],
        )
        domain = normalize_domain(clean_url)
        snippet = choose_web_snippet(
            objective="",
            title=title,
            domain=domain,
            candidates=snippet_candidates,
        )
        entries.append(
            {
                "url": clean_url,
                "title": title,
                "domain": domain,
                "snippet": snippet,
            }
        )

    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "web_search_call":
            call = _model_to_dict(item)
            action = call.get("action") if isinstance(call.get("action"), dict) else {}
            sources = action.get("sources") if isinstance(action, dict) else None
            if isinstance(sources, list):
                for source in sources:
                    source_dict = _model_to_dict(source)
                    add_entry(
                        _clean_text(source_dict.get("url")),
                        title_candidates=_title_candidates_from_metadata(source_dict),
                        snippet_candidates=_snippet_candidates_from_metadata(source_dict),
                    )
            results = call.get("results") if isinstance(call, dict) else None
            if isinstance(results, list):
                for result in results:
                    result_dict = _model_to_dict(result)
                    add_entry(
                        _clean_text(result_dict.get("url")),
                        title_candidates=_title_candidates_from_metadata(result_dict),
                        snippet_candidates=_snippet_candidates_from_metadata(result_dict),
                    )
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
                add_entry(url, title_candidates=[title], snippet_candidates=[])
    return entries


def _infer_title_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = normalize_domain(parsed.netloc or "") or "Web source"
        path_segments = [segment for segment in parsed.path.split("/") if segment.strip()]
        for segment in reversed(path_segments):
            cleaned = re.sub(r"[-_]+", " ", segment).strip()
            cleaned = re.sub(r"\.[a-z0-9]{1,5}$", "", cleaned, flags=re.IGNORECASE)
            if not cleaned:
                continue
            token_count = len(re.findall(r"[a-zA-Z0-9]+", cleaned))
            if token_count >= 2 or len(cleaned) >= 14:
                return cleaned.strip(" -|:").title()
        return host
    except Exception:
        return "Web source"


def domain_matches_policy(domain: str, allowed_domains: Sequence[str] | None, blocked_domains: Sequence[str] | None) -> bool:
    normalized = normalize_domain(domain)
    if not normalized:
        return False

    blocked = [normalize_domain(item) for item in (blocked_domains or []) if normalize_domain(item)]
    for blocked_domain in blocked:
        if normalized == blocked_domain or normalized.endswith(f".{blocked_domain}"):
            return False

    allowed = [normalize_domain(item) for item in (allowed_domains or []) if normalize_domain(item)]
    if not allowed:
        return True
    for allowed_domain in allowed:
        if normalized == allowed_domain or normalized.endswith(f".{allowed_domain}"):
            return True
    return False


@dataclass(frozen=True)
class WebCandidateResult:
    items: List[Dict[str, Any]]
    filtered_out_by_policy: bool


@dataclass(frozen=True)
class EvidenceBuildResult:
    evidence_pack: List[EvidenceItem]
    web_unavailable_objectives: List[str]
    objectives_without_evidence: List[str]


@dataclass(frozen=True)
class TopicRelatednessResult:
    relation: Literal["closely related", "somewhat related", "unrelated"]
    reason: str


@dataclass(frozen=True)
class TopicBridgeContext:
    relation: Literal["closely related", "somewhat related"]
    previous_topic: str
    previous_objectives: List[str]
    previous_summary: str
    reason: str = ""


@dataclass(frozen=True)
class RetrievalPlan:
    section_limit: int
    doc_results_per_objective: int
    web_results_per_objective: int
    max_evidence_items: int


class OpenAIOperationTimeoutError(RuntimeError):
    def __init__(self, message: str = MODULE_TIMEOUT_MESSAGE) -> None:
        super().__init__(message)


def web_search_tool_args() -> Dict[str, Any]:
    tool = {"type": "web_search", "search_context_size": "medium"}
    if "filters" in tool:
        raise AssertionError("web_search tool args must not include filters.")
    return tool


def _extract_web_candidates(
    client: OpenAI,
    *,
    subject: str = "",
    topic: str,
    audience_level: str,
    objective: str,
    max_results: int,
    web_recency_days: int,
    allowed_domains: Sequence[str] | None,
    blocked_domains: Sequence[str] | None,
    preferred_domains: Sequence[str] | None = None,
    academic_mode: bool = False,
    source_preference: str = "",
    prefer_high_trust_sources: bool = False,
) -> WebCandidateResult:
    def choose_best(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        objective_terms = {
            term
            for term in re.findall(r"[a-zA-Z]+", objective.lower())
            if len(term) >= 4
        }

        def score(item: Dict[str, Any]) -> tuple[int, int, int, int]:
            title_text = str(item.get("title", "")).lower()
            snippet_text = str(item.get("snippet", "")).lower()
            title_hits = sum(1 for term in objective_terms if term in title_text)
            snippet_hits = sum(1 for term in objective_terms if term in snippet_text)
            source_quality = _source_quality_score(
                domain=str(item.get("domain", "")),
                url=str(item.get("url", "")),
                preferred_domains=preferred_domains,
                academic_mode=academic_mode,
                prefer_high_trust_sources=prefer_high_trust_sources,
            )
            quality = len(snippet_text)
            rank = int(item.get("_rank", 0))
            return (title_hits * 3 + snippet_hits, source_quality, quality, -rank)

        best = max(candidates, key=score)
        return [{key: value for key, value in best.items() if not str(key).startswith("_")}]

    normalized_allowed = [normalize_domain(item) for item in (allowed_domains or []) if normalize_domain(item)]
    normalized_blocked = [normalize_domain(item) for item in (blocked_domains or []) if normalize_domain(item)]

    subject_prefix = f"Subject: {subject}\n" if _clean_text(subject) else ""
    prompt = (
        subject_prefix
        + (
        f"Topic: {topic}\n"
        f"Audience level: {audience_level}\n"
        f"Learning objective: {objective}\n"
        f"Recency requirement: prioritize sources from the last {max(1, web_recency_days)} days.\n\n"
        "Search the web and return evidence snippets for teaching this objective. "
        "Only use sources that satisfy domain policy constraints."
        )
    )
    if academic_mode and not normalized_allowed:
        prompt += " Prefer official educational resources, universities, .edu/.gov pages, and major reference sources over commercial course-listing pages."
    if _clean_text(source_preference) == "Beginner-friendly":
        prompt += " Prefer clear, beginner-friendly educational/reference sources that explain ideas plainly."
    elif _clean_text(source_preference) == "Mixed":
        prompt += " Balance high-trust educational/reference sources with clear beginner-friendly explanations."
    if prefer_high_trust_sources:
        prompt += " Prefer high-trust sources when possible."
    response = _call_openai(
        "web_retrieval",
        lambda: client.responses.create(
            model=settings.retrieval_model,
            input=prompt,
            tools=[web_search_tool_args()],
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
            timeout=settings.retrieval_timeout_seconds,
        ),
    )
    payload = parse_json_object(response_text(response))
    raw_items = payload.get("items", [])
    source_entries = _extract_web_sources_from_response(response)
    source_by_url = {entry["url"]: entry for entry in source_entries if entry.get("url")}

    retrieved_at = utc_now()
    parsed: List[Dict[str, Any]] = []
    had_candidate_sources = False
    policy_applied = bool(normalized_allowed or normalized_blocked)
    if isinstance(raw_items, list):
        for raw_index, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                continue
            url = _clean_text(raw.get("url"))
            source_meta = source_by_url.get(url, {})
            title = choose_web_title(
                url=url,
                candidates=[
                    *(_title_candidates_from_metadata(raw)),
                    source_meta.get("title", ""),
                ],
            )
            domain = normalize_domain(url)
            snippet = choose_web_snippet(
                objective=objective,
                title=title,
                domain=domain,
                candidates=[
                    *(_snippet_candidates_from_metadata(raw)),
                    source_meta.get("snippet", ""),
                ],
            )
            if not url or not snippet:
                continue
            had_candidate_sources = True
            if source_entries and url not in source_by_url:
                continue
            if not domain_matches_policy(domain, normalized_allowed, normalized_blocked):
                continue
            parsed.append(
                {
                    "source_type": "web",
                    "domain": domain,
                    "title": title,
                    "url": url,
                    "doc_name": None,
                    "location": None,
                    "snippet": snippet,
                    "retrieved_at": retrieved_at,
                    "_rank": raw_index,
                }
            )

    if not parsed and source_entries:
        for source_index, source in enumerate(source_entries):
            url = _clean_text(source.get("url"))
            domain = normalize_domain(url)
            if not domain:
                continue
            had_candidate_sources = True
            if not domain_matches_policy(domain, normalized_allowed, normalized_blocked):
                continue
            title = choose_web_title(
                url=url,
                candidates=[source.get("title", "")],
            )
            snippet = choose_web_snippet(
                objective=objective,
                title=title,
                domain=domain,
                candidates=[source.get("snippet", ""), source.get("title", "")],
            )
            if not snippet:
                continue
            parsed.append(
                {
                    "source_type": "web",
                    "domain": domain,
                    "title": title,
                    "url": url,
                    "doc_name": None,
                    "location": None,
                    "snippet": snippet,
                    "retrieved_at": retrieved_at,
                    "_rank": source_index,
                }
            )

    filtered_out_by_policy = policy_applied and had_candidate_sources and not parsed
    return WebCandidateResult(
        items=choose_best(parsed[: max(1, max_results)]),
        filtered_out_by_policy=filtered_out_by_policy,
    )


def build_evidence_pack(
    client: OpenAI,
    *,
    subject: str | None = None,
    topic: str,
    audience_level: str,
    learning_objectives: Sequence[str],
    allow_web: bool,
    vector_store_id: str | None,
    source_policy: SourcePolicy | None = None,
    start_index: int = 1,
    fast_mode: bool = False,
    stage_timer: StageTimingLogger | None = None,
    source_preference: str = "",
    prefer_high_trust_sources: bool = False,
) -> EvidenceBuildResult:
    policy = source_policy or SourcePolicy(allow_web=allow_web)
    plan = retrieval_plan(fast_mode)
    effective_allow_web = bool(policy.allow_web)
    allowed_domains = policy.allowed_domains
    blocked_domains = policy.blocked_domains
    preferred_domains, academic_mode = (
        ([], False)
        if allowed_domains
        else _default_source_preferences(
            subject=subject or "",
            topic=topic,
            source_preference=source_preference,
            prefer_high_trust_sources=prefer_high_trust_sources,
        )
    )
    target_objectives: List[str] = []
    for item in learning_objectives:
        normalized = normalize_learning_goal_text(
            item,
            subject=subject or "",
            topic=topic,
            audience_level=audience_level,
        )
        if normalized:
            target_objectives.append(normalized)
    target_objectives = target_objectives[: plan.section_limit]
    if not target_objectives:
        fallback_goal = normalize_learning_goal_text(
            topic,
            subject=subject or "",
            topic=topic,
            audience_level=audience_level,
        )
        target_objectives = [fallback_goal or "Learning goal"]

    if not vector_store_id and not effective_allow_web:
        return EvidenceBuildResult(
            evidence_pack=[],
            web_unavailable_objectives=[],
            objectives_without_evidence=target_objectives,
        )

    objective_states: List[Dict[str, Any]] = []
    retrieval_errors: List[str] = []
    web_unavailable_objectives: List[str] = []
    for objective_index, objective_text in enumerate(target_objectives):
        objective_doc_count = 0
        objective_web_count = 0
        objective_items: List[Dict[str, Any]] = []
        objective_doc_items: List[Dict[str, Any]] = []
        objective_web_items: List[Dict[str, Any]] = []
        if vector_store_id:
            try:
                doc_stage = f"doc_retrieval_{objective_index + 1}"
                if stage_timer is not None:
                    stage_timer.start(doc_stage, objective=objective_text)
                doc_items = _extract_doc_candidates(
                    client,
                    vector_store_id=vector_store_id,
                    subject=subject or "",
                    topic=topic,
                    audience_level=audience_level,
                    objective=objective_text,
                    max_results=plan.doc_results_per_objective,
                )
                objective_doc_count = len(doc_items)
                objective_doc_items = list(doc_items)
                objective_items.extend(doc_items)
                if stage_timer is not None:
                    stage_timer.end(doc_stage, objective=objective_text, result_count=objective_doc_count)
                    stage_timer.log_event(
                        "doc_retrieval.summary",
                        objective=objective_text,
                        result_count=objective_doc_count,
                    )
            except OpenAIOperationTimeoutError:
                if stage_timer is not None:
                    stage_timer.end(doc_stage, objective=objective_text, status="timeout")
                raise
            except Exception as exc:
                if stage_timer is not None:
                    stage_timer.end(doc_stage, objective=objective_text, status="error")
                retrieval_errors.append(f"doc({objective_text}): {type(exc).__name__}: {exc}")
        if effective_allow_web:
            try:
                web_stage = f"web_retrieval_{objective_index + 1}"
                if stage_timer is not None:
                    stage_timer.start(web_stage, objective=objective_text)
                web_result = _extract_web_candidates(
                    client,
                    subject=subject or "",
                    topic=topic,
                    audience_level=audience_level,
                    objective=objective_text,
                    max_results=plan.web_results_per_objective,
                    web_recency_days=policy.web_recency_days,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                    preferred_domains=preferred_domains,
                    academic_mode=academic_mode,
                    source_preference=source_preference,
                    prefer_high_trust_sources=prefer_high_trust_sources,
                )
                objective_web_count = len(web_result.items)
                objective_web_items = list(web_result.items)
                objective_items.extend(web_result.items)
                if stage_timer is not None:
                    stage_timer.end(web_stage, objective=objective_text, result_count=objective_web_count)
                    stage_timer.log_event(
                        "web_retrieval.summary",
                        objective=objective_text,
                        result_count=objective_web_count,
                    )
                if web_result.filtered_out_by_policy:
                    web_unavailable_objectives.append(objective_text)
            except OpenAIOperationTimeoutError:
                if stage_timer is not None:
                    stage_timer.end(web_stage, objective=objective_text, status="timeout")
                raise
            except Exception as exc:
                if stage_timer is not None:
                    stage_timer.end(web_stage, objective=objective_text, status="error")
                retrieval_errors.append(f"web({objective_text}): {type(exc).__name__}: {exc}")
        objective_states.append(
            {
                "objective": objective_text,
                "items": objective_items,
                "doc_items": objective_doc_items,
                "web_items": objective_web_items,
            }
        )

    if not any(state["items"] for state in objective_states) and retrieval_errors:
        detail = " | ".join(retrieval_errors[:4])
        raise RuntimeError(f"Evidence retrieval failed: {detail}")

    doc_pool: List[Dict[str, Any]] = []
    for state in objective_states:
        doc_pool.extend(state["doc_items"])

    objectives_without_evidence: List[str] = []
    for state in objective_states:
        if state["items"]:
            continue
        reused_doc_items = _best_matching_doc_items(
            state["objective"],
            doc_pool,
            limit=max(1, plan.doc_results_per_objective),
        )
        if reused_doc_items:
            state["items"].extend(reused_doc_items)
            state["doc_items"].extend(reused_doc_items)
            continue
        objectives_without_evidence.append(state["objective"])

    deduped: List[Dict[str, Any]] = []
    seen = set()
    remaining_groups = [list(state["items"]) for state in objective_states if state["items"]]
    while remaining_groups and len(deduped) < plan.max_evidence_items:
        next_groups: List[List[Dict[str, Any]]] = []
        for group in remaining_groups:
            chosen_item: Dict[str, Any] | None = None
            while group and chosen_item is None:
                candidate = group.pop(0)
                key = (
                    candidate.get("source_type"),
                    candidate.get("url") or candidate.get("doc_name") or candidate.get("title"),
                    candidate.get("snippet"),
                )
                if key in seen:
                    continue
                seen.add(key)
                chosen_item = candidate
            if chosen_item is not None:
                deduped.append(chosen_item)
            if group:
                next_groups.append(group)
            if len(deduped) >= plan.max_evidence_items:
                break
        remaining_groups = next_groups

    evidence_pack: List[EvidenceItem] = []
    index = max(1, start_index)
    for item in deduped:
        evidence_pack.append(
            EvidenceItem(
                evidence_id=f"E{index:03d}",
                source_type=item["source_type"],
                domain=item.get("domain"),
                title=item["title"],
                url=item.get("url"),
                doc_name=item.get("doc_name"),
                location=item.get("location"),
                snippet=item["snippet"],
                retrieved_at=item.get("retrieved_at") or utc_now(),
            )
        )
        index += 1
    return EvidenceBuildResult(
        evidence_pack=evidence_pack,
        web_unavailable_objectives=sorted(set(web_unavailable_objectives)),
        objectives_without_evidence=sorted(set(objectives_without_evidence)),
    )


def build_unverified_section(
    *,
    section_id: str,
    objective_index: int,
    learning_goal: str,
    heading: str,
    reason: str,
) -> ModuleSection:
    return ModuleSection(
        section_id=section_id,
        objective_index=objective_index,
        learning_goal=learning_goal,
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
    goals = selected_learning_objectives(request)
    subject_text = _clean_text(request.subject)
    sections: List[ModuleSection] = []
    for idx, goal in enumerate(goals):
        sections.append(
            build_unverified_section(
                section_id=f"section-{idx + 1}",
                objective_index=idx,
                learning_goal=goal,
                heading=learning_goal_to_heading(goal or f"Section {idx + 1}"),
                reason=reason,
            )
        )
    return Module(
        module_id=module_id,
        title=f"{request.topic} - Grounded Tutoring Module",
        overview=(
            f"Introductory module for {request.audience_level} learners on {request.topic}"
            f"{f' in {subject_text}' if subject_text else ''}."
        ),
        sections=sections,
        glossary=[],
        mcqs=[],
    )


def detect_topic_relatedness(
    client: OpenAI,
    *,
    topic: str,
    learning_objectives: Sequence[str],
    previous_context: TopicContinuityContext,
) -> TopicRelatednessResult:
    payload = {
        "new_module": {
            "topic": topic,
            "learning_objectives": [_clean_text(item) for item in learning_objectives if _clean_text(item)],
        },
        "previous_module": previous_context.model_dump(mode="json"),
    }
    response = _call_openai(
        "topic_relatedness",
        lambda: client.responses.create(
            model=settings.generation_model,
            instructions=(
                "Compare the new module topic to the previous module topic for tutoring continuity.\n"
                "Return 'closely related' when the new topic directly builds on or extends the earlier one.\n"
                "Return 'somewhat related' when the topics are adjacent or commonly connected in learning sequences.\n"
                "Return 'unrelated' when the learner would not expect a meaningful bridge."
            ),
            input=json.dumps(payload, ensure_ascii=True),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "topic_relatedness",
                    "schema": TOPIC_RELATEDNESS_SCHEMA,
                    "strict": True,
                }
            },
            temperature=0,
            max_output_tokens=250,
            timeout=settings.retrieval_timeout_seconds,
        ),
    )
    parsed = parse_json_object(response_text(response))
    relation = str(parsed.get("relation", "unrelated")).strip().lower()
    if relation not in {"closely related", "somewhat related", "unrelated"}:
        relation = "unrelated"
    reason = _clean_text(parsed.get("reason"))
    return TopicRelatednessResult(relation=relation, reason=reason)


def build_topic_bridge_context(
    relatedness: TopicRelatednessResult,
    previous_context: TopicContinuityContext,
) -> TopicBridgeContext | None:
    if relatedness.relation not in {"closely related", "somewhat related"}:
        return None
    return TopicBridgeContext(
        relation=relatedness.relation,
        previous_topic=_clean_text(previous_context.topic) or "the previous topic",
        previous_objectives=[_clean_text(item) for item in previous_context.learning_objectives if _clean_text(item)],
        previous_summary=_clean_text(previous_context.module_summary),
        reason=_clean_text(relatedness.reason),
    )


def _bridge_overview_prefix(request: ModuleGenerateRequest, topic_bridge: TopicBridgeContext) -> str:
    if topic_bridge.relation == "closely related":
        return (
            f"This module builds directly from the previous module on {topic_bridge.previous_topic}, "
            f"helping learners carry that foundation into {request.topic}."
        )
    return (
        f"This module connects to the previous module on {topic_bridge.previous_topic} "
        f"and helps learners shift into the related topic of {request.topic}."
    )


def _bridge_section_intro(request: ModuleGenerateRequest, topic_bridge: TopicBridgeContext) -> str:
    relation_text = "directly builds on" if topic_bridge.relation == "closely related" else "connects with"
    return (
        f"Bridge from the last module: if learners just studied {topic_bridge.previous_topic}, "
        f"this section {relation_text} that work and turns to {request.topic}."
    )


def _apply_topic_bridge(
    module: Module,
    *,
    request: ModuleGenerateRequest,
    topic_bridge: TopicBridgeContext | None,
) -> Module:
    if topic_bridge is None:
        return module

    overview_prefix = _bridge_overview_prefix(request, topic_bridge)
    overview = str(module.overview or "").strip()
    if topic_bridge.previous_topic.lower() not in overview.lower():
        overview = f"{overview_prefix} {overview}".strip()

    sections = [section.model_copy(deep=True) for section in module.sections]
    if sections:
        first_section = sections[0]
        intro = _bridge_section_intro(request, topic_bridge)
        content = str(first_section.content or "").strip()
        if topic_bridge.previous_topic.lower() not in content.lower():
            content = f"{intro}\n\n{content}".strip()
        sections[0] = first_section.model_copy(update={"content": content})

    return module.model_copy(update={"overview": overview, "sections": sections})


def _enforce_objective_section_structure(
    module: Module,
    request: ModuleGenerateRequest,
    *,
    objectives_without_evidence: Sequence[str] | None = None,
) -> Module:
    goals = selected_learning_objectives(request)
    missing_goals = set(item.strip() for item in (objectives_without_evidence or []) if item and item.strip())
    enforced_sections: List[ModuleSection] = []
    for idx, goal in enumerate(goals):
        if idx < len(module.sections):
            base = module.sections[idx]
            citations = list(base.citations)
            unverified = bool(base.unverified)
            reason = _clean_text(base.unverified_reason)
            if goal in missing_goals and not citations:
                unverified = True
                reason = reason or "No retrievable evidence was available for this learning objective."
            enforced_sections.append(
                base.model_copy(
                    update={
                        "section_id": f"section-{idx + 1}",
                        "objective_index": idx,
                        "learning_goal": goal,
                        "heading": _normalize_section_heading(_clean_text(base.heading), goal or f"Section {idx + 1}"),
                        "content": _polish_section_content(base.content)
                        or "No verifiable instructional content could be generated from retrieved evidence.",
                        "unverified": unverified,
                        "unverified_reason": reason,
                    }
                )
            )
            continue

        fallback_reason = "No section content was generated for this learning objective."
        if goal in missing_goals:
            fallback_reason = "No retrievable evidence was available for this learning objective."
        enforced_sections.append(
            build_unverified_section(
                section_id=f"section-{idx + 1}",
                objective_index=idx,
                learning_goal=goal,
                heading=learning_goal_to_heading(goal or f"Section {idx + 1}"),
                reason=fallback_reason,
            )
        )

    overview = _clean_text(module.overview) or f"Learning module for {request.topic}."
    return module.model_copy(update={"overview": overview, "sections": enforced_sections})


def generate_module_from_evidence(
    client: OpenAI,
    *,
    request: ModuleGenerateRequest,
    evidence_pack: Sequence[EvidenceItem],
    module_id: str,
    web_unavailable_objectives: Sequence[str] | None = None,
    objectives_without_evidence: Sequence[str] | None = None,
    topic_bridge: TopicBridgeContext | None = None,
) -> Module:
    if not evidence_pack:
        return build_unverified_module(
            module_id=module_id,
            request=request,
            reason="No evidence was retrieved from the configured sources.",
        )

    evidence_payload = [item.model_dump(mode="json") for item in evidence_pack]
    allowed_ids = [item.evidence_id for item in evidence_pack]
    target_goals = selected_learning_objectives(request)
    target_section_count = len(target_goals)
    personalization = build_personalization_context(request)
    prompt_payload = {
        "module_request": {
            "subject": request.subject,
            "topic": request.topic,
            "audience_level": request.audience_level,
            "learning_objectives": target_goals,
            "personalization": personalization,
            "fast_mode": request.fast_mode,
        },
        "web_unavailable_objectives": list(web_unavailable_objectives or []),
        "objectives_without_evidence": list(objectives_without_evidence or []),
        "allowed_evidence_ids": allowed_ids,
        "evidence_pack": evidence_payload,
    }
    if topic_bridge is not None:
        prompt_payload["topic_continuity"] = {
            "relation": topic_bridge.relation,
            "previous_topic": topic_bridge.previous_topic,
            "previous_objectives": topic_bridge.previous_objectives,
            "previous_summary": topic_bridge.previous_summary,
            "reason": topic_bridge.reason,
        }

    response = _call_openai(
        "module_generation",
        lambda: client.responses.create(
            model=settings.generation_model,
            instructions=(
                "You are a tutoring-module generator.\n"
                "Use only evidence from evidence_pack. Never invent facts.\n"
                f"Create exactly {target_section_count} sections, one per learning objective in order.\n"
                "Each section must include objective_index and learning_goal.\n"
                "Every verified section must cite evidence_id values from allowed_evidence_ids.\n"
                "If an objective appears in objectives_without_evidence, mark that section unverified.\n"
                "If an objective appears in web_unavailable_objectives, web evidence was filtered by policy.\n"
                "Use module_request.personalization to shape tone, depth, pacing, explanation style, and whether to start with basics or move ahead.\n"
                "If module_request.personalization.confusion_points is present, give extra support around that confusion when evidence allows.\n"
                "Use the subject field to keep framing and examples in the right discipline.\n"
                "Avoid repetitive tutorial framing such as 'This section introduces...' or 'Here, you will learn...'. Write in direct instructional prose.\n"
                "If topic_continuity is present and relation is not unrelated, add a brief learner-friendly bridge in the overview and first section.\n"
                "If fast_mode is true, keep sections concise and return at most 2 MCQs.\n"
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
            timeout=settings.generation_timeout_seconds,
        ),
    )
    payload = parse_json_object(response_text(response))
    module = Module.model_validate(payload)
    module = module.model_copy(update={"module_id": module_id})
    module = _enforce_objective_section_structure(
        module,
        request,
        objectives_without_evidence=objectives_without_evidence,
    )
    if request.fast_mode:
        module = module.model_copy(update={"mcqs": list(module.mcqs[:2])})
    module = _apply_topic_bridge(module, request=request, topic_bridge=topic_bridge)
    return module.model_copy(
        update={
            "sections": [
                section.model_copy(
                    update={
                        "heading": _normalize_section_heading(section.heading, section.learning_goal),
                        "content": _polish_section_content(section.content) or section.content,
                    }
                )
                for section in module.sections
            ]
        }
    )


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
            objective_index=target_section.objective_index,
            learning_goal=target_section.learning_goal,
            heading=target_section.heading,
            reason="No evidence was retrieved for this regeneration request.",
        )

    prompt_payload = {
        "module_context": {
            "module_id": module.module_id,
            "module_title": module.title,
            "subject": request.subject,
            "topic": request.topic,
            "audience_level": request.audience_level,
            "learning_objectives": selected_learning_objectives(request),
            "personalization": build_personalization_context(request),
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

    response = _call_openai(
        "section_generation",
        lambda: client.responses.create(
            model=settings.generation_model,
            instructions=(
                "Regenerate one section using only evidence_pack.\n"
                "Use module_context.personalization to keep the same tone, depth, pacing, and learner support.\n"
                "Avoid repetitive tutorial framing such as 'This section introduces...' or 'Here, you will learn...'. Write in direct instructional prose.\n"
                "Return objective_index and learning_goal fields for the section.\n"
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
            timeout=settings.generation_timeout_seconds,
        ),
    )
    payload = parse_json_object(response_text(response))
    section = ModuleSection.model_validate(payload)
    return section.model_copy(
        update={
            "section_id": target_section.section_id,
            "objective_index": target_section.objective_index,
            "learning_goal": target_section.learning_goal,
            "heading": target_section.heading,
            "content": _polish_section_content(section.content) or section.content,
        }
    )


def _default_tutor_fallback(message: str) -> ModuleAskResponse:
    return ModuleAskResponse(
        answer=message,
        citations=[],
        unverified=True,
    )


def _module_tutor_context(module: Module) -> Dict[str, Any]:
    return {
        "module_id": module.module_id,
        "module_title": module.title,
        "overview": module.overview,
        "sections": [
            {
                "section_id": section.section_id,
                "objective_index": section.objective_index,
                "learning_goal": section.learning_goal,
                "heading": section.heading,
                "content": section.content,
                "citations": list(section.citations),
                "unverified": section.unverified,
            }
            for section in module.sections
        ],
        "glossary": [item.model_dump(mode="json") for item in module.glossary],
        "mcqs": [item.model_dump(mode="json") for item in module.mcqs],
    }


def _assignment_fallback(module: Module) -> ModuleAssignmentResponse:
    criteria = []
    for learning_goal in _supported_rubric_criteria(module)[:3]:
        criteria.append(
            RubricCriterion.model_validate(
                {
                    "criteria": learning_goal,
                    "levels": [
                        {"score": 4, "description": "Explains the idea accurately, clearly, and with supporting detail from the module."},
                        {"score": 3, "description": "Explains the idea correctly with minor gaps or limited detail."},
                        {"score": 2, "description": "Shows partial understanding but misses important details or connections."},
                        {"score": 1, "description": "Shows limited understanding or gives an unsupported response."},
                    ],
                }
            )
        )

    if not criteria:
        criteria.append(
            RubricCriterion.model_validate(
                {
                    "criteria": "Use the key ideas from the module accurately.",
                    "levels": [
                        {"score": 4, "description": "Uses the main ideas accurately and clearly."},
                        {"score": 3, "description": "Uses most main ideas accurately."},
                        {"score": 2, "description": "Uses some main ideas but with gaps."},
                        {"score": 1, "description": "Shows little accurate use of the module ideas."},
                    ],
                }
            )
        )

    return ModuleAssignmentResponse(
        prompt=(
            f"Write a short response that teaches the most important ideas from {module.title}. "
            "Use evidence and vocabulary from the module, and include one concrete example."
        ),
        rubric=criteria,
    )


def _supported_rubric_criteria(module: Module) -> List[str]:
    supported = [
        _clean_text(section.learning_goal) or _clean_text(section.heading)
        for section in module.sections
        if _clean_text(section.learning_goal) or _clean_text(section.heading)
    ]
    if supported:
        return supported[:5]
    fallback = _clean_text(module.overview) or _clean_text(module.title) or "Use the key ideas from the module accurately."
    return [fallback]


def _normalize_assignment_payload(
    payload: Dict[str, Any],
    *,
    module: Module,
) -> ModuleAssignmentResponse:
    prompt = _clean_text(payload.get("prompt"))
    rubric_items = payload.get("rubric", [])
    try:
        raw_rubric = [RubricCriterion.model_validate(item) for item in rubric_items if isinstance(item, dict)]
    except Exception:
        raw_rubric = []

    supported_criteria = _supported_rubric_criteria(module)
    rubric = [
        item.model_copy(update={"criteria": supported_criteria[index]})
        for index, item in enumerate(raw_rubric[: len(supported_criteria)])
    ]

    if not prompt or not rubric:
        return _assignment_fallback(module)
    return ModuleAssignmentResponse(prompt=prompt, rubric=rubric)


def _unsupported_grade_response(message: str) -> ModuleGradeResponse:
    return ModuleGradeResponse(
        score=0,
        feedback=message,
        breakdown=[],
        unverified=True,
    )


def _normalize_grade_payload(
    payload: Dict[str, Any],
    *,
    rubric: Sequence[RubricCriterion],
    fallback_feedback: str,
) -> ModuleGradeResponse:
    score = int(payload.get("score", 0) or 0)
    feedback = _clean_text(payload.get("feedback")) or fallback_feedback
    unverified = bool(payload.get("unverified"))
    rubric_max = sum(max((level.score for level in criterion.levels), default=4) for criterion in rubric) or 1
    score = max(0, min(score, 100))

    rubric_by_criteria = {_clean_text(item.criteria): item for item in rubric}
    breakdown_items: List[ModuleGradeBreakdownItem] = []
    for raw_item in payload.get("breakdown", []):
        if not isinstance(raw_item, dict):
            continue
        criteria_key = _clean_text(raw_item.get("criteria"))
        if not criteria_key or criteria_key not in rubric_by_criteria:
            continue
        rubric_item = rubric_by_criteria[criteria_key]
        max_score = max((level.score for level in rubric_item.levels), default=4)
        awarded = int(raw_item.get("score", 0) or 0)
        awarded = max(0, min(awarded, max_score))
        item_feedback = _clean_text(raw_item.get("feedback")) or "No criterion-specific feedback was provided."
        breakdown_items.append(
            ModuleGradeBreakdownItem(
                criteria=rubric_item.criteria,
                score=awarded,
                max_score=max_score,
                feedback=item_feedback,
            )
        )

    if not breakdown_items:
        if "saved sources" not in feedback.lower() and "cannot" not in feedback.lower():
            feedback = f"{feedback} I cannot verify a detailed rubric-based grade from this module's saved sources."
        return ModuleGradeResponse(score=0, feedback=feedback, breakdown=[], unverified=True)

    if len(breakdown_items) < len(rubric):
        unverified = True

    max_points = sum(item.max_score for item in breakdown_items) or rubric_max
    awarded_points = sum(item.score for item in breakdown_items)
    normalized_score = round((awarded_points / max_points) * 100)
    if not unverified:
        score = normalized_score

    return ModuleGradeResponse(
        score=score if unverified else normalized_score,
        feedback=feedback,
        breakdown=breakdown_items,
        unverified=unverified,
    )


def _normalize_tutor_answer_payload(
    payload: Dict[str, Any],
    *,
    evidence_pack: Sequence[EvidenceItem],
    fallback_answer: str,
) -> ModuleAskResponse:
    answer = _clean_text(payload.get("answer"))
    citations = [
        str(item).strip()
        for item in payload.get("citations", [])
        if str(item).strip()
    ]
    citations = list(dict.fromkeys(citations))
    allowed_ids = {item.evidence_id for item in evidence_pack}
    citations = [item for item in citations if item in allowed_ids]
    unverified = bool(payload.get("unverified"))

    if not answer:
        answer = fallback_answer
        unverified = True

    if not citations:
        unverified = True
        if "saved sources" not in answer.lower() and "cannot" not in answer.lower():
            answer = f"{answer} I cannot verify more than that from this module's saved sources."

    return ModuleAskResponse(
        answer=answer,
        citations=citations,
        unverified=unverified,
    )


def _tutor_mode_instruction(mode: str) -> str:
    if mode == "simpler":
        return "Use simpler language, shorter sentences, and one concrete example when possible."
    if mode == "more_detailed":
        return "Give a fuller explanation with a bit more detail and structure, but stay concise."
    return "Answer at a normal tutoring level."


def answer_question_from_module(
    client: OpenAI,
    *,
    module: Module,
    question: str,
    evidence_pack: Sequence[EvidenceItem],
    mode: str = "default",
    quiz_prompt: str | None = None,
) -> ModuleAskResponse:
    clean_question = _clean_text(question)
    clean_quiz_prompt = _clean_text(quiz_prompt)
    if not clean_question and mode != "quiz_me":
        return _default_tutor_fallback("Please ask a more specific question about this module.")

    if not evidence_pack:
        return _default_tutor_fallback(
            "I cannot answer that confidently from this module because no cached source evidence is available. "
            "Please refresh sources or ask about material already shown in the module."
        )

    base_payload = {
        "module_context": _module_tutor_context(module),
        "allowed_evidence_ids": [item.evidence_id for item in evidence_pack],
        "evidence_pack": [item.model_dump(mode="json") for item in evidence_pack],
    }

    if mode == "quiz_me" and not clean_quiz_prompt:
        prompt_payload = dict(base_payload)
        prompt_payload["request"] = "Create one short quiz question based on the module."
        response = _call_openai(
            "module_tutor_quiz_question",
            lambda: client.responses.create(
                model=settings.generation_model,
                instructions=(
                    "Generate one short quiz question based only on module_context and evidence_pack.\n"
                    "The answer field must contain only the quiz question.\n"
                    "Citations must reference the evidence_id values that support the question.\n"
                    "If you cannot make a supported quiz question, set unverified=true and answer cautiously."
                ),
                input=json.dumps(prompt_payload, ensure_ascii=True),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "grounded_quiz_question",
                        "schema": ASK_RESPONSE_SCHEMA,
                        "strict": True,
                    }
                },
                temperature=0.2,
                max_output_tokens=400,
                timeout=settings.generation_timeout_seconds,
            ),
        )
        return _normalize_tutor_answer_payload(
            parse_json_object(response_text(response)),
            evidence_pack=evidence_pack,
            fallback_answer="I could not make a supported quiz question from this module's saved sources.",
        )

    if mode == "quiz_me":
        prompt_payload = dict(base_payload)
        prompt_payload["quiz_prompt"] = clean_quiz_prompt
        prompt_payload["learner_answer"] = clean_question
        response = _call_openai(
            "module_tutor_quiz_feedback",
            lambda: client.responses.create(
                model=settings.generation_model,
                instructions=(
                    "You are giving feedback on one learner answer.\n"
                    "Use only module_context and evidence_pack.\n"
                    "The answer field should briefly say what the learner got right, what needs correction, and what the supported answer is.\n"
                    "Citations must use evidence_id values from allowed_evidence_ids.\n"
                    "If the saved evidence cannot support clear feedback, answer cautiously and set unverified=true."
                ),
                input=json.dumps(prompt_payload, ensure_ascii=True),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "grounded_quiz_feedback",
                        "schema": ASK_RESPONSE_SCHEMA,
                        "strict": True,
                    }
                },
                temperature=0.1,
                max_output_tokens=700,
                timeout=settings.generation_timeout_seconds,
            ),
        )
        return _normalize_tutor_answer_payload(
            parse_json_object(response_text(response)),
            evidence_pack=evidence_pack,
            fallback_answer="I could not verify clear quiz feedback from this module's saved sources.",
        )

    prompt_payload = dict(base_payload)
    prompt_payload["question"] = clean_question
    response = _call_openai(
        "module_tutor_answer",
        lambda: client.responses.create(
            model=settings.generation_model,
            instructions=(
                "Answer the learner's question using only module_context and evidence_pack.\n"
                "Treat module_context as supporting wording, but only mark an answer verified when it is backed by "
                "specific evidence_id values from allowed_evidence_ids.\n"
                "If the answer cannot be supported from cached evidence, answer cautiously, return no citations, "
                "and set unverified=true.\n"
                f"{_tutor_mode_instruction(mode)}\n"
                "Do not use tools. Do not invent facts. Keep the answer teacher-friendly."
            ),
            input=json.dumps(prompt_payload, ensure_ascii=True),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "grounded_module_answer",
                    "schema": ASK_RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
            temperature=0.1,
            max_output_tokens=900,
            timeout=settings.generation_timeout_seconds,
        ),
    )
    return _normalize_tutor_answer_payload(
        parse_json_object(response_text(response)),
        evidence_pack=evidence_pack,
        fallback_answer="I could not find enough support in this module's saved sources to answer that confidently.",
    )


def generate_assignment_from_module(
    client: OpenAI,
    *,
    module: Module,
    evidence_pack: Sequence[EvidenceItem],
) -> ModuleAssignmentResponse:
    prompt_payload = {
        "module_context": _module_tutor_context(module),
        "evidence_pack": [item.model_dump(mode="json") for item in evidence_pack],
        "allowed_evidence_ids": [item.evidence_id for item in evidence_pack],
    }
    response = _call_openai(
        "module_assignment",
        lambda: client.responses.create(
            model=settings.generation_model,
            instructions=(
                "Create one assignment prompt and a rubric using only module_context and evidence_pack.\n"
                "Do not invent criteria that are not grounded in the module's sections, glossary, or supported ideas.\n"
                "Return 3 to 5 rubric criteria.\n"
                "Each criterion must contain exactly four levels with scores 4, 3, 2, and 1 in that order.\n"
                "Keep the assignment teacher-friendly and appropriate for the module audience."
            ),
            input=json.dumps(prompt_payload, ensure_ascii=True),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "module_assignment",
                    "schema": ASSIGNMENT_RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
            temperature=0.2,
            max_output_tokens=1400,
            timeout=settings.generation_timeout_seconds,
        ),
    )
    return _normalize_assignment_payload(
        parse_json_object(response_text(response)),
        module=module,
    )


def grade_assignment_from_module(
    client: OpenAI,
    *,
    module: Module,
    evidence_pack: Sequence[EvidenceItem],
    student_response: str,
    rubric: Sequence[RubricCriterion],
) -> ModuleGradeResponse:
    clean_response = _clean_text(student_response)
    if not clean_response:
        return _unsupported_grade_response("Please provide a student response to grade.")
    if not evidence_pack:
        return _unsupported_grade_response(
            "I cannot grade this confidently because the module has no saved source evidence available."
        )

    prompt_payload = {
        "module_context": _module_tutor_context(module),
        "evidence_pack": [item.model_dump(mode="json") for item in evidence_pack],
        "allowed_evidence_ids": [item.evidence_id for item in evidence_pack],
        "rubric": [item.model_dump(mode="json") for item in rubric],
        "student_response": clean_response,
    }
    response = _call_openai(
        "module_assignment_grading",
        lambda: client.responses.create(
            model=settings.generation_model,
            instructions=(
                "Grade the student response using only module_context, evidence_pack, and the provided rubric.\n"
                "Do not invent new criteria.\n"
                "Use the rubric criteria exactly as given.\n"
                "For each breakdown item, use one of the rubric's score levels for that criterion.\n"
                "If the module's saved evidence cannot support a confident grade, set unverified=true and answer cautiously.\n"
                "Keep the feedback specific, constructive, and teacher-friendly."
            ),
            input=json.dumps(prompt_payload, ensure_ascii=True),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "module_assignment_grade",
                    "schema": GRADE_RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
            temperature=0.1,
            max_output_tokens=1500,
            timeout=settings.generation_timeout_seconds,
        ),
    )
    return _normalize_grade_payload(
        parse_json_object(response_text(response)),
        rubric=rubric,
        fallback_feedback="I could not verify a detailed rubric-based grade from this module's saved sources.",
    )


def enforce_quality_gate(module: Module, evidence_pack: Sequence[EvidenceItem]) -> Module:
    evidence_ids = {item.evidence_id for item in evidence_pack}
    normalized_sections: List[ModuleSection] = []
    objective_indexes: List[int] = []

    for section in module.sections:
        objective_indexes.append(section.objective_index)
        if not _clean_text(section.learning_goal):
            raise ValueError(f"Section '{section.heading}' is missing learning_goal.")
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

    expected_indexes = list(range(len(module.sections)))
    if sorted(objective_indexes) != expected_indexes:
        raise ValueError("Section objective_index values must span 0..n-1 with no gaps.")

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
    goals = selected_learning_objectives(request)
    if 0 <= section_index < len(goals):
        candidate = _clean_text(goals[section_index])
        if candidate:
            return candidate
    return _clean_text(fallback_heading) or "Regenerated section objective"
