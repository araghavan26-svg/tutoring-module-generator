from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocsUploadItem(BaseModel):
    file_id: str
    vector_store_file_id: Optional[str] = None
    filename: str
    bytes: int = Field(..., ge=0)
    status: str
    indexed_at: datetime = Field(default_factory=utc_now)


class DocsUploadResponse(BaseModel):
    vector_store_id: str
    docs: List[DocsUploadItem] = Field(default_factory=list)


def normalize_domain(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or parsed.path or "").strip().lower()
    host = host.split("/", 1)[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


class SourcePolicy(BaseModel):
    allow_web: bool = False
    web_recency_days: int = Field(default=30, ge=1, le=3650)
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None

    @field_validator("allowed_domains", "blocked_domains")
    @classmethod
    def normalize_domains(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned: List[str] = []
        for item in value:
            normalized = normalize_domain(str(item or ""))
            if normalized:
                cleaned.append(normalized)
        if not cleaned:
            return None
        return sorted(set(cleaned))


class TopicContinuityContext(BaseModel):
    topic: str = Field(..., min_length=1)
    learning_objectives: List[str] = Field(default_factory=list)
    module_summary: str = Field(default="")


def split_learning_request(value: str) -> Tuple[Optional[str], str]:
    cleaned = (value or "").strip()
    if not cleaned:
        return None, ""
    for separator in (" - ", ":", " – ", " — "):
        if separator not in cleaned:
            continue
        left, right = (part.strip() for part in cleaned.split(separator, 1))
        if left and right:
            return left, right
    return None, cleaned


class ModuleGenerateRequest(BaseModel):
    learning_request: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    audience_level: str = Field(..., min_length=1)
    learner_level: Optional[str] = None
    custom_level_description: Optional[str] = None
    current_familiarity: Optional[str] = None
    learning_purpose: Optional[str] = None
    explanation_style: Optional[str] = None
    confusion_points: Optional[str] = None
    source_preference: Optional[Literal["General", "Academic / Educational", "Beginner-friendly", "Mixed"]] = None
    prefer_high_trust_sources: bool = False
    learning_objectives: List[str] = Field(..., min_length=1)
    allow_web: bool = False
    vector_store_id: Optional[str] = None
    source_policy: Optional[SourcePolicy] = None
    related_to_previous: bool = False
    fast_mode: bool = False

    @field_validator("learning_objectives")
    @classmethod
    def validate_objectives(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("learning_objectives must contain at least one non-empty objective.")
        return cleaned

    @field_validator(
        "subject",
        "learning_request",
        "topic",
        "audience_level",
        "learner_level",
        "custom_level_description",
        "current_familiarity",
        "learning_purpose",
        "explanation_style",
        "confusion_points",
        "source_preference",
        mode="before",
    )
    @classmethod
    def clean_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value

    @model_validator(mode="after")
    def normalize_personalization(self) -> "ModuleGenerateRequest":
        if self.learning_request:
            parsed_subject, parsed_topic = split_learning_request(self.learning_request)
            if parsed_subject and not self.subject:
                self.subject = parsed_subject
            if parsed_topic and not self.topic:
                self.topic = parsed_topic

        if self.learner_level == "Custom":
            if not self.custom_level_description:
                raise ValueError("custom_level_description is required when learner_level is Custom.")
            self.audience_level = self.custom_level_description
        elif self.learner_level and not self.audience_level:
            self.audience_level = self.learner_level

        self.topic = (self.topic or "").strip()
        if not self.topic:
            raise ValueError("topic must not be empty.")
        self.subject = (self.subject or "").strip() or None
        self.audience_level = self.audience_level.strip()
        if not self.audience_level:
            raise ValueError("audience_level must not be empty.")
        return self

    def effective_source_policy(self) -> SourcePolicy:
        if self.source_policy is not None:
            return self.source_policy
        return SourcePolicy(allow_web=self.allow_web)

    def personalization_context(self) -> dict:
        return {
            "learning_request": self.learning_request,
            "subject": self.subject,
            "learner_level": self.learner_level or self.audience_level,
            "audience_level": self.audience_level,
            "custom_level_description": self.custom_level_description,
            "current_familiarity": self.current_familiarity,
            "learning_purpose": self.learning_purpose,
            "explanation_style": self.explanation_style,
            "confusion_points": self.confusion_points,
            "source_preference": self.source_preference,
            "prefer_high_trust_sources": self.prefer_high_trust_sources,
        }


class EvidenceItem(BaseModel):
    evidence_id: str = Field(..., min_length=2)
    source_type: Literal["doc", "web"]
    domain: Optional[str] = None
    title: str = Field(..., min_length=1)
    url: Optional[str] = None
    doc_name: Optional[str] = None
    location: Optional[str] = None
    snippet: str = Field(..., min_length=1)
    retrieved_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_source_fields(self) -> "EvidenceItem":
        if self.source_type == "web" and not self.url:
            raise ValueError("Web evidence must include a URL.")
        if self.source_type == "web":
            domain = normalize_domain(self.domain or self.url or "")
            if not domain:
                raise ValueError("Web evidence must include a domain.")
            self.domain = domain
        if self.source_type == "doc" and not self.doc_name:
            self.doc_name = self.title
        if self.source_type == "doc":
            self.domain = None
        return self


class ModuleSection(BaseModel):
    section_id: str = Field(..., min_length=1)
    objective_index: int = Field(default=0, ge=0)
    learning_goal: str = Field(default="")
    heading: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    citations: List[str] = Field(default_factory=list)
    unverified: bool = False
    unverified_reason: str = ""


class GlossaryItem(BaseModel):
    term: str = Field(..., min_length=1)
    definition: str = Field(..., min_length=1)


class McqItem(BaseModel):
    question: str = Field(..., min_length=1)
    options: List[str] = Field(default_factory=list, min_length=2)
    answer_index: int = Field(..., ge=0)
    explanation: str = Field(default="")


class Module(BaseModel):
    module_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    overview: str = Field(default="")
    sections: List[ModuleSection] = Field(default_factory=list, min_length=1)
    glossary: List[GlossaryItem] = Field(default_factory=list)
    mcqs: List[McqItem] = Field(default_factory=list)
    evidence_pack: List[EvidenceItem] = Field(default_factory=list)
    source_policy: Optional[SourcePolicy] = None
    share_enabled: bool = False
    share_id: Optional[str] = None

    @model_validator(mode="after")
    def normalize_share_state(self) -> "Module":
        if not self.share_enabled:
            self.share_id = None
        elif self.share_id is not None:
            self.share_id = str(self.share_id).strip() or None
        return self


class ModuleGenerateResponse(BaseModel):
    module: Module
    evidence_pack: List[EvidenceItem] = Field(default_factory=list)


class ModuleLibraryItem(BaseModel):
    module_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    section_count: int = Field(..., ge=0)
    share_enabled: bool = False
    share_id: Optional[str] = None


VersionAction = Literal["generated", "section_improved", "sources_refreshed"]


class ModuleVersionSummary(BaseModel):
    version_id: str = Field(..., min_length=1)
    module_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=utc_now)
    action: VersionAction


class ModuleAskRequest(BaseModel):
    question: str = Field(default="")
    mode: Literal["default", "simpler", "more_detailed", "quiz_me"] = "default"
    quiz_prompt: Optional[str] = None

    @model_validator(mode="after")
    def validate_question(self) -> "ModuleAskRequest":
        clean_question = self.question.strip()
        clean_quiz_prompt = (self.quiz_prompt or "").strip()

        self.question = clean_question
        self.quiz_prompt = clean_quiz_prompt or None

        if self.mode == "quiz_me":
            if self.quiz_prompt and len(clean_question) < 1:
                raise ValueError("Provide a learner answer when responding to a quiz question.")
            return self

        if len(clean_question) < 2:
            raise ValueError("question must contain at least two characters.")
        return self


class ModuleAskResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    citations: List[str] = Field(default_factory=list)
    unverified: bool = False


class RubricLevel(BaseModel):
    score: int = Field(..., ge=1, le=4)
    description: str = Field(..., min_length=1)


class RubricCriterion(BaseModel):
    criteria: str = Field(..., min_length=1)
    levels: List[RubricLevel] = Field(..., min_length=4, max_length=4)

    @field_validator("levels")
    @classmethod
    def validate_levels(cls, value: List[RubricLevel]) -> List[RubricLevel]:
        scores = [item.score for item in value]
        if scores != [4, 3, 2, 1]:
            raise ValueError("levels must use scores 4, 3, 2, 1 in order.")
        return value


class ModuleAssignmentResponse(BaseModel):
    prompt: str = Field(..., min_length=1)
    rubric: List[RubricCriterion] = Field(..., min_length=1)


class ModuleGradeBreakdownItem(BaseModel):
    criteria: str = Field(..., min_length=1)
    score: int = Field(..., ge=0, le=4)
    max_score: int = Field(default=4, ge=1, le=4)
    feedback: str = Field(..., min_length=1)


class ModuleGradeRequest(BaseModel):
    student_response: str = Field(..., min_length=1)
    rubric: List[RubricCriterion] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_payload(self) -> "ModuleGradeRequest":
        self.student_response = self.student_response.strip()
        if not self.student_response:
            raise ValueError("student_response must not be empty.")
        return self


class ModuleGradeResponse(BaseModel):
    score: int = Field(..., ge=0, le=100)
    feedback: str = Field(..., min_length=1)
    breakdown: List[ModuleGradeBreakdownItem] = Field(default_factory=list)
    unverified: bool = False


class ModuleDeleteResponse(BaseModel):
    module_id: str = Field(..., min_length=1)
    deleted: bool = True


class ModuleShareRequest(BaseModel):
    enabled: bool = True


class ModuleShareResponse(BaseModel):
    module_id: str = Field(..., min_length=1)
    share_enabled: bool = False
    share_id: Optional[str] = None
    share_url: Optional[str] = None


class SectionRegenerateRequest(BaseModel):
    section_index: Optional[int] = Field(default=None, ge=0)
    section_id: Optional[str] = None
    section_heading: Optional[str] = None
    learning_objective: Optional[str] = None
    instructions: Optional[str] = None
    allow_web: Optional[bool] = None
    vector_store_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_target(self) -> "SectionRegenerateRequest":
        if self.section_index is None and not self.section_id and not self.section_heading:
            raise ValueError("Provide section_index, section_id, or section_heading.")
        return self


class SectionRegenerateResponse(BaseModel):
    module: Module
    regenerated_section_index: int = Field(..., ge=0)
    evidence_pack: List[EvidenceItem] = Field(default_factory=list)


class SectionRegenerateByIdRequest(BaseModel):
    instructions: Optional[str] = None
    refresh_sources: bool = False


class RefreshSourcesRequest(BaseModel):
    source_policy: Optional[SourcePolicy] = None


class RefreshSourcesResponse(BaseModel):
    module_id: str
    source_policy: SourcePolicy
    vector_store_id: Optional[str] = None
    evidence_count: int = Field(..., ge=0)
    doc_count: int = Field(..., ge=0)
    web_count: int = Field(..., ge=0)
    refreshed_at: datetime = Field(default_factory=utc_now)
