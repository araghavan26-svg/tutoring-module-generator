from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional
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


class ModuleGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    audience_level: str = Field(..., min_length=1)
    learning_objectives: List[str] = Field(..., min_length=1)
    allow_web: bool = False
    vector_store_id: Optional[str] = None
    source_policy: Optional[SourcePolicy] = None

    @field_validator("learning_objectives")
    @classmethod
    def validate_objectives(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("learning_objectives must contain at least one non-empty objective.")
        return cleaned

    def effective_source_policy(self) -> SourcePolicy:
        if self.source_policy is not None:
            return self.source_policy
        return SourcePolicy(allow_web=self.allow_web)


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


class ModuleGenerateResponse(BaseModel):
    module: Module
    evidence_pack: List[EvidenceItem] = Field(default_factory=list)


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
