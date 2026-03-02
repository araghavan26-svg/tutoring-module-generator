from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

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


class ModuleGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    audience_level: str = Field(..., min_length=1)
    learning_objectives: List[str] = Field(..., min_length=1)
    allow_web: bool = False
    vector_store_id: Optional[str] = None

    @field_validator("learning_objectives")
    @classmethod
    def validate_objectives(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("learning_objectives must contain at least one non-empty objective.")
        return cleaned


class EvidenceItem(BaseModel):
    evidence_id: str = Field(..., min_length=2)
    source_type: Literal["doc", "web"]
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
        if self.source_type == "doc" and not self.doc_name:
            self.doc_name = self.title
        return self


class ModuleSection(BaseModel):
    section_id: str = Field(..., min_length=1)
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
    sections: List[ModuleSection] = Field(default_factory=list, min_length=1)
    glossary: List[GlossaryItem] = Field(default_factory=list)
    mcqs: List[McqItem] = Field(default_factory=list)


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

