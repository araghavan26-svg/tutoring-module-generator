from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TUTOR_DATA_DIR", str(ROOT_DIR / ".test_data"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app
from app.models import EvidenceItem, Module, ModuleGenerateRequest, ModuleSection
from app.store import module_store


@pytest.fixture(autouse=True)
def clear_module_store() -> None:
    module_store.clear()
    yield
    module_store.clear()


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_request() -> ModuleGenerateRequest:
    return ModuleGenerateRequest(
        topic="Photosynthesis",
        audience_level="Middle school",
        learning_objectives=["Explain the core idea", "Use one simple example", "Check understanding"],
        allow_web=True,
    )


@pytest.fixture()
def sample_module() -> Module:
    return Module(
        module_id="module-fixture",
        title="Photosynthesis module",
        overview="Overview for Photosynthesis.",
        sections=[
            ModuleSection(
                section_id="section-1",
                objective_index=0,
                learning_goal="Explain the core idea",
                heading="Start here",
                content="Grounded explanation for Photosynthesis.",
                citations=["E001"],
                unverified=False,
                unverified_reason="",
            ),
            ModuleSection(
                section_id="section-2",
                objective_index=1,
                learning_goal="Use one simple example",
                heading="Example",
                content="Example section for Photosynthesis.",
                citations=["E001"],
                unverified=False,
                unverified_reason="",
            ),
            ModuleSection(
                section_id="section-3",
                objective_index=2,
                learning_goal="Check understanding",
                heading="Practice",
                content="Practice section for Photosynthesis.",
                citations=["E001"],
                unverified=False,
                unverified_reason="",
            ),
        ],
        glossary=[],
        mcqs=[],
    )


@pytest.fixture()
def sample_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id="E001",
            source_type="web",
            domain="kids.britannica.com",
            title="Photosynthesis",
            url="https://kids.britannica.com/students/article/photosynthesis/123456",
            snippet="Photosynthesis is how plants make food using sunlight.",
        )
    ]
