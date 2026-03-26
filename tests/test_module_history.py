from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("TUTOR_DATA_DIR", str(ROOT_DIR / ".test_data"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app
from app.models import EvidenceItem, Module, ModuleGenerateRequest, ModuleSection
from app.openai_service import EvidenceBuildResult
from app.store import module_store


def _request(topic: str = "Photosynthesis") -> ModuleGenerateRequest:
    return ModuleGenerateRequest(
        topic=topic,
        audience_level="Middle school",
        learning_objectives=[
            "Explain the core idea",
            "Use one simple example",
            "Check understanding",
        ],
        allow_web=True,
    )


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id="E001",
            source_type="web",
            domain="kids.britannica.com",
            title="Photosynthesis for Students",
            url="https://kids.britannica.com/students/article/photosynthesis/123",
            snippet="Plants use sunlight, water, and carbon dioxide to make food.",
        )
    ]


def _module(module_id: str, *, content: str = "Original section content.") -> Module:
    return Module(
        module_id=module_id,
        title="Photosynthesis Module",
        overview="A grounded overview of photosynthesis.",
        sections=[
            ModuleSection(
                section_id="section-1",
                objective_index=0,
                learning_goal="Explain the core idea",
                heading="What it is",
                content=content,
                citations=["E001"],
                unverified=False,
                unverified_reason="",
            )
        ],
        glossary=[],
        mcqs=[],
    )


class ModuleHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        module_store.clear()

    def tearDown(self) -> None:
        module_store.clear()

    def test_generation_creates_version_history_entry(self) -> None:
        request = _request()
        evidence_pack = _evidence()

        def fake_generate_module_from_evidence(*args, **kwargs) -> Module:
            return _module(kwargs["module_id"])

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.build_evidence_pack",
            return_value=EvidenceBuildResult(
                evidence_pack=evidence_pack,
                web_unavailable_objectives=[],
                objectives_without_evidence=[],
            ),
        ), patch(
            "app.main.generate_module_from_evidence",
            side_effect=fake_generate_module_from_evidence,
        ), patch(
            "app.main.enforce_quality_gate",
            side_effect=lambda module, evidence_pack: module,
        ):
            with TestClient(app) as client:
                response = client.post("/v1/modules/generate", json=request.model_dump(mode="json"))
                self.assertEqual(response.status_code, 200)
                module_id = response.json()["module"]["module_id"]

                history_response = client.get(f"/v1/modules/{module_id}/history")

        self.assertEqual(history_response.status_code, 200)
        history = history_response.json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["action"], "generated")

    def test_section_improvement_creates_version_history_entry(self) -> None:
        request = _request()
        module_id = "module-history"
        evidence_pack = _evidence()
        module_store.save(
            module_id,
            request,
            _module(module_id),
            evidence_pack,
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        improved_section = ModuleSection(
            section_id="section-1",
            objective_index=0,
            learning_goal="Explain the core idea",
            heading="What it is",
            content="Improved section content.",
            citations=["E001"],
            unverified=False,
            unverified_reason="",
        )

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.generate_section_from_evidence",
            return_value=improved_section,
        ):
            with TestClient(app) as client:
                response = client.post(
                    f"/v1/modules/{module_id}/sections/section-1/regenerate",
                    json={"instructions": "Make it clearer.", "refresh_sources": False},
                )
                self.assertEqual(response.status_code, 200)
                history_response = client.get(f"/v1/modules/{module_id}/history")

        self.assertEqual(history_response.status_code, 200)
        history = history_response.json()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["action"], "section_improved")
        self.assertEqual(history[1]["action"], "generated")

    def test_revert_restores_previous_module_snapshot(self) -> None:
        request = _request()
        module_id = "module-revert"
        evidence_pack = _evidence()
        original_module = _module(module_id, content="Original section content.")
        improved_module = _module(module_id, content="Improved section content.")

        module_store.save(
            module_id,
            request,
            original_module,
            evidence_pack,
            source_policy=request.effective_source_policy(),
            action="generated",
        )
        module_store.save(
            module_id,
            request,
            improved_module,
            evidence_pack,
            source_policy=request.effective_source_policy(),
            action="section_improved",
        )

        history = module_store.history(module_id)
        generated_version = next(item for item in history if item.action == "generated")

        with TestClient(app) as client:
            response = client.post(f"/v1/modules/{module_id}/revert/{generated_version.version_id}")

        self.assertEqual(response.status_code, 200)
        restored = response.json()
        self.assertEqual(restored["sections"][0]["content"], "Original section content.")
        current_record = module_store.get(module_id)
        self.assertIsNotNone(current_record)
        self.assertEqual(current_record.module.sections[0].content, "Original section content.")


if __name__ == "__main__":
    unittest.main()
