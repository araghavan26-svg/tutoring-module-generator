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
from app.models import EvidenceItem, Module, ModuleAskResponse, ModuleGenerateRequest, ModuleSection
from app.store import module_store


def _request() -> ModuleGenerateRequest:
    return ModuleGenerateRequest(
        topic="Photosynthesis",
        audience_level="Middle school",
        learning_objectives=[
            "Explain what photosynthesis is",
            "Identify key vocabulary",
            "Apply the idea in a simple example",
        ],
        allow_web=True,
    )


def _module(module_id: str) -> Module:
    return Module(
        module_id=module_id,
        title="Photosynthesis Module",
        overview="This module explains how plants use sunlight to make food.",
        sections=[
            ModuleSection(
                section_id="section-1",
                objective_index=0,
                learning_goal="Explain what photosynthesis is",
                heading="What it is",
                content="Photosynthesis is the process plants use to make food from sunlight, water, and carbon dioxide.",
                citations=["E001"],
                unverified=False,
                unverified_reason="",
            )
        ],
        glossary=[],
        mcqs=[],
    )


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id="E001",
            source_type="web",
            domain="kids.britannica.com",
            title="Photosynthesis for Students",
            url="https://kids.britannica.com/students/article/photosynthesis/123",
            snippet="Photosynthesis lets plants use sunlight, water, and carbon dioxide to make sugar.",
        )
    ]


class TutorModeTests(unittest.TestCase):
    def setUp(self) -> None:
        module_store.clear()

    def tearDown(self) -> None:
        module_store.clear()

    def test_supported_question_returns_grounded_answer_with_citations(self) -> None:
        module_id = "module-tutor"
        request = _request()
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.answer_question_from_module",
            return_value=ModuleAskResponse(
                answer="Photosynthesis is how plants use sunlight to make sugar from water and carbon dioxide.",
                citations=["E001"],
                unverified=False,
            ),
        ):
            with TestClient(app) as client:
                response = client.post(
                    f"/v1/modules/{module_id}/ask",
                    json={"question": "What is photosynthesis?"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["citations"], ["E001"])
        self.assertFalse(payload["unverified"])
        self.assertIn("sunlight", payload["answer"])

    def test_unsupported_question_returns_cautious_unverified_answer(self) -> None:
        module_id = "module-tutor"
        request = _request()
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.answer_question_from_module",
            return_value=ModuleAskResponse(
                answer="I cannot answer that confidently from this module's saved sources.",
                citations=[],
                unverified=True,
            ),
        ):
            with TestClient(app) as client:
                response = client.post(
                    f"/v1/modules/{module_id}/ask",
                    json={"question": "How does cellular respiration compare to photosynthesis in animals?"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["citations"], [])
        self.assertTrue(payload["unverified"])
        self.assertIn("saved sources", payload["answer"])

    def test_quiz_mode_forwards_quiz_prompt_to_helper(self) -> None:
        module_id = "module-tutor"
        request = _request()
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )
        captured = {}

        def fake_answer_question_from_module(*args, **kwargs) -> ModuleAskResponse:
            captured["mode"] = kwargs.get("mode")
            captured["quiz_prompt"] = kwargs.get("quiz_prompt")
            captured["question"] = kwargs.get("question")
            return ModuleAskResponse(
                answer="That is correct. Plants make sugar during photosynthesis.",
                citations=["E001"],
                unverified=False,
            )

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.answer_question_from_module",
            side_effect=fake_answer_question_from_module,
        ):
            with TestClient(app) as client:
                response = client.post(
                    f"/v1/modules/{module_id}/ask",
                    json={
                        "question": "Plants make sugar.",
                        "mode": "quiz_me",
                        "quiz_prompt": "What do plants make during photosynthesis?",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["mode"], "quiz_me")
        self.assertEqual(captured["quiz_prompt"], "What do plants make during photosynthesis?")
        self.assertEqual(captured["question"], "Plants make sugar.")


if __name__ == "__main__":
    unittest.main()
