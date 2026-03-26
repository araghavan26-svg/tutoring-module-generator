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
from app.models import (
    EvidenceItem,
    Module,
    ModuleAssignmentResponse,
    ModuleGenerateRequest,
    ModuleGradeBreakdownItem,
    ModuleGradeResponse,
    ModuleSection,
    RubricCriterion,
)
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
                content="Photosynthesis is how plants use sunlight, water, and carbon dioxide to make sugar.",
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


def _rubric() -> list[RubricCriterion]:
    return [
        RubricCriterion.model_validate(
            {
                "criteria": "Explain the core process accurately.",
                "levels": [
                    {"score": 4, "description": "Explains the process accurately with all key parts."},
                    {"score": 3, "description": "Explains the process mostly accurately with a small gap."},
                    {"score": 2, "description": "Shows partial understanding of the process."},
                    {"score": 1, "description": "Shows limited or inaccurate understanding."},
                ],
            }
        )
    ]


class AssignmentGradingTests(unittest.TestCase):
    def setUp(self) -> None:
        module_store.clear()

    def tearDown(self) -> None:
        module_store.clear()

    def test_assignment_generation_returns_prompt_and_rubric_structure(self) -> None:
        module_id = "module-assignment"
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
            "app.main.generate_assignment_from_module",
            return_value=ModuleAssignmentResponse(
                prompt="Write a paragraph explaining photosynthesis and give one everyday example.",
                rubric=_rubric(),
            ),
        ):
            with TestClient(app) as client:
                response = client.post(f"/v1/modules/{module_id}/assignment")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("photosynthesis", payload["prompt"].lower())
        self.assertEqual(payload["rubric"][0]["criteria"], "Explain the core process accurately.")
        self.assertEqual([level["score"] for level in payload["rubric"][0]["levels"]], [4, 3, 2, 1])

    def test_grading_returns_structured_breakdown(self) -> None:
        module_id = "module-grading"
        request = _request()
        rubric = _rubric()
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.grade_assignment_from_module",
            return_value=ModuleGradeResponse(
                score=85,
                feedback="Strong explanation overall, with one small factual gap.",
                breakdown=[
                    ModuleGradeBreakdownItem(
                        criteria="Explain the core process accurately.",
                        score=3,
                        max_score=4,
                        feedback="The response explained the main process, but left out carbon dioxide.",
                    )
                ],
                unverified=False,
            ),
        ):
            with TestClient(app) as client:
                response = client.post(
                    f"/v1/modules/{module_id}/grade",
                    json={
                        "student_response": "Plants use sunlight and water to make food.",
                        "rubric": [item.model_dump(mode="json") for item in rubric],
                    },
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["score"], 85)
        self.assertFalse(payload["unverified"])
        self.assertEqual(payload["breakdown"][0]["criteria"], "Explain the core process accurately.")
        self.assertEqual(payload["breakdown"][0]["score"], 3)

    def test_unsupported_grading_returns_cautious_output(self) -> None:
        module_id = "module-grading"
        request = _request()
        rubric = _rubric()
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.grade_assignment_from_module",
            return_value=ModuleGradeResponse(
                score=0,
                feedback="I cannot grade this confidently from the module's saved sources.",
                breakdown=[],
                unverified=True,
            ),
        ):
            with TestClient(app) as client:
                response = client.post(
                    f"/v1/modules/{module_id}/grade",
                    json={
                        "student_response": "This answer talks about cellular respiration instead.",
                        "rubric": [item.model_dump(mode="json") for item in rubric],
                    },
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["unverified"])
        self.assertEqual(payload["breakdown"], [])
        self.assertIn("saved sources", payload["feedback"])

    def test_assignment_markdown_export_returns_prompt_and_rubric_levels(self) -> None:
        module_id = "module-assignment-export"
        request = _request()
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )
        assignment = ModuleAssignmentResponse(
            prompt="Write a paragraph explaining photosynthesis and include one real-life example.",
            rubric=_rubric(),
        )

        with TestClient(app) as client:
            response = client.post(
                f"/v1/modules/{module_id}/assignment/export/markdown",
                json=assignment.model_dump(mode="json"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/markdown; charset=utf-8")
        self.assertIn("# Assignment", response.text)
        self.assertIn("## Prompt", response.text)
        self.assertIn("## Rubric", response.text)
        self.assertIn("### 1. Explain the core process accurately.", response.text)
        self.assertIn("- 4: Explains the process accurately with all key parts.", response.text)
        self.assertIn("- 1: Shows limited or inaccurate understanding.", response.text)

    def test_assignment_json_export_returns_structured_assignment(self) -> None:
        module_id = "module-assignment-export-json"
        request = _request()
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )
        assignment = ModuleAssignmentResponse(
            prompt="Write a paragraph explaining photosynthesis and include one real-life example.",
            rubric=_rubric(),
        )

        with TestClient(app) as client:
            response = client.post(
                f"/v1/modules/{module_id}/assignment/export/json",
                json=assignment.model_dump(mode="json"),
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prompt"], assignment.prompt)
        self.assertEqual(payload["rubric"][0]["criteria"], "Explain the core process accurately.")
        self.assertEqual([level["score"] for level in payload["rubric"][0]["levels"]], [4, 3, 2, 1])


if __name__ == "__main__":
    unittest.main()
