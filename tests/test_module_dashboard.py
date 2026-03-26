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
from app.store import PersistentModuleStore, module_store


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
        overview="Plants use sunlight to make food.",
        sections=[
            ModuleSection(
                section_id="section-1",
                objective_index=0,
                learning_goal="Explain what photosynthesis is",
                heading="What it is",
                content="Photosynthesis uses sunlight, water, and carbon dioxide.",
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


class ModuleDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        module_store.clear()

    def tearDown(self) -> None:
        module_store.clear()

    def test_saved_module_appears_in_dashboard_after_generation(self) -> None:
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
                dashboard = client.get("/modules")

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("Photosynthesis Module", dashboard.text)
        self.assertIn(module_id, dashboard.text)
        self.assertIn(f"/modules/{module_id}", dashboard.text)
        self.assertIn(f"/v1/modules/{module_id}/export/markdown", dashboard.text)

    def test_deleting_module_removes_it_from_dashboard(self) -> None:
        request = _request()
        module_id = "module-delete"
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        with TestClient(app) as client:
            delete_response = client.post(f"/v1/modules/{module_id}/delete")
            dashboard = client.get("/modules")

        self.assertEqual(delete_response.status_code, 200)
        self.assertNotIn("Photosynthesis Module", dashboard.text)
        self.assertNotIn(module_id, dashboard.text)

    def test_saved_modules_still_appear_after_restart(self) -> None:
        request = _request()
        module_id = "module-restart"
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        restarted_store = PersistentModuleStore(data_dir=Path(os.environ["TUTOR_DATA_DIR"]))
        with patch("app.main.module_store", restarted_store):
            with TestClient(app) as client:
                dashboard = client.get("/modules")

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("Photosynthesis Module", dashboard.text)
        self.assertIn(module_id, dashboard.text)


if __name__ == "__main__":
    unittest.main()
