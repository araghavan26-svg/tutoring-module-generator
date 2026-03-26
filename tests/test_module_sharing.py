from __future__ import annotations

import os
import sys
import tempfile
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


def _module(module_id: str, *, share_enabled: bool = False, share_id: str | None = None) -> Module:
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
        share_enabled=share_enabled,
        share_id=share_id,
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


class ModuleSharingTests(unittest.TestCase):
    def setUp(self) -> None:
        module_store.clear()

    def tearDown(self) -> None:
        module_store.clear()

    def test_shared_page_loads(self) -> None:
        request = _request()
        module_id = "module-share"
        module_store.save(
            module_id,
            request,
            _module(module_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        with TestClient(app) as client:
            share_response = client.post(f"/v1/modules/{module_id}/share", json={"enabled": True})
            self.assertEqual(share_response.status_code, 200)
            share_payload = share_response.json()
            shared_page = client.get(f"/shared/{share_payload['share_id']}")

        self.assertEqual(shared_page.status_code, 200)
        self.assertIn("Photosynthesis Module", shared_page.text)
        self.assertIn("This is a read-only shared copy.", shared_page.text)
        self.assertIn("Open source", shared_page.text)
        self.assertNotIn("Improve this section", shared_page.text)
        self.assertNotIn("Refresh sources", shared_page.text)
        self.assertNotIn("Delete", shared_page.text)

    def test_disabling_share_invalidates_link(self) -> None:
        request = _request()
        module_id = "module-disable-share"
        original_share_id = "share-disabled"
        module_store.save(
            module_id,
            request,
            _module(module_id, share_enabled=True, share_id=original_share_id),
            _evidence(),
            source_policy=request.effective_source_policy(),
            action="generated",
        )

        with TestClient(app) as client:
            disable_response = client.post(f"/v1/modules/{module_id}/share", json={"enabled": False})
            shared_page = client.get(f"/shared/{original_share_id}")

        self.assertEqual(disable_response.status_code, 200)
        self.assertEqual(shared_page.status_code, 404)
        self.assertIn("This shared module is no longer available.", shared_page.text)
        self.assertIn('href="/"', shared_page.text)

    def test_share_persists_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            store = PersistentModuleStore(data_dir=data_dir)
            request = _request()
            store.save(
                "module-restart-share",
                request,
                _module("module-restart-share", share_enabled=True, share_id="share-persisted"),
                _evidence(),
                source_policy=request.effective_source_policy(),
                action="generated",
            )

            restarted_store = PersistentModuleStore(data_dir=data_dir)
            with patch("app.main.module_store", restarted_store):
                with TestClient(app) as client:
                    shared_page = client.get("/shared/share-persisted")
                    dashboard = client.get("/modules")

        self.assertEqual(shared_page.status_code, 200)
        self.assertIn("Photosynthesis Module", shared_page.text)
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("Sharing: On", dashboard.text)
        self.assertIn("/shared/share-persisted", dashboard.text)


if __name__ == "__main__":
    unittest.main()
