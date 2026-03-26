from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.models import EvidenceItem, Module, ModuleGenerateRequest, ModuleSection
from app.store import PersistentModuleStore


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


def _module(module_id: str, *, content: str) -> Module:
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
                content=content,
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


class StorePersistenceTests(unittest.TestCase):
    def test_save_then_reload_keeps_module_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            store = PersistentModuleStore(data_dir=data_dir)
            request = _request()
            store.save(
                "module-1",
                request,
                _module("module-1", content="Original content."),
                _evidence(),
                source_policy=request.effective_source_policy(),
                action="generated",
            )

            reloaded = PersistentModuleStore(data_dir=data_dir)
            record = reloaded.get("module-1")

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.module.title, "Photosynthesis Module")
            self.assertEqual(record.module.sections[0].content, "Original content.")
            self.assertEqual(record.evidence_pack[0].evidence_id, "E001")
            self.assertTrue((data_dir / "module_store.json").exists())

    def test_revert_still_works_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            store = PersistentModuleStore(data_dir=data_dir)
            request = _request()
            evidence_pack = _evidence()
            store.save(
                "module-1",
                request,
                _module("module-1", content="Original content."),
                evidence_pack,
                source_policy=request.effective_source_policy(),
                action="generated",
            )
            store.save(
                "module-1",
                request,
                _module("module-1", content="Improved content."),
                evidence_pack,
                source_policy=request.effective_source_policy(),
                action="section_improved",
            )

            reloaded = PersistentModuleStore(data_dir=data_dir)
            generated_version = next(item for item in reloaded.history("module-1") if item.action == "generated")
            restored = reloaded.revert("module-1", generated_version.version_id)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.sections[0].content, "Original content.")

            restarted_again = PersistentModuleStore(data_dir=data_dir)
            record = restarted_again.get("module-1")
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.module.sections[0].content, "Original content.")

    def test_history_persists_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            store = PersistentModuleStore(data_dir=data_dir)
            request = _request()
            evidence_pack = _evidence()
            store.save(
                "module-1",
                request,
                _module("module-1", content="Original content."),
                evidence_pack,
                source_policy=request.effective_source_policy(),
                action="generated",
            )
            store.save(
                "module-1",
                request,
                _module("module-1", content="Improved content."),
                evidence_pack,
                source_policy=request.effective_source_policy(),
                action="section_improved",
            )
            store.save(
                "module-1",
                request,
                _module("module-1", content="Improved content."),
                evidence_pack,
                source_policy=request.effective_source_policy(),
                action="sources_refreshed",
            )

            reloaded = PersistentModuleStore(data_dir=data_dir)
            actions = [item.action for item in reloaded.history("module-1")]

            self.assertEqual(len(actions), 3)
            self.assertIn("generated", actions)
            self.assertIn("section_improved", actions)
            self.assertIn("sources_refreshed", actions)


if __name__ == "__main__":
    unittest.main()
