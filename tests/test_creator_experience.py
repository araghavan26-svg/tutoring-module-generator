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

from app.main import app, _module_export_markdown
from app.models import EvidenceItem, Module, ModuleGenerateRequest, ModuleSection
from app.openai_service import (
    EvidenceBuildResult,
    TopicBridgeContext,
    TopicRelatednessResult,
    _apply_topic_bridge,
    build_personalization_context,
    learning_goal_to_heading,
    normalize_learning_goal_text,
)
from app.store import module_store


def _dummy_request(topic: str, *, related_to_previous: bool = False) -> ModuleGenerateRequest:
    return ModuleGenerateRequest(
        topic=topic,
        audience_level="Middle school",
        learning_objectives=["Explain the core idea", "Use one simple example", "Check understanding"],
        allow_web=True,
        related_to_previous=related_to_previous,
    )


def _dummy_module(module_id: str, topic: str, *, overview: str | None = None) -> Module:
    return Module(
        module_id=module_id,
        title=f"{topic} module",
        overview=overview or f"Overview for {topic}.",
        sections=[
            ModuleSection(
                section_id="section-1",
                objective_index=0,
                learning_goal="Explain the core idea",
                heading="Start here",
                content=f"Grounded explanation for {topic}.",
                citations=["E001"],
                unverified=False,
                unverified_reason="",
            ),
            ModuleSection(
                section_id="section-2",
                objective_index=1,
                learning_goal="Use one simple example",
                heading="Example",
                content=f"Example section for {topic}.",
                citations=["E001"],
                unverified=False,
                unverified_reason="",
            ),
            ModuleSection(
                section_id="section-3",
                objective_index=2,
                learning_goal="Check understanding",
                heading="Practice",
                content=f"Practice section for {topic}.",
                citations=["E001"],
                unverified=False,
                unverified_reason="",
            ),
        ],
        glossary=[],
        mcqs=[],
    )


class CreatorExperienceTests(unittest.TestCase):
    def setUp(self) -> None:
        module_store.clear()

    def tearDown(self) -> None:
        module_store.clear()

    def test_create_page_includes_progress_ui(self) -> None:
        with TestClient(app) as client:
            response = client.get("/create")
            landing = client.get("/app")
            disclaimer = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(landing.status_code, 200)
        self.assertEqual(disclaimer.status_code, 200)
        self.assertIn("What do you want to learn?", response.text)
        self.assertIn('id="learning_request"', response.text)
        self.assertIn('id="create-progress"', response.text)
        self.assertIn('id="create-progress-message"', response.text)
        self.assertLess(response.text.index('id="create-submit"'), response.text.index('id="create-progress"'))
        self.assertIn("Progress is approximate, not exact.", response.text)
        self.assertIn("View Saved Modules", landing.text)
        self.assertIn("Before You Use This System", disclaimer.text)
        self.assertIn("I Understand — Continue", disclaimer.text)
        self.assertIn('action="/app"', disclaimer.text)
        self.assertIn("Personalize your learning", response.text)
        self.assertIn("Current familiarity with the topic", response.text)
        self.assertIn("Learning purpose", response.text)
        self.assertIn("Explanation style", response.text)
        self.assertIn("Describe your level", response.text)
        self.assertIn('id="source_preference"', response.text)
        self.assertIn("Prefer high-trust sources", response.text)
        self.assertIn('id="preset-learn-scratch"', response.text)
        self.assertIn('id="preset-study-test"', response.text)
        self.assertIn('id="preset-quick-review"', response.text)
        self.assertIn("What specifically confuses you or what do you want extra help with?", response.text)

        create_js = (ROOT_DIR / "static" / "create.js").read_text()
        self.assertIn("Finding sources...", create_js)
        self.assertIn("Reading your materials...", create_js)
        self.assertIn("Building lesson sections...", create_js)
        self.assertIn("Generating practice questions...", create_js)
        self.assertIn("Module generation took too long. Please try fewer goals or fewer sources.", create_js)
        self.assertIn("fast_mode: true", create_js)
        self.assertIn("toggleCustomLevelField", create_js)
        self.assertIn("Please describe your level.", create_js)
        self.assertIn("learning_request: learningRequest", create_js)
        self.assertIn("source_preference: sourcePreference || \"General\"", create_js)
        self.assertIn("prefer_high_trust_sources: preferHighTrustSources", create_js)
        self.assertIn("setActivePreset", create_js)
        self.assertIn("current_familiarity: currentFamiliarity || null", create_js)
        self.assertIn("learning_purpose: learningPurpose || null", create_js)
        self.assertIn("explanation_style: explanationStyle || null", create_js)
        self.assertIn("confusion_points: confusionPoints || null", create_js)

    def test_demo_polish_landing_create_and_dashboard_empty_states(self) -> None:
        with TestClient(app) as client:
            disclaimer = client.get("/")
            landing = client.get("/app")
            create = client.get("/create?sample=1")
            dashboard = client.get("/modules")

        self.assertEqual(disclaimer.status_code, 200)
        self.assertEqual(landing.status_code, 200)
        self.assertEqual(create.status_code, 200)
        self.assertEqual(dashboard.status_code, 200)

        self.assertIn("Before You Use This System", disclaimer.text)
        self.assertIn("AI Limitations", disclaimer.text)
        self.assertIn("API Key Notice", disclaimer.text)
        self.assertIn("Usage Rules", disclaimer.text)
        self.assertIn("What this app does", landing.text)
        self.assertIn("Try Sample Module", landing.text)
        self.assertIn("View Saved Modules", landing.text)
        self.assertIn("Create", landing.text)
        self.assertIn("Tutor", landing.text)
        self.assertIn("Assess", landing.text)
        self.assertIn("Best experience: use Try Sample Module to start quickly.", landing.text)

        self.assertIn('id="create-demo-banner"', create.text)
        self.assertIn("Use sample document", create.text)
        self.assertIn("For the smoothest demo, keep web sources on and leave the sample document checked.", create.text)

        self.assertIn('id="library-banner"', dashboard.text)
        self.assertIn('id="empty-library"', dashboard.text)
        self.assertIn("No saved modules yet.", dashboard.text)
        self.assertIn("Create your first module", dashboard.text)
        self.assertIn("Try Sample Module", dashboard.text)

        create_js = (ROOT_DIR / "static" / "create.js").read_text()
        self.assertIn("Generate Sample Module", create_js)
        self.assertIn("Preparing the sample module...", create_js)
        self.assertIn("Demo mode is ready.", create_js)

    def test_demo_reset_helper_exists(self) -> None:
        reset_helper = ROOT_DIR / "reset_demo_state.command"
        self.assertTrue(reset_helper.exists())

        reset_script = reset_helper.read_text()
        self.assertIn("module_store.clear()", reset_script)
        self.assertIn("Demo state reset.", reset_script)

    def test_start_launcher_is_the_primary_demo_path(self) -> None:
        launcher = ROOT_DIR / "start_app.command"
        self.assertTrue(launcher.exists())

        launcher_text = launcher.read_text()
        readme_text = (ROOT_DIR / "README.md").read_text()

        self.assertIn("Launching tutoring module app...", launcher_text)
        self.assertIn("python3 -m venv .venv", launcher_text)
        self.assertIn("-m pip install -r requirements.txt", launcher_text)
        self.assertIn('-m uvicorn app.main:app --reload --port 8000', launcher_text)
        self.assertIn('open "http://127.0.0.1:8000"', launcher_text)

        self.assertIn("## Quick Start", readme_text)
        self.assertIn("1. Double-click `start_app.command`", readme_text)
        self.assertIn("Advanced Terminal Launch (Optional)", readme_text)

    def test_markdown_export_renders_clickable_web_citations(self) -> None:
        module_data = {
            "module_id": "module-1",
            "title": "Photosynthesis",
            "overview": "A grounded overview.",
            "sections": [
                {
                    "heading": "What it is",
                    "learning_goal": "Explain the process",
                    "content": "Plants make food using light.",
                    "citations": ["E001", "E002"],
                }
            ],
            "glossary": [],
            "mcqs": [],
        }
        footnotes = [
            {
                "footnote_id": "E001",
                "source_type": "web",
                "domain": "vocabulary.com",
                "title": "Photosynthesis definition",
                "url": "https://www.vocabulary.com/dictionary/photosynthesis",
                "doc_name": None,
                "location": None,
                "snippet": "Photosynthesis is the process plants use to make food.",
            },
            {
                "footnote_id": "E002",
                "source_type": "doc",
                "domain": None,
                "title": "sample.txt",
                "url": None,
                "doc_name": "sample.txt",
                "location": "page:1",
                "snippet": "Chlorophyll helps plants capture sunlight.",
            },
        ]

        markdown = _module_export_markdown(module_data, footnotes)

        self.assertIn("[Photosynthesis definition](https://www.vocabulary.com/dictionary/photosynthesis)", markdown)
        self.assertIn("Web source: vocabulary.com", markdown)
        self.assertIn("**sample.txt**", markdown)
        self.assertIn("Document source | page:1", markdown)

    def test_personalization_context_shapes_generation_guidance(self) -> None:
        request = ModuleGenerateRequest(
            subject="Biology",
            topic="Photosynthesis",
            audience_level="Adult beginner returning to science",
            learner_level="Custom",
            custom_level_description="Adult beginner returning to science",
            current_familiarity="Brand new",
            learning_purpose="Understand a confusing concept",
            explanation_style="Step-by-step",
            confusion_points="I mix up chlorophyll and glucose production.",
            learning_objectives=["Explain the process", "Use the key vocabulary"],
            allow_web=True,
        )

        personalization = build_personalization_context(request)

        self.assertEqual(personalization["subject"], "Biology")
        self.assertIn("Start from the basics", personalization["starting_point"])
        self.assertIn("Slow down", personalization["pacing_guidance"])
        self.assertIn("clear sequence", personalization["tone_guidance"])
        self.assertIn("chlorophyll", personalization["focus_guidance"])

    def test_learning_goal_normalization_removes_first_person_phrasing(self) -> None:
        request = ModuleGenerateRequest(
            subject="Statistics",
            topic="Probability",
            audience_level="AP Statistics",
            learner_level="Advanced",
            current_familiarity="I know the basics",
            learning_purpose="Review for a test",
            explanation_style="Concise review",
            learning_objectives=["I want to understand statistics at a AP level"],
            allow_web=True,
        )

        normalized = normalize_learning_goal_text(
            "I want to understand statistics at a AP level",
            request=request,
            subject=request.subject or "",
            topic=request.topic,
            audience_level=request.audience_level,
        )

        self.assertNotIn("I want to", normalized)
        self.assertNotIn(" at a AP level", normalized)
        self.assertEqual(normalized, "Understand core AP Statistics concepts and expectations")

    def test_advanced_ap_personalization_strengthens_guidance(self) -> None:
        request = ModuleGenerateRequest(
            subject="Statistics",
            topic="Probability",
            audience_level="AP Statistics",
            learner_level="Advanced",
            current_familiarity="I know the basics",
            learning_purpose="Review for a test",
            explanation_style="Formal/academic",
            learning_objectives=["Analyze probability models"],
            allow_web=True,
        )

        personalization = build_personalization_context(request)

        self.assertIn("course-level", personalization["starting_point"])
        self.assertIn("precise disciplinary vocabulary", personalization["tone_guidance"])
        self.assertIn("rigorous distinctions", personalization["depth_guidance"])
        self.assertIn("review and application", personalization["section_emphasis"])

    def test_heading_cleanup_is_cleaner_than_raw_goal_wording(self) -> None:
        raw_goal = "I want to understand statistics at a AP level"
        normalized_goal = "Understand core AP Statistics concepts and expectations"

        heading = learning_goal_to_heading(normalized_goal)

        self.assertNotEqual(heading, raw_goal)
        self.assertNotIn("I want to", heading)
        self.assertEqual(heading, "Core AP Statistics Concepts and Expectations")

    def test_combined_learning_request_splits_subject_and_topic(self) -> None:
        request = ModuleGenerateRequest(
            learning_request="Statistics - Probability",
            audience_level="Advanced / AP",
            learner_level="Advanced / AP",
            current_familiarity="I know the basics",
            learning_purpose="Review for a test",
            explanation_style="Concise review",
            learning_objectives=["Explain the difference between probability models"],
            allow_web=True,
        )

        self.assertEqual(request.subject, "Statistics")
        self.assertEqual(request.topic, "Probability")

    def test_learning_request_without_separator_keeps_topic_and_leaves_subject_optional(self) -> None:
        request = ModuleGenerateRequest(
            learning_request="Photosynthesis",
            audience_level="Beginner",
            learner_level="Beginner",
            current_familiarity="Brand new",
            learning_purpose="Learn from scratch",
            explanation_style="Step-by-step",
            learning_objectives=["Explain the process"],
            allow_web=True,
        )

        self.assertIsNone(request.subject)
        self.assertEqual(request.topic, "Photosynthesis")

    def test_generate_endpoint_accepts_and_passes_personalization_fields(self) -> None:
        evidence_pack = [
            EvidenceItem(
                evidence_id="E001",
                source_type="web",
                domain="kids.britannica.com",
                title="Photosynthesis",
                url="https://kids.britannica.com/students/article/photosynthesis/123456",
                snippet="Photosynthesis is how plants make food using sunlight.",
            )
        ]
        evidence_result = EvidenceBuildResult(
            evidence_pack=evidence_pack,
            web_unavailable_objectives=[],
            objectives_without_evidence=[],
        )
        captured: dict[str, object] = {}

        def fake_generate_module_from_evidence(*args, **kwargs) -> Module:
            captured["request"] = kwargs.get("request")
            return _dummy_module("module-new", "Photosynthesis", overview="A personalized module overview.")

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.build_evidence_pack",
            return_value=evidence_result,
        ), patch(
            "app.main.generate_module_from_evidence",
            side_effect=fake_generate_module_from_evidence,
        ), patch(
            "app.main.enforce_quality_gate",
            side_effect=lambda module, evidence_pack: module,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/modules/generate",
                    json={
                        "learning_request": "Biology - Photosynthesis",
                        "audience_level": "Adult beginner returning to science",
                        "learner_level": "Custom",
                        "custom_level_description": "Adult beginner returning to science",
                        "current_familiarity": "Brand new",
                        "learning_purpose": "Understand a confusing concept",
                        "explanation_style": "Step-by-step",
                        "confusion_points": "I mix up chlorophyll and glucose production.",
                        "source_preference": "Academic / Educational",
                        "prefer_high_trust_sources": True,
                        "learning_objectives": [
                            "Explain what photosynthesis is",
                            "Identify key vocabulary",
                        ],
                        "allow_web": True,
                    },
                )

        self.assertEqual(response.status_code, 200)
        request = captured.get("request")
        self.assertIsNotNone(request)
        self.assertEqual(request.subject, "Biology")
        self.assertEqual(request.topic, "Photosynthesis")
        self.assertEqual(request.learner_level, "Custom")
        self.assertEqual(request.custom_level_description, "Adult beginner returning to science")
        self.assertEqual(request.audience_level, "Adult beginner returning to science")
        self.assertEqual(request.current_familiarity, "Brand new")
        self.assertEqual(request.learning_purpose, "Understand a confusing concept")
        self.assertEqual(request.explanation_style, "Step-by-step")
        self.assertEqual(request.source_preference, "Academic / Educational")
        self.assertTrue(request.prefer_high_trust_sources)
        self.assertIn("chlorophyll", request.confusion_points)

    def test_module_editor_js_renders_clickable_web_and_doc_citations(self) -> None:
        editor_js = (ROOT_DIR / "static" / "module_editor.js").read_text()
        dashboard_js = (ROOT_DIR / "static" / "modules_dashboard.js").read_text()
        module_editor_html = (ROOT_DIR / "templates" / "module_editor.html").read_text()
        dashboard_html = (ROOT_DIR / "templates" / "modules_dashboard.html").read_text()

        self.assertIn('<a\n            class="citation-card citation-link"', editor_js)
        self.assertIn('href="${escapeAttribute(evidence.url)}"', editor_js)
        self.assertIn('target="_blank"', editor_js)
        self.assertIn('rel="noopener noreferrer"', editor_js)
        self.assertIn("Open source", editor_js)
        self.assertIn("citation-doc-card", editor_js)
        self.assertIn("Document source", editor_js)
        self.assertIn('/v1/modules/${moduleId}/history', editor_js)
        self.assertIn('/v1/modules/${moduleId}/revert/${versionId}', editor_js)
        self.assertIn('/v1/modules/${moduleId}/ask', editor_js)
        self.assertIn('/v1/modules/${moduleId}/share', editor_js)
        self.assertIn('/v1/modules/${moduleId}/assignment', editor_js)
        self.assertIn('/v1/modules/${moduleId}/grade', editor_js)
        self.assertIn('/v1/modules/${moduleId}/assignment/export/markdown', editor_js)
        self.assertIn('/v1/modules/${moduleId}/assignment/export/json', editor_js)
        self.assertIn("Ask a question about this module", module_editor_html)
        self.assertIn('id="share-module"', module_editor_html)
        self.assertIn('id="share-badge"', module_editor_html)
        self.assertIn('id="create-assignment"', module_editor_html)
        self.assertIn('id="assignment-panel"', module_editor_html)
        self.assertIn('id="assignment-grade"', module_editor_html)
        self.assertIn('id="assignment-export-markdown"', module_editor_html)
        self.assertIn('id="assignment-export-json"', module_editor_html)
        self.assertIn("value=\"quiz_me\"", module_editor_html)
        self.assertIn('id="tutor-suggestions"', module_editor_html)
        self.assertIn("share-module-btn", dashboard_html)
        self.assertIn("library-share-badge", dashboard_html)
        self.assertIn("Can you explain", editor_js)
        self.assertIn("Grounded", editor_js)
        self.assertIn("Unverified", editor_js)
        self.assertIn("shareBadgeEl.hidden = !shareEnabled;", editor_js)
        self.assertIn('sharePanelEl.hidden = !shareEnabled;', editor_js)
        self.assertIn('copyShareBtn.hidden = !shareEnabled;', editor_js)
        self.assertIn('shareLinkEl.value = shareEnabled ? shareUrl : "";', editor_js)
        self.assertIn("Sharing disabled. Existing shared links will no longer work.", editor_js)
        self.assertIn("shareBadge.hidden = !shareEnabled;", dashboard_js)
        self.assertIn("shareWrap.hidden = !shareEnabled;", dashboard_js)
        self.assertIn('shareInput.value = shareEnabled ? shareUrl : "";', dashboard_js)
        self.assertIn("Create an assignment before grading a response.", editor_js)
        self.assertIn("Creating an assignment and rubric from the current module...", editor_js)
        self.assertIn("Grading the response against the rubric...", editor_js)
        self.assertIn("Preparing the assignment Markdown export...", editor_js)
        self.assertIn("Assignment exported successfully.", editor_js)
        self.assertIn("Tutor is reading the module and drafting an answer...", editor_js)
        self.assertIn("Improving this section from the saved sources...", editor_js)
        self.assertIn("Sources refreshed successfully.", editor_js)
        self.assertIn("section-card-loading", editor_js)
        self.assertIn("No versions yet.", editor_js)
        self.assertIn('setButtonBusy(tutorAskBtn, true, "Thinking...")', editor_js)
        self.assertIn('setButtonBusy(assignmentGradeBtn, true, "Grading...")', editor_js)
        self.assertIn('setButtonBusy(assignmentExportMarkdownBtn, true, "Exporting...")', editor_js)
        self.assertIn('showBanner("Deleting the saved module...", "info")', dashboard_js)
        self.assertIn('setStatus(enabling ? "Turning sharing on..." : "Turning sharing off...", "loading")', dashboard_js)

    def test_module_editor_uses_html_rendering_for_citations_not_plain_text(self) -> None:
        editor_js = (ROOT_DIR / "static" / "module_editor.js").read_text()

        self.assertIn("sectionsEl.innerHTML = sections", editor_js)
        self.assertIn("${renderSectionCitations(section)}", editor_js)

    def test_apply_topic_bridge_updates_overview_and_first_section(self) -> None:
        request = _dummy_request("Probability", related_to_previous=True)
        module = _dummy_module("module-2", "Probability", overview="This module introduces simple probability ideas.")
        bridge = TopicBridgeContext(
            relation="somewhat related",
            previous_topic="Percentiles",
            previous_objectives=["Read percentile rankings", "Compare percentile positions"],
            previous_summary="The last module explained how percentile ranks compare values within a group.",
            reason="Both topics help learners interpret uncertainty and comparison in data.",
        )

        bridged = _apply_topic_bridge(module, request=request, topic_bridge=bridge)

        self.assertIn("Percentiles", bridged.overview)
        self.assertIn("Probability", bridged.overview)
        self.assertIn("Percentiles", bridged.sections[0].content)
        self.assertIn("Probability", bridged.sections[0].content)

    def test_generate_endpoint_passes_topic_bridge_when_requested(self) -> None:
        previous_request = _dummy_request("Percentiles")
        previous_module = _dummy_module(
            "module-prev",
            "Percentiles",
            overview="This module explained how percentile ranks compare values in a set.",
        )
        previous_evidence = [
            EvidenceItem(
                evidence_id="E001",
                source_type="web",
                domain="vocabulary.com",
                title="Percentile definition",
                url="https://www.vocabulary.com/dictionary/percentile",
                snippet="A percentile tells what percent of values fall below a score.",
            )
        ]
        module_store.save(
            "module-prev",
            previous_request,
            previous_module,
            previous_evidence,
            source_policy=previous_request.effective_source_policy(),
        )

        evidence_pack = [
            EvidenceItem(
                evidence_id="E001",
                source_type="web",
                domain="kids.britannica.com",
                title="Probability",
                url="https://kids.britannica.com/students/article/probability/123456",
                snippet="Probability describes how likely something is to happen.",
            )
        ]
        evidence_result = EvidenceBuildResult(
            evidence_pack=evidence_pack,
            web_unavailable_objectives=[],
            objectives_without_evidence=[],
        )
        captured: dict[str, object] = {}

        def fake_generate_module_from_evidence(*args, **kwargs) -> Module:
            captured["topic_bridge"] = kwargs.get("topic_bridge")
            return _dummy_module("module-new", "Probability", overview="This module introduces probability.")

        with patch("app.main.get_openai_client", return_value=object()), patch(
            "app.main.build_evidence_pack",
            return_value=evidence_result,
        ), patch(
            "app.main.detect_topic_relatedness",
            return_value=TopicRelatednessResult(
                relation="somewhat related",
                reason="Percentiles and probability are both used to interpret quantitative situations.",
            ),
        ), patch(
            "app.main.generate_module_from_evidence",
            side_effect=fake_generate_module_from_evidence,
        ), patch(
            "app.main.enforce_quality_gate",
            side_effect=lambda module, evidence_pack: module,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/modules/generate",
                    json={
                        "topic": "Probability",
                        "audience_level": "Middle school",
                        "learning_objectives": [
                            "Explain what probability means",
                            "Use probability words correctly",
                            "Apply the idea to a simple example",
                        ],
                        "allow_web": True,
                        "related_to_previous": True,
                    },
                )

        self.assertEqual(response.status_code, 200)
        topic_bridge = captured.get("topic_bridge")
        self.assertIsNotNone(topic_bridge)
        self.assertEqual(topic_bridge.previous_topic, "Percentiles")
        self.assertEqual(topic_bridge.relation, "somewhat related")
        self.assertEqual(topic_bridge.previous_summary, previous_module.overview)


if __name__ == "__main__":
    unittest.main()
