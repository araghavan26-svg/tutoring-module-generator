from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.models import SourcePolicy
from app.openai_service import (
    WebCandidateResult,
    WEB_SNIPPET_FALLBACK,
    _default_source_preferences,
    _extract_doc_candidates,
    _extract_web_candidates,
    build_evidence_pack,
    human_readable_snippet,
)


class _FakeDocCall:
    type = "file_search_call"

    def __init__(self, results):
        self._results = results

    def model_dump(self, mode: str = "python", exclude_none: bool = True):
        return {"results": self._results}


class _FakeDocResponse:
    output_text = ""

    def __init__(self, results):
        self.output = [_FakeDocCall(results)]


class _FakeWebCall:
    type = "web_search_call"

    def __init__(self, sources):
        self._sources = sources

    def model_dump(self, mode: str = "python", exclude_none: bool = True):
        return {"action": {"sources": self._sources}}


class _FakeWebResponse:
    def __init__(self, items, sources):
        self.output_text = json.dumps({"items": items})
        self.output = [_FakeWebCall(sources)]


class _FakeResponses:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.responses = _FakeResponses(response)


class EvidenceExtractionTests(unittest.TestCase):
    def test_human_readable_snippet_strips_serialized_json(self) -> None:
        raw = '{"items":[{"title":"Photosynthesis","snippet":"Plants use sunlight to make food."}]}'

        snippet = human_readable_snippet(raw)

        self.assertEqual(snippet, "Plants use sunlight to make food.")
        self.assertNotIn('{"items":', snippet)

    def test_extract_web_candidates_returns_clean_single_best_snippet(self) -> None:
        client = _FakeClient(
            _FakeWebResponse(
                items=[
                    {
                        "title": "kids.britannica.com",
                        "url": "https://kids.britannica.com/students/article/photosynthesis/123",
                        "snippet": '{"items":[{"snippet":"Plants use sunlight to make food."}]}',
                    },
                    {
                        "title": "www.vocabulary.com",
                        "url": "https://www.vocabulary.com/dictionary/photosynthesis",
                        "snippet": "A process used by plants.",
                    },
                ],
                sources=[
                    {
                        "title": "Photosynthesis for Students",
                        "url": "https://kids.britannica.com/students/article/photosynthesis/123",
                        "snippet": "Plants use sunlight to make food.",
                    },
                    {
                        "title": "Photosynthesis Definition",
                        "url": "https://www.vocabulary.com/dictionary/photosynthesis",
                        "snippet": "A process used by plants.",
                    },
                ],
            )
        )

        result = _extract_web_candidates(
            client,
            topic="Photosynthesis",
            audience_level="Middle school",
            objective="Explain how plants use sunlight",
            max_results=2,
            web_recency_days=30,
            allowed_domains=["kids.britannica.com", "www.vocabulary.com"],
            blocked_domains=None,
        )

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0]["domain"], "kids.britannica.com")
        self.assertEqual(result.items[0]["title"], "Photosynthesis for Students")
        self.assertNotEqual(result.items[0]["title"], "kids.britannica.com")
        self.assertEqual(result.items[0]["snippet"], "Plants use sunlight to make food.")
        self.assertNotEqual(result.items[0]["snippet"], "kids.britannica.com")
        self.assertNotIn('{"items":', result.items[0]["snippet"])
        self.assertNotEqual(result.items[0]["snippet"], "Relevant source retrieved via web search.")

    def test_extract_web_candidates_recovers_from_domain_only_title_and_snippet(self) -> None:
        client = _FakeClient(
            _FakeWebResponse(
                items=[
                    {
                        "title": "kids.britannica.com",
                        "url": "https://kids.britannica.com/students/article/photosynthesis/123",
                        "snippet": "kids.britannica.com",
                    }
                ],
                sources=[
                    {
                        "headline": "Photosynthesis for Students",
                        "url": "https://kids.britannica.com/students/article/photosynthesis/123",
                        "description": "Plants use sunlight, water, and carbon dioxide to make food.",
                    }
                ],
            )
        )

        result = _extract_web_candidates(
            client,
            topic="Photosynthesis",
            audience_level="Middle school",
            objective="Explain how plants use sunlight",
            max_results=1,
            web_recency_days=30,
            allowed_domains=["kids.britannica.com"],
            blocked_domains=None,
        )

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0]["title"], "Photosynthesis for Students")
        self.assertNotEqual(result.items[0]["title"], "kids.britannica.com")
        self.assertEqual(
            result.items[0]["snippet"],
            "Plants use sunlight, water, and carbon dioxide to make food.",
        )
        self.assertNotEqual(result.items[0]["snippet"], "kids.britannica.com")

    def test_extract_web_candidates_uses_human_readable_fallback_when_no_snippet_exists(self) -> None:
        client = _FakeClient(
            _FakeWebResponse(
                items=[
                    {
                        "title": "kids.britannica.com",
                        "url": "https://kids.britannica.com/students/article/photosynthesis/123",
                        "snippet": "kids.britannica.com",
                    }
                ],
                sources=[
                    {
                        "title": "Photosynthesis for Students",
                        "url": "https://kids.britannica.com/students/article/photosynthesis/123",
                    }
                ],
            )
        )

        result = _extract_web_candidates(
            client,
            topic="Photosynthesis",
            audience_level="Middle school",
            objective="Identify key vocabulary",
            max_results=1,
            web_recency_days=30,
            allowed_domains=["kids.britannica.com"],
            blocked_domains=None,
        )

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0]["title"], "Photosynthesis for Students")
        self.assertEqual(result.items[0]["snippet"], WEB_SNIPPET_FALLBACK)
        self.assertNotEqual(result.items[0]["snippet"], "kids.britannica.com")

    def test_extract_web_candidates_prefers_stronger_academic_domains_by_default(self) -> None:
        client = _FakeClient(
            _FakeWebResponse(
                items=[
                    {
                        "title": "Statistics course guide",
                        "url": "https://bestcolleges.com/courses/statistics-overview",
                        "snippet": "Statistics courses cover data and probability.",
                    },
                    {
                        "title": "AP Statistics",
                        "url": "https://www.khanacademy.org/math/ap-statistics",
                        "snippet": "AP Statistics studies data, probability, and inference.",
                    },
                ],
                sources=[
                    {
                        "title": "Statistics course guide",
                        "url": "https://bestcolleges.com/courses/statistics-overview",
                        "snippet": "Statistics courses cover data and probability.",
                    },
                    {
                        "title": "AP Statistics",
                        "url": "https://www.khanacademy.org/math/ap-statistics",
                        "snippet": "AP Statistics studies data, probability, and inference.",
                    },
                ],
            )
        )

        result = _extract_web_candidates(
            client,
            subject="Statistics",
            topic="Probability",
            audience_level="AP Statistics",
            objective="Understand core AP Statistics concepts and expectations",
            max_results=2,
            web_recency_days=30,
            allowed_domains=None,
            blocked_domains=None,
            preferred_domains=["khanacademy.org", "collegeboard.org", "britannica.com"],
            academic_mode=True,
        )

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0]["domain"], "khanacademy.org")

    def test_source_preferences_add_beginner_friendly_and_high_trust_defaults(self) -> None:
        preferred_domains, academic_mode = _default_source_preferences(
            subject="Biology",
            topic="Photosynthesis",
            source_preference="Beginner-friendly",
            prefer_high_trust_sources=True,
        )

        self.assertIn("kids.britannica.com", preferred_domains)
        self.assertIn("khanacademy.org", preferred_domains)
        self.assertIn("britannica.com", preferred_domains)
        self.assertTrue(academic_mode)

    def test_extract_doc_candidates_uses_chunk_location_for_txt(self) -> None:
        client = _FakeClient(
            _FakeDocResponse(
                results=[
                    {"filename": "sample.txt", "text": "Photosynthesis uses light.", "attributes": {}},
                    {"filename": "sample.txt", "text": "Plants also need water.", "attributes": {}},
                ]
            )
        )

        results = _extract_doc_candidates(
            client,
            vector_store_id="vs_test",
            topic="Photosynthesis",
            audience_level="Middle school",
            objective="Explain the process",
            max_results=2,
        )

        self.assertEqual(results[0]["location"], "chunk:1")
        self.assertEqual(results[1]["location"], "chunk:2")
        self.assertNotEqual(results[0]["location"], "unknown")

    def test_build_evidence_pack_reuses_doc_evidence_for_later_objective_when_web_is_filtered(self) -> None:
        first_objective = "Explain the process"
        later_objective = "Connect photosynthesis to plant growth"
        with patch(
            "app.openai_service._extract_doc_candidates",
            side_effect=[
                [
                    {
                        "source_type": "doc",
                        "title": "sample.txt",
                        "url": None,
                        "doc_name": "sample.txt",
                        "location": "chunk:1",
                        "snippet": "Photosynthesis helps plants make sugar for growth and survival.",
                        "retrieved_at": None,
                    }
                ],
                [],
            ],
        ), patch(
            "app.openai_service._extract_web_candidates",
            side_effect=[
                WebCandidateResult(items=[], filtered_out_by_policy=False),
                WebCandidateResult(items=[], filtered_out_by_policy=True),
            ],
        ):
            result = build_evidence_pack(
                client=object(),
                topic="Photosynthesis",
                audience_level="Middle school",
                learning_objectives=[first_objective, later_objective],
                allow_web=True,
                vector_store_id="vs_test",
                source_policy=SourcePolicy(
                    allow_web=True,
                    allowed_domains=["kids.britannica.com"],
                    blocked_domains=None,
                ),
                start_index=1,
            )

        self.assertEqual(len(result.evidence_pack), 1)
        self.assertEqual(result.evidence_pack[0].source_type, "doc")
        self.assertEqual(result.objectives_without_evidence, [])
        self.assertEqual(result.web_unavailable_objectives, [later_objective])


if __name__ == "__main__":
    unittest.main()
