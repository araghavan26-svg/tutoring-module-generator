from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.openai_service import _extract_web_candidates, web_search_tool_args


class _FakeResponse:
    output_text = '{"items": []}'
    output = []


class _FakeResponses:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse()


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class WebSearchToolArgsTests(unittest.TestCase):
    def test_helper_never_includes_filters(self) -> None:
        tool = web_search_tool_args()
        self.assertEqual(tool.get("type"), "web_search")
        self.assertNotIn("filters", tool)

    def test_extract_web_candidates_call_never_passes_filters(self) -> None:
        client = _FakeClient()
        _extract_web_candidates(
            client,
            topic="Photosynthesis",
            audience_level="Middle school",
            objective="Explain light-dependent reactions",
            max_results=2,
            web_recency_days=30,
            allowed_domains=None,
            blocked_domains=None,
        )
        self.assertIsNotNone(client.responses.last_kwargs)
        tools = client.responses.last_kwargs.get("tools", [])
        self.assertTrue(tools)
        for tool in tools:
            self.assertNotIn("filters", tool)


if __name__ == "__main__":
    unittest.main()
