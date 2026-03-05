from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.models import normalize_domain
from app.openai_service import domain_matches_policy


class DomainPolicyTests(unittest.TestCase):
    def test_normalize_domain_strips_scheme_path_and_www(self) -> None:
        self.assertEqual(normalize_domain("https://www.vocabulary.com/lists/12345"), "vocabulary.com")
        self.assertEqual(normalize_domain("http://kids.britannica.com/science/photosynthesis"), "kids.britannica.com")
        self.assertEqual(normalize_domain("WWW.EXAMPLE.ORG/path?q=1"), "example.org")

    def test_domain_matches_allowlist(self) -> None:
        allowed = ["kids.britannica.com", "www.vocabulary.com"]
        self.assertTrue(domain_matches_policy("https://www.vocabulary.com/word", allowed, None))
        self.assertTrue(domain_matches_policy("kids.britannica.com", allowed, None))
        self.assertFalse(domain_matches_policy("example.com", allowed, None))

    def test_blocklist_always_excludes(self) -> None:
        allowed = ["vocabulary.com"]
        blocked = ["www.vocabulary.com"]
        self.assertFalse(domain_matches_policy("www.vocabulary.com", allowed, blocked))
        self.assertFalse(domain_matches_policy("sub.www.vocabulary.com", allowed, blocked))
        self.assertTrue(domain_matches_policy("kids.vocabulary.com", allowed, ["other.com"]))


if __name__ == "__main__":
    unittest.main()

