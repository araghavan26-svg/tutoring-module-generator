#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from app.config import ensure_openai_api_key
from app.main import app


AZURE_WEB_SEARCH_HINT = "If using Azure, switch web_search to web_search_preview."


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _get_error_detail(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        return (getattr(response, "text", "") or "").strip() or "Unknown error"
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail is not None:
            return str(detail)
    return str(payload)


def _maybe_print_azure_hint(detail: str) -> None:
    text = (detail or "").lower()
    if "web_search" in text and (
        "invalid" in text
        or "unknown" in text
        or "unsupported" in text
        or "tool" in text
        or "not allowed" in text
    ):
        print(AZURE_WEB_SEARCH_HINT, file=sys.stderr)


def _short_source(evidence: Dict[str, Any]) -> str:
    if str(evidence.get("source_type", "")).strip() == "web":
        return str(evidence.get("url", "")).strip() or "missing-url"
    return str(evidence.get("doc_name", "")).strip() or "doc-source"


def _validate_grounding_response(body: Dict[str, Any]) -> List[Tuple[str, str]]:
    module = body.get("module", {})
    sections = module.get("sections", [])
    evidence_pack = body.get("evidence_pack", [])

    _assert(isinstance(sections, list), "Response module.sections must be a list.")
    _assert(len(sections) >= 3, "Expected at least 3 sections.")
    _assert(isinstance(evidence_pack, list), "Response evidence_pack must be a list.")

    evidence_by_id: Dict[str, Dict[str, Any]] = {}
    for item in evidence_pack:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id", "")).strip()
        if evidence_id:
            evidence_by_id[evidence_id] = item

    report_rows: List[Tuple[str, str]] = []

    for idx, section in enumerate(sections):
        _assert(isinstance(section, dict), f"Section at index {idx} is not an object.")
        heading = str(section.get("heading", "")).strip() or f"Section {idx + 1}"
        citations = section.get("citations") or []
        unverified = bool(section.get("unverified", False))

        _assert(isinstance(citations, list), f"Section '{heading}' citations must be a list.")
        _assert(bool(citations) or unverified, f"Section '{heading}' must have citations or unverified=true.")

        first_citation_summary = "none (unverified)"
        for citation_id in citations:
            evidence_id = str(citation_id).strip()
            _assert(evidence_id, f"Section '{heading}' contains empty citation id.")
            evidence = evidence_by_id.get(evidence_id)
            _assert(evidence is not None, f"Section '{heading}' references unknown evidence id '{evidence_id}'.")

            snippet = str(evidence.get("snippet", "")).strip()
            _assert(snippet, f"Evidence '{evidence_id}' must include a non-empty snippet.")

            source_type = str(evidence.get("source_type", "")).strip()
            if source_type == "web":
                url = str(evidence.get("url", "")).strip()
                _assert(url, f"Web evidence '{evidence_id}' must include a non-empty url.")

            if first_citation_summary.startswith("none"):
                title = str(evidence.get("title", "")).strip() or "Untitled source"
                first_citation_summary = f"{title} | {_short_source(evidence)}"

        report_rows.append((heading, first_citation_summary))

    return report_rows


def _post_or_fail(client: TestClient, path: str, *, json_body: Dict[str, Any] | None = None, files: Any = None) -> Dict[str, Any]:
    response = client.post(path, json=json_body, files=files)
    if response.status_code >= 400:
        detail = _get_error_detail(response)
        _maybe_print_azure_hint(detail)
        raise AssertionError(f"{path} failed with {response.status_code}: {detail}")
    payload = response.json()
    _assert(isinstance(payload, dict), f"{path} response is not a JSON object.")
    return payload


def run_web_smoke_test(client: TestClient, topic: str, audience_level: str) -> None:
    payload = {
        "topic": topic,
        "audience_level": audience_level,
        "learning_objectives": [
            "Explain the core concept in plain language",
            "Identify key terms and definitions",
            "Apply the concept in a simple example",
        ],
        "allow_web": True,
        "vector_store_id": None,
    }

    body = _post_or_fail(client, "/v1/modules/generate", json_body=payload)
    report_rows = _validate_grounding_response(body)

    print("Web grounding smoke test: PASS")
    for heading, summary in report_rows:
        print(f"- {heading}: {summary}")


def run_docs_only_smoke_test(client: TestClient, sample_file: Path, audience_level: str) -> None:
    _assert(sample_file.exists(), f"Sample file not found: {sample_file}")
    _assert(sample_file.is_file(), f"Sample path is not a file: {sample_file}")

    content_type = "text/plain"
    files = [("files", (sample_file.name, sample_file.read_bytes(), content_type))]
    upload_body = _post_or_fail(client, "/v1/docs/upload", files=files)
    vector_store_id = str(upload_body.get("vector_store_id", "")).strip()
    _assert(vector_store_id, "Upload response missing vector_store_id.")
    docs = upload_body.get("docs", [])
    _assert(isinstance(docs, list) and docs, "Upload response must include at least one doc metadata item.")
    indexed_docs = 0
    for item in docs:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        vector_file_id = str(item.get("vector_store_file_id", "")).strip()
        if status in {"failed", "cancelled"}:
            continue
        if vector_file_id:
            indexed_docs += 1
    _assert(indexed_docs >= 1, "Expected at least one uploaded doc indexed into the vector store.")

    payload = {
        "topic": "Grounding docs-only smoke test",
        "audience_level": audience_level,
        "learning_objectives": [
            "Summarize the provided document",
            "Extract key points from the provided document",
            "Teach one example based on the provided document",
        ],
        "allow_web": False,
        "vector_store_id": vector_store_id,
    }
    body = _post_or_fail(client, "/v1/modules/generate", json_body=payload)
    report_rows = _validate_grounding_response(body)

    module = body.get("module", {})
    sections = module.get("sections", [])
    evidence_pack = body.get("evidence_pack", [])
    evidence_by_id = {
        str(item.get("evidence_id", "")).strip(): item
        for item in evidence_pack
        if isinstance(item, dict) and str(item.get("evidence_id", "")).strip()
    }

    cited_ids: List[str] = []
    doc_grounded_sections = 0
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_has_valid_doc = False
        for citation_id in section.get("citations") or []:
            evidence_id = str(citation_id).strip()
            if evidence_id:
                cited_ids.append(evidence_id)
                evidence = evidence_by_id.get(evidence_id)
                if isinstance(evidence, dict) and str(evidence.get("source_type", "")).strip() == "doc":
                    doc_name = str(evidence.get("doc_name", "")).strip()
                    location = str(evidence.get("location", "")).strip()
                    snippet = str(evidence.get("snippet", "")).strip()
                    if doc_name and location and snippet:
                        section_has_valid_doc = True
        if section_has_valid_doc:
            doc_grounded_sections += 1

    _assert(cited_ids, "Docs-only test expected at least one cited evidence item.")
    for evidence_id in cited_ids:
        evidence = evidence_by_id.get(evidence_id)
        _assert(evidence is not None, f"Docs-only citation '{evidence_id}' missing from evidence_pack.")
        _assert(
            str(evidence.get("source_type", "")).strip() == "doc",
            f"Docs-only citation '{evidence_id}' must have source_type='doc'.",
        )
        _assert(str(evidence.get("doc_name", "")).strip(), f"Docs-only citation '{evidence_id}' missing doc_name.")
        _assert(str(evidence.get("location", "")).strip(), f"Docs-only citation '{evidence_id}' missing location.")
        _assert(str(evidence.get("snippet", "")).strip(), f"Docs-only citation '{evidence_id}' missing snippet.")
    _assert(
        doc_grounded_sections >= 2,
        "Docs-only test expected at least 2 sections with doc citations containing doc_name/location/snippet.",
    )

    print("Docs-only smoke test: PASS")
    print(f"- sample file: {sample_file}")
    for heading, summary in report_rows:
        print(f"- {heading}: {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end smoke test for grounding and citations.")
    parser.add_argument(
        "--topic",
        default="Photosynthesis",
        help="Topic used for web-enabled smoke test.",
    )
    parser.add_argument(
        "--audience-level",
        default="High school",
        help="Audience level used for module generation.",
    )
    parser.add_argument(
        "--sample-file",
        default="samples/sample.txt",
        help="Local txt file for docs-only smoke test (runs by default unless --skip-doc-test).",
    )
    parser.add_argument(
        "--skip-doc-test",
        action="store_true",
        help="Skip the docs-only smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_openai_api_key()

    sample_file_arg = Path(args.sample_file)
    sample_file = sample_file_arg if sample_file_arg.is_absolute() else (ROOT_DIR / sample_file_arg)
    run_doc_test = not args.skip_doc_test

    with TestClient(app) as client:
        run_web_smoke_test(client, topic=args.topic, audience_level=args.audience_level)
        if run_doc_test:
            run_docs_only_smoke_test(client, sample_file=sample_file, audience_level=args.audience_level)
        else:
            print("Docs-only smoke test: SKIPPED")
            print(f"- rerun without --skip-doc-test to validate docs grounding using '{sample_file}'.")

    print("Smoke test complete: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Smoke test FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
