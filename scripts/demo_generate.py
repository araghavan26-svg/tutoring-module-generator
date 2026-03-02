#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict
from urllib import error, request as urllib_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call /v1/modules/generate and print section citations + snippets.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL.")
    parser.add_argument("--topic", default="Photosynthesis", help="Module topic.")
    parser.add_argument("--audience-level", default="High school", help="Audience level.")
    parser.add_argument(
        "--objective",
        action="append",
        dest="objectives",
        help="Learning objective (repeat flag for multiple).",
    )
    parser.add_argument("--allow-web", action="store_true", help="Enable web search evidence.")
    parser.add_argument("--vector-store-id", default=None, help="Optional vector store id from /v1/docs/upload.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    objectives = args.objectives or [
        "Explain the core process",
        "Describe key terms",
        "Apply the concept to a simple example",
    ]
    payload = {
        "topic": args.topic,
        "audience_level": args.audience_level,
        "learning_objectives": objectives,
        "allow_web": bool(args.allow_web),
        "vector_store_id": args.vector_store_id,
    }
    url = f"{args.base_url.rstrip('/')}/v1/modules/generate"
    body_raw = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body_raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=args.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"Request failed ({exc.code}): {detail}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    module = body.get("module", {})
    evidence_pack = body.get("evidence_pack", [])
    evidence_by_id: Dict[str, dict] = {item.get("evidence_id"): item for item in evidence_pack}

    print(f"Module: {module.get('title', 'Untitled')} ({module.get('module_id', 'n/a')})")
    print(f"Evidence items: {len(evidence_pack)}")

    for section in module.get("sections", []):
        heading = section.get("heading", "Untitled section")
        print(f"\nSection: {heading}")
        if section.get("unverified"):
            print(f"  unverified: {section.get('unverified_reason', '').strip() or 'No reason provided.'}")

        citation_ids = section.get("citations", []) or []
        if not citation_ids:
            print("  citations: none")
            continue

        for evidence_id in citation_ids:
            evidence = evidence_by_id.get(evidence_id) or {}
            title = evidence.get("title", "Unknown title")
            url_or_doc = evidence.get("url") or evidence.get("doc_name") or "unknown-source"
            snippet = evidence.get("snippet", "").strip()
            print(f"  - {evidence_id}: {title} | {url_or_doc}")
            print(f"    snippet: {snippet}")

    print("\nRaw JSON response:")
    print(json.dumps(body, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
