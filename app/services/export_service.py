from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..models import EvidenceItem, ModuleAssignmentResponse
from ..openai_service import human_readable_snippet


def build_module_footnotes(module_data: Dict[str, Any], evidence_pack: Sequence[EvidenceItem]) -> List[Dict[str, Any]]:
    evidence_by_id: Dict[str, EvidenceItem] = {
        item.evidence_id: item
        for item in evidence_pack
        if getattr(item, "evidence_id", "")
    }

    footnotes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for section in module_data.get("sections", []):
        citations = section.get("citations", []) if isinstance(section, dict) else []
        if not isinstance(citations, list):
            continue
        for citation_id in citations:
            citation_key = str(citation_id).strip()
            if not citation_key or citation_key in seen:
                continue
            seen.add(citation_key)
            evidence = evidence_by_id.get(citation_key)
            if evidence is None:
                continue
            footnotes.append(
                {
                    "footnote_id": citation_key,
                    "source_type": evidence.source_type,
                    "domain": evidence.domain,
                    "title": evidence.title,
                    "url": evidence.url,
                    "doc_name": evidence.doc_name,
                    "location": evidence.location,
                    "snippet": human_readable_snippet(evidence.snippet),
                    "retrieved_at": evidence.retrieved_at,
                }
            )
    return footnotes


def module_export_payload(module_data: Dict[str, Any], evidence_pack: Sequence[EvidenceItem]) -> Dict[str, Any]:
    return {
        "module_id": module_data.get("module_id"),
        "title": module_data.get("title"),
        "overview": module_data.get("overview", ""),
        "sections": module_data.get("sections", []),
        "glossary": module_data.get("glossary", []),
        "mcqs": module_data.get("mcqs", []),
        "footnotes": build_module_footnotes(module_data, evidence_pack),
    }


def share_url(request: Any, share_id: str | None) -> str | None:
    share_key = str(share_id or "").strip()
    if not share_key:
        return None
    return str(request.url_for("shared_module_page", share_id=share_key))


def footnote_markdown(note: Dict[str, Any]) -> str:
    footnote_id = str(note.get("footnote_id", "")).strip()
    title = str(note.get("title", "Untitled source")).strip() or "Untitled source"
    url = str(note.get("url", "") or "").strip()
    domain = str(note.get("domain", "") or "").strip()
    doc_name = str(note.get("doc_name", "") or "").strip()
    location = str(note.get("location", "") or "").strip()
    snippet = str(note.get("snippet", "") or "").strip()

    if url:
        label = f"[{title}]({url})"
        meta = f"Web source: {domain or 'external source'}"
    else:
        label = f"**{doc_name or title}**"
        meta_parts = ["Document source"]
        if location:
            meta_parts.append(location)
        meta = " | ".join(meta_parts)

    lines = [f"[^{footnote_id}]: {label}"]
    lines.append(f"    - {meta}")
    if snippet:
        lines.append(f"    - Snippet: {snippet}")
    return "\n".join(lines)


def module_export_markdown(module_data: Dict[str, Any], footnotes: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    title = str(module_data.get("title", "Untitled module")).strip() or "Untitled module"
    overview = str(module_data.get("overview", "")).strip() or "No overview provided."
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overview")
    lines.append(overview)
    lines.append("")

    lines.append("## Sections")
    sections = module_data.get("sections", [])
    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading", f"Section {idx + 1}")).strip() or f"Section {idx + 1}"
        learning_goal = str(section.get("learning_goal", "")).strip() or "No learning goal provided."
        content = str(section.get("content", "")).strip() or "No content provided."
        lines.append(f"### {idx + 1}. {heading}")
        lines.append(f"Learning Goal: {learning_goal}")
        lines.append("")
        lines.append(content)
        citations = section.get("citations", [])
        if isinstance(citations, list) and citations:
            refs = ", ".join(f"[^{str(item).strip()}]" for item in citations if str(item).strip())
            lines.append("")
            lines.append(f"Citations: {refs}")
        lines.append("")

    lines.append("## Glossary")
    glossary = module_data.get("glossary", [])
    if isinstance(glossary, list) and glossary:
        for item in glossary:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            definition = str(item.get("definition", "")).strip()
            if term and definition:
                lines.append(f"- **{term}**: {definition}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## MCQs")
    mcqs = module_data.get("mcqs", [])
    if isinstance(mcqs, list) and mcqs:
        for idx, mcq in enumerate(mcqs):
            if not isinstance(mcq, dict):
                continue
            question = str(mcq.get("question", f"Question {idx + 1}")).strip()
            lines.append(f"{idx + 1}. {question}")
            options = mcq.get("options", [])
            if isinstance(options, list):
                for option_idx, option in enumerate(options):
                    letter = chr(ord("A") + option_idx)
                    lines.append(f"   - {letter}. {str(option).strip()}")
            answer_index = mcq.get("answer_index")
            explanation = str(mcq.get("explanation", "")).strip()
            lines.append(f"   - Answer index: {answer_index}")
            if explanation:
                lines.append(f"   - Explanation: {explanation}")
            lines.append("")
    else:
        lines.append("- (none)")
        lines.append("")

    lines.append("## Footnotes")
    if footnotes:
        for note in footnotes:
            lines.append(footnote_markdown(note))
    else:
        lines.append("- (none)")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def assignment_export_markdown(assignment: ModuleAssignmentResponse) -> str:
    lines: List[str] = []
    lines.append("# Assignment")
    lines.append("")
    lines.append("## Prompt")
    lines.append(assignment.prompt)
    lines.append("")
    lines.append("## Rubric")
    for index, criterion in enumerate(assignment.rubric, start=1):
        lines.append(f"### {index}. {criterion.criteria}")
        for level in criterion.levels:
            lines.append(f"- {level.score}: {level.description}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
