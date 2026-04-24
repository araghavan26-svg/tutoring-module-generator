from __future__ import annotations

from typing import Any, Dict

from ..config import settings
from ..errors import NotFoundError
from ..services.export_service import build_module_footnotes, share_url
from ..services.retrieval_helpers import current_evidence_pack, current_source_policy, format_timestamp_for_ui
from ..store import PersistentModuleStore


SAMPLES_DIR = settings.project_root / "samples"


def create_page_context(*, store: PersistentModuleStore) -> Dict[str, Any]:
    sample_doc_path = SAMPLES_DIR / "sample.txt"
    latest_record = store.latest()
    return {
        "title": "Create Module",
        "sample_doc_exists": sample_doc_path.exists(),
        "has_previous_module": latest_record is not None,
        "previous_topic": latest_record.request.topic if latest_record is not None else "",
    }


def modules_dashboard_context(*, request: Any, store: PersistentModuleStore) -> Dict[str, Any]:
    modules = store.list_modules()
    module_rows = []
    for item in modules:
        row = item.model_dump(mode="json")
        row["created_label"] = format_timestamp_for_ui(item.created_at)
        row["updated_label"] = format_timestamp_for_ui(item.updated_at)
        row["share_url"] = share_url(request, item.share_id) if item.share_enabled else None
        module_rows.append(row)
    return {"title": "Saved Modules", "modules": module_rows}


def module_editor_context(*, request: Any, module_id: str, store: PersistentModuleStore) -> Dict[str, Any]:
    record = store.get(module_id)
    if record is None:
        raise NotFoundError("Module not found.", error_code="module_not_found")
    module_data = record.module.model_dump(mode="json")
    module_data["evidence_pack"] = [item.model_dump(mode="json") for item in current_evidence_pack(record)]
    module_data["source_policy"] = current_source_policy(record).model_dump(mode="json")
    module_data["share_url"] = share_url(request, record.module.share_id) if record.module.share_enabled else None
    return {
        "title": "Module Editor",
        "module_id": module_id,
        "initial_module": module_data,
        "initial_history": [item.model_dump(mode="json") for item in store.history(module_id)],
    }


def shared_module_context(*, request: Any, share_id: str, store: PersistentModuleStore) -> Dict[str, Any]:
    record = store.get_by_share_id(share_id)
    if record is None:
        raise NotFoundError("Shared module is unavailable.", error_code="shared_module_unavailable")
    module_data = record.module.model_dump(mode="json")
    footnotes = build_module_footnotes(module_data, current_evidence_pack(record))
    return {
        "title": record.module.title,
        "module": module_data,
        "footnotes": footnotes,
        "footnotes_by_id": {item["footnote_id"]: item for item in footnotes},
    }
