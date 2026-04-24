from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional
from uuid import uuid4

from .config import settings
from .models import (
    EvidenceItem,
    Module,
    ModuleGenerateRequest,
    ModuleLibraryItem,
    ModuleVersionSummary,
    SourcePolicy,
    TopicContinuityContext,
    VersionAction,
    utc_now,
)

DEFAULT_DATA_DIR = settings.data_dir
STORE_FILENAME = "module_store.json"


def continuity_context_from_record(request: ModuleGenerateRequest, module: Module) -> TopicContinuityContext:
    return TopicContinuityContext(
        topic=request.topic,
        learning_objectives=list(request.learning_objectives or []),
        module_summary=str(module.overview or "").strip(),
    )


def _parse_datetime(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return utc_now()
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return utc_now()


@dataclass
class ModuleRecord:
    request: ModuleGenerateRequest
    module: Module
    evidence_pack: list[EvidenceItem]
    source_policy: SourcePolicy
    previous_context: TopicContinuityContext | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class ModuleVersionRecord:
    version_id: str
    module_id: str
    timestamp: datetime
    action: VersionAction
    module_snapshot: dict[str, Any]

    def summary(self) -> ModuleVersionSummary:
        return ModuleVersionSummary(
            version_id=self.version_id,
            module_id=self.module_id,
            timestamp=self.timestamp,
            action=self.action,
        )


class PersistentModuleStore:
    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self._file_path = self._data_dir / STORE_FILENAME
        self._records: Dict[str, ModuleRecord] = {}
        self._versions: Dict[str, list[ModuleVersionRecord]] = {}
        self._lock = RLock()
        self.load_from_disk()

    def _serialize_locked(self) -> dict[str, Any]:
        return {
            "modules": {
                module_id: {
                    "request": record.request.model_dump(mode="json"),
                    "module": record.module.model_dump(mode="json"),
                    "evidence_pack": [item.model_dump(mode="json") for item in record.evidence_pack],
                    "source_policy": record.source_policy.model_dump(mode="json"),
                    "previous_context": (
                        record.previous_context.model_dump(mode="json")
                        if record.previous_context is not None
                        else None
                    ),
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                }
                for module_id, record in self._records.items()
            },
            "versions": {
                module_id: [
                    {
                        "version_id": item.version_id,
                        "module_id": item.module_id,
                        "timestamp": item.timestamp.isoformat(),
                        "action": item.action,
                        "module_snapshot": item.module_snapshot,
                    }
                    for item in versions
                ]
                for module_id, versions in self._versions.items()
            },
        }

    def _write_to_disk_locked(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        payload = self._serialize_locked()
        temp_path = self._file_path.with_suffix(f"{self._file_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        temp_path.replace(self._file_path)

    def load_from_disk(self) -> None:
        with self._lock:
            if not self._file_path.exists():
                self._records = {}
                self._versions = {}
                return

            payload = json.loads(self._file_path.read_text(encoding="utf-8"))
            raw_modules = payload.get("modules", {})
            raw_versions = payload.get("versions", {})
            records: Dict[str, ModuleRecord] = {}
            versions: Dict[str, list[ModuleVersionRecord]] = {}

            if isinstance(raw_modules, dict):
                for module_id, raw_record in raw_modules.items():
                    if not isinstance(raw_record, dict):
                        continue
                    request = ModuleGenerateRequest.model_validate(raw_record.get("request") or {})
                    module = Module.model_validate(raw_record.get("module") or {})
                    evidence_pack = [
                        EvidenceItem.model_validate(item)
                        for item in (raw_record.get("evidence_pack") or [])
                        if isinstance(item, dict)
                    ]
                    source_policy_data = raw_record.get("source_policy")
                    if isinstance(source_policy_data, dict):
                        source_policy = SourcePolicy.model_validate(source_policy_data)
                    else:
                        source_policy = module.source_policy or request.effective_source_policy()
                    previous_context_data = raw_record.get("previous_context")
                    previous_context = (
                        TopicContinuityContext.model_validate(previous_context_data)
                        if isinstance(previous_context_data, dict)
                        else None
                    )
                    updated_at = _parse_datetime(raw_record.get("updated_at"))
                    created_at = _parse_datetime(raw_record.get("created_at"))
                    if "created_at" not in raw_record:
                        raw_module_versions = raw_versions.get(module_id, []) if isinstance(raw_versions, dict) else []
                        parsed_timestamps = [
                            _parse_datetime(item.get("timestamp"))
                            for item in raw_module_versions
                            if isinstance(item, dict)
                        ]
                        if parsed_timestamps:
                            created_at = min(parsed_timestamps)
                        else:
                            created_at = updated_at
                    records[module_id] = ModuleRecord(
                        request=request,
                        module=module,
                        evidence_pack=evidence_pack,
                        source_policy=source_policy,
                        previous_context=previous_context,
                        created_at=created_at,
                        updated_at=updated_at,
                    )

            if isinstance(raw_versions, dict):
                for module_id, raw_module_versions in raw_versions.items():
                    if not isinstance(raw_module_versions, list):
                        continue
                    parsed_versions: list[ModuleVersionRecord] = []
                    for item in raw_module_versions:
                        if not isinstance(item, dict):
                            continue
                        parsed_versions.append(
                            ModuleVersionRecord(
                                version_id=str(item.get("version_id") or uuid4()),
                                module_id=str(item.get("module_id") or module_id),
                                timestamp=_parse_datetime(item.get("timestamp")),
                                action=str(item.get("action") or "generated"),
                                module_snapshot=item.get("module_snapshot") or {},
                            )
                        )
                    versions[module_id] = parsed_versions

            self._records = records
            self._versions = versions

    def save(
        self,
        module_id: str,
        request: ModuleGenerateRequest,
        module: Module,
        evidence_pack: list[EvidenceItem],
        source_policy: SourcePolicy | None = None,
        action: VersionAction | None = None,
    ) -> None:
        effective_source_policy = source_policy or module.source_policy or request.effective_source_policy()
        cached_evidence = [item.model_copy(deep=True) for item in evidence_pack]
        persisted_module = module.model_copy(
            deep=True,
            update={
                "source_policy": effective_source_policy,
                "evidence_pack": cached_evidence,
            },
        )
        with self._lock:
            existing = self._records.get(module_id)
            if existing is not None:
                previous_context = existing.previous_context
                created_at = existing.created_at
            else:
                latest_other = self.latest(exclude_module_id=module_id)
                previous_context = (
                    continuity_context_from_record(latest_other.request, latest_other.module)
                    if latest_other is not None
                    else None
                )
                created_at = utc_now()
            self._records[module_id] = ModuleRecord(
                request=request,
                module=persisted_module,
                evidence_pack=cached_evidence,
                source_policy=effective_source_policy,
                previous_context=previous_context,
                created_at=created_at,
                updated_at=utc_now(),
            )
            if action is not None:
                self._versions.setdefault(module_id, []).append(
                    ModuleVersionRecord(
                        version_id=str(uuid4()),
                        module_id=module_id,
                        timestamp=utc_now(),
                        action=action,
                        module_snapshot=persisted_module.model_dump(mode="json"),
                    )
                )
            self._write_to_disk_locked()

    def get(self, module_id: str) -> Optional[ModuleRecord]:
        with self._lock:
            return self._records.get(module_id)

    def history(self, module_id: str) -> list[ModuleVersionSummary]:
        with self._lock:
            versions = list(self._versions.get(module_id, []))
        versions.sort(key=lambda item: item.timestamp, reverse=True)
        return [version.summary() for version in versions]

    def list_modules(self) -> list[ModuleLibraryItem]:
        with self._lock:
            items = [
                ModuleLibraryItem(
                    module_id=module_id,
                    title=record.module.title,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    section_count=len(record.module.sections),
                    share_enabled=record.module.share_enabled,
                    share_id=record.module.share_id,
                )
                for module_id, record in self._records.items()
            ]
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return items

    def get_by_share_id(self, share_id: str) -> Optional[ModuleRecord]:
        share_key = str(share_id or "").strip()
        if not share_key:
            return None
        with self._lock:
            for record in self._records.values():
                if record.module.share_enabled and record.module.share_id == share_key:
                    return record
        return None

    def set_share_enabled(self, module_id: str, enabled: bool) -> Optional[Module]:
        with self._lock:
            record = self._records.get(module_id)
            if record is None:
                return None

            next_share_id = record.module.share_id if enabled else None
            if enabled and not next_share_id:
                next_share_id = str(uuid4())

            updated_module = record.module.model_copy(
                deep=True,
                update={
                    "share_enabled": enabled,
                    "share_id": next_share_id,
                },
            )
            self._records[module_id] = ModuleRecord(
                request=record.request,
                module=updated_module,
                evidence_pack=[item.model_copy(deep=True) for item in record.evidence_pack],
                source_policy=record.source_policy,
                previous_context=record.previous_context,
                created_at=record.created_at,
                updated_at=utc_now(),
            )
            self._write_to_disk_locked()
            return updated_module.model_copy(deep=True)

    def revert(self, module_id: str, version_id: str) -> Optional[Module]:
        with self._lock:
            record = self._records.get(module_id)
            if record is None:
                return None
            version = next(
                (item for item in self._versions.get(module_id, []) if item.version_id == version_id),
                None,
            )
            if version is None:
                return None

            restored_module = Module.model_validate(version.module_snapshot)
            restored_source_policy = restored_module.source_policy or record.source_policy
            restored_evidence = [item.model_copy(deep=True) for item in restored_module.evidence_pack]
            updated_request = record.request.model_copy(
                update={
                    "allow_web": restored_source_policy.allow_web,
                    "source_policy": restored_source_policy,
                }
            )
            self._records[module_id] = ModuleRecord(
                request=updated_request,
                module=restored_module.model_copy(deep=True),
                evidence_pack=restored_evidence,
                source_policy=restored_source_policy,
                previous_context=record.previous_context,
                created_at=record.created_at,
                updated_at=utc_now(),
            )
            self._write_to_disk_locked()
            return self._records[module_id].module.model_copy(deep=True)

    def delete(self, module_id: str) -> bool:
        with self._lock:
            if module_id not in self._records:
                return False
            self._records.pop(module_id, None)
            self._versions.pop(module_id, None)
            self._write_to_disk_locked()
            return True

    def latest(self, exclude_module_id: str | None = None) -> Optional[ModuleRecord]:
        with self._lock:
            candidates = [
                record
                for key, record in self._records.items()
                if exclude_module_id is None or key != exclude_module_id
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda item: item.updated_at)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._versions.clear()
            if self._file_path.exists():
                self._file_path.unlink()
            temp_path = self._file_path.with_suffix(f"{self._file_path.suffix}.tmp")
            if temp_path.exists():
                temp_path.unlink()


module_store = PersistentModuleStore()
