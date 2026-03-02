from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Dict, Optional

from .models import EvidenceItem, Module, ModuleGenerateRequest, utc_now


@dataclass
class ModuleRecord:
    request: ModuleGenerateRequest
    module: Module
    evidence_pack: list[EvidenceItem]
    updated_at: datetime = field(default_factory=utc_now)


class InMemoryModuleStore:
    def __init__(self) -> None:
        self._records: Dict[str, ModuleRecord] = {}
        self._lock = RLock()

    def save(
        self,
        module_id: str,
        request: ModuleGenerateRequest,
        module: Module,
        evidence_pack: list[EvidenceItem],
    ) -> None:
        with self._lock:
            self._records[module_id] = ModuleRecord(
                request=request,
                module=module,
                evidence_pack=evidence_pack,
            )

    def get(self, module_id: str) -> Optional[ModuleRecord]:
        with self._lock:
            return self._records.get(module_id)


module_store = InMemoryModuleStore()
