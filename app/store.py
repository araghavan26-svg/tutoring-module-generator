from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Dict, Optional

from .models import EvidenceItem, Module, ModuleGenerateRequest, SourcePolicy, utc_now


@dataclass
class ModuleRecord:
    request: ModuleGenerateRequest
    module: Module
    evidence_pack: list[EvidenceItem]
    source_policy: SourcePolicy
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
        source_policy: SourcePolicy | None = None,
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
            self._records[module_id] = ModuleRecord(
                request=request,
                module=persisted_module,
                evidence_pack=cached_evidence,
                source_policy=effective_source_policy,
            )

    def get(self, module_id: str) -> Optional[ModuleRecord]:
        with self._lock:
            return self._records.get(module_id)


module_store = InMemoryModuleStore()
