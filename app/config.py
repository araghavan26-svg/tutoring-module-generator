from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
MISSING_KEY_MESSAGE = (
    "Missing OPENAI_API_KEY. Copy .env.example to .env and paste your own OpenAI API key."
)


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    env_file_path: Path
    data_dir: Path
    retrieval_model: str
    generation_model: str
    doc_results_per_objective: int
    web_results_per_objective: int
    max_evidence_items: int
    fast_section_limit: int
    fast_doc_results_per_objective: int
    fast_web_results_per_objective: int
    fast_max_evidence_items: int
    openai_timeout_seconds: float
    retrieval_timeout_seconds: float
    generation_timeout_seconds: float
    upload_timeout_seconds: float
    vector_poll_attempts: int
    vector_poll_sleep_seconds: float
    log_level: str


def _load_dotenv_if_present() -> None:
    if not ENV_FILE_PATH.exists():
        return

    try:
        content = ENV_FILE_PATH.read_text(encoding="utf-8")
    except OSError:
        return

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _str_env(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw or default


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    _load_dotenv_if_present()
    raw_data_dir = os.getenv("TUTOR_DATA_DIR", "").strip()
    data_dir = Path(raw_data_dir).expanduser() if raw_data_dir else PROJECT_ROOT / "data"
    return AppSettings(
        project_root=PROJECT_ROOT,
        env_file_path=ENV_FILE_PATH,
        data_dir=data_dir,
        retrieval_model=_str_env("TUTOR_RETRIEVAL_MODEL", "gpt-4.1-mini"),
        generation_model=_str_env("TUTOR_GENERATION_MODEL", "gpt-4.1-mini"),
        doc_results_per_objective=_int_env("DOC_RESULTS_PER_OBJECTIVE", 2),
        web_results_per_objective=_int_env("WEB_RESULTS_PER_OBJECTIVE", 2),
        max_evidence_items=_int_env("MAX_EVIDENCE_ITEMS", 10),
        fast_section_limit=_int_env("FAST_SECTION_LIMIT", 3),
        fast_doc_results_per_objective=_int_env("FAST_DOC_RESULTS_PER_OBJECTIVE", 1),
        fast_web_results_per_objective=_int_env("FAST_WEB_RESULTS_PER_OBJECTIVE", 1),
        fast_max_evidence_items=_int_env("FAST_MAX_EVIDENCE_ITEMS", 5),
        openai_timeout_seconds=_float_env("OPENAI_TIMEOUT_SECONDS", 25.0),
        retrieval_timeout_seconds=_float_env("OPENAI_RETRIEVAL_TIMEOUT_SECONDS", 18.0),
        generation_timeout_seconds=_float_env("OPENAI_GENERATION_TIMEOUT_SECONDS", 28.0),
        upload_timeout_seconds=_float_env("OPENAI_UPLOAD_TIMEOUT_SECONDS", 30.0),
        vector_poll_attempts=_int_env("VECTOR_POLL_ATTEMPTS", 25),
        vector_poll_sleep_seconds=_float_env("VECTOR_POLL_SLEEP_SECONDS", 0.4),
        log_level=_str_env("TUTOR_LOG_LEVEL", "INFO").upper(),
    )


settings = get_settings()


def get_openai_api_key() -> str:
    _load_dotenv_if_present()
    return os.getenv("OPENAI_API_KEY", "").strip()


def ensure_openai_api_key() -> str:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    return api_key
