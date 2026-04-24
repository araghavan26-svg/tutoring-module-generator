from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import ensure_openai_api_key, settings
from .errors import NotFoundError, register_exception_handlers
from .logging_utils import configure_logging, get_logger
from .models import (
    DocsUploadResponse,
    Module,
    ModuleAssignmentResponse,
    ModuleAskRequest,
    ModuleAskResponse,
    ModuleDeleteResponse,
    ModuleGenerateRequest,
    ModuleGenerateResponse,
    ModuleGradeRequest,
    ModuleGradeResponse,
    ModuleShareRequest,
    ModuleShareResponse,
    ModuleVersionSummary,
    RefreshSourcesRequest,
    RefreshSourcesResponse,
    SectionRegenerateByIdRequest,
    SectionRegenerateRequest,
    SectionRegenerateResponse,
)
from .openai_client import get_openai_client
from .openai_service import (
    answer_question_from_module,
    build_evidence_pack,
    build_topic_bridge_context,
    build_unverified_module,
    detect_topic_relatedness,
    enforce_quality_gate,
    generate_assignment_from_module,
    generate_module_from_evidence,
    generate_section_from_evidence,
    grade_assignment_from_module,
)
from .services.docs_service import upload_documents
from .services.export_service import module_export_markdown as _module_export_markdown
from .services.module_service import (
    ask_module_question_response,
    create_module_assignment_response,
    delete_module_response,
    export_assignment_json_payload,
    export_assignment_markdown_text,
    export_module_json_payload,
    export_module_markdown_text,
    generate_module_response,
    grade_assignment_response,
    module_history,
    refresh_sources_response,
    regenerate_section_by_id_response,
    regenerate_section_response,
    revert_module_version,
    toggle_module_share_response,
)
from .services.view_service import create_page_context, module_editor_context, modules_dashboard_context, shared_module_context
from .store import module_store


configure_logging()
logger = get_logger("tutoring_module_api")

app = FastAPI(
    title="Grounded Tutoring Module API",
    version="1.0.0",
    description=(
        "Two-step tutoring-module generator: (1) retrieve evidence with tools, "
        "(2) generate module from evidence with tools disabled."
    ),
)
register_exception_handlers(app, logger=logger)

PROJECT_ROOT = settings.project_root
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
SAMPLES_DIR = PROJECT_ROOT / "samples"
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.on_event("startup")
def enforce_byok_startup_check() -> None:
    ensure_openai_api_key()
    module_store.load_from_disk()


def _generate_module_response(payload: ModuleGenerateRequest) -> ModuleGenerateResponse:
    return generate_module_response(
        payload,
        store=module_store,
        client_provider=get_openai_client,
        evidence_builder=build_evidence_pack,
        module_generator=generate_module_from_evidence,
        quality_gate=enforce_quality_gate,
        relatedness_detector=detect_topic_relatedness,
        topic_bridge_builder=build_topic_bridge_context,
        unverified_module_builder=build_unverified_module,
    )


def _simple_create_context(*, error: str | None = None, form: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "title": "Create a Module",
        "error": error,
        "source_note": (
            "No documents uploaded. The module will use web search if enabled, "
            "otherwise it may mark sections as unverified."
        ),
        "form": form
        or {
            "topic": "",
            "audience_level": "Middle school",
            "learning_objectives": "",
            "allow_web": True,
        },
    }


@app.get("/")
def simple_module_page(request: Request) -> Response:
    return templates.TemplateResponse(request, "simple_create.html", _simple_create_context())


@app.get("/disclaimer")
def disclaimer_page(request: Request) -> Response:
    return templates.TemplateResponse(request, "disclaimer.html", {"title": "Before You Use This System"})


@app.get("/app")
def app_homepage(request: Request) -> Response:
    return templates.TemplateResponse(request, "index.html", {"title": "ModuleForge"})


@app.get("/create")
def create_page(request: Request) -> Response:
    return templates.TemplateResponse(request, "create.html", create_page_context(store=module_store))


@app.get("/modules")
def modules_dashboard_page(request: Request) -> Response:
    return templates.TemplateResponse(request, "modules_dashboard.html", modules_dashboard_context(request=request, store=module_store))


@app.get("/modules/{module_id}")
def module_editor_page(request: Request, module_id: str) -> Response:
    return templates.TemplateResponse(
        request,
        "module_editor.html",
        module_editor_context(request=request, module_id=module_id, store=module_store),
    )


@app.get("/shared/{share_id}")
def shared_module_page(request: Request, share_id: str) -> Response:
    try:
        context = shared_module_context(request=request, share_id=share_id, store=module_store)
    except NotFoundError:
        return templates.TemplateResponse(
            request,
            "shared_unavailable.html",
            {"title": "Shared Module Unavailable"},
            status_code=404,
        )
    return templates.TemplateResponse(request, "shared_module.html", context)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "OK"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/docs/upload", response_model=DocsUploadResponse)
async def upload_docs(files: List[UploadFile] = File(...)) -> DocsUploadResponse:
    return await upload_documents(files, client_provider=get_openai_client)


@app.post("/ui/modules/generate")
def simple_generate_module_page(
    request: Request,
    topic: str = Form(...),
    audience_level: str = Form(...),
    learning_objectives: str = Form(...),
    allow_web: bool = Form(False),
) -> Response:
    form = {
        "topic": topic,
        "audience_level": audience_level,
        "learning_objectives": learning_objectives,
        "allow_web": allow_web,
    }
    objectives = [line.strip() for line in learning_objectives.splitlines() if line.strip()]
    if not topic.strip():
        return templates.TemplateResponse(
            request,
            "simple_create.html",
            _simple_create_context(error="Please enter a topic.", form=form),
            status_code=400,
        )
    if not objectives:
        return templates.TemplateResponse(
            request,
            "simple_create.html",
            _simple_create_context(error="Please add at least one learning objective.", form=form),
            status_code=400,
        )

    try:
        response = _generate_module_response(
            ModuleGenerateRequest(
                topic=topic,
                audience_level=audience_level,
                learning_objectives=objectives,
                allow_web=allow_web,
                fast_mode=True,
            )
        )
    except Exception:
        logger.exception("Simple UI module generation failed")
        return templates.TemplateResponse(
            request,
            "simple_create.html",
            _simple_create_context(
                error=(
                    "We could not generate the module just now. One goal is okay, and no document upload is required. "
                    "Try again, or turn on web search if you want the app to look for sources automatically."
                ),
                form=form,
            ),
            status_code=500,
        )

    evidence_by_id = {item.evidence_id: item for item in response.evidence_pack}
    return templates.TemplateResponse(
        request,
        "simple_results.html",
        {
            "title": response.module.title,
            "module": response.module,
            "evidence_by_id": evidence_by_id,
        },
    )


@app.post("/v1/modules/generate", response_model=ModuleGenerateResponse)
def generate_module(payload: ModuleGenerateRequest) -> ModuleGenerateResponse:
    return _generate_module_response(payload)


@app.post("/v1/modules/{module_id}/regenerate", response_model=SectionRegenerateResponse)
def regenerate_section(module_id: str, payload: SectionRegenerateRequest) -> SectionRegenerateResponse:
    return regenerate_section_response(
        module_id,
        payload,
        store=module_store,
        client_provider=get_openai_client,
        evidence_builder=build_evidence_pack,
        section_generator=generate_section_from_evidence,
        quality_gate=enforce_quality_gate,
    )


@app.post("/v1/modules/{module_id}/refresh_sources", response_model=RefreshSourcesResponse)
def refresh_sources(module_id: str, payload: RefreshSourcesRequest) -> RefreshSourcesResponse:
    return refresh_sources_response(module_id, payload, store=module_store, client_provider=get_openai_client)


@app.post("/v1/modules/{module_id}/sections/{section_id}/regenerate", response_model=SectionRegenerateResponse)
def regenerate_section_by_id(
    module_id: str,
    section_id: str,
    payload: SectionRegenerateByIdRequest,
) -> SectionRegenerateResponse:
    return regenerate_section_by_id_response(
        module_id,
        section_id,
        payload,
        store=module_store,
        client_provider=get_openai_client,
        section_generator=generate_section_from_evidence,
    )


@app.get("/v1/modules/{module_id}/export/json")
def export_module_json(module_id: str) -> Dict[str, Any]:
    return export_module_json_payload(module_id, store=module_store)


@app.get("/v1/modules/{module_id}/export/markdown")
def export_module_markdown(module_id: str) -> Response:
    return Response(content=export_module_markdown_text(module_id, store=module_store), media_type="text/markdown")


@app.get("/v1/modules/{module_id}/history", response_model=list[ModuleVersionSummary])
def get_module_history(module_id: str) -> list[ModuleVersionSummary]:
    return module_history(module_id, store=module_store)


@app.post("/v1/modules/{module_id}/revert/{version_id}", response_model=Module)
def revert_module(module_id: str, version_id: str) -> Module:
    return revert_module_version(module_id, version_id, store=module_store)


@app.post("/v1/modules/{module_id}/ask", response_model=ModuleAskResponse)
def ask_module_question(module_id: str, payload: ModuleAskRequest) -> ModuleAskResponse:
    return ask_module_question_response(
        module_id,
        payload,
        store=module_store,
        client_provider=get_openai_client,
        answer_generator=answer_question_from_module,
    )


@app.post("/v1/modules/{module_id}/assignment", response_model=ModuleAssignmentResponse)
def create_module_assignment(module_id: str) -> ModuleAssignmentResponse:
    return create_module_assignment_response(
        module_id,
        store=module_store,
        client_provider=get_openai_client,
        assignment_generator=generate_assignment_from_module,
    )


@app.post("/v1/modules/{module_id}/assignment/export/json", response_model=ModuleAssignmentResponse)
def export_assignment_json(module_id: str, payload: ModuleAssignmentResponse) -> ModuleAssignmentResponse:
    return export_assignment_json_payload(module_id, payload, store=module_store)


@app.post("/v1/modules/{module_id}/assignment/export/markdown")
def export_assignment_markdown(module_id: str, payload: ModuleAssignmentResponse) -> Response:
    return Response(content=export_assignment_markdown_text(module_id, payload, store=module_store), media_type="text/markdown")


@app.post("/v1/modules/{module_id}/grade", response_model=ModuleGradeResponse)
def grade_module_assignment(module_id: str, payload: ModuleGradeRequest) -> ModuleGradeResponse:
    return grade_assignment_response(
        module_id,
        payload,
        store=module_store,
        client_provider=get_openai_client,
        grading_generator=grade_assignment_from_module,
    )


@app.post("/v1/modules/{module_id}/share", response_model=ModuleShareResponse)
def toggle_module_share(request: Request, module_id: str, payload: ModuleShareRequest) -> ModuleShareResponse:
    return toggle_module_share_response(request, module_id, enabled=payload.enabled, store=module_store)


@app.post("/v1/modules/{module_id}/delete", response_model=ModuleDeleteResponse)
def delete_module(module_id: str) -> ModuleDeleteResponse:
    return delete_module_response(module_id, store=module_store)
