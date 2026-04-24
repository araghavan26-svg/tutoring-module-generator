from __future__ import annotations

from unittest.mock import patch

from app.models import ModuleGenerateResponse


def test_create_page_shows_combined_input_and_local_progress(client):
    quick_create = client.get("/")
    disclaimer = client.get("/disclaimer")
    app_home = client.get("/app")
    response = client.get("/create?sample=1")

    assert quick_create.status_code == 200
    assert "Create a tutoring module" in quick_create.text
    assert 'name="topic"' in quick_create.text
    assert 'name="audience_level"' in quick_create.text
    assert 'name="learning_objectives"' in quick_create.text
    assert 'name="allow_web"' in quick_create.text
    assert 'action="/ui/modules/generate"' in quick_create.text
    assert "No documents uploaded. The module will use web search if enabled" in quick_create.text
    assert "One objective is enough to generate a focused module." in quick_create.text
    assert "fewer goals" not in quick_create.text
    assert "fewer sources" not in quick_create.text
    assert disclaimer.status_code == 200
    assert "Before You Use This System" in disclaimer.text
    assert "I Understand — Continue" in disclaimer.text
    assert 'action="/app"' in disclaimer.text
    assert app_home.status_code == 200
    assert "What this app does" in app_home.text
    assert response.status_code == 200
    assert 'id="learning_request"' in response.text
    assert 'id="source_preference"' in response.text
    assert 'id="preset-learn-scratch"' in response.text
    assert response.text.index('id="create-submit"') < response.text.index('id="create-progress"')
    assert "This will create a grounded lesson, vocabulary, practice questions, and optional assessment tools." in response.text


def test_simple_form_submission_renders_results_page(client, sample_module, sample_evidence):
    with patch(
        "app.main._generate_module_response",
        return_value=ModuleGenerateResponse(module=sample_module, evidence_pack=sample_evidence),
    ) as generate_response:
        response = client.post(
            "/ui/modules/generate",
            data={
                "topic": "Photosynthesis",
                "audience_level": "Middle school",
                "learning_objectives": "Explain the core idea\nUse one simple example",
                "allow_web": "true",
            },
        )

    assert response.status_code == 200
    assert "Generated module" in response.text
    assert "Start here" in response.text
    assert "Grounded explanation for Photosynthesis." in response.text
    assert "kids.britannica.com" in response.text
    assert "Open source" in response.text

    request = generate_response.call_args.args[0]
    assert request.topic == "Photosynthesis"
    assert request.audience_level == "Middle school"
    assert request.learning_objectives == ["Explain the core idea", "Use one simple example"]
    assert request.allow_web is True


def test_simple_form_accepts_one_objective_with_no_uploaded_documents(client, sample_module, sample_evidence):
    with patch(
        "app.main._generate_module_response",
        return_value=ModuleGenerateResponse(module=sample_module, evidence_pack=sample_evidence),
    ) as generate_response:
        response = client.post(
            "/ui/modules/generate",
            data={
                "topic": "Photosynthesis",
                "audience_level": "Middle school",
                "learning_objectives": "Explain the core idea",
            },
        )

    assert response.status_code == 200
    assert "Generated module" in response.text
    assert "fewer goals" not in response.text
    assert "fewer sources" not in response.text

    request = generate_response.call_args.args[0]
    assert request.learning_objectives == ["Explain the core idea"]
    assert request.allow_web is False


def test_simple_form_generation_error_does_not_blame_single_goal_or_missing_uploads(client):
    with patch("app.main._generate_module_response", side_effect=RuntimeError("upstream failed")):
        response = client.post(
            "/ui/modules/generate",
            data={
                "topic": "Photosynthesis",
                "audience_level": "Middle school",
                "learning_objectives": "Explain the core idea",
            },
        )

    assert response.status_code == 500
    assert "One goal is okay" in response.text
    assert "no document upload is required" in response.text
    assert "fewer goals" not in response.text
    assert "fewer sources" not in response.text


def test_dashboard_and_shared_routes_use_feature_contexts(client, sample_request, sample_module, sample_evidence):
    from app.store import module_store

    module_store.save("module-ui-pytest", sample_request, sample_module, sample_evidence, action="generated")
    shared_module = module_store.set_share_enabled("module-ui-pytest", True)

    dashboard = client.get("/modules")
    shared = client.get(f"/shared/{shared_module.share_id}")

    assert dashboard.status_code == 200
    assert "Saved Modules" in dashboard.text
    assert shared.status_code == 200
    assert "Photosynthesis module" in shared.text
