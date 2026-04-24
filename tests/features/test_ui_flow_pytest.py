from __future__ import annotations


def test_create_page_shows_combined_input_and_local_progress(client):
    disclaimer = client.get("/")
    app_home = client.get("/app")
    response = client.get("/create?sample=1")

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
