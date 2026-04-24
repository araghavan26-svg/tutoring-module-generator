from __future__ import annotations

from unittest.mock import patch

from app.models import ModuleAskResponse, ModuleAssignmentResponse, ModuleGradeBreakdownItem, ModuleGradeResponse, RubricCriterion, RubricLevel
from app.openai_service import EvidenceBuildResult
from app.store import module_store


def test_generate_endpoint_with_combined_learning_request_passes_source_preferences(client, sample_evidence, sample_module):
    captured = {}
    evidence_result = EvidenceBuildResult(
        evidence_pack=sample_evidence,
        web_unavailable_objectives=[],
        objectives_without_evidence=[],
    )

    def fake_generate_module_from_evidence(*args, **kwargs):
        captured["request"] = kwargs.get("request")
        return sample_module.model_copy(update={"module_id": "module-new"})

    with patch("app.main.get_openai_client", return_value=object()), patch(
        "app.main.build_evidence_pack",
        return_value=evidence_result,
    ), patch(
        "app.main.generate_module_from_evidence",
        side_effect=fake_generate_module_from_evidence,
    ), patch(
        "app.main.enforce_quality_gate",
        side_effect=lambda module, evidence_pack: module,
    ):
        response = client.post(
            "/v1/modules/generate",
            json={
                "learning_request": "Biology - Photosynthesis",
                "audience_level": "Advanced / AP",
                "learner_level": "Advanced / AP",
                "current_familiarity": "I know the basics",
                "learning_purpose": "Review for a test",
                "explanation_style": "Concise review",
                "source_preference": "Academic / Educational",
                "prefer_high_trust_sources": True,
                "learning_objectives": ["Explain the process", "Use the key vocabulary"],
                "allow_web": True,
            },
        )

    assert response.status_code == 200
    request = captured["request"]
    assert request.subject == "Biology"
    assert request.topic == "Photosynthesis"
    assert request.source_preference == "Academic / Educational"
    assert request.prefer_high_trust_sources is True


def test_tutor_endpoint_uses_service_layer_with_grounded_answer(client, sample_request, sample_module, sample_evidence):
    module_store.save("module-tutor-pytest", sample_request, sample_module, sample_evidence, action="generated")

    with patch("app.main.get_openai_client", return_value=object()), patch(
        "app.main.answer_question_from_module",
        return_value=ModuleAskResponse(answer="Plants use sunlight to make sugar.", citations=["E001"], unverified=False),
    ):
        response = client.post(
            "/v1/modules/module-tutor-pytest/ask",
            json={"question": "What is the most important idea here?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["unverified"] is False
    assert payload["citations"] == ["E001"]


def test_assignment_and_grading_endpoints_use_service_layer(client, sample_request, sample_module, sample_evidence):
    module_store.save("module-assess-pytest", sample_request, sample_module, sample_evidence, action="generated")
    assignment = ModuleAssignmentResponse(
        prompt="Explain photosynthesis and give one example.",
        rubric=[
            RubricCriterion(
                criteria="Use the core process accurately.",
                levels=[
                    RubricLevel(score=4, description="Strong"),
                    RubricLevel(score=3, description="Good"),
                    RubricLevel(score=2, description="Partial"),
                    RubricLevel(score=1, description="Weak"),
                ],
            )
        ],
    )
    grade = ModuleGradeResponse(
        score=85,
        feedback="Mostly accurate and grounded.",
        breakdown=[
            ModuleGradeBreakdownItem(
                criteria="Use the core process accurately.",
                score=3,
                max_score=4,
                feedback="One minor omission.",
            )
        ],
        unverified=False,
    )

    with patch("app.main.get_openai_client", return_value=object()), patch(
        "app.main.generate_assignment_from_module",
        return_value=assignment,
    ), patch(
        "app.main.grade_assignment_from_module",
        return_value=grade,
    ):
        assignment_response = client.post("/v1/modules/module-assess-pytest/assignment")
        grade_response = client.post(
            "/v1/modules/module-assess-pytest/grade",
            json={"student_response": "Plants use sunlight.", "rubric": assignment.model_dump(mode="json")["rubric"]},
        )

    assert assignment_response.status_code == 200
    assert grade_response.status_code == 200
    assert grade_response.json()["score"] == 85
