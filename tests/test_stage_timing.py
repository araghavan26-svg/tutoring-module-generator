from __future__ import annotations

import logging
import sys
import time
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.models import ModuleGenerateRequest
from app.openai_service import retrieval_plan_for_request, selected_learning_objectives
from app.stage_timing import StageTimingLogger


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class StageTimingTests(unittest.TestCase):
    def test_stage_timing_logger_records_durations_and_logs(self) -> None:
        logger = logging.getLogger("stage-timing-test")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = _ListHandler()
        logger.handlers = [handler]

        timer = StageTimingLogger(logger=logger, request_id="req-123")
        timer.start("evidence_retrieval", objective_count=3)
        time.sleep(0.001)
        duration = timer.end("evidence_retrieval", objective_count=3)
        total_duration = timer.finish(topic="Photosynthesis")

        self.assertGreater(duration, 0.0)
        self.assertGreater(total_duration, 0.0)
        self.assertIn("evidence_retrieval", timer.stage_durations)
        self.assertTrue(any("evidence_retrieval.start" in message for message in handler.messages))
        self.assertTrue(any("evidence_retrieval.end" in message for message in handler.messages))
        self.assertTrue(any("generation.total" in message for message in handler.messages))

    def test_fast_mode_limits_objectives_and_retrieval_fanout(self) -> None:
        request = ModuleGenerateRequest(
            topic="Photosynthesis",
            audience_level="Middle school",
            learning_objectives=[
                "Explain the process",
                "Identify the inputs",
                "Identify the outputs",
                "Connect the process to plant growth",
            ],
            allow_web=True,
            fast_mode=True,
        )

        goals = selected_learning_objectives(request)
        plan = retrieval_plan_for_request(request)

        self.assertEqual(len(goals), min(3, len(request.learning_objectives)))
        self.assertEqual(plan.doc_results_per_objective, 1)
        self.assertEqual(plan.web_results_per_objective, 1)
        self.assertEqual(plan.max_evidence_items, 5)


if __name__ == "__main__":
    unittest.main()
