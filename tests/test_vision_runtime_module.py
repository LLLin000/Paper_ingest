from pathlib import Path

from ingest.qa_telemetry import _read_jsonl
from ingest.vision_faults import FaultEvent, VisionLLMCallEvent
from ingest.vision_runtime import (
    VisionImageDataUrlCache,
    VisionRequestBudget,
    VisionRuntimeContext,
    append_vision_llm_call_event,
    append_vision_runtime_event,
)


def test_vision_runtime_module_emits_runtime_and_llm_events(tmp_path: Path) -> None:
    qa_dir = tmp_path / "qa"
    budget = VisionRequestBudget(limit=3)
    accepted, _, _, _ = budget.try_consume()
    assert accepted is True

    cache = VisionImageDataUrlCache()
    runtime = VisionRuntimeContext(budget=budget, image_cache=cache)

    append_vision_llm_call_event(
        qa_dir,
        VisionLLMCallEvent(
            stage="vision",
            page=1,
            attempt=1,
            model="fake",
            endpoint="https://example.test/v1/chat/completions",
            success=True,
            parse_success=True,
            validation_success=True,
            error_type="none",
            http_status=200,
            prompt_chars=10,
            response_chars=10,
            response_preview="{}",
            timestamp="2026-03-18T00:00:00+00:00",
        ),
    )
    append_vision_runtime_event(qa_dir, runtime, pages_done=1, total_pages=2)

    llm_events = _read_jsonl(qa_dir / "vision_llm_calls.jsonl")
    runtime_events = _read_jsonl(qa_dir / "vision_runtime.jsonl")
    fault_summary = (qa_dir / "fault_summary.json").read_text(encoding="utf-8")

    assert llm_events[-1]["model"] == "fake"
    assert runtime_events[-1]["budget_consumed"] == 1
    assert "window_event_count" in fault_summary


def test_fault_event_dataclass_shape() -> None:
    event = FaultEvent(
        stage="vision",
        fault="parse-failure",
        page=1,
        retry_attempts=0,
        fallback_used=True,
        status="degraded",
    )

    assert event.stage == "vision"
