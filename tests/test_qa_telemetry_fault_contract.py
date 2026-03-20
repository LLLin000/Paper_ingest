import json
from pathlib import Path

from ingest.qa_telemetry import append_fault_events


def test_append_fault_events_enriches_stage_fault_contract(tmp_path: Path) -> None:
    qa_dir = tmp_path / "qa"
    append_fault_events(
        qa_dir,
        [
            {
                "stage": "vision",
                "fault": "parse-failure",
                "retry_attempts": 1,
                "fallback_used": True,
                "status": "degraded",
            }
        ],
    )

    payload = json.loads((qa_dir / "fault_injection.json").read_text(encoding="utf-8"))
    assert len(payload) == 1

    event = payload[0]
    assert event["stage"] == "vision"
    assert event["reason_category"] == event["category"]
    assert event["retryable"] is False
    assert event["degraded_output_used"] is True
    assert event["upstream_artifacts"] == ["text/blocks_norm.jsonl", "pages/p*.png"]
