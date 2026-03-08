from ingest.reading import apply_pipeline_fallback_degradation


def test_apply_pipeline_fallback_degradation_marks_degraded_when_fallback_used() -> None:
    summary_status = {
        "doc_id": "demo",
        "status": "full",
        "reason_codes": [],
        "metrics": {
            "narrative_fact_count": 12,
            "themes_count": 3,
            "synthesis_present": True,
            "narrative_evidence_ratio": 1.0,
        },
    }

    updated = apply_pipeline_fallback_degradation(
        summary_status,
        reading_fallback_count=2,
        vision_fallback_pages=5,
    )

    assert updated["status"] == "degraded"
    assert "reading_model_fallback_used" in updated["reason_codes"]
    assert "vision_fallback_used" in updated["reason_codes"]


def test_apply_pipeline_fallback_degradation_preserves_full_when_no_fallback() -> None:
    summary_status = {
        "doc_id": "demo",
        "status": "full",
        "reason_codes": [],
        "metrics": {
            "narrative_fact_count": 12,
            "themes_count": 3,
            "synthesis_present": True,
            "narrative_evidence_ratio": 1.0,
        },
    }

    updated = apply_pipeline_fallback_degradation(
        summary_status,
        reading_fallback_count=0,
        vision_fallback_pages=0,
    )

    assert updated["status"] == "full"
    assert updated["reason_codes"] == []
