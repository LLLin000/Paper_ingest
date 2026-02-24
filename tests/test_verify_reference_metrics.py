# pyright: reportMissingTypeStubs=false
import json
from pathlib import Path
from typing import Any

import pytest

from ingest import verify as verify_module


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def test_reference_quality_metrics_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = "ref_metrics"
    run_root = tmp_path / "run"
    doc_dir = run_root / doc_id

    monkeypatch.setattr(verify_module, "RUN_ROOT", run_root)

    # Create synthetic artifacts
    api_refs = [
        {"title": "A", "doi": "10.1/abc", "source": "crossref", "confidence": 0.9},
        {"title": "B", "source": "semantic_scholar", "confidence": 0.8},
    ]
    merged_refs = [
        {"title": "A", "doi": "10.1/abc", "sources": [{"provider": "crossref", "kind": "api"}]},
        {"title": "B", "sources": [{"provider": "pdf", "kind": "pdf"}]},
    ]
    pdf_catalog = [
        {"title": "B", "raw_text": "B ref"}
    ]

    _write_jsonl(doc_dir / "refs" / "references_api.jsonl", api_refs)
    _write_jsonl(doc_dir / "refs" / "references_merged.jsonl", merged_refs)
    _write_jsonl(doc_dir / "citations" / "reference_catalog.jsonl", pdf_catalog)

    # Run verification
    rc = verify_module.verify(doc_id)
    assert rc == 0

    report_path = doc_dir / "qa" / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "reference" in report["gates"]
    ref_gate = report["gates"]["reference"]
    details = ref_gate.get("details", {})

    assert details["api_reference_count"] == 2
    assert details["merged_reference_count"] == 2
    assert details["pdf_reference_count"] == 1
    assert "dedupe_rate" in details
    assert "identifier_completeness" in details


def test_reference_quality_missing_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = "ref_missing"
    run_root = tmp_path / "run"
    doc_dir = run_root / doc_id

    monkeypatch.setattr(verify_module, "RUN_ROOT", run_root)

    # No reference artifacts created
    # Create run directory so verify can run even when artifacts missing
    doc_dir.mkdir(parents=True, exist_ok=True)

    rc = verify_module.verify(doc_id)
    assert rc == 0

    report_path = doc_dir / "qa" / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "reference" in report["gates"]
    ref_gate = report["gates"]["reference"]
    assert ref_gate["status"] in {"not_evaluated", "degraded"}
