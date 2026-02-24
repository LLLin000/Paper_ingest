# pyright: reportUnusedCallResult=false

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _create_strict_required_layout(run_dir: Path, doc_id: str) -> None:
    for dirname in (
        "pages",
        "text",
        "vision",
        "figures_tables",
        "paragraphs",
        "citations",
        "reading",
        "obsidian",
        "qa",
    ):
        (run_dir / dirname).mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "qa" / "report.json",
        {
            "doc_id": doc_id,
            "overall_status": "pass",
            "hard_stop": False,
            "gates": {},
        },
    )
    _write_json(
        run_dir / "qa" / "stage_status.json",
        {"stage": "verify", "status": "pass", "hard_stop": False, "reason": "ok"},
    )
    _write_json(
        run_dir / "qa" / "runtime_safety.json",
        {"network_deny_mode": True, "egress_attempt_targets": []},
    )


def test_validate_strict_reports_schema_error_with_artifact_and_schema_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "bad_doc"
    schemas_dir = tmp_path / "schemas"
    _create_strict_required_layout(run_dir, doc_id="bad_doc")
    _ = (run_dir / "obsidian" / "bad_doc.md").write_text("# note\n", encoding="utf-8")

    _write_json(schemas_dir / "manifest.schema.json", {"type": "object", "required": ["doc_id"]})
    _write_json(run_dir / "manifest.json", {"pipeline_version": "0.1.0"})

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingest.validate",
            "--run",
            str(run_dir),
            "--schemas",
            str(schemas_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "Validation failed:" in output
    assert "manifest.json" in output
    assert "manifest.schema.json" in output
    assert "required property" in output


def test_validate_non_placeholder_execution_on_existing_run() -> None:
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "run" / "struct07full"
    schemas_dir = root / "schemas"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingest.validate",
            "--run",
            str(run_dir),
            "--schemas",
            str(schemas_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "Validation passed for run:" in output
    assert "Bootstrap placeholder" not in output


def test_validate_strict_fails_on_missing_required_artifact_with_actionable_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "missing_doc"
    schemas_dir = tmp_path / "schemas"

    _create_strict_required_layout(run_dir, doc_id="missing_doc")
    _write_json(schemas_dir / "manifest.schema.json", {"type": "object", "required": ["doc_id"]})
    _write_json(run_dir / "manifest.json", {"doc_id": "missing_doc"})

    note_path = run_dir / "obsidian" / "missing_doc.md"
    if note_path.exists():
        note_path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingest.validate",
            "--run",
            str(run_dir),
            "--schemas",
            str(schemas_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    output = result.stdout + result.stderr
    assert "Validation failed:" in output
    assert "missing required artifact for strict validation" in output
    assert "obsidian" in output
    assert "missing_doc.md" in output


def test_validate_strict_reports_actionable_path_for_qa_report_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "bad_report"
    schemas_dir = tmp_path / "schemas"
    _create_strict_required_layout(run_dir, doc_id="bad_report")
    _ = (run_dir / "obsidian" / "bad_report.md").write_text("# note\n", encoding="utf-8")

    _write_json(schemas_dir / "manifest.schema.json", {"type": "object", "required": ["doc_id"]})
    _write_json(run_dir / "manifest.json", {"doc_id": "bad_report"})
    _write_json(
        run_dir / "qa" / "report.json",
        {
            "doc_id": "bad_report",
            "overall_status": "pass",
            "hard_stop": False,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingest.validate",
            "--run",
            str(run_dir),
            "--schemas",
            str(schemas_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "report.json" in output
    assert "$.gates" in output
    assert "missing required key" in output


def test_validate_strict_reports_actionable_path_for_malformed_vision_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "bad_vision"
    schemas_dir = tmp_path / "schemas"
    _create_strict_required_layout(run_dir, doc_id="bad_vision")
    _ = (run_dir / "obsidian" / "bad_vision.md").write_text("# note\n", encoding="utf-8")

    _write_json(schemas_dir / "manifest.schema.json", {"type": "object", "required": ["doc_id"]})
    _write_json(run_dir / "manifest.json", {"doc_id": "bad_vision"})
    _write_json(
        run_dir / "vision" / "p001_out.json",
        {
            "page": 1,
            "reading_order": [],
            "merge_groups": [],
            "role_labels": {},
            "confidence": 0.1,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingest.validate",
            "--run",
            str(run_dir),
            "--schemas",
            str(schemas_dir),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "p001_out.json" in output
    assert "$.fallback_used" in output
    assert "missing required key" in output
