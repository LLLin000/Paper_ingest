# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingest import cli  # noqa: E402
from ingest import verify as verify_module  # noqa: E402


def test_verify_stage_smoke_creates_qa_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = "smoke_verify"
    run_root = tmp_path / "run"
    input_pdf = tmp_path / "smoke.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n% smoke test fixture\n")
    doc_dir = run_root / doc_id

    monkeypatch.setattr(cli, "RUN_ROOT", run_root)
    monkeypatch.setattr(verify_module, "RUN_ROOT", run_root)
    monkeypatch.setattr(verify_module, "EVAL_ROOT", tmp_path / "eval" / "golden")

    result = CliRunner().invoke(
        cli.app,
        ["--pdf", str(input_pdf), "--doc_id", doc_id, "--stage", "verify"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Verifying run" in result.stdout
    assert "QA Report:" in result.stdout

    report_path = doc_dir / "qa" / "report.json"
    stage_status_path = doc_dir / "qa" / "stage_status.json"
    runtime_safety_path = doc_dir / "qa" / "runtime_safety.json"

    assert report_path.exists()
    assert stage_status_path.exists()
    assert runtime_safety_path.exists()

    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["doc_id"] == doc_id
    assert report["overall_status"] in {"pass", "degraded", "fail", "not_evaluated"}
    assert isinstance(report.get("gates"), dict)


def test_cli_requires_pdf_or_doc_id() -> None:
    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 1
