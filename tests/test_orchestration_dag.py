import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ingest import cli
from ingest.manifest import Manifest
from ingest.orchestration import execute_levelized_dag


def _make_manifest(doc_id: str = "dag_task5_test") -> Manifest:
    return Manifest(
        doc_id=doc_id,
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="b" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def test_execute_levelized_dag_respects_dependencies_and_parallel_levels() -> None:
    events: list[tuple[str, str, float]] = []
    lock = threading.Lock()

    def make_job(name: str, delay: float):
        def _job() -> str:
            with lock:
                events.append(("start", name, time.perf_counter()))
            time.sleep(delay)
            with lock:
                events.append(("end", name, time.perf_counter()))
            return name

        return _job

    jobs = {
        "extractor": make_job("extractor", 0.02),
        "overlay": make_job("overlay", 0.08),
        "vision": make_job("vision", 0.03),
        "paragraphs": make_job("paragraphs", 0.02),
        "citations": make_job("citations", 0.04),
        "figures_tables": make_job("figures_tables", 0.08),
        "reading": make_job("reading", 0.02),
        "render": make_job("render", 0.01),
    }

    results = execute_levelized_dag(jobs, max_workers=2)

    def t(kind: str, stage: str) -> float:
        for event_kind, event_stage, ts in events:
            if event_kind == kind and event_stage == stage:
                return ts
        raise AssertionError(f"Missing {kind} event for {stage}")

    assert list(results.keys()) == [
        "extractor",
        "overlay",
        "vision",
        "paragraphs",
        "citations",
        "figures_tables",
        "reading",
        "render",
    ]

    assert t("start", "overlay") >= t("end", "extractor")
    assert t("start", "vision") >= t("end", "extractor")
    assert t("start", "paragraphs") >= t("end", "vision")
    assert t("start", "citations") >= t("end", "paragraphs")
    assert t("start", "figures_tables") >= t("end", "paragraphs")
    assert t("start", "reading") >= max(t("end", "citations"), t("end", "figures_tables"))
    assert t("start", "render") >= t("end", "reading")

    assert t("start", "overlay") < t("end", "vision")
    assert t("start", "vision") < t("end", "overlay")


def test_run_full_emits_deterministic_stage_output_order(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    run_root = tmp_path / "run"
    monkeypatch.setattr(cli, "RUN_ROOT", run_root)

    def _extractor(*, run_dir: Path, manifest: Manifest, inject_missing_font_stats: bool):
        _ = (run_dir, manifest, inject_missing_font_stats)
        time.sleep(0.02)
        return 2, 20

    def _overlay(*, run_dir: Path, manifest: Manifest):
        _ = (run_dir, manifest)
        time.sleep(0.08)
        return 2, 18

    def _vision(*, run_dir: Path, manifest: Manifest, inject_malformed_json: bool):
        _ = (run_dir, manifest, inject_malformed_json)
        time.sleep(0.01)
        return 2, 20

    def _paragraphs(*, run_dir: Path, manifest: Manifest):
        _ = (run_dir, manifest)
        return 8, 20

    def _citations(*, run_dir: Path, manifest: Manifest):
        _ = (run_dir, manifest)
        return 6, 6

    def _figures_tables(*, run_dir: Path, manifest: Manifest):
        _ = (run_dir, manifest)
        time.sleep(0.06)
        return 3, 3

    def _reading(*, run_dir: Path, manifest: Manifest, inject_malformed_json: bool):
        _ = (run_dir, manifest, inject_malformed_json)
        return 4, 2, 5, 3, 0

    def _render(*, run_dir: Path, manifest: Manifest):
        _ = (run_dir, manifest)
        return 7, run_dir / "obsidian" / "summary.md"

    monkeypatch.setattr(cli, "run_extractor", _extractor)
    monkeypatch.setattr(cli, "run_overlay", _overlay)
    monkeypatch.setattr(cli, "run_vision", _vision)
    monkeypatch.setattr(cli, "run_paragraphs", _paragraphs)
    monkeypatch.setattr(cli, "run_citations", _citations)
    monkeypatch.setattr(cli, "run_figures_tables", _figures_tables)
    monkeypatch.setattr(cli, "run_reading", _reading)
    monkeypatch.setattr(cli, "run_render", _render)
    monkeypatch.setattr(cli, "run_verify", lambda doc_id: 0)

    cli._run_full(
        manifest=_make_manifest("dag_order_test"),
        inject_vision_malformed_json=False,
        inject_reading_malformed_json=False,
        inject_missing_font_stats=False,
    )

    output = capsys.readouterr().out
    expected_markers = [
        "[1/8] Extractor stage...",
        "[2/8] Overlay stage...",
        "[3/8] Vision stage...",
        "[4/8] Paragraphs stage...",
        "[5/8] Citations stage...",
        "[6/8] Figures-tables stage...",
        "[7/8] Reading stage...",
        "[8/8] Render stage...",
        "[9/9] Verify stage...",
    ]

    positions = [output.index(marker) for marker in expected_markers]
    assert positions == sorted(positions)
    assert "Verification PASSED" in output
