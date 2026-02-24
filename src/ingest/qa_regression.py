"""Benchmark QA regression runner and trend KPI summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import cast


DEFAULT_DOC_IDS = ("struct05full", "struct06full", "struct07full")
FAULT_CATEGORIES = ("config/auth", "request-contract", "parse/validation", "timeout/retry")

JsonDict = dict[str, object]


def _as_dict(value: object) -> JsonDict:
    if isinstance(value, dict):
        return cast(JsonDict, value)
    return {}


def _load_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        loaded = cast(object, json.load(f))
    return _as_dict(loaded)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(cast(int | float | str, value))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(cast(int | float | str, value))
    except (TypeError, ValueError):
        return default


def _tail_lines(value: str, size: int = 5) -> list[str]:
    return value.strip().splitlines()[-size:]


def _snapshot_doc(run_root: Path, doc_id: str) -> JsonDict:
    qa_dir = run_root / doc_id / "qa"
    report = _load_json(qa_dir / "report.json")
    stage_status = _load_json(qa_dir / "stage_status.json")
    fault_summary = _load_json(qa_dir / "fault_summary.json")

    gates = _as_dict(report.get("gates"))
    quality_gates: list[JsonDict] = []
    for raw_gate in gates.values():
        gate = _as_dict(raw_gate)
        if "golden_available" in _as_dict(gate.get("details")):
            quality_gates.append(gate)

    evaluated_gate_count = sum(
        1 for gate in quality_gates if str(gate.get("status", "")).lower() != "not_evaluated"
    )
    total_quality_gates = len(quality_gates)
    citation_coverage = _as_float(_as_dict(gates.get("citation")).get("value"))

    by_category_src = _as_dict(fault_summary.get("window_failure_counts_by_category"))
    by_category: JsonDict = {
        category: _as_int(by_category_src.get(category), default=0) for category in FAULT_CATEGORIES
    }

    window_failure_count_raw = _as_int(fault_summary.get("window_failure_count_raw"), default=0)
    timeout_count = _as_int(by_category.get("timeout/retry"), default=0)
    timeout_rate = (timeout_count / window_failure_count_raw) if window_failure_count_raw > 0 else 0.0

    return {
        "doc_id": doc_id,
        "overall_status": str(report.get("overall_status", "unknown")),
        "verify_status": str(stage_status.get("status", "unknown")),
        "evaluated_gate_count": evaluated_gate_count,
        "total_quality_gates": total_quality_gates,
        "citation_coverage": citation_coverage,
        "window_failure_count_raw": window_failure_count_raw,
        "window_failure_counts_by_category": by_category,
        "timeout_rate": timeout_rate,
        "artifact_paths": {
            "report": str((qa_dir / "report.json").as_posix()),
            "stage_status": str((qa_dir / "stage_status.json").as_posix()),
            "fault_summary": str((qa_dir / "fault_summary.json").as_posix()),
        },
    }


def _aggregate_kpis(snapshots: list[JsonDict]) -> JsonDict:
    doc_count = len(snapshots)
    evaluated_total = sum(_as_int(snapshot.get("evaluated_gate_count"), default=0) for snapshot in snapshots)
    quality_total = sum(_as_int(snapshot.get("total_quality_gates"), default=0) for snapshot in snapshots)
    evaluated_gate_coverage = (evaluated_total / quality_total) if quality_total > 0 else 0.0

    citations = [_as_float(snapshot.get("citation_coverage")) for snapshot in snapshots]
    citation_values = [value for value in citations if value is not None]
    citation_coverage = (sum(citation_values) / len(citation_values)) if citation_values else 0.0

    degraded_doc_count_by_category: JsonDict = {}
    degraded_rate_by_category: JsonDict = {}
    for category in FAULT_CATEGORIES:
        count = sum(
            1
            for snapshot in snapshots
            if _as_int(_as_dict(snapshot.get("window_failure_counts_by_category")).get(category), default=0) > 0
        )
        degraded_doc_count_by_category[category] = count
        degraded_rate_by_category[category] = (count / doc_count) if doc_count > 0 else 0.0

    timeout_failure_count = sum(
        _as_int(_as_dict(snapshot.get("window_failure_counts_by_category")).get("timeout/retry"), default=0)
        for snapshot in snapshots
    )
    window_failure_count_raw = sum(
        _as_int(snapshot.get("window_failure_count_raw"), default=0) for snapshot in snapshots
    )
    timeout_rate = (timeout_failure_count / window_failure_count_raw) if window_failure_count_raw > 0 else 0.0

    return {
        "doc_count": doc_count,
        "evaluated_gate_coverage": evaluated_gate_coverage,
        "evaluated_gate_count": evaluated_total,
        "evaluated_gate_total": quality_total,
        "citation_coverage": citation_coverage,
        "degraded_doc_count_by_category": degraded_doc_count_by_category,
        "degraded_rate_by_category": degraded_rate_by_category,
        "timeout_rate": timeout_rate,
        "timeout_failure_count": timeout_failure_count,
        "window_failure_count_raw": window_failure_count_raw,
    }


def _compute_delta(baseline: JsonDict, optimized: JsonDict) -> JsonDict:
    delta: JsonDict = {}
    for field in (
        "evaluated_gate_coverage",
        "citation_coverage",
        "timeout_rate",
        "window_failure_count_raw",
        "timeout_failure_count",
    ):
        before = _as_float(baseline.get(field))
        after = _as_float(optimized.get(field))
        if before is not None and after is not None:
            delta[field] = after - before

    category_delta: JsonDict = {}
    base_rates = _as_dict(baseline.get("degraded_rate_by_category"))
    opt_rates = _as_dict(optimized.get("degraded_rate_by_category"))
    for category in FAULT_CATEGORIES:
        before = _as_float(base_rates.get(category), default=0.0) or 0.0
        after = _as_float(opt_rates.get(category), default=0.0) or 0.0
        category_delta[category] = after - before
    delta["degraded_rate_by_category"] = category_delta
    return delta


def _run_pipeline_for_doc(doc_id: str) -> list[JsonDict]:
    commands = [
        [sys.executable, "-m", "ingest.cli", "--doc_id", doc_id, "--stage", "full"],
        [sys.executable, "-m", "ingest.cli", "--doc_id", doc_id, "--stage", "verify"],
    ]
    results: list[JsonDict] = []
    for command in commands:
        proc = subprocess.run(command, check=False, capture_output=True, text=True)
        results.append(
            {
                "command": " ".join(command),
                "exit_code": proc.returncode,
                "stdout_tail": _tail_lines(proc.stdout),
                "stderr_tail": _tail_lines(proc.stderr),
            }
        )
    return results


def run_regression(run_root: Path, output: Path, doc_ids: tuple[str, ...]) -> JsonDict:
    baseline_snapshots = [_snapshot_doc(run_root=run_root, doc_id=doc_id) for doc_id in doc_ids]
    baseline_kpis = _aggregate_kpis(baseline_snapshots)

    command_results: dict[str, list[JsonDict]] = {
        doc_id: _run_pipeline_for_doc(doc_id) for doc_id in doc_ids
    }

    optimized_snapshots = [_snapshot_doc(run_root=run_root, doc_id=doc_id) for doc_id in doc_ids]
    optimized_kpis = _aggregate_kpis(optimized_snapshots)

    runs: list[JsonDict] = []
    for baseline_snapshot, optimized_snapshot in zip(baseline_snapshots, optimized_snapshots):
        doc_id = str(baseline_snapshot.get("doc_id", ""))
        runs.append(
            {
                "doc_id": doc_id,
                "baseline": baseline_snapshot,
                "optimized": optimized_snapshot,
                "delta": _compute_delta(_aggregate_kpis([baseline_snapshot]), _aggregate_kpis([optimized_snapshot])),
                "commands": command_results.get(doc_id, []),
            }
        )

    summary: JsonDict = {
        "schema_version": "task7.qa_regression.v1",
        "benchmark_doc_ids": list(doc_ids),
        "kpis": {
            "baseline": baseline_kpis,
            "optimized": optimized_kpis,
            "delta": _compute_delta(baseline_kpis, optimized_kpis),
        },
        "runs": runs,
        "traceability": {
            "run_window_semantics": "timeout and degraded category rates use fault_summary.window_* fields from each run/<doc_id>/qa/fault_summary.json",
            "kpi_sources": {
                "evaluated_gate_coverage": "run/<doc_id>/qa/report.json gates.*.status with details.golden_available",
                "citation_coverage": "run/<doc_id>/qa/report.json gates.citation.value",
                "degraded_rate_by_category": "run/<doc_id>/qa/fault_summary.json window_failure_counts_by_category",
                "timeout_rate": "run/<doc_id>/qa/fault_summary.json category timeout/retry divided by window_failure_count_raw",
            },
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        _ = f.write("\n")
    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark full->verify regression and emit trend KPI summary."
    )
    _ = parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="Benchmark doc id. Repeat for multiple docs. Defaults to struct05full/struct06full/struct07full.",
    )
    _ = parser.add_argument(
        "--run-root",
        default="run",
        help="Run root containing run/<doc_id>/qa artifacts.",
    )
    _ = parser.add_argument(
        "--output",
        default="run/qa_regression_trend_summary.json",
        help="Output summary JSON path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    doc_ids_arg = cast(list[str] | None, getattr(args, "doc_ids", None))
    selected = doc_ids_arg if doc_ids_arg else list(DEFAULT_DOC_IDS)
    doc_ids = tuple(sorted({doc_id.strip() for doc_id in selected if doc_id and doc_id.strip()}))
    if not doc_ids:
        print("No doc ids provided.", file=sys.stderr)
        return 1

    run_root = Path(cast(str, getattr(args, "run_root", "run")))
    output = Path(cast(str, getattr(args, "output", "run/qa_regression_trend_summary.json")))
    summary = run_regression(run_root=run_root, output=output, doc_ids=doc_ids)
    delta = _as_dict(_as_dict(summary.get("kpis")).get("delta"))

    delta_line = (
        f"KPI delta: evaluated_gate_coverage={_as_float(delta.get('evaluated_gate_coverage'), 0.0):+.6f}, "
        f"citation_coverage={_as_float(delta.get('citation_coverage'), 0.0):+.6f}, "
        f"timeout_rate={_as_float(delta.get('timeout_rate'), 0.0):+.6f}"
    )
    print(f"Wrote regression summary: {output.as_posix()}")
    print(f"Docs: {', '.join(doc_ids)}")
    print(delta_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
