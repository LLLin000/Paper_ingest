import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingest.manifest import Manifest
from ingest.reading import Fact, build_summary_status, run_reading
from ingest.render import run_render


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_summary_status_full_mode_when_thresholds_pass() -> None:
    paragraphs = []
    facts = []
    for idx in range(12):
        para_id = f"para_{idx:04d}"
        paragraphs.append({
            "para_id": para_id,
            "role": "Body",
            "section_path": ["Introduction"],
            "clean_roles": ["body_text"],
            "text": f"Body paragraph {idx} with substantive narrative evidence and quantitative findings.",
            "evidence_pointer": {"source_block_ids": [f"b{idx:04d}"]},
        })
        facts.append(
            Fact(
                fact_id=f"fact_{idx:04d}",
                para_id=para_id,
                category="result",
                statement=f"Narrative finding {idx} reports statistically significant improvement.",
                quote="Narrative finding quote",
                evidence_pointer={"page": 1, "bbox": [0, 0, 1, 1], "source_block_ids": [f"b{idx:04d}"]},
            )
        )

    themes = {
        "themes": [
            {"theme_id": "t1"},
            {"theme_id": "t2"},
            {"theme_id": "t3"},
        ]
    }
    synthesis = {
        "executive_summary": "This synthesis summarizes the paper with grounded evidence.",
        "key_evidence_lines": [
            {"line_id": "l1", "statement": "Evidence line 1", "fact_ids": ["fact_0000", "fact_0001"]},
            {"line_id": "l2", "statement": "Evidence line 2", "fact_ids": ["fact_0002", "fact_0003"]},
            {"line_id": "l3", "statement": "Evidence line 3", "fact_ids": ["fact_0004", "fact_0005"]},
        ],
    }

    summary_status = build_summary_status(
        doc_id="doc_full",
        facts=facts,
        themes=themes,
        synthesis=synthesis,
        paragraphs=paragraphs,
        clean_role_by_block={},
        clean_document_metrics={
            "ordering_confidence_low": False,
            "section_boundary_unstable": False,
        },
    )

    assert summary_status["status"] == "full"
    assert summary_status["reason_codes"] == []
    assert summary_status["metrics"]["narrative_fact_count"] == 12
    assert summary_status["metrics"]["themes_count"] == 3
    assert summary_status["metrics"]["synthesis_present"] is True
    assert summary_status["metrics"]["narrative_evidence_ratio"] == 1.0


def test_build_summary_status_degraded_mode_sets_explicit_missing_reasons() -> None:
    summary_status = build_summary_status(
        doc_id="doc_degraded",
        facts=[],
        themes={},
        synthesis={},
        paragraphs=[],
        clean_role_by_block={},
        clean_document_metrics={
            "ordering_confidence_low": False,
            "section_boundary_unstable": False,
        },
    )

    assert summary_status["status"] == "degraded"
    reason_codes = set(summary_status["reason_codes"])
    assert "missing_facts" in reason_codes
    assert "missing_themes" in reason_codes
    assert "missing_synthesis" in reason_codes
    assert "missing_key_evidence_fact_links" in reason_codes
    assert "low_narrative_coverage" in reason_codes


def test_renderer_mode_is_deterministic_from_summary_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "doc_render"
    manifest = Manifest(
        doc_id="doc_render",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="b" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    _write_json(
        run_dir / "reading" / "paper_profile.json",
        {
            "paper_type": "original_research",
            "paper_type_confidence": 0.9,
            "research_problem": "Test problem",
            "claimed_contribution": "Test contribution",
            "reading_strategy": "methods_first",
        },
    )
    _write_json(run_dir / "reading" / "logic_graph.json", {"nodes": [], "edges": [], "argument_flow": {"premises": [], "core_claims": [], "conclusions": []}})
    _write_json(run_dir / "reading" / "themes.json", {"themes": [{"theme_id": "t1"}], "cross_theme_links": [], "contradictions": []})
    _write_json(
        run_dir / "reading" / "synthesis.json",
        {
            "executive_summary": "Executive summary text.",
            "key_evidence_lines": [
                {"line_id": "line_1", "statement": "Key finding statement", "fact_ids": ["fact_1"], "strength": "moderate", "is_strong_claim": False}
            ],
            "figure_table_slots": [],
        },
    )
    _write_jsonl(
        run_dir / "reading" / "facts.jsonl",
        [
            {
                "fact_id": "fact_1",
                "para_id": "para_1",
                "category": "result",
                "statement": "Fact statement with quantitative comparison.",
                "quote": "Fact quote",
                "evidence_pointer": {"page": 1, "bbox": [0, 0, 1, 1], "source_block_ids": ["b1"]},
            }
        ],
    )
    _write_jsonl(
        run_dir / "paragraphs" / "paragraphs.jsonl",
        [
            {
                "para_id": "para_1",
                "role": "Body",
                "section_path": ["Introduction"],
                "text": "Paragraph text containing narrative evidence for rendering.",
                "evidence_pointer": {"page": 1, "bbox_union": [0, 0, 1, 1], "source_block_ids": ["b1"]},
            }
        ],
    )

    _write_json(
        run_dir / "qa" / "summary_status.json",
        {
            "doc_id": "doc_render",
            "status": "full",
            "reason_codes": [],
            "metrics": {
                "narrative_fact_count": 12,
                "themes_count": 3,
                "synthesis_present": True,
                "narrative_evidence_ratio": 0.95,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    _, output_path = run_render(run_dir, manifest)
    full_render = output_path.read_text(encoding="utf-8")
    assert "summary_status: full" in full_render
    assert "## 研究问题与假设 (Research Question & Hypothesis)" in full_render

    _write_json(
        run_dir / "qa" / "summary_status.json",
        {
            "doc_id": "doc_render",
            "status": "degraded",
            "reason_codes": ["missing_synthesis"],
            "metrics": {
                "narrative_fact_count": 1,
                "themes_count": 1,
                "synthesis_present": False,
                "narrative_evidence_ratio": 0.0,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    _, output_path = run_render(run_dir, manifest)
    degraded_render = output_path.read_text(encoding="utf-8")
    assert "summary_status: degraded" in degraded_render
    assert "## 摘要质量状态 (Summary Quality Status)" in degraded_render
    assert "## 降级摘要 (Degraded Summary)" in degraded_render
    assert "## 研究问题与假设 (Research Question & Hypothesis)" not in degraded_render
    assert "无提取的关键发现。" not in degraded_render


def test_run_reading_always_emits_summary_status_artifact(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "doc_reading"
    _write_jsonl(
        run_dir / "paragraphs" / "paragraphs.jsonl",
        [
            {
                "para_id": "para_0001",
                "page_span": {"start": 1, "end": 1},
                "role": "Body",
                "section_path": ["Introduction"],
                "clean_roles": ["body_text"],
                "text": "This study reports meaningful narrative outcomes with quantitative improvements.",
                "evidence_pointer": {"page": 1, "bbox_union": [0, 0, 1, 1], "source_block_ids": ["b1"]},
                "neighbors": {"prev": None, "next": None},
                "confidence": 0.95,
            }
        ],
    )
    _write_json(
        run_dir / "qa" / "clean_document_metrics.json",
        {
            "ordering_confidence_low": False,
            "section_boundary_unstable": False,
        },
    )

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")

    def _fake_call(prompt: str, max_tokens: int = 4000) -> tuple[str, dict[str, Any]]:
        _ = max_tokens
        meta: dict[str, Any] = {
            "model": "fake",
            "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
            "success": True,
            "error_type": "none",
            "http_status": 200,
            "prompt_chars": len(prompt),
            "response_chars": 0,
            "response_preview": "",
        }
        if "Analyze this academic paper" in prompt:
            raw = json.dumps(
                {
                    "paper_type": "original_research",
                    "paper_type_confidence": 0.9,
                    "research_problem": "test",
                    "claimed_contribution": "test",
                    "reading_strategy": "methods_first",
                }
            )
        elif "Construct an argument flow graph" in prompt:
            raw = json.dumps({"nodes": [], "edges": [], "argument_flow": {"premises": [], "core_claims": [], "conclusions": []}})
        elif "Refine deterministic local fact candidates" in prompt:
            raw = json.dumps(
                [
                    {
                        "fact_id": "fact_1",
                        "para_id": "para_0001",
                        "category": "result",
                        "statement": "Narrative fact statement for gate computation.",
                        "quote": "Narrative fact statement for gate computation.",
                        "evidence_pointer": {"page": 1, "bbox": [0, 0, 1, 1], "source_block_ids": ["b1"]},
                    }
                ]
            )
        elif "Group related facts into themes" in prompt:
            raw = json.dumps({"themes": [{"theme_id": "t1", "name": "Theme"}], "cross_theme_links": [], "contradictions": []})
        elif "Generate executive summary" in prompt:
            raw = json.dumps(
                {
                    "executive_summary": "Short synthesis",
                    "key_evidence_lines": [{"line_id": "l1", "statement": "Line", "fact_ids": ["fact_1"]}],
                    "figure_table_slots": [],
                }
            )
        else:
            raw = "{}"
        meta["response_chars"] = len(raw)
        meta["response_preview"] = raw[:300]
        return raw, meta

    monkeypatch.setattr("ingest.reading.call_siliconflow", _fake_call)

    manifest = Manifest(
        doc_id="doc_reading",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="c" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    run_reading(run_dir, manifest=manifest)

    summary_status_path = run_dir / "qa" / "summary_status.json"
    assert summary_status_path.exists()
    summary_status = json.loads(summary_status_path.read_text(encoding="utf-8"))
    assert summary_status["doc_id"] == "doc_reading"
    assert summary_status["status"] in {"full", "degraded"}
    assert isinstance(summary_status["reason_codes"], list)
    assert set(summary_status["metrics"].keys()) == {
        "narrative_fact_count",
        "themes_count",
        "synthesis_present",
        "narrative_evidence_ratio",
    }
