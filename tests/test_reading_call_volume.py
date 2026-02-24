import json
from datetime import datetime, timezone
from pathlib import Path

from ingest.manifest import Manifest
from ingest.reading import build_local_fact_candidates, run_reading, select_analysis_paragraphs


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_manifest() -> Manifest:
    return Manifest(
        doc_id="reading_task3_test",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="a" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _paragraph(i: int) -> dict[str, object]:
    return {
        "para_id": f"para_{i:04d}",
        "page_span": {"start": i // 5 + 1, "end": i // 5 + 1},
        "role": "Body",
        "section_path": ["Introduction"],
        "text": (
            f"This study reports result {i} with a significant increase in response rate by {i + 3} percent "
            "compared with baseline treatment outcomes."
        ),
        "evidence_pointer": {
            "page": i // 5 + 1,
            "bbox_union": [10.0, float(20 + i), 200.0, float(40 + i)],
            "source_block_ids": [f"p{i // 5 + 1}_b{i}"],
        },
        "neighbors": {"prev": None, "next": None},
        "confidence": 0.95,
    }


def test_build_local_fact_candidates_produces_contract_ready_candidates() -> None:
    paragraphs = [_paragraph(i) for i in range(6)]

    candidates = build_local_fact_candidates(paragraphs)

    assert len(candidates) == 6
    first = candidates[0]
    assert first["para_id"].startswith("para_")
    assert first["category"] in {
        "result",
        "statistics",
        "comparison",
        "definition",
        "mechanism",
        "limitation",
        "recommendation",
        "background",
        "none",
    }
    assert "statement" in first and len(str(first["statement"]).split()) >= 6
    assert "evidence_pointer" in first


def test_run_reading_uses_fewer_global_facts_calls_than_legacy_batch_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run" / "task3"
    (run_dir / "paragraphs").mkdir(parents=True, exist_ok=True)
    paragraphs = [_paragraph(i) for i in range(25)]
    _write_jsonl(run_dir / "paragraphs" / "paragraphs.jsonl", paragraphs)

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")

    def _fake_call(prompt: str, max_tokens: int = 4000):
        _ = max_tokens
        meta = {
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
            raw = json.dumps({
                "paper_type": "review",
                "paper_type_confidence": 0.9,
                "research_problem": "test",
                "claimed_contribution": "test",
                "reading_strategy": "evidence_synthesis",
            })
        elif "Construct an argument flow graph" in prompt:
            raw = json.dumps({"nodes": [], "edges": [], "argument_flow": {"premises": [], "core_claims": [], "conclusions": []}})
        elif "Refine deterministic local fact candidates" in prompt:
            raw = "[]"
        elif "Group related facts into themes" in prompt:
            raw = json.dumps({"themes": [], "cross_theme_links": [], "contradictions": []})
        elif "Generate executive summary" in prompt:
            raw = json.dumps({"executive_summary": "ok", "key_evidence_lines": [], "figure_table_slots": []})
        else:
            raw = "{}"
        meta["response_chars"] = len(raw)
        meta["response_preview"] = raw[:300]
        return raw, meta

    monkeypatch.setattr("ingest.reading.call_siliconflow", _fake_call)

    run_reading(run_dir, manifest=_make_manifest())

    analysis_count = len(select_analysis_paragraphs(paragraphs))
    old_facts_calls = (analysis_count + 10 - 1) // 10

    llm_calls_path = run_dir / "qa" / "llm_calls.jsonl"
    with open(llm_calls_path, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    new_facts_calls = len([e for e in events if str(e.get("step", "")).startswith("facts_global_")])

    assert old_facts_calls >= 3
    assert new_facts_calls == 1
    assert new_facts_calls < old_facts_calls
