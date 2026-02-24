import json
from pathlib import Path

from ingest.verify import GateStatus, compute_provenance_gate


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _golden_with_data() -> dict[str, object]:
    return {
        "reading_order_pairs": [
            {
                "left_para_id": "p_left",
                "right_para_id": "p_right",
                "expected_order": "left_before_right",
            }
        ]
    }


def test_provenance_anchor_allows_statement_fallback_with_paragraph_mapping(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "reading" / "synthesis.json",
        {
            "key_evidence_lines": [
                {
                    "line_id": "line_1",
                    "statement": "The intervention improved outcomes.",
                    "fact_ids": ["fact_1"],
                    "is_strong_claim": False,
                }
            ]
        },
    )
    _write_jsonl(
        run_dir / "reading" / "facts.jsonl",
        [
            {
                "fact_id": "fact_1",
                "para_id": "para_1",
                "category": "background",
                "statement": "The intervention improved outcomes.",
                "quote": "",
                "evidence_pointer": {
                    "page": 3,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "source_block_ids": ["p3_b1"],
                },
            }
        ],
    )
    _write_jsonl(
        run_dir / "paragraphs" / "paragraphs.jsonl",
        [
            {
                "para_id": "para_1",
                "text": "The intervention improved outcomes in the treatment group.",
                "page_span": {"start": 3, "end": 3},
                "evidence_pointer": {"pages": [3], "bbox_union": [1.0, 2.0, 3.0, 4.0]},
            }
        ],
    )

    gate = compute_provenance_gate(run_dir, _golden_with_data())

    assert gate.status == GateStatus.PASS
    assert gate.value == 1.0
    assert gate.details["anchored_atomic_claims"] == 1
    assert gate.details["total_atomic_claims"] == 1
    assert gate.details["anchor_type_counts"]["paragraph_statement_fallback"] == 1


def test_provenance_failure_reports_unanchored_claim_reasons(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "reading" / "synthesis.json",
        {
            "key_evidence_lines": [
                {
                    "line_id": "line_missing_anchor",
                    "statement": "A strong claim without valid paragraph mapping.",
                    "fact_ids": ["fact_bad"],
                    "is_strong_claim": True,
                }
            ]
        },
    )
    _write_jsonl(
        run_dir / "reading" / "facts.jsonl",
        [
            {
                "fact_id": "fact_bad",
                "para_id": "para_missing",
                "category": "result",
                "statement": "A strong claim without valid paragraph mapping.",
                "quote": "",
                "evidence_pointer": {
                    "page": 5,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "source_block_ids": ["p5_b2"],
                },
            }
        ],
    )
    _write_jsonl(
        run_dir / "paragraphs" / "paragraphs.jsonl",
        [
            {
                "para_id": "other_para",
                "text": "Unrelated paragraph.",
                "page_span": {"start": 5, "end": 5},
                "evidence_pointer": {"pages": [5], "bbox_union": [1.0, 2.0, 3.0, 4.0]},
            }
        ],
    )

    gate = compute_provenance_gate(run_dir, _golden_with_data())

    assert gate.status == GateStatus.FAIL
    assert gate.details["anchored_atomic_claims"] == 0
    assert gate.details["strong_claim_unsupported_count"] == 1
    assert gate.details["unanchored_claims"]
    first = gate.details["unanchored_claims"][0]
    assert first["line_id"] == "line_missing_anchor"
    assert any("missing_paragraph_mapping" in reason for reason in first["missing_anchor_reasons"])
