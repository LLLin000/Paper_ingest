import json
from pathlib import Path

from ingest.verify import GateStatus, compute_citation_gate


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_doi_pmid_precision_excludes_non_truth_identifier_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "citations" / "cite_anchors.jsonl",
        [
            {"anchor_id": "a_truth", "anchor_type": "citation_marker", "anchor_text": "[1]"},
            {"anchor_id": "a_extra", "anchor_type": "citation_marker", "anchor_text": "[2]"},
        ],
    )
    _write_jsonl(
        run_dir / "citations" / "cite_map.jsonl",
        [
            {"anchor_id": "a_truth", "mapped_ref_key": "doi:10.1000/xyz"},
            {"anchor_id": "a_extra", "mapped_ref_key": "doi:10.2000/extra"},
        ],
    )

    golden = {
        "citation_truth": [
            {"marker_id": "a_truth", "expected_ref_key": "doi:10.1000/xyz"},
        ]
    }

    gate = compute_citation_gate(run_dir, golden)

    assert gate.status == GateStatus.PASS
    assert gate.details["doi_pmid_precision"] == 1.0
    assert gate.details["correct_doi_pmid"] == 1
    assert gate.details["extracted_doi_pmid"] == 1
    assert gate.details["unscored_extracted_doi_pmid"] == 1


def test_doi_pmid_precision_counts_missing_truth_identifier_as_incorrect(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "citations" / "cite_anchors.jsonl",
        [{"anchor_id": "a_truth", "anchor_type": "citation_marker", "anchor_text": "[1]"}],
    )
    _write_jsonl(
        run_dir / "citations" / "cite_map.jsonl",
        [{"anchor_id": "a_truth", "mapped_ref_key": None}],
    )

    golden = {
        "citation_truth": [
            {"marker_id": "a_truth", "expected_ref_key": "pmid:12345678"},
        ]
    }

    gate = compute_citation_gate(run_dir, golden)

    assert gate.details["doi_pmid_precision"] == 0.0
    assert gate.details["correct_doi_pmid"] == 0
    assert gate.details["extracted_doi_pmid"] == 1
    assert gate.details["missing_doi_pmid"] == 1


def test_doi_pmid_precision_allows_multi_identifier_matches(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "citations" / "cite_anchors.jsonl",
        [{"anchor_id": "a_truth", "anchor_type": "citation_marker", "anchor_text": "[1]"}],
    )
    _write_jsonl(
        run_dir / "citations" / "cite_map.jsonl",
        [{"anchor_id": "a_truth", "mapped_ref_key": "pmid:7654321"}],
    )

    golden = {
        "citation_truth": [
            {
                "marker_id": "a_truth",
                "expected_ref_key": ["doi:10.1000/abc", "pmid:7654321"],
            },
        ]
    }

    gate = compute_citation_gate(run_dir, golden)

    assert gate.details["doi_pmid_precision"] == 1.0
    assert gate.details["correct_doi_pmid"] == 1
    semantics = gate.details["doi_pmid_precision_semantics"]
    assert "multi_identifier_handling" in semantics


def test_citation_coverage_uses_golden_truth_marker_scope(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "citations" / "cite_anchors.jsonl",
        [
            {"anchor_id": "a_truth", "anchor_type": "citation_marker", "anchor_text": "[1]"},
            {"anchor_id": "a_extra_1", "anchor_type": "citation_marker", "anchor_text": "[2]"},
            {"anchor_id": "a_extra_2", "anchor_type": "citation_marker", "anchor_text": "[3]"},
        ],
    )
    _write_jsonl(
        run_dir / "citations" / "cite_map.jsonl",
        [
            {"anchor_id": "a_truth", "mapped_ref_key": "doi:10.1000/xyz"},
            {"anchor_id": "a_extra_1", "mapped_ref_key": None},
            {"anchor_id": "a_extra_2", "mapped_ref_key": None},
        ],
    )

    golden = {
        "citation_truth": [
            {"marker_id": "a_truth", "expected_ref_key": "doi:10.1000/xyz"},
        ]
    }

    gate = compute_citation_gate(run_dir, golden)

    assert gate.status == GateStatus.PASS
    assert gate.value == 1.0
    assert gate.details["mapped_markers"] == 1
    assert gate.details["total_markers"] == 1
    assert gate.details["coverage_scope"] == "golden_citation_truth"
