import json
from pathlib import Path

from ingest.paragraphs import run_paragraphs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_paragraphs_writes_structure_quality_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "structure_quality_doc"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    vision_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        text_dir / "blocks_norm.jsonl",
        [
            {
                "block_id": "p1_title",
                "page": 1,
                "bbox_pt": [80.0, 70.0, 520.0, 120.0],
                "text": "Structure Quality Study",
                "is_header_footer_candidate": False,
                "is_heading_candidate": True,
            },
            {
                "block_id": "p1_b1",
                "page": 1,
                "bbox_pt": [80.0, 180.0, 520.0, 260.0],
                "text": "This paragraph begins the section properly.",
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
            },
        ],
    )

    with open(vision_dir / "p001_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 1,
                "reading_order": ["p1_title", "p1_b1"],
                "merge_groups": [],
                "role_labels": {
                    "p1_title": "Heading",
                    "p1_b1": "Body",
                },
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    run_paragraphs(run_dir)

    structure_quality_path = run_dir / "qa" / "structure_quality.json"
    assert structure_quality_path.exists()

    payload = json.loads(structure_quality_path.read_text(encoding="utf-8"))
    assert payload["doc_id"] == "structure_quality_doc"
    assert payload["parser_backend"] == "builtin"
    assert isinstance(payload["ordering_confidence_low"], bool)
    assert isinstance(payload["section_boundary_unstable"], bool)
    assert isinstance(payload["reference_region_ambiguous"], bool)
    assert isinstance(payload["caption_linking_partial"], bool)
