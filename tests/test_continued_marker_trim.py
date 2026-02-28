import json
from pathlib import Path

from ingest.paragraphs import run_paragraphs


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_continued_marker_trim(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "continued_marker"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        text_dir / "blocks_norm.jsonl",
        [
            {
                "block_id": "p1_title",
                "page": 1,
                "bbox_pt": [80.0, 70.0, 520.0, 120.0],
                "text": "Continued Marker Study",
                "is_header_footer_candidate": False,
                "is_heading_candidate": True,
            },
            {
                "block_id": "p1_b1",
                "page": 1,
                "bbox_pt": [80.0, 140.0, 520.0, 170.0],
                "text": "( Continued )",
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
            },
            {
                "block_id": "p1_b2",
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
                "reading_order": ["p1_title", "p1_b1", "p1_b2"],
                "merge_groups": [],
                "role_labels": {
                    "p1_title": "Heading",
                    "p1_b1": "Body",
                    "p1_b2": "Body",
                },
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()

    # The marker-only paragraph should be removed and not appear in the clean document
    assert "( Continued )" not in clean_doc
    # Title containing 'Continued' is expected; ensure the marker-only paragraph
    # specifically was removed and the real body paragraph remains.
    assert "Continued Marker Study" in clean_doc
    assert "This paragraph begins the section properly." in clean_doc
