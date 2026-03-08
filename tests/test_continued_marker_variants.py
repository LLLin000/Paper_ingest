import json
from pathlib import Path
from ingest.paragraphs import run_paragraphs


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_various_continued_markers(tmp_path):
    run_dir = tmp_path / "run" / "continued_variants"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"

    # blocks_norm: title, marker-only paragraph, real paragraph
    blocks = [
        {"block_id": "b1", "page": 1, "bbox_pt": [0, 0, 100, 10], "text": "A Study on Things"},
    ]

    variants = ["( Continued )", "(Continued).", "Continued", "Cont.", "( cont )"]
    for i, var in enumerate(variants, start=2):
        blocks.append({"block_id": f"b{i}", "page": 1, "bbox_pt": [0, i * 10, 100, i * 10 + 8], "text": var})
        blocks.append({"block_id": f"b{i}x", "page": 1, "bbox_pt": [0, i * 10 + 4, 100, i * 10 + 14], "text": "This is the first real paragraph."})

    _write_jsonl(text_dir / "blocks_norm.jsonl", blocks)

    # Minimal vision outputs: everything labeled Body
    _write_jsonl(vision_dir / "p001_out.json", [{"confidence_by_page": {}, "merge_groups_by_page": {}, "role_labels_by_page": {}}])

    paras, blocks_count = run_paragraphs(run_dir)

    # Read produced clean_document.md
    clean_md = (text_dir / "clean_document.md").read_text(encoding="utf-8")

    # Ensure none of the marker-only variants appear in clean_document
    for var in variants:
        assert var not in clean_md

    # Ensure real paragraph text present
    assert "This is the first real paragraph." in clean_md
