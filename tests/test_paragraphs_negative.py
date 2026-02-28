from pathlib import Path
import json

from ingest.paragraphs import run_paragraphs


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_section_start_lowercase_but_legit(tmp_path: Path):
    run_dir = tmp_path / "run" / "section_fragments_neg"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a heading followed by a legitimate lowercase technical term
    # that should NOT be merged (e.g., gene names that start lowercase)
    blocks = [
        {"block_id": "b1", "page": 1, "bbox_pt": [40.0, 50.0, 520.0, 80.0], "text": "3. Results", "is_heading_candidate": True},
        {"block_id": "b2", "page": 1, "bbox_pt": [40.0, 90.0, 520.0, 110.0], "text": "e coli strains were used.", "is_heading_candidate": False},
        {"block_id": "b3", "page": 1, "bbox_pt": [40.0, 120.0, 520.0, 200.0], "text": "Data shows a significant increase.", "is_heading_candidate": False},
    ]
    _write_jsonl(text_dir / "blocks_norm.jsonl", blocks)

    _write_jsonl(vision_dir / "p001_out.json", [{"page": 1, "reading_order": ["b1", "b2", "b3"], "merge_groups": [], "role_labels": {"b1": "Heading", "b2": "Body", "b3": "Body"}, "confidence": 0.95}])

    paras, blocks_count = run_paragraphs(run_dir)

    clean_md = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    # Ensure the lowercase technical start remained present and not merged away
    assert "e coli strains were used." in clean_md


def test_section_start_legitimate_lowercase_sentence_is_preserved(tmp_path: Path):
    run_dir = tmp_path / "run" / "section_legit_lowercase_sentence"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {"block_id": "b1", "page": 1, "bbox_pt": [40.0, 50.0, 520.0, 80.0], "text": "3. Results", "is_heading_candidate": True},
        {
            "block_id": "b2",
            "page": 1,
            "bbox_pt": [40.0, 90.0, 520.0, 160.0],
            "text": "e coli strains were used. Data shows a significant increase.",
            "is_heading_candidate": False,
        },
    ]
    _write_jsonl(text_dir / "blocks_norm.jsonl", blocks)

    _write_jsonl(
        vision_dir / "p001_out.json",
        [
            {
                "page": 1,
                "reading_order": ["b1", "b2"],
                "merge_groups": [],
                "role_labels": {"b1": "Heading", "b2": "Body"},
                "confidence": 0.95,
            }
        ],
    )

    _ = run_paragraphs(run_dir)

    clean_md = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    main_body = clean_md.split("## Main Body", maxsplit=1)[-1]
    assert "### 3. Results" in clean_md
    assert "e coli strains were used." in main_body
    assert "Data shows a significant increase." in main_body
