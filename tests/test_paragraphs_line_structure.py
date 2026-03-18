import json
from pathlib import Path

from ingest.paragraphs import run_paragraphs


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_paragraphs_uses_block_line_sidecar_to_split_leading_headings(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "line_heading_doc"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        text_dir / "blocks_norm.jsonl",
        [
            {
                "block_id": "p2_b2",
                "page": 2,
                "bbox_pt": [72.0, 72.0, 520.0, 160.0],
                "text": (
                    "Results Identification of three distinct subtypes of tendinopathy "
                    "based on transcriptome profiles and clinical features "
                    "To better understand the heterogeneity of tendinopathy and identify potential subtypes, "
                    "we collected clinical and transcriptomic data."
                ),
                "font_stats": {"avg_size": 11.5, "is_bold": False},
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            }
        ],
    )
    _write_jsonl(
        text_dir / "block_lines.jsonl",
        [
            {
                "block_id": "p2_b2",
                "page": 2,
                "bbox_pt": [72.0, 72.0, 520.0, 160.0],
                "lines": [
                    {
                        "line_index": 0,
                        "text": "Results",
                        "bbox_pt": [72.0, 72.0, 160.0, 88.0],
                        "font_stats": {"avg_size": 16.0, "is_bold": True, "dominant_font": "Times-Bold"},
                    },
                    {
                        "line_index": 1,
                        "text": "Identification of three distinct subtypes of tendinopathy based on",
                        "bbox_pt": [72.0, 92.0, 520.0, 108.0],
                        "font_stats": {"avg_size": 13.0, "is_bold": True, "dominant_font": "Times-Bold"},
                    },
                    {
                        "line_index": 2,
                        "text": "transcriptome profiles and clinical features",
                        "bbox_pt": [72.0, 110.0, 360.0, 126.0],
                        "font_stats": {"avg_size": 13.0, "is_bold": True, "dominant_font": "Times-Bold"},
                    },
                    {
                        "line_index": 3,
                        "text": (
                            "To better understand the heterogeneity of tendinopathy and identify potential subtypes, "
                            "we collected clinical and transcriptomic data."
                        ),
                        "bbox_pt": [72.0, 132.0, 520.0, 160.0],
                        "font_stats": {"avg_size": 10.0, "is_bold": False, "dominant_font": "Times-Roman"},
                    },
                ],
            }
        ],
    )

    with open(vision_dir / "p002_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 2,
                "reading_order": ["p2_b2"],
                "merge_groups": [],
                "role_labels": {"p2_b2": "Body"},
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    paragraphs_count, blocks_count = run_paragraphs(run_dir)

    assert paragraphs_count == 3
    assert blocks_count == 1

    paragraphs_rows = [
        json.loads(line)
        for line in (run_dir / "paragraphs" / "paragraphs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(paragraphs_rows) == 3
    assert paragraphs_rows[0]["section_path"] == ["Results"]
    assert paragraphs_rows[0]["role"] == "Heading"
    assert paragraphs_rows[1]["section_path"] == [
        "Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features"
    ]
    assert paragraphs_rows[1]["role"] == "Heading"
    assert paragraphs_rows[2]["role"] == "Body"
    assert paragraphs_rows[2]["text"].startswith("To better understand the heterogeneity of tendinopathy")
    assert paragraphs_rows[2]["evidence_pointer"]["source_line_spans"] == [{"block_id": "p2_b2", "start": 3, "end": 3}]

    clean_document = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert "### Results" in clean_document
    assert "### Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features" in clean_document
    assert "To better understand the heterogeneity of tendinopathy" in clean_document
    assert clean_document.count("Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features") == 1


def test_run_paragraphs_splits_subtle_nonbold_heading_from_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "subtle_heading_doc"
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
                "bbox_pt": [70.0, 50.0, 280.0, 90.0],
                "text": "Line Split Study",
                "font_stats": {"avg_size": 18.0, "is_bold": True},
                "is_header_footer_candidate": False,
                "is_heading_candidate": True,
                "column_guess": 1,
            },
            {
                "block_id": "p2_b0",
                "page": 2,
                "bbox_pt": [70.0, 120.0, 420.0, 260.0],
                "text": (
                    "RNA sequence Total RNA was extracted from the tissue using TRIzol Reagent "
                    "according to the manufacturer's instructions."
                ),
                "font_stats": {"avg_size": 8.3, "is_bold": False},
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
        ],
    )
    _write_jsonl(
        text_dir / "block_lines.jsonl",
        [
            {
                "block_id": "p2_b0",
                "page": 2,
                "bbox_pt": [70.0, 120.0, 420.0, 260.0],
                "lines": [
                    {
                        "line_index": 0,
                        "text": "RNA sequence",
                        "bbox_pt": [70.0, 120.0, 150.0, 140.0],
                        "font_stats": {"avg_size": 8.72, "is_bold": False, "dominant_font": "AdvOT70e1938e"},
                    },
                    {
                        "line_index": 1,
                        "text": "Total RNA was extracted from the tissue using TRIzol Reagent",
                        "bbox_pt": [70.0, 145.0, 420.0, 165.0],
                        "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "AdvOTdd63dae3"},
                    },
                    {
                        "line_index": 2,
                        "text": "according to the manufacturer's instructions.",
                        "bbox_pt": [70.0, 170.0, 330.0, 190.0],
                        "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "AdvOTdd63dae3"},
                    },
                ],
            }
        ],
    )

    with open(vision_dir / "p001_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 1,
                "reading_order": ["p1_title"],
                "merge_groups": [],
                "role_labels": {"p1_title": "Heading"},
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(vision_dir / "p002_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 2,
                "reading_order": ["p2_b0"],
                "merge_groups": [],
                "role_labels": {"p2_b0": "Heading"},
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    paragraph_count, _ = run_paragraphs(run_dir)

    assert paragraph_count == 3

    rows = [
        json.loads(line)
        for line in (run_dir / "paragraphs" / "paragraphs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["role"] for row in rows] == ["Heading", "Heading", "Body"]
    assert rows[1]["text"] == "RNA sequence"
    assert rows[2]["text"].startswith("Total RNA was extracted from the tissue")

    clean_document = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert "### RNA sequence" in clean_document
    assert "### RNA sequence Total" not in clean_document


def test_run_paragraphs_recursively_splits_nested_nonbold_heading_prefix_lines(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "nested_heading_doc"
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
                "bbox_pt": [70.0, 50.0, 280.0, 90.0],
                "text": "Line Split Study",
                "font_stats": {"avg_size": 18.0, "is_bold": True},
                "is_header_footer_candidate": False,
                "is_heading_candidate": True,
                "column_guess": 1,
            },
            {
                "block_id": "p2_b2",
                "page": 2,
                "bbox_pt": [72.0, 72.0, 520.0, 220.0],
                "text": (
                    "Results Identification of three distinct subtypes of tendinopathy based on "
                    "transcriptome profiles and clinical features "
                    "To better understand the heterogeneity of tendinopathy and identify potential subtypes, "
                    "we collected clinical and transcriptomic data from diseased tendons."
                ),
                "font_stats": {"avg_size": 8.6, "is_bold": False},
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
        ],
    )
    _write_jsonl(
        text_dir / "block_lines.jsonl",
        [
            {
                "block_id": "p2_b2",
                "page": 2,
                "bbox_pt": [72.0, 72.0, 520.0, 220.0],
                "lines": [
                    {
                        "line_index": 0,
                        "text": "Results",
                        "bbox_pt": [72.0, 72.0, 160.0, 88.0],
                        "font_stats": {"avg_size": 10.71, "is_bold": False, "dominant_font": "AdvOT3d287b35.B"},
                    },
                    {
                        "line_index": 1,
                        "text": "Identification of three distinct subtypes of tendinopathy based on",
                        "bbox_pt": [72.0, 92.0, 520.0, 108.0],
                        "font_stats": {"avg_size": 8.72, "is_bold": False, "dominant_font": "AdvOT70e1938e"},
                    },
                    {
                        "line_index": 2,
                        "text": "transcriptome profiles and clinical features",
                        "bbox_pt": [72.0, 110.0, 360.0, 126.0],
                        "font_stats": {"avg_size": 8.72, "is_bold": False, "dominant_font": "AdvOT70e1938e"},
                    },
                    {
                        "line_index": 3,
                        "text": "To better understand the heterogeneity of tendinopathy and identify potential subtypes,",
                        "bbox_pt": [72.0, 132.0, 520.0, 148.0],
                        "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "AdvOTdd63dae3"},
                    },
                    {
                        "line_index": 4,
                        "text": "we collected clinical and transcriptomic data from diseased tendons.",
                        "bbox_pt": [72.0, 150.0, 520.0, 166.0],
                        "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "AdvOTdd63dae3"},
                    },
                ],
            }
        ],
    )

    with open(vision_dir / "p001_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 1,
                "reading_order": ["p1_title"],
                "merge_groups": [],
                "role_labels": {"p1_title": "Heading"},
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(vision_dir / "p002_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 2,
                "reading_order": ["p2_b2"],
                "merge_groups": [],
                "role_labels": {"p2_b2": "Heading"},
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    paragraph_count, _ = run_paragraphs(run_dir)

    assert paragraph_count == 4

    rows = [
        json.loads(line)
        for line in (run_dir / "paragraphs" / "paragraphs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["role"] for row in rows] == ["Heading", "Heading", "Heading", "Body"]
    assert rows[1]["text"] == "Results"
    assert rows[2]["text"] == (
        "Identification of three distinct subtypes of tendinopathy based on "
        "transcriptome profiles and clinical features"
    )
    assert rows[2]["evidence_pointer"]["source_line_spans"] == [{"block_id": "p2_b2", "start": 1, "end": 2}]
    assert rows[3]["text"].startswith("To better understand the heterogeneity of tendinopathy")

    clean_document = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert "### Results" in clean_document
    assert (
        "### Identification of three distinct subtypes of tendinopathy based on "
        "transcriptome profiles and clinical features"
    ) in clean_document
    assert "Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features To better understand" not in clean_document


def test_run_paragraphs_uses_approved_heading_hints_to_split_overlong_heading_block_without_line_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "hint_split_overlong_heading"
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
                "bbox_pt": [70.0, 50.0, 280.0, 90.0],
                "text": "Heading Hint Split Study",
                "font_stats": {"avg_size": 18.0, "is_bold": True},
                "is_header_footer_candidate": False,
                "is_heading_candidate": True,
                "column_guess": 1,
            },
            {
                "block_id": "p2_b2",
                "page": 2,
                "bbox_pt": [72.0, 72.0, 520.0, 220.0],
                "text": (
                    "Results Identification of three distinct subtypes of tendinopathy based on "
                    "transcriptome profiles and clinical features "
                    "To better understand the heterogeneity of tendinopathy and identify potential subtypes, "
                    "we collected clinical and transcriptomic data from diseased tendons."
                ),
                "font_stats": {"avg_size": 8.6, "is_bold": False},
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
        ],
    )

    with open(vision_dir / "p001_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 1,
                "reading_order": ["p1_title"],
                "merge_groups": [],
                "role_labels": {"p1_title": "Heading"},
                "confidence": 0.95,
                "fallback_used": False,
                "embedded_headings": [],
                "embedded_heading_reviewed_block_ids": [],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(vision_dir / "p002_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 2,
                "reading_order": ["p2_b2"],
                "merge_groups": [],
                "role_labels": {"p2_b2": "Heading"},
                "confidence": 0.95,
                "fallback_used": False,
                "embedded_headings": [
                    {"block_id": "p2_b2", "heading_text": "Results", "confidence": 0.99},
                    {
                        "block_id": "p2_b2",
                        "heading_text": (
                            "Identification of three distinct subtypes of tendinopathy based on "
                            "transcriptome profiles and clinical features"
                        ),
                        "confidence": 0.99,
                    },
                ],
                "embedded_heading_reviewed_block_ids": ["p2_b2"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    paragraph_count, _ = run_paragraphs(run_dir)

    assert paragraph_count == 4

    rows = [
        json.loads(line)
        for line in (run_dir / "paragraphs" / "paragraphs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["role"] for row in rows] == ["Heading", "Heading", "Heading", "Body"]
    assert rows[1]["text"] == "Results"
    assert rows[2]["text"] == (
        "Identification of three distinct subtypes of tendinopathy based on "
        "transcriptome profiles and clinical features"
    )
    assert rows[3]["text"].startswith("To better understand the heterogeneity of tendinopathy")


def test_run_paragraphs_splits_mixed_merge_group_by_approved_heading_hints_without_line_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "hint_split_merge_group"
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
                "bbox_pt": [70.0, 50.0, 280.0, 90.0],
                "text": "Heading Hint Split Study",
                "font_stats": {"avg_size": 18.0, "is_bold": True},
                "is_header_footer_candidate": False,
                "is_heading_candidate": True,
                "column_guess": 1,
            },
            {
                "block_id": "p10_b2",
                "page": 10,
                "bbox_pt": [39.6, 562.6, 294.8, 731.9],
                "text": (
                    "RNA sequence Total RNA was extracted from the tissue using TRIzol Reagent "
                    "according to the manufacturer's instructions."
                ),
                "font_stats": {"avg_size": 8.3, "is_bold": False},
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
            {
                "block_id": "p10_b5",
                "page": 10,
                "bbox_pt": [306.1, 637.5, 561.3, 731.9],
                "text": (
                    "Classification based on transcriptomic and clinical data "
                    "Based on the NMF classification, we further categorized samples by the presence "
                    "of a red hyperemia state in tendons."
                ),
                "font_stats": {"avg_size": 8.3, "is_bold": False},
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 2,
            },
        ],
    )

    with open(vision_dir / "p001_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 1,
                "reading_order": ["p1_title"],
                "merge_groups": [],
                "role_labels": {"p1_title": "Heading"},
                "confidence": 0.95,
                "fallback_used": False,
                "embedded_headings": [],
                "embedded_heading_reviewed_block_ids": [],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(vision_dir / "p010_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 10,
                "reading_order": ["p10_b2", "p10_b5"],
                "merge_groups": [{"group_id": "group_4", "block_ids": ["p10_b2", "p10_b5"]}],
                "role_labels": {"p10_b2": "Body", "p10_b5": "Body"},
                "confidence": 0.95,
                "fallback_used": False,
                "embedded_headings": [
                    {"block_id": "p10_b2", "heading_text": "RNA sequence", "confidence": 1.0},
                    {
                        "block_id": "p10_b5",
                        "heading_text": "Classification based on transcriptomic and clinical data",
                        "confidence": 1.0,
                    },
                ],
                "embedded_heading_reviewed_block_ids": ["p10_b2", "p10_b5"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    paragraph_count, _ = run_paragraphs(run_dir)

    assert paragraph_count == 5

    rows = [
        json.loads(line)
        for line in (run_dir / "paragraphs" / "paragraphs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["role"] for row in rows] == ["Heading", "Heading", "Body", "Heading", "Body"]
    assert rows[1]["text"] == "RNA sequence"
    assert rows[2]["text"].startswith("Total RNA was extracted from the tissue")
    assert rows[3]["text"] == "Classification based on transcriptomic and clinical data"
    assert rows[4]["text"].startswith("Based on the NMF classification")
