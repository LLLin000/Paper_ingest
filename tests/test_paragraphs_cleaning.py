import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

import ingest.paragraphs as paragraphs_mod
from ingest.paragraphs import Paragraph, classify_clean_blocks, run_paragraphs, suppress_non_narrative_main_body_entries


def _write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_vision_outputs(
    vision_dir: Path,
    blocks: Any,
    role_labels_by_page: dict[int, dict[str, str]],
) -> None:
    grouped: dict[int, list[dict[str, Any]]] = {}

    for block in blocks:
        raw_page = block.get("page", 1)
        if isinstance(raw_page, bool):
            page = int(raw_page)
        elif isinstance(raw_page, int):
            page = raw_page
        elif isinstance(raw_page, float):
            page = int(raw_page)
        elif isinstance(raw_page, str) and raw_page.isdigit():
            page = int(raw_page)
        else:
            page = 1
        grouped.setdefault(page, []).append(block)

    def bbox_xy(block: dict[str, object]) -> tuple[float, float]:
        bbox_obj = block.get("bbox_pt", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(bbox_obj, list) or len(bbox_obj) < 2:
            return (0.0, 0.0)
        x_obj, y_obj = bbox_obj[0], bbox_obj[1]
        x0 = float(x_obj) if isinstance(x_obj, (int, float, str)) else 0.0
        y0 = float(y_obj) if isinstance(y_obj, (int, float, str)) else 0.0
        return (x0, y0)

    for page, page_blocks in grouped.items():
        ordered = sorted(
            page_blocks,
            key=lambda b: (
                bbox_xy(b)[1],
                bbox_xy(b)[0],
                str(b.get("block_id", "")),
            ),
        )
        payload = {
            "page": page,
            "reading_order": [str(b.get("block_id", "")) for b in ordered],
            "merge_groups": [],
            "role_labels": role_labels_by_page.get(page, {}),
            "confidence": 0.95,
            "fallback_used": False,
        }
        with open(vision_dir / f"p{page:03d}_out.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _section_text(markdown: str, section_title: str) -> str:
    pattern = rf"^## {re.escape(section_title)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def test_classify_clean_blocks_combines_signals() -> None:
    blocks = {
        "p1_b0": {
            "block_id": "p1_b0",
            "page": 1,
            "bbox_pt": [20.0, 10.0, 140.0, 20.0],
            "text": "AJR:200, May 2013",
            "is_header_footer_candidate": True,
        },
        "p1_b1": {
            "block_id": "p1_b1",
            "page": 1,
            "bbox_pt": [-1.0, 80.0, 8.0, 640.0],
            "text": "Downloaded from example by 1.2.3.4",
            "is_header_footer_candidate": False,
        },
        "p2_b1": {
            "block_id": "p2_b1",
            "page": 2,
            "bbox_pt": [-1.0, 82.0, 8.0, 638.0],
            "text": "Downloaded from example by 9.9.9.9",
            "is_header_footer_candidate": False,
        },
        "p1_b2": {
            "block_id": "p1_b2",
            "page": 1,
            "bbox_pt": [60.0, 200.0, 420.0, 260.0],
            "text": "This is core body content used for analysis.",
            "is_header_footer_candidate": False,
        },
    }
    role_labels = {
        1: {"p1_b0": "HeaderFooter", "p1_b1": "Body", "p1_b2": "Body"},
        2: {"p2_b1": "Body"},
    }

    kept_blocks, annotated = classify_clean_blocks(blocks, role_labels)
    by_id = {str(row.get("block_id", "")): row for row in annotated}

    assert by_id["p1_b0"]["is_nuisance"] is True
    assert "vision_role_header_footer" in by_id["p1_b0"]["nuisance_reasons"]
    assert "extractor_header_footer_candidate" in by_id["p1_b0"]["nuisance_reasons"]

    assert by_id["p1_b1"]["is_nuisance"] is True
    assert "near_side_margin" in by_id["p1_b1"]["nuisance_reasons"]
    assert any(str(reason).startswith("repeated_template_") for reason in by_id["p1_b1"]["nuisance_reasons"])

    assert by_id["p2_b1"]["is_nuisance"] is True
    assert by_id["p1_b2"]["is_nuisance"] is False
    assert by_id["p1_b2"]["clean_role"] == "body_text"
    assert set(kept_blocks.keys()) == {"p1_b2"}


def test_classify_clean_blocks_assigns_semantic_roles_and_role_based_keep_set() -> None:
    blocks = {
        "p1_title": {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 100.0, 520.0, 170.0],
            "text": "Rotator Cable MRI Study With Histologic Correlation",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 20.0, "is_bold": True},
        },
        "p1_heading": {
            "block_id": "p1_heading",
            "page": 1,
            "bbox_pt": [90.0, 220.0, 320.0, 250.0],
            "text": "Materials and Methods",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 11.0, "is_bold": True},
        },
        "p1_body": {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [90.0, 280.0, 520.0, 360.0],
            "text": "This paragraph provides the principal study details and outcomes for analysis.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
            "font_stats": {"avg_size": 9.0, "is_bold": False},
        },
        "p1_author": {
            "block_id": "p1_author",
            "page": 1,
            "bbox_pt": [90.0, 180.0, 420.0, 210.0],
            "text": "Soterios Gyftopoulos 1 Jenny Bencardino 1",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
            "font_stats": {"avg_size": 8.5, "is_bold": False},
        },
        "p1_aff": {
            "block_id": "p1_aff",
            "page": 1,
            "bbox_pt": [90.0, 590.0, 520.0, 640.0],
            "text": "1 Department of Radiology, NYU Langone Medical Center, New York, NY 10016",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
            "font_stats": {"avg_size": 7.0, "is_bold": False},
        },
        "p1_ref": {
            "block_id": "p1_ref",
            "page": 1,
            "bbox_pt": [90.0, 660.0, 520.0, 710.0],
            "text": "1. Smith J. Rotator cable mechanics. AJR. 2013;200:1101-1105.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
            "font_stats": {"avg_size": 8.0, "is_bold": False},
        },
        "p1_fig": {
            "block_id": "p1_fig",
            "page": 1,
            "bbox_pt": [120.0, 500.0, 540.0, 560.0],
            "text": "Fig. 1 - Measurement diagram and landmarks.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
            "font_stats": {"avg_size": 7.5, "is_bold": True},
        },
    }
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_heading": "Heading",
            "p1_body": "Body",
            "p1_author": "Body",
            "p1_aff": "Body",
            "p1_ref": "Body",
            "p1_fig": "FigureCaption",
        }
    }

    kept_blocks, annotated = classify_clean_blocks(blocks, role_labels)
    by_id = {str(row.get("block_id", "")): row for row in annotated}

    assert by_id["p1_title"]["clean_role"] == "main_title"
    assert by_id["p1_heading"]["clean_role"] == "section_heading"
    assert by_id["p1_body"]["clean_role"] == "body_text"
    assert by_id["p1_author"]["clean_role"] == "author_meta"
    assert by_id["p1_aff"]["clean_role"] == "affiliation_meta"
    assert by_id["p1_ref"]["clean_role"] == "reference_entry"
    assert by_id["p1_fig"]["clean_role"] == "figure_caption"

    assert set(kept_blocks.keys()) == {"p1_title", "p1_heading", "p1_body", "p1_fig", "p1_ref"}


def test_run_paragraphs_writes_clean_artifacts_and_filters_nuisance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "doc_test"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        text_dir / "blocks_norm.jsonl",
        [
            {
                "block_id": "p1_b0",
                "page": 1,
                "bbox_pt": [20.0, 8.0, 120.0, 20.0],
                "text": "Header line",
                "is_header_footer_candidate": True,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
            {
                "block_id": "p1_b1",
                "page": 1,
                "bbox_pt": [60.0, 360.0, 420.0, 460.0],
                "text": "OBJECTIVE. Main paragraph text for downstream analysis. MATERIALS AND METHODS. Details follow in body.",
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
            {
                "block_id": "p1_b2",
                "page": 1,
                "bbox_pt": [60.0, 80.0, 300.0, 110.0],
                "text": "Soterios Gyftopoulos 1 Jenny Bencardino 1",
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
            {
                "block_id": "p1_b3",
                "page": 1,
                "bbox_pt": [60.0, 660.0, 520.0, 710.0],
                "text": "1. Smith J. Rotator cable mechanics. AJR. 2013;200:1101-1105.",
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
            {
                "block_id": "p1_b4",
                "page": 1,
                "bbox_pt": [60.0, 500.0, 520.0, 560.0],
                "text": "Fig. 1 - Measurement diagram and landmarks.",
                "is_header_footer_candidate": False,
                "is_heading_candidate": False,
                "column_guess": 1,
            },
            {
                "block_id": "p1_b5",
                "page": 1,
                "bbox_pt": [60.0, 300.0, 320.0, 340.0],
                "text": "Materials and Methods",
                "is_header_footer_candidate": False,
                "is_heading_candidate": True,
                "column_guess": 1,
            },
        ],
    )

    with open(vision_dir / "p001_out.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": 1,
                "reading_order": ["p1_b0", "p1_b2", "p1_b5", "p1_b1", "p1_b4", "p1_b3"],
                "merge_groups": [],
                "role_labels": {
                    "p1_b0": "HeaderFooter",
                    "p1_b1": "Body",
                    "p1_b2": "Body",
                    "p1_b3": "Body",
                    "p1_b4": "FigureCaption",
                    "p1_b5": "Heading",
                },
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    paragraph_count, block_count = run_paragraphs(run_dir)

    assert paragraph_count == 4
    assert block_count == 6

    blocks_clean_path = text_dir / "blocks_clean.jsonl"
    clean_doc_path = text_dir / "clean_document.md"
    paragraphs_path = run_dir / "paragraphs" / "paragraphs.jsonl"
    assert blocks_clean_path.exists()
    assert clean_doc_path.exists()
    assert paragraphs_path.exists()

    with open(blocks_clean_path, "r", encoding="utf-8") as f:
        cleaned_rows = [json.loads(line) for line in f if line.strip()]
    removed = [row for row in cleaned_rows if row.get("is_nuisance")]
    assert len(removed) == 1
    assert removed[0]["block_id"] == "p1_b0"
    assert removed[0]["nuisance_reasons"]
    author_rows = [row for row in cleaned_rows if row.get("block_id") == "p1_b2"]
    assert len(author_rows) == 1
    assert author_rows[0]["is_nuisance"] is False
    assert author_rows[0]["clean_role"] == "author_meta"

    with open(clean_doc_path, "r", encoding="utf-8") as f:
        clean_doc = f.read()
    assert "## Authors" in clean_doc
    assert "## Abstract / Objective" in clean_doc
    assert "## Main Body" in clean_doc
    assert "## Figures and Tables" in clean_doc
    assert "## References" in clean_doc
    assert "Main paragraph text" in clean_doc
    assert "Header line" not in clean_doc
    assert "1. Smith J." in clean_doc

    main_body_text = clean_doc.split("## Main Body", maxsplit=1)[-1]
    assert "Soterios Gyftopoulos" not in main_body_text

    with open(paragraphs_path, "r", encoding="utf-8") as f:
        paragraphs = [json.loads(line) for line in f if line.strip()]
        assert len(paragraphs) == 4
    assert any("Main paragraph text" in str(p.get("text", "")) for p in paragraphs)


def test_main_body_suppressions_emit_trace_artifact_and_zero_target_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "suppression_trace_doc"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 60.0, 520.0, 120.0],
            "text": "Suppression Trace Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p3_body",
            "page": 3,
            "bbox_pt": [80.0, 220.0, 520.0, 320.0],
            "text": "Narrative paragraph remains in clean main body.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p3_page_number",
            "page": 3,
            "bbox_pt": [82.0, 332.0, 110.0, 348.0],
            "text": "5",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p3_updates",
            "page": 3,
            "bbox_pt": [80.0, 356.0, 320.0, 378.0],
            "text": "Check for updates",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p3_aff_like",
            "page": 3,
            "bbox_pt": [80.0, 392.0, 520.0, 432.0],
            "text": "1 Department of Radiology, Example University, New York, NY 10016",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {"p1_title": "Heading"},
        3: {
            "p3_body": "Body",
            "p3_page_number": "Body",
            "p3_updates": "Body",
            "p3_aff_like": "Body",
        },
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")
    assert "Narrative paragraph remains" in main_body
    assert "Check for updates" not in main_body
    assert "Department of Radiology" not in main_body
    assert "\n5\n" not in main_body

    suppressions_path = run_dir / "qa" / "clean_document_suppressions.jsonl"
    assert suppressions_path.exists()
    suppression_rows = _read_jsonl(suppressions_path)
    rule_ids = {str(row.get("rule_id", "")) for row in suppression_rows}
    assert rule_ids <= {"page_number_only", "check_for_updates_banner", "affiliation_address_line"}
    for row in suppression_rows:
        assert row.get("doc_id") == "suppression_trace_doc"
        assert row.get("para_id") is None or isinstance(row.get("para_id"), str)
        assert isinstance(row.get("source_block_ids"), list)
        assert row.get("normalized_text")
        assert row.get("observed_at")

    with open(run_dir / "qa" / "clean_document_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["standalone_page_number_count"] == 0
    assert metrics["check_for_updates_count"] == 0


def test_suppress_non_narrative_main_body_entries_emits_expected_rules() -> None:
    entries = [
        (
            "5",
            Paragraph(
                para_id="para_page",
                page_span={"start": 3, "end": 3},
                role="Body",
                text="",
                evidence_pointer={"source_block_ids": ["b_page"]},
            ),
        ),
        (
            "Check for updates",
            Paragraph(
                para_id="para_updates",
                page_span={"start": 3, "end": 3},
                role="Body",
                text="",
                evidence_pointer={"source_block_ids": ["b_updates"]},
            ),
        ),
        (
            "1 Department of Radiology, Example University, New York, NY 10016",
            Paragraph(
                para_id="para_aff",
                page_span={"start": 3, "end": 3},
                role="Body",
                text="",
                evidence_pointer={"source_block_ids": ["b_aff"]},
            ),
        ),
        (
            "Narrative line that should stay.",
            Paragraph(
                para_id="para_body",
                page_span={"start": 3, "end": 3},
                role="Body",
                text="",
                evidence_pointer={"source_block_ids": ["b_body"]},
            ),
        ),
    ]

    filtered_entries, suppressions = suppress_non_narrative_main_body_entries(entries, doc_id="doc_unit")
    kept_texts = [text for text, _ in filtered_entries]
    assert kept_texts == ["Narrative line that should stay."]

    rule_ids = {str(row.get("rule_id", "")) for row in suppressions}
    assert rule_ids == {"page_number_only", "check_for_updates_banner", "affiliation_address_line"}
    for row in suppressions:
        assert row["doc_id"] == "doc_unit"
        assert isinstance(row["para_id"], str)
        assert isinstance(row["source_block_ids"], list)
        assert row["normalized_text"]
        assert row["observed_at"]


def test_chen2024_struct07_suppression_preserves_numbered_narrative_lines() -> None:
    entries = [
        (
            "5",
            Paragraph(
                para_id="para_page",
                page_span={"start": 3, "end": 3},
                role="Body",
                text="",
                evidence_pointer={"source_block_ids": ["b_page"]},
            ),
        ),
        (
            "Check for updates Rotator cuff tears",
            Paragraph(
                para_id="para_banner",
                page_span={"start": 3, "end": 3},
                role="Body",
                text="",
                evidence_pointer={"source_block_ids": ["b_banner"]},
            ),
        ),
        (
            "1 Department of Radiology, Example University, New York, NY 10016",
            Paragraph(
                para_id="para_aff",
                page_span={"start": 3, "end": 3},
                role="Body",
                text="",
                evidence_pointer={"source_block_ids": ["b_aff"]},
            ),
        ),
        (
            "The 5-year follow-up cohort retained 38 patients with stable outcomes.",
            Paragraph(
                para_id="para_keep_numeric",
                page_span={"start": 3, "end": 3},
                role="Body",
                text="",
                evidence_pointer={"source_block_ids": ["b_keep_numeric"]},
            ),
        ),
    ]

    filtered_entries, suppressions = suppress_non_narrative_main_body_entries(entries, doc_id="chen2024_struct07")
    kept_texts = [text for text, _ in filtered_entries]
    assert kept_texts == ["The 5-year follow-up cohort retained 38 patients with stable outcomes."]

    suppressed_rules = {str(item.get("rule_id", "")) for item in suppressions}
    assert suppressed_rules == {"page_number_only", "check_for_updates_banner", "affiliation_address_line"}
    assert all(item.get("original_text") != kept_texts[0] for item in suppressions)


def test_single_column_layout_separates_general_metadata_from_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "single_column_meta"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [70.0, 70.0, 520.0, 130.0],
            "text": "Cross Journal MRI Cohort Analysis",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 18.0, "is_bold": True},
        },
        {
            "block_id": "p1_auth",
            "page": 1,
            "bbox_pt": [80.0, 145.0, 520.0, 180.0],
            "text": "Jane Doe 1 John Roe 2",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_aff",
            "page": 1,
            "bbox_pt": [80.0, 190.0, 520.0, 240.0],
            "text": "1 Department of Radiology, Example University, New York, NY",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_kw",
            "page": 1,
            "bbox_pt": [80.0, 250.0, 520.0, 278.0],
            "text": "Keywords: magnetic resonance imaging, tendon, cohort",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_doi",
            "page": 1,
            "bbox_pt": [80.0, 282.0, 520.0, 305.0],
            "text": "DOI: 10.1148/example.2026.10101",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_recv",
            "page": 1,
            "bbox_pt": [80.0, 309.0, 520.0, 336.0],
            "text": "Received January 3, 2026; accepted February 2, 2026",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_jmeta",
            "page": 1,
            "bbox_pt": [80.0, 24.0, 520.0, 48.0],
            "text": "Journal of Imaging Vol 12 Issue 4 Pages 200-214",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 360.0, 520.0, 500.0],
            "text": "This narrative paragraph reports the generalized findings and remains in the body section.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]

    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_auth": "Body",
            "p1_aff": "Body",
            "p1_kw": "Body",
            "p1_doi": "Body",
            "p1_recv": "Body",
            "p1_jmeta": "Body",
            "p1_body": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    _ = run_paragraphs(run_dir)

    cleaned_rows = _read_jsonl(text_dir / "blocks_clean.jsonl")
    by_id = {str(row.get("block_id", "")): row for row in cleaned_rows}
    assert by_id["p1_kw"]["clean_role"] == "keywords"
    assert by_id["p1_doi"]["clean_role"] == "doi"
    assert by_id["p1_recv"]["clean_role"] == "received"
    assert by_id["p1_jmeta"]["clean_role"] == "journal_meta"

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    assert "## Document Metadata" in clean_doc
    assert "### Keywords" in clean_doc
    assert "### DOI" in clean_doc
    assert "### Submission Timeline" in clean_doc
    main_body_text = clean_doc.split("## Main Body", maxsplit=1)[-1]
    assert "Keywords:" not in main_body_text
    assert "DOI:" not in main_body_text


def test_two_column_layout_keeps_section_assembly_stable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "two_column_layout"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 500.0, 120.0],
            "text": "Generalized Processing Across Layouts",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 17.0, "is_bold": True},
        },
        {
            "block_id": "p1_h1",
            "page": 1,
            "bbox_pt": [70.0, 150.0, 270.0, 180.0],
            "text": "Methods",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_lbody",
            "page": 1,
            "bbox_pt": [70.0, 195.0, 280.0, 330.0],
            "text": "Left column narrative body with reproducible preprocessing details.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_rbody",
            "page": 1,
            "bbox_pt": [320.0, 200.0, 540.0, 345.0],
            "text": "Right column narrative body with model evaluation details.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p2_h1",
            "page": 2,
            "bbox_pt": [70.0, 120.0, 270.0, 150.0],
            "text": "METHODS",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p2_body",
            "page": 2,
            "bbox_pt": [70.0, 170.0, 540.0, 290.0],
            "text": "Follow-up narrative paragraph confirms cross-page section continuity.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_aff",
            "page": 1,
            "bbox_pt": [70.0, 345.0, 540.0, 390.0],
            "text": "1 Department of Biomedical Engineering, Example Institute",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]

    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_h1": "Heading",
            "p1_lbody": "Body",
            "p1_rbody": "Body",
            "p1_aff": "Body",
        },
        2: {
            "p2_h1": "Heading",
            "p2_body": "Body",
        },
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    assert clean_doc.count("### Methods") == 1
    main_body = clean_doc.split("## Main Body", maxsplit=1)[-1]
    assert "Left column narrative body" in main_body
    assert "Right column narrative body" in main_body
    assert "Department of Biomedical Engineering" not in main_body


def test_two_column_layout_prefers_column_continuity_over_row_interleave(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "two_column_continuity"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Column Continuity Stress Test",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 17.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [70.0, 140.0, 280.0, 170.0],
            "text": "4.2. Conductive Polymers",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_top_full",
            "page": 1,
            "bbox_pt": [70.0, 178.0, 540.0, 214.0],
            "text": "This opening context spans both columns before detailed discussion.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_l1",
            "page": 1,
            "bbox_pt": [70.0, 220.0, 280.0, 270.0],
            "text": "Left sequence A describes polymer backbone characteristics.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_r1",
            "page": 1,
            "bbox_pt": [320.0, 228.0, 540.0, 278.0],
            "text": "Right sequence A reports electrical conductivity measurements.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_l2",
            "page": 1,
            "bbox_pt": [70.0, 282.0, 280.0, 332.0],
            "text": "Left sequence B covers stability and processability trends.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_r2",
            "page": 1,
            "bbox_pt": [320.0, 292.0, 540.0, 342.0],
            "text": "Right sequence B summarizes in vitro assay outcomes.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_bottom_full",
            "page": 1,
            "bbox_pt": [70.0, 356.0, 540.0, 396.0],
            "text": "This closing context also spans both columns after the detailed block.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]

    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_h": "Heading",
            "p1_top_full": "Body",
            "p1_l1": "Body",
            "p1_r1": "Body",
            "p1_l2": "Body",
            "p1_r2": "Body",
            "p1_bottom_full": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    idx_top = main_body.find("opening context spans both columns")
    idx_l1 = main_body.find("Left sequence A")
    idx_l2 = main_body.find("Left sequence B")
    idx_r1 = main_body.find("Right sequence A")
    idx_r2 = main_body.find("Right sequence B")
    idx_bottom = main_body.find("closing context also spans both columns")

    assert min(idx_top, idx_l1, idx_l2, idx_r1, idx_r2, idx_bottom) >= 0
    assert idx_l1 < idx_l2 < idx_r1 < idx_r2
    assert idx_top < idx_bottom


def test_multiline_author_extraction_is_complete_with_superscript_patterns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "multiline_authors"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Rotator Cable Outcome Validation",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 17.0, "is_bold": True},
        },
        {
            "block_id": "p1_auth_a",
            "page": 1,
            "bbox_pt": [80.0, 135.0, 520.0, 165.0],
            "text": "Soterios Gyftopoulos1 Jenny Bencardino1 Gregory Nevsky1",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_auth_b",
            "page": 1,
            "bbox_pt": [80.0, 170.0, 520.0, 205.0],
            "text": "Yousef Soofi3 Panna Desai3 Laith Jazrawi2 Michael P. Recht1",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_aff_1",
            "page": 1,
            "bbox_pt": [80.0, 215.0, 520.0, 250.0],
            "text": "1 Department of Radiology, NYU Langone Medical Center, New York, NY",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_aff_2",
            "page": 1,
            "bbox_pt": [80.0, 255.0, 520.0, 285.0],
            "text": "2 Department of Orthopaedic Surgery, NYU Langone Medical Center, New York, NY",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 340.0, 520.0, 470.0],
            "text": "This is the narrative body content and should remain in main body only.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_auth_a": "Body",
            "p1_auth_b": "Body",
            "p1_aff_1": "Body",
            "p1_aff_2": "Body",
            "p1_body": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()

    assert "## Authors" in clean_doc
    for expected_name in [
        "Soterios Gyftopoulos",
        "Jenny Bencardino",
        "Gregory Nevsky",
        "Yousef Soofi",
        "Panna Desai",
        "Laith Jazrawi",
        "Michael P. Recht",
    ]:
        assert expected_name in clean_doc

    main_body = clean_doc.split("## Main Body", maxsplit=1)[-1]
    assert "Soterios Gyftopoulos" not in main_body
    assert "Michael P. Recht" not in main_body


def test_metadata_family_never_leaks_into_main_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "metadata_leakage_gate"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Cross-Layout Narrative Integrity",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_author",
            "page": 1,
            "bbox_pt": [80.0, 135.0, 520.0, 165.0],
            "text": "Jane Doe1 John Roe2",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_aff",
            "page": 1,
            "bbox_pt": [80.0, 175.0, 520.0, 220.0],
            "text": "1 Department of Imaging, Example University Hospital",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_kw",
            "page": 1,
            "bbox_pt": [80.0, 225.0, 520.0, 250.0],
            "text": "Keywords: shoulder MRI, tendon, anatomy",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_doi",
            "page": 1,
            "bbox_pt": [80.0, 255.0, 520.0, 278.0],
            "text": "DOI: 10.1000/example.999",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_jcite",
            "page": 1,
            "bbox_pt": [80.0, 282.0, 520.0, 304.0],
            "text": "AJR 2013; 200:1101-1105",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_issn",
            "page": 1,
            "bbox_pt": [80.0, 306.0, 520.0, 326.0],
            "text": "0361-803X/13/2005-1101",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 330.0, 520.0, 470.0],
            "text": "This body paragraph provides analysis and should remain the only narrative in main body.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_author": "Body",
            "p1_aff": "Body",
            "p1_kw": "Body",
            "p1_doi": "Body",
            "p1_jcite": "Body",
            "p1_issn": "Body",
            "p1_body": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = clean_doc.split("## Main Body", maxsplit=1)[-1]

    assert "Jane Doe" not in main_body
    assert "Department of Imaging" not in main_body
    assert "Keywords:" not in main_body
    assert "DOI:" not in main_body
    assert "AJR 2013; 200:1101-1105" not in main_body
    assert "0361-803X/13/2005-1101" not in main_body
    assert "only narrative in main body" in main_body


def test_affiliations_and_document_metadata_are_stably_partitioned(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "metadata_partition"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Stable Partitioning Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_aff_1",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 520.0, 185.0],
            "text": "1 Department of Radiology, Example Medical Center",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_aff_2",
            "page": 1,
            "bbox_pt": [80.0, 190.0, 520.0, 225.0],
            "text": "2 Department of Pathology, Example Medical Center",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_received",
            "page": 1,
            "bbox_pt": [80.0, 230.0, 520.0, 255.0],
            "text": "Received January 1, 2026; accepted February 1, 2026",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_kw",
            "page": 1,
            "bbox_pt": [80.0, 258.0, 520.0, 283.0],
            "text": "Keywords: tendon, cable, MRI",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_doi",
            "page": 1,
            "bbox_pt": [80.0, 286.0, 520.0, 310.0],
            "text": "DOI: 10.1148/example.12345",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 340.0, 520.0, 470.0],
            "text": "Narrative paragraph remains in the body while metadata stays partitioned.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_aff_1": "Body",
            "p1_aff_2": "Body",
            "p1_received": "Body",
            "p1_kw": "Body",
            "p1_doi": "Body",
            "p1_body": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()

    assert "## Affiliations" in clean_doc
    affiliations = _section_text(clean_doc, "Affiliations")
    assert "Department of Radiology" in affiliations
    assert "Department of Pathology" in affiliations
    assert "DOI:" not in affiliations
    assert "Keywords:" not in affiliations

    assert "## Document Metadata" in clean_doc
    metadata_block = _section_text(clean_doc, "Document Metadata")
    assert "DOI: 10.1148/example.12345" in metadata_block
    assert "Keywords: tendon, cable, MRI" in metadata_block
    assert "Department of Radiology" not in metadata_block


def test_figures_section_rejects_long_body_like_caption_text(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "caption_admission_gate"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    long_narrative = (
        "There is no consensus in terms of MRI appearance and this paragraph describes methods, "
        "results, and discussion in a long narrative body style that should not be listed as a caption. "
        "It continues with additional prose to exceed caption-like structure and readability constraints."
    )
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Caption Gate Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 340.0, 520.0, 470.0],
            "text": "Main narrative body paragraph remains in body section.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_fig_bad",
            "page": 1,
            "bbox_pt": [80.0, 520.0, 520.0, 620.0],
            "text": long_narrative,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_fig_ok",
            "page": 1,
            "bbox_pt": [80.0, 625.0, 520.0, 670.0],
            "text": "Fig. 2 - MRI gross anatomic correlation in sagittal plane.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body": "Body",
            "p1_fig_bad": "FigureCaption",
            "p1_fig_ok": "FigureCaption",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    figures_section = _section_text(clean_doc, "Figures and Tables")
    assert "Fig. 2 - MRI gross anatomic correlation" in figures_section
    assert long_narrative not in figures_section


def test_abstract_content_is_not_repeated_in_main_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "abstract_dedup_gate"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    abstract_like = (
        "OBJECTIVE. Evaluate signal characteristics of the rotator cable in MRI. "
        "MATERIALS AND METHODS. We analyzed forty studies with consensus review."
    )
    duplicated_body = abstract_like + " Additional trailing details appear in source extraction but should be deduplicated."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Abstract Dedup Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_abs",
            "page": 1,
            "bbox_pt": [80.0, 160.0, 520.0, 250.0],
            "text": abstract_like,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_dup",
            "page": 1,
            "bbox_pt": [80.0, 320.0, 520.0, 440.0],
            "text": duplicated_body,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 445.0, 520.0, 520.0],
            "text": "Independent methods paragraph with non-overlapping body content.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_abs": "Body", "p1_dup": "Body", "p1_body": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")
    assert "OBJECTIVE. Evaluate signal characteristics" not in main_body
    assert "Independent methods paragraph" in main_body


def test_reference_entries_are_normalized_and_prefix_stripped(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "reference_normalization"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    merged_ref = (
        "References 1. Clark JM, Harryman DT II. Tendons and capsule. J Bone Joint Surg Am 1992; 74:713-725. "
        "2. Burkhart SS. Suspension bridge model. Clin Orthop Relat Res 1992; 284:144-152."
    )
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Reference Normalization Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 220.0, 520.0, 300.0],
            "text": "Body paragraph remains present.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_ref",
            "page": 1,
            "bbox_pt": [80.0, 610.0, 520.0, 710.0],
            "text": merged_ref,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_body": "Body", "p1_ref": "ReferenceList"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    refs = _section_text(clean_doc, "References")
    assert "References 1." not in refs
    assert "1. Clark JM, Harryman DT II." in refs
    assert "2. Burkhart SS." in refs


def test_main_body_drops_leading_orphan_continuation_fragment(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "orphan_fragment_guard"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    orphan_fragment = "don (Fig. 6). The posterior insertion was not clearly visualized on arthroscopy."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Cross Column Flow Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_orphan",
            "page": 1,
            "bbox_pt": [350.0, 130.0, 540.0, 180.0],
            "text": orphan_fragment,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_main",
            "page": 1,
            "bbox_pt": [350.0, 220.0, 540.0, 420.0],
            "text": "Results Cadaveric Study. A linear band of hypointense signal intensity was found and described in coherent opening narrative.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_orphan": "Body", "p1_main": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")
    first_body_paragraph = ""
    for line in main_body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("###"):
            continue
        first_body_paragraph = stripped
        break

    assert "don (Fig. 6)" not in main_body
    assert "Results Cadaveric Study" in main_body
    assert first_body_paragraph
    assert re.match(r"^[A-Z]", first_body_paragraph) is not None


def test_body_boundary_repair_merges_hyphen_wrap_fragments(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "body_hyphen_repair"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Boundary Repair Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_b1",
            "page": 1,
            "bbox_pt": [80.0, 210.0, 520.0, 300.0],
            "text": "These observations were consistent on MR arthrograph-",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_b2",
            "page": 1,
            "bbox_pt": [80.0, 305.0, 520.0, 390.0],
            "text": "ic images and matched gross anatomic findings.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_b1": "Body", "p1_b2": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")
    assert "arthrograph-" not in main_body
    assert "arthrographic images" in main_body


def test_body_boundary_repair_does_not_break_semantic_hyphen_tokens(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "semantic_hyphen_guard"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Semantic Hyphen Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_b1",
            "page": 1,
            "bbox_pt": [80.0, 210.0, 520.0, 290.0],
            "text": "A double-blind protocol used a 3-T scanner for acquisition.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_b2",
            "page": 1,
            "bbox_pt": [80.0, 300.0, 520.0, 380.0],
            "text": "Results remained stable across independent cohorts.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_b1": "Body", "p1_b2": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")
    assert "double-blind" in main_body
    assert "3-T scanner" in main_body
    assert "doubleblind" not in main_body


def test_body_boundary_repair_does_not_merge_across_sections(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "no_cross_section_merge"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Section Merge Guard Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h_results",
            "page": 1,
            "bbox_pt": [80.0, 160.0, 300.0, 190.0],
            "text": "Results",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_res",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 270.0],
            "text": "Signal on MR arthrograph-",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_h_disc",
            "page": 1,
            "bbox_pt": [80.0, 300.0, 300.0, 330.0],
            "text": "Discussion",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_disc",
            "page": 1,
            "bbox_pt": [80.0, 340.0, 520.0, 420.0],
            "text": "ic interpretation was aligned with pathology review.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_h_results": "Heading",
            "p1_res": "Body",
            "p1_h_disc": "Heading",
            "p1_disc": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    results_section = _section_text(clean_doc, "Main Body")
    assert "arthrographic interpretation" not in results_section
    assert "Signal on MR arthrograph-" in clean_doc
    assert "ic interpretation was aligned" in clean_doc


def test_hyphen_wrap_merges_unicode_letter_tail_with_ligature(tmp_path: Path) -> None:
    """Test that hyphen-wrap merging works with Unicode letters including ligatures.
    
    For example: 'effe-' + 'tive' should merge to 'effective' when positions
    indicate continuity (same column, reasonable vertical gap).
    """
    from ingest.paragraphs import should_merge_hyphen_wrap
    
    # Standard case: 4+ Unicode letters (e.g., 'effect-' + 'tive')
    assert should_merge_hyphen_wrap("effect-", "tive results", prev_x_rel=0.3, next_x_rel=0.3, prev_y_rel=0.8, next_y_rel=0.2) is True
    
    # Ligature case: 'effe-' (partial) + 'ctive' should merge
    assert should_merge_hyphen_wrap("effe-", "ctive improvement", prev_x_rel=0.3, next_x_rel=0.3, prev_y_rel=0.8, next_y_rel=0.2) is True
    
    assert should_merge_hyphen_wrap("de-", "scription details", prev_x_rel=0.3, next_x_rel=0.65, prev_y_rel=0.8, next_y_rel=0.15) is True
    
    # Same column, reasonable gap
    assert should_merge_hyphen_wrap("consist-", "ent data", prev_x_rel=0.4, next_x_rel=0.4, prev_y_rel=0.7, next_y_rel=0.3) is True
    
    # Reject when next doesn't start with lowercase
    assert should_merge_hyphen_wrap("effect-", "Results", prev_x_rel=0.3, next_x_rel=0.3, prev_y_rel=0.8, next_y_rel=0.2) is False
    
    # Reject large vertical gap in same column
    assert should_merge_hyphen_wrap("effect-", "tive", prev_x_rel=0.3, next_x_rel=0.3, prev_y_rel=0.3, next_y_rel=0.9) is False


def test_hyphen_wrap_short_tail_merge_with_strict_continuity(tmp_path: Path) -> None:
    """Test that short-tail hyphen-wrap (2-3 letters) merges only under strict continuity.
    
    Short tails like 're-' should only merge when:
    1. Left-to-right column swap (left column bottom -> right column top)
    2. Vertical wrap-back (next starts above where previous ended)
    3. Next starts with lowercase letter
    """
    from ingest.paragraphs import should_merge_hyphen_wrap
    
    # Short tail 're-' should merge with 'cruitment' only under strict left->right column swap
    # prev in left column (x_rel=0.3), next in right column (x_rel=0.65)
    # prev at bottom (y_rel=0.8), next at top (y_rel=0.15)
    assert should_merge_hyphen_wrap("re-", "cruitment process", prev_x_rel=0.3, next_x_rel=0.65, prev_y_rel=0.8, next_y_rel=0.15) is True
    
    # Short tail 'de-' should merge with 'novo analysis' under strict continuity
    assert should_merge_hyphen_wrap("de-", "novo synthesis", prev_x_rel=0.25, next_x_rel=0.7, prev_y_rel=0.85, next_y_rel=0.1) is True
    
    # Reject: short tail but same column (no swap)
    assert should_merge_hyphen_wrap("re-", "cruitment", prev_x_rel=0.3, next_x_rel=0.3, prev_y_rel=0.8, next_y_rel=0.2) is False
    
    # Reject: short tail with right column but no vertical wrap-back
    assert should_merge_hyphen_wrap("re-", "cruitment", prev_x_rel=0.3, next_x_rel=0.65, prev_y_rel=0.3, next_y_rel=0.5) is False
    
    # Reject: short tail but next starts with uppercase
    assert should_merge_hyphen_wrap("re-", "Recruitment", prev_x_rel=0.3, next_x_rel=0.65, prev_y_rel=0.8, next_y_rel=0.15) is False
    
    # Reject: short tail 're-' but no column swap - fails strict continuity
    assert should_merge_hyphen_wrap("re-", "search results", prev_x_rel=0.4, next_x_rel=0.4, prev_y_rel=0.7, next_y_rel=0.2) is False


def test_hyphen_wrap_acronym_numeric_and_same_row_cross_gutter_cases() -> None:
    from ingest.paragraphs import should_merge_hyphen_wrap

    assert (
        should_merge_hyphen_wrap(
            "upregulation of BMP-",
            "2, BMP-4, IGF-2 pathways were detected",
            prev_x_rel=0.72,
            next_x_rel=0.74,
            prev_y_rel=0.78,
            next_y_rel=0.86,
        )
        is True
    )

    assert (
        should_merge_hyphen_wrap(
            "there remains some contro-",
            "versy regarding treatment effects",
            prev_x_rel=0.30,
            next_x_rel=0.78,
            prev_y_rel=0.90,
            next_y_rel=0.90,
        )
        is True
    )


def test_lowercase_continuation_allows_conservative_cross_page_merge() -> None:
    from ingest.paragraphs import should_merge_lowercase_continuation

    assert (
        should_merge_lowercase_continuation(
            "In short, the distinctive material architecture enables robust transport",
            "physicochemical properties of graphene broaden biomedical use",
            prev_x_rel=0.78,
            next_x_rel=0.22,
            prev_y_rel=0.83,
            next_y_rel=0.20,
            prev_page=10,
            next_page=11,
        )
        is True
    )

    assert (
        should_merge_lowercase_continuation(
            "In short, the distinctive material architecture enables robust transport",
            "row 1 2 3 4 5",
            prev_x_rel=0.78,
            next_x_rel=0.22,
            prev_y_rel=0.83,
            next_y_rel=0.20,
            prev_page=10,
            next_page=11,
        )
        is False
    )


def test_trim_numeric_leading_continuation_sentence_preserves_remainder() -> None:
    from ingest.paragraphs import trim_numeric_leading_continuation_sentence

    text = (
        "2, BMP-4, IGF-2, and VEGF in osteoblasts. "
        "Notably, continuous ES has shown stronger osteogenic effects."
    )
    trimmed = trim_numeric_leading_continuation_sentence(text)
    assert trimmed == "Notably, continuous ES has shown stronger osteogenic effects."


def test_merge_citation_tail_continuation_is_conservative() -> None:
    from ingest.paragraphs import merge_citation_tail_continuation_paragraphs, should_merge_citation_tail_continuation

    previous = (
        "These signals facilitated cell recruitment and promoted cartilage formation. "
        "Damaraju et al. [ 159c ]"
    )
    next_text = "developed a pliable 3D electrospun scaffold with improved piezoelectricity."
    assert should_merge_citation_tail_continuation(previous, next_text) is True

    reference_like_previous = "1. Damaraju et al. [ 159c ]"
    assert should_merge_citation_tail_continuation(reference_like_previous, next_text) is False

    paragraphs = [
        previous,
        next_text,
        "These findings support further biomaterial optimization.",
    ]
    merged = merge_citation_tail_continuation_paragraphs(paragraphs)
    assert len(merged) == 2
    assert merged[0].endswith("developed a pliable 3D electrospun scaffold with improved piezoelectricity.")


@pytest.mark.parametrize(
    "layout_name, blocks, role_labels, nuisance_ids, kept_ids",
    [
        (
            "side_watermark",
            {
                "p1_wm": {
                    "block_id": "p1_wm",
                    "page": 1,
                    "bbox_pt": [-2.0, 100.0, 9.0, 700.0],
                    "text": "Downloaded from sample by 1.2.3.4",
                    "is_header_footer_candidate": False,
                },
                "p2_wm": {
                    "block_id": "p2_wm",
                    "page": 2,
                    "bbox_pt": [-2.0, 90.0, 9.0, 690.0],
                    "text": "Downloaded from sample by 8.8.8.8",
                    "is_header_footer_candidate": False,
                },
                "p1_body": {
                    "block_id": "p1_body",
                    "page": 1,
                    "bbox_pt": [70.0, 220.0, 520.0, 360.0],
                    "text": "Normal body paragraph one for the side watermark layout.",
                    "is_header_footer_candidate": False,
                },
                "p2_body": {
                    "block_id": "p2_body",
                    "page": 2,
                    "bbox_pt": [70.0, 220.0, 520.0, 360.0],
                    "text": "Normal body paragraph two for the side watermark layout.",
                    "is_header_footer_candidate": False,
                },
            },
            {1: {"p1_wm": "Body", "p1_body": "Body"}, 2: {"p2_wm": "Body", "p2_body": "Body"}},
            {"p1_wm", "p2_wm"},
            {"p1_body", "p2_body"},
        ),
        (
            "heavy_footer",
            {
                "p1_f": {
                    "block_id": "p1_f",
                    "page": 1,
                    "bbox_pt": [60.0, 742.0, 540.0, 774.0],
                    "text": "Journal of Imaging 2026;12(4):200-214",
                    "is_header_footer_candidate": True,
                },
                "p2_f": {
                    "block_id": "p2_f",
                    "page": 2,
                    "bbox_pt": [60.0, 742.0, 540.0, 774.0],
                    "text": "Journal of Imaging 2026;12(4):200-214",
                    "is_header_footer_candidate": True,
                },
                "p3_f": {
                    "block_id": "p3_f",
                    "page": 3,
                    "bbox_pt": [60.0, 742.0, 540.0, 774.0],
                    "text": "Journal of Imaging 2026;12(4):200-214",
                    "is_header_footer_candidate": True,
                },
                "p1_body": {
                    "block_id": "p1_body",
                    "page": 1,
                    "bbox_pt": [70.0, 180.0, 520.0, 320.0],
                    "text": "Primary body paragraph page one for heavy footer layout.",
                    "is_header_footer_candidate": False,
                },
                "p2_body": {
                    "block_id": "p2_body",
                    "page": 2,
                    "bbox_pt": [70.0, 180.0, 520.0, 320.0],
                    "text": "Primary body paragraph page two for heavy footer layout.",
                    "is_header_footer_candidate": False,
                },
                "p3_body": {
                    "block_id": "p3_body",
                    "page": 3,
                    "bbox_pt": [70.0, 180.0, 520.0, 320.0],
                    "text": "Primary body paragraph page three for heavy footer layout.",
                    "is_header_footer_candidate": False,
                },
            },
            {
                1: {"p1_f": "Body", "p1_body": "Body"},
                2: {"p2_f": "Body", "p2_body": "Body"},
                3: {"p3_f": "Body", "p3_body": "Body"},
            },
            {"p1_f", "p2_f", "p3_f"},
            {"p1_body", "p2_body", "p3_body"},
        ),
    ],
)
def test_nuisance_filtering_handles_side_watermark_and_heavy_footer(
    layout_name: str,
    blocks: dict[str, dict[str, object]],
    role_labels: dict[int, dict[str, str]],
    nuisance_ids: set[str],
    kept_ids: set[str],
) -> None:
    kept_blocks, annotated = classify_clean_blocks(blocks, role_labels)
    by_id = {str(row.get("block_id", "")): row for row in annotated}

    for block_id in nuisance_ids:
        assert by_id[block_id]["is_nuisance"] is True, f"{layout_name}: expected nuisance {block_id}"

    for block_id in kept_ids:
        assert by_id[block_id]["is_nuisance"] is False, f"{layout_name}: expected kept {block_id}"


def test_llm_refinement_drops_noise_from_main_body_with_mocked_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run" / "llm_drop_noise"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "General Narrative Robustness",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body_1",
            "page": 1,
            "bbox_pt": [80.0, 210.0, 520.0, 300.0],
            "text": "This paragraph presents reproducible acquisition and preprocessing details.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_noise",
            "page": 1,
            "bbox_pt": [80.0, 310.0, 520.0, 360.0],
            "text": "Click here to subscribe and receive article alerts and navigation tips.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_body_2",
            "page": 1,
            "bbox_pt": [80.0, 370.0, 520.0, 460.0],
            "text": "This paragraph reports stable findings across independent validation folds.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body_1": "Body",
            "p1_noise": "Body",
            "p1_body_2": "Body",
        }
    }
    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    monkeypatch.setenv("PARAGRAPHS_LLM_REFINE", "1")
    monkeypatch.setenv("PARAGRAPHS_LLM_MAX_CHUNKS", "1")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    def _mock_drop_noise(prompt: str, config: object) -> tuple[str, dict[str, object]]:
        payload = json.loads(prompt)
        rows = payload.get("window_rows", [])
        target_ids = set(payload.get("target_para_ids", []))
        decisions = []
        for row in rows:
            para_id = str(row.get("para_id", ""))
            if para_id not in target_ids:
                continue
            text = str(row.get("text", "")).lower()
            is_noise = "subscribe" in text and "alerts" in text
            decisions.append(
                {
                    "para_id": para_id,
                    "keep": not is_noise,
                    "continuity_group": "g_noise" if is_noise else "g1",
                    "confidence": 0.96,
                    "reason": "promotional/navigation noise" if is_noise else "narrative",
                }
            )
        return (
            json.dumps({"decisions": decisions}),
            {
                "model": getattr(config, "model", "mock"),
                "endpoint": getattr(config, "endpoint", "mock"),
                "success": True,
                "error_type": "none",
                "http_status": 200,
                "prompt_chars": len(prompt),
                "response_chars": 300,
            },
        )

    monkeypatch.setattr(paragraphs_mod, "call_siliconflow_for_paragraphs", _mock_drop_noise)

    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")
    assert "subscribe and receive article alerts" not in main_body
    assert "reproducible acquisition" in main_body
    assert "stable findings across independent validation folds" in main_body

    telemetry_path = run_dir / "qa" / "paragraphs_llm_calls.jsonl"
    assert telemetry_path.exists()
    telemetry_rows = _read_jsonl(telemetry_path)
    assert telemetry_rows
    assert telemetry_rows[0]["stage"] == "paragraphs"
    assert any(str(row.get("step", "")) == "llm_refine_applied_summary" for row in telemetry_rows)


def test_llm_refinement_preserves_order_when_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run" / "llm_order_stable"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Continuity Ordering Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_a",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 260.0],
            "text": "First narrative paragraph begins with setup and context.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_b",
            "page": 1,
            "bbox_pt": [80.0, 265.0, 520.0, 330.0],
            "text": "Second narrative paragraph explains method and rationale.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_c",
            "page": 1,
            "bbox_pt": [80.0, 335.0, 520.0, 400.0],
            "text": "Third narrative paragraph summarizes results and implication.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_a": "Body", "p1_b": "Body", "p1_c": "Body"}}
    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    monkeypatch.setenv("PARAGRAPHS_LLM_REFINE", "1")
    monkeypatch.setenv("PARAGRAPHS_LLM_MAX_CHUNKS", "1")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    def _mock_keep_all(prompt: str, config: object) -> tuple[str, dict[str, object]]:
        payload = json.loads(prompt)
        target_ids = [str(value) for value in payload.get("target_para_ids", [])]
        return (
            json.dumps(
                {
                    "decisions": [
                        {
                            "para_id": para_id,
                            "keep": True,
                            "continuity_group": "g_main",
                            "confidence": 0.91,
                            "reason": "continuous narrative",
                        }
                        for para_id in target_ids
                    ]
                }
            ),
            {
                "model": "mock",
                "endpoint": "mock",
                "success": True,
                "error_type": "none",
                "http_status": 200,
                "prompt_chars": len(prompt),
                "response_chars": 200,
            },
        )

    monkeypatch.setattr(paragraphs_mod, "call_siliconflow_for_paragraphs", _mock_keep_all)

    _ = run_paragraphs(run_dir)
    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    idx_a = main_body.find("First narrative paragraph")
    idx_b = main_body.find("Second narrative paragraph")
    idx_c = main_body.find("Third narrative paragraph")
    assert idx_a >= 0 and idx_b >= 0 and idx_c >= 0
    assert idx_a < idx_b < idx_c


def test_llm_refinement_falls_back_when_parse_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run" / "llm_fallback_parse_error"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Fallback Reliability Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 220.0, 520.0, 300.0],
            "text": "Narrative content should remain when LLM output is unavailable.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_body": "Body"}}
    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    monkeypatch.setenv("PARAGRAPHS_LLM_REFINE", "1")
    monkeypatch.setenv("PARAGRAPHS_LLM_MAX_CHUNKS", "1")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setattr(
        paragraphs_mod,
        "call_siliconflow_for_paragraphs",
        lambda prompt, config: (
            "not-json-response",
            {
                "model": getattr(config, "model", "mock"),
                "endpoint": getattr(config, "endpoint", "mock"),
                "success": True,
                "error_type": "none",
                "http_status": 200,
                "prompt_chars": len(prompt),
                "response_chars": 17,
            },
        ),
    )

    _ = run_paragraphs(run_dir)
    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")
    assert "Narrative content should remain" in main_body

    telemetry_path = run_dir / "qa" / "paragraphs_llm_calls.jsonl"
    telemetry_rows = _read_jsonl(telemetry_path)
    assert telemetry_rows
    assert any(row.get("parse_success") is False for row in telemetry_rows)


def test_llm_refinement_disabled_emits_skip_telemetry_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run" / "llm_disabled_telemetry"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Disabled Telemetry Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 220.0, 520.0, 300.0],
            "text": "Body paragraph should remain unchanged when LLM refine is disabled.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_body": "Body"}}
    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    monkeypatch.setenv("PARAGRAPHS_LLM_REFINE", "0")
    _ = run_paragraphs(run_dir)

    telemetry_path = run_dir / "qa" / "paragraphs_llm_calls.jsonl"
    assert telemetry_path.exists()
    telemetry_rows = _read_jsonl(telemetry_path)
    assert telemetry_rows
    assert any(str(row.get("step", "")) == "llm_refine_disabled" for row in telemetry_rows)


def test_table_noise_filter_drops_tabular_content_from_main_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_gate"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    table_noise_text = "Advantage Disadvantage Benefit Limitation"
    narrative_text = "The study results demonstrate significant improvements in patient outcomes."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Table Noise Gate Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body_good",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 300.0],
            "text": narrative_text,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_table_noise",
            "page": 1,
            "bbox_pt": [80.0, 310.0, 520.0, 350.0],
            "text": table_noise_text,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body_good": "Body",
            "p1_table_noise": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert narrative_text in main_body
    assert "Advantage Disadvantage" not in main_body


def test_table_noise_filter_preserves_nearby_narrative(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_narrative_preserve"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    units_text = "Scanner Field Strength 3 T 1.5 T 7 T"
    narrative_before = "We evaluated the imaging quality across different scanner configurations."
    narrative_after = "Results showed consistent signal-to-noise ratios across all field strengths."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Units Column Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_before",
            "page": 1,
            "bbox_pt": [80.0, 180.0, 520.0, 240.0],
            "text": narrative_before,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_units",
            "page": 1,
            "bbox_pt": [80.0, 250.0, 520.0, 290.0],
            "text": units_text,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_after",
            "page": 1,
            "bbox_pt": [80.0, 300.0, 520.0, 360.0],
            "text": narrative_after,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_before": "Body",
            "p1_units": "Body",
            "p1_after": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert narrative_before in main_body
    assert narrative_after in main_body
    assert "Scanner Field Strength" not in main_body
    assert "3 T 1.5 T" not in main_body


def test_table_noise_filter_handles_numeric_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_numeric"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    numeric_row = "Slice Thickness 1.0 1.5 2.0 3.0 mm"
    narrative = "The imaging protocol used varying slice thicknesses to optimize resolution."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Numeric Row Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 280.0],
            "text": narrative,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_numeric",
            "page": 1,
            "bbox_pt": [80.0, 290.0, 520.0, 320.0],
            "text": numeric_row,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body": "Body",
            "p1_numeric": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert narrative in main_body
    assert "Slice Thickness" not in main_body


def test_table_noise_filter_does_not_affect_normal_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_no_regression"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    normal_body = "The patient cohort consisted of forty subjects with confirmed diagnosis."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Normal Body Regression Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 300.0],
            "text": normal_body,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert normal_body in main_body
    assert "patient cohort" in main_body


def test_table_noise_filter_drops_long_continued_table(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_continued"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    continued_table = (
        "### Table 1. (Continued). Type Conductivity [S cm - 1] Advantages Disadvantages "
        "Applications References PPy 10 3 -5 x 10 4 Simple to manufacture High stability "
        "High conductivity PANi 10 2 -10 8 High conductivity low cost Outstanding mechanical "
        "properties Polythiophene (PT) 10 -1 -10 -4 Strong thermal stability High optical "
        "performance 10 -13 -10 -2 Outstanding acid and alkali resistance Good resistance to "
        "organic solvents High density polyethylene (HDPE) Polyphenylene sulfide (PPS)"
    )
    narrative = "The conductive polymer composites show promising applications in biomedical devices."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Long Table Continuation Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 280.0],
            "text": narrative,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_table_cont",
            "page": 1,
            "bbox_pt": [80.0, 290.0, 520.0, 450.0],
            "text": continued_table,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body": "Body",
            "p1_table_cont": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert narrative in main_body
    assert "Table 1. (Continued)" not in main_body
    assert "Advantages Disadvantages" not in main_body


def test_table_noise_filter_keeps_long_narrative_paragraph(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_long_narrative"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    long_narrative = (
        "The study investigated the effects of various conductive polymer composites on cell viability "
        "and proliferation rates. Results demonstrated that PPy-coated substrates showed enhanced "
        "neuronal attachment and growth compared to control groups. Statistical analysis revealed "
        "significant differences in metabolic activity between treatment groups at p less than 0.05. "
        "These findings suggest potential applications in neural tissue engineering and biosensing "
        "platforms. Further studies are needed to optimize coating parameters and long-term stability."
    )
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Long Narrative Retention Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 400.0],
            "text": long_narrative,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert "conductive polymer composites" in main_body
    assert "cell viability" in main_body
    assert "neuronal attachment" in main_body


def test_table_noise_filter_drops_citation_bracket_line(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_citation"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    citation_bracket = "[ 306 ]"
    narrative = "The scaffold showed excellent biocompatibility in vitro."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Citation Bracket Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 280.0],
            "text": narrative,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_cite",
            "page": 1,
            "bbox_pt": [80.0, 290.0, 520.0, 310.0],
            "text": citation_bracket,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body": "Body",
            "p1_cite": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert narrative in main_body
    assert "[ 306 ]" not in main_body


def test_table_noise_filter_drops_fragment_row(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_fragment"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    fragment_row = "Poor liquidity Osteogenic differentiation; Vascularization; Chondrocyte proliferation"
    narrative = "The composite materials demonstrated favorable mechanical properties."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Fragment Row Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 280.0],
            "text": narrative,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_frag",
            "page": 1,
            "bbox_pt": [80.0, 290.0, 520.0, 330.0],
            "text": fragment_row,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body": "Body",
            "p1_frag": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert narrative in main_body
    assert "Poor liquidity" not in main_body


def test_table_noise_filter_keeps_concise_narrative(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "table_noise_concise_narrative"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    concise_narrative = "MXene shows high specific capacity."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Concise Narrative Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 250.0],
            "text": concise_narrative,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_body": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    main_body = _section_text(clean_doc, "Main Body")

    assert "MXene shows high specific capacity" in main_body


def test_column_aware_merge_prevents_cross_column_merge(tmp_path: Path) -> None:
    """Cross-column adjacent fragments should NOT be merged."""
    from ingest.paragraphs import (
        should_merge_hyphen_wrap,
        should_merge_lowercase_continuation,
    )
    
    left_text = "The material shows excellent conductivity"
    right_text = "and mechanical strength"
    
    # Same y level but different x positions (left column vs right column)
    # left x_rel ~ 0.2, right x_rel ~ 0.8
    # With x_rel difference > 0.2, should NOT merge
    assert not should_merge_hyphen_wrap(left_text, right_text, 0.2, 0.8, 0.3, 0.3)
    assert not should_merge_lowercase_continuation(left_text, right_text, 0.2, 0.8, 0.3, 0.3)


def test_column_aware_merge_preserves_same_column_hyphen_wrap(tmp_path: Path) -> None:
    """Same-column hyphen wrap fragments should still merge."""
    from ingest.paragraphs import should_merge_hyphen_wrap
    
    prev_text = "The conductive polymer is charac-"
    next_text = "terized by high flexibility"
    
    # Same column (both x_rel ~ 0.25) and close y positions
    assert should_merge_hyphen_wrap(prev_text, next_text, 0.25, 0.28, 0.3, 0.32)


def test_column_aware_merge_preserves_same_column_lowercase(tmp_path: Path) -> None:
    """Same-column lowercase continuation should still merge."""
    from ingest.paragraphs import should_merge_lowercase_continuation
    
    prev_text = "The composite material demonstrates excellent electrical conductivity"
    next_text = "and superior mechanical properties"
    
    # Same column and close y positions
    assert should_merge_lowercase_continuation(prev_text, next_text, 0.25, 0.28, 0.3, 0.32)


def test_y_gap_guard_prevents_vertical_gap_merge(tmp_path: Path) -> None:
    """Large y-gap between paragraphs should prevent merge."""
    from ingest.paragraphs import should_merge_lowercase_continuation
    
    prev_text = "The conductive polymer is characterized by high flexibility and"
    next_text = "biocompatibility"
    
    # Same x but large y-gap (different sections) - gap > 0.4 threshold
    # prev at y=0.1, next at y=0.6 (gap = 0.5 > 0.4)
    assert not should_merge_lowercase_continuation(prev_text, next_text, 0.25, 0.25, 0.1, 0.6)


def test_section_numbering_preserved_in_clean_document(tmp_path: Path) -> None:
    """Section numbering should be preserved (1. Introduction, not . Introduction)."""
    from ingest.paragraphs import run_paragraphs
    from ingest.manifest import Manifest
    
    run_dir = tmp_path / "run" / "section_numbering"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)
    
    # Create blocks with numbered section headings
    blocks = [
        {
            "block_id": "p1_heading",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 100.0],
            "text": "1. Introduction",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 14.0, "is_bold": True},
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 120.0, 520.0, 200.0],
            "text": "Bone defects are prevalent worldwide.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p2_heading",
            "page": 2,
            "bbox_pt": [80.0, 70.0, 520.0, 100.0],
            "text": "2.1. Bioelectricity in Bone",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 13.0, "is_bold": True},
        },
        {
            "block_id": "p2_body",
            "page": 2,
            "bbox_pt": [80.0, 120.0, 520.0, 200.0],
            "text": "Natural bone exhibits piezoelectric properties.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    _write_jsonl(text_dir / "blocks_norm.jsonl", blocks)
    _write_jsonl(
        vision_dir / "merge_groups.jsonl",
        [{"page": page, "groups": []} for page in [1, 2]],
    )
    _write_jsonl(
        vision_dir / "confidence.jsonl",
        [{"page": page, "confidence_scores": {}} for page in [1, 2]],
    )
    _write_jsonl(
        vision_dir / "overlay.jsonl",
        [
            {
                "page": 1,
                "predictions": [
                    {"block_id": "p1_heading", "label": "Heading", "confidence": 0.95},
                    {"block_id": "p1_body", "label": "Body", "confidence": 0.95},
                ],
            },
            {
                "page": 2,
                "predictions": [
                    {"block_id": "p2_heading", "label": "Heading", "confidence": 0.95},
                    {"block_id": "p2_body", "label": "Body", "confidence": 0.95},
                ],
            },
        ],
    )
    
    from datetime import datetime, timezone
    manifest = Manifest(
        doc_id="section_numbering",
        input_pdf_path="test.pdf",
        input_pdf_sha256="a" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    run_paragraphs(run_dir, manifest)
    
    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    
    # Numbering should be preserved
    assert "### 1. Introduction" in clean_doc
    assert "### 2.1. Bioelectricity in Bone" in clean_doc
    # Should NOT have stripped numbering
    assert "### . Introduction" not in clean_doc
    assert "### . Bioelectricity in Bone" not in clean_doc


def test_front_matter_fragments_excluded_from_main_body(tmp_path: Path) -> None:
    """Abstract fragments and ORCID/DOI metadata should not appear in Main Body."""
    from ingest.paragraphs import run_paragraphs
    from ingest.manifest import Manifest
    
    run_dir = tmp_path / "run" / "front_matter_filter"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)
    
    # Create blocks including abstract fragments and metadata
    blocks = [
        {
            "block_id": "p1_abstract",
            "page": 1,
            "bbox_pt": [80.0, 200.0, 520.0, 280.0],
            "text": "The incidence of bone defects is increasing. Electroactive biomaterials are emphasized as a promising approach.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_orcid",
            "page": 1,
            "bbox_pt": [80.0, 290.0, 520.0, 310.0],
            "text": "The ORCID identification number(s) for the author(s) of this article can be found under https://doi.org/10.1002/test",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p2_heading",
            "page": 2,
            "bbox_pt": [80.0, 70.0, 520.0, 100.0],
            "text": "1. Introduction",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 14.0, "is_bold": True},
        },
        {
            "block_id": "p2_body",
            "page": 2,
            "bbox_pt": [80.0, 120.0, 520.0, 200.0],
            "text": "Bone defects represent a significant clinical challenge.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    _write_jsonl(text_dir / "blocks_norm.jsonl", blocks)
    _write_jsonl(
        vision_dir / "merge_groups.jsonl",
        [{"page": page, "groups": []} for page in [1, 2]],
    )
    _write_jsonl(
        vision_dir / "confidence.jsonl",
        [{"page": page, "confidence_scores": {}} for page in [1, 2]],
    )
    _write_jsonl(
        vision_dir / "overlay.jsonl",
        [
            {
                "page": 1,
                "predictions": [
                    {"block_id": "p1_abstract", "label": "Body", "confidence": 0.95},
                    {"block_id": "p1_orcid", "label": "Body", "confidence": 0.95},
                ],
            },
            {
                "page": 2,
                "predictions": [
                    {"block_id": "p2_heading", "label": "Heading", "confidence": 0.95},
                    {"block_id": "p2_body", "label": "Body", "confidence": 0.95},
                ],
            },
        ],
    )
    
    from datetime import datetime, timezone
    manifest = Manifest(
        doc_id="front_matter_filter",
        input_pdf_path="test.pdf",
        input_pdf_sha256="a" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    run_paragraphs(run_dir, manifest)
    
    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()
    
    # Abstract should be extracted
    assert "## Abstract / Objective" in clean_doc or "emphasized" in clean_doc
    # ORCID should NOT appear in Main Body
    main_body = clean_doc.split("## Main Body", maxsplit=1)[-1]
    assert "ORCID" not in main_body
    assert "doi.org" not in main_body
    # Abstract fragment with "is emphasized" should not appear in Main Body
    assert "is emphasized" not in main_body or clean_doc.index("is emphasized") < clean_doc.index("## Main Body")


def test_table_caption_filtered_from_main_body(tmp_path: Path) -> None:
    """Table captions like 'Table 1. Summary of...' should be filtered from Main Body."""
    from ingest.paragraphs import looks_like_table_noise
    
    # Table captions should be detected as noise
    assert looks_like_table_noise("Table 1. Summary of the representative conductive biomaterials")
    assert looks_like_table_noise("Table 2. (Continued).")
    assert looks_like_table_noise("Table 1 (Continued)")
    
    # Narrative text should NOT be detected as noise
    assert not looks_like_table_noise("Conductive biomaterials are widely used in tissue engineering.")
    assert not looks_like_table_noise("Table 1 shows the summary")  # Not a caption pattern


def test_column_aware_sorting_applied_at_paragraphs_stage(tmp_path: Path) -> None:
    """Column-aware sorting should be applied in build_neighbors, not just render."""
    from ingest.paragraphs import build_neighbors, Paragraph
    
    # Create paragraphs simulating two-column layout
    # Need enough paragraphs to trigger two-column detection (>= 4 per page)
    paragraphs = [
        # Left column top
        Paragraph(
            para_id="left_p1",
            page_span={"start": 1, "end": 1},
            role="Body",
            section_path=None,
            text="Left column first paragraph",
            evidence_pointer={"pages": [1], "bbox_union": [50.0, 100.0, 250.0, 150.0], "source_block_ids": []},
            neighbors={},
            confidence=0.95,
            provenance={"source": "test", "strategy": "test", "notes": ""},
        ),
        # Left column bottom
        Paragraph(
            para_id="left_p2",
            page_span={"start": 1, "end": 1},
            role="Body",
            section_path=None,
            text="Left column second paragraph",
            evidence_pointer={"pages": [1], "bbox_union": [50.0, 200.0, 250.0, 250.0], "source_block_ids": []},
            neighbors={},
            confidence=0.95,
            provenance={"source": "test", "strategy": "test", "notes": ""},
        ),
        # Right column top
        Paragraph(
            para_id="right_p1",
            page_span={"start": 1, "end": 1},
            role="Body",
            section_path=None,
            text="Right column first paragraph",
            evidence_pointer={"pages": [1], "bbox_union": [400.0, 100.0, 550.0, 150.0], "source_block_ids": []},
            neighbors={},
            confidence=0.95,
            provenance={"source": "test", "strategy": "test", "notes": ""},
        ),
        # Right column bottom
        Paragraph(
            para_id="right_p2",
            page_span={"start": 1, "end": 1},
            role="Body",
            section_path=None,
            text="Right column second paragraph",
            evidence_pointer={"pages": [1], "bbox_union": [400.0, 200.0, 550.0, 250.0], "source_block_ids": []},
            neighbors={},
            confidence=0.95,
            provenance={"source": "test", "strategy": "test", "notes": ""},
        ),
        # Full-width header (top of page)
        Paragraph(
            para_id="header",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=["Introduction"],
            text="1. Introduction",
            evidence_pointer={"pages": [1], "bbox_union": [50.0, 50.0, 550.0, 90.0], "source_block_ids": []},
            neighbors={},
            confidence=0.95,
            provenance={"source": "test", "strategy": "test", "notes": ""},
        ),
    ]
    
    # build_neighbors should apply column-aware sorting
    sorted_paras = build_neighbors(paragraphs)
    
    # Order should be: header (full-width top), left_p1, left_p2, right_p1, right_p2
    sorted_ids = [p.para_id for p in sorted_paras]
    assert sorted_ids[0] == "header", f"Expected header first, got {sorted_ids}"
    assert sorted_ids[1] == "left_p1", f"Expected left_p1 second, got {sorted_ids}"
    assert sorted_ids[2] == "left_p2", f"Expected left_p2 third, got {sorted_ids}"
    assert "right_p1" in sorted_ids[3:] or "right_p2" in sorted_ids[3:], f"Right column should come after left, got {sorted_ids}"
    
    # Neighbors should be set correctly
    assert sorted_paras[0].neighbors.get("next_para_id") == sorted_paras[1].para_id
    assert sorted_paras[1].neighbors.get("next_para_id") == sorted_paras[2].para_id


def test_section_leading_lowercase_fragment_sentence_is_trimmed_conservatively(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "section_lowercase_fragment_trim"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Lowercase Fragment Trim Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": "2.2. Bioelectricity in Bone",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 300.0],
            "text": "bone lies within the collagen matrix. The piezoelectric response remains stable under loading.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h": "Heading", "p1_body": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)

    _ = run_paragraphs(run_dir)

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_doc = f.read()

    section_text = _section_text(clean_doc, "Main Body")
    assert "### 2.2. Bioelectricity in Bone" in clean_doc
    assert "bone lies within the collagen matrix." not in section_text
    assert "The piezoelectric response remains stable under loading." in section_text


def test_section_leading_citation_prefix_is_trimmed_when_remainder_is_fresh(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "section_citation_prefix_trim"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Citation Prefix Trim Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h1",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": "2.2. Bioelectricity in Bone",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_b1",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 290.0],
            "text": "[ 30 ] Collagen molecules, possessing a triple helix structure, exhibit self-assembly under stress.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h1": "Heading", "p1_b1": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert "### 2.2. Bioelectricity in Bone" in clean_doc
    assert "[ 30 ] Collagen molecules" not in clean_doc
    assert "Collagen molecules, possessing a triple helix structure" in clean_doc


def test_section_leading_lowercase_continuation_can_merge_to_previous_section(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "section_lowercase_cross_merge"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    previous_tail = "Conductive channels provide stable signaling support for regenerative interfaces"
    leading_lowercase = "can be designed so that they directly impact endogenous stem cells near injured sites."
    retained_second = "Piezoelectric biomaterials are increasingly used for controlled stimulation."

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Cross-Section Merge Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h4",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 520.0, 182.0],
            "text": "4. Conductive Biomaterials",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_b4",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 270.0],
            "text": previous_tail,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_h5",
            "page": 1,
            "bbox_pt": [80.0, 285.0, 520.0, 317.0],
            "text": "5. Piezoelectric Biomaterials",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_b5a",
            "page": 1,
            "bbox_pt": [80.0, 330.0, 540.0, 395.0],
            "text": leading_lowercase,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_b5b",
            "page": 1,
            "bbox_pt": [80.0, 405.0, 540.0, 470.0],
            "text": retained_second,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_h4": "Heading",
            "p1_b4": "Body",
            "p1_h5": "Heading",
            "p1_b5a": "Body",
            "p1_b5b": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert previous_tail in clean_doc

    section5_match = re.search(
        r"^### 5\. Piezoelectric Biomaterials\n\n(.*?)(?=^### |^## |\Z)",
        clean_doc,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section5_match is not None
    section5_text = section5_match.group(1)
    assert leading_lowercase not in section5_text
    assert retained_second in section5_text


def test_section_leading_modal_sentence_then_citation_prefix_is_chained(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "section_modal_citation_chain"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    leading_modal = "can be designed so that they directly impact endogenous stem cells near injured sites."
    citation_body = "[ 30 ] Collagen molecules, possessing a triple helix structure, exhibit self-assembly under stress."

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Modal Citation Chain Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": "2.2. Bioelectricity in Bone",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 320.0],
            "text": f"{leading_modal} {citation_body}",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h": "Heading", "p1_body": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert leading_modal not in clean_doc
    assert "[ 30 ] Collagen molecules" not in clean_doc
    assert "Collagen molecules, possessing a triple helix structure" in clean_doc


def test_section_leading_truncated_token_sentence_is_trimmed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "section_truncated_token_trim"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    truncated_sentence = "cated that the PLLA-MATN3 scaffold effectively enhanced cartilage formation."
    fresh_sentence = "At present, the simulation of pericellular electrical microenvironments has been preliminarily realized."

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Truncated Token Section-Start Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 280.0, 182.0],
            "text": "8.3. ECM Formation",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 320.0],
            "text": f"{truncated_sentence} {fresh_sentence}",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h": "Heading", "p1_body": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert re.search(r"^### 8\.3\. ECM Formation\n\ncated that", clean_doc, flags=re.MULTILINE) is None
    assert truncated_sentence not in clean_doc
    assert fresh_sentence in clean_doc


def test_main_body_trim_suspicious_lowercase_fragment_positive(tmp_path: Path) -> None:
    """Positive: Main Body should drop a suspicious 1-token lowercase opener when followed by a fresh sentence."""
    run_dir = tmp_path / "run" / "mb_trim_positive"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {"block_id": "p1_frag", "page": 1, "bbox_pt": [320.0, 120.0, 540.0, 150.0], "text": "tive", "is_header_footer_candidate": False},
        {"block_id": "p1_main", "page": 1, "bbox_pt": [320.0, 200.0, 540.0, 320.0], "text": "The study demonstrates reproducible findings.", "is_header_footer_candidate": False},
    ]
    role_labels = {1: {"p1_frag": "Body", "p1_main": "Body"}}
    _write_jsonl(text_dir / "blocks_norm.jsonl", blocks)
    _write_vision_outputs(vision_dir, blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    main_body = _section_text(clean_doc, "Main Body")
    assert "tive" not in main_body
    assert "The study demonstrates reproducible findings." in main_body


def test_main_body_preserves_legitimate_lowercase_start_negative(tmp_path: Path) -> None:
    """Negative: Legitimate lowercase starts like 'e coli' should not be dropped."""
    run_dir = tmp_path / "run" / "mb_trim_negative"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {"block_id": "p1_frag", "page": 1, "bbox_pt": [320.0, 120.0, 540.0, 150.0], "text": "e coli is a gram-negative bacterium.", "is_header_footer_candidate": False},
        {"block_id": "p1_main", "page": 1, "bbox_pt": [320.0, 200.0, 540.0, 320.0], "text": "This paragraph follows and is independent.", "is_header_footer_candidate": False},
    ]
    role_labels = {1: {"p1_frag": "Body", "p1_main": "Body"}}
    _write_jsonl(text_dir / "blocks_norm.jsonl", blocks)
    _write_vision_outputs(vision_dir, blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    main_body = _section_text(clean_doc, "Main Body")
    assert "e coli is a gram-negative bacterium." in main_body
    assert "This paragraph follows and is independent." in main_body


def test_same_section_hyphen_wrap_merges_left_bottom_to_right_top(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "same_section_hyphen_column_transition"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Same-Section Hyphen Transition Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 280.0, 182.0],
            "text": "8.3. ECM Formation",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_left_bottom",
            "page": 1,
            "bbox_pt": [60.0, 675.0, 280.0, 710.0],
            "text": "Prior studies in cartilage models indi-",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_right_top",
            "page": 1,
            "bbox_pt": [320.0, 90.0, 540.0, 140.0],
            "text": "cated that aligned electroactivity supports ECM formation.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_h": "Heading",
            "p1_left_bottom": "Body",
            "p1_right_top": "Body",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    with open(text_dir / "layout_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"page_layouts": {"1": {"column_count": 2}}}, f, ensure_ascii=False)

    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert "Prior studies in cartilage models indi-" not in clean_doc
    assert re.search(r"^### 8\.3\. ECM Formation\n\ncated that", clean_doc, flags=re.MULTILINE) is None
    assert "Prior studies in cartilage models indicated that aligned electroactivity supports ECM formation." in clean_doc


def test_orphan_section_heading_is_demoted_to_body_text(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "orphan_heading_demoted"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Orphan Heading Demotion Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h_orphan",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": "6. Discussion",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h_orphan": "Heading"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert "### 6. Discussion" not in clean_doc
    assert re.search(r"^6\. Discussion$", clean_doc, flags=re.MULTILINE) is not None


def test_orphan_section_heading_text_is_preserved_not_dropped(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "orphan_heading_text_preserved"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    first_heading = "4. Methods"
    second_heading = "5. Future Work"
    body_text = "The protocol used controlled stimulation and calibrated measurements."

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Heading Retention Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h1",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": first_heading,
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 280.0],
            "text": body_text,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_h2",
            "page": 1,
            "bbox_pt": [80.0, 300.0, 340.0, 332.0],
            "text": second_heading,
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_h1": "Heading",
            "p1_body": "Body",
            "p1_h2": "Heading",
        }
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert "### 5. Future Work" not in clean_doc
    assert clean_doc.count(second_heading) == 1
    assert clean_doc.find(second_heading) > clean_doc.find(body_text)


def test_section_heading_with_real_body_is_kept_as_heading(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "heading_with_real_body"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Legitimate Heading Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": "3. Results",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 285.0],
            "text": "Electrical cues increased migration and matrix deposition across all groups.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h": "Heading", "p1_body": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    assert "### 3. Results" in clean_doc
    assert "Electrical cues increased migration and matrix deposition across all groups." in clean_doc


def test_clean_document_drops_page_number_only_line_in_section_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "drop_page_number_line"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Artifact Filtering Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": "2. Results",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_page_num",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 120.0, 220.0],
            "text": "5",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 230.0, 540.0, 310.0],
            "text": "Cell migration increased significantly under electrical stimulation.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h": "Heading", "p1_page_num": "Body", "p1_body": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    main_body = _section_text(clean_doc, "Main Body")
    assert re.search(r"(?m)^\s*5\s*$", main_body) is None
    assert "Cell migration increased significantly under electrical stimulation." in main_body


def test_clean_document_drops_check_for_updates_banner_line_in_section_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "drop_check_for_updates_line"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Artifact Filtering Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": "2. Results",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_banner",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 235.0],
            "text": "Check for updates in this article",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 240.0, 540.0, 320.0],
            "text": "Aligned fibers promoted osteogenic differentiation in vitro.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h": "Heading", "p1_banner": "Body", "p1_body": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    main_body = _section_text(clean_doc, "Main Body")
    assert re.search(r"(?im)^\s*check\s+for\s+updates\b", main_body) is None
    assert "Aligned fibers promoted osteogenic differentiation in vitro." in main_body


def test_clean_document_keeps_legitimate_content_line_in_section_body(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "keep_legitimate_numeric_line"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    legitimate_line = "Participants were screened and 120 met inclusion criteria."
    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Artifact Filtering Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 320.0, 182.0],
            "text": "2. Results",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 195.0, 540.0, 275.0],
            "text": legitimate_line,
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {1: {"p1_title": "Heading", "p1_h": "Heading", "p1_body": "Body"}}

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    clean_doc = (text_dir / "clean_document.md").read_text(encoding="utf-8")
    main_body = _section_text(clean_doc, "Main Body")
    assert legitimate_line in main_body


def test_run_paragraphs_writes_clean_document_metrics_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "metrics_doc"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Metrics Instrumentation Study",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        {
            "block_id": "p1_h_results",
            "page": 1,
            "bbox_pt": [80.0, 150.0, 300.0, 180.0],
            "text": "2. Results",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p1_body",
            "page": 1,
            "bbox_pt": [80.0, 190.0, 540.0, 280.0],
            "text": "The post- operative cohort maintained stable outcomes across follow-up.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
        {
            "block_id": "p1_h_orphan",
            "page": 1,
            "bbox_pt": [80.0, 320.0, 330.0, 350.0],
            "text": "5. Future Work",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
        },
        {
            "block_id": "p3_aff_like",
            "page": 3,
            "bbox_pt": [80.0, 180.0, 540.0, 240.0],
            "text": "Department of Radiology, Example University, New York, NY 10016",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
        },
    ]
    role_labels = {
        1: {
            "p1_title": "Heading",
            "p1_h_results": "Heading",
            "p1_body": "Body",
            "p1_h_orphan": "Heading",
        },
        3: {"p3_aff_like": "Body"},
    }

    typed_blocks = cast(list[dict[str, object]], blocks)
    _write_jsonl(text_dir / "blocks_norm.jsonl", typed_blocks)
    _write_vision_outputs(vision_dir, typed_blocks, role_labels)
    _ = run_paragraphs(run_dir)

    metrics_path = run_dir / "qa" / "clean_document_metrics.json"
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_keys = {
        "orphan_heading_count",
        "standalone_page_number_count",
        "check_for_updates_count",
        "affiliation_leak_count",
        "ordering_confidence_low",
        "section_boundary_unstable",
        "hyphen_wrap_count",
    }
    assert required_keys.issubset(metrics.keys())
    assert metrics.get("doc_id") == "metrics_doc"
    assert isinstance(metrics["orphan_heading_count"], int)
    assert metrics["orphan_heading_count"] >= 0
    assert metrics["hyphen_wrap_count"] == 0
    assert isinstance(metrics["affiliation_leak_count"], int)
    assert metrics["affiliation_leak_count"] >= 0
    assert isinstance(metrics["ordering_confidence_low"], bool)
    assert isinstance(metrics["section_boundary_unstable"], bool)


def test_normalize_inline_hyphen_wrap_artifacts_repairs_soft_and_hard_wraps() -> None:
    from ingest.paragraphs import normalize_inline_hyphen_wrap_artifacts

    assert normalize_inline_hyphen_wrap_artifacts("electro- magnetic response") == "electro-magnetic response"
    assert normalize_inline_hyphen_wrap_artifacts("micro\u00ad scope imaging") == "microscope imaging"


def test_normalize_inline_hyphen_wrap_artifacts_keeps_non_continuation_cases() -> None:
    from ingest.paragraphs import normalize_inline_hyphen_wrap_artifacts

    assert normalize_inline_hyphen_wrap_artifacts("inter- Related signaling") == "inter- Related signaling"
    assert normalize_inline_hyphen_wrap_artifacts("COVID- 19 cohort") == "COVID- 19 cohort"
