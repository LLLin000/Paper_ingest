import json
from pathlib import Path
from typing import Any

import pymupdf

from ingest import extractor, paragraphs
from ingest.manifest import LLMSettings, Manifest, RenderConfig, ToolchainInfo


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_extractor_writes_block_lines_sidecar(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 72), "Results", fontsize=18, fontname="helv")
    page.insert_text(
        (72, 84),
        "To better understand the heterogeneity of tendinopathy.",
        fontsize=10,
        fontname="helv",
    )
    pdf_path = tmp_path / "line-sidecar.pdf"
    doc.save(pdf_path)
    doc.close()

    run_dir = tmp_path / "run" / "extractor_sidecar_doc"
    manifest = Manifest(
        doc_id="extractor_sidecar_doc",
        input_pdf_path=str(pdf_path),
        input_pdf_sha256="0" * 64,
        started_at_utc="2026-03-08T00:00:00Z",
        toolchain=ToolchainInfo(python_version="3.12", package_lock_hash=""),
        model_config=LLMSettings(text_model="none", vision_model="none", temperature=0.0),
        render_config=RenderConfig(dpi=144, scale=2.0),
        pipeline_version="test",
    )

    total_pages, total_blocks = extractor.run_extractor(run_dir, manifest=manifest)

    assert total_pages == 1
    assert total_blocks == 1

    with open(run_dir / "text" / "block_lines.jsonl", "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 1
    record = rows[0]
    assert record["block_id"] == "p1_b0"
    assert record["page"] == 1
    assert len(record["lines"]) == 2
    assert record["lines"][0]["text"] == "Results"
    assert record["lines"][0]["font_stats"]["avg_size"] > record["lines"][1]["font_stats"]["avg_size"]
    assert record["lines"][1]["text"].startswith("To better understand")


def test_run_paragraphs_splits_mixed_block_using_block_lines_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "line_split_doc"
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
                "bbox_pt": [70.0, 50.0, 300.0, 90.0],
                "text": "Line Split Study",
                "font_stats": {"avg_size": 18.0, "is_bold": True},
                "is_header_footer_candidate": False,
                "is_heading_candidate": True,
                "column_guess": 1,
            },
            {
                "block_id": "p2_b0",
                "page": 2,
                "bbox_pt": [70.0, 120.0, 360.0, 210.0],
                "text": (
                    "Results To better understand the heterogeneity of tendinopathy "
                    "and identify potential subtypes, we collected clinical data."
                ),
                "font_stats": {"avg_size": 11.5, "is_bold": True},
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
                "bbox_pt": [70.0, 120.0, 360.0, 210.0],
                "lines": [
                    {
                        "line_index": 0,
                        "text": "Results",
                        "bbox_pt": [70.0, 120.0, 180.0, 145.0],
                        "avg_size": 18.0,
                        "is_bold": True,
                        "is_italic": False,
                        "dominant_font": "Helvetica-Bold",
                    },
                    {
                        "line_index": 1,
                        "text": (
                            "To better understand the heterogeneity of tendinopathy "
                            "and identify potential subtypes, we collected clinical data."
                        ),
                        "bbox_pt": [70.0, 150.0, 360.0, 210.0],
                        "avg_size": 10.0,
                        "is_bold": False,
                        "is_italic": False,
                        "dominant_font": "Helvetica",
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
                "role_labels": {"p2_b0": "Body"},
                "confidence": 0.95,
                "fallback_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    paragraph_count, _ = paragraphs.run_paragraphs(run_dir)

    assert paragraph_count == 3

    with open(run_dir / "paragraphs" / "paragraphs.jsonl", "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    texts = [str(row.get("text", "")) for row in rows]
    roles = [str(row.get("role", "")) for row in rows]

    assert "Results" in texts
    assert any(text.startswith("To better understand the heterogeneity") for text in texts)
    assert roles.count("Heading") == 2
    assert roles.count("Body") == 1

    with open(text_dir / "clean_document.md", "r", encoding="utf-8") as f:
        clean_document = f.read()
    assert "## Results" in clean_document
    assert "To better understand the heterogeneity of tendinopathy" in clean_document


def test_prepare_blocks_for_aggregation_recursively_splits_parent_and_subsection_headings() -> None:
    blocks = {
        "p2_b2": {
            "block_id": "p2_b2",
            "page": 2,
            "bbox_pt": [39.68, 346.91, 294.90, 560.65],
            "text": (
                "Results Identification of three distinct subtypes of tendinopathy based "
                "on transcriptome profiles and clinical features To better understand the "
                "heterogeneity of tendinopathy and identify potential subtypes, we collected "
                "clinical and transcriptomic data from 126 human-diseased tendons."
            ),
            "font_stats": {"avg_size": 8.8, "is_bold": False},
            "clean_role": "body_text",
        }
    }
    role_labels = {2: {"p2_b2": "Body"}}
    block_lines = {
        "p2_b2": [
            {
                "line_index": 0,
                "text": "Results",
                "bbox_pt": [39.68, 346.91, 76.67, 357.62],
                "font_stats": {"avg_size": 10.7, "is_bold": False, "dominant_font": "FontA"},
            },
            {
                "line_index": 1,
                "text": "Identification of three distinct subtypes of tendinopathy based",
                "bbox_pt": [39.68, 359.09, 293.84, 367.92],
                "font_stats": {"avg_size": 8.7, "is_bold": False, "dominant_font": "FontB"},
            },
            {
                "line_index": 2,
                "text": "on transcriptome profiles and clinical features",
                "bbox_pt": [39.68, 369.80, 226.91, 378.64],
                "font_stats": {"avg_size": 8.7, "is_bold": False, "dominant_font": "FontB"},
            },
            {
                "line_index": 3,
                "text": (
                    "To better understand the heterogeneity of tendinopathy and identify "
                    "potential subtypes, we collected clinical and transcriptomic data from "
                    "126 human-diseased tendons."
                ),
                "bbox_pt": [39.68, 381.04, 294.78, 399.97],
                "font_stats": {"avg_size": 8.2, "is_bold": False, "dominant_font": "FontC"},
            },
        ]
    }

    prepared_blocks, prepared_roles, expansion = paragraphs.prepare_blocks_for_aggregation(
        blocks,
        role_labels,
        block_lines_by_block=block_lines,
    )

    assert expansion["p2_b2"] == ["p2_b2__ls0", "p2_b2__ls1__ls0", "p2_b2__ls1__ls1"]
    assert prepared_roles[2]["p2_b2__ls0"] == "Heading"
    assert prepared_roles[2]["p2_b2__ls1__ls0"] == "Heading"
    assert prepared_roles[2]["p2_b2__ls1__ls1"] == "Body"
    assert prepared_blocks["p2_b2__ls0"]["text"] == "Results"
    assert prepared_blocks["p2_b2__ls1__ls0"]["text"].startswith("Identification of three distinct")
    assert prepared_blocks["p2_b2__ls1__ls1"]["text"].startswith("To better understand the heterogeneity")


def test_prepare_blocks_for_aggregation_splits_heading_on_font_transition_without_bold() -> None:
    blocks = {
        "p2_b8": {
            "block_id": "p2_b8",
            "page": 2,
            "bbox_pt": [306.14, 434.21, 561.38, 646.31],
            "text": (
                "Association of three distinct tendinopathy subtypes with clinical features "
                "To understand the clinical characteristics of these biochemically and "
                "clinically-defined tendinopathy subtypes, we examined the differences in "
                "clinical features between these subtypes."
            ),
            "font_stats": {"avg_size": 8.4, "is_bold": False},
            "clean_role": "body_text",
        }
    }
    role_labels = {2: {"p2_b8": "Body"}}
    block_lines = {
        "p2_b8": [
            {
                "line_index": 0,
                "text": "Association of three distinct tendinopathy subtypes with clinical",
                "bbox_pt": [306.14, 434.21, 561.31, 442.93],
                "font_stats": {"avg_size": 8.7, "is_bold": False, "dominant_font": "HeadingFont"},
            },
            {
                "line_index": 1,
                "text": "features",
                "bbox_pt": [306.14, 444.87, 339.16, 453.58],
                "font_stats": {"avg_size": 8.7, "is_bold": False, "dominant_font": "HeadingFont"},
            },
            {
                "line_index": 2,
                "text": (
                    "To understand the clinical characteristics of these biochemically and "
                    "clinically-defined tendinopathy subtypes, we examined the differences "
                    "in clinical features between these subtypes."
                ),
                "bbox_pt": [306.14, 455.99, 561.25, 474.92],
                "font_stats": {"avg_size": 8.2, "is_bold": False, "dominant_font": "BodyFont"},
            },
        ]
    }

    prepared_blocks, prepared_roles, expansion = paragraphs.prepare_blocks_for_aggregation(
        blocks,
        role_labels,
        block_lines_by_block=block_lines,
    )

    assert expansion["p2_b8"] == ["p2_b8__ls0", "p2_b8__ls1"]
    assert prepared_roles[2]["p2_b8__ls0"] == "Heading"
    assert prepared_roles[2]["p2_b8__ls1"] == "Body"
    assert prepared_blocks["p2_b8__ls0"]["text"] == (
        "Association of three distinct tendinopathy subtypes with clinical features"
    )
    assert prepared_blocks["p2_b8__ls1"]["text"].startswith("To understand the clinical characteristics")


def test_prepare_blocks_for_aggregation_keeps_body_like_second_line_out_of_heading_prefix() -> None:
    blocks = {
        "p12_b11": {
            "block_id": "p12_b11",
            "page": 12,
            "bbox_pt": [306.14, 304.11, 561.36, 357.11],
            "text": (
                "Code availability Data analyses were carried out using either an assortment of R system "
                "software (http://www.r-project.org, 4.2.0) packages including those of Bioconductor or "
                "original R code. This paper does not report original code."
            ),
            "font_stats": {"avg_size": 8.36, "is_bold": False},
            "clean_role": "body_text",
        }
    }
    role_labels = {12: {"p12_b11": "Body"}}
    block_lines = {
        "p12_b11": [
            {
                "line_index": 0,
                "text": "Code availability",
                "bbox_pt": [306.14, 304.11, 390.84, 314.82],
                "font_stats": {"avg_size": 10.71, "is_bold": False, "dominant_font": "HeadingFont"},
            },
            {
                "line_index": 1,
                "text": "Data analyses were carried out using either an assortment of R system",
                "bbox_pt": [306.14, 316.81, 561.24, 325.03],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "line_index": 2,
                "text": "software (http://www.r-project.org, 4.2.0) packages including those of",
                "bbox_pt": [306.14, 327.47, 561.36, 335.68],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "line_index": 3,
                "text": "Bioconductor or original R code. This paper does not report original code.",
                "bbox_pt": [306.14, 338.18, 561.24, 357.11],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
        ]
    }

    prepared_blocks, prepared_roles, expansion = paragraphs.prepare_blocks_for_aggregation(
        blocks,
        role_labels,
        block_lines_by_block=block_lines,
    )

    assert expansion["p12_b11"] == ["p12_b11__ls0", "p12_b11__ls1"]
    assert prepared_roles[12]["p12_b11__ls0"] == "Heading"
    assert prepared_roles[12]["p12_b11__ls1"] == "Body"
    assert prepared_blocks["p12_b11__ls0"]["text"] == "Code availability"
    assert prepared_blocks["p12_b11__ls1"]["text"].startswith(
        "Data analyses were carried out using either an assortment of R system"
    )


def test_prepare_blocks_for_aggregation_splits_short_common_heading_and_body() -> None:
    blocks = {
        "p14_b3": {
            "block_id": "p14_b3",
            "page": 14,
            "bbox_pt": [39.68, 636.11, 294.91, 664.41],
            "text": "Competing interests The authors declare no competing interest",
            "font_stats": {"avg_size": 10.2, "is_bold": True},
            "clean_role": "section_heading",
        }
    }
    role_labels = {14: {"p14_b3": "Heading"}}
    block_lines = {
        "p14_b3": [
            {
                "line_index": 0,
                "text": "Competing interests",
                "bbox_pt": [39.68, 636.11, 164.52, 649.42],
                "font_stats": {"avg_size": 10.6, "is_bold": True, "dominant_font": "HeadingFont"},
            },
            {
                "line_index": 1,
                "text": "The authors declare no competing interest",
                "bbox_pt": [39.68, 650.30, 294.91, 664.41],
                "font_stats": {"avg_size": 9.2, "is_bold": False, "dominant_font": "BodyFont"},
            },
        ]
    }

    prepared_blocks, prepared_roles, expansion = paragraphs.prepare_blocks_for_aggregation(
        blocks,
        role_labels,
        block_lines_by_block=block_lines,
    )

    assert expansion["p14_b3"] == ["p14_b3__ls0", "p14_b3__ls1"]
    assert prepared_roles[14]["p14_b3__ls0"] == "Heading"
    assert prepared_roles[14]["p14_b3__ls1"] == "Body"
    assert prepared_blocks["p14_b3__ls0"]["text"] == "Competing interests"
    assert prepared_blocks["p14_b3__ls1"]["text"] == "The authors declare no competing interest"


def test_prepare_blocks_for_aggregation_splits_short_references_heading_from_first_entry() -> None:
    blocks = {
        "p12_b12": {
            "block_id": "p12_b12",
            "page": 12,
            "bbox_pt": [306.14, 405.11, 561.36, 460.11],
            "text": "References 1. Millar, N. L. et al. Tendinopathy. Nat. Rev. Dis. Primers 7 (2021).",
            "font_stats": {"avg_size": 10.0, "is_bold": True},
            "clean_role": "section_heading",
        }
    }
    role_labels = {12: {"p12_b12": "Heading"}}
    block_lines = {
        "p12_b12": [
            {
                "line_index": 0,
                "text": "References",
                "bbox_pt": [306.14, 405.11, 384.30, 418.42],
                "font_stats": {"avg_size": 10.4, "is_bold": True, "dominant_font": "HeadingFont"},
            },
            {
                "line_index": 1,
                "text": "1.",
                "bbox_pt": [306.14, 420.10, 314.20, 433.40],
                "font_stats": {"avg_size": 9.1, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "line_index": 2,
                "text": "Millar, N. L. et al. Tendinopathy. Nat. Rev. Dis. Primers 7 (2021).",
                "bbox_pt": [318.10, 420.10, 561.36, 460.11],
                "font_stats": {"avg_size": 9.1, "is_bold": False, "dominant_font": "BodyFont"},
            },
        ]
    }

    prepared_blocks, prepared_roles, expansion = paragraphs.prepare_blocks_for_aggregation(
        blocks,
        role_labels,
        block_lines_by_block=block_lines,
    )

    assert expansion["p12_b12"] == ["p12_b12__ls0", "p12_b12__ls1"]
    assert prepared_roles[12]["p12_b12__ls0"] == "Heading"
    assert prepared_roles[12]["p12_b12__ls1"] == "Body"
    assert prepared_blocks["p12_b12__ls0"]["text"] == "References"
    assert prepared_blocks["p12_b12__ls1"]["text"].startswith("1. Millar, N. L. et al.")


def test_prepare_blocks_for_aggregation_splits_page1_front_matter_into_metadata_and_title() -> None:
    blocks = {
        "p1_b0": {
            "block_id": "p1_b0",
            "page": 1,
            "bbox_pt": [39.68, 94.61, 561.33, 161.80],
            "text": (
                "Article https://doi.org/10.1038/s41467-024-53826-w "
                "Classification of distinct tendinopathy subtypes for precision therapeutics"
            ),
            "font_stats": {"avg_size": 20.2, "is_bold": False},
            "clean_role": "reference_entry",
        }
    }
    role_labels = {1: {"p1_b0": "Body"}}
    block_lines = {
        "p1_b0": [
            {
                "line_index": 0,
                "text": "Article",
                "bbox_pt": [39.68, 94.61, 70.66, 104.57],
                "font_stats": {"avg_size": 9.96, "is_bold": False, "dominant_font": "FrontMatter"},
            },
            {
                "line_index": 1,
                "text": "https://doi.org/10.1038/s41467-024-53826-w",
                "bbox_pt": [396.90, 96.18, 561.32, 104.15],
                "font_stats": {"avg_size": 7.97, "is_bold": False, "dominant_font": "FrontMatter"},
            },
            {
                "line_index": 2,
                "text": "Classification of distinct tendinopathy",
                "bbox_pt": [39.68, 108.00, 487.42, 139.08],
                "font_stats": {"avg_size": 25.90, "is_bold": False, "dominant_font": "TitleFont"},
            },
            {
                "line_index": 3,
                "text": "subtypes for precision therapeutics",
                "bbox_pt": [39.68, 135.89, 456.40, 161.80],
                "font_stats": {"avg_size": 25.90, "is_bold": False, "dominant_font": "TitleFont"},
            },
        ]
    }

    prepared_blocks, prepared_roles, expansion = paragraphs.prepare_blocks_for_aggregation(
        blocks,
        role_labels,
        block_lines_by_block=block_lines,
    )

    assert expansion["p1_b0"] == ["p1_b0__fm0", "p1_b0__fm1", "p1_b0__fm2"]
    assert prepared_blocks["p1_b0__fm0"]["clean_role"] == "journal_meta"
    assert prepared_blocks["p1_b0__fm1"]["clean_role"] == "doi"
    assert prepared_blocks["p1_b0__fm2"]["clean_role"] == "main_title"
    assert prepared_roles[1]["p1_b0__fm0"] == "HeaderFooter"
    assert prepared_roles[1]["p1_b0__fm1"] == "HeaderFooter"
    assert prepared_roles[1]["p1_b0__fm2"] == "Heading"


def test_prepare_blocks_for_aggregation_does_not_split_compact_graphic_label_cluster() -> None:
    blocks = {
        "p4_b36": {
            "block_id": "p4_b36",
            "page": 4,
            "bbox_pt": [278.01, 106.41, 528.43, 146.93],
            "text": "HYE N-HYE -log10 (P-value) 6",
            "font_stats": {"avg_size": 8.5, "is_bold": False},
            "clean_role": "body_text",
        }
    }
    role_labels = {4: {"p4_b36": "Body"}}
    block_lines = {
        "p4_b36": [
            {
                "line_index": 0,
                "text": "HYE",
                "bbox_pt": [278.01, 106.41, 297.94, 116.11],
                "font_stats": {"avg_size": 9.69, "is_bold": False, "dominant_font": "ArialMT"},
            },
            {
                "line_index": 1,
                "text": "N-HYE",
                "bbox_pt": [278.01, 118.47, 308.18, 128.16],
                "font_stats": {"avg_size": 9.69, "is_bold": False, "dominant_font": "ArialMT"},
            },
            {
                "line_index": 2,
                "text": "-log10 (P-value)",
                "bbox_pt": [474.81, 129.06, 528.43, 137.18],
                "font_stats": {"avg_size": 7.90, "is_bold": False, "dominant_font": "ArialMT"},
            },
            {
                "line_index": 3,
                "text": "6",
                "bbox_pt": [499.42, 139.02, 503.81, 146.93],
                "font_stats": {"avg_size": 7.90, "is_bold": False, "dominant_font": "ArialMT"},
            },
        ]
    }

    prepared_blocks, prepared_roles, expansion = paragraphs.prepare_blocks_for_aggregation(
        blocks,
        role_labels,
        block_lines_by_block=block_lines,
    )

    assert expansion["p4_b36"] == ["p4_b36"]
    assert prepared_roles[4]["p4_b36"] == "Body"
    assert prepared_blocks["p4_b36"]["text"] == "HYE N-HYE -log10 (P-value) 6"
