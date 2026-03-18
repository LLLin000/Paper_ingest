from ingest.paragraphs import Paragraph, render_clean_document


def _section_text(markdown: str, section_title: str) -> str:
    import re

    pattern = rf"^## {re.escape(section_title)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def test_render_clean_document_redirects_stray_caption_tail_out_of_main_body() -> None:
    tail_text = (
        "key drivers for each subtype plotted in its location in the network. "
        "Each point represents a gene. Points of different colors represent "
        "different gene co-expression modules."
    )
    paragraphs = [
        Paragraph(
            para_id="title",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["b_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "body1"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="body1",
            page_span={"start": 7, "end": 7},
            role="Body",
            section_path=["Results", "Tendinopathy subtypes translated to animal models uncover the potential for personalized therapy"],
            text=(
                "We conducted subtype determination of animal models based on characteristic genes "
                "using NTP. The collagenase model was classified as Hw, while the other"
            ),
            evidence_pointer={"pages": [7], "bbox_union": [40.0, 300.0, 295.0, 410.0], "source_block_ids": ["b_body1"]},
            neighbors={"prev_para_id": "title", "next_para_id": "cap_head"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="cap_head",
            page_span={"start": 7, "end": 7},
            role="FigureCaption",
            section_path=None,
            text="Fig. 3 | Key pathogenesis and key regulatory genes of the three distinct rotator cuff tendinopathy subtypes.",
            evidence_pointer={"pages": [7], "bbox_union": [40.0, 520.0, 295.0, 620.0], "source_block_ids": ["b_cap_head"]},
            neighbors={"prev_para_id": "body1", "next_para_id": "body2"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="body2",
            page_span={"start": 7, "end": 7},
            role="Body",
            section_path=["Clinical data to verify the therapeutic effects of glucocorticoids on different tendinopathy subtypes"],
            text="We further compared the differences in prognostic improvement among patients with different tendinopathy subtypes.",
            evidence_pointer={"pages": [7], "bbox_union": [40.0, 630.0, 295.0, 700.0], "source_block_ids": ["b_body2"]},
            neighbors={"prev_para_id": "cap_head", "next_para_id": "cap_tail"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="cap_tail",
            page_span={"start": 7, "end": 7},
            role="Body",
            section_path=None,
            text=tail_text,
            evidence_pointer={"pages": [7], "bbox_union": [320.0, 720.0, 560.0, 780.0], "source_block_ids": ["b_cap_tail"]},
            neighbors={"prev_para_id": "body2", "next_para_id": None},
            confidence=0.45,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "b_title", "clean_role": "main_title"},
        {"block_id": "b_body1", "clean_role": "body_text"},
        {"block_id": "b_cap_head", "clean_role": "figure_caption"},
        {"block_id": "b_body2", "clean_role": "body_text"},
        {"block_id": "b_cap_tail", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="mixed_caption_tail")

    main_body = _section_text(clean_doc, "Main Body")
    figures_tables = _section_text(clean_doc, "Figures and Tables")

    assert tail_text not in main_body
    assert tail_text in figures_tables


def test_render_clean_document_uses_mixed_group_review_to_redirect_caption_tail() -> None:
    tail_text = "Signal pathways with characteristic changes and treatment recommendation for subtype Ir."
    paragraphs = [
        Paragraph(
            para_id="title_reviewed_tail",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["b_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "body_reviewed_tail"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="body_reviewed_tail",
            page_span={"start": 7, "end": 7},
            role="Body",
            section_path=["Results", "Tendinopathy subtypes translated to animal models uncover the potential for personalized therapy"],
            text="We conducted subtype determination of animal models based on characteristic genes using NTP.",
            evidence_pointer={"pages": [7], "bbox_union": [40.0, 300.0, 295.0, 410.0], "source_block_ids": ["b_body"]},
            neighbors={"prev_para_id": "title_reviewed_tail", "next_para_id": "cap_head_reviewed_tail"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="cap_head_reviewed_tail",
            page_span={"start": 7, "end": 7},
            role="FigureCaption",
            section_path=None,
            text="Fig. 3 | Key pathogenesis and key regulatory genes of the three distinct rotator cuff tendinopathy subtypes.",
            evidence_pointer={"pages": [7], "bbox_union": [40.0, 520.0, 295.0, 620.0], "source_block_ids": ["b_cap_head"]},
            neighbors={"prev_para_id": "body_reviewed_tail", "next_para_id": "cap_tail_reviewed_tail"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="cap_tail_reviewed_tail",
            page_span={"start": 7, "end": 7},
            role="Body",
            section_path=None,
            text=tail_text,
            evidence_pointer={"pages": [7], "bbox_union": [320.0, 640.0, 560.0, 700.0], "source_block_ids": ["b_cap_tail"]},
            neighbors={"prev_para_id": "cap_head_reviewed_tail", "next_para_id": None},
            confidence=0.45,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "b_title", "clean_role": "main_title"},
        {"block_id": "b_body", "clean_role": "body_text"},
        {"block_id": "b_cap_head", "clean_role": "figure_caption"},
        {"block_id": "b_cap_tail", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(
        paragraphs,
        annotated_blocks,
        doc_id="mixed_group_review_caption_tail",
        mixed_group_reviews_by_page={
            7: {
                "b_cap_tail": {
                    "block_id": "b_cap_tail",
                    "decision": "caption_tail",
                    "caption_kind": "figure",
                    "confidence": 0.97,
                }
            }
        },
    )

    main_body = _section_text(clean_doc, "Main Body")
    figures_tables = _section_text(clean_doc, "Figures and Tables")

    assert tail_text not in main_body
    assert tail_text in figures_tables


def test_render_clean_document_merges_figure_caption_role_continuation_without_prefix() -> None:
    tail_text = (
        "key drivers for each subtype plotted in its location in the network. "
        "Each point represents a gene. Points of different colors represent "
        "different gene co-expression modules. (KDGs, key driver genes). "
        "H Biological insight into the three distinct tendinopathy subtypes."
    )
    paragraphs = [
        Paragraph(
            para_id="title_caption_role_tail",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["b_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "cap_head"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="cap_head",
            page_span={"start": 7, "end": 7},
            role="FigureCaption",
            section_path=None,
            text="Fig. 3 | Key pathogenesis and key regulatory genes of the three distinct rotator cuff tendinopathy subtypes.",
            evidence_pointer={"pages": [7], "bbox_union": [40.0, 520.0, 295.0, 620.0], "source_block_ids": ["b_cap_head"]},
            neighbors={"prev_para_id": "title_caption_role_tail", "next_para_id": "cap_tail"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="cap_tail",
            page_span={"start": 7, "end": 7},
            role="FigureCaption",
            section_path=None,
            text=tail_text,
            evidence_pointer={"pages": [7], "bbox_union": [320.0, 520.0, 560.0, 690.0], "source_block_ids": ["b_cap_tail"]},
            neighbors={"prev_para_id": "cap_head", "next_para_id": None},
            confidence=0.95,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "b_title", "clean_role": "main_title"},
        {"block_id": "b_cap_head", "clean_role": "figure_caption"},
        {"block_id": "b_cap_tail", "clean_role": "figure_caption"},
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="caption_role_tail")

    main_body = _section_text(clean_doc, "Main Body")
    figures_tables = _section_text(clean_doc, "Figures and Tables")

    assert tail_text not in main_body
    assert tail_text in figures_tables
