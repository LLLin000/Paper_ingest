from ingest.paragraphs import Paragraph, classify_clean_blocks, render_clean_document
import ingest.paragraphs as paragraphs_mod


def _section_text(markdown: str, section_title: str) -> str:
    import re

    pattern = rf"^## {re.escape(section_title)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def test_infer_metadata_role_rejects_decimal_leading_body_sentence_as_affiliation() -> None:
    text = "94.7% of red hyperemia tendons were identified as Subtype I, while"
    words = paragraphs_mod.word_tokens(text)

    result = paragraphs_mod.infer_metadata_role(
        text=text,
        low=text.lower(),
        words=words,
        page=2,
        y_rel=0.25,
        tcase_ratio=paragraphs_mod.title_case_ratio(words),
        digits=paragraphs_mod.digit_token_count(words),
        sentence_ratio=paragraphs_mod.sentence_like_ratio(words),
    )

    assert result is None


def test_classify_clean_blocks_recovers_textual_figure_caption_from_fallback_body_label() -> None:
    blocks = {
        "p1_title": {
            "block_id": "p1_title",
            "page": 1,
            "bbox_pt": [80.0, 70.0, 520.0, 120.0],
            "text": "Classification of distinct tendinopathy subtypes for precision therapeutics",
            "is_header_footer_candidate": False,
            "is_heading_candidate": True,
            "font_stats": {"avg_size": 16.0, "is_bold": True},
        },
        "p3_figcap": {
            "block_id": "p3_figcap",
            "page": 3,
            "bbox_pt": [40.0, 552.0, 295.0, 660.0],
            "text": "Fig. 1 | Identification of three distinct subtypes of rotator cuff tendinopathy. A Heat map of NMF consensus matrix and average silhouette-width plots.",
            "is_header_footer_candidate": False,
            "is_heading_candidate": False,
            "font_stats": {"avg_size": 7.0, "is_bold": False},
        },
    }
    role_labels = {
        1: {"p1_title": "Heading"},
        3: {"p3_figcap": "Body"},
    }

    _, annotated = classify_clean_blocks(blocks, role_labels)
    by_id = {str(row.get("block_id", "")): row for row in annotated}

    assert by_id["p3_figcap"]["clean_role"] == "figure_caption"


def test_render_clean_document_filters_tang_chart_fragments_from_main_body() -> None:
    narrative = (
        "We then aimed to determine whether subgroups of tendinopathy exhibited molecular differences "
        "and validated the separation by PCA and pathway analysis."
    )
    figure_caption = (
        "Fig. 1 | Identification of three distinct subtypes of rotator cuff tendinopathy. "
        "A Heat map of NMF consensus matrix and average silhouette-width plots."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["b_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_body"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_body",
            page_span={"start": 3, "end": 3},
            role="Body",
            section_path=None,
            text=narrative,
            evidence_pointer={"pages": [3], "bbox_union": [40.0, 200.0, 290.0, 280.0], "source_block_ids": ["b_body"]},
            neighbors={"prev_para_id": "p_title", "next_para_id": "p_axis"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_axis",
            page_span={"start": 3, "end": 3},
            role="Body",
            section_path=None,
            text="1.0 0.8",
            evidence_pointer={"pages": [3], "bbox_union": [170.0, 90.0, 180.0, 122.0], "source_block_ids": ["b_axis"]},
            neighbors={"prev_para_id": "p_body", "next_para_id": "p_label"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_label",
            page_span={"start": 3, "end": 3},
            role="Body",
            section_path=None,
            text="Brunet",
            evidence_pointer={"pages": [3], "bbox_union": [240.0, 116.0, 250.0, 140.0], "source_block_ids": ["b_label"]},
            neighbors={"prev_para_id": "p_axis", "next_para_id": "p_or"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_or",
            page_span={"start": 3, "end": 3},
            role="Body",
            section_path=None,
            text="Feature OR 8.87 Veins",
            evidence_pointer={"pages": [3], "bbox_union": [138.0, 172.0, 224.0, 180.0], "source_block_ids": ["b_or"]},
            neighbors={"prev_para_id": "p_label", "next_para_id": "p_kegg"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_kegg",
            page_span={"start": 3, "end": 3},
            role="Body",
            section_path=None,
            text="Cytokine-cytokine receptor interaction CAMs Rap1 pathway Calcium pathway VSM contraction NLRI -log10 (P-Value)",
            evidence_pointer={"pages": [3], "bbox_union": [216.0, 443.0, 286.0, 459.0], "source_block_ids": ["b_kegg"]},
            neighbors={"prev_para_id": "p_or", "next_para_id": "p_figcap"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_figcap",
            page_span={"start": 3, "end": 3},
            role="Body",
            section_path=None,
            text=figure_caption,
            evidence_pointer={"pages": [3], "bbox_union": [40.0, 552.0, 295.0, 660.0], "source_block_ids": ["b_figcap"]},
            neighbors={"prev_para_id": "p_kegg", "next_para_id": None},
            confidence=0.95,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "b_title", "clean_role": "main_title"},
        {"block_id": "b_body", "clean_role": "body_text"},
        {"block_id": "b_axis", "clean_role": "body_text"},
        {"block_id": "b_label", "clean_role": "body_text"},
        {"block_id": "b_or", "clean_role": "body_text"},
        {"block_id": "b_kegg", "clean_role": "body_text"},
        {"block_id": "b_figcap", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_clean")
    main_body = _section_text(clean_doc, "Main Body")

    assert narrative in main_body
    assert "1.0 0.8" not in main_body
    assert "Brunet" not in main_body
    assert "Feature OR 8.87 Veins" not in main_body
    assert "Cytokine-cytokine receptor interaction" not in main_body
    assert figure_caption not in main_body


def test_render_clean_document_prefers_sanitized_metadata_title_over_check_for_updates() -> None:
    paragraphs = [
        Paragraph(
            para_id="p_bad_title",
            page_span={"start": 1, "end": 1},
            role="Body",
            section_path=None,
            text="Check for updates",
            evidence_pointer={"pages": [1], "bbox_union": [52.0, 269.0, 120.0, 277.0], "source_block_ids": ["b_bad_title"]},
            neighbors={"prev_para_id": None, "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [
        {
            "block_id": "b_doi_title",
            "page": 1,
            "clean_role": "doi",
            "is_nuisance": False,
            "text": "Article https://doi.org/10.1038/s41467-024-53826-w Classification of distinct tendinopathy subtypes for precision therapeutics",
            "font_stats": {"avg_size": 20.2},
            "bbox_pt": [40.0, 95.0, 561.0, 162.0],
        },
        {
            "block_id": "b_bad_title",
            "page": 1,
            "clean_role": "body_text",
            "is_nuisance": False,
            "text": "Check for updates",
            "font_stats": {"avg_size": 8.0},
            "bbox_pt": [52.0, 269.0, 120.0, 277.0],
        },
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_title")

    assert clean_doc.startswith("# Classification of distinct tendinopathy subtypes for precision therapeutics")
    assert "- https://doi.org/10.1038/s41467-024-53826-w" in clean_doc
    assert "Article https://doi.org/10.1038/s41467-024-53826-w Classification" not in clean_doc

def test_render_clean_document_filters_short_tang_plot_labels_from_main_body() -> None:
    narrative = "Subtype Ir exhibited the worst shoulder joint function across the compared clinical readouts."
    label_texts = [
        "Hw Iw Ir 2",
        "Iw vs. N",
        "White tendon",
        "Joint synovium",
        "Anteflexion ( )",
        "D-Dimer ( /L)",
        "Hypoxia Glycolysis EMT UPR Mtorc1 signaling IFN response Kras signaling up Complement IL6 jak stat3 signaling Inflammatory response Gross view under arthroscopy Molecular clinical subtype",
    ]

    paragraphs = [
        Paragraph(
            para_id="p_title_labels",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["bl_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_label_0"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [{"block_id": "bl_title", "clean_role": "main_title"}]

    for idx, label in enumerate(label_texts):
        block_id = f"bl_label_{idx}"
        next_id = f"p_label_{idx + 1}" if idx + 1 < len(label_texts) else "p_body_labels"
        paragraphs.append(
            Paragraph(
                para_id=f"p_label_{idx}",
                page_span={"start": 3, "end": 3},
                role="Body",
                section_path=None,
                text=label,
                evidence_pointer={"pages": [3], "bbox_union": [120.0, 90.0 + idx * 24.0, 260.0, 104.0 + idx * 24.0], "source_block_ids": [block_id]},
                neighbors={"prev_para_id": "p_title_labels" if idx == 0 else f"p_label_{idx - 1}", "next_para_id": next_id},
                confidence=0.5,
                provenance={"source": "test", "notes": "low_confidence no_section_path"},
            )
        )
        annotated_blocks.append({"block_id": block_id, "clean_role": "body_text"})

    paragraphs.append(
        Paragraph(
            para_id="p_body_labels",
            page_span={"start": 3, "end": 3},
            role="Body",
            section_path=None,
            text=narrative,
            evidence_pointer={"pages": [3], "bbox_union": [40.0, 420.0, 290.0, 500.0], "source_block_ids": ["bl_body"]},
            neighbors={"prev_para_id": f"p_label_{len(label_texts) - 1}", "next_para_id": None},
            confidence=0.95,
            provenance={"source": "test"},
        )
    )
    annotated_blocks.append({"block_id": "bl_body", "clean_role": "body_text"})

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_labels")
    main_body = _section_text(clean_doc, "Main Body")

    assert narrative in main_body
    for label in label_texts:
        assert label not in main_body


def test_split_embedded_section_heading_recovers_results_and_sentence_case_subheading() -> None:
    text = (
        "In heterogeneous diseases such as colorectal cancer, transcriptome sequencing can offer a deep "
        "insight into the pathology of the affected tissues. Results Identification of three distinct "
        "subtypes of tendinopathy based on transcriptome profiles and clinical features To better "
        "understand the heterogeneity of tendinopathy and identify potential subtypes, we collected "
        "clinical and transcriptomic data from 126 human-diseased tendons."
    )

    prefix, heading, suffix = paragraphs_mod.split_embedded_section_heading(text)

    assert prefix.endswith("affected tissues.")
    assert heading == "Results"
    assert suffix.startswith("Identification of three distinct subtypes of tendinopathy")


def test_render_clean_document_recovers_embedded_unnumbered_headings_from_mixed_body_block() -> None:
    mixed_text = (
        "In heterogeneous diseases such as colorectal cancer, transcriptome sequencing can offer a deep "
        "insight into the pathology of the affected tissues. Results Identification of three distinct "
        "subtypes of tendinopathy based on transcriptome profiles and clinical features To better "
        "understand the heterogeneity of tendinopathy and identify potential subtypes, we collected "
        "clinical and transcriptomic data from 126 human-diseased tendons."
    )
    association_text = (
        "Association of three distinct tendinopathy subtypes with clinical features To understand the "
        "clinical characteristics of these biochemically and clinically-defined tendinopathy subtypes, "
        "we examined the differences in clinical features between these subtypes."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_heading_recovery",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["bh_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_mixed_body"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_mixed_body",
            page_span={"start": 2, "end": 2},
            role="Body",
            section_path=None,
            text=mixed_text,
            evidence_pointer={"pages": [2], "bbox_union": [40.0, 180.0, 295.0, 742.0], "source_block_ids": ["bh_mixed"]},
            neighbors={"prev_para_id": "p_title_heading_recovery", "next_para_id": "p_assoc_body"},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
        Paragraph(
            para_id="p_assoc_body",
            page_span={"start": 2, "end": 2},
            role="Body",
            section_path=None,
            text=association_text,
            evidence_pointer={"pages": [2], "bbox_union": [306.0, 440.0, 561.0, 742.0], "source_block_ids": ["bh_assoc"]},
            neighbors={"prev_para_id": "p_mixed_body", "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "bh_title", "clean_role": "main_title"},
        {"block_id": "bh_mixed", "clean_role": "body_text"},
        {"block_id": "bh_assoc", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_embedded_heading")
    main_body = _section_text(clean_doc, "Main Body")

    assert "### Results" in clean_doc
    assert "### Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features" in clean_doc
    assert "### Association of three distinct tendinopathy subtypes with clinical features" in clean_doc
    assert "To better understand the heterogeneity of tendinopathy" in main_body
    assert "To understand the clinical characteristics of these biochemically and clinically-defined tendinopathy subtypes" in main_body


def test_render_clean_document_uses_vision_embedded_heading_hints_to_limit_mixed_block_splits() -> None:
    mixed_text = (
        "In heterogeneous diseases such as colorectal cancer, transcriptome sequencing can offer a deep "
        "insight into the pathology of the affected tissues. Results Identification of three distinct "
        "subtypes of tendinopathy based on transcriptome profiles and clinical features To better "
        "understand the heterogeneity of tendinopathy and identify potential subtypes, we collected "
        "clinical and transcriptomic data from 126 human-diseased tendons. Principal compo- nent "
        "analysis (PCA) and heatmap of differentially expressed genes (DEGs) demonstrated a clear "
        "distinction between diseased and normal tendon samples. Therefore, based on the RNA-seq "
        "data, tendinopathy can be classified into two distinct molecular subtypes."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_vision_hint",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["vh_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_mixed_vision_hint"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_mixed_vision_hint",
            page_span={"start": 2, "end": 2},
            role="Body",
            section_path=None,
            text=mixed_text,
            evidence_pointer={"pages": [2], "bbox_union": [40.0, 180.0, 295.0, 742.0], "source_block_ids": ["vh_mixed"]},
            neighbors={"prev_para_id": "p_title_vision_hint", "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "vh_title", "clean_role": "main_title"},
        {"block_id": "vh_mixed", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(
        paragraphs,
        annotated_blocks,
        doc_id="tang_vision_heading_hint",
        embedded_heading_hints_by_page={
            2: {
                "vh_mixed": [
                    "Results",
                    "Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features",
                ]
            }
        },
        reviewed_block_ids_by_page={2: {"vh_mixed"}},
    )
    main_body = _section_text(clean_doc, "Main Body")

    assert "### Results" in clean_doc
    assert "### Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features" in clean_doc
    assert "### Principal compo- nent analysis (PCA) and heatmap of differentially expressed genes (DEGs)" not in clean_doc
    assert "### Therefore, based on the" not in clean_doc
    assert "Principal compo-nent analysis (PCA) and heatmap of differentially expressed genes (DEGs) demonstrated a clear distinction" in main_body
    assert "Therefore, based on the RNA-seq data, tendinopathy can be classified into two distinct molecular subtypes." in main_body


def test_render_clean_document_uses_vision_hint_to_trim_heading_and_suppress_false_followups() -> None:
    block_text = (
        "RNA sequence Total RNA was extracted from the tissue using TRIzol Reagent according to the "
        "manufacturer instructions. After quantification by TBS380, Paired-end libraries were sequenced "
        "with the Illumina HiSeq PE 2X150bp read length."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_rna_hint",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["vr_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_rna_hint"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_rna_hint",
            page_span={"start": 10, "end": 10},
            role="Body",
            section_path=None,
            text=block_text,
            evidence_pointer={"pages": [10], "bbox_union": [40.0, 560.0, 295.0, 732.0], "source_block_ids": ["vr_rna"]},
            neighbors={"prev_para_id": "p_title_rna_hint", "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "vr_title", "clean_role": "main_title"},
        {"block_id": "vr_rna", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(
        paragraphs,
        annotated_blocks,
        doc_id="tang_rna_heading_hint",
        embedded_heading_hints_by_page={10: {"vr_rna": ["RNA sequence"]}},
        reviewed_block_ids_by_page={10: {"vr_rna"}},
    )
    main_body = _section_text(clean_doc, "Main Body")

    assert "### RNA sequence" in clean_doc
    assert "### RNA sequence Total" not in clean_doc
    assert "### After quantification by TBS380," not in clean_doc
    assert "Total RNA was extracted from the tissue using TRIzol Reagent" in main_body
    assert "After quantification by TBS380, Paired-end libraries were sequenced" in main_body


def test_render_clean_document_uses_line_evidence_to_reject_false_embedded_heading() -> None:
    body_text = (
        "To better understand the heterogeneity of tendinopathy and identify potential subtypes, "
        "we collected clinical and transcriptomic data from 126 human-diseased tendons. "
        "Principal compo- nent analysis (PCA) and heatmap of differentially expressed genes (DEGs) "
        "demonstrated a clear distinction between the transcriptomes of diseased and normal tendon samples."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_line_guard",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["lg_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_body_line_guard"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_body_line_guard",
            page_span={"start": 2, "end": 2},
            role="Body",
            section_path=None,
            text=body_text,
            evidence_pointer={
                "pages": [2],
                "bbox_union": [40.0, 240.0, 295.0, 520.0],
                "source_block_ids": ["lg_body"],
                "source_line_spans": [{"block_id": "lg_body", "start": 0, "end": 5}],
            },
            neighbors={"prev_para_id": "p_title_line_guard", "next_para_id": None},
            confidence=0.95,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "lg_title", "clean_role": "main_title"},
        {"block_id": "lg_body", "clean_role": "body_text"},
    ]
    block_lines_by_block = {
        "lg_body": [
            {
                "line_index": 0,
                "text": "To better understand the heterogeneity of tendinopathy and identify",
                "bbox_pt": [40.0, 240.0, 295.0, 252.0],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "line_index": 1,
                "text": "potential subtypes, we collected clinical and transcriptomic data from",
                "bbox_pt": [40.0, 254.0, 295.0, 266.0],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "line_index": 2,
                "text": "126 human-diseased tendons. Principal compo-",
                "bbox_pt": [40.0, 268.0, 295.0, 280.0],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "line_index": 3,
                "text": "nent analysis (PCA) and heatmap of differentially expressed genes",
                "bbox_pt": [40.0, 282.0, 295.0, 294.0],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "line_index": 4,
                "text": "(DEGs) demonstrated a clear distinction between the transcriptomes of",
                "bbox_pt": [40.0, 296.0, 295.0, 308.0],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "line_index": 5,
                "text": "diseased and normal tendon samples.",
                "bbox_pt": [40.0, 310.0, 295.0, 322.0],
                "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"},
            },
        ]
    }

    clean_doc, _ = render_clean_document(
        paragraphs,
        annotated_blocks,
        doc_id="tang_line_evidence_guard",
        block_lines_by_block=block_lines_by_block,
    )
    main_body = _section_text(clean_doc, "Main Body")

    assert "### Principal compo- nent analysis (PCA) and heatmap of differentially expressed genes (DEGs)" not in clean_doc
    assert "Principal compo-nent analysis (PCA) and heatmap of differentially expressed genes (DEGs) demonstrated a clear distinction" in main_body


def test_render_clean_document_rejects_discourse_clause_as_embedded_heading_without_line_evidence() -> None:
    body_text = (
        "Conservative treatment modalities also often elicit heterogeneous drug responses. "
        "These methods can be employed to alleviate symptoms in patients with mild conditions. "
        "For instance, glucocorticoids have been widely used in tendon-related diseases, but their use "
        "Tendinopathy is a term used to describe a complex, multi-faceted tendon pathology characterized by pain."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_discourse_guard",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["dg_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_body_discourse_guard"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_body_discourse_guard",
            page_span={"start": 1, "end": 1},
            role="Body",
            section_path=None,
            text=body_text,
            evidence_pointer={"pages": [1], "bbox_union": [40.0, 180.0, 295.0, 320.0], "source_block_ids": ["dg_body"]},
            neighbors={"prev_para_id": "p_title_discourse_guard", "next_para_id": None},
            confidence=0.95,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "dg_title", "clean_role": "main_title"},
        {"block_id": "dg_body", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(
        paragraphs,
        annotated_blocks,
        doc_id="tang_discourse_heading_guard",
    )
    main_body = _section_text(clean_doc, "Main Body")

    assert "### For instance, glucocorticoids have been widely used in tendon-related diseases, but their use" not in clean_doc
    assert "For instance, glucocorticoids have been widely used in tendon-related diseases, but their use Tendinopathy is a term used to describe a complex, multi-faceted tendon pathology characterized by pain." in main_body


def test_render_clean_document_uses_source_block_lines_when_source_line_spans_are_missing() -> None:
    body_text_p2 = (
        "The heatmap illustrated the top 100 upregulated DEGs among the three subtypes and their "
        "associated top Gene Ontology biological process terms (Fig. 1F). The Venn diagram clearly "
        "illustrated DEGs that were specifically upregulated or downregulated in each of the three "
        "subtypes, as well as those commonly upregulated or downregulated genes (Fig. 1G, H). "
        "Enrichment analysis using Kyoto Encyclopedia of Genes and Genomes (KEGG) terms revealed that "
        "the pathways specifically upregulated in subtype Hw mainly included the HIF-1 pathway and "
        "several metabolic pathways, while in subtype Iw, it was Cytokine-cytokine receptor "
        "interaction, and in subtype Ir, they were mainly related to inflammation (Fig. 1I). "
        "The genes commonly altered in the three subtypes were enriched in metabolic and inflammatory "
        "pathways (Supplementary Fig. 3). In summary, subtype Hw was similar to Subtype H, primarily "
        "upregulating hypoxia and metabolic pathways. Subtype Iw and Ir were similar to Subtype I, "
        "mainly upregulating pathways related to inflammation."
    )
    body_text_p5_left = (
        "With regards to blood testing, the three distinct subtypes also exhibited differences, "
        "particularly in coagulation function. The results showed that subtype Ir had significantly "
        "higher levels than subtype Hw in D-Dimer, Prothrombin Time, and International Normalized "
        "Ratio (Fig. 2E and Supplementary Fig. 6A, B). PT activity in subtype Ir was significantly "
        "lower than that in subtype Hw (Supplementary Fig. 6C). The examination results of coagulation "
        "function indicated that patients with subtype Ir might have thrombosis or that their blood "
        "was in a hypercoagulable state."
    )
    body_text_p5_right = (
        "(Fig. 3D). Major modules were annotated by significantly associated KEGG terms (Fig. 3E). "
        "The module most correlated with the upregulated gene set in subtype Hw was c1-23, with the "
        "representative KEGG term being Glycolysis/Gluconeogenesis. For subtype Ir, the module was "
        "c1-22, with the representative KEGG pathway being Cytokine-cytokine receptor interaction. "
        "In the downregulated gene set, the modules most correlated with subtype Hw were c1-19 and "
        "c1-27, while those most correlated with subtype Ir were c1-20 and c1-27 (Fig. 3F)."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_source_line_fallback",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["slf_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_body_source_line_fallback_p2"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_body_source_line_fallback_p2",
            page_span={"start": 2, "end": 2},
            role="Body",
            section_path=None,
            text=body_text_p2,
            evidence_pointer={"pages": [2], "bbox_union": [306.0, 145.0, 561.0, 421.0], "source_block_ids": ["slf_p2"]},
            neighbors={"prev_para_id": "p_title_source_line_fallback", "next_para_id": "p_body_source_line_fallback_p5_left"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_body_source_line_fallback_p5_left",
            page_span={"start": 5, "end": 5},
            role="Body",
            section_path=None,
            text=body_text_p5_left,
            evidence_pointer={"pages": [5], "bbox_union": [40.0, 49.0, 295.0, 260.0], "source_block_ids": ["slf_p5_left"]},
            neighbors={"prev_para_id": "p_body_source_line_fallback_p2", "next_para_id": "p_body_source_line_fallback_p5_right"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_body_source_line_fallback_p5_right",
            page_span={"start": 5, "end": 5},
            role="Body",
            section_path=None,
            text=body_text_p5_right,
            evidence_pointer={"pages": [5], "bbox_union": [306.0, 49.0, 561.0, 175.0], "source_block_ids": ["slf_p5_right"]},
            neighbors={"prev_para_id": "p_body_source_line_fallback_p5_left", "next_para_id": None},
            confidence=0.95,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "slf_title", "clean_role": "main_title"},
        {"block_id": "slf_p2", "clean_role": "body_text"},
        {"block_id": "slf_p5_left", "clean_role": "body_text"},
        {"block_id": "slf_p5_right", "clean_role": "body_text"},
    ]
    block_lines_by_block = {
        "slf_p2": [
            {"line_index": 0, "text": "The heatmap illustrated the top 100 upregulated DEGs among the three subtypes and their", "bbox_pt": [306.0, 145.0, 561.0, 157.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 1, "text": "associated top Gene Ontology biological process terms (Fig. 1F). The Venn diagram clearly", "bbox_pt": [306.0, 159.0, 561.0, 171.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 2, "text": "illustrated DEGs that were specifically upregulated or downregulated in each of the three", "bbox_pt": [306.0, 173.0, 561.0, 185.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 3, "text": "subtypes, as well as those commonly upregulated or downregulated genes (Fig. 1G, H).", "bbox_pt": [306.0, 187.0, 561.0, 199.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 4, "text": "Enrichment analysis using Kyoto Encyclopedia of Genes and Genomes (KEGG) terms revealed that", "bbox_pt": [306.0, 201.0, 561.0, 213.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 5, "text": "the pathways specifically upregulated in subtype Hw mainly included the HIF-1 pathway and", "bbox_pt": [306.0, 215.0, 561.0, 227.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 6, "text": "several metabolic pathways, while in subtype Iw, it was Cytokine-cytokine receptor", "bbox_pt": [306.0, 229.0, 561.0, 241.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 7, "text": "interaction, and in subtype Ir, they were mainly related to inflammation (Fig. 1I).", "bbox_pt": [306.0, 243.0, 561.0, 255.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 8, "text": "The genes commonly altered in the three subtypes were enriched in metabolic and inflammatory", "bbox_pt": [306.0, 257.0, 561.0, 269.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 9, "text": "pathways (Supplementary Fig. 3). In summary, subtype Hw was similar to Subtype H, primarily", "bbox_pt": [306.0, 271.0, 561.0, 283.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 10, "text": "upregulating hypoxia and metabolic pathways. Subtype Iw and", "bbox_pt": [306.0, 285.0, 561.0, 297.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 11, "text": "Ir were similar to Subtype I, mainly upregulating pathways related to inflammation.", "bbox_pt": [306.0, 299.0, 561.0, 311.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
        ],
        "slf_p5_left": [
            {"line_index": 0, "text": "With regards to blood testing, the three distinct subtypes also exhibited differences,", "bbox_pt": [40.0, 49.0, 295.0, 61.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 1, "text": "particularly in coagulation function. The results showed that subtype Ir had significantly", "bbox_pt": [40.0, 63.0, 295.0, 75.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 2, "text": "higher levels than subtype Hw in D-Dimer, Prothrombin Time, and International Normalized", "bbox_pt": [40.0, 77.0, 295.0, 89.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 3, "text": "Ratio (Fig. 2E and Supplementary Fig. 6A, B). PT activity in subtype", "bbox_pt": [40.0, 91.0, 295.0, 103.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 4, "text": "Ir was significantly lower than that in subtype Hw (Supplementary Fig. 6C). The examination", "bbox_pt": [40.0, 105.0, 295.0, 117.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 5, "text": "results of coagulation function indicated that patients with subtype Ir might have thrombosis", "bbox_pt": [40.0, 119.0, 295.0, 131.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 6, "text": "or that their blood was in a hypercoagulable state.", "bbox_pt": [40.0, 133.0, 295.0, 145.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
        ],
        "slf_p5_right": [
            {"line_index": 0, "text": "(Fig. 3D). Major modules were annotated by significantly associated KEGG terms (Fig. 3E).", "bbox_pt": [306.0, 49.0, 561.0, 61.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 1, "text": "The module most correlated with the upregulated gene set in subtype Hw was c1-23, with the", "bbox_pt": [306.0, 63.0, 561.0, 75.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 2, "text": "representative KEGG term being Glycolysis/Gluconeogenesis. For subtype", "bbox_pt": [306.0, 77.0, 561.0, 89.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 3, "text": "Ir, the module was c1-22, with the representative KEGG pathway being Cytokine-cytokine", "bbox_pt": [306.0, 91.0, 561.0, 103.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 4, "text": "receptor interaction. In the downregulated gene set, the modules most correlated with", "bbox_pt": [306.0, 105.0, 561.0, 117.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 5, "text": "subtype Hw were c1-19 and c1-27, while those most correlated with subtype Ir were c1-20", "bbox_pt": [306.0, 119.0, 561.0, 131.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
            {"line_index": 6, "text": "and c1-27 (Fig. 3F).", "bbox_pt": [306.0, 133.0, 561.0, 145.0], "font_stats": {"avg_size": 8.22, "is_bold": False, "dominant_font": "BodyFont"}},
        ],
    }

    clean_doc, _ = render_clean_document(
        paragraphs,
        annotated_blocks,
        doc_id="tang_source_block_line_fallback",
        block_lines_by_block=block_lines_by_block,
    )
    main_body = _section_text(clean_doc, "Main Body")

    assert "### 1F). The" not in clean_doc
    assert "### Enrichment analysis using Kyoto Encyclopedia of Genes and Genomes" not in clean_doc
    assert "### Subtype Iw and" not in clean_doc
    assert "### PT activity in subtype" not in clean_doc
    assert "### For subtype" not in clean_doc
    assert "The Venn diagram clearly illustrated DEGs" in main_body
    assert "Enrichment analysis using Kyoto Encyclopedia of Genes and Genomes (KEGG) terms revealed" in main_body
    assert "Subtype Iw and Ir were similar to Subtype I" in main_body
    assert "PT activity in subtype Ir was significantly lower" in main_body
    assert "For subtype Ir, the module was c1-22" in main_body


def test_render_clean_document_prefers_explicit_figure_caption_over_duplicate_heading_body_from_same_block() -> None:
    duplicate_caption = (
        "Fig. 1 | Identification of three distinct subtypes of rotator cuff tendinopathy. "
        "F The heat map depicted the relative abundance of DEGs in three tendinopathy subtypes. "
        "G Venn diagram of DEGs upregulated in each of the three tendinopathy subtypes compared to normal tendons. "
        "H Venn diagram of DEGs downregulated in each of the three tendinopathy subtypes compared to normal tendons. "
        "I Representative KEGG terms associated with genes that were specifically upregulated or downregulated in each subtype compared to normal tendons."
    )
    duplicate_body = (
        "KEGG terms associated with genes that were specifically upregulated or downregulated in each subtype "
        "compared to normal tendons. The selection criteria for DEGs were a Log2 fold change > |2| with an "
        "adjusted P value < 0.01."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_duplicate_caption",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["dc_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_duplicate_heading"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_duplicate_heading",
            page_span={"start": 3, "end": 3},
            role="Heading",
            section_path=["I Representative"],
            text="I Representative",
            evidence_pointer={"pages": [3], "bbox_union": [306.0, 551.0, 561.0, 688.0], "source_block_ids": ["dc_shared"]},
            neighbors={"prev_para_id": "p_title_duplicate_caption", "next_para_id": "p_duplicate_body"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_duplicate_body",
            page_span={"start": 3, "end": 3},
            role="Body",
            section_path=None,
            text=duplicate_body,
            evidence_pointer={"pages": [3], "bbox_union": [306.0, 551.0, 561.0, 688.0], "source_block_ids": ["dc_shared"]},
            neighbors={"prev_para_id": "p_duplicate_heading", "next_para_id": "p_explicit_caption"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_explicit_caption",
            page_span={"start": 3, "end": 3},
            role="FigureCaption",
            section_path=None,
            text=duplicate_caption,
            evidence_pointer={"pages": [3], "bbox_union": [40.0, 551.0, 561.0, 688.0], "source_block_ids": ["dc_other", "dc_shared"]},
            neighbors={"prev_para_id": "p_duplicate_body", "next_para_id": None},
            confidence=0.95,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "dc_title", "clean_role": "main_title"},
        {"block_id": "dc_shared", "clean_role": "body_text"},
        {"block_id": "dc_other", "clean_role": "figure_caption"},
    ]

    clean_doc, _ = render_clean_document(
        paragraphs,
        annotated_blocks,
        doc_id="tang_duplicate_caption_precedence",
    )
    main_body = _section_text(clean_doc, "Main Body")
    figures_section = _section_text(clean_doc, "Figures and Tables")

    assert "### I Representative" not in clean_doc
    assert "KEGG terms associated with genes that were specifically upregulated or downregulated in each subtype compared to normal tendons." not in main_body
    assert "I Representative KEGG terms associated with genes that were specifically upregulated or downregulated in each subtype compared to normal tendons." in figures_section


def test_render_clean_document_recovers_embedded_caption_and_same_page_continuation() -> None:
    caption_head = (
        "1 0 Hw Iw Ir Fig. 2 | Association of three distinct rotator cuff tendinopathy subtypes "
        "with clinical features. A The association of the three tendinopathy subtypes with the top "
        "10 clinical features."
    )
    caption_tail = (
        "tendinopathy subtypes (Hw, n = 35; Iw, n = 12; Ir, n = 16). G Representative HE images "
        "of diseased tendons from the three distinct tendinopathy subtypes. Scale bar: 100 μm. "
        "The P values for tendon and joint synovium color was determined using the Corrected Chi-Square Test."
    )
    narrative = "Subtype Ir also exhibited the highest proportion of hyperemia in the tendon and joint synovium."

    paragraphs = [
        Paragraph(
            para_id="p_title_caption",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["bc_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_caption_head"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_caption_head",
            page_span={"start": 4, "end": 4},
            role="Body",
            section_path=None,
            text=caption_head,
            evidence_pointer={"pages": [4], "bbox_union": [40.0, 540.0, 295.0, 708.0], "source_block_ids": ["bc_head"]},
            neighbors={"prev_para_id": "p_title_caption", "next_para_id": "p_noise_mid"},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
        Paragraph(
            para_id="p_noise_mid",
            page_span={"start": 4, "end": 4},
            role="Body",
            section_path=None,
            text="Ir",
            evidence_pointer={"pages": [4], "bbox_union": [478.0, 400.0, 484.0, 410.0], "source_block_ids": ["bc_noise"]},
            neighbors={"prev_para_id": "p_caption_head", "next_para_id": "p_caption_tail"},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
        Paragraph(
            para_id="p_caption_tail",
            page_span={"start": 4, "end": 4},
            role="Body",
            section_path=None,
            text=caption_tail,
            evidence_pointer={"pages": [4], "bbox_union": [306.0, 582.0, 561.0, 698.0], "source_block_ids": ["bc_tail"]},
            neighbors={"prev_para_id": "p_noise_mid", "next_para_id": "p_body_caption"},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
        Paragraph(
            para_id="p_body_caption",
            page_span={"start": 5, "end": 5},
            role="Body",
            section_path=None,
            text=narrative,
            evidence_pointer={"pages": [5], "bbox_union": [40.0, 80.0, 290.0, 160.0], "source_block_ids": ["bc_body"]},
            neighbors={"prev_para_id": "p_caption_tail", "next_para_id": None},
            confidence=0.95,
            provenance={"source": "test"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "bc_title", "clean_role": "main_title"},
        {"block_id": "bc_head", "clean_role": "body_text"},
        {"block_id": "bc_noise", "clean_role": "body_text"},
        {"block_id": "bc_tail", "clean_role": "body_text"},
        {"block_id": "bc_body", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_caption_cont")
    main_body = _section_text(clean_doc, "Main Body")
    figures_section = _section_text(clean_doc, "Figures and Tables")

    assert narrative in main_body
    assert "Fig. 2 | Association of three distinct rotator cuff tendinopathy subtypes" not in main_body
    assert "Scale bar: 100 μm." not in main_body
    assert "### Figure Captions" in clean_doc
    assert "Fig. 2 | Association of three distinct rotator cuff tendinopathy subtypes" in figures_section
    assert "Scale bar: 100 μm." in figures_section


def test_is_narrow_figure_fragment_paragraph_keeps_embedded_caption_head() -> None:
    para = Paragraph(
        para_id="p_embedded_caption_head",
        page_span={"start": 4, "end": 4},
        role="Body",
        section_path=None,
        text=(
            "1 0 Hw Iw Ir Fig. 2 | Association of three distinct rotator cuff tendinopathy "
            "subtypes with clinical features. A The association of the three tendinopathy "
            "subtypes with the top 10 clinical features"
        ),
        evidence_pointer={"pages": [4], "bbox_union": [40.0, 540.0, 295.0, 708.0], "source_block_ids": ["bc_head"]},
        neighbors={"prev_para_id": None, "next_para_id": None},
        confidence=0.5,
        provenance={"source": "test", "notes": "low_confidence no_section_path"},
    )

    result = paragraphs_mod.is_narrow_figure_fragment_paragraph(
        para,
        para.text,
        {4: [0.0, 0.0, 560.0, 800.0]},
    )

    assert result is False


def test_render_clean_document_demotes_mislabeled_figure_caption_body_paragraph() -> None:
    narrative = (
        "showing elevated gene expression related to hypoxia, while Subtype I showed elevated gene "
        "expression associated with inflammation. Next, we analyzed the differences in clinical "
        "features between the two subtypes. We ranked these features by p-value and selected the "
        "10 features with the smallest p-values for presentation in Fig. 1D. We conducted binary "
        "logistic regression analysis on these 10 clinical features and calculated their odds ratio (OR)."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_demote",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["bd_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_false_caption"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_false_caption",
            page_span={"start": 2, "end": 2},
            role="FigureCaption",
            section_path=None,
            text=narrative,
            evidence_pointer={"pages": [2], "bbox_union": [306.0, 49.0, 561.0, 742.0], "source_block_ids": ["bd_false_caption"]},
            neighbors={"prev_para_id": "p_title_demote", "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "bd_title", "clean_role": "main_title"},
        {"block_id": "bd_false_caption", "clean_role": "figure_caption"},
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_false_caption_demote")
    main_body = _section_text(clean_doc, "Main Body")
    figures_section = _section_text(clean_doc, "Figures and Tables")

    assert "We conducted binary logistic regression analysis" in main_body
    assert "We conducted binary logistic regression analysis" not in figures_section


def test_render_clean_document_demotes_panel_reference_prefix_from_figure_captions() -> None:
    narrative = (
        "(Fig. 3D). Major modules were annotated by significantly associated KEGG terms "
        "(Fig. 3E). The module most correlated with the upregulated gene set in subtype Hw "
        "was c1-23."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_panel_ref",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["bp_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_panel_ref"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_panel_ref",
            page_span={"start": 5, "end": 5},
            role="FigureCaption",
            section_path=None,
            text=narrative,
            evidence_pointer={"pages": [5], "bbox_union": [306.0, 49.0, 561.0, 175.0], "source_block_ids": ["bp_panel_ref"]},
            neighbors={"prev_para_id": "p_title_panel_ref", "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "bp_title", "clean_role": "main_title"},
        {"block_id": "bp_panel_ref", "clean_role": "figure_caption"},
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_panel_ref_demote")
    main_body = _section_text(clean_doc, "Main Body")
    figures_section = _section_text(clean_doc, "Figures and Tables")

    assert "Major modules were annotated by significantly associated KEGG terms" in main_body
    assert "Major modules were annotated by significantly associated KEGG terms" not in figures_section


def test_render_clean_document_does_not_enable_llm_refine_implicitly_from_api_key(monkeypatch) -> None:
    narrative = (
        "(Fig. 3D). Major modules were annotated by significantly associated KEGG terms "
        "(Fig. 3E). The module most correlated with the upregulated gene set in subtype Hw "
        "was c1-23."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_panel_ref_api_key",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["bp_title_api_key"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_panel_ref_api_key"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_panel_ref_api_key",
            page_span={"start": 5, "end": 5},
            role="FigureCaption",
            section_path=None,
            text=narrative,
            evidence_pointer={"pages": [5], "bbox_union": [306.0, 49.0, 561.0, 175.0], "source_block_ids": ["bp_panel_ref_api_key"]},
            neighbors={"prev_para_id": "p_title_panel_ref_api_key", "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "bp_title_api_key", "clean_role": "main_title"},
        {"block_id": "bp_panel_ref_api_key", "clean_role": "figure_caption"},
    ]

    def _drop_everything(*_args, **_kwargs):
        return (
            '{"decisions":[{"para_id":"p_panel_ref_api_key","keep":false,"confidence":0.99,"reason":"drop"}]}',
            {
                "model": "fake-paragraphs",
                "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
                "success": True,
                "error_type": "none",
                "http_status": 200,
                "prompt_chars": 10,
                "response_chars": 10,
            },
        )

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.delenv("PARAGRAPHS_LLM_REFINE", raising=False)
    monkeypatch.setattr("ingest.paragraphs.call_siliconflow_for_paragraphs", _drop_everything)

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_panel_ref_api_key")
    main_body = _section_text(clean_doc, "Main Body")

    assert "Major modules were annotated by significantly associated KEGG terms" in main_body


def test_render_clean_document_appends_same_page_lowercase_caption_continuation() -> None:
    head = (
        "Fig. 5 | Summary of characteristics of the three distinct rotator cuff tendinopathy "
        "subtypes. The major clinical features, molecular features and treatment recommendation "
        "of three distinct rotator cuff tendinopathy subtypes are summarized. The clinical "
        "features mainly include the ROM of the shoulder joint before surgery, OSS, histology, "
        "and results of blood tests. Molecular features focus on the"
    )
    tail = (
        "signal pathways with characteristic changes. The treatment recommendation is primarily "
        "directed towards specific subtypes of patients who are suitable for glucocorticoid therapy."
    )
    later_body = (
        "between red hyperemia and white non-hyperemia. Blood tests comprised commonly utilized "
        "clinical assessments, including coagulation function and lipid profile."
    )

    paragraphs = [
        Paragraph(
            para_id="p_title_lowercase_cont",
            page_span={"start": 1, "end": 1},
            role="Heading",
            section_path=None,
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            evidence_pointer={"pages": [1], "bbox_union": [80.0, 70.0, 520.0, 120.0], "source_block_ids": ["bl_title"]},
            neighbors={"prev_para_id": None, "next_para_id": "p_caption_head_lowercase"},
            confidence=0.95,
            provenance={"source": "test"},
        ),
        Paragraph(
            para_id="p_caption_head_lowercase",
            page_span={"start": 10, "end": 10},
            role="FigureCaption",
            section_path=None,
            text=head,
            evidence_pointer={"pages": [10], "bbox_union": [40.0, 296.0, 295.0, 343.0], "source_block_ids": ["bl_head"]},
            neighbors={"prev_para_id": "p_title_lowercase_cont", "next_para_id": "p_caption_tail_lowercase"},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
        Paragraph(
            para_id="p_caption_tail_lowercase",
            page_span={"start": 10, "end": 10},
            role="Body",
            section_path=None,
            text=tail,
            evidence_pointer={"pages": [10], "bbox_union": [306.0, 296.0, 560.0, 333.0], "source_block_ids": ["bl_tail"]},
            neighbors={"prev_para_id": "p_caption_head_lowercase", "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
        Paragraph(
            para_id="p_later_body",
            page_span={"start": 10, "end": 10},
            role="Body",
            section_path=None,
            text=later_body,
            evidence_pointer={"pages": [10], "bbox_union": [40.0, 498.0, 295.0, 560.0], "source_block_ids": ["bl_later"]},
            neighbors={"prev_para_id": "p_caption_tail_lowercase", "next_para_id": None},
            confidence=0.5,
            provenance={"source": "test", "notes": "low_confidence no_section_path"},
        ),
    ]
    annotated_blocks = [
        {"block_id": "bl_title", "clean_role": "main_title"},
        {"block_id": "bl_head", "clean_role": "figure_caption"},
        {"block_id": "bl_tail", "clean_role": "body_text"},
        {"block_id": "bl_later", "clean_role": "body_text"},
    ]

    clean_doc, _ = render_clean_document(paragraphs, annotated_blocks, doc_id="tang_lowercase_caption_cont")
    main_body = _section_text(clean_doc, "Main Body")
    figures_section = _section_text(clean_doc, "Figures and Tables")

    assert "signal pathways with characteristic changes" not in main_body
    assert "signal pathways with characteristic changes" in figures_section
    assert "between red hyperemia and white non-hyperemia" in main_body
    assert "between red hyperemia and white non-hyperemia" not in figures_section
