from ingest.vision import (
    BlockCandidate,
    VisionOutput,
    expand_vision_output_to_raw_blocks,
    load_block_line_records,
    normalize_blocks_for_vision_page,
)


def test_normalize_blocks_for_vision_page_merges_regrouped_blocks_and_filters_header_band() -> None:
    blocks = [
        BlockCandidate(
            block_id="header_1",
            text="Nature Communications | Article",
            bbox_pt=[30.0, 10.0, 560.0, 24.0],
            bbox_px=[60, 20, 1120, 48],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
        ),
        BlockCandidate(
            block_id="b1",
            text="Paragraph part one",
            bbox_pt=[40.0, 100.0, 290.0, 145.0],
            bbox_px=[80, 200, 580, 290],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
        ),
        BlockCandidate(
            block_id="b2",
            text="Paragraph part two",
            bbox_pt=[40.0, 148.0, 290.0, 188.0],
            bbox_px=[80, 296, 580, 376],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
        ),
        BlockCandidate(
            block_id="b3",
            text="Results",
            bbox_pt=[310.0, 100.0, 500.0, 128.0],
            bbox_px=[620, 200, 1000, 256],
            column_guess=2,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
        ),
    ]

    normalized, preset_roles = normalize_blocks_for_vision_page(
        page=1,
        blocks=blocks,
        document_profile={
            "header_band_pt": [0.0, 0.0, 595.0, 30.0],
            "footer_band_pt": [0.0, 760.0, 595.0, 792.0],
        },
        paragraph_regrouping_hints_by_page={1: [["b1", "b2"]]},
    )

    assert preset_roles == {"header_1": "HeaderFooter"}
    assert [block.block_id for block in normalized] == ["p001_rg000", "b3"]
    assert normalized[0].source_block_ids == ["b1", "b2"]
    assert normalized[0].text == "Paragraph part one Paragraph part two"
    assert normalized[1].source_block_ids == ["b3"]


def test_expand_vision_output_to_raw_blocks_restores_source_block_ids_and_hidden_roles() -> None:
    normalized_blocks = [
        BlockCandidate(
            block_id="p001_rg000",
            text="Paragraph part one Paragraph part two",
            bbox_pt=[40.0, 100.0, 290.0, 188.0],
            bbox_px=[80, 200, 580, 376],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["b1", "b2"],
        ),
        BlockCandidate(
            block_id="b3",
            text="Results",
            bbox_pt=[310.0, 100.0, 500.0, 128.0],
            bbox_px=[620, 200, 1000, 256],
            column_guess=2,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
            source_block_ids=["b3"],
        ),
    ]
    raw_blocks = [
        BlockCandidate(
            block_id="header_1",
            text="Nature Communications | Article",
            bbox_pt=[30.0, 10.0, 560.0, 24.0],
            bbox_px=[60, 20, 1120, 48],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["header_1"],
        ),
        BlockCandidate(
            block_id="b1",
            text="Paragraph part one",
            bbox_pt=[40.0, 100.0, 290.0, 145.0],
            bbox_px=[80, 200, 580, 290],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["b1"],
        ),
        BlockCandidate(
            block_id="b2",
            text="Paragraph part two",
            bbox_pt=[40.0, 148.0, 290.0, 188.0],
            bbox_px=[80, 296, 580, 376],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["b2"],
        ),
        BlockCandidate(
            block_id="b3",
            text="Results",
            bbox_pt=[310.0, 100.0, 500.0, 128.0],
            bbox_px=[620, 200, 1000, 256],
            column_guess=2,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
            source_block_ids=["b3"],
        ),
    ]
    output = VisionOutput(
        page=1,
        reading_order=["p001_rg000", "b3"],
        merge_groups=[{"group_id": "g1", "block_ids": ["p001_rg000", "b3"]}],
        role_labels={"p001_rg000": "Body", "b3": "Heading"},
        confidence=0.9,
        fallback_used=False,
    )

    expanded = expand_vision_output_to_raw_blocks(
        output,
        normalized_blocks,
        raw_blocks,
        hidden_role_labels={"header_1": "HeaderFooter"},
    )

    assert expanded.reading_order == ["b1", "b2", "b3"]
    assert expanded.merge_groups == [{"group_id": "g1", "block_ids": ["b1", "b2", "b3"]}]
    assert expanded.role_labels == {
        "b1": "Body",
        "b2": "Body",
        "b3": "Heading",
        "header_1": "HeaderFooter",
    }


def test_normalize_blocks_for_vision_page_splits_mixed_block_using_block_lines() -> None:
    blocks = [
        BlockCandidate(
            block_id="p1_b0",
            text="Results To better understand the heterogeneity of tendinopathy.",
            bbox_pt=[40.0, 100.0, 300.0, 170.0],
            bbox_px=[80, 200, 600, 340],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
        )
    ]

    normalized, preset_roles = normalize_blocks_for_vision_page(
        page=1,
        blocks=blocks,
        document_profile={
            "heading_font_profile": {"avg_size": 16.0, "bold_ratio": 0.9, "dominant_fonts": ["Helvetica-Bold"]},
            "body_font_profile": {"avg_size": 10.0, "bold_ratio": 0.1, "dominant_fonts": ["Helvetica"]},
        },
        block_lines_by_block={
            "p1_b0": [
                {
                    "line_index": 0,
                    "text": "Results",
                    "bbox_pt": [40.0, 100.0, 160.0, 122.0],
                    "font_stats": {"avg_size": 18.0, "is_bold": True, "dominant_font": "Helvetica-Bold"},
                },
                {
                    "line_index": 1,
                    "text": "To better understand the heterogeneity of tendinopathy.",
                    "bbox_pt": [40.0, 128.0, 300.0, 170.0],
                    "font_stats": {"avg_size": 10.0, "is_bold": False, "dominant_font": "Helvetica"},
                },
            ]
        },
    )

    assert preset_roles == {}
    assert [block.block_id for block in normalized] == ["p1_b0__nl0", "p1_b0__nl1"]
    assert normalized[0].text == "Results"
    assert normalized[0].source_block_ids == ["p1_b0"]
    assert normalized[1].text == "To better understand the heterogeneity of tendinopathy."
    assert normalized[1].source_block_ids == ["p1_b0"]


def test_expand_vision_output_to_raw_blocks_turns_line_split_headings_into_embedded_hints() -> None:
    normalized_blocks = [
        BlockCandidate(
            block_id="p1_b0__nl0",
            text="Results",
            bbox_pt=[40.0, 100.0, 160.0, 122.0],
            bbox_px=[80, 200, 320, 244],
            column_guess=1,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
        ),
        BlockCandidate(
            block_id="p1_b0__nl1",
            text="To better understand the heterogeneity of tendinopathy.",
            bbox_pt=[40.0, 128.0, 300.0, 170.0],
            bbox_px=[80, 256, 600, 340],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
        ),
    ]
    raw_blocks = [
        BlockCandidate(
            block_id="p1_b0",
            text="Results To better understand the heterogeneity of tendinopathy.",
            bbox_pt=[40.0, 100.0, 300.0, 170.0],
            bbox_px=[80, 200, 600, 340],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
        )
    ]
    output = VisionOutput(
        page=1,
        reading_order=["p1_b0__nl0", "p1_b0__nl1"],
        merge_groups=[{"group_id": "g1", "block_ids": ["p1_b0__nl0", "p1_b0__nl1"]}],
        role_labels={"p1_b0__nl0": "Heading", "p1_b0__nl1": "Body"},
        confidence=0.9,
        fallback_used=False,
    )

    expanded = expand_vision_output_to_raw_blocks(
        output,
        normalized_blocks,
        raw_blocks,
        hidden_role_labels=None,
    )

    assert expanded.reading_order == ["p1_b0"]
    assert expanded.merge_groups == [{"group_id": "g1", "block_ids": ["p1_b0"]}]
    assert expanded.role_labels == {"p1_b0": "Body"}
    assert expanded.embedded_headings == [{"block_id": "p1_b0", "heading_text": "Results", "confidence": 1.0}]


def test_normalize_blocks_for_vision_page_does_not_split_compact_panel_label_cluster() -> None:
    blocks = [
        BlockCandidate(
            block_id="p3_b37",
            text="Ir Inflammation red tendon",
            bbox_pt=[487.78, 253.95, 533.01, 279.32],
            bbox_px=[975, 507, 1066, 558],
            column_guess=2,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["p3_b37"],
        )
    ]

    normalized, preset_roles = normalize_blocks_for_vision_page(
        page=3,
        blocks=blocks,
        document_profile={
            "heading_font_profile": {"avg_size": 10.7, "bold_ratio": 0.1, "dominant_fonts": ["Times-Bold"]},
            "body_font_profile": {"avg_size": 8.2, "bold_ratio": 0.0, "dominant_fonts": ["Times-Roman"]},
        },
        block_lines_by_block={
            "p3_b37": [
                {
                    "line_index": 0,
                    "text": "Ir",
                    "bbox_pt": [506.80, 253.95, 513.98, 264.70],
                    "font_stats": {"avg_size": 10.75, "is_bold": True, "dominant_font": "Arial-BoldMT"},
                },
                {
                    "line_index": 1,
                    "text": "Inflammation",
                    "bbox_pt": [487.78, 264.05, 533.01, 272.15],
                    "font_stats": {"avg_size": 7.89, "is_bold": True, "dominant_font": "Arial-BoldMT"},
                },
                {
                    "line_index": 2,
                    "text": "red tendon",
                    "bbox_pt": [491.29, 271.21, 529.50, 279.32],
                    "font_stats": {"avg_size": 7.89, "is_bold": True, "dominant_font": "Arial-BoldMT"},
                },
            ]
        },
    )

    assert preset_roles == {}
    assert [block.block_id for block in normalized] == ["p3_b37"]
    assert normalized[0].text == "Ir Inflammation red tendon"


def test_normalize_blocks_for_vision_page_splits_page1_front_matter_title_block() -> None:
    blocks = [
        BlockCandidate(
            block_id="p1_b0",
            text=(
                "Article https://doi.org/10.1038/s41467-024-53826-w "
                "Classification of distinct tendinopathy subtypes for precision therapeutics"
            ),
            bbox_pt=[36.0, 20.0, 560.0, 82.0],
            bbox_px=[72, 40, 1120, 164],
            column_guess=1,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
        )
    ]

    normalized, hidden_roles = normalize_blocks_for_vision_page(
        page=1,
        blocks=blocks,
        document_profile={
            "heading_font_profile": {
                "avg_size": 16.0,
                "bold_ratio": 0.9,
                "dominant_fonts": ["Helvetica-Bold"],
            },
            "body_font_profile": {
                "avg_size": 10.0,
                "bold_ratio": 0.1,
                "dominant_fonts": ["Helvetica"],
            },
        },
        block_lines_by_block={
            "p1_b0": [
                {
                    "line_index": 0,
                    "text": "Article",
                    "bbox_pt": [36.0, 20.0, 86.0, 33.0],
                    "font_stats": {"avg_size": 10.5, "is_bold": True, "dominant_font": "Helvetica-Bold"},
                },
                {
                    "line_index": 1,
                    "text": "https://doi.org/10.1038/s41467-024-53826-w",
                    "bbox_pt": [360.0, 20.0, 560.0, 33.0],
                    "font_stats": {"avg_size": 9.8, "is_bold": False, "dominant_font": "Helvetica"},
                },
                {
                    "line_index": 2,
                    "text": "Classification of distinct tendinopathy",
                    "bbox_pt": [36.0, 40.0, 315.0, 60.0],
                    "font_stats": {"avg_size": 18.0, "is_bold": True, "dominant_font": "Helvetica-Bold"},
                },
                {
                    "line_index": 3,
                    "text": "subtypes for precision therapeutics",
                    "bbox_pt": [36.0, 61.0, 285.0, 82.0],
                    "font_stats": {"avg_size": 18.0, "is_bold": True, "dominant_font": "Helvetica-Bold"},
                },
            ]
        },
    )

    assert hidden_roles == {}
    assert [block.text for block in normalized] == [
        "Article",
        "https://doi.org/10.1038/s41467-024-53826-w",
        "Classification of distinct tendinopathy subtypes for precision therapeutics",
    ]
    assert [block.is_heading_candidate for block in normalized] == [False, False, True]


def test_normalize_blocks_for_vision_page_does_not_split_page1_author_name_block() -> None:
    blocks = [
        BlockCandidate(
            block_id="p1_b1",
            text="Chenqi Tang 1,2,3,4,5,6,7,14, Zetao Wang1,2,3,4,6,7,8,14",
            bbox_pt=[36.0, 90.0, 560.0, 130.0],
            bbox_px=[72, 180, 1120, 260],
            column_guess=1,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b1"],
        )
    ]

    normalized, hidden_roles = normalize_blocks_for_vision_page(
        page=1,
        blocks=blocks,
        document_profile={
            "heading_font_profile": {
                "avg_size": 16.0,
                "bold_ratio": 0.9,
                "dominant_fonts": ["Helvetica-Bold"],
            },
            "body_font_profile": {
                "avg_size": 10.0,
                "bold_ratio": 0.1,
                "dominant_fonts": ["Helvetica"],
            },
        },
        block_lines_by_block={
            "p1_b1": [
                {
                    "line_index": 0,
                    "text": "Chenqi Tang",
                    "bbox_pt": [36.0, 90.0, 120.0, 103.0],
                    "font_stats": {"avg_size": 12.0, "is_bold": True, "dominant_font": "Helvetica-Bold"},
                },
                {
                    "line_index": 1,
                    "text": "1,2,3,4,5,6,7,14, Zetao Wang1,2,3,4,6,7,8,14",
                    "bbox_pt": [36.0, 104.0, 560.0, 130.0],
                    "font_stats": {"avg_size": 9.0, "is_bold": False, "dominant_font": "Helvetica"},
                },
            ]
        },
    )

    assert hidden_roles == {}
    assert [block.block_id for block in normalized] == ["p1_b1"]


def test_expand_vision_output_to_raw_blocks_keeps_only_title_hint_from_front_matter_split() -> None:
    normalized_blocks = [
        BlockCandidate(
            block_id="p1_b0__fm0",
            text="Article",
            bbox_pt=[36.0, 20.0, 86.0, 33.0],
            bbox_px=[72, 40, 172, 66],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
            source_line_span=[0, 0],
            source_kind="front_matter_split",
        ),
        BlockCandidate(
            block_id="p1_b0__fm1",
            text="https://doi.org/10.1038/s41467-024-53826-w",
            bbox_pt=[360.0, 20.0, 560.0, 33.0],
            bbox_px=[720, 40, 1120, 66],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
            source_line_span=[1, 1],
            source_kind="front_matter_split",
        ),
        BlockCandidate(
            block_id="p1_b0__fm2",
            text="Classification of distinct tendinopathy subtypes for precision therapeutics",
            bbox_pt=[36.0, 40.0, 315.0, 82.0],
            bbox_px=[72, 80, 630, 164],
            column_guess=1,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
            source_line_span=[2, 3],
            source_kind="front_matter_split",
        ),
    ]
    raw_blocks = [
        BlockCandidate(
            block_id="p1_b0",
            text=(
                "Article https://doi.org/10.1038/s41467-024-53826-w "
                "Classification of distinct tendinopathy subtypes for precision therapeutics"
            ),
            bbox_pt=[36.0, 20.0, 560.0, 82.0],
            bbox_px=[72, 40, 1120, 164],
            column_guess=1,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
            source_block_ids=["p1_b0"],
        )
    ]
    output = VisionOutput(
        page=1,
        reading_order=["p1_b0__fm0", "p1_b0__fm1", "p1_b0__fm2"],
        merge_groups=[{"group_id": "g1", "block_ids": ["p1_b0__fm0", "p1_b0__fm1", "p1_b0__fm2"]}],
        role_labels={"p1_b0__fm0": "Heading", "p1_b0__fm1": "ReferenceList", "p1_b0__fm2": "Heading"},
        confidence=0.9,
        fallback_used=False,
    )

    expanded = expand_vision_output_to_raw_blocks(
        output,
        normalized_blocks,
        raw_blocks,
        hidden_role_labels=None,
    )

    assert expanded.role_labels == {"p1_b0": "Body"}
    assert expanded.embedded_headings == [
        {
            "block_id": "p1_b0",
            "heading_text": "Classification of distinct tendinopathy subtypes for precision therapeutics",
            "confidence": 1.0,
        }
    ]
