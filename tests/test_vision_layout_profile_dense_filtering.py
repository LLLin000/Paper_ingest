from ingest.vision import BlockCandidate, select_hierarchical_fine_layout_block_ids


def test_select_hierarchical_fine_layout_block_ids_ignores_header_footer_band_and_micro_figure_text() -> None:
    blocks = [
        BlockCandidate(
            block_id="header_block",
            text="Article https://doi.org/10.1038/s41467-024-53826-w",
            bbox_pt=[30.0, 10.0, 560.0, 22.0],
            bbox_px=[60, 20, 1120, 44],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=True,
        ),
        BlockCandidate(
            block_id="axis_tick",
            text="0 5 10 Log2 of FC",
            bbox_pt=[320.0, 140.0, 420.0, 165.0],
            bbox_px=[640, 280, 840, 330],
            column_guess=2,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
        ),
        BlockCandidate(
            block_id="body_para",
            text="To better understand the heterogeneity of tendinopathy and identify potential subtypes.",
            bbox_pt=[40.0, 390.0, 295.0, 470.0],
            bbox_px=[80, 780, 590, 940],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
        ),
    ]
    projected_regions = {
        "text_block_ids": ["header_block", "axis_tick", "body_para"],
        "caption_block_ids": [],
    }
    document_profile = {
        "header_band_pt": [0.0, 0.0, 595.0, 30.0],
        "footer_band_pt": [0.0, 760.0, 595.0, 791.0],
        "body_font_profile": {"avg_size": 8.2},
    }

    selected = select_hierarchical_fine_layout_block_ids(blocks, projected_regions, document_profile)

    assert "header_block" not in selected
    assert "axis_tick" not in selected
    assert "body_para" in selected
