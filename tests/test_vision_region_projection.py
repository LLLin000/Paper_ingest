from ingest.vision import (
    BlockCandidate,
    build_coarse_region_fallback_output,
    project_blocks_to_layout_regions,
    prune_non_narrative_text_block_ids,
    select_hierarchical_fine_layout_block_ids,
)


def _block(block_id: str, bbox_px: list[int], text: str) -> BlockCandidate:
    return BlockCandidate(
        block_id=block_id,
        text=text,
        bbox_pt=[float(v) for v in bbox_px],
        bbox_px=bbox_px,
        column_guess=1,
        is_heading_candidate=False,
        is_header_footer_candidate=False,
    )


def test_project_blocks_to_layout_regions_excludes_figure_microblocks_from_text_layout() -> None:
    blocks = [
        _block("body_1", [40, 40, 260, 120], "Results To better understand the heterogeneity of tendinopathy."),
        _block("caption_1", [40, 130, 260, 170], "Fig. 1 | Summary of the distinct subtypes."),
        _block("axis_tick_1", [500, 120, 520, 134], "0"),
        _block("axis_tick_2", [500, 160, 520, 174], "5"),
        _block("legend_1", [560, 260, 700, 284], "Inflammation"),
    ]
    coarse_layout = {
        "page": 1,
        "text_regions": [{"region_id": "text_1", "bbox_px": [0, 0, 320, 125]}],
        "caption_regions": [{"region_id": "cap_1", "bbox_px": [0, 125, 320, 190]}],
        "figure_regions": [{"region_id": "fig_1", "bbox_px": [320, 0, 900, 700]}],
        "table_regions": [],
        "header_footer_regions": [],
        "confidence": 0.9,
    }

    projected = project_blocks_to_layout_regions(blocks, coarse_layout)

    assert projected["text_block_ids"] == ["body_1"]
    assert projected["caption_block_ids"] == ["caption_1"]
    assert set(projected["figure_block_ids"]) == {"axis_tick_1", "axis_tick_2", "legend_1"}


def test_prune_non_narrative_text_block_ids_drops_distant_microblock_cluster() -> None:
    blocks = [
        _block("body_1", [40, 40, 320, 120], "Results To better understand the heterogeneity of tendinopathy."),
        _block("body_2", [40, 130, 360, 190], "We combined transcriptomic features with clinical characteristics."),
        _block("axis_tick_1", [1800, 140, 1820, 154], "0"),
        _block("axis_tick_2", [1800, 190, 1820, 204], "5"),
        _block("axis_tick_3", [1800, 240, 1820, 254], "10"),
        _block("legend_1", [1880, 300, 1980, 328], "Inflammation"),
        _block("legend_2", [1880, 340, 1960, 368], "Hypoxia"),
    ]

    kept_ids = prune_non_narrative_text_block_ids(
        blocks,
        ["body_1", "body_2", "axis_tick_1", "axis_tick_2", "axis_tick_3", "legend_1", "legend_2"],
    )

    assert kept_ids == ["body_1", "body_2"]


def test_project_blocks_to_layout_regions_uses_overlap_when_center_misses_caption_region() -> None:
    blocks = [
        _block("caption_1", [40, 40, 420, 220], "Fig. 1 | Summary of the distinct subtypes."),
    ]
    coarse_layout = {
        "page": 1,
        "text_regions": [],
        "caption_regions": [{"region_id": "cap_1", "bbox_px": [320, 60, 520, 240]}],
        "figure_regions": [],
        "table_regions": [],
        "header_footer_regions": [],
        "confidence": 0.9,
    }

    projected = project_blocks_to_layout_regions(blocks, coarse_layout)

    assert projected["caption_block_ids"] == ["caption_1"]
    assert projected["unassigned_block_ids"] == []


def test_select_hierarchical_fine_layout_block_ids_falls_back_to_projected_text_subset_when_pruned_empty() -> None:
    blocks = [
        _block("axis_tick_1", [1800, 140, 1820, 154], "0"),
        _block("axis_tick_2", [1800, 190, 1820, 204], "5"),
        _block("axis_tick_3", [1800, 240, 1820, 254], "10"),
        _block("legend_1", [1880, 300, 1980, 328], "Inflammation"),
        _block("legend_2", [1880, 340, 1960, 368], "Hypoxia"),
        _block("caption_1", [40, 360, 320, 420], "Fig. 1 | Summary of the distinct subtypes."),
    ]
    projected = {
        "text_block_ids": ["axis_tick_1", "axis_tick_2", "axis_tick_3", "legend_1", "legend_2"],
        "caption_block_ids": ["caption_1"],
        "figure_block_ids": [],
        "table_block_ids": [],
        "header_footer_block_ids": [],
        "unassigned_block_ids": [],
    }

    selected = select_hierarchical_fine_layout_block_ids(blocks, projected)

    assert selected == ["axis_tick_1", "axis_tick_2", "axis_tick_3", "legend_1", "legend_2", "caption_1"]


def test_build_coarse_region_fallback_output_marks_non_narrative_unassigned_cluster_as_sidebar() -> None:
    blocks = [
        _block("header_1", [20, 10, 400, 40], "Nature Communications"),
        _block("axis_tick_1", [1800, 140, 1820, 154], "0"),
        _block("axis_tick_2", [1800, 190, 1820, 204], "5"),
        _block("axis_tick_3", [1800, 240, 1820, 254], "10"),
        _block("legend_1", [1880, 300, 1980, 328], "Inflammation"),
        _block("legend_2", [1880, 340, 1960, 368], "Hypoxia"),
    ]
    projected = {
        "text_block_ids": [],
        "caption_block_ids": [],
        "figure_block_ids": [],
        "table_block_ids": [],
        "header_footer_block_ids": ["header_1"],
        "unassigned_block_ids": ["axis_tick_1", "axis_tick_2", "axis_tick_3", "legend_1", "legend_2"],
    }

    output = build_coarse_region_fallback_output(page=1, blocks=blocks, projected_regions=projected)

    assert output.source == "coarse_fallback"
    assert output.fallback_used is True
    assert output.role_labels["header_1"] == "HeaderFooter"
    assert output.role_labels["axis_tick_1"] == "Sidebar"
    assert output.role_labels["legend_1"] == "Sidebar"
