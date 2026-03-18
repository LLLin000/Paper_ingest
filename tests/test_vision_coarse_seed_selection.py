from ingest.vision import BlockCandidate, select_coarse_layout_seed_blocks


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


def test_select_coarse_layout_seed_blocks_prioritizes_informative_text_over_repetitive_microblocks() -> None:
    blocks = [
        _block("header_1", [20, 10, 420, 50], "Nature Communications"),
        _block("caption_1", [30, 340, 780, 420], "Fig. 1 | Classification of distinct tendinopathy subtypes."),
        _block("body_1", [40, 80, 760, 200], "Results To better understand the heterogeneity of tendinopathy."),
    ]
    for idx in range(3, 80):
        blocks.append(_block(f"micro_{idx}", [1700, 60 + idx * 20, 1735, 75 + idx * 20], "0 5 10"))

    selected = select_coarse_layout_seed_blocks(blocks, max_blocks=12)
    selected_ids = [block.block_id for block in selected]

    assert len(selected) <= 12
    assert "header_1" in selected_ids
    assert "caption_1" in selected_ids
    assert "body_1" in selected_ids
    assert len([block_id for block_id in selected_ids if block_id.startswith("micro_")]) < len(selected_ids)
