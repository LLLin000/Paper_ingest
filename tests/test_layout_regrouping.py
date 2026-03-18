from ingest.layout_analyzer import find_same_paragraph_block_groups


def test_find_same_paragraph_block_groups_merges_vertically_adjacent_same_column_blocks() -> None:
    groups = find_same_paragraph_block_groups(
        blocks=[
            {
                "block_id": "b1",
                "page": 1,
                "bbox_pt": [40.0, 100.0, 295.0, 180.0],
                "text": "Paragraph part one",
                "font_stats": {"avg_size": 8.2, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "block_id": "b2",
                "page": 1,
                "bbox_pt": [40.5, 183.0, 295.5, 220.0],
                "text": "Paragraph part two",
                "font_stats": {"avg_size": 8.2, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "block_id": "b3",
                "page": 1,
                "bbox_pt": [40.0, 260.0, 295.0, 320.0],
                "text": "New paragraph",
                "font_stats": {"avg_size": 8.2, "is_bold": False, "dominant_font": "BodyFont"},
            },
        ],
        page_profile={
            "body_line_gap_pt": 6.0,
            "column_regions": [[0.0, 0.0, 297.0, 791.0], [297.0, 0.0, 595.0, 791.0]],
            "header_band_pt": [0.0, 0.0, 0.0, 0.0],
            "footer_band_pt": [0.0, 0.0, 0.0, 0.0],
        },
    )

    assert groups == [["b1", "b2"]]


def test_find_same_paragraph_block_groups_avoids_crossing_caption_like_block() -> None:
    groups = find_same_paragraph_block_groups(
        blocks=[
            {
                "block_id": "b1",
                "page": 1,
                "bbox_pt": [40.0, 100.0, 295.0, 180.0],
                "text": "Paragraph part one",
                "font_stats": {"avg_size": 8.2, "is_bold": False, "dominant_font": "BodyFont"},
            },
            {
                "block_id": "b2",
                "page": 1,
                "bbox_pt": [40.5, 183.0, 295.5, 220.0],
                "text": "Fig. 1 | Caption block",
                "font_stats": {"avg_size": 7.0, "is_bold": False, "dominant_font": "CaptionFont"},
            },
        ],
        page_profile={
            "body_line_gap_pt": 6.0,
            "column_regions": [[0.0, 0.0, 297.0, 791.0], [297.0, 0.0, 595.0, 791.0]],
            "header_band_pt": [0.0, 0.0, 0.0, 0.0],
            "footer_band_pt": [0.0, 0.0, 0.0, 0.0],
        },
    )

    assert groups == []
