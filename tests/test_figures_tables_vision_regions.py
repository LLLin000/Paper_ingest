from ingest.figures_tables import build_vision_region_block_hints


def test_build_vision_region_block_hints_recovers_caption_and_header_footer_from_regions() -> None:
    blocks_by_page = {
        4: [
            {
                "block_id": "p4_header",
                "page": 4,
                "bbox_pt": [10.0, 8.0, 180.0, 20.0],
                "text": "Nature Communications",
            },
            {
                "block_id": "p4_caption",
                "page": 4,
                "bbox_pt": [30.0, 120.0, 260.0, 150.0],
                "text": "Fig. 1 | Classification of distinct tendinopathy subtypes.",
            },
            {
                "block_id": "p4_micro",
                "page": 4,
                "bbox_pt": [400.0, 200.0, 420.0, 210.0],
                "text": "0 5 10",
            },
        ]
    }
    vision_by_page = {
        4: {
            "page": 4,
            "source": "model",
            "role_labels": {},
            "figure_regions": [{"region_id": "fig_1", "bbox_pt": [350.0, 160.0, 500.0, 280.0]}],
            "table_regions": [],
            "caption_regions": [{"region_id": "cap_1", "bbox_pt": [0.0, 100.0, 320.0, 170.0]}],
            "header_footer_regions": [{"region_id": "hf_1", "bbox_pt": [0.0, 0.0, 250.0, 30.0]}],
        }
    }

    caption_hints, header_footer_hints = build_vision_region_block_hints(blocks_by_page, vision_by_page)

    assert 4 in caption_hints
    assert caption_hints[4][0]["text"] == "Fig. 1 | Classification of distinct tendinopathy subtypes."
    assert 4 in header_footer_hints
    assert header_footer_hints[4] == [[10.0, 8.0, 180.0, 20.0]]
