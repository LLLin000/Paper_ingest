from ingest.paragraphs import aggregate_paragraphs


def test_aggregate_paragraphs_uses_layout_regrouping_hints_when_merge_group_missing() -> None:
    blocks = {
        "b1": {
            "block_id": "b1",
            "page": 1,
            "bbox_pt": [40.0, 100.0, 295.0, 180.0],
            "text": "Paragraph part one",
            "font_stats": {"avg_size": 8.2},
        },
        "b2": {
            "block_id": "b2",
            "page": 1,
            "bbox_pt": [40.0, 183.0, 295.0, 220.0],
            "text": "Paragraph part two",
            "font_stats": {"avg_size": 8.2},
        },
    }

    paragraphs = aggregate_paragraphs(
        blocks=blocks,
        merge_groups_by_page={1: []},
        role_labels_by_page={1: {"b1": "Body", "b2": "Body"}},
        confidence_by_page={1: 0.9},
        paragraph_regrouping_hints_by_page={1: [["b1", "b2"]]},
    )

    assert len(paragraphs) == 1
    assert paragraphs[0].text == "Paragraph part one Paragraph part two"
    assert paragraphs[0].evidence_pointer["source_block_ids"] == ["b1", "b2"]


def test_aggregate_paragraphs_does_not_duplicate_existing_vision_merge_group() -> None:
    blocks = {
        "b1": {
            "block_id": "b1",
            "page": 1,
            "bbox_pt": [40.0, 100.0, 295.0, 180.0],
            "text": "Paragraph part one",
            "font_stats": {"avg_size": 8.2},
        },
        "b2": {
            "block_id": "b2",
            "page": 1,
            "bbox_pt": [40.0, 183.0, 295.0, 220.0],
            "text": "Paragraph part two",
            "font_stats": {"avg_size": 8.2},
        },
    }

    paragraphs = aggregate_paragraphs(
        blocks=blocks,
        merge_groups_by_page={1: [{"group_id": "g1", "block_ids": ["b1", "b2"]}]},
        role_labels_by_page={1: {"b1": "Body", "b2": "Body"}},
        confidence_by_page={1: 0.9},
        paragraph_regrouping_hints_by_page={1: [["b1", "b2"]]},
    )

    assert len(paragraphs) == 1
    assert paragraphs[0].text == "Paragraph part one Paragraph part two"
    assert paragraphs[0].evidence_pointer["source_block_ids"] == ["b1", "b2"]
