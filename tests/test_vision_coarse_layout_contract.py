from ingest.vision import validate_coarse_layout_output


def test_validate_coarse_layout_output_accepts_region_schema() -> None:
    payload = {
        "page": 4,
        "text_regions": [{"region_id": "text_1", "bbox_px": [0, 0, 100, 100]}],
        "caption_regions": [{"region_id": "cap_1", "bbox_px": [100, 100, 200, 160]}],
        "figure_regions": [{"region_id": "fig_1", "bbox_px": [100, 160, 300, 400]}],
        "table_regions": [],
        "header_footer_regions": [{"region_id": "hf_1", "bbox_px": [0, 0, 300, 20]}],
        "confidence": 0.82,
    }

    ok, reason = validate_coarse_layout_output(payload, page=4)

    assert ok is True
    assert reason == "ok"


def test_validate_coarse_layout_output_rejects_missing_required_region_arrays() -> None:
    payload = {
        "page": 4,
        "text_regions": [{"region_id": "text_1", "bbox_px": [0, 0, 100, 100]}],
        "figure_regions": [{"region_id": "fig_1", "bbox_px": [100, 160, 300, 400]}],
        "confidence": 0.82,
    }

    ok, reason = validate_coarse_layout_output(payload, page=4)

    assert ok is False
    assert reason == "missing_region_array:caption_regions"
