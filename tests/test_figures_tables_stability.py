from ingest.figures_tables import (
    FigureTableAsset,
    canonicalize_caption_text,
    deduplicate_assets,
    is_figure_caption,
    is_table_caption,
)


def test_detect_inline_figure_caption_in_body_text() -> None:
    text = "Diagnosis text before caption. Fig. 4 | Surgical anatomy of rotator cuff tears."
    ok, num, remainder = is_figure_caption(text)
    assert ok is True
    assert num == "4"
    assert remainder.startswith("Surgical anatomy")


def test_detect_inline_table_caption_in_body_text() -> None:
    text = "Some narrative context. Table 2 | Quantitative criteria for diagnosis"
    ok, num, remainder = is_table_caption(text)
    assert ok is True
    assert num == "2"
    assert remainder.startswith("Quantitative criteria")


def test_canonicalize_caption_text_trims_prefix_noise() -> None:
    text = "Diagnosis section. Table 1 | Differential diagnosis of rotator cuff pathology"
    normalized = canonicalize_caption_text(text, "table")
    assert normalized == "Table 1 | Differential diagnosis of rotator cuff pathology"


def test_deduplicate_assets_prefers_caption_linked_asset_for_same_asset_id() -> None:
    assets = [
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=9,
            bbox_px=[100, 100, 500, 500],
            caption_text=None,
            caption_id=None,
            source_para_id=None,
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.4,
        ),
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=10,
            bbox_px=[120, 120, 520, 520],
            caption_text="Table 1 | Differential diagnosis",
            caption_id="para_tbl_1",
            source_para_id="para_tbl_1",
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.5,
        ),
    ]

    deduped = deduplicate_assets(assets)
    assert len(deduped) == 1
    kept = deduped[0]
    assert kept.asset_id == "tbl_001"
    assert kept.caption_id == "para_tbl_1"
    assert kept.page == 10


def test_canonicalize_caption_text_collapses_duplicate_pipe_noise() -> None:
    text = "Table 1 | | Biosafety evaluation of electrical biomaterials"
    normalized = canonicalize_caption_text(text, "table")
    assert normalized == "Table 1 | Biosafety evaluation of electrical biomaterials"

def test_propose_table_bbox_from_blocks_captures_multi_column_table_extent() -> None:
    from ingest.figures_tables import propose_table_bbox_from_blocks

    caption_para = {
        "evidence_pointer": {"bbox_union": [50.8, 75.8, 120.0, 83.8]},
        "text": "Table 2. (Continued).",
    }
    blocks_on_page = [
        {"bbox_pt": [50.8, 98.7, 290.7, 114.6], "text": "Piezoelectric Biomaterials Type Piezoelectric Coefficient Advantages"},
        {"bbox_pt": [431.2, 98.7, 537.7, 105.7], "text": "Applications References"},
        {"bbox_pt": [124.8, 121.0, 311.6, 139.0], "text": "Chitin d33 = 9.49 pC N−1 High hydrophilic material; Good biodegradability"},
        {"bbox_pt": [323.6, 163.9, 399.6, 180.8], "text": "Poor mechanical strength; Low molding efficiency"},
        {"bbox_pt": [40.0, 560.0, 550.0, 620.0], "text": "This narrative paragraph describes methods and results in full sentences."},
    ]

    bbox = propose_table_bbox_from_blocks(
        caption_para=caption_para,
        blocks_on_page=blocks_on_page,
        page_width=595.0,
        page_height=842.0,
    )

    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x0 <= 60.0
    assert x1 >= 530.0
    assert y0 >= 83.0
    assert y0 <= 110.0
    assert y1 >= 180.0


def test_propose_table_bbox_from_blocks_returns_none_without_table_like_blocks() -> None:
    from ingest.figures_tables import propose_table_bbox_from_blocks

    caption_para = {
        "evidence_pointer": {"bbox_union": [50.0, 75.0, 150.0, 85.0]},
        "text": "Table 1. (Continued).",
    }
    blocks_on_page = [
        {"bbox_pt": [60.0, 120.0, 540.0, 180.0], "text": "This section discusses clinical translation and future directions."},
        {"bbox_pt": [60.0, 200.0, 540.0, 260.0], "text": "Additional narrative text with complete sentence structure."},
    ]

    bbox = propose_table_bbox_from_blocks(
        caption_para=caption_para,
        blocks_on_page=blocks_on_page,
        page_width=595.0,
        page_height=842.0,
    )

    assert bbox is None

def test_propose_table_bbox_from_blocks_ignores_distant_sentence_cluster() -> None:
    from ingest.figures_tables import propose_table_bbox_from_blocks

    caption_para = {
        "evidence_pointer": {"bbox_union": [50.8, 75.8, 120.0, 83.8]},
        "text": "Table 2. (Continued).",
    }
    blocks_on_page = [
        {"bbox_pt": [50.8, 98.7, 290.7, 114.6], "text": "Piezoelectric Biomaterials Type Piezoelectric Coefficient Advantages"},
        {"bbox_pt": [431.2, 98.7, 537.7, 105.7], "text": "Applications References"},
        {"bbox_pt": [124.8, 121.0, 311.6, 139.0], "text": "Chitin d33 = 9.49 pC N-1 High hydrophilic material; Good biodegradability"},
        {"bbox_pt": [410.6, 163.9, 487.8, 220.7], "text": "Osteogenic differentiation; Vascularization; Immunomodulation; ECM formation"},
        {"bbox_pt": [334.2, 482.7, 487.8, 539.5], "text": "High cost; Brittle; Difficult to process Chondrogenic differentiation"},
        {
            "bbox_pt": [305.9, 569.0, 546.9, 643.8],
            "text": "abling the spontaneous generation of significant longitudinal polarization, thereby facilitating the piezoelectric effect.[170] The piezoelectric coefficient of this particular material is comparatively lower than that of conventional piezoelectric ceramics; however, it still possesses the ability to generate a piezopotential",
        },
    ]

    bbox = propose_table_bbox_from_blocks(
        caption_para=caption_para,
        blocks_on_page=blocks_on_page,
        page_width=595.0,
        page_height=842.0,
    )

    assert bbox is not None
    _, _, _, y1 = bbox
    assert y1 < 560.0

