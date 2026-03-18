from ingest.layout_analyzer import build_document_layout_profile


def test_build_document_layout_profile_detects_stable_header_footer_bands() -> None:
    profile = build_document_layout_profile(
        pages_blocks={
            2: [
                {
                    "block_id": "p2_h",
                    "page": 2,
                    "bbox_pt": [30.0, 10.0, 560.0, 22.0],
                    "text": "Article https://doi.org/10.1038/s41467-024-53826-w",
                    "font_stats": {"avg_size": 8.0, "is_bold": False},
                }
            ],
            3: [
                {
                    "block_id": "p3_h",
                    "page": 3,
                    "bbox_pt": [30.0, 11.0, 560.0, 23.0],
                    "text": "Article https://doi.org/10.1038/s41467-024-53826-w",
                    "font_stats": {"avg_size": 8.0, "is_bold": False},
                }
            ],
            4: [
                {
                    "block_id": "p4_f",
                    "page": 4,
                    "bbox_pt": [30.0, 770.0, 560.0, 786.0],
                    "text": "Nature Communications | (2024) 15:9460",
                    "font_stats": {"avg_size": 8.0, "is_bold": False},
                }
            ],
        },
        pages_dimensions={2: (595.0, 791.0), 3: (595.0, 791.0), 4: (595.0, 791.0)},
    )

    assert profile["header_band_pt"][1] < 40.0
    assert profile["footer_band_pt"][1] > 740.0


def test_build_document_layout_profile_extracts_body_heading_and_caption_priors() -> None:
    profile = build_document_layout_profile(
        pages_blocks={
            2: [
                {
                    "block_id": "p2_head",
                    "page": 2,
                    "bbox_pt": [40.0, 340.0, 295.0, 385.0],
                    "text": "Results Identification of three distinct subtypes",
                    "font_stats": {"avg_size": 12.5, "is_bold": True, "dominant_font": "BoldFont"},
                },
                {
                    "block_id": "p2_body",
                    "page": 2,
                    "bbox_pt": [40.0, 390.0, 295.0, 720.0],
                    "text": "To better understand the heterogeneity of tendinopathy and identify potential subtypes.",
                    "font_stats": {"avg_size": 8.2, "is_bold": False, "dominant_font": "BodyFont"},
                },
                {
                    "block_id": "p2_cap",
                    "page": 2,
                    "bbox_pt": [40.0, 735.0, 295.0, 770.0],
                    "text": "Fig. 1 | Identification of three distinct subtypes of rotator cuff tendinopathy.",
                    "font_stats": {"avg_size": 7.0, "is_bold": False, "dominant_font": "CaptionFont"},
                },
            ]
        },
        pages_dimensions={2: (595.0, 791.0)},
    )

    assert profile["body_font_profile"]["avg_size"] == 8.2
    assert profile["heading_font_profile"]["is_bold_ratio"] == 1.0
    assert profile["caption_font_profile"]["avg_size"] == 7.0
