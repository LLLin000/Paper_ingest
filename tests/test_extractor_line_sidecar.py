from typing import Any

import ingest.extractor as extractor


class _FakePage:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def get_text(self, mode: str) -> dict[str, Any]:
        assert mode == "dict"
        return self._payload


def test_extract_block_line_records_from_page_preserves_line_style_metadata() -> None:
    page = _FakePage(
        {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "bbox": [72.0, 72.0, 220.0, 88.0],
                            "spans": [
                                {
                                    "text": "Results",
                                    "bbox": [72.0, 72.0, 140.0, 88.0],
                                    "size": 16.0,
                                    "font": "Times-Bold",
                                }
                            ],
                        },
                        {
                            "bbox": [72.0, 92.0, 420.0, 110.0],
                            "spans": [
                                {
                                    "text": "Identification of three distinct subtypes",
                                    "bbox": [72.0, 92.0, 420.0, 110.0],
                                    "size": 13.0,
                                    "font": "Times-Bold",
                                }
                            ],
                        },
                        {
                            "bbox": [72.0, 118.0, 520.0, 136.0],
                            "spans": [
                                {
                                    "text": "To better understand the heterogeneity of tendinopathy.",
                                    "bbox": [72.0, 118.0, 520.0, 136.0],
                                    "size": 10.0,
                                    "font": "Times-Roman",
                                }
                            ],
                        },
                    ],
                }
            ]
        }
    )

    records = extractor.extract_block_line_records_from_page(page, page_num=1)

    assert len(records) == 1
    record = records[0]
    assert record["block_id"] == "p1_b0"
    assert [line["text"] for line in record["lines"]] == [
        "Results",
        "Identification of three distinct subtypes",
        "To better understand the heterogeneity of tendinopathy.",
    ]
    assert record["lines"][0]["font_stats"]["avg_size"] == 16.0
    assert record["lines"][0]["font_stats"]["is_bold"] is True
    assert record["lines"][1]["font_stats"]["dominant_font"] == "Times-Bold"
    assert record["lines"][2]["font_stats"]["is_bold"] is False
