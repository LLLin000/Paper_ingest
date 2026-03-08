from ingest import extractor


def test_compose_line_text_from_spans_keeps_superscript_digit_attached() -> None:
    spans = [
        {
            "text": "other purely carbon-based materials possessing",
            "bbox": [305.9241943359375, 240.5608367919922, 481.3848876953125, 249.60794067382812],
        },
        {"text": " sp", "bbox": [481.3848876953125, 240.5608367919922, 490.242919921875, 249.60794067382812]},
        {"text": "2", "bbox": [490.24493408203125, 239.66676330566406, 493.2337341308594, 245.69815063476562]},
        {"text": " ", "bbox": [493.2337341308594, 237.4281463623047, 495.2939453125, 249.60714721679688]},
        {"text": "hybridized or-", "bbox": [495.2939453125, 240.56004333496094, 546.866943359375, 249.60714721679688]},
    ]

    out = extractor.compose_line_text_from_spans(spans)

    assert "sp2 hybridized" in out
    assert "sp 2" not in out


def test_compose_line_text_from_spans_inserts_space_for_large_gap() -> None:
    spans = [
        {"text": "electrical", "bbox": [0.0, 0.0, 30.0, 10.0]},
        {"text": "stimulation", "bbox": [34.5, 0.0, 65.0, 10.0]},
    ]

    out = extractor.compose_line_text_from_spans(spans)

    assert out == "electrical stimulation"
