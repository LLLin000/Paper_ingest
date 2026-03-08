from ingest.citations import extract_inline_anchors


def _para(text: str, para_id: str = "p1") -> dict[str, dict[str, object]]:
    return {
        para_id: {
            "para_id": para_id,
            "role": "Body",
            "section_path": ["Introduction"],
            "text": text,
            "page_span": {"start": 1, "end": 1},
            "evidence_pointer": {"bbox_union": [0.0, 0.0, 100.0, 20.0]},
        }
    }


def test_extract_inline_anchors_skips_single_parenthetical_numbers() -> None:
    paragraphs = _para("Results improved (1) with n=20 compared with baseline.")
    anchors = extract_inline_anchors(paragraphs)
    assert anchors == []


def test_extract_inline_anchors_keeps_parenthetical_ranges_or_lists() -> None:
    paragraphs = _para("Prior studies (1,2;4-5) support this finding.")
    anchors = extract_inline_anchors(paragraphs)
    marker_texts = sorted(a.anchor_text for a in anchors)
    assert marker_texts == ["[1]", "[2]", "[4]", "[5]"]


def test_extract_inline_anchors_keeps_single_parenthetical_with_ref_context() -> None:
    paragraphs = _para("See ref (3) for methodological details.")
    anchors = extract_inline_anchors(paragraphs)
    marker_texts = sorted(a.anchor_text for a in anchors)
    assert marker_texts == ["[3]"]
