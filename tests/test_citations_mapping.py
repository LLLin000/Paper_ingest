from ingest.citations import (
    CiteAnchor,
    build_paragraph_spatial_index,
    find_nearest_para,
    map_citation_to_reference,
    should_demote_anchor_to_structural_link,
    should_demote_non_bibliography_internal_anchor,
)


def _anchor(anchor_text: str, link_type: str = "internal", anchor_type: str = "citation_marker") -> CiteAnchor:
    return CiteAnchor(
        anchor_id="a1",
        page=1,
        anchor_bbox=[0.0, 0.0, 0.0, 0.0],
        anchor_text=anchor_text,
        nearest_para_id="p1",
        link_type=link_type,
        anchor_type=anchor_type,
    )


def test_demote_single_digit_when_reference_marker_missing() -> None:
    assert should_demote_anchor_to_structural_link(_anchor("1"), marker_to_key={2: "doi:10.1000/x"})


def test_keep_single_digit_when_reference_marker_exists() -> None:
    assert not should_demote_anchor_to_structural_link(_anchor("1"), marker_to_key={1: "doi:10.1000/x"})


def test_keep_bracketed_single_digit_even_when_missing() -> None:
    assert not should_demote_anchor_to_structural_link(_anchor("[1]"), marker_to_key={2: "doi:10.1000/x"})


def test_demote_single_digit_for_external_anchor_when_missing() -> None:
    assert should_demote_anchor_to_structural_link(
        _anchor("1", link_type="external"),
        marker_to_key={2: "doi:10.1000/x"},
    )


def test_named_destination_maps_when_marker_index_missing() -> None:
    anchor = _anchor("[7]", link_type="internal")
    setattr(anchor, "dest_name", "adfm202314079-bib-0007")

    mapping = map_citation_to_reference(
        anchor=anchor,
        paragraphs={},
        reference_paras={},
        reference_entries=[],
        marker_to_key={},
    )

    assert mapping.strategy_used == "internal_dest"
    assert mapping.mapped_ref_key == "bib:dest_0007"


def test_non_bibliography_named_destination_is_demoted() -> None:
    anchor = _anchor("[1]", link_type="internal")
    setattr(anchor, "dest_name", "adfm202314079-fig-0001")

    assert should_demote_non_bibliography_internal_anchor(anchor) is True


def test_bibliography_named_destination_is_not_demoted() -> None:
    anchor = _anchor("[1]", link_type="internal")
    setattr(anchor, "dest_name", "adfm202314079-bib-0001")

    assert should_demote_non_bibliography_internal_anchor(anchor) is False


def _para(start: int, y0: float, end: int | None = None) -> dict[str, object]:
    return {
        "page_span": {"start": start, "end": end if end is not None else start},
        "evidence_pointer": {"bbox_union": [0.0, y0, 100.0, y0 + 20.0]},
    }


def test_build_paragraph_spatial_index_sorts_by_y_within_page() -> None:
    paragraphs = {
        "p3": _para(start=2, y0=300.0),
        "p1": _para(start=2, y0=100.0),
        "p2": _para(start=2, y0=200.0),
    }

    index = build_paragraph_spatial_index(paragraphs)

    assert [para_id for _, para_id in index.by_page_entries[2]] == ["p1", "p2", "p3"]


def test_find_nearest_para_prefers_same_page_candidates() -> None:
    paragraphs = {
        "same_page_far": _para(start=4, y0=500.0),
        "same_page_near": _para(start=4, y0=210.0),
        "other_page_closer_y": _para(start=3, y0=205.0),
    }
    index = build_paragraph_spatial_index(paragraphs)

    nearest = find_nearest_para([0.0, 200.0, 10.0, 210.0], page=4, paragraphs=paragraphs, paragraph_index=index)

    assert nearest == "same_page_near"


def test_find_nearest_para_falls_back_to_closest_available_page() -> None:
    paragraphs = {
        "p2": _para(start=2, y0=250.0),
        "p7": _para(start=7, y0=80.0),
    }
    index = build_paragraph_spatial_index(paragraphs)

    nearest = find_nearest_para([0.0, 240.0, 10.0, 250.0], page=5, paragraphs=paragraphs, paragraph_index=index)

    assert nearest == "p7"


def test_extractor_hot_path_no_global_page_refilter_pattern() -> None:
    extractor_source = __import__("ingest.extractor", fromlist=["run_extractor"])
    with open(extractor_source.__file__, "r", encoding="utf-8") as f:
        source_text = f.read()

    assert "[b for b in all_raw_blocks if b.page == page_num]" not in source_text
