from ingest.citations import is_reference_paragraph


def test_reference_paragraph_does_not_trigger_on_body_with_doi_only() -> None:
    para = {
        "role": "Body",
        "section_path": ["Introduction"],
        "text": "This review appears in Nature Reviews and links to https://doi.org/10.1038/s41572-024-00492-3 for context.",
    }
    assert is_reference_paragraph(para) is False


def test_reference_paragraph_detects_numbered_reference_entry_with_signals() -> None:
    para = {
        "role": "Body",
        "section_path": ["Main Body"],
        "text": "1. Tempelhof, Rupp, Seil, R. Age-related prevalence of rotator cuff tears. J. Shoulder Surg. 1999. doi:10.1000/test",
    }
    assert is_reference_paragraph(para) is True


def test_reference_paragraph_accepts_reference_list_role() -> None:
    para = {
        "role": "ReferenceList",
        "section_path": ["Unknown"],
        "text": "non-standard reference text",
    }
    assert is_reference_paragraph(para) is True


def test_reference_paragraph_does_not_trigger_on_body_with_many_inline_markers() -> None:
    para = {
        "role": "Body",
        "section_path": ["Introduction"],
        "text": "Clinical outcomes were mixed [1] [2] [3] [4] [5] and compared across cohorts in 2024.",
    }
    assert is_reference_paragraph(para) is False
