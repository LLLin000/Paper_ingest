from ingest.reference_providers_impl import (
    _normalize_crossref_reference_entry,
    _fetch_crossref,
    _rewrite_author_prefixed_title,
    NormalizedReference,
)
import ingest.reference_providers_impl as prov
import urllib.parse as _up


def test_normalize_crossref_reference_entry_structured():
    ref = {
        "article-title": "An interesting result",
        "author": "Doe, J; Smith, A",
        "year": "2020",
        "DOI": "10.1234/example.doi",
        "journal-title": "Journal of Testing",
        "URL": "https://example.org/ref",
    }
    nr = _normalize_crossref_reference_entry(ref, confidence=0.85)
    assert isinstance(nr, NormalizedReference)
    assert nr.title == "An interesting result"
    assert nr.doi == "10.1234/example.doi"
    assert nr.year == 2020
    assert "Journal" in nr.venue


def test_fetch_crossref_references_list(monkeypatch):
    # Mock payload where Crossref returns a message with 'reference' list
    sample_payload = {
        "message": {
            "title": ["Paper Title"],
            "reference": [
                {"article-title": "Ref One", "DOI": "10.1/refone", "author": "A"},
                {"unstructured": "Ref Two - something", "author": "B"},
                {"DOI": "10.1/refdoi-only"},
            ],
        }
    }

    calls: list[str] = []

    def _mock_request_json(url, timeout_sec=5.0):
        calls.append(url)
        return sample_payload, None

    monkeypatch.setattr(prov, "_request_json", _mock_request_json)
    # ensure hydrations are allowed (env override)
    monkeypatch.setenv("CROSSREF_HYDRATE_MAX", "20")

    refs, status = _fetch_crossref({"doi": "10.1000/testdoi", "title": ""})
    assert status["provider"] == "crossref"
    assert status["status"] == "ok"
    assert status["reason"] == "references"
    assert status["records"] == 3
    assert len(refs) == 3
    assert all(isinstance(r, NormalizedReference) for r in refs)
    # initial crossref call should have been made (URI-encoding may change exact string)
    assert any("/works/" in c and "testdoi" in c for c in calls)


def test_normalize_crossref_reference_entry_keeps_doi_only_reference():
    ref = {
        "DOI": "10.1234/doi-only",
    }
    nr = _normalize_crossref_reference_entry(ref, confidence=0.85)
    assert nr.doi == "10.1234/doi-only"


def test_title_sanitization_removes_author_prefix():
    ref = {"unstructured": "Davies, M. R.; Smith, J. et al. - A study of something interesting"}
    nr = _normalize_crossref_reference_entry(ref, confidence=0.85)
    assert "Davies" not in nr.title
    assert nr.title.startswith("A study") or "study" in nr.title


def test_unstructured_author_prefix_extracts_clean_title_and_authors():
    raw = (
        "Abate, M., Smith, J. et al. Oxidative stress and inflammation in tendon injury. "
        "J Shoulder Elbow Surg. 2021;30:100-110."
    )
    nr = _normalize_crossref_reference_entry({"unstructured": raw}, confidence=0.85)
    assert nr.title.startswith("Oxidative stress and inflammation in tendon injury")
    assert nr.authors
    assert nr.authors[0].startswith("Abate, M")
    assert not nr.title.startswith("Abate,")


def test_title_sanitization_keeps_normal_title():
    ref = {"unstructured": "A clear and normal title without author prefix"}
    nr = _normalize_crossref_reference_entry(ref, confidence=0.85)
    assert nr.title == "A clear and normal title without author prefix"


def test_unstructured_with_no_clear_split_keeps_previous_behavior():
    raw = "Complex adaptive systems in medicine and biology"
    nr = _normalize_crossref_reference_entry({"unstructured": raw}, confidence=0.85)
    assert nr.title == raw
    assert nr.authors == []


def test_structured_entry_does_not_use_unstructured_override():
    ref = {
        "article-title": "Structured title should win",
        "author": "Doe, J; Smith, A",
        "unstructured": "Abate, M., Smith, J. et al. Oxidative stress and inflammation in tendon injury.",
    }
    nr = _normalize_crossref_reference_entry(ref, confidence=0.85)
    assert nr.title == "Structured title should win"
    assert nr.authors == ["Doe", "J", "Smith", "A"]


def test_second_pass_extracts_title_from_author_prefixed_title_string():
    raw = (
        "Surname, X., Surname, Y. Actual title sentence with enough words for parsing. "
        "Journal of Testing. 2020;10:100-110."
    )
    nr = _normalize_crossref_reference_entry({"unstructured": raw}, confidence=0.85)
    assert nr.title == "Actual title sentence with enough words for parsing"


def test_second_pass_keeps_title_when_next_segment_is_not_title():
    raw = "Surname, X., Surname, Y. J Shoulder Elbow Surg. 2020;10:100-110."
    nr = _normalize_crossref_reference_entry({"unstructured": raw}, confidence=0.85)
    assert nr.title == raw


def test_rewrite_author_prefixed_title_positive_multi_author_case():
    raw = (
        "Abate, M., Smith, J., Brown, R. Oxidative stress and inflammation in tendon injury progression. "
        "J Shoulder Elbow Surg. 2021;30:100-110."
    )
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == "Oxidative stress and inflammation in tendon injury progression"


def test_rewrite_author_prefixed_title_negative_case():
    raw = "Abate, M., Smith, J. J Shoulder Elbow Surg. 2021;30:100-110."
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == raw


def test_rewrite_author_prefixed_title_preserves_clean_title():
    raw = "A clean and descriptive title for rotator cuff outcomes"
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == raw


def test_rewrite_author_prefixed_title_full_citation_long_title():
    raw = (
        "Fucentese, S. F., von Roll, A. L., Pfirrmann, C. W. & Gerber, C. "
        "Evolution of nonoperatively treated symptomatic isolated full-thickness supraspinatus tears. "
        "J. Bone Joint Surg. Am. 94, 801-808 (2012)."
    )
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == "Evolution of nonoperatively treated symptomatic isolated full-thickness supraspinatus tears"


def test_rewrite_author_prefixed_title_full_citation_one_word_title():
    raw = "Millar, N. L. et al. Tendinopathy. Nat. Rev. Dis. Primers 7, 1 (2021)."
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == "Tendinopathy"


def test_rewrite_author_prefixed_title_full_citation_short_valid_phrase():
    raw = "Tresoldi, I. et al. Tendon's ultrastructure. Muscles Ligaments Tendons J. 3, 2-6 (2013)."
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == "Tendon's ultrastructure"


def test_rewrite_author_prefixed_title_full_citation_colon_and_numbers():
    raw = (
        "Le, B. T., Wu, X. L., Lam, P. H. & Murrell, G. A. "
        "Factors predicting rotator cuff retears: analysis of 1000 repairs in 2 cohorts. "
        "Am. J. Sports Med. 42, 1134-1142 (2014)."
    )
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == "Factors predicting rotator cuff retears: analysis of 1000 repairs in 2 cohorts"


def test_rewrite_author_prefixed_title_full_citation_negative_venue_second_sentence():
    raw = "Hamada, K., Yamanaka, K. Clin. Orthop. Relat. Res. 469, 2452 (2011)."
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == raw


def test_rewrite_author_prefixed_title_residual_full_citation_patterns():
    examples = [
        (
            "Hamada, K., Yamanaka, K., Uchiyama, Y., Mikasa, T. & Mikasa, M. "
            "A radiographic classification of massive rotator cuff tear arthritis. "
            "Clin. Orthop. Relat. Res. 469, 2452 (2011).",
            "A radiographic classification of massive rotator cuff tear arthritis",
        ),
        (
            "Kim, S. K., Nguyen, C., Jones, K. B. & Tashjian, R. Z. "
            "A genome-wide association study for shoulder impingement and rotator cuff disease. "
            "J. Shoulder Elbow Surg. 30, 2134-2145 (2021).",
            "A genome-wide association study for shoulder impingement and rotator cuff disease",
        ),
        (
            "Sheean, A. J., Hartzler, R. U. & Burkhart, S. S. "
            "Arthroscopic rotator cuff repair in 2019: linked, double row repair for achieving higher healing rates and optimal clinical outcomes. "
            "Arthroscopy 35, 2749-2755 (2019).",
            "Arthroscopic rotator cuff repair in 2019: linked, double row repair for achieving higher healing rates and optimal clinical outcomes",
        ),
        (
            "Yanik, E. L., Chamberlain, A. M. & Keener, J. D. "
            "Trends in rotator cuff repair rates and comorbidity burden among commercially insured patients younger than the age of 65 years, United States 2007-2016. "
            "JSES Rev. Rep. Tech. 1, 309-316 (2021).",
            "Trends in rotator cuff repair rates and comorbidity burden among commercially insured patients younger than the age of 65 years, United States 2007-2016",
        ),
    ]

    for raw, expected in examples:
        rewritten = _rewrite_author_prefixed_title(raw)
        assert rewritten == expected


def test_rewrite_author_prefixed_title_negative_non_title_second_sentence():
    raw = "Kim, S. K., Nguyen, C. J. Shoulder Elbow Surg. 30, 2134-2145 (2021)."
    rewritten = _rewrite_author_prefixed_title(raw)
    assert rewritten == raw


def test_title_sanitization_handles_initials_prefix():
    # Titles that start with initials like "M. R. Davies - Title" should strip
    ref = {"unstructured": "M. R. Davies - On the role of X in Y"}
    nr = _normalize_crossref_reference_entry(ref, confidence=0.85)
    assert "Davies" not in nr.title
    assert nr.title.startswith("On the role") or "role" in nr.title


def test_title_sanitization_does_not_strip_when_not_author_like():
    # Ensure we do not strip content that looks like a legitimate title start
    ref = {"unstructured": "DNA-binding proteins: a comprehensive review"}
    nr = _normalize_crossref_reference_entry(ref, confidence=0.85)
    # Should remain unchanged because it does not match author-prefix pattern
    assert nr.title == "DNA-binding proteins: a comprehensive review"


def test_regression_false_positive_examples():
    # The following cases were previously over-stripped; ensure we no longer
    # accept cleaned titles that start with conjunction fragments or lowercase
    examples = [
        ("A., Toms, A. P. & Hing, C. B. The diagnostic accuracy of MRI for the detection of partial- and full-thickness rotator cuff tears in adults. Magn. Reson. Imaging 30, 336-346 (2012).",),
        ("Albright, L. A. & Tashjian, R. Z. Significant association of full-thickness rotator cuff tears and estrogen-related receptor-β (ESRRB). J. Shoulder Elbow Surg. 24, e31-e35 (2015).",),
        ("Amer, Y. & Shen, K. C. Tenotomy-induced muscle atrophy is sex-specific and independent of NFκB. Elife 11, e82016 (2022).",),
        ("Josserand, L., Garaud, P. & Walch, G. Long-term outcomes of reverse total shoulder arthroplasty: a follow-up of a previous study. J. Bone Joint Surg. Am. 99, 454-461 (2017).",),
        ("Sheasha, G. & Grawe, B. M. Type 2 retear after arthroscopic single-row, double-row and suture bridge rotator cuff repair: a systematic review. Eur. J. Orthop. Surg. Traumatol. 29, 373-382 (2019).",),
    ]

    for (raw,) in examples:
        nr = _normalize_crossref_reference_entry({"unstructured": raw}, confidence=0.85)
        # Title should not have been truncated to start with a lowercase fragment
        assert nr.title and nr.title[0].isupper()


def test_hydrate_prefers_cleaner_title(monkeypatch):
    # sparse reference with noisy unstructured title containing author prefix
    sample_payload = {
        "message": {
            "title": ["Paper Title"],
            "reference": [
                {"unstructured": "Davies, M. R. et al. - Noisy Title", "DOI": "10.1/cleantitle"},
            ],
        }
    }

    # hydration payload provides a cleaner title
    hydrate_payload = {"message": {"title": ["Clean Hydrated Title"], "author": [{"given": "Jane", "family": "Doe"}], "issued": {"date-parts": [[2020]]}, "container-title": ["Journal"], "URL": "https://doi.org/10.1/cleantitle", "DOI": "10.1/cleantitle"}}

    def _mock_request_json(url, timeout_sec=5.0):
        if "/works/" in url and "cleantitle" in url:
            return hydrate_payload, None
        return sample_payload, None

    monkeypatch.setattr(prov, "_request_json", _mock_request_json)
    monkeypatch.setenv("CROSSREF_HYDRATE_MAX", "5")

    refs, status = _fetch_crossref({"doi": "10.1000/testdoi", "title": ""})
    assert len(refs) == 1
    r = refs[0]
    # hydrated cleaner title should be preferred
    assert r.title == "Clean Hydrated Title"


def test_hydrate_complete_record_with_noisy_title(monkeypatch):
    sample_payload = {
        "message": {
            "title": ["Paper Title"],
            "reference": [
                {
                    "article-title": "Abate, M. et al. Oxidative stress and inflammation in tendon injury.",
                    "author": "Abate, M.; Smith, J.",
                    "year": "2021",
                    "journal-title": "Journal of Testing",
                    "URL": "https://example.org/reference",
                    "DOI": "10.1/complete-noisy",
                },
            ],
        }
    }

    hydrate_payload = {
        "message": {
            "title": ["Hydrated clean title for tendon injury"],
            "author": [{"given": "Marco", "family": "Abate"}],
            "issued": {"date-parts": [[2021]]},
            "container-title": ["Journal of Testing"],
            "URL": "https://doi.org/10.1/complete-noisy",
            "DOI": "10.1/complete-noisy",
        }
    }

    calls: list[str] = []

    def _mock_request_json(url, timeout_sec=5.0):
        calls.append(url)
        if "/works/" in url and "complete-noisy" in _up.unquote(url):
            return hydrate_payload, None
        return sample_payload, None

    monkeypatch.setattr(prov, "_request_json", _mock_request_json)
    monkeypatch.setenv("CROSSREF_HYDRATE_MAX", "5")

    refs, _ = _fetch_crossref({"doi": "10.1000/testdoi", "title": ""})
    assert len(refs) == 1
    assert refs[0].title == "Hydrated clean title for tendon injury"
    assert any("complete-noisy" in _up.unquote(c) for c in calls)


def test_hydrate_sparse_reference(monkeypatch):
    # initial Crossref message with one sparse DOI-only reference
    sample_payload = {
        "message": {
            "title": ["Paper Title"],
            "reference": [
                {"DOI": "10.1/sparse"},
            ],
        }
    }

    # hydration payload for the sparse DOI
    hydrate_payload = {
        "message": {
            "title": ["Hydrated Title"],
            "author": [{"given": "Jane", "family": "Doe"}],
            "issued": {"date-parts": [[2019]]},
            "container-title": ["Hydrated Journal"],
            "URL": "https://doi.org/10.1/sparse",
            "DOI": "10.1/sparse",
        }
    }

    calls: list[str] = []

    def _mock_request_json(url, timeout_sec=5.0):
        calls.append(url)
        # Distinguish initial work call vs hydration call (allow for URL-encoding)
        if "/works/" in url and "testdoi" in url:
            return sample_payload, None
        if "/works/" in url and "sparse" in url:
            return hydrate_payload, None
        return None, "not_found"

    monkeypatch.setattr(prov, "_request_json", _mock_request_json)

    refs, status = _fetch_crossref({"doi": "10.1000/testdoi", "title": ""})
    assert status["provider"] == "crossref"
    assert status["status"] == "ok"
    assert status["reason"] == "references"
    assert len(refs) == 1
    r = refs[0]
    # After hydration, sparse DOI-only entry should have title/authors/year/venue/url
    assert r.title == "Hydrated Title"
    assert r.authors and r.authors[0].startswith("Jane")
    assert r.year == 2019
    assert "Hydrated Journal" in r.venue
    assert r.url and "10.1/sparse" in r.url
    # ensure hydrate call occurred once (allow URL-encoding)
    assert any(("/works/" in _up.unquote(c) and "sparse" in _up.unquote(c)) for c in calls)
