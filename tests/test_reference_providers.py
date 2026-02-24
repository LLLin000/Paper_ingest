from ingest.reference_providers_impl import NormalizedReference
from ingest.reference_providers import collect_api_references
import ingest.reference_providers_impl as impl


def test_collect_api_references_handles_provider_exception(monkeypatch) -> None:
    def _pubmed_provider(_: dict[str, object]):
        return [], {"provider": "pubmed", "status": "ok", "reason": None, "records": 0}

    def _raise_provider(_: dict[str, object]):
        raise RuntimeError("provider down")

    def _openalex_provider(_: dict[str, object]):
        return [
            NormalizedReference(
                title="Example Paper",
                authors=["Alice Smith"],
                year=2024,
                doi="10.1000/example",
                pmid=None,
                arxiv=None,
                venue="ExampleConf",
                url="https://example.org/paper",
                source="openalex",
                confidence=0.8,
            )
        ], {"provider": "openalex", "status": "ok", "reason": None, "records": 1}

    def _arxiv_provider(_: dict[str, object]):
        return [], {"provider": "arxiv", "status": "ok", "reason": None, "records": 0}

    monkeypatch.setattr("ingest.reference_providers_impl._fetch_pubmed", _pubmed_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_crossref", _raise_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_openalex", _openalex_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_arxiv", _arxiv_provider)

    refs, status = collect_api_references({"title": "Example", "arxiv": "2401.01234"})

    assert len(refs) == 1
    assert set(refs[0].keys()) == {
        "title",
        "authors",
        "year",
        "doi",
        "pmid",
        "arxiv",
        "venue",
        "url",
        "source",
        "confidence",
        "source_chain",
        "filled_fields",
    }
    error_statuses = [entry for entry in status if entry["status"] == "error"]
    assert error_statuses
    assert error_statuses[0]["reason"] == "provider_exception:RuntimeError"


def test_collect_api_references_adds_chain_and_filled_fields(monkeypatch) -> None:
    def _pubmed_provider(_: dict[str, object]):
        return [
            NormalizedReference(
                title="Example Paper",
                authors=[],
                year=2020,
                doi="10.1000/example",
                pmid="12345678",
                arxiv=None,
                venue="",
                url=None,
                source="pubmed",
                confidence=0.7,
            )
        ], {"provider": "pubmed", "status": "ok", "reason": None, "records": 1}

    def _crossref_provider(_: dict[str, object]):
        return [
            NormalizedReference(
                title="Example Paper",
                authors=["Alice Smith"],
                year=2020,
                doi="10.1000/example",
                pmid=None,
                arxiv=None,
                venue="Journal",
                url="https://doi.org/10.1000/example",
                source="crossref",
                confidence=0.9,
            )
        ], {"provider": "crossref", "status": "ok", "reason": None, "records": 1}

    def _openalex_provider(_: dict[str, object]):
        return [], {"provider": "openalex", "status": "ok", "reason": None, "records": 0}

    monkeypatch.setattr("ingest.reference_providers_impl._fetch_pubmed", _pubmed_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_crossref", _crossref_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_openalex", _openalex_provider)

    refs, _ = collect_api_references({"title": "Example"})

    assert len(refs) == 1
    ref = refs[0]
    assert ref["source_chain"] == ["pubmed", "crossref"]
    assert "authors" in ref["filled_fields"]
    assert "doi" in ref["filled_fields"]


def test_fetch_pubmed_prefers_pubmed_references_via_elink(monkeypatch) -> None:
    def _mock_request_json(url: str, timeout_sec: float = 5.0):
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["999"]}}, None
        if "elink.fcgi" in url:
            return {
                "linksets": [
                    {
                        "linksetdbs": [
                            {"links": ["111", "222"]},
                        ]
                    }
                ]
            }, None
        if "esummary.fcgi" in url and "id=111%2C222" in url:
            return {
                "result": {
                    "uids": ["111", "222"],
                    "111": {
                        "uid": "111",
                        "title": "Ref One",
                        "pubdate": "2019",
                        "authors": [{"name": "A One"}],
                        "source": "Journal A",
                        "articleids": [{"idtype": "doi", "value": "10.1000/ref1"}],
                    },
                    "222": {
                        "uid": "222",
                        "title": "Ref Two",
                        "pubdate": "2020",
                        "authors": [{"name": "B Two"}],
                        "source": "Journal B",
                        "articleids": [{"idtype": "doi", "value": "10.1000/ref2"}],
                    },
                }
            }, None
        return None, "unexpected_url"

    monkeypatch.setattr(impl, "_request_json", _mock_request_json)

    refs, status = impl._fetch_pubmed({"doi": "10.1000/demo"})
    assert status["provider"] == "pubmed"
    assert status["status"] == "ok"
    assert status["reason"] == "references"
    assert status["records"] == 2
    assert [r.title for r in refs] == ["Ref One", "Ref Two"]


def test_fetch_pubmed_falls_back_to_seed_match_when_no_links(monkeypatch) -> None:
    def _mock_request_json(url: str, timeout_sec: float = 5.0):
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["999"]}}, None
        if "elink.fcgi" in url:
            return {"linksets": [{"linksetdbs": []}]}, None
        if "esummary.fcgi" in url and "id=999" in url:
            return {
                "result": {
                    "uids": ["999"],
                    "999": {
                        "uid": "999",
                        "title": "Primary Paper",
                        "pubdate": "2021",
                        "authors": [{"name": "P Author"}],
                        "source": "Journal Primary",
                        "articleids": [{"idtype": "doi", "value": "10.1000/primary"}],
                    },
                }
            }, None
        return None, "unexpected_url"

    monkeypatch.setattr(impl, "_request_json", _mock_request_json)

    refs, status = impl._fetch_pubmed({"doi": "10.1000/demo"})
    assert status["provider"] == "pubmed"
    assert status["status"] == "ok"
    assert status["reason"] == "seed_match"
    assert status["records"] == 1
    assert refs[0].title == "Primary Paper"


def test_pubmed_backfill_record_fills_missing_fields(monkeypatch) -> None:
    def _mock_request_json(url: str, timeout_sec: float = 5.0):
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["7654321"]}}, None
        if "esummary.fcgi" in url and "id=7654321" in url:
            return {
                "result": {
                    "uids": ["7654321"],
                    "7654321": {
                        "uid": "7654321",
                        "title": "Backfilled Title",
                        "pubdate": "2018 Jan",
                        "authors": [{"name": "Alpha One"}, {"name": "Beta Two"}],
                        "source": "Backfill Journal",
                        "articleids": [{"idtype": "doi", "value": "10.1000/backfill"}],
                    },
                }
            }, None
        return None, "unexpected_url"

    monkeypatch.setattr(impl, "_request_json", _mock_request_json)

    fields, status = impl._pubmed_backfill_record(
        NormalizedReference(
            title="",
            authors=[],
            year=None,
            doi="10.1000/backfill",
            pmid=None,
            arxiv=None,
            venue="",
            url=None,
            source="crossref",
            confidence=0.85,
        )
    )

    assert status["status"] == "ok"
    assert status["reason"] == "filled"
    assert fields["title"] == "Backfilled Title"
    assert fields["authors"] == ["Alpha One", "Beta Two"]
    assert fields["year"] == 2018
    assert fields["venue"] == "Backfill Journal"
    assert fields["pmid"] == "7654321"
    assert fields["url"] == "https://pubmed.ncbi.nlm.nih.gov/7654321/"


def test_collect_api_references_backfill_failure_leaves_record_unchanged(monkeypatch) -> None:
    def _pubmed_provider(_: dict[str, object]):
        return [], {"provider": "pubmed", "status": "ok", "reason": None, "records": 0}

    def _crossref_provider(_: dict[str, object]):
        return [
            NormalizedReference(
                title="Sparse Ref",
                authors=[],
                year=None,
                doi="10.1000/sparse",
                pmid=None,
                arxiv=None,
                venue="",
                url=None,
                source="crossref",
                confidence=0.9,
            )
        ], {"provider": "crossref", "status": "ok", "reason": None, "records": 1}

    def _openalex_provider(_: dict[str, object]):
        return [], {"provider": "openalex", "status": "ok", "reason": None, "records": 0}

    def _mock_request_json(url: str, timeout_sec: float = 5.0):
        if "esearch.fcgi" in url:
            return None, "timeout"
        return None, "unexpected_url"

    monkeypatch.setattr("ingest.reference_providers_impl._fetch_pubmed", _pubmed_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_crossref", _crossref_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_openalex", _openalex_provider)
    monkeypatch.setattr(impl, "_request_json", _mock_request_json)

    refs, _ = collect_api_references({"title": "Example"})

    assert len(refs) == 1
    ref = refs[0]
    assert ref["authors"] == []
    assert ref["year"] is None
    assert ref["venue"] == ""
    assert ref["url"] is None
    assert ref["source_chain"] == ["crossref"]
    assert "pubmed_backfill" not in ref["source_chain"]


def test_collect_api_references_appends_pubmed_backfill_on_fill(monkeypatch) -> None:
    def _pubmed_provider(_: dict[str, object]):
        return [], {"provider": "pubmed", "status": "ok", "reason": None, "records": 0}

    def _crossref_provider(_: dict[str, object]):
        return [
            NormalizedReference(
                title="Sparse Ref",
                authors=[],
                year=None,
                doi="10.1000/sparse-fill",
                pmid=None,
                arxiv=None,
                venue="",
                url=None,
                source="crossref",
                confidence=0.9,
            )
        ], {"provider": "crossref", "status": "ok", "reason": None, "records": 1}

    def _openalex_provider(_: dict[str, object]):
        return [], {"provider": "openalex", "status": "ok", "reason": None, "records": 0}

    def _mock_request_json(url: str, timeout_sec: float = 5.0):
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["99123"]}}, None
        if "esummary.fcgi" in url and "id=99123" in url:
            return {
                "result": {
                    "uids": ["99123"],
                    "99123": {
                        "uid": "99123",
                        "title": "Filled By PubMed",
                        "pubdate": "2022 Feb",
                        "authors": [{"name": "Jane Doe"}],
                        "source": "Filled Journal",
                        "articleids": [{"idtype": "doi", "value": "10.1000/sparse-fill"}],
                    },
                }
            }, None
        return None, "unexpected_url"

    monkeypatch.setattr("ingest.reference_providers_impl._fetch_pubmed", _pubmed_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_crossref", _crossref_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_openalex", _openalex_provider)
    monkeypatch.setattr(impl, "_request_json", _mock_request_json)

    refs, _ = collect_api_references({"title": "Example"})

    assert len(refs) == 1
    ref = refs[0]
    assert ref["authors"] == ["Jane Doe"]
    assert ref["year"] == 2022
    assert ref["venue"] == "Filled Journal"
    assert ref["pmid"] == "99123"
    assert ref["url"] == "https://pubmed.ncbi.nlm.nih.gov/99123/"
    assert ref["source_chain"] == ["crossref", "pubmed_backfill"]
    assert "authors" in ref["filled_fields"]
    assert "year" in ref["filled_fields"]
    assert "venue" in ref["filled_fields"]
    assert "url" in ref["filled_fields"]


def test_pubmed_backfill_title_only_strong_match_accepted(monkeypatch) -> None:
    def _mock_request_json(url: str, timeout_sec: float = 5.0):
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["2001"]}}, None
        if "esummary.fcgi" in url and "id=2001" in url:
            return {
                "result": {
                    "uids": ["2001"],
                    "2001": {
                        "uid": "2001",
                        "title": "Backfilled title for tendon injury",
                        "pubdate": "2019",
                        "authors": [{"name": "Alpha One"}],
                        "source": "Journal Match",
                        "articleids": [{"idtype": "doi", "value": "10.1000/title-match"}],
                    },
                }
            }, None
        return None, "unexpected_url"

    monkeypatch.setattr(impl, "_request_json", _mock_request_json)

    fields, status = impl._pubmed_backfill_record(
        NormalizedReference(
            title="Backfilled Title for Tendon Injury",
            authors=["Alpha One"],
            year=2018,
            doi=None,
            pmid=None,
            arxiv=None,
            venue="",
            url=None,
            source="crossref",
            confidence=0.85,
        )
    )

    assert status["status"] == "ok"
    assert status["reason"] == "filled"
    assert fields["title"] == "Backfilled title for tendon injury"
    assert fields["year"] == 2019
    assert fields["pmid"] == "2001"


def test_pubmed_backfill_title_only_weak_match_rejected(monkeypatch) -> None:
    def _mock_request_json(url: str, timeout_sec: float = 5.0):
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["2002"]}}, None
        if "esummary.fcgi" in url and "id=2002" in url:
            return {
                "result": {
                    "uids": ["2002"],
                    "2002": {
                        "uid": "2002",
                        "title": "Completely unrelated methods paper",
                        "pubdate": "2018",
                        "authors": [{"name": "Other Author"}],
                        "source": "Journal Other",
                    },
                }
            }, None
        return None, "unexpected_url"

    monkeypatch.setattr(impl, "_request_json", _mock_request_json)

    fields, status = impl._pubmed_backfill_record(
        NormalizedReference(
            title="Backfilled Title for Tendon Injury",
            authors=["Alpha One"],
            year=2018,
            doi=None,
            pmid=None,
            arxiv=None,
            venue="",
            url=None,
            source="crossref",
            confidence=0.85,
        )
    )

    assert fields == {}
    assert status["status"] == "ok"
    assert status["reason"] == "uncertain_title_match"


def test_collect_api_references_mutual_fill_trace_is_deterministic(monkeypatch) -> None:
    def _pubmed_provider(_: dict[str, object]):
        return [], {"provider": "pubmed", "status": "ok", "reason": None, "records": 0}

    def _crossref_provider(_: dict[str, object]):
        return [
            NormalizedReference(
                title="Mutual Fill Paper",
                authors=[],
                year=None,
                doi=None,
                pmid=None,
                arxiv=None,
                venue="",
                url=None,
                source="crossref",
                confidence=0.9,
            )
        ], {"provider": "crossref", "status": "ok", "reason": None, "records": 1}

    def _openalex_provider(_: dict[str, object]):
        return [], {"provider": "openalex", "status": "ok", "reason": None, "records": 0}

    def _mock_request_json(url: str, timeout_sec: float = 5.0):
        if "eutils.ncbi.nlm.nih.gov" in url and "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["3001"]}}, None
        if "eutils.ncbi.nlm.nih.gov" in url and "esummary.fcgi" in url and "id=3001" in url:
            return {
                "result": {
                    "uids": ["3001"],
                    "3001": {
                        "uid": "3001",
                        "title": "Different article title",
                        "pubdate": "2021",
                        "authors": [{"name": "Mismatch"}],
                        "source": "Mismatch Journal",
                    },
                }
            }, None
        if "api.openalex.org/works?" in url:
            return {
                "results": [
                    {
                        "display_name": "Mutual Fill Paper",
                        "publication_year": 2021,
                        "ids": {"doi": "https://doi.org/10.1000/mutual"},
                        "authorships": [],
                        "primary_location": {},
                    }
                ]
            }, None
        if "api.crossref.org/works/10.1000%2Fmutual" in url:
            return {
                "message": {
                    "title": ["Mutual Fill Paper"],
                    "author": [{"given": "Jane", "family": "Doe"}],
                    "issued": {"date-parts": [[2021]]},
                    "container-title": ["Mutual Journal"],
                    "URL": "https://doi.org/10.1000/mutual",
                    "DOI": "10.1000/mutual",
                }
            }, None
        return None, "unexpected_url"

    monkeypatch.setattr("ingest.reference_providers_impl._fetch_pubmed", _pubmed_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_crossref", _crossref_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_openalex", _openalex_provider)
    monkeypatch.setattr(impl, "_request_json", _mock_request_json)

    refs, _ = collect_api_references({"title": "Example"})

    assert len(refs) == 1
    ref = refs[0]
    assert ref["doi"] == "10.1000/mutual"
    assert ref["authors"] == ["Jane Doe"]
    assert ref["venue"] == "Mutual Journal"
    assert ref["url"] == "https://doi.org/10.1000/mutual"
    assert ref["source_chain"] == ["crossref", "openalex_backfill", "crossref_backfill"]
    assert "doi" in ref["filled_fields"]
    assert "authors" in ref["filled_fields"]


def test_collect_api_references_adaptive_max_grows_with_sparse_count(monkeypatch) -> None:
    def _pubmed_provider(_: dict[str, object]):
        return [], {"provider": "pubmed", "status": "ok", "reason": None, "records": 0}

    def _crossref_provider(_: dict[str, object]):
        refs: list[NormalizedReference] = []
        for idx in range(200):
            refs.append(
                NormalizedReference(
                    title=f"Sparse {idx}",
                    authors=[],
                    year=None,
                    doi=None,
                    pmid=None,
                    arxiv=None,
                    venue="",
                    url=None,
                    source="crossref",
                    confidence=0.9,
                )
            )
        return refs, {"provider": "crossref", "status": "ok", "reason": None, "records": len(refs)}

    def _openalex_provider(_: dict[str, object]):
        return [], {"provider": "openalex", "status": "ok", "reason": None, "records": 0}

    monkeypatch.delenv("REFERENCE_BACKFILL_MAX", raising=False)
    monkeypatch.delenv("PUBMED_BACKFILL_MAX", raising=False)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_pubmed", _pubmed_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_crossref", _crossref_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_openalex", _openalex_provider)
    monkeypatch.setattr(impl, "_pubmed_backfill_record", lambda _r: ({}, {"provider": "pubmed_backfill", "status": "ok", "reason": None, "records": 0}))
    monkeypatch.setattr(impl, "_openalex_backfill_record", lambda _r: ({}, {"provider": "openalex_backfill", "status": "ok", "reason": None, "records": 0}))
    monkeypatch.setattr(impl, "_crossref_backfill_record", lambda _r: ({}, {"provider": "crossref_backfill", "status": "ok", "reason": None, "records": 0}))

    _, statuses = collect_api_references({"title": "Example"})
    status_by_provider = {s["provider"]: s for s in statuses}

    assert status_by_provider["pubmed_backfill"]["max"] == 340
    attempted_total = (
        status_by_provider["pubmed_backfill"]["attempted"]
        + status_by_provider["openalex_backfill"]["attempted"]
        + status_by_provider["crossref_backfill"]["attempted"]
    )
    assert attempted_total == 340


def test_collect_api_references_env_override_wins_over_adaptive_default(monkeypatch) -> None:
    def _pubmed_provider(_: dict[str, object]):
        return [], {"provider": "pubmed", "status": "ok", "reason": None, "records": 0}

    def _crossref_provider(_: dict[str, object]):
        refs: list[NormalizedReference] = []
        for idx in range(200):
            refs.append(
                NormalizedReference(
                    title=f"Sparse {idx}",
                    authors=[],
                    year=None,
                    doi=None,
                    pmid=None,
                    arxiv=None,
                    venue="",
                    url=None,
                    source="crossref",
                    confidence=0.9,
                )
            )
        return refs, {"provider": "crossref", "status": "ok", "reason": None, "records": len(refs)}

    def _openalex_provider(_: dict[str, object]):
        return [], {"provider": "openalex", "status": "ok", "reason": None, "records": 0}

    monkeypatch.setenv("REFERENCE_BACKFILL_MAX", "30")
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_pubmed", _pubmed_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_crossref", _crossref_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_openalex", _openalex_provider)
    monkeypatch.setattr(impl, "_pubmed_backfill_record", lambda _r: ({}, {"provider": "pubmed_backfill", "status": "ok", "reason": None, "records": 0}))
    monkeypatch.setattr(impl, "_openalex_backfill_record", lambda _r: ({}, {"provider": "openalex_backfill", "status": "ok", "reason": None, "records": 0}))
    monkeypatch.setattr(impl, "_crossref_backfill_record", lambda _r: ({}, {"provider": "crossref_backfill", "status": "ok", "reason": None, "records": 0}))

    _, statuses = collect_api_references({"title": "Example"})
    status_by_provider = {s["provider"]: s for s in statuses}

    assert status_by_provider["pubmed_backfill"]["max"] == 30
    attempted_total = (
        status_by_provider["pubmed_backfill"]["attempted"]
        + status_by_provider["openalex_backfill"]["attempted"]
        + status_by_provider["crossref_backfill"]["attempted"]
    )
    assert attempted_total == 30


def test_collect_api_references_backfill_max_is_hard_bounded(monkeypatch) -> None:
    def _pubmed_provider(_: dict[str, object]):
        return [], {"provider": "pubmed", "status": "ok", "reason": None, "records": 0}

    def _crossref_provider(_: dict[str, object]):
        refs: list[NormalizedReference] = []
        for idx in range(500):
            refs.append(
                NormalizedReference(
                    title=f"Sparse {idx}",
                    authors=[],
                    year=None,
                    doi=None,
                    pmid=None,
                    arxiv=None,
                    venue="",
                    url=None,
                    source="crossref",
                    confidence=0.9,
                )
            )
        return refs, {"provider": "crossref", "status": "ok", "reason": None, "records": len(refs)}

    def _openalex_provider(_: dict[str, object]):
        return [], {"provider": "openalex", "status": "ok", "reason": None, "records": 0}

    monkeypatch.setenv("REFERENCE_BACKFILL_MAX", "9999")
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_pubmed", _pubmed_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_crossref", _crossref_provider)
    monkeypatch.setattr("ingest.reference_providers_impl._fetch_openalex", _openalex_provider)
    monkeypatch.setattr(impl, "_pubmed_backfill_record", lambda _r: ({}, {"provider": "pubmed_backfill", "status": "ok", "reason": None, "records": 0}))
    monkeypatch.setattr(impl, "_openalex_backfill_record", lambda _r: ({}, {"provider": "openalex_backfill", "status": "ok", "reason": None, "records": 0}))
    monkeypatch.setattr(impl, "_crossref_backfill_record", lambda _r: ({}, {"provider": "crossref_backfill", "status": "ok", "reason": None, "records": 0}))

    _, statuses = collect_api_references({"title": "Example"})
    status_by_provider = {s["provider"]: s for s in statuses}

    assert status_by_provider["pubmed_backfill"]["max"] == 1000
    attempted_total = (
        status_by_provider["pubmed_backfill"]["attempted"]
        + status_by_provider["openalex_backfill"]["attempted"]
        + status_by_provider["crossref_backfill"]["attempted"]
    )
    assert attempted_total == 1000
