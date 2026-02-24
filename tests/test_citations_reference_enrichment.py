import json
from pathlib import Path

from ingest.citations import run_citations


def _write_manifest(run_dir: Path, pdf_path: Path) -> None:
    payload = {
        "doc_id": "unit_doc",
        "input_pdf_path": str(pdf_path),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def test_citations_reference_enrichment_provider_failure_is_non_fatal(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "unit_doc"
    run_dir.mkdir(parents=True)

    pdf_path = run_dir / "input.pdf"
    pdf_path.write_bytes(b"not-a-valid-pdf")
    _write_manifest(run_dir, pdf_path)

    text_dir = run_dir / "text"
    text_dir.mkdir(parents=True)
    with open(text_dir / "clean_document.md", "w", encoding="utf-8") as f:
        f.write("# Example Paper\n\n## Authors\n\n- Jane Doe\n\n### DOI\n\n- DOI:10.1000/example\n")

    def fake_collect_api_references(_identity: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        return [
            {
                "title": "Example API Reference",
                "authors": ["Jane Doe", "John Roe"],
                "year": 2022,
                "doi": "10.1000/example.ref",
                "pmid": None,
                "arxiv": None,
                "venue": "Journal of Tests",
                "url": "https://example.org/reference",
                "source": "semantic_scholar",
                "confidence": 0.81,
                "source_chain": ["pubmed", "crossref", "openalex"],
                "filled_fields": ["title", "authors", "year", "doi", "venue", "url"],
            }
        ], [
            {"provider": "pubmed", "status": "ok", "reason": None, "records": 0},
            {"provider": "crossref", "status": "error", "reason": "timeout", "records": 0},
            {"provider": "openalex", "status": "ok", "reason": None, "records": 1},
            {"provider": "arxiv", "status": "skipped", "reason": "missing_arxiv_id", "records": 0},
        ]

    # patch via public collect_api_references abstraction
    monkeypatch.setattr("ingest.citations.collect_api_references", fake_collect_api_references)

    anchors, mappings = run_citations(run_dir)
    assert anchors == 0
    assert mappings == 0

    assert (run_dir / "citations" / "cite_anchors.jsonl").exists()
    assert (run_dir / "citations" / "cite_map.jsonl").exists()
    assert (run_dir / "citations" / "reference_catalog.jsonl").exists()

    doc_identity_path = run_dir / "refs" / "doc_identity.json"
    assert doc_identity_path.exists()
    with open(doc_identity_path, "r", encoding="utf-8") as f:
        doc_identity = json.load(f)
    assert isinstance(doc_identity.get("provider_status"), list)
    assert any(s.get("status") == "error" for s in doc_identity["provider_status"])

    references_api_path = run_dir / "refs" / "references_api.jsonl"
    assert references_api_path.exists()
    with open(references_api_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 1
    expected_keys = {
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
    assert set(lines[0].keys()) == expected_keys
