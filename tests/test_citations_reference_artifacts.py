import json
from pathlib import Path

from ingest.citations import run_citations


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_citations_writes_reference_api_artifacts_without_breaking_existing_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"not a real pdf")

    manifest = {
        "doc_id": "demo",
        "input_pdf_path": str(pdf_path),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    _write_jsonl(
        run_dir / "paragraphs" / "paragraphs.jsonl",
        [
            {
                "para_id": "p1",
                "role": "BodyText",
                "text": "As shown in [1], this is grounded.",
                "page_span": {"start": 1, "end": 1},
                "evidence_pointer": {"bbox_union": [0.0, 100.0, 300.0, 130.0]},
                "section_path": ["Introduction"],
            },
            {
                "para_id": "p2",
                "role": "ReferenceList",
                "text": "[1] Doe, J. Example reference. 2020. doi:10.1000/x",
                "page_span": {"start": 1, "end": 1},
                "evidence_pointer": {"bbox_union": [0.0, 700.0, 300.0, 760.0]},
                "section_path": ["References"],
            },
        ],
    )

    def _fake_identity(_: Path, __: dict[str, object]) -> dict[str, object]:
        return {
            "doc_id": "demo",
            "doi": "10.1000/x",
            "arxiv": None,
            "title": "Demo Title",
            "authors": ["Demo Author"],
            "identifier_source": "test",
            "title_source": "test",
            "status": "ok",
        }

    def _fake_collect(_: dict[str, object]):
        return [
            {
                "title": "API Paper",
                "authors": ["A. Author"],
                "year": 2020,
                "doi": "10.1000/x",
                "pmid": None,
                "arxiv": None,
                "venue": "Journal",
                "url": "https://example.org",
                "source": "crossref",
                "confidence": 0.8764,
                "source_chain": ["pubmed", "crossref"],
                "filled_fields": ["title", "authors", "year", "doi", "venue", "url"],
                "extra_ignored": "x",
            }
        ], [{"provider": "crossref", "status": "ok", "reason": None, "records": 1}]

    monkeypatch.setattr("ingest.citations.build_doc_identity", _fake_identity)
    # ensure citations imports new provider abstraction
    monkeypatch.setattr("ingest.citations.collect_api_references", _fake_collect)

    run_citations(run_dir)

    assert (run_dir / "citations" / "cite_anchors.jsonl").exists()
    assert (run_dir / "citations" / "cite_map.jsonl").exists()
    assert (run_dir / "citations" / "reference_catalog.jsonl").exists()
    assert (run_dir / "refs" / "doc_identity.json").exists()
    assert (run_dir / "refs" / "references_api.jsonl").exists()

    identity = json.loads((run_dir / "refs" / "doc_identity.json").read_text(encoding="utf-8"))
    assert identity["provider_status"][0]["provider"] == "crossref"

    line = (run_dir / "refs" / "references_api.jsonl").read_text(encoding="utf-8").strip()
    record = json.loads(line)
    required_keys = {
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
    }
    assert required_keys.issubset(set(record.keys()))
    assert record["confidence"] == 0.876
    assert "source_chain" in record
    assert isinstance(record["source_chain"], list)
    assert "filled_fields" in record
    assert isinstance(record["filled_fields"], list)

    # merged references should exist and be deduped (one entry for DOI)
    merged_path = run_dir / "refs" / "references_merged.jsonl"
    assert merged_path.exists()
    merged_lines = [json.loads(l) for l in merged_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # expect single merged record since API and PDF share same DOI
    assert len(merged_lines) == 1
    merged_rec = merged_lines[0]
    # doi preserved and sources include both api and pdf kinds
    assert merged_rec.get("doi") == "10.1000/x"
    providers = [s.get("provider") for s in merged_rec.get("sources", [])]
    assert any((p and p.startswith("crossref")) or p == "api" for p in providers)
    assert any(p and p.startswith("pdf:") for p in providers)


def test_citations_degrades_when_reference_collection_raises(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "demo2"
    run_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = tmp_path / "demo2.pdf"
    pdf_path.write_bytes(b"not a real pdf")
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"doc_id": "demo2", "input_pdf_path": str(pdf_path)}, f)

    _write_jsonl(
        run_dir / "paragraphs" / "paragraphs.jsonl",
        [
            {
                "para_id": "p1",
                "role": "BodyText",
                "text": "Body text",
                "page_span": {"start": 1, "end": 1},
                "evidence_pointer": {"bbox_union": [0.0, 100.0, 300.0, 130.0]},
                "section_path": ["Intro"],
            }
        ],
    )

    monkeypatch.setattr("ingest.citations.build_doc_identity", lambda *_: {"doc_id": "demo2", "title": "Demo2"})

    def _raise_collect(_: dict[str, object]):
        raise RuntimeError("boom")

    monkeypatch.setattr("ingest.citations.collect_api_references", _raise_collect)

    run_citations(run_dir)

    identity = json.loads((run_dir / "refs" / "doc_identity.json").read_text(encoding="utf-8"))
    assert identity["provider_status"][0]["status"] == "error"
    assert identity["provider_status"][0]["reason"] == "collection_exception:RuntimeError"
