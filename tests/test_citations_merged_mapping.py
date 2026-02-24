import json
from pathlib import Path

from ingest.citations import run_citations


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_mapping_prefers_merged(tmp_path, monkeypatch):
    run_dir = tmp_path / "run" / "merged_pref"
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "m.pdf"
    pdf.write_bytes(b"x")
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"doc_id": "merged_pref", "input_pdf_path": str(pdf)}, f)

    _write_jsonl(
        run_dir / "paragraphs" / "paragraphs.jsonl",
        [
            {"para_id": "p1", "role": "BodyText", "text": "See [1]", "page_span": {"start": 1}, "evidence_pointer": {"bbox_union": [0,0,0,0]}},
            {"para_id": "p2", "role": "ReferenceList", "text": "[1] X. 2020.", "page_span": {"start": 1}, "evidence_pointer": {"bbox_union": [0,0,0,0]}},
        ],
    )

    refs_dir = run_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    merged = {"title": "X", "year": 2020, "doi": "10.1/merged", "confidence": 0.9, "sources": [{"provider": "crossref", "confidence": 0.9, "kind": "api"}]}
    with open(refs_dir / "references_merged.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(merged) + "\n")

    monkeypatch.setattr("ingest.citations.build_doc_identity", lambda *_: {"doc_id": "merged_pref", "title": "X"})
    monkeypatch.setattr("ingest.citations.collect_api_references", lambda *_: ([], []))

    anchors, maps = run_citations(run_dir)
    assert maps >= 0
    mpath = run_dir / "citations" / "cite_map.jsonl"
    assert mpath.exists()
    lines = [json.loads(l) for l in mpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines
    # ensure at least one mapping points to the merged DOI
    assert any(l.get("mapped_ref_key") == "doi:10.1/merged" for l in lines)
