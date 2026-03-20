import json
from pathlib import Path

from ingest.manifest import create_manifest
from ingest.parser_backend import DEFAULT_PARSER_BACKEND, resolve_parser_backend


def test_create_manifest_records_parser_backend(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% parser backend fixture\n")

    manifest, run_dir = create_manifest(pdf_path=pdf_path, run_root=tmp_path / "run")

    assert manifest.parser_backend == DEFAULT_PARSER_BACKEND

    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["parser_backend"] == DEFAULT_PARSER_BACKEND


def test_resolve_parser_backend_falls_back_to_builtin() -> None:
    backend = resolve_parser_backend("future-mineru")

    assert backend.name == DEFAULT_PARSER_BACKEND
    assert backend.metadata()["parser_backend"] == DEFAULT_PARSER_BACKEND
