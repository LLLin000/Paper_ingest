import json
from pathlib import Path

from ingest.manifest import Manifest
from ingest.orchestration import write_pipeline_dag_artifact


def test_write_pipeline_dag_artifact_records_stage_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "dag_artifact_doc"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        doc_id="dag_artifact_doc",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="d" * 64,
        started_at_utc="2026-03-18T00:00:00+00:00",
    )

    artifact_path = write_pipeline_dag_artifact(run_dir=run_dir, manifest=manifest)

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["doc_id"] == "dag_artifact_doc"
    assert payload["parser_backend"] == "builtin"
    assert payload["levels"][0] == ["extractor"]
    assert payload["dependencies"]["reading"] == ["citations", "figures_tables"]
    assert payload["stages"]["vision"]["cache_scope"] == "page"
    assert payload["stages"]["reading"]["invalidated_by"] == [
        "paragraphs/paragraphs.jsonl",
        "citations/cite_map.jsonl",
        "figures_tables/figure_table_index.jsonl",
        "figures_tables/figure_table_links.json",
        "qa/structure_quality.json",
        "manifest.model_config.reading_model",
        "manifest.model_config.prompt_bundle_version",
    ]
