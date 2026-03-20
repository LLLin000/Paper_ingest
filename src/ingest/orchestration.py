"""DAG-like orchestration helpers for full pipeline execution."""

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .manifest import Manifest

# Default parallelism settings
DEFAULT_MAX_WORKERS = int(os.environ.get("PIPELINE_MAX_WORKERS", "4"))
MAX_LLM_PARALLEL_CALLS = int(os.environ.get("PIPELINE_MAX_LLM_CALLS", "4"))

StageJob = Callable[[], Any]

# Explicit full-run dependency graph for orchestration and documentation.
FULL_PIPELINE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "extractor": (),
    "overlay": ("extractor",),
    "vision": ("extractor",),
    "paragraphs": ("vision",),
    "citations": ("paragraphs",),
    "figures_tables": ("paragraphs", "vision"),
    "reading": ("citations", "figures_tables"),
    "render": ("reading",),
}

# Levelized DAG plan that allows safe parallel execution by level.
FULL_PIPELINE_LEVELS: tuple[tuple[str, ...], ...] = (
    ("extractor",),
    ("overlay", "vision"),
    ("paragraphs",),
    ("citations", "figures_tables"),
    ("reading",),
    ("render",),
)

FULL_PIPELINE_STAGE_METADATA: dict[str, dict[str, Any]] = {
    "extractor": {
        "inputs": ["manifest.json", "input_pdf"],
        "outputs": ["pages/p*.png", "text/blocks_raw.jsonl", "text/blocks_norm.jsonl", "text/layout_analysis.json"],
        "cache_scope": "document",
        "invalidated_by": ["input_pdf", "manifest.render_config", "manifest.parser_backend"],
    },
    "overlay": {
        "inputs": ["pages/p*.png", "text/blocks_norm.jsonl"],
        "outputs": ["pages/p*_annot.png"],
        "cache_scope": "page",
        "invalidated_by": ["pages/p*.png", "text/blocks_norm.jsonl"],
    },
    "vision": {
        "inputs": ["pages/p*.png", "text/blocks_norm.jsonl", "text/layout_analysis.json"],
        "outputs": ["vision/p*_in.json", "vision/p*_out.json"],
        "cache_scope": "page",
        "invalidated_by": [
            "pages/p*.png",
            "text/blocks_norm.jsonl",
            "text/layout_analysis.json",
            "manifest.model_config.vision_model",
            "manifest.model_config.prompt_bundle_version",
            "manifest.parser_backend",
        ],
    },
    "paragraphs": {
        "inputs": ["text/blocks_norm.jsonl", "vision/p*_out.json"],
        "outputs": ["paragraphs/paragraphs.jsonl", "text/clean_document.md", "qa/clean_document_metrics.json", "qa/structure_quality.json"],
        "cache_scope": "document",
        "invalidated_by": ["text/blocks_norm.jsonl", "vision/p*_out.json", "manifest.parser_backend"],
    },
    "citations": {
        "inputs": ["paragraphs/paragraphs.jsonl", "manifest.json"],
        "outputs": ["citations/cite_anchors.jsonl", "citations/cite_map.jsonl", "citations/reference_catalog.jsonl"],
        "cache_scope": "document",
        "invalidated_by": ["paragraphs/paragraphs.jsonl", "manifest.input_pdf_path"],
    },
    "figures_tables": {
        "inputs": ["vision/p*_out.json", "paragraphs/paragraphs.jsonl", "manifest.json"],
        "outputs": ["figures_tables/figure_table_index.jsonl", "figures_tables/figure_table_links.json"],
        "cache_scope": "document",
        "invalidated_by": ["vision/p*_out.json", "paragraphs/paragraphs.jsonl", "manifest.input_pdf_path"],
    },
    "reading": {
        "inputs": [
            "paragraphs/paragraphs.jsonl",
            "citations/cite_map.jsonl",
            "figures_tables/figure_table_index.jsonl",
            "figures_tables/figure_table_links.json",
            "qa/structure_quality.json",
        ],
        "outputs": [
            "reading/paper_profile.json",
            "reading/logic_graph.json",
            "reading/facts.jsonl",
            "reading/themes.json",
            "reading/synthesis.json",
            "qa/summary_status.json",
        ],
        "cache_scope": "document",
        "invalidated_by": [
            "paragraphs/paragraphs.jsonl",
            "citations/cite_map.jsonl",
            "figures_tables/figure_table_index.jsonl",
            "figures_tables/figure_table_links.json",
            "qa/structure_quality.json",
            "manifest.model_config.reading_model",
            "manifest.model_config.prompt_bundle_version",
        ],
    },
    "render": {
        "inputs": ["reading/synthesis.json", "reading/facts.jsonl", "qa/summary_status.json"],
        "outputs": ["obsidian/<doc_id>.md"],
        "cache_scope": "document",
        "invalidated_by": ["reading/synthesis.json", "reading/facts.jsonl", "qa/summary_status.json"],
    },
}


def build_pipeline_dag_artifact(manifest: Manifest) -> dict[str, Any]:
    return {
        "doc_id": manifest.doc_id,
        "parser_backend": getattr(manifest, "parser_backend", "builtin"),
        "dependencies": {stage: list(deps) for stage, deps in FULL_PIPELINE_DEPENDENCIES.items()},
        "levels": [list(level) for level in FULL_PIPELINE_LEVELS],
        "stages": FULL_PIPELINE_STAGE_METADATA,
    }


def write_pipeline_dag_artifact(run_dir: Path, manifest: Manifest) -> Path:
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = qa_dir / "pipeline_dag.json"
    artifact_path.write_text(
        json.dumps(build_pipeline_dag_artifact(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return artifact_path


def execute_levelized_dag(
    jobs: dict[str, StageJob],
    levels: tuple[tuple[str, ...], ...] = FULL_PIPELINE_LEVELS,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Execute levelized DAG jobs with deterministic result ordering."""
    results: dict[str, Any] = {}

    for level in levels:
        if len(level) == 1:
            stage_name = level[0]
            results[stage_name] = jobs[stage_name]()
            continue

        with ThreadPoolExecutor(max_workers=min(max_workers, len(level))) as executor:
            futures: dict[str, Future[Any]] = {
                stage_name: executor.submit(jobs[stage_name])
                for stage_name in level
            }
            for stage_name in level:
                results[stage_name] = futures[stage_name].result()

    return results
