"""PDF Ingest CLI - Contract-driven PDF reading pipeline."""

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .manifest import create_manifest, load_manifest, Manifest
from .extractor import run_extractor
from .overlay import run_overlay
from .vision import run_vision
from .paragraphs import run_paragraphs
from .citations import run_citations
from .figures_tables import run_figures_tables
from .reading import run_reading
from .render import run_render
from .verify import verify as run_verify
from .orchestration import execute_levelized_dag

app = typer.Typer(
    name="ingest_pdf",
    help="PDF document ingestion CLI for evidence-traceable Obsidian literature notes.",
    add_completion=False,
)

RUN_ROOT = Path("run")


class Stage(str, Enum):
    INIT_ONLY = "init-only"
    EXTRACTOR = "extractor"
    OVERLAY = "overlay"
    VISION = "vision"
    PARAGRAPHS = "paragraphs"
    CITATIONS = "citations"
    FIGURES_TABLES = "figures-tables"
    READING = "reading"
    RENDER = "render"
    FULL = "full"
    VERIFY = "verify"


@app.command()
def run(
    pdf: Optional[Path] = typer.Option(
        None,
        "--pdf",
        help="Path to input PDF file. Required for init-only or when manifest is missing.",
        exists=False,
    ),
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc_id",
        help="Document identifier. If omitted, computed from PDF SHA256 prefix.",
    ),
    stage: Stage = typer.Option(
        Stage.FULL,
        "--stage",
        help="Pipeline stage to execute.",
        case_sensitive=False,
    ),
    inject_vision_malformed_json: bool = typer.Option(
        False,
        "--inject-vision-malformed-json",
        help="[Testing] Inject malformed JSON in vision stage.",
        hidden=True,
    ),
    inject_reading_malformed_json: bool = typer.Option(
        False,
        "--inject-reading-malformed-json",
        help="[Testing] Inject malformed JSON in reading stage.",
        hidden=True,
    ),
    inject_missing_font_stats: bool = typer.Option(
        False,
        "--inject-missing-font-stats",
        help="[Testing] Inject missing font stats in extractor.",
        hidden=True,
    ),
) -> None:
    typer.echo(f"PDF Ingest CLI v0.1.0")
    typer.echo(f"Stage: {stage.value}")
    
    if pdf:
        typer.echo(f"Input PDF: {pdf}")
    
    if stage == Stage.INIT_ONLY:
        if not pdf:
            typer.echo("Error: --pdf is required for init-only stage.", err=True)
            raise typer.Exit(code=1)
        if not pdf.exists():
            typer.echo(f"Error: PDF file not found: {pdf}", err=True)
            raise typer.Exit(code=1)
        return _run_init_only(pdf, doc_id)
    
    if not doc_id and not pdf:
        typer.echo("Error: Either --pdf or --doc_id is required.", err=True)
        raise typer.Exit(code=1)
    
    manifest = _resolve_manifest(pdf, doc_id)
    
    typer.echo(f"Document ID: {manifest.doc_id}")
    
    if stage == Stage.EXTRACTOR:
        return _run_extractor(manifest, inject_missing_font_stats)
    
    if stage == Stage.OVERLAY:
        return _run_overlay(manifest)
    
    if stage == Stage.VISION:
        return _run_vision(manifest, inject_vision_malformed_json)
    
    if stage == Stage.PARAGRAPHS:
        return _run_paragraphs(manifest)
    
    if stage == Stage.CITATIONS:
        return _run_citations(manifest)
    
    if stage == Stage.FIGURES_TABLES:
        return _run_figures_tables(manifest)
    
    if stage == Stage.READING:
        return _run_reading(manifest, inject_reading_malformed_json)
    
    if stage == Stage.RENDER:
        return _run_render(manifest)
    
    if stage == Stage.FULL:
        return _run_full(manifest, inject_vision_malformed_json, inject_reading_malformed_json, inject_missing_font_stats)
    
    if stage == Stage.VERIFY:
        return _run_verify(manifest.doc_id)
    
    typer.echo("Pipeline bootstrap complete. Ready for T2+ implementation.")


def _run_init_only(pdf: Path, doc_id: Optional[str]) -> None:
    manifest, run_dir = create_manifest(
        pdf_path=pdf,
        doc_id=doc_id,
        run_root=RUN_ROOT,
    )
    typer.echo(f"Document ID: {manifest.doc_id}")
    typer.echo(f"Run directory: {run_dir}")
    typer.echo(f"Manifest created: {run_dir / 'manifest.json'}")
    typer.echo(f"PDF SHA256: {manifest.input_pdf_sha256[:16]}...")
    typer.echo("Init-only stage complete.")


def _run_extractor(manifest: Manifest, inject_missing_font_stats: bool) -> None:
    run_dir = RUN_ROOT / manifest.doc_id
    typer.echo(f"Run directory: {run_dir}")
    
    total_pages, total_blocks = run_extractor(
        run_dir=run_dir,
        manifest=manifest,
        inject_missing_font_stats=inject_missing_font_stats,
    )
    
    typer.echo(f"Rendered {total_pages} pages to {run_dir / 'pages'}")
    typer.echo(f"Extracted {total_blocks} text blocks")
    typer.echo(f"Raw blocks: {run_dir / 'text' / 'blocks_raw.jsonl'}")
    typer.echo(f"Norm blocks: {run_dir / 'text' / 'blocks_norm.jsonl'}")
    
    if inject_missing_font_stats:
        typer.echo("Fault injection: missing-font-stats enabled")
        typer.echo(f"Fault log: {run_dir / 'qa' / 'fault_injection.json'}")
    
    typer.echo("Extractor stage complete.")


def _run_overlay(manifest: Manifest) -> None:
    run_dir = RUN_ROOT / manifest.doc_id
    typer.echo(f"Run directory: {run_dir}")
    
    pages_processed, blocks_drawn = run_overlay(
        run_dir=run_dir,
        manifest=manifest,
    )
    
    typer.echo(f"Processed {pages_processed} pages")
    typer.echo(f"Drew {blocks_drawn} block overlays")
    typer.echo(f"Output: {run_dir / 'pages' / 'p*_annot.png'}")
    typer.echo("Overlay stage complete.")


def _run_vision(manifest: Manifest, inject_vision_malformed_json: bool) -> None:
    run_dir = RUN_ROOT / manifest.doc_id
    typer.echo(f"Run directory: {run_dir}")
    
    pages_processed, blocks_processed = run_vision(
        run_dir=run_dir,
        manifest=manifest,
        inject_malformed_json=inject_vision_malformed_json,
    )
    
    typer.echo(f"Processed {pages_processed} pages")
    typer.echo(f"Analyzed {blocks_processed} blocks")
    typer.echo(f"Input files: {run_dir / 'vision' / 'p*_in.json'}")
    typer.echo(f"Output files: {run_dir / 'vision' / 'p*_out.json'}")
    
    if inject_vision_malformed_json:
        typer.echo("Fault injection: malformed-json enabled")
        typer.echo(f"Fault log: {run_dir / 'qa' / 'fault_injection.json'}")
    
    typer.echo("Vision stage complete.")


def _run_paragraphs(manifest: Manifest) -> None:
    run_dir = RUN_ROOT / manifest.doc_id
    typer.echo(f"Run directory: {run_dir}")
    
    paragraphs_created, blocks_processed = run_paragraphs(
        run_dir=run_dir,
        manifest=manifest,
    )
    
    typer.echo(f"Processed {blocks_processed} blocks")
    typer.echo(f"Created {paragraphs_created} paragraphs")
    typer.echo(f"Output: {run_dir / 'paragraphs' / 'paragraphs.jsonl'}")
    typer.echo("Paragraphs stage complete.")


def _run_citations(manifest: Manifest) -> None:
    run_dir = RUN_ROOT / manifest.doc_id
    typer.echo(f"Run directory: {run_dir}")
    
    anchors_created, mappings_created = run_citations(
        run_dir=run_dir,
        manifest=manifest,
    )
    
    typer.echo(f"Extracted {anchors_created} citation anchors")
    typer.echo(f"Created {mappings_created} citation mappings")
    typer.echo(f"Output: {run_dir / 'citations' / 'cite_anchors.jsonl'}")
    typer.echo(f"Output: {run_dir / 'citations' / 'cite_map.jsonl'}")
    typer.echo(f"Output: {run_dir / 'refs' / 'doc_identity.json'}")
    typer.echo(f"Output: {run_dir / 'refs' / 'references_api.jsonl'}")
    typer.echo("Citations stage complete.")


def _run_figures_tables(manifest: Manifest) -> None:
    run_dir = RUN_ROOT / manifest.doc_id
    typer.echo(f"Run directory: {run_dir}")
    
    assets_created, images_cropped = run_figures_tables(
        run_dir=run_dir,
        manifest=manifest,
    )
    
    typer.echo(f"Extracted {assets_created} figure/table assets")
    typer.echo(f"Cropped {images_cropped} images")
    typer.echo(f"Index: {run_dir / 'figures_tables' / 'figure_table_index.jsonl'}")
    typer.echo(f"Links: {run_dir / 'figures_tables' / 'figure_table_links.json'}")
    typer.echo("Figures-tables stage complete.")


def _run_reading(manifest: Manifest, inject_reading_malformed_json: bool) -> None:
    run_dir = RUN_ROOT / manifest.doc_id
    typer.echo(f"Run directory: {run_dir}")
    
    facts_count, themes_count, lines_count, slots_count, fallback_count = run_reading(
        run_dir=run_dir,
        manifest=manifest,
        inject_malformed_json=inject_reading_malformed_json,
    )
    
    typer.echo(f"Extracted {facts_count} facts")
    typer.echo(f"Grouped into {themes_count} themes")
    typer.echo(f"Generated {lines_count} key evidence lines")
    typer.echo(f"Created {slots_count} figure/table slots")
    if fallback_count > 0:
        typer.echo(f"Fallback used: {fallback_count} times")
    typer.echo(f"Profile: {run_dir / 'reading' / 'paper_profile.json'}")
    typer.echo(f"Logic: {run_dir / 'reading' / 'logic_graph.json'}")
    typer.echo(f"Facts: {run_dir / 'reading' / 'facts.jsonl'}")
    typer.echo(f"Themes: {run_dir / 'reading' / 'themes.json'}")
    typer.echo(f"Synthesis: {run_dir / 'reading' / 'synthesis.json'}")
    
    if inject_reading_malformed_json:
        typer.echo("Fault injection: malformed-json enabled")
        typer.echo(f"Fault log: {run_dir / 'qa' / 'fault_injection.json'}")
    
    typer.echo("Reading stage complete.")


def _run_render(manifest: Manifest) -> None:
    run_dir = RUN_ROOT / manifest.doc_id
    typer.echo(f"Run directory: {run_dir}")
    
    sections_created, output_path = run_render(
        run_dir=run_dir,
        manifest=manifest,
    )
    
    typer.echo(f"Created {sections_created} sections")
    typer.echo(f"Output: {output_path}")
    typer.echo("Render stage complete.")


def _run_full(
    manifest: Manifest,
    inject_vision_malformed_json: bool,
    inject_reading_malformed_json: bool,
    inject_missing_font_stats: bool,
) -> None:
    """Run full pipeline with DAG-like orchestration and fixed output order."""
    typer.echo("=" * 50)
    typer.echo("Running FULL pipeline")
    typer.echo("=" * 50)
    
    run_dir = RUN_ROOT / manifest.doc_id
    
    stage_jobs = {
        "extractor": lambda: run_extractor(
            run_dir=run_dir,
            manifest=manifest,
            inject_missing_font_stats=inject_missing_font_stats,
        ),
        "overlay": lambda: run_overlay(
            run_dir=run_dir,
            manifest=manifest,
        ),
        "vision": lambda: run_vision(
            run_dir=run_dir,
            manifest=manifest,
            inject_malformed_json=inject_vision_malformed_json,
        ),
        "paragraphs": lambda: run_paragraphs(
            run_dir=run_dir,
            manifest=manifest,
        ),
        "citations": lambda: run_citations(
            run_dir=run_dir,
            manifest=manifest,
        ),
        "figures_tables": lambda: run_figures_tables(
            run_dir=run_dir,
            manifest=manifest,
        ),
        "reading": lambda: run_reading(
            run_dir=run_dir,
            manifest=manifest,
            inject_malformed_json=inject_reading_malformed_json,
        ),
        "render": lambda: run_render(
            run_dir=run_dir,
            manifest=manifest,
        ),
    }

    results = execute_levelized_dag(stage_jobs)

    typer.echo("\n[1/8] Extractor stage...")
    total_pages, total_blocks = results["extractor"]
    typer.echo(f"  Extracted {total_blocks} blocks from {total_pages} pages")
    
    typer.echo("\n[2/8] Overlay stage...")
    pages_processed, blocks_drawn = results["overlay"]
    typer.echo(f"  Drew {blocks_drawn} block overlays on {pages_processed} pages")
    
    typer.echo("\n[3/8] Vision stage...")
    pages_processed, blocks_processed = results["vision"]
    typer.echo(f"  Analyzed {blocks_processed} blocks on {pages_processed} pages")
    
    typer.echo("\n[4/8] Paragraphs stage...")
    paragraphs_created, blocks_processed = results["paragraphs"]
    typer.echo(f"  Created {paragraphs_created} paragraphs from {blocks_processed} blocks")
    
    typer.echo("\n[5/8] Citations stage...")
    anchors_created, mappings_created = results["citations"]
    typer.echo(f"  Extracted {anchors_created} citation anchors, {mappings_created} mappings")
    
    typer.echo("\n[6/8] Figures-tables stage...")
    assets_created, images_cropped = results["figures_tables"]
    typer.echo(f"  Extracted {assets_created} assets, cropped {images_cropped} images")
    
    typer.echo("\n[7/8] Reading stage...")
    facts_count, themes_count, lines_count, slots_count, fallback_count = results["reading"]
    typer.echo(f"  Extracted {facts_count} facts, {themes_count} themes")
    typer.echo(f"  Generated {lines_count} evidence lines, {slots_count} figure/table slots")
    
    typer.echo("\n[8/8] Render stage...")
    sections_created, output_path = results["render"]
    typer.echo(f"  Created {sections_created} sections -> {output_path}")
    
    typer.echo("\n" + "=" * 50)
    typer.echo("Full pipeline complete!")
    typer.echo("=" * 50)
    
    typer.echo("\n[9/9] Verify stage...")
    exit_code = run_verify(manifest.doc_id)
    if exit_code != 0:
        typer.echo(f"Verification FAILED (exit code {exit_code})", err=True)
        raise typer.Exit(code=exit_code)
    
    typer.echo("Verification PASSED")
    typer.echo(f"QA Report: run/{manifest.doc_id}/qa/report.json")


def _run_verify(doc_id: str) -> None:
    """Run verification on an existing run."""
    typer.echo(f"Verifying run: {doc_id}")
    
    exit_code = run_verify(doc_id)
    
    if exit_code == 0:
        typer.echo("Verification PASSED")
    else:
        typer.echo(f"Verification FAILED (exit code {exit_code})", err=True)
    
    run_dir = RUN_ROOT / doc_id
    if (run_dir / "qa" / "report.json").exists():
        typer.echo(f"QA Report: {run_dir / 'qa' / 'report.json'}")
    if (run_dir / "qa" / "stage_status.json").exists():
        typer.echo(f"Stage Status: {run_dir / 'qa' / 'stage_status.json'}")
    
    raise typer.Exit(code=exit_code)


def _resolve_manifest(pdf: Optional[Path], doc_id: Optional[str]) -> Manifest:
    if doc_id:
        run_dir = RUN_ROOT / doc_id
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            return load_manifest(run_dir)
    
    if pdf:
        if not pdf.exists():
            typer.echo(f"Error: PDF file not found: {pdf}", err=True)
            raise typer.Exit(code=1)
        manifest, _ = create_manifest(pdf_path=pdf, doc_id=doc_id, run_root=RUN_ROOT)
        return manifest
    
    typer.echo(f"Error: No manifest found for doc_id: {doc_id}", err=True)
    raise typer.Exit(code=1)


def main() -> None:
    """Main entry point for console script and module execution."""
    app()


if __name__ == "__main__":
    main()
