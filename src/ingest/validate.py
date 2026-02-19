"""Schema validation entrypoint for PDF ingest pipeline artifacts.

Validation command contract per blueprint:
  python -m ingest.validate --run run/<id> --schemas schemas/ --strict

Pass criteria:
  - Exit code 0
  - No schema errors
  - No silent truncation errors
"""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="validate",
    help="Validate run artifacts against JSON schemas.",
    add_completion=False,
)


@app.command()
def validate(
    run: Path = typer.Option(
        ...,
        "--run",
        help="Path to run directory (e.g., run/<doc_id>)",
        exists=False,
    ),
    schemas: Path = typer.Option(
        Path("schemas"),
        "--schemas",
        help="Path to schemas directory",
        exists=False,
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on any validation warning",
    ),
) -> None:
    """Validate all JSON artifacts in a run directory against schemas.
    
    \b
    Exit codes:
        0 - All validations passed
        1 - Validation errors found
        2 - Missing required artifacts
    """
    typer.echo(f"[Bootstrap placeholder] Validation not yet implemented.")
    typer.echo(f"Run directory: {run}")
    typer.echo(f"Schemas directory: {schemas}")
    typer.echo(f"Strict mode: {strict}")
    raise typer.Exit(code=0)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
