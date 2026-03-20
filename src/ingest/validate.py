"""Schema validation entrypoint for PDF ingest pipeline artifacts.

Strict mode is used for QA release gating and fails on missing artifacts,
schema violations, and cross-file invariant violations.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import typer

try:
    from jsonschema import Draft7Validator
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    Draft7Validator = None

app = typer.Typer(
    name="validate",
    help="Validate run artifacts against JSON schemas.",
    add_completion=False,
)

_ARTIFACT_SCHEMA_MAP: dict[str, str] = {
    "manifest.json": "manifest.schema.json",
}

_STRICT_REQUIRED_DIRECTORIES: tuple[str, ...] = (
    "pages",
    "text",
    "vision",
    "figures_tables",
    "paragraphs",
    "citations",
    "reading",
    "obsidian",
)

_STRICT_REQUIRED_FILES: tuple[str, ...] = (
    "manifest.json",
    "qa/report.json",
    "qa/stage_status.json",
    "qa/runtime_safety.json",
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]


class _ValidationErrorLike(Protocol):
    path: object
    message: str


def _read_json_dict(path: Path) -> dict[str, JsonValue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path}: line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON type in {path}: expected object, got {type(payload).__name__}")

    normalized = cast(dict[str, JsonValue], {str(key): value for key, value in payload.items()})
    return normalized


def _format_validation_error(
    *, artifact_path: Path, schema_path: Path, error: _ValidationErrorLike
) -> str:
    location = "$"
    path_parts = cast(object, error.path)
    if isinstance(path_parts, list | tuple):
        location = "$." + ".".join(str(part) for part in path_parts)
    return f"{artifact_path} against {schema_path} at {location}: {error.message}"


def _cross_file_invariant_errors(run: Path, manifest_obj: Mapping[str, JsonValue]) -> list[str]:
    errors: list[str] = []
    expected_doc_id = run.name
    actual_doc_id = manifest_obj.get("doc_id")

    if isinstance(actual_doc_id, str) and actual_doc_id != expected_doc_id:
        errors.append(
            f"{run / 'manifest.json'} doc_id mismatch: expected '{expected_doc_id}' from run path, got '{actual_doc_id}'"
        )

    report_path = run / "qa" / "report.json"
    if report_path.exists():
        try:
            report_obj = _read_json_dict(report_path)
        except ValueError as exc:
            errors.append(str(exc))
            return errors

        report_doc_id = report_obj.get("doc_id")
        if isinstance(actual_doc_id, str) and report_doc_id != actual_doc_id:
            errors.append(
                f"{report_path} doc_id mismatch: expected '{actual_doc_id}' from manifest.json, got '{report_doc_id}'"
            )

    if isinstance(actual_doc_id, str):
        expected_note = run / "obsidian" / f"{actual_doc_id}.md"
        if not expected_note.exists():
            errors.append(
                f"missing required artifact for strict validation: {expected_note} (invariant: obsidian note must match manifest doc_id)"
            )

    return errors


def _strict_required_artifact_errors(run: Path) -> list[str]:
    missing: list[str] = []

    for directory in _STRICT_REQUIRED_DIRECTORIES:
        target = run / directory
        if not target.is_dir():
            missing.append(f"missing required artifact for strict validation: {target} (required directory)")

    for rel_file in _STRICT_REQUIRED_FILES:
        target = run / rel_file
        if not target.exists():
            missing.append(f"missing required artifact for strict validation: {target} (required file)")

    return missing


def _strict_additional_artifact_errors(run: Path, doc_id: str) -> list[str]:
    errors: list[str] = []

    def expect_object_key(
        obj: Mapping[str, JsonValue],
        *,
        artifact_path: Path,
        key: str,
        expected_type: str | None = None,
    ) -> None:
        if key not in obj:
            errors.append(f"{artifact_path} at $.{key}: missing required key")
            return
        value = obj.get(key)
        if expected_type == "object" and not isinstance(value, dict):
            errors.append(
                f"{artifact_path} at $.{key}: expected object, got {type(value).__name__}"
            )
        if expected_type == "array" and not isinstance(value, list):
            errors.append(
                f"{artifact_path} at $.{key}: expected array, got {type(value).__name__}"
            )
        if expected_type == "string" and not isinstance(value, str):
            errors.append(
                f"{artifact_path} at $.{key}: expected string, got {type(value).__name__}"
            )
        if expected_type == "boolean" and not isinstance(value, bool):
            errors.append(
                f"{artifact_path} at $.{key}: expected boolean, got {type(value).__name__}"
            )

    def read_if_exists(path: Path) -> dict[str, JsonValue] | None:
        if not path.exists():
            return None
        try:
            return _read_json_dict(path)
        except ValueError as exc:
            errors.append(str(exc))
            return None

    report_path = run / "qa" / "report.json"
    report_obj = read_if_exists(report_path)
    if report_obj is not None:
        expect_object_key(report_obj, artifact_path=report_path, key="doc_id", expected_type="string")
        expect_object_key(report_obj, artifact_path=report_path, key="overall_status", expected_type="string")
        expect_object_key(report_obj, artifact_path=report_path, key="hard_stop", expected_type="boolean")
        expect_object_key(report_obj, artifact_path=report_path, key="gates", expected_type="object")
        report_doc_id = report_obj.get("doc_id")
        if isinstance(report_doc_id, str) and report_doc_id != doc_id:
            errors.append(
                f"{report_path} at $.doc_id: expected '{doc_id}' from manifest/run, got '{report_doc_id}'"
            )

    stage_status_path = run / "qa" / "stage_status.json"
    stage_status_obj = read_if_exists(stage_status_path)
    if stage_status_obj is not None:
        expect_object_key(stage_status_obj, artifact_path=stage_status_path, key="stage", expected_type="string")
        expect_object_key(stage_status_obj, artifact_path=stage_status_path, key="status", expected_type="string")
        expect_object_key(
            stage_status_obj,
            artifact_path=stage_status_path,
            key="hard_stop",
            expected_type="boolean",
        )
        expect_object_key(stage_status_obj, artifact_path=stage_status_path, key="reason", expected_type="string")

    runtime_safety_path = run / "qa" / "runtime_safety.json"
    runtime_safety_obj = read_if_exists(runtime_safety_path)
    if runtime_safety_obj is not None:
        expect_object_key(
            runtime_safety_obj,
            artifact_path=runtime_safety_path,
            key="network_deny_mode",
            expected_type="boolean",
        )
        expect_object_key(
            runtime_safety_obj,
            artifact_path=runtime_safety_path,
            key="egress_attempt_targets",
            expected_type="array",
        )

    structure_quality_path = run / "qa" / "structure_quality.json"
    structure_quality_obj = read_if_exists(structure_quality_path)
    if structure_quality_obj is not None:
        expect_object_key(structure_quality_obj, artifact_path=structure_quality_path, key="doc_id", expected_type="string")
        expect_object_key(
            structure_quality_obj,
            artifact_path=structure_quality_path,
            key="ordering_confidence_low",
            expected_type="boolean",
        )
        expect_object_key(
            structure_quality_obj,
            artifact_path=structure_quality_path,
            key="section_boundary_unstable",
            expected_type="boolean",
        )
        expect_object_key(
            structure_quality_obj,
            artifact_path=structure_quality_path,
            key="reference_region_ambiguous",
            expected_type="boolean",
        )
        expect_object_key(
            structure_quality_obj,
            artifact_path=structure_quality_path,
            key="caption_linking_partial",
            expected_type="boolean",
        )

    profile_path = run / "reading" / "paper_profile.json"
    profile_obj = read_if_exists(profile_path)
    if profile_obj is not None:
        for key in (
            "paper_type",
            "paper_type_confidence",
            "research_problem",
            "claimed_contribution",
            "reading_strategy",
        ):
            expect_object_key(profile_obj, artifact_path=profile_path, key=key)

    synthesis_path = run / "reading" / "synthesis.json"
    synthesis_obj = read_if_exists(synthesis_path)
    if synthesis_obj is not None:
        expect_object_key(synthesis_obj, artifact_path=synthesis_path, key="executive_summary")
        expect_object_key(
            synthesis_obj,
            artifact_path=synthesis_path,
            key="key_evidence_lines",
            expected_type="array",
        )
        expect_object_key(
            synthesis_obj,
            artifact_path=synthesis_path,
            key="figure_table_slots",
            expected_type="array",
        )

    for vision_path in sorted((run / "vision").glob("p*_out.json")):
        vision_obj = read_if_exists(vision_path)
        if vision_obj is None:
            continue
        for key in ("page", "reading_order", "merge_groups", "role_labels", "confidence", "fallback_used"):
            expect_object_key(vision_obj, artifact_path=vision_path, key=key)

    return errors


def validate_run_artifacts(run: Path, schemas: Path, strict: bool) -> list[str]:
    if Draft7Validator is None:
        raise RuntimeError(
            "jsonschema is required for validation. Install dev dependencies (pip install -e .[dev])."
        )

    issues: list[str] = []
    if not run.exists() or not run.is_dir():
        return [f"run directory not found: {run}"]
    if not schemas.exists() or not schemas.is_dir():
        return [f"schemas directory not found: {schemas}"]

    if strict:
        issues.extend(_strict_required_artifact_errors(run))

    for artifact_name, schema_name in _ARTIFACT_SCHEMA_MAP.items():
        artifact_path = run / artifact_name
        schema_path = schemas / schema_name
        if not schema_path.exists():
            continue

        if not artifact_path.exists():
            if strict:
                issues.append(
                    f"missing required artifact for strict validation: {artifact_path} (schema: {schema_path})"
                )
            continue

        try:
            artifact_obj = _read_json_dict(artifact_path)
            schema_obj = _read_json_dict(schema_path)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        validator = Draft7Validator(schema_obj)
        validation_errors = sorted(validator.iter_errors(artifact_obj), key=lambda item: tuple(item.path))
        for error in validation_errors:
            issues.append(
                _format_validation_error(
                    artifact_path=artifact_path,
                    schema_path=schema_path,
                    error=error,
                )
            )

        if strict and artifact_name == "manifest.json":
            issues.extend(_cross_file_invariant_errors(run, artifact_obj))

    if strict:
        strict_doc_id = run.name
        manifest_path = run / "manifest.json"
        if manifest_path.exists():
            try:
                manifest_obj = _read_json_dict(manifest_path)
                manifest_doc_id = manifest_obj.get("doc_id")
                if isinstance(manifest_doc_id, str) and manifest_doc_id:
                    strict_doc_id = manifest_doc_id
            except ValueError:
                pass
        issues.extend(_strict_additional_artifact_errors(run, strict_doc_id))

    return issues


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
    """Validate run artifacts in a run directory against schemas."""
    try:
        issues = validate_run_artifacts(run=run, schemas=schemas, strict=strict)
    except RuntimeError as exc:
        typer.echo(f"Validation setup error: {exc}")
        raise typer.Exit(code=1) from exc

    if issues:
        missing_artifact_issues = [
            issue for issue in issues if issue.startswith("missing required artifact for strict validation:")
        ]
        typer.echo("Validation failed:")
        for issue in issues:
            typer.echo(f"- {issue}")
        raise typer.Exit(code=2 if missing_artifact_issues else 1)

    typer.echo(f"Validation passed for run: {run}")
    raise typer.Exit(code=0)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
