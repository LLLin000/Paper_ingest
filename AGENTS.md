# AGENTS.md

Repository guidance for coding agents working in `D:\L\AI\Paper_ingest`.

## Project Overview

- Language: Python (`requires-python >=3.11`).
- Packaging: `hatchling` backend in `pyproject.toml`.
- Layout: src-layout package under `src/ingest/`.
- Console script: `ingest_pdf = ingest.cli:main`.
- Domain: contract-driven PDF ingestion pipeline with structured outputs and Obsidian rendering.

## Key Source Files

- `pyproject.toml` - build metadata, dependencies, script entrypoint.
- `.sisyphus/plans/pdf-blueprint-contracts.md` - normative IO and behavior contracts.
- `.sisyphus/plans/pdf-review-system.md` - implementation and QA plan context.
- `src/ingest/*.py` - pipeline implementation by stage.

Read contract files before making behavioral changes.

## Setup Commands

- Editable install:
  - `pip install -e .`
- Dev install (includes pytest/jsonschema extras):
  - `pip install -e .[dev]`
- Lockfile install (if desired):
  - `pip install -r requirements.lock`

## Build / Lint / Test / Verify

No dedicated linter config files were found (no ruff/black/mypy config in repo root).

### Core checks

- CLI check (module form):
  - `python -m ingest.cli --help`
- CLI check (console script form):
  - `ingest_pdf --help`
- Test suite:
  - `pytest`
- Test discovery only:
  - `pytest --collect-only`

### Run a single test (important)

- Single test file:
  - `pytest tests/path/test_file.py`
- Single test function:
  - `pytest tests/path/test_file.py::test_name`
- Single class test:
  - `pytest tests/path/test_file.py::TestClass::test_name`
- Target via keyword expression:
  - `pytest -k "expr"`

### Pipeline execution examples

- Init-only stage:
  - `ingest_pdf --pdf "example_pdf/<file>.pdf" --doc_id demo01 --stage init-only`
- Full pipeline:
  - `ingest_pdf --pdf "example_pdf/<file>.pdf" --doc_id demo01 --stage full`
- Verify existing run:
  - `ingest_pdf --doc_id demo01 --stage verify`

### Contract validator command

- From blueprint contract:
  - `python -m ingest.validate --run run/<id> --schemas schemas/ --strict`

### Packaging notes

- `hatchling` backend is configured, but `hatch` CLI may not be installed in every environment.
- `python -m build` requires `build` package and may be unavailable by default.
- Prefer install + CLI verification unless packaging tasks are explicitly requested.

## Repository Layout Notes

- `src/ingest/` - stage modules (`extractor`, `overlay`, `vision`, `paragraphs`, `citations`, `reading`, `render`, `verify`).
- `schemas/` - JSON schema files.
- `prompts/` - prompt templates for model-call stages.
- `eval/golden/` - golden labels for quality metrics.
- `example_pdf/` - smoke/stress input files.
- `run/<doc_id>/` - generated per-document artifacts.

## Code Style Guidelines

Follow existing conventions in `src/ingest/*.py`.

### Imports

- Order imports as: standard library, third-party, local package.
- Keep imports explicit; avoid wildcard imports.
- Use relative imports inside package modules where current code does.

### Formatting

- 4-space indentation.
- Triple-double-quoted module/function docstrings for non-trivial APIs.
- Keep functions focused; split helpers instead of large mixed-responsibility functions.
- Use blank lines between logical blocks for readability.

### Types

- Add type hints for new/changed functions and return values.
- Prefer concrete modern typing (`list[str]`, `dict[str, Any]`, `tuple[int, int]`).
- Keep annotation style consistent within the edited file.
- Use dataclasses for transient structured records.
- Use Pydantic models for persisted contract-bound objects (manifest/config schemas).

### Naming

- `snake_case` for functions, variables, filenames.
- `PascalCase` for classes, dataclasses, enums.
- `UPPER_SNAKE_CASE` for constants.
- Keep contract field names exact (for example `mapped_ref_key`, `figure_table_slots`).

### Error Handling

- Fail explicitly on missing/invalid inputs; avoid silent failures.
- CLI layer should return non-zero exit on user/actionable errors (`typer.Exit(code=...)`).
- Lower layers should raise specific exceptions when possible.
- Only catch exceptions when adding meaningful fallback/recovery behavior.
- If fallback paths are triggered, emit required QA/status artifacts per contracts.

### File IO and JSON/JSONL

- Use UTF-8 for text and JSON files.
- JSONL must be one JSON object per line.
- Preserve Unicode text where needed (`ensure_ascii=False` is already used in outputs).
- Preserve deterministic IDs and contract-required keys across reruns.

### Change Scope

- Make minimal, targeted edits.
- Bugfixes: do not perform broad refactors unless requested.
- Keep schema/artifact name compatibility with blueprint contracts.

### Patch Tool Rule

- When using patch/apply_patch, paths MUST be workspace-relative paths (for example `src/ingest/cli.py`).
- Do NOT use absolute paths in patch headers; absolute paths can cause patch application to fail.

## Testing and Validation Expectations

- For logic changes, run focused checks first (single test or closest stage command).
- Then run broader checks (`pytest`, or stage + verify commands as appropriate).
- For schema-sensitive changes, run contract validator.
- If no tests cover changed behavior, add focused tests where practical.

## Cursor / Copilot Rules Status

Checked locations:

- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

Current repository status:

- No Cursor rules found.
- No Copilot instruction file found.

If those files are later added, treat them as high-priority policy and update this file.

## Agent Checklist Before Handoff

1. Confirm commands referenced in your response exist in this repo context.
2. Confirm changed outputs remain contract-compliant (`.sisyphus/plans/pdf-blueprint-contracts.md`).
3. Avoid introducing dependencies unless required and requested.
4. Avoid editing generated `run/` artifacts unless task explicitly targets fixtures/samples.
5. Report any environment gaps clearly (missing tools, no tests collected, etc.).
