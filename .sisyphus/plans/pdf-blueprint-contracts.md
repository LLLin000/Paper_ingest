# PDF Blueprint Contracts (v1)

## Purpose
This file materializes the implementation contracts referenced by the work plan.
It is the normative spec for v1 schema and interface behavior.

## Canonical Inputs
- Primary corpus path: `example_pdf/`
- Default smoke file: `example_pdf/Bedi 等 - 2024 - Rotator cuff tears.pdf`
- Secondary stress files:
  - `example_pdf/Yao 等 - 2025 - Advances in Electrical Materials for Bone and Cartilage Regeneration Developments, Challenges, and.pdf`
  - `example_pdf/Mengiardi 等 - 2004 - Frozen Shoulder MR Arthrographic Findings.pdf`

## Run Layout Contract
For `doc_id=<id>`, pipeline output root is `run/<id>/`.

Required directories:
- `run/<id>/pages/`
- `run/<id>/text/`
- `run/<id>/vision/`
- `run/<id>/figures_tables/`
- `run/<id>/paragraphs/`
- `run/<id>/citations/`
- `run/<id>/reading/`
- `run/<id>/obsidian/`

Required top-level metadata:
- `run/<id>/manifest.json`

Final note output (canonical):
- `run/<id>/obsidian/<id>.md`

## Manifest Contract
Required keys:
- `doc_id`
- `input_pdf_path`
- `input_pdf_sha256`
- `started_at_utc`
- `toolchain` (python version, package lock hash)
- `model_config` (text model, vision model, temperature)
- `render_config` (`dpi`, `scale`)
- `pipeline_version`

## Data Contracts (minimum)

### vision/pXXX_in.json
Each record must include:
- `page`
- `image_path`
- `blocks` (array of block candidates with id/text/bbox)
- `constraints`

### vision/pXXX_out.json
Each record must include:
- `page`
- `reading_order` (ordered block ids)
- `merge_groups` (group id + block ids)
- `role_labels` (group/region role labels)
- `confidence`
- `fallback_used` (boolean)

Optional but recommended page-level region outputs:
- `figure_regions` (array)
- `table_regions` (array)

Allowed role labels (enum):
- `Body`
- `Heading`
- `FigureCaption`
- `TableCaption`
- `Footnote`
- `ReferenceList`
- `Sidebar`
- `HeaderFooter`

### reading/paper_profile.json
Must include:
- `paper_type`
- `paper_type_confidence`
- `research_problem`
- `claimed_contribution`
- `reading_strategy`

### reading/logic_graph.json
Must include:
- `nodes`
- `edges`
- `argument_flow`

### reading/themes.json
Must include:
- `themes`
- `cross_theme_links`
- `contradictions` (nullable)

### figures_tables/figure_table_index.jsonl
Each record must include:
- `asset_id` (`fig_*` or `tbl_*`)
- `asset_type` (`figure` or `table`)
- `page`
- `bbox_px`
- `caption_text`
- `caption_id` (if detectable)
- `source_para_id` (nullable)
- `image_path` (for cropped asset)
- `text_content` (OCR/vision recognized text, nullable)
- `summary_content` (LLM summarized description, nullable)
- `confidence`

### figures_tables/figure_table_links.json
Must include:
- `by_section` (map section -> ordered asset ids)
- `by_fact` (map fact_id -> asset ids)
- `by_synthesis_slot` (map synthesis slot id -> asset ids)

### text/blocks_raw.jsonl
Each record must include:
- `block_id` (stable within document)
- `page` (1-based)
- `bbox_pt` (`[x0,y0,x1,y1]` in PDF points)
- `text`
- `font_stats` (object, may be partial)

### text/blocks_norm.jsonl
Extends `blocks_raw` with:
- `is_heading_candidate`
- `is_header_footer_candidate`
- `column_guess`

### paragraphs/paragraphs.jsonl
Each record must include:
- `para_id`
- `page_span`
- `role`
- `section_path`
- `text`
- `evidence_pointer` (page, bbox union, source block ids)
- `neighbors`
- `confidence`

### citations/cite_anchors.jsonl
Each record must include:
- `anchor_id`
- `page`
- `anchor_bbox`
- `anchor_text` (if available)
- `nearest_para_id`
- `link_type` (`internal` or `external`)

### citations/cite_map.jsonl
Each record must include:
- `anchor_id`
- `mapped_ref_key` (nullable)
- `strategy_used` (`internal_dest`, `regex`, `fuzzy`, `none`)
- `confidence`
- `reason` (required when unmapped)

### reading/facts.jsonl
Each record must include:
- `fact_id`
- `para_id`
- `category`
- `statement`
- `quote` (<=30 words unless truncated with reason)
- `evidence_pointer`

### reading/synthesis.json
Must include:
- `executive_summary`
- `key_evidence_lines` (array)
- each `key_evidence_line` must include `fact_ids` (>=1)
- `figure_table_slots` (array of placement slots in final summary)

Each `figure_table_slot` must include:
- `slot_id`
- `position_hint` (`inline_after_claim`, `section_appendix`, `end_of_summary`)
- `asset_ids`
- `render_mode` (`content_only` or `full_asset_embed`)

## ID Stability Rules
- `doc_id` deterministic default: SHA256 of input PDF bytes, first 16 hex chars.
- `block_id`: `p{page}_b{index}` by extractor order.
- `para_id`: hash of ordered source `block_id` list.
- Re-run with same input and same config must preserve IDs.

## Coordinate Rules
- PDF-native coordinates use points (`pt`) with origin from PyMuPDF convention.
- Rendered coordinates use pixels (`px`) at fixed `dpi` and `scale`.
- Any `bbox_px` must be derived from `bbox_pt * scale` and document this in output metadata.

## Schema Validation Contract
Schema files (to be created in implementation):
- `schemas/manifest.schema.json`
- `schemas/blocks_raw.schema.json`
- `schemas/blocks_norm.schema.json`
- `schemas/vision_in.schema.json`
- `schemas/vision_out.schema.json`
- `schemas/figure_table_index.schema.json`
- `schemas/figure_table_links.schema.json`
- `schemas/paragraph.schema.json`
- `schemas/cite_anchor.schema.json`
- `schemas/cite_map.schema.json`
- `schemas/paper_profile.schema.json`
- `schemas/logic_graph.schema.json`
- `schemas/fact.schema.json`
- `schemas/theme.schema.json`
- `schemas/synthesis.schema.json`
- `schemas/qa_report.schema.json`

Validation command contract:
`python -m ingest.validate --run run/<id> --schemas schemas/ --strict`

Pass criteria:
- Exit code `0`
- No schema errors
- No silent truncation errors

## Security and Runtime Contract
- No outbound network during parsing/extraction/normalization/validation stages.
- Model-call stages are explicitly allowed network egress if provider is remote:
  - `vision`
  - `reading`
- Per-document hard timeout required.
- On timeout/failure, emit partial artifacts plus explicit status file with reason.

Verification requirements for runtime safety:
- `run/<id>/qa/runtime_safety.json` must include:
  - `network_block_enforced` (boolean)
  - `timeout_seconds`
  - `timed_out` (boolean)
  - `stage`
- `verify` stage must fail if any parse-stage network egress attempt is detected.

## CLI Contract (canonical)
Implementation must expose both forms:
- module: `python -m ingest.cli`
- console script: `ingest_pdf`

Canonical syntax:
- `ingest_pdf --pdf "<path>" --doc_id <id> --stage <stage>`

Allowed stage enum:
- `init-only`
- `extractor`
- `overlay`
- `vision`
- `paragraphs`
- `citations`
- `figures-tables`
- `reading`
- `render`
- `full`
- `verify`

Determinism rule with doc ids:
- If `--doc_id` is provided, it is authoritative.
- If omitted, compute deterministic default from PDF hash prefix.

Stage input rule:
- For stages run with only `--doc_id`, implementation must load `run/<id>/manifest.json` and use:
  - `input_pdf_path`
  - `render_config`
  - prior stage artifacts
- If manifest is missing/invalid, emit explicit status failure artifact and non-zero exit.

## Packaging/Bootstrap Contract
Bootstrap must create:
- `pyproject.toml` with console script mapping `ingest_pdf = ingest.cli:main`
- install instruction contract: `pip install -e .`
- lockfile for dependencies (tool choice documented in bootstrap task output)

Task-0 acceptance is valid only if both invocations work:
- `python -m ingest.cli --help`
- `ingest_pdf --help`

## Model-Call Contract (Vision + Reading)
Configuration source order:
1. explicit CLI flags
2. env vars
3. `run/<id>/manifest.json` defaults

Manifest must record resolved values:
- `model_config.vision_provider`
- `model_config.vision_model`
- `model_config.reading_provider`
- `model_config.reading_model`
- `model_config.temperature`
- `model_config.prompt_bundle_version`

Prompt bundle contract:
- `prompts/vision_page_structure.md`
- `prompts/reading_profile.md`
- `prompts/reading_logic.md`
- `prompts/reading_facts.md`
- `prompts/reading_themes.md`
- `prompts/reading_synthesis.md`

Minimum prompt requirements (all model-call prompts):
- enforce strict JSON-only output (no prose wrapper)
- require explicit evidence fields (`page`, `bbox`, `quote`, source ids)
- require `missing_information` when evidence is absent
- forbid unsupported claims and implicit inference without linked facts
- define truncation behavior: when truncated, set explicit truncation metadata

No silent provider/model switching rule:
- Any fallback provider/model change must be recorded in:
  - `run/<id>/qa/model_events.json`
- Unrecorded switching is verify-fail.

## Citation Key Normalization
`mapped_ref_key` must use one of:
- `doi:<lowercase_doi>`
- `pmid:<digits>`
- `bib:<author_year_titlehash8>`

Conflict policy:
- Duplicate keys with different reference strings are invalid and must fail citation gate.

## Quality Gate Metric Definitions
All gate metrics are computed in `run/<id>/qa/report.json` and produced by:
- `python -m ingest.verify --run run/<id>`

Ground-truth/evaluation data contract:
- `eval/golden/<doc_id>.json` contains annotated targets for:
  - paragraph reading order pairs
  - column boundaries and interleave labels
  - citation marker -> reference mapping truth
  - figure/table -> caption links
- If no golden file exists for a doc, report must mark metrics as `not_evaluated` (never silently pass).

Required structure for `eval/golden/<doc_id>.json`:
- `doc_id` (string)
- `reading_order_pairs` (array of `{left_para_id,right_para_id,expected_order}`)
- `multi_column_pages` (array of page numbers)
- `interleave_labels` (array of `{page,has_interleave}`)
- `citation_truth` (array of `{marker_id,expected_ref_key}`)
- `figure_caption_truth` (array of `{figure_id,expected_caption_id}`)

Illustrative example:
```json
{
  "doc_id": "demo01",
  "reading_order_pairs": [{"left_para_id": "p1a", "right_para_id": "p1b", "expected_order": "left_before_right"}],
  "multi_column_pages": [2, 3],
  "interleave_labels": [{"page": 2, "has_interleave": false}],
  "citation_truth": [{"marker_id": "m12", "expected_ref_key": "doi:10.1038/s41572-024-00492-3"}],
  "figure_caption_truth": [{"figure_id": "fig2", "expected_caption_id": "cap2"}]
}
```

Metric definitions:
- `provenance_coverage = anchored_atomic_claims / total_atomic_claims`
- `strong_claim_unsupported_count = count(strong_claims_without_anchor)`
- `paragraph_inversion_rate = inverted_paragraph_pairs / evaluated_paragraph_pairs`
- `cross_column_interleave_page_rate = pages_with_interleave / pages_with_columns`
- `citation_mapping_coverage = mapped_markers / total_markers`
- `doi_pmid_precision = correct_doi_pmid / extracted_doi_pmid`
- `figure_caption_precision = correct_figure_caption_links / evaluated_figure_caption_links`
- `caption_id_retention = captions_with_id / captions_expected_with_id`
- `silent_truncation_count = count(truncations_without_metadata)`

Operational definitions:
- `atomic_claim`: minimal factual statement asserting one proposition.
- `strong_claim`: numeric, causal, comparative efficacy, or guideline-like recommendation statement.

Computation source rules (mandatory):
- `atomic_claim` universe is `reading/synthesis.json.key_evidence_lines[]` (one entry = one claim).
- `anchored_atomic_claim` requires:
  - non-empty `fact_ids[]`
  - every linked `fact_id` resolves in `reading/facts.jsonl`
  - each resolved fact has non-empty `quote` and `evidence_pointer.page` + `evidence_pointer.bbox`.
- `strong_claim` is true when claim text contains numeric/comparative/causal cues OR any linked fact category in `{result,statistics,comparison,limitation}`.

Gate pass/fail thresholds:
- provenance pass if `provenance_coverage >= 0.95` and `strong_claim_unsupported_count == 0`
- reading order pass if `paragraph_inversion_rate <= 0.02` and `cross_column_interleave_page_rate <= 0.01`
- citation pass if `citation_mapping_coverage >= 0.98` and `doi_pmid_precision >= 0.99`
- figure-caption pass if `figure_caption_precision >= 0.90` and `caption_id_retention >= 0.95`
- truncation pass if `silent_truncation_count == 0`

Metric procedure notes (mandatory):
- `paragraph_inversion_rate` computed only on `Body` paragraph pairs in golden annotations.
- `cross_column_interleave_page_rate` computed on pages annotated as multi-column in golden file.
- `doi_pmid_precision` computed against golden citation truth for the same document.
- `figure_caption_precision` computed against golden figure-caption link truth.

## Hard-Stop vs Soft-Degrade Matrix
For single-stage execution (`--stage <x>`):
- hard-stop (non-zero exit): schema-invalid critical outputs, missing manifest for doc-only stages, verify blocker failure.
- soft-degrade (zero exit with quality flag): low-confidence extraction/recognition where fallback succeeded and required artifacts are still emitted.

For `--stage full`:
- hard-stop if any release blocker fails (`provenance`, `reading_order`, `citation`).
- otherwise continue with degradation labels (`quality: low`, `limitations`) if non-blocker gates fail.

Status artifact contract:
- `run/<id>/qa/stage_status.json` must include per-stage:
  - `stage`
  - `status` (`pass`, `degraded`, `failed`)
  - `hard_stop` (boolean)
  - `reason`

## Fault Injection Contract (for acceptance testing)
Implementation must support deterministic fault hooks:
- `--inject-vision-malformed-json`
- `--inject-reading-malformed-json`
- `--inject-missing-font-stats`

Expected behavior:
- Fault hook triggers retry/fallback path.
- Fallback event must be logged in `run/<id>/qa/fault_injection.json` with:
  - `stage`
  - `fault`
  - `retry_attempts`
  - `fallback_used`
  - `status`

## Runtime Safety Enforcement Mechanism
Parse stages (`init-only`, `extractor`, `overlay`, `paragraphs`, `citations`, `verify`) must run under network-deny mode.

Network-deny evidence requirements:
- `run/<id>/qa/runtime_safety.json` includes:
  - `network_deny_mode: true`
  - `egress_attempt_count`
  - `egress_attempt_targets` (array)
- Verify stage fails if `egress_attempt_count > 0` for parse stages.

## Golden Corpus Regression Contract
Minimum corpus for regression:
- `example_pdf/Bedi 等 - 2024 - Rotator cuff tears.pdf`
- `example_pdf/Yao 等 - 2025 - Advances in Electrical Materials for Bone and Cartilage Regeneration Developments, Challenges, and.pdf`
- `example_pdf/Mengiardi 等 - 2004 - Frozen Shoulder MR Arthrographic Findings.pdf`

Minimum report file:
- `run/<id>/qa/report.json`

Report must include:
- provenance coverage
- reading-order inversion rate
- citation mapping coverage and precision
- figure-caption association precision
- truncation count and reasons
