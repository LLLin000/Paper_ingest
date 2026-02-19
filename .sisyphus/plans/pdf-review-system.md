# Stable PDF Close-Reading System Plan (PDF -> Obsidian)

## Context

### Original Request

Build a stable end-to-end pipeline that ingests arbitrary academic PDFs and outputs evidence-traceable Obsidian Markdown literature notes, plus structured cognition artifacts (facts/themes/logic) for review writing.

### Normative Blueprint

- Contract source file: `.sisyphus/plans/pdf-blueprint-contracts.md`
- This contract file is mandatory reference for all implementation tasks.

### Interview Summary

- User supplied a complete blueprint (A-E modules) with explicit stage responsibilities and IO contracts.
- Core requirement: every key conclusion must trace to evidence (`page + bbox + quote + para_id`), with strict anti-hallucination rules.
- Robustness is mandatory: schema validation, fallback chains, and graceful degradation.

### Metis Review (addressed)

- Added explicit MVP boundaries and non-goals.
- Added hard-stop vs soft-degrade policy.
- Added reproducibility and run metadata requirements.
- Added measurable acceptance criteria for traceability and quality gates.

### Corpus Reality Check (sample PDFs)

- Verified with mixed-era real papers in `example_pdf/` (2025 review journals + 2004 radiology paper).
- Confirmed high-risk patterns: two-column flow, boxed sidebars, figure/table-dense pages, very long references, and extraction artifacts (ligature/OCR-like noise).
- Plan updated to enforce quantified gates for reading order, provenance, and citation mapping.

---

## Work Objectives

### Core Objective

Deliver a reproducible, contract-driven PDF reading system that converts layout-rich papers into structured, evidence-grounded artifacts and Obsidian notes suitable for literature synthesis.

### Concrete Deliverables

- CLI entrypoint for single-document ingest (`ingest_pdf --pdf <path> --doc_id <id>`).
- Standard run workspace with deterministic manifest.
- Intermediate artifacts:
  - `text/blocks_raw.jsonl`
  - `text/blocks_norm.jsonl`
  - `vision/pXXX_in.json`
  - `vision/pXXX_out.json`
  - `figures_tables/figure_table_index.jsonl`
  - `figures_tables/figure_table_links.json`
  - `paragraphs/paragraphs.jsonl`
  - `citations/cite_anchors.jsonl`
  - `citations/cite_map.jsonl`
  - `reading/paper_profile.json`
  - `reading/logic_graph.json`
  - `reading/facts.jsonl`
  - `reading/themes.json`
  - `reading/synthesis.json`
- Final Obsidian note: `run/{doc_id}/obsidian/{doc_id}.md` (plus optional split notes).

### Definition of Done

- [ ] Pipeline runs from PDF input to final markdown output without manual edits.
- [ ] 100% of key synthesis statements reference `fact_id` entries with valid evidence pointers.
- [ ] Fallback/degradation paths are executed and logged for simulated failure cases.
- [ ] All produced JSON outputs pass schema checks defined in `.sisyphus/plans/pdf-blueprint-contracts.md`.

### Must Have

- Universal Core chain: Orientation -> Logic -> Facts -> Themes -> Synthesis.
- Evidence-first generation and validation.
- Missing data expressed as `missing_information` (or `category=none` for fact fallback), never fabricated.
- Figure/table extraction and recognition pipeline with explicit placement into final synthesis slots.
- ?????????????????????????

### Must NOT Have (Guardrails)

- No unsupported synthesis claims.
- No silent model/provider switching.
- No v1 scope creep into plugin UI, multi-PDF synthesis, or heavy OCR-first architecture.

### v1 Release Blockers

- Blocker 1: Claim provenance + hallucination containment.
- Blocker 2: Two-column/box reading-order integrity.
- Blocker 3: In-text citation -> reference mapping completeness.

---

## Verification Strategy

### Test Decision

- Infrastructure exists: NO (not present in current workspace).
- User wants tests: Manual-first in this plan (with explicit command/output checks).
- Framework: none (deferred), but add schema and pipeline smoke scripts.

### Manual QA Rules

- Each module task includes direct command execution and expected artifact checks.
- For visual/structure corrections, store page overlays and compare merged paragraph coverage.
- For synthesis, run traceability audit script to ensure each key conclusion resolves to facts/evidence.
- Stage failure handling must follow hard-stop/soft-degrade matrix in `.sisyphus/plans/pdf-blueprint-contracts.md`.

### Quantified Quality Gates (must pass)

- Provenance gate: >=95% atomic claims anchored to evidence; 0 strong claims (numeric/causal/clinical) without evidence.
- Reading-order gate: <=2% paragraph inversions; <=1% pages with cross-column interleaving; sidebar text tagged/excluded on >=95% pages where present.
- Citation gate: >=98% in-text citation markers mapped; DOI/PMID precision >=99%; 0 duplicate/conflicting citation keys.
- Figure-caption gate: >=90% correct associations; caption identifier retention >=95% when present.
- Figure/table extraction gate: >=95% detectable figure/table regions indexed; >=90% assets have non-empty `caption_text` or `text_content`; synthesis placement slots resolved for >=90% cited assets.
- Truncation gate: 0 silent truncations; every truncation must be explicit with reason and dropped-span metadata.
- Determinism gate: same input PDF -> same `doc_id`, same output note path and stable internal IDs.
- Runtime safety gate: no network egress during parse/extract/normalize/validate stages; hard timeouts enforced with explicit status artifacts.
- Regression gate: golden corpus scoring for all critical gates with regression alert on >=3% absolute drop.

---

## Task Flow

`T0 Bootstrap/Contracts -> T1 Skeleton -> T2 Extractor -> T3 Overlay -> T4 Vision -> T5 Paragraph Builder -> T6 Citation Mapper -> T6.5 Figure/Table Extract+Recognize -> T7 Reading Engine -> T8 Obsidian Renderer -> T9 End-to-End QA`

### Parallelization

- T3 (overlay) can proceed in parallel with T2 late-stage normalization once raw blocks exist.
- T6 citation mapper can begin after T5 paragraph outputs exist.
- T6.5 figure/table extraction can run after T4 vision regions and before T7 synthesis.
- T8 renderer depends on T7 outputs.

### Execution Ordering Rule

- T6.5 must complete before T7 final synthesis pass. T7 consumes `figures_tables/figure_table_links.json` to build `figure_table_slots`.

---

## TODOs

- [x] 0. Bootstrap repository and contract baseline
  - What to do:
    - Define implementation runtime (Python 3.11+) and dependency lockfile.
    - Create package layout and CLI module path for `ingest_pdf`.
    - Add schema folder and validator entrypoint contract.
    - Add `pyproject.toml` console-script mapping and editable install path (`pip install -e .`).
    - Create prompt bundle files for vision + reading stages per contract.
    - Confirm `.sisyphus/plans/pdf-blueprint-contracts.md` is treated as normative spec.
  - Must NOT do:
    - Do not begin extractor logic before bootstrap and contracts are committed.
  - Parallelizable: NO
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md`
    - `.sisyphus/plans/pdf-review-system.md`
  - Acceptance criteria:
    - Command: `python -V`
    - Expected: Python 3.11+.
    - Command: `python -m ingest.cli --help`
    - Expected: CLI help renders canonical `--stage` enum (`init-only`, `extractor`, `overlay`, `vision`, `paragraphs`, `citations`, `reading`, `render`, `full`, `verify`).
    - Command: `ingest_pdf --help`
    - Expected: console script available and consistent with module CLI.

- [x] 1. Build run skeleton and manifest contract
  - What to do:
    - Define run directory structure under `run/{doc_id}/`.
    - Record config/model/version/input checksum in `manifest.json`.
    - Standardize path conventions for downstream modules.
  - Must NOT do:
    - Do not hardcode machine-specific absolute paths.
  - Parallelizable: NO
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (Run Layout + Manifest Contract).
  - Acceptance criteria:
    - Command: `ingest_pdf --pdf "example_pdf/Bedi 等 - 2024 - Rotator cuff tears.pdf" --doc_id test01 --stage init-only`
    - Expected: `run/test01/manifest.json` and all module folders created.
    - Evidence: terminal output + directory listing.

- [x] 2. Implement Text/Layout Extractor (A)
  - What to do:
    - Render `pages/pXXX.png` via PyMuPDF.
    - Extract line-level blocks to `text/blocks_raw.jsonl`.
    - Compute `blocks_norm.jsonl` with font statistics, heading candidates, header/footer repeat hints, and column guess.
  - Must NOT do:
    - Do not perform semantic rewriting in this module.
  - Parallelizable: NO (foundation stage)
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (`blocks_raw`/`blocks_norm` contracts + coordinate rules).
  - Acceptance criteria:
    - Command: `ingest_pdf --pdf "example_pdf/Bedi 等 - 2024 - Rotator cuff tears.pdf" --doc_id test01 --stage extractor`
    - Expected: pages and both jsonl files exist with non-empty records.
    - Fallback check: `ingest_pdf --doc_id test01 --stage extractor --inject-missing-font-stats`
    - Expected: output still includes bbox/text contract and `run/test01/qa/fault_injection.json` records fallback event.

- [x] 3. Implement overlay generation for visual debugging (B.1)
  - What to do:
    - Convert PT bbox to PX using render scale.
    - Draw block rectangles and IDs on annotated page images.
  - Must NOT do:
    - Do not mutate raw extraction data.
  - Parallelizable: YES (after T2 raw blocks)
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (coordinate conversion and bbox contract).
  - Acceptance criteria:
    - Command: `ingest_pdf --doc_id test01 --stage overlay`
    - Expected: `pages/pXXX_annot.png` for each page.

- [x] 4. Implement Vision Structure Corrector with fallback chain (B.2-B.4)
  - What to do:
    - Build per-page vision input package with strict constraints.
    - Parse model output JSON (`reading_order`, `merge_groups`, `role_labels`, confidence).
    - Add retry-once for parse errors; then rule-based fallback (column+y sort, spacing merge, caption regex).
  - Must NOT do:
    - Do not accept free-form non-JSON outputs.
  - Parallelizable: NO
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (`vision` IO contract, role enum, model-call contract).
  - Acceptance criteria:
    - Command: `ingest_pdf --doc_id test01 --stage vision`
    - Expected: `vision/pXXX_out.json` per page.
    - Fault injection: `ingest_pdf --doc_id test01 --stage vision --inject-vision-malformed-json`
    - Expected: malformed JSON triggers retry then fallback and logs in `run/test01/qa/fault_injection.json`.

- [x] 5. Build paragraph canonicalizer
  - What to do:
    - Aggregate merge groups across pages into `paragraphs/paragraphs.jsonl`.
    - Compute union bbox, neighbor links, section path where possible.
    - Preserve confidence metadata.
  - Must NOT do:
    - Do not drop unresolved paragraphs; mark uncertainty instead.
  - Parallelizable: NO
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (`paragraphs/paragraphs.jsonl` contract).
  - Acceptance criteria:
    - Command: `ingest_pdf --doc_id test01 --stage paragraphs`
    - Expected: every paragraph has `para_id`, `text`, `evidence_pointer`.

- [x] 6. Implement citation mapper (link-first, on-demand)
  - What to do:
    - Extract `page.get_links()` anchors to `cite_anchors.jsonl`.
    - Detect reference zone via role labels or heading fallback.
    - Map cite -> ref via internal destination, regex, fuzzy fallback.
  - Must NOT do:
    - Do not block pipeline if full citation mapping is impossible.
  - Parallelizable: YES (after T5)
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (`cite_anchors`/`cite_map` contracts).
  - Acceptance criteria:
    - Command: `ingest_pdf --doc_id test01 --stage citations`
    - Expected: anchors file + map file; unresolved entries include explicit reason.

- [x] 7. Implement reading engine (Universal Core D)
  - What to do:
    - D1 profile: classify paper type and reading strategy.
    - D2 logic graph: argument flow scaffold.
    - D3 facts: paragraph-level facts with quotes (<=30 words), fallback `category=none` if needed.
    - D4 themes: grouped facts + cross-theme links.
    - D5 synthesis: executive summary + traceable key lines.
    - Build figure/table placement slots (`figure_table_slots`) and link assets into summary structure.
    - Run final synthesis pass only after T6.5 artifacts exist.
  - Must NOT do:
    - Do not emit synthesis claims lacking `fact_id` trace.
  - Parallelizable: Partial (D1/D2 can overlap with early fact extraction prep)
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (`paper_profile`/`logic_graph`/`facts`/`themes`/`synthesis` + model-call contract).
  - Acceptance criteria:
    - Command: `ingest_pdf --doc_id test01 --stage reading`
    - Expected: all reading artifacts exist and pass schema checks.
    - Trace audit: every synthesis key line resolves to fact and evidence pointer.
    - Placement audit: every referenced figure/table slot resolves to existing asset ids.
    - Fault injection: `ingest_pdf --doc_id test01 --stage reading --inject-reading-malformed-json`
    - Expected: retry/fallback path logged in `run/test01/qa/fault_injection.json`.

- [x] 6.5. Implement figure/table extraction + recognition pipeline
  
  - What to do:
    - Detect figure/table regions from Text+Vision outputs and crop assets.
    - Build `figures_tables/figure_table_index.jsonl` with caption, OCR/vision text, and optional summary content.
    - Build `figures_tables/figure_table_links.json` to map assets to section/fact/synthesis slots.
    - Support two synthesis render modes: `content_only` and `full_asset_embed`.
  - Must NOT do:
    - Do not inject figure/table claims into summary without asset linkage and evidence metadata.
  - Parallelizable: YES (after T4/T5; before T7 final synthesis)
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (`figure_table_index`, `figure_table_links`, synthesis slot contract).
    - `example_pdf/Yao 等 - 2025 - Advances in Electrical Materials for Bone and Cartilage Regeneration Developments, Challenges, and.pdf` (high figure/table density).
  - Acceptance criteria:
    - Command: `ingest_pdf --doc_id test01 --stage figures-tables`
    - Expected: `figures_tables/figure_table_index.jsonl` and `figures_tables/figure_table_links.json` exist.
    - Expected: each asset has `asset_type`, `page`, `bbox_px`, and at least one of `caption_text`/`text_content`.

- [x] 8. Implement Obsidian renderer (E)
  - What to do:
    - Render `run/{doc_id}/obsidian/{doc_id}.md` with frontmatter, profile, logic, themes, evidence index, citations.
    - Render figure/table content in synthesis at slot positions (`content_only`), and optionally embed full assets (`full_asset_embed`).
    - Optional split notes for facts/graph/citations.
  - Must NOT do:
    - Do not include untraceable narrative text.
  - Parallelizable: NO (depends on T7)
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (deterministic IDs and output contract).
    - `obsidian-markdown` skill conventions for frontmatter and markdown syntax.
  - Acceptance criteria:
    - Command: `ingest_pdf --doc_id test01 --stage render`
    - Expected: markdown file exists at `run/test01/obsidian/test01.md`.
    - Expected frontmatter keys: `doc_id`, `paper_type`, `source_pdf`, `quality`.
    - Expected required sections: `Paper Profile`, `Logic Graph`, `Themes`, `Evidence Index`, `Citations`.
    - Expected figure/table section behavior: slot-based placement is preserved; embed mode follows synthesis slot `render_mode`.
    - Expected traceability invariant: every synthesis key line references >=1 `fact_id`; every referenced fact resolves to evidence pointer with page+bbox+quote.

- [x] 9. Add quality gates, degradation labels, and E2E verification
  - What to do:
    - Implement structural gates (coverage, role recognition, confidence thresholds).
    - Implement reading gates (fact count floor, quote completeness, synthesis traceability).
    - Add degradation label (`quality: low`, `limitations`) when extraction reliability is poor.
    - Add end-to-end report summarizing pass/fail, fallback usage, unresolved citations.
    - Build golden-corpus regression harness from `example_pdf/` and publish per-doc gate scores.
    - Add `eval/golden/*.json` annotation files required for compute-only metrics (reading order, citation correctness, figure-caption correctness).
    - Define annotation workflow for golden files: generate stable ids after `paragraphs`/`citations` stages, then annotate truth mappings and save to `eval/golden/<doc_id>.json`.
    - Implement deterministic fault injection hooks (`--inject-vision-malformed-json`, `--inject-reading-malformed-json`, `--inject-missing-font-stats`) and collect `run/<id>/qa/fault_injection.json`.
    - Implement runtime safety evidence output `run/<id>/qa/runtime_safety.json` for parse-stage network-deny checks.
    - Emit `run/<id>/qa/stage_status.json` and enforce hard-stop/soft-degrade matrix.
    - Enforce hard fail for v1 release blockers (provenance, reading order, citation mapping).
  - Must NOT do:
    - Do not silently pass failed gates.
  - Parallelizable: NO
  - References:
    - `.sisyphus/plans/pdf-blueprint-contracts.md` (schema validation, runtime safety, regression contract).
    - `example_pdf/Bedi 等 - 2024 - Rotator cuff tears.pdf` (two-column + boxed content + long references).
    - `example_pdf/Yao 等 - 2025 - Advances in Electrical Materials for Bone and Cartilage Regeneration Developments, Challenges, and.pdf` (figure/table-heavy review with dense captions).
    - `example_pdf/Mengiardi 等 - 2004 - Frozen Shoulder MR Arthrographic Findings.pdf` (older formatting style, dense metadata and variable layout cues).
  - Acceptance criteria:
    - Command: `ingest_pdf --pdf "example_pdf/Yao 等 - 2025 - Advances in Electrical Materials for Bone and Cartilage Regeneration Developments, Challenges, and.pdf" --doc_id stress01 --stage full`
    - Expected: final markdown + machine-readable QA report with explicit gate outcomes.
    - Expected: QA report includes all quantified gate values and PASS/FAIL per gate.
    - Expected: release blockers failing => non-zero verify exit status.
    - Command: `ingest_pdf --doc_id stress01 --stage verify`
    - Expected: computed metrics and gate formulas from contract file are present in `run/stress01/qa/report.json`.

---

## Commit Strategy

> Apply this section only after repository is git-enabled.

- After T1-T2: `feat(ingest): add run skeleton and text extraction contracts`
- After T3-T5: `feat(layout): add overlay, vision correction, and paragraph canonicalizer`
- After T6-T7: `feat(reading): add citation mapping and universal core reading engine`
- After T8-T9: `feat(render): add obsidian output and quality gate enforcement`

Each commit requires corresponding stage command execution and artifact existence checks.

---

## Success Criteria

### Verification Commands

```bash
ingest_pdf --pdf "example_pdf/Bedi 等 - 2024 - Rotator cuff tears.pdf" --doc_id demo01 --stage full
ingest_pdf --doc_id demo01 --stage verify
```

### Final Checklist

- [ ] Output markdown exists and is readable in Obsidian.
- [ ] All key conclusions are evidence-traceable.
- [ ] Fallback/degrade behavior is explicit and logged.
- [ ] Schema validation passes for all JSON artifacts.
- [ ] No unsupported synthesis statement remains.
- [ ] All v1 release blockers pass on golden corpus.
