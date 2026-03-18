# Hierarchical Vision Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dense-page-aware hierarchical vision layout path so layout reasoning is no longer dominated by figure-internal micro text, while keeping heading recovery and figure/table recognition under vision control.

**Architecture:** Keep raw OCR/extractor blocks as the source of truth, but introduce a coarse vision pass for dense pages that predicts page regions first (`text_regions`, `caption_regions`, `figure_regions`, `table_regions`, `header_footer_regions`). After region projection, cluster micro text blocks inside dense figure/table areas and run a lightweight coherence classifier that only decides whether a cluster is narrative, non-narrative graphic text, tabular-like, or uncertain. Then run the existing fine-grained layout logic only on text/caption regions plus narrative/uncertain clusters, which avoids circular dependence on pre-labeled layout blocks while shrinking the high-cost full-page prompt on dense figure pages.

**Tech Stack:** Python 3.11, existing `ingest.vision` + `ingest.paragraphs` pipeline, JSON/JSONL contracts, pytest.

### Task 1: Add Dense-Page Classification

**Files:**
- Modify: `src/ingest/vision.py`
- Test: `tests/test_vision_dense_page_detection.py`

**Step 1: Write the failing test**

```python
def test_classify_dense_layout_page_flags_microblock_figure_page() -> None:
    blocks = [
        BlockCandidate(block_id=f"b{i}", text="0 1 2", bbox_pt=[0, 0, 10, 10], bbox_px=[0, 0, 10, 10], column_guess=1, is_heading_candidate=False, is_header_footer_candidate=False)
        for i in range(120)
    ]
    mode = classify_page_layout_mode(blocks)
    assert mode == "hierarchical"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_dense_page_detection.py::test_classify_dense_layout_page_flags_microblock_figure_page -q`

Expected: FAIL because `classify_page_layout_mode` does not exist.

**Step 3: Write minimal implementation**

- Add `classify_page_layout_mode(blocks)` to `src/ingest/vision.py`
- Start with a simple deterministic heuristic:
  - `hierarchical` when block count is high and average block text is short
  - otherwise `direct`
- Emit page-class telemetry for later tuning

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_dense_page_detection.py::test_classify_dense_layout_page_flags_microblock_figure_page -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_dense_page_detection.py src/ingest/vision.py
git commit -m "feat: classify dense pages for hierarchical vision"
```

### Task 2: Define Coarse Layout Pass Contract

**Files:**
- Modify: `src/ingest/vision.py`
- Modify: `.sisyphus/plans/pdf-blueprint-contracts.md`
- Test: `tests/test_vision_coarse_layout_contract.py`

**Step 1: Write the failing test**

```python
def test_validate_coarse_layout_output_accepts_region_schema() -> None:
    payload = {
        "page": 4,
        "text_regions": [{"region_id": "text_1", "bbox_px": [0, 0, 100, 100]}],
        "caption_regions": [],
        "figure_regions": [{"region_id": "fig_1", "bbox_px": [100, 100, 300, 300]}],
        "table_regions": [],
        "header_footer_regions": [],
        "confidence": 0.82,
    }
    ok, reason = validate_coarse_layout_output(payload, page=4)
    assert ok is True
    assert reason == "ok"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_coarse_layout_contract.py::test_validate_coarse_layout_output_accepts_region_schema -q`

Expected: FAIL because validator does not exist.

**Step 3: Write minimal implementation**

- Add parser/validator helpers for a coarse-layout JSON payload
- Keep schema intentionally small:
  - `page`
  - `text_regions`
  - `caption_regions`
  - `figure_regions`
  - `table_regions`
  - `header_footer_regions`
  - `confidence`
- Update contract notes in `.sisyphus/plans/pdf-blueprint-contracts.md`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_coarse_layout_contract.py::test_validate_coarse_layout_output_accepts_region_schema -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_coarse_layout_contract.py src/ingest/vision.py .sisyphus/plans/pdf-blueprint-contracts.md
git commit -m "feat: add coarse vision layout contract"
```

### Task 3: Add Coarse Layout Prompt and Dense-Page Execution Path

**Files:**
- Modify: `src/ingest/vision.py`
- Create: `prompts/vision_page_regions.md`
- Test: `tests/test_vision_hierarchical_layout_flow.py`

**Step 1: Write the failing test**

```python
def test_run_vision_uses_coarse_layout_path_for_dense_page(tmp_path: Path, monkeypatch) -> None:
    # Build one dense page and assert the first prompt is region-only rather than full block layout.
    ...
    assert coarse_calls == 1
    assert fine_calls == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_hierarchical_layout_flow.py::test_run_vision_uses_coarse_layout_path_for_dense_page -q`

Expected: FAIL because `run_vision` only has the direct path today.

**Step 3: Write minimal implementation**

- Add `build_coarse_layout_prompt(...)`
- Add `process_page_hierarchical(...)`
- In `run_vision`, dispatch by page mode:
  - `direct` -> current path
  - `hierarchical` -> coarse pass first
- Save coarse output alongside existing page artifacts, for example inside `vision/pXXX_regions.json`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_hierarchical_layout_flow.py::test_run_vision_uses_coarse_layout_path_for_dense_page -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_hierarchical_layout_flow.py prompts/vision_page_regions.md src/ingest/vision.py
git commit -m "feat: add hierarchical vision path for dense pages"
```

### Task 4: Project Raw Blocks Into Coarse Regions

**Files:**
- Modify: `src/ingest/vision.py`
- Test: `tests/test_vision_region_projection.py`

**Step 1: Write the failing test**

```python
def test_project_blocks_to_regions_excludes_figure_microblocks_from_text_layout() -> None:
    blocks = [...]
    coarse = {
        "text_regions": [{"region_id": "text_1", "bbox_px": [0, 0, 200, 300]}],
        "figure_regions": [{"region_id": "fig_1", "bbox_px": [220, 0, 800, 600]}],
        "caption_regions": [{"region_id": "cap_1", "bbox_px": [220, 610, 800, 700]}],
    }
    mapping = project_blocks_to_layout_regions(blocks, coarse)
    assert "axis_tick_1" not in mapping["text_block_ids"]
    assert "body_1" in mapping["text_block_ids"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_region_projection.py::test_project_blocks_to_regions_excludes_figure_microblocks_from_text_layout -q`

Expected: FAIL because projector does not exist.

**Step 3: Write minimal implementation**

- Add `project_blocks_to_layout_regions(...)`
- Use overlap ratio / center-point rules
- Keep raw blocks untouched; produce region-scoped subsets only
- Mark blocks with region assignment metadata for downstream consumers

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_region_projection.py::test_project_blocks_to_regions_excludes_figure_microblocks_from_text_layout -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_region_projection.py src/ingest/vision.py
git commit -m "feat: project raw blocks into coarse layout regions"
```

### Task 5: Add Micro-Block Coherence Classification

**Files:**
- Modify: `src/ingest/vision.py`
- Test: `tests/test_vision_microblock_coherence.py`

**Step 1: Write the failing test**

```python
def test_classify_microblock_cluster_marks_axis_legend_group_non_narrative(monkeypatch) -> None:
    cluster = {
        "region_id": "fig_1_cluster_0",
        "texts": ["0", "5", "10", "Hw", "Iw", "Ir", "Inflammation", "Log2 of FC", "Up", "Down"],
    }
    label = classify_microblock_cluster_coherence(cluster)
    assert label == "non_narrative_graphic"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_microblock_coherence.py::test_classify_microblock_cluster_marks_axis_legend_group_non_narrative -q`

Expected: FAIL because coherence classifier does not exist.

**Step 3: Write minimal implementation**

- Add `classify_microblock_cluster_coherence(...)`
- Start with cheap deterministic signals:
  - low sentence continuity
  - many short labels / numbers / abbreviations
  - low stopword ratio
- Keep the API shaped for a future small-model call:
  - return one of `narrative`, `non_narrative_graphic`, `tabular_like`, `uncertain`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_microblock_coherence.py::test_classify_microblock_cluster_marks_axis_legend_group_non_narrative -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_microblock_coherence.py src/ingest/vision.py
git commit -m "feat: add microblock coherence classifier"
```

### Task 6: Run Fine Layout Only On Text and Caption Regions

**Files:**
- Modify: `src/ingest/vision.py`
- Test: `tests/test_vision_hierarchical_layout_flow.py`

**Step 1: Write the failing test**

```python
def test_hierarchical_fine_layout_uses_only_text_caption_blocks(tmp_path: Path, monkeypatch) -> None:
    ...
    assert set(fine_input_block_ids) == {"body_1", "heading_1", "caption_1"}
    assert "axis_tick_1" not in fine_input_block_ids
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_hierarchical_layout_flow.py::test_hierarchical_fine_layout_uses_only_text_caption_blocks -q`

Expected: FAIL because the current fine pass still sees every block.

**Step 3: Write minimal implementation**

- Reuse the existing prompt/validator path for the fine pass
- Feed only region-projected text/caption blocks into that fine pass
- For blocks outside the fine pass, synthesize safe defaults:
  - figure/table region members should not become main-body layout participants
  - header/footer region members should default to `HeaderFooter`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_hierarchical_layout_flow.py::test_hierarchical_fine_layout_uses_only_text_caption_blocks -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_hierarchical_layout_flow.py src/ingest/vision.py
git commit -m "feat: scope fine layout to text and caption regions"
```

### Task 7: Preserve Heading Recovery on Overlong and Mixed Blocks

**Files:**
- Modify: `src/ingest/vision.py`
- Modify: `src/ingest/paragraphs.py`
- Test: `tests/test_vision_region_heading_fallback.py`
- Test: `tests/test_paragraphs_line_structure.py`

**Step 1: Write the failing test**

```python
def test_hierarchical_layout_keeps_region_heading_review_for_text_region_blocks(...) -> None:
    ...
    assert out["embedded_headings"] == [...]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_region_heading_fallback.py -k hierarchical -q`

Expected: FAIL because heading review is not region-aware yet.

**Step 3: Write minimal implementation**

- Ensure region heading review runs on fine-layout text blocks after region projection
- Preserve existing approved-heading hint path in `paragraphs`
- Keep the new pre-aggregation hint-based split path for overlong blocks and mixed merge groups

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_region_heading_fallback.py tests/test_paragraphs_line_structure.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_region_heading_fallback.py tests/test_paragraphs_line_structure.py src/ingest/vision.py src/ingest/paragraphs.py
git commit -m "feat: preserve heading recovery in hierarchical layout flow"
```

### Task 8: Feed Figure/Table Regions Into Figures-Tables Stage

**Files:**
- Modify: `src/ingest/figures_tables.py`
- Modify: `src/ingest/vision.py`
- Test: `tests/test_figures_tables_region_inputs.py`

**Step 1: Write the failing test**

```python
def test_figures_tables_prefers_coarse_vision_regions_on_dense_pages(tmp_path: Path) -> None:
    ...
    assert chosen_bbox == [expected values]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_figures_tables_region_inputs.py::test_figures_tables_prefers_coarse_vision_regions_on_dense_pages -q`

Expected: FAIL because figures/tables stage does not consume coarse layout regions.

**Step 3: Write minimal implementation**

- Store coarse region outputs in page artifacts
- Let `figures_tables` prefer those regions as hints for crop search on dense pages
- Keep old path as fallback

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_figures_tables_region_inputs.py::test_figures_tables_prefers_coarse_vision_regions_on_dense_pages -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_figures_tables_region_inputs.py src/ingest/figures_tables.py src/ingest/vision.py
git commit -m "feat: reuse coarse layout regions for figure-table extraction"
```

### Task 9: Add Retry Policy By Page Class

**Files:**
- Modify: `src/ingest/vision.py`
- Test: `tests/test_vision_fallback_retries.py`

**Step 1: Write the failing test**

```python
def test_dense_page_hierarchical_mode_uses_reduced_full_page_retries(tmp_path: Path, monkeypatch) -> None:
    ...
    assert full_page_attempts == 1
    assert coarse_attempts == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_fallback_retries.py::test_dense_page_hierarchical_mode_uses_reduced_full_page_retries -q`

Expected: FAIL because retry policy is page-agnostic today.

**Step 3: Write minimal implementation**

- Add separate retry knobs for:
  - direct full-page layout
  - coarse layout pass
  - fine region layout pass
- Dense/hierarchical pages should spend fewer retries on the expensive full-page path

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_fallback_retries.py::test_dense_page_hierarchical_mode_uses_reduced_full_page_retries -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_fallback_retries.py src/ingest/vision.py
git commit -m "feat: tune retry policy by page layout mode"
```

### Task 10: Verify Real Documents and Telemetry

**Files:**
- Modify: `src/ingest/vision.py`
- Modify: `src/ingest/paragraphs.py`
- Modify: `src/ingest/figures_tables.py`
- Verify: `run/verify_tang2024_linesplit_20260308`
- Verify: `run/verify_yao2025_*` or equivalent real dense-figure sample

**Step 1: Add telemetry fields**

- Record per page:
  - `layout_mode`
  - `coarse_regions_count`
  - `text_layout_block_count`
  - `figure_region_block_count`
  - `timeout_stage`

**Step 2: Run focused regressions**

Run:

```bash
pytest tests/test_vision_region_heading_fallback.py tests/test_vision_fallback_retries.py tests/test_paragraphs_line_structure.py tests/test_figures_tables_region_inputs.py -q
```

Expected: PASS

**Step 3: Run real-document checks**

Run:

```bash
python -m ingest.cli --doc_id verify_tang2024_linesplit_20260308 --stage vision
python -m ingest.cli --doc_id verify_tang2024_linesplit_20260308 --stage paragraphs
python -m ingest.cli --doc_id verify_tang2024_linesplit_20260308 --stage figures_tables
```

Check:
- Dense pages no longer spend 6 full-page vision calls before fallback
- `clean_document.md` still keeps true headings such as `Results`, `RNA sequence`
- dense figure pages do not reintroduce chart-text leaks into Main Body

**Step 4: Commit**

```bash
git add src/ingest/vision.py src/ingest/paragraphs.py src/ingest/figures_tables.py
git commit -m "feat: verify hierarchical layout on dense real documents"
```

## Design Notes

- Do **not** delete raw OCR blocks. They remain the contract-bound source for downstream evidence.
- The coarse pass should be low-semantic: region segmentation only, not final heading/body semantics.
- The coherence classifier is a body-vs-non-body aid only; it must not become the final figure/table/heading semantic authority.
- The fine pass remains where heading/caption/body role recovery happens.
- `paragraphs` should consume fine-layout outputs plus approved heading hints, never coarse-region guesses directly.
- `figures_tables` should use coarse `figure_regions/table_regions` as geometric hints, not as final truth.

## Success Criteria

- Dense pages like Tang `page 4/6/8` stop timing out repeatedly on the same giant full-page prompt.
- True section headings remain recoverable by vision on pages like Tang `page 2/10`.
- Figure/table micro text no longer dominates `reading_order` and `merge_groups`.
- The coherence classifier reliably removes obvious axis/legend micro-clusters from main-body layout inputs without suppressing short true headings.
- `clean_document` improves without losing coverage of real body text.
