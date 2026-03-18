# Layout Profile And Mixed Paragraphs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add document/journal layout profiling, stronger block regrouping, dense-figure pruning, and mixed paragraph splitting so difficult PDFs like Tang2024 stop leaking figure text and mixed caption/body fragments into `clean_document`.

**Architecture:** Introduce a document-level layout profile built from extractor output and early page evidence, then use that profile to improve block-level decisions before `vision` and `paragraphs` consume them. The implementation should stay weakly semantic at the layout stage: infer header/footer bands, body/caption/heading typography priors, and same-paragraph regrouping candidates without pre-deciding final roles. Then tighten the dense-page path using the profile, and finally add mixed-paragraph splitting plus region-level fallback for ambiguous groups.

**Tech Stack:** Python 3.11, `src/ingest/extractor.py`, `src/ingest/layout_analyzer.py`, `src/ingest/vision.py`, `src/ingest/paragraphs.py`, JSON/JSONL artifacts, pytest.

### Task 1: Add Document Layout Profile Contract

**Files:**
- Modify: `src/ingest/layout_analyzer.py`
- Modify: `src/ingest/extractor.py`
- Test: `tests/test_layout_profile.py`

**Step 1: Write the failing test**

```python
def test_build_document_layout_profile_detects_stable_header_footer_bands() -> None:
    profile = build_document_layout_profile(
        pages_blocks={
            2: [{"block_id": "p2_h", "page": 2, "bbox_pt": [30, 10, 560, 22], "text": "Article https://doi.org/..."}],
            3: [{"block_id": "p3_h", "page": 3, "bbox_pt": [30, 11, 560, 23], "text": "Article https://doi.org/..."}],
            4: [{"block_id": "p4_f", "page": 4, "bbox_pt": [30, 770, 560, 786], "text": "Nature Communications | (2024) 15:9460"}],
        },
        pages_dimensions={2: (595, 791), 3: (595, 791), 4: (595, 791)},
    )
    assert profile["header_band_pt"][1] < 40
    assert profile["footer_band_pt"][1] > 740
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout_profile.py::test_build_document_layout_profile_detects_stable_header_footer_bands -q`

Expected: FAIL because `build_document_layout_profile` does not exist.

**Step 3: Write minimal implementation**

- Add `build_document_layout_profile(...)` in `src/ingest/layout_analyzer.py`
- Include:
  - `header_band_pt`
  - `footer_band_pt`
  - `column_count_mode`
  - `body_font_profile`
  - `heading_font_profile`
  - `caption_font_profile`
- Save artifact from extractor to `text/document_layout_profile.json`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_layout_profile.py::test_build_document_layout_profile_detects_stable_header_footer_bands -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_layout_profile.py src/ingest/layout_analyzer.py src/ingest/extractor.py
git commit -m "feat: add document layout profile"
```

### Task 2: Add Same-Paragraph Block Regrouping

**Files:**
- Modify: `src/ingest/layout_analyzer.py`
- Modify: `src/ingest/extractor.py`
- Test: `tests/test_layout_regrouping.py`

**Step 1: Write the failing test**

```python
def test_find_same_paragraph_block_groups_merges_vertically_adjacent_same_column_blocks() -> None:
    groups = find_same_paragraph_block_groups(
        blocks=[
            {"block_id": "b1", "page": 1, "bbox_pt": [40, 100, 295, 180], "text": "Paragraph part one", "font_stats": {"avg_size": 8.2}},
            {"block_id": "b2", "page": 1, "bbox_pt": [40, 183, 295, 220], "text": "Paragraph part two", "font_stats": {"avg_size": 8.2}},
            {"block_id": "b3", "page": 1, "bbox_pt": [40, 260, 295, 320], "text": "New paragraph", "font_stats": {"avg_size": 8.2}},
        ],
        page_profile={"body_line_gap_pt": 6.0, "column_regions": [[0, 0, 297, 791], [297, 0, 595, 791]]},
    )
    assert groups == [["b1", "b2"]]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout_regrouping.py::test_find_same_paragraph_block_groups_merges_vertically_adjacent_same_column_blocks -q`

Expected: FAIL because regrouping helper does not exist.

**Step 3: Write minimal implementation**

- Add a weak-semantic regrouping helper that considers:
  - same page
  - same column
  - left-edge alignment tolerance
  - horizontal overlap
  - small vertical gap relative to learned line gap
  - similar font size
  - not crossing header/footer or figure/caption bands
- Save regrouping hints into `text/layout_analysis.json`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_layout_regrouping.py::test_find_same_paragraph_block_groups_merges_vertically_adjacent_same_column_blocks -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_layout_regrouping.py src/ingest/layout_analyzer.py src/ingest/extractor.py
git commit -m "feat: add same-paragraph block regrouping"
```

### Task 3: Use Layout Profile In Vision Dense-Page Filtering

**Files:**
- Modify: `src/ingest/vision.py`
- Test: `tests/test_vision_layout_profile_dense_filtering.py`

**Step 1: Write the failing test**

```python
def test_select_hierarchical_fine_layout_block_ids_ignores_header_footer_band_and_micro_figure_text() -> None:
    ...
    assert "header_block" not in selected
    assert "axis_tick" not in selected
    assert "body_para" in selected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_layout_profile_dense_filtering.py::test_select_hierarchical_fine_layout_block_ids_ignores_header_footer_band_and_micro_figure_text -q`

Expected: FAIL because layout profile is not consulted.

**Step 3: Write minimal implementation**

- Load `document_layout_profile.json` into dense-page path
- Filter candidate blocks using:
  - stable header/footer bands
  - caption typography prior
  - body typography prior
  - regrouping hints where useful

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_layout_profile_dense_filtering.py::test_select_hierarchical_fine_layout_block_ids_ignores_header_footer_band_and_micro_figure_text -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_layout_profile_dense_filtering.py src/ingest/vision.py
git commit -m "feat: use layout profile in dense vision filtering"
```

### Task 4: Add Mixed Paragraph Split Before Render

**Files:**
- Modify: `src/ingest/paragraphs.py`
- Test: `tests/test_paragraphs_mixed_split.py`

**Step 1: Write the failing test**

```python
def test_split_mixed_body_and_caption_tail_keeps_caption_explanation_out_of_main_body() -> None:
    ...
    assert "Each point represents a gene." not in main_body
    assert "Each point represents a gene." in figures_tables
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_paragraphs_mixed_split.py::test_split_mixed_body_and_caption_tail_keeps_caption_explanation_out_of_main_body -q`

Expected: FAIL because mixed paragraph split does not exist.

**Step 3: Write minimal implementation**

- Add a preprocessing split step before `render_clean_document`
- Use:
  - source lines
  - region hints
  - caption continuation markers
  - heading boundaries
- Keep it limited to low-confidence or multi-region paragraphs

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_paragraphs_mixed_split.py::test_split_mixed_body_and_caption_tail_keeps_caption_explanation_out_of_main_body -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_paragraphs_mixed_split.py src/ingest/paragraphs.py
git commit -m "feat: split mixed caption and body paragraphs"
```

### Task 5: Add Region-Level Vision Review For Ambiguous Mixed Groups

**Files:**
- Modify: `src/ingest/vision.py`
- Modify: `src/ingest/paragraphs.py`
- Test: `tests/test_vision_mixed_group_review.py`

**Step 1: Write the failing test**

```python
def test_region_review_marks_mixed_group_caption_tail_as_caption() -> None:
    ...
    assert reviewed["decision"] == "caption_tail"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_mixed_group_review.py::test_region_review_marks_mixed_group_caption_tail_as_caption -q`

Expected: FAIL because mixed-group region review does not exist.

**Step 3: Write minimal implementation**

- Trigger only for ambiguous mixed groups after profile-guided filtering
- Crop local region
- Ask region vision to choose among:
  - `body`
  - `heading`
  - `caption_tail`
  - `graphic_text`
- Feed result back into `paragraphs`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_mixed_group_review.py::test_region_review_marks_mixed_group_caption_tail_as_caption -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vision_mixed_group_review.py src/ingest/vision.py src/ingest/paragraphs.py
git commit -m "feat: add region review for mixed layout groups"
```

### Task 6: Add Journal Layout Profile Cache

**Files:**
- Modify: `src/ingest/layout_analyzer.py`
- Modify: `src/ingest/extractor.py`
- Test: `tests/test_journal_layout_profiles.py`

**Step 1: Write the failing test**

```python
def test_match_journal_layout_profile_prefers_nature_communications_defaults() -> None:
    profile = match_journal_layout_profile(
        metadata_title="Nature Communications",
        doi="https://doi.org/10.1038/s41467-024-53826-w",
    )
    assert profile["journal_key"] == "nature_communications"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_journal_layout_profiles.py::test_match_journal_layout_profile_prefers_nature_communications_defaults -q`

Expected: FAIL because journal profile cache does not exist.

**Step 3: Write minimal implementation**

- Add a small in-repo profile registry for known journals
- Use it as soft prior only
- Merge with document profile, never override strong document evidence

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_journal_layout_profiles.py::test_match_journal_layout_profile_prefers_nature_communications_defaults -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_journal_layout_profiles.py src/ingest/layout_analyzer.py src/ingest/extractor.py
git commit -m "feat: add journal layout profile cache"
```

### Task 7: Tang And Regression Verification

**Files:**
- Modify: `tests/test_tang_clean_document_regressions.py`
- Test: `tests/test_paragraphs_cleaning.py`
- Test: `tests/test_figures_tables_vision_regions.py`

**Step 1: Write the failing test**

- Add one Tang regression per fixed failure class:
  - dense-figure micro-text before Methods
  - mixed caption/body paragraph
  - methods tail contamination
  - same-paragraph split false negative

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_tang_clean_document_regressions.py -q
```

Expected: targeted failures reproducing current Tang issues.

**Step 3: Write minimal implementation**

- Only after the relevant upstream tasks are in place
- Re-run Tang pipeline and confirm the targeted leaks are gone

**Step 4: Run verification**

Run:

```bash
pytest tests/test_paragraphs_cleaning.py tests/test_tang_clean_document_regressions.py tests/test_figures_tables_vision_regions.py -q
python -m ingest.cli --doc_id verify_tang2024_20260306 --stage paragraphs
```

Expected:
- tests PASS
- Tang `clean_document.md` cleaner in Methods and figure-heavy sections

**Step 5: Commit**

```bash
git add tests/test_tang_clean_document_regressions.py tests/test_paragraphs_cleaning.py tests/test_figures_tables_vision_regions.py
git commit -m "test: cover Tang layout profile regressions"
```
