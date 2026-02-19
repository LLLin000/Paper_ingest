# PyMuPDF Dependency Research - T0 Bootstrap

**Date**: 2026-02-18
**Scope**: Dependency baseline for page rendering and text block extraction setup (bootstrap only)

---

## 1. Official Sources

### Installation Documentation
- **PyPI**: https://pypi.org/project/PyMuPDF/
- **Official Docs**: https://pymupdf.readthedocs.io/
- **GitHub**: https://github.com/pymupdf/pymupdf

### Current Python Version Requirements
| Version | Python Requirement |
|---------|---------------------|
| 1.27.x (latest) | >=3.10 |
| 1.26.x | >=3.9 |
| 1.24.x | >=3.8 |

**Recommendation**: For Python 3.11+, use `PyMuPDF>=1.26.0` to ensure full compatibility.

---

## 2. Dependency Baseline

### Core Dependency
```toml
# pyproject.toml
dependencies = [
    "pymupdf>=1.26.0",
]
```

### No Mandatory External Dependencies
PyMuPDF has **no mandatory external dependencies**. However, optional features require:
- `pillow` - for image processing
- `pymupdf-fonts` - for additional font support
- `fonttools` - for font manipulation in tests

---

## 3. Basic API Entry Points (Reference Only)

### Document Operations
```python
import pymupdf
doc = pymupdf.open("document.pdf")  # Open PDF
page = doc.load_page(page_number)    # Load page (0-indexed)
doc.close()                          # Close document
```

### Text Extraction Methods
```python
text = page.get_text()               # Plain text (UTF-8)
text_dict = page.get_text("dict")   # Text with block structure
text_html = page.get_text("html")   # HTML formatted
```

### Page Rendering (for thumbnail/preview)
```python
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Render at 2x scale
pix.save("output.png")
```

---

## 4. Windows Path/Unicode CLI Risks

### Known Issues

1. **MSVCP140.dll Dependency** (Windows)
   - Error: `ImportError: DLL load failed while importing _extra`
   - Cause: Missing Visual C++ Redistributables
   - Fix: Install [Microsoft Visual C++ Redistributables](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)

2. **Unicode Filenames in CLI Arguments**
   - Python 3.x handles `sys.argv` as proper Unicode on Windows
   - However, shell encoding issues may arise in older Windows terminal environments
   - **Risk**: Filenames with characters outside system locale may fail in CLI context
   - **Mitigation**: Use `pathlib.Path` for file operations; avoid raw CLI argument passing for Unicode paths

3. **Path Handling Best Practices**
   - Always use `pathlib.Path` for cross-platform path handling
   - Convert to string with `.resolve()` before passing to PyMuPDF
   - Avoid shell expansion - pass full paths explicitly

---

## 5. Lockfile/Constraints Recommendations

### For Deterministic Environment Setup

#### Option A: poetry.lock (if using Poetry)
```toml
[tool.poetry.dependencies]
pymupdf = "^1.26"
```

#### Option B: pip constraints file
```
# constraints.txt
pymupdf>=1.26.0,<1.28.0
```

#### Option C: pip-tools requirements.in
```
# requirements.in
pymupdf>=1.26.0

# requirements.txt (generated)
pymupdf==1.26.7
```

### Version Pinning Strategy
- **Minimum**: 1.26.0 (supports Python 3.9+, includes TEXT_CLIP feature)
- **Maximum**: <1.28.0 (allow patch updates, avoid breaking changes)
- **Exact for CI**: Pin to specific version (e.g., `pymupdf==1.26.7`)

---

## 6. References

1. PyMuPDF Official Installation: https://pymupdf.readthedocs.io/en/latest/installation.html
2. PyPI Package Details: https://pypi.org/project/PyMuPDF/
3. Python 3.x Windows CLI Unicode: https://docs.python.org/3/using/windows.html
4. PyMuPDF GitHub: https://github.com/pymupdf/pymupdf

---
*End of research notes*

---

## T1 Fix: Pydantic `model_config` Name Collision with basedpyright

**Date**: 2026-02-18

### Issue
basedpyright reported `reportIncompatibleVariableOverride` errors when a Pydantic model class had:
1. A field with `alias="model_config"` (to emit JSON key `model_config`)
2. A nested model class named `ModelConfig`

The error claimed `ModelConfig` was overriding `BaseModel.model_config` (a `ConfigDict`).

### Root Cause
- Pydantic v2 `BaseModel` has class attribute `model_config: ConfigDict`
- basedpyright saw class name `ModelConfig` as potentially conflicting with `model_config` attribute
- LSP also reported error at wrong line (133 instead of actual location)

### Fix Applied
1. Renamed `ModelConfig` class to `LLMSettings` to avoid name similarity
2. Used `serialization_alias="model_config"` + `validation_alias=AliasChoices(...)` instead of `alias=`
3. Added `# type: ignore[misc]` on class definition (belt-and-suspenders)

### Verification
- Acceptance command passes
- Manifest JSON output still has `model_config` key (contract preserved)
- Python syntax check passes
- Module imports correctly

### Note
LSP diagnostics showed stale errors (referenced old class name `ModelConfig`) - likely due to basedpyright cache. Runtime behavior is correct.

---

## T1 Fix #2: Remove unnecessary ConfigDict override

**Date**: 2026-02-18

### Final Fix
Removed `model_config = ConfigDict()` from `Manifest` class entirely since no special config was needed. Pydantic v2 uses sensible defaults.

### Changes
- Removed `ConfigDict` import
- Removed `model_config = ConfigDict()` line from `Manifest` class
- Removed `# type: ignore[misc]` comment

### Verification
```bash
$ basedpyright src/ingest/manifest.py
0 errors, 6 warnings, 0 notes
```

All warnings are minor (unused imports, deprecated Optional syntax) - no errors. Acceptance test passes.

---

## T2 Extractor: PyMuPDF Type Stubs Missing

**Date**: 2026-02-18

### Issue

PyMuPDF does not ship with type stubs. basedpyright reports `reportMissingTypeStubs` warning and numerous `reportUnknownMemberType` / `reportUnknownVariableType` warnings for all PyMuPDF API calls.

### Impact

- LSP shows warnings but no errors
- Runtime behavior is correct
- Code completion/intellisense limited for PyMuPDF APIs

### Mitigation

- Added runtime type checking (`isinstance`) for all PyMuPDF return values
- Used `Any` type annotation where dynamic typing is unavoidable
- All extractor tests pass despite warnings

### Potential Fix

Could add `py.typed` or type stubs to project, but this is low priority since:
- PyMuPDF is a vendored dependency
- Type safety at boundaries is more important than internal typing
- Runtime type checking provides adequate safety

---

## T2 Extractor: Page Rendering Memory Usage

**Date**: 2026-02-18

### Observation

Rendering 20 pages at 150 DPI with 2.0 scale creates ~1.2MB PNG files per page. Total pages directory size for test document: ~21MB.

### Not Blocking

Memory usage is acceptable for typical documents (<100 pages). For very large documents, could consider:
- Streaming page processing (render/write one at a time)
- Configurable DPI/scale via manifest
- Lazy rendering on demand

No action needed for v1.

---

## T2 Extractor: Fault Injection Event Volume

**Date**: 2026-02-18

### Observation

With `--inject-missing-font-stats`, one fault event is logged per affected block (2406 events for test document).

### Acceptable Behavior

Contract requires fallback events to be logged (lines 410-416). Per-block granularity provides full traceability. If file size becomes a concern, could:
- Add summary-level event with count
- Clear file between test runs
- Add max_events config

No change needed for v1.

---

## T3 Overlay: PIL Type Stubs

**Date**: 2026-02-18

### Issue

Pillow does not ship with complete type stubs. basedpyright reports `reportUnknownMemberType` and `reportAny` warnings for PIL API calls.

### Impact

- LSP shows warnings but no errors
- Runtime behavior is correct
- Code completion limited for PIL APIs

### Mitigation

- Used explicit type annotations where possible
- Accepted `Any` type for JSON-parsed data
- All overlay tests pass despite warnings

### Not Blocking

This is a known limitation of Pillow's type support. No action needed for v1.

---

## T3 Overlay: Font Availability

**Date**: 2026-02-18

### Observation

Font availability varies across platforms. The overlay module tries multiple font paths with fallback to default.

### Potential Issue

On systems without Arial, DejaVuSans, or Helvetica, the default font may be small or pixelated.

### Mitigation

- Fallback chain covers Windows, Linux, macOS
- Default font is functional for block IDs
- Could add bundled font in future if needed

No change needed for v1.

---

## T4 Vision: SiliconFlow API Timeout Handling

**Date**: 2026-02-18

### Observation

The SiliconFlow API call uses a 60-second timeout. For documents with many pages, this could potentially timeout on slower connections.

### Mitigation

- 60 seconds is generous for text-only LLM calls
- Fallback ensures processing continues even on timeout
- Could make timeout configurable via environment variable in future

### Not Blocking

API timeout triggers fallback path which is contract-compliant behavior.

---

## T4 Vision: No API Key Behavior

**Date**: 2026-02-18

### Observation

When `SILICONFLOW_API_KEY` is not set, the vision stage returns an error JSON response which triggers fallback.

### Design Decision

This is intentional - the fallback path is used when:
1. API key not configured
2. API call fails
3. JSON parse fails (even after retry)

### Acceptable

Fallback produces deterministic, contract-compliant output. Users without API access still get functional vision stage.

---

## Progress Note (2026-02-18)

- Task 4 (Vision Structure Corrector) completed. No blocking issues identified. All fault injection paths verified working.

---

## T4 Vision: Optional/prompt Type Checker Fix

**Date**: 2026-02-18

### Issue

basedpyright reported:
1. `Optional` not imported (used on multiple lines)
2. `prompt` possibly unbound in retry path

### Fix Applied

1. Added `Optional` to typing imports
2. Moved `prompt = build_prompt(input_pkg)` before the `inject_malformed` conditional to ensure it's always defined

### Verification

Both vision commands execute successfully with 0 LSP errors (warnings only).

---

## T5 Paragraph: Type Checker Fix (2026-02-18)

### Issue

Basedpyright LSP diagnostics showed stale/incorrect type errors after code edits:
- `load_vision_outputs` return type mismatch (cache issue)
- singleton paragraph `page_span` and `text.strip()` type errors
- `sort_key` tuple return type
- `confidence_by_page` argument type

### Fix Applied

Ran `basedpyright src/ingest/paragraphs.py` directly which showed **0 errors**:
- LSP diagnostics were stale (did not reflect code changes)
- CLI tool shows correct type status
- All type annotations are correct

### Verification

```
$ basedpyright src/ingest/paragraphs.py
0 errors, 51 warnings, 0 notes
```

Runtime behavior unchanged - acceptance test passes:
```
$ ingest_pdf --doc_id test01 --stage paragraphs
Processed 2406 blocks
Created 1131 paragraphs
```

### Note

LSP cache invalidation may be needed in VS Code after code changes. Use CLI tool `basedpyright` for accurate diagnostics.

---

## T5 Paragraph: Type Fix Summary (2026-02-18)

### Fix Applied

Fixed type errors in `src/ingest/paragraphs.py`:
- Changed `load_vision_outputs` return type to use `dict[int, Any]` for confidence_by_page
- Changed `aggregate_paragraphs` parameter type to accept `dict[int, Any]`
- Fixed indentation error in `build_singleton_paragraph`
- Used explicit type checking for text.strip() with isinstance()

### Verification

```
$ basedpyright src/ingest/paragraphs.py
0 errors, 59 warnings, 0 notes

$ ingest_pdf --doc_id test01 --stage paragraphs
Processed 2406 blocks
Created 1131 paragraphs
```

### Note

LSP diagnostics in this environment may show stale errors. Use CLI tool `basedpyright` for accurate results.

---

## T5 Paragraph: Merge Group Quality

**Date**: 2026-02-18

### Observation

Vision merge groups may include unrelated blocks (e.g., "Primer" + "Rotator cuff tears" in test01 page 1).

### Impact

- Some paragraph text may be incorrectly merged
- Downstream stages (citations, reading) may receive noisy input
- Confidence score reflects vision stage quality

### Mitigation

- Provenance metadata tracks source merge_group
- Low confidence marked in notes
- Future: enhance vision merge detection or add paragraph splitting

### Not Blocking

Contract requires aggregation only - quality depends on vision stage. Explicit uncertainty metadata signals potential issues.

---

## T5 Paragraph: Section Path Inference

**Date**: 2026-02-18

### Observation

Section path is only extracted from Heading role blocks. Many documents use implicit headings (larger font, bold) without explicit Heading role.

### Impact

- section_path may be null for documents without explicit headings
- Reading pipeline may lack section context

### Mitigation

- Null section_path is contract-compliant (line 135)
- Uncertainty metadata (`no_section_path`) signals missing data
- Could enhance with font-based heading detection in future

### Not Blocking

Explicit null with uncertainty metadata satisfies contract.

---

## T6 Citations: PyMuPDF Link Kind Constants (2026-02-18)

### Issue

PyMuPDF constants like `pymupdf.LINK_INTERNAL`, `pymupdf.LINK_URI` are not available in all versions.

### Error

```
AttributeError: module 'pymupdf' has no attribute 'LINK_INTERNAL'
```

### Fix Applied

Use numeric values instead:
- kind == 3: internal (goto)
- kind in (1, 2): external (URI/URL)
- kind == 4: external (Gotob)

### Verification

```bash
$ ingest_pdf --doc_id test01 --stage citations
Extracted 360 citation anchors
Created 360 citation mappings
```

---

## T6 Citations: Link Bounding Box Extraction (2026-02-18)

### Issue

Initial implementation used `link.get("bbox", ...)` which returned [0,0,0,0] for all links.

### Root Cause

PyMuPDF uses `'from'` key for link rect, not `'bbox'`.

### Fix Applied

```python
rect = link.get("from")
if rect:
    bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
```

### Verification

cite_anchors.jsonl now contains proper bbox values:
```
"anchor_bbox": [391.10, 39.86, 561.26, 50.86]
```

---

## T6 Citations: DOI Extraction with URL Parameters (2026-02-18)

### Issue

Crossmark URLs embed DOI in query string: `http://crossmark.crossref.org/dialog/?doi=10.1038/...&domain=pdf`

Extracted DOI included URL parameters.

### Fix Applied

Strip both `?param=value` and `&param=value` patterns:
```python
if '&' in doi:
    doi = doi.split('&')[0]
if '?' in doi:
    doi = doi.split('?')[0]
```

### Verification

cite_map.jsonl now shows clean DOI:
```
"mapped_ref_key": "doi:10.1038/s41572-024-00492-3"
```

---

## T6 Citations: High Unmapped Rate (2026-02-18)

### Observation

Out of 360 citation anchors:
- 10 mapped via regex (DOI extraction)
- 350 marked as "none"

### Reason

Most links in test document are:
- ORCID URLs (external)
- mailto links
- Crossref/Crossmark URLs without DOI

These don't match reference paragraphs in the document.

### Not Blocking

Contract explicitly allows partial mapping:
- Unmapped entries retained with explicit `reason` field
- Strategy "none" signals unsuccessful mapping
- No hard-stop on partial mapping (contract line 389)

---

## T6 Citations: Pydantic Manifest Handling (2026-02-18)

### Issue

`manifest.get("input_pdf_path")` failed because `manifest` is a Pydantic model, not a dict.

### Error

```
AttributeError: 'Manifest' object has no attribute 'get'
```

### Fix Applied

Convert Pydantic model to dict:
```python
manifest_data = manifest.model_dump()
pdf_path = Path(manifest_data.get("input_pdf_path", ""))
```

### Verification

Citation stage runs successfully with manifest object passed from CLI.

---

## T6 Citations: Type Checker Fix (2026-02-19)

### Issue

basedpyright reported 3 errors in `fuzzy_match_reference`:
1. Index 2 out of range for tuple (line 182)
2. Type mismatch in return statement (line 183)

### Fix Applied

Changed:
```python
best = max(candidates, key=lambda x: x[2])
return (best[0], best[2])
```
To:
```python
best = max(candidates, key=lambda x: x[1])
return best
```

Also added explicit type annotation: `candidates: list[tuple[str, float]] = []`

### Verification

```
$ basedpyright src/ingest/citations.py
0 errors, 77 warnings, 0 notes

$ ingest_pdf --doc_id test01 --stage citations
Extracted 360 citation anchors
Created 360 citation mappings
```

---

## T6.5 Figure/Table Extraction Issues

**Date**: 2026-02-19

### Issue: Type Mismatch in bbox Conversion

**Symptom**: 
basedpyright error: `Argument of type "list[int] | Any" cannot be assigned to parameter "bbox_pt" of type "list[float]"`

**Root Cause**:
PyMuPDF returns bbox coordinates as integers, but `pt_to_px_bbox` function signature expected `list[float]`.

**Fix Applied**:
Added explicit float conversion:
```python
bbox_px = pt_to_px_bbox([float(c) for c in bbox_pt], dpi, scale)
```

### Issue: Caption-Only Entries Without Images

**Symptom**:
Some caption paragraphs don't have corresponding embedded images in the PDF.

**Resolution**:
- Created fallback to estimate figure/table location based on caption position
- Uses heuristic: figures appear above their captions
- Generates placeholder crops with lower confidence (0.4 vs 0.6)

### Issue: Unused Imports Warning

**Symptom**:
basedpyright warning: `Import "hashlib" is not accessed`

**Resolution**:
- hashlib was imported but not used (planned for future deterministic IDs)
- Currently using simple sequential counters
- Warning accepted as minor issue

### Issue: Unused Variables Warning

**Symptom**:
basedpyright warnings for `fig_num`, `fig_remainder`, `tbl_num`, `tbl_remainder` variables.

**Resolution**:
- Variables captured from regex match but not used in baseline
- Accepted as they may be useful for enhanced detection in future

### Issue: Type Narrowing Limitation

**Symptom**:
basedpyright error persists at line 471 despite explicit float conversion in list comprehension.

**Root Cause**:
`bbox_pt` comes from `dict.get()` with type `list[int] | Any`. Basedpyright doesn't narrow types through list comprehensions for union types - this is a known type-checker limitation.

**Resolution**:
- Explicit `[float(c) for c in bbox_pt]` is correct at runtime
- Code works as verified by acceptance tests
- Error is false positive from type checker
- No runtime impact confirmed

### Issue: Type Invariance with list[T]

**Symptom**:
`list[int]` not assignable to `list[float | int]` due to list invariance.

**Root Cause**:
Python's `list` is invariant - `list[int]` is not a subtype of `list[float | int]` even though `int` is a subtype of `float | int`.

**Fix Applied**:
Changed `pt_to_px_bbox` signature from `list[float]` to `Sequence[float | int]`:
```python
def pt_to_px_bbox(bbox_pt: Sequence[float | int], dpi: int, scale: float) -> list[int]:
```
`Sequence` is covariant, so it accepts `list[int]`.

**Verification**:
```
$ python -m basedpyright src/ingest/figures_tables.py
0 errors, 121 warnings, 0 notes
```
Command runs successfully with expected output.

---

## T7 Reading Engine: Type Annotation Fixes (2026-02-19)

### Issue: Bare `dict` Annotations

**Symptom**:
basedpyright errors: `Expected type arguments for generic class "dict"` on multiple functions.

**Fix Applied**:
Replaced bare `dict` with `dict[str, Any]`:
- `parse_logic_graph()` return type
- `parse_themes()` return type  
- `parse_synthesis()` return type
- `build_profile_prompt()` parameter types
- `build_logic_prompt()` parameter types
- `build_facts_prompt()` parameter types
- `build_themes_prompt()` parameter types
- `build_synthesis_prompt()` parameter types
- `generate_fallback_logic_graph()` return type
- `generate_fallback_facts()` parameter type
- `generate_fallback_themes()` return type
- `generate_fallback_synthesis()` parameter/return types

### Issue: Record Dict Type in Facts Write Loop

**Symptom**:
basedpyright errors on lines adding `quote_truncated` (bool) and `truncation_reason` (nullable str) to record dict.

**Fix Applied**:
Added explicit type annotation:
```python
record: dict[str, Any] = { ... }
```

### Verification

```
$ basedpyright src/ingest/reading.py
0 errors, 175 warnings, 0 notes
```

Acceptance test passes:
```
$ ingest_pdf --doc_id test01 --stage reading
Extracted 1131 facts, 0 themes, 3 figure slots, 4 fallbacks
```

Fault injection test passes:
```
$ ingest_pdf --doc_id test01 --stage reading --inject-reading-malformed-json
Fault injection: malformed-json enabled
```

### Issue: F-string Format Specifier Error

**Symptom**:
```
ValueError: Invalid format specifier
```
when building JSON templates in reading stage.

**Root Cause**:
F-string interprets `{` as format specifier. JSON templates contain curly braces which conflict.

**Fix Applied**:
Use string concatenation instead of f-strings:
```python
# BAD:
template = f'{{"page": {page}, "reading_order": []}}'

# GOOD:
template = '{"page": ' + str(page) + ', "reading_order": []}'
```

### Issue: None Caption Handling

**Symptom**:
Figure/table assets may have null `caption_text` field.

**Error**:
```
KeyError: 'caption_text'
```

**Fix Applied**:
Use `or` fallback for nullable fields:
```python
caption = asset.get("caption_text") or "No caption"
```

### Issue: Import Organization

**Symptom**:
basedpyright warnings for unused imports in reading.py.

**Resolution**:
- Organized imports by standard library, third-party, local
- Removed unused `time` import
- Warnings are acceptable (59 warnings, 0 errors)

### Issue: Themes Stage Fallback Quality

**Observation**:
When LLM fails, themes stage produces empty themes array (fallback). This results in:
- 0 themes extracted
- No cross-theme links
- Empty synthesis thematic content

**Not Blocking**:
- Contract allows fallback output
- Deterministic behavior maintained
- API failures trigger fallback gracefully
- Future: enhance fallback with rule-based theme extraction

### Issue: Facts Batching Memory

**Observation**:
Processing 1131 paragraphs in batches of 30 creates ~38 API calls per document.

**Not Blocking**:
- Each call is lightweight (30 paragraphs text)
- Timeout (60s) handles slow responses
- Fallback extracts all facts deterministically if API fails
- Future: consider async parallelization if performance needed

---

## T8 Obsidian Renderer Issues (2026-02-19)

### Issue: Unused Import Warning

**Symptom**:
basedpyright warning: `Import "load_manifest" is not accessed`

**Resolution**:
- Removed unused import from render.py

### Issue: Type Annotation Bulk

**Observation**:
The render.py module processes dynamic JSON data from multiple sources. This results in many `Any` type warnings from basedpyright.

**Not Blocking**:
- All warnings are informational (reportAny, reportUnknownMemberType)
- No type errors
- Runtime behavior correct as verified by acceptance test
- Dynamic JSON parsing inherently lacks static type information

### Issue: Missing Figure/Table Images

**Observation**:
When render_mode is `full_asset_embed`, the renderer checks if image_path exists. For test01, images exist but may not render properly in all Obsidian configurations.

**Not Blocking**:
- Falls back gracefully to content_only display
- Embed path uses relative path `../figures_tables/assets/` for portability
- User can switch to content_only mode if images fail to render

---

## T8 Obsidian Renderer: Type Annotation Fix (2026-02-19)

### Issue: Missing Type Arguments

**Symptom**:
basedpyright reported `reportMissingTypeArgument` errors for bare generics in function signatures.

**Fix Applied**:
1. Removed unused `load_manifest` import
2. Changed `_load_jsonl` return type from `list[dict]` to `list[dict[str, Any]]`
3. Changed `_determine_quality` parameter types:
   - `paper_profile: dict` -> `paper_profile: dict[str, Any]`
   - `synthesis: dict` -> `synthesis: dict[str, Any]`
   - `facts: list` -> `facts: list[dict[str, Any]]`
   - `cite_map: list` -> `cite_map: list[dict[str, Any]]`

### Verification

```
$ basedpyright src/ingest/render.py 2>&1 | grep -E "error"
(no errors - warnings only)

$ ingest_pdf --doc_id test01 --stage render
Created 7 sections
Output: run\test01/obsidian/test01.md
Render stage complete.
```

---

## T9 Quality Gates: Type Annotation Fix (2026-02-19)

### Issue: Bare `dict` Annotations in verify.py

**Symptom**:
basedpyright reported multiple `reportMissingTypeArgument` errors for bare generics:
- `dict` in dataclass fields
- `dict` in function return types and parameters
- `report_data` variable type inference issues causing indexing errors

### Fix Applied

1. Changed dataclass field:
   ```python
   details: dict = field(default_factory=dict)
   ```
   To:
   ```python
   details: dict[str, Any] = field(default_factory=dict)
   ```

2. Changed function signatures:
   ```python
   def load_jsonl(path: Path) -> list[dict]:
   def load_json(path: Path) -> Optional[dict]:
   def load_golden(doc_id: str) -> Optional[dict]:
   def _has_evaluation_data(golden: Optional[dict]) -> bool:
   def compute_provenance_gate(run_dir: Path, golden: Optional[dict]) -> GateResult:
   def compute_reading_order_gate(run_dir: Path, golden: Optional[dict]) -> GateResult:
   def compute_citation_gate(run_dir: Path, golden: Optional[dict]) -> GateResult:
   def compute_figure_caption_gate(run_dir: Path, golden: Optional[dict]) -> GateResult:
   def compute_runtime_safety(run_dir: Path) -> dict:
   def _has_valid_evidence(fact: dict) -> bool:
   ```
   To use `dict[str, Any]` instead of bare `dict`.

3. Fixed `report_data` variable with explicit type:
   ```python
   report_data: dict[str, Any] = { ... }
   ```

### Verification

```
$ basedpyright src/ingest/verify.py --level error
0 errors, 0 warnings, 0 notes
```

Acceptance test still passes:
```
$ ingest_pdf --doc_id test01 --stage verify
Verification PASSED
QA Report: run\test01\qa\report.json
```

QA artifacts emit correctly:
- `run/test01/qa/report.json` - gate values with thresholds
- `run/test01/qa/stage_status.json` - stage status with hard_stop flag
- `run/test01/qa/runtime_safety.json` - network deny evidence

### Note

Gate formulas and runtime behavior unchanged. Only type annotations fixed to satisfy type checker.
