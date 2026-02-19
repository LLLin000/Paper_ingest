# T0 Bootstrap Implementation Decisions

## Date: 2026-02-18

---

## Package Structure Decision

**Decision**: Use `src/ingest/` layout with `hatchling` build backend.

**Rationale**:
- `src/` layout prevents accidental imports from working directory
- `hatchling` is modern, minimal, and PyPA-recommended
- Package name `pdf-ingest` maps to import path `ingest`

**Import path**: `from ingest.cli import main`

---

## CLI Framework Decision

**Decision**: Use Typer for CLI implementation.

**Rationale**:
- Type-hint friendly, auto-generates help from signatures
- Used in T0 research baseline (learnings.md)
- Consistent with pydantic for type validation

---

## Stage Enum Naming

**Decision**: Use lowercase-hyphenated values matching contract spec.

**Stage values** (from `pdf-blueprint-contracts.md` lines 246-254):
- `init-only`, `extractor`, `overlay`, `vision`, `paragraphs`
- `citations`, `figures-tables`, `reading`, `render`, `full`, `verify`

**Note**: `figures-tables` hyphenated per contract (not underscore).

---

## Schema Validation Approach

**Decision**: Create JSON Schema files in `schemas/` folder.

**Rationale**:
- Contract requires `python -m ingest.validate --run run/<id> --schemas schemas/ --strict`
- JSON Schema allows language-agnostic validation
- Pydantic v2 can export to JSON Schema for consistency

**Bootstrap**: Created `manifest.schema.json` and `_index.json` as foundation.

---

## Prompt Bundle Organization

**Decision**: Store prompts as markdown files in `prompts/` folder.

**Files created per contract (lines 293-298)**:
- `vision_page_structure.md`
- `reading_profile.md`
- `reading_logic.md`
- `reading_facts.md`
- `reading_themes.md`
- `reading_synthesis.md`

**Rationale**:
- Markdown allows embedded examples and formatting
- Version-controlled alongside code
- `prompt_bundle_version` in manifest tracks changes

---

## Dependency Pinning

**Decision**: Use range constraints in pyproject.toml dependencies.

```toml
dependencies = [
    "pymupdf>=1.26.0,<1.28.0",
    "typer>=0.9.0",
    "pydantic>=2.0.0",
]
```

**Rationale**:
- Allows patch updates while preventing breaking changes
- PyMuPDF <1.28.0 prevents untested major versions
- Exact pinning deferred to requirements.txt for CI

---

## Lockfile Implementation

**Decision**: Use `requirements.lock` with exact pinned versions.

**File**: `requirements.lock` in project root.

**Format**: Standard pip requirements format with `==` pinning.

**Rationale**:
- Contract requires "lockfile for dependencies" (line 270)
- `requirements.lock` is conventional and pip-native
- Includes all transitive dependencies for full reproducibility
- Grouped by dependency source for maintainability

**Usage**:
```bash
pip install -r requirements.lock
```

**Regeneration**:
After `pyproject.toml` changes, update lockfile by inspecting installed versions.

---

## T1 Run Skeleton and Manifest Contract Decisions

**Date**: 2026-02-18

### Field Alias for model_config

**Decision**: Use `llm_config` as field name with `alias="model_config"` for serialization.

**Rationale**:
- Pydantic `BaseModel` reserves `model_config` as class attribute for configuration
- Contract requires `model_config` as JSON key for LLM settings
- Alias allows both correct Python naming and contract-compliant JSON output
- `ConfigDict(populate_by_name=True)` allows loading by either name

### SHA256 Chunk Size

**Decision**: Use 8192 bytes for chunked file reading.

**Rationale**:
- Memory-efficient for large PDF files
- Standard chunk size for hashlib operations
- No performance penalty for small files

### Lockfile Hash Prefix Length

**Decision**: Use first 16 characters of SHA256 hash.

**Rationale**:
- Consistent with `doc_id` computation pattern
- Sufficient uniqueness for lockfile version tracking
- Short enough for human-readable output

### Path Resolution Strategy

**Decision**: Always use `Path.resolve()` before storing paths in manifest.

**Rationale**:
- Windows/unicode path handling requires absolute paths
- Resolves symlinks and normalizes path separators
- Ensures manifest paths work regardless of CWD at runtime
- Contract prohibits machine-specific absolute paths, but `resolve()` produces consistent output for the same logical path

### Default Model Config Fields

**Decision**: Use empty strings for model placeholders with default temperature 0.0.

**Rationale**:
- Allows downstream stages to populate via CLI/env/manifest precedence
- Temperature 0.0 ensures deterministic LLM output (important for reproducibility)
- Empty strings indicate "not yet configured" vs None which breaks JSON schema

---

## T2 Extractor Stage Decisions

**Date**: 2026-02-18

### Line-Level Block Extraction

**Decision**: Extract at line level rather than block or span level.

**Rationale**:
- Contract specifies "line-level blocks" (lines 116-122)
- PyMuPDF's block level is too coarse (may merge multiple paragraphs)
- Span level is too fine (individual text fragments)
- Line level provides good balance for downstream processing

### Font Stats as dict[str, Any]

**Decision**: Use `dict[str, Any]` for font_stats field.

**Rationale**:
- PyMuPDF returns heterogeneous dict with various types
- Strong typing would require complex union types
- Any is acceptable for extraction-stage data that gets normalized downstream
- basedpyright `reportExplicitAny` warning accepted as trade-off

### Coordinate System

**Decision**: Store bbox as `[x0, y0, x1, y1]` in PDF points.

**Rationale**:
- Contract specifies `bbox_pt` in PDF points (lines 117-120)
- PyMuPDF uses same coordinate system natively
- No conversion needed at extraction stage
- Pixel coordinates (`bbox_px`) computed later by vision stage with dpi/scale

### Column Guess Binary Split

**Decision**: Use simple left/right binary column guess (1 or 2).

**Rationale**:
- Contract specifies `column_guess: int` with no further specification
- Simple heuristic sufficient for v1
- Multi-column detection (>2 columns) deferred to vision overlay stage
- Single-column pages will still get column=1 assignment

### Fault Injection Event Per Block

**Decision**: Log one fault event per affected block to `fault_injection.json`.

**Rationale**:
- Provides granular traceability for testing
- Contract requires "fallback event must be logged" (lines 410-416)
- Per-block logging allows precise verification of fault injection scope
- File can be cleared between test runs if needed

### Heading Detection Thresholds

**Decision**: Use 1.1x average body size threshold for heading detection.

**Rationale**:
- Provides reasonable sensitivity for typical academic papers
- Combined with bold requirement reduces false positives
- 150-character text length cap excludes long paragraphs
- Thresholds are conservative to minimize false positives at extraction stage

---

## T3 Overlay Stage Decisions

**Date**: 2026-02-18

### Coordinate Conversion Formula

**Decision**: Use same zoom formula as extractor rendering.

**Formula**: `zoom = dpi / 72.0 * scale`

**Rationale**:
- Ensures pixel-perfect alignment between overlay rectangles and rendered page content
- Matches PyMuPDF's internal coordinate system (72 points per inch)
- Consistent with `render_page_to_png()` implementation

### Output File Naming

**Decision**: Use `pXXX_annot.png` suffix for annotated pages.

**Rationale**:
- Clear distinction from original `pXXX.png` files
- Deterministic naming allows downstream stages to locate overlays
- Preserves original page images for comparison

### Block Visualization Style

**Decision**: Green outline (2px) with white background label.

**Rationale**:
- Green provides good contrast on typical document backgrounds
- 2px width is visible without obscuring content
- White label background ensures text readability
- Block ID positioned above bbox (or inside if at page top)

### Font Loading Strategy

**Decision**: Try multiple font paths with fallback to default.

**Paths tried**:
1. `arial.ttf` (Windows)
2. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` (Linux)
3. `/System/Library/Fonts/Helvetica.ttc` (macOS)
4. `ImageFont.load_default()` (fallback)

**Rationale**:
- Cross-platform compatibility without external font dependencies
- Graceful degradation ensures overlay works on all systems
- Default font is functional but may be small

### Dependency Addition

**Decision**: Add `pillow>=10.0.0` to dependencies.

**Rationale**:
- Pillow is the standard Python imaging library
- Required for image loading, drawing, and saving
- Version 10.0.0+ supports modern Python versions
- No additional dependencies required

---

## T4 Vision Stage Decisions

**Date**: 2026-02-18

### SiliconFlow API Provider

**Decision**: Use SiliconFlow as default vision model provider.

**Rationale**:
- Contract allows network egress for model-call stages (lines 219-223)
- SiliconFlow offers cost-effective models suitable for document layout analysis
- No additional SDK required - standard HTTP API with urllib
- Configurable via environment variables only (no hardcoded secrets)

### Default Model Selection

**Decision**: Use `Qwen/Qwen2.5-7B-Instruct` as default vision model.

**Rationale**:
- Lightweight and cost-effective for structured output tasks
- Sufficient capability for layout analysis and role labeling
- Overridable via `SILICONFLOW_VISION_MODEL` environment variable
- 7B parameter size balances speed and accuracy

### HTTP Library Choice

**Decision**: Use stdlib `urllib.request` instead of `requests` or `httpx`.

**Rationale**:
- Zero additional dependencies
- Sufficient for simple POST requests
- No need for connection pooling or advanced features
- Timeout support built-in

### Retry Strategy

**Decision**: Retry exactly once on parse failure, then fallback.

**Rationale**:
- Contract specifies "retry-once" (lines B.2-B.4)
- Avoids infinite retry loops
- Second failure indicates persistent issue requiring fallback
- Log retry attempt count for debugging

### Fallback Reading Order

**Decision**: Sort by `(column_guess, bbox_pt[1], bbox_pt[0])`.

**Rationale**:
- Column-first ordering handles multi-column layouts correctly
- Y-coordinate secondary sort ensures top-to-bottom within columns
- X-coordinate tertiary sort handles edge cases
- Deterministic for same input (contract requirement)

### Merge Group Threshold

**Decision**: Use 20pt vertical gap threshold for merge groups.

**Rationale**:
- Typical line spacing in academic papers is 12-16pt
- 20pt threshold allows for slight variations while avoiding false merges
- Only merges same-column, same-role (Body) blocks
- Configurable in future if needed

---

## T5 Paragraph Stage Decisions

**Date**: 2026-02-18

### Para ID Hash Algorithm

**Decision**: Use SHA256 with 12-character hex prefix.

**Format**: `para_{12_char_hex}`

**Rationale**:
- Contract requires stable ID across reruns (line 183-184)
- SHA256 provides cryptographic stability
- 12-char prefix balances uniqueness with readability
- No external dependencies (stdlib hashlib)

### Evidence Pointer Fields

**Decision**: Store `pages` as list, not singular page.

**Fields**:
- `pages`: list[int] - all pages covered by paragraph
- `bbox_union`: list[float] - [x0, y0, x1, y1] union bbox
- `source_block_ids`: list[str] - source block IDs

**Rationale**:
- Paragraphs may span multiple pages
- Union bbox enables downstream to locate content
- Source block IDs enable provenance tracing

### Neighbor Link Strategy

**Decision**: Sort by (page_start, bbox_y) then link sequentially.

**Rationale**:
- Matches reading order semantics
- Simple and deterministic
- Covers single and multi-page paragraphs

### Singleton Block Handling

**Decision**: Include singleton blocks as paragraphs, skip HeaderFooter.

**Rationale**:
- No data loss - all text content preserved
- HeaderFooter filtered per contract role labels
- Provenance tracks singleton vs merged origin

### Section Path Extraction

**Decision**: Extract from Heading role blocks, nullable otherwise.

**Implementation**:
- Clean heading text (remove numbering prefix)
- Return list of heading strings or null

**Rationale**:
- Heading signals provide section context
- Null acceptable per contract (line 135)
- Cleaned text removes false headings

### Uncertainty Metadata

**Decision**: Append to provenance.notes field as space-separated tags.

**Tags**:
- `low_confidence` - confidence < 0.6
- `no_section_path` - section_path is null
- `no_merge_group` - singleton paragraph

**Rationale**:
- Explicit uncertainty per contract requirement
- Append-only preserves history
- Space-separated allows parsing

---

## T6 Citation Mapper Decisions (2026-02-18)

### Link Type Detection

**Decision**: Use numeric kind values instead of PyMuPDF constants.

**Implementation**:
- kind == 3: internal
- kind in (1, 2): external
- kind == 4: external (Gotob)

**Rationale**:
- pymupdf.LINK_* constants not available in all versions
- Numeric values consistent across PyMuPDF releases

### DOI Extraction

**Decision**: Strip URL parameters from extracted DOI strings.

**Implementation**:
- Handle both `?param=value` and `&param=value` patterns
- Apply after initial regex extraction

**Rationale**:
- Crossmark URLs embed DOI in query string
- Clean DOI format required per contract (doi:10.xxxx/xxxxx)

### Anchor ID Generation

**Decision**: Use SHA256 hash of page_index_dest tuple.

**Implementation**:
- Format: `anchor_{sha256[:12]}`
- Components: page number, link index, destination/uri

**Rationale**:
- Deterministic across runs
- Collision-resistant
- Matches para_id pattern from T5

### Unmapped Entry Handling

**Decision**: Retain unmapped entries with strategy "none" and explicit reason.

**Reasons**:
- "empty anchor_text" - no text in link
- "no reference paragraphs available" - none detected
- "no matching reference found" - strategies exhausted

**Rationale**:
- Contract requires no hard-stop on partial mapping
- Explicit reason enables debugging
- Strategy "none" signals unsuccessful mapping

---

## T6.5 Figure/Table Extraction Stage Decisions

**Date**: 2026-02-19

### Detection Approach

**Decision**: Use dual-source detection (PDF images + caption text).

**Rationale**:
- PDF image extraction provides precise bounding boxes from PyMuPDF
- Caption text patterns identify figure/table context
- Combining both provides robust detection even with partial data

### Caption Pattern Implementation

**Decision**: Use regex patterns for Figure/Table caption detection.

**Patterns**:
- Figure: `^(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)[\s\.\-:]*(.*)`
- Table: `^(?:Table)\s*(\d+[a-zA-Z]?)[\s\.\-:]*(.*)`

**Rationale**:
- Matches common academic paper caption formats
- Captures figure/table numbers for potential future use
- Case-insensitive for robustness

### Caption-to-Image Linking

**Decision**: Link captions to images via vertical proximity analysis.

**Algorithm**:
- Caption should appear below figure (y0 of caption > y1 of image)
- Distance should be small (caption_y0 - image_y1 < 200pt)
- Select caption with minimum distance

**Rationale**:
- Academic papers typically place figures above captions
- Proximity provides reliable linking without ML
- Falls back gracefully when no match found

### Asset ID Format

**Decision**: Use `fig_NNN` and `tbl_NNN` format with 3-digit zero-padding.

**Rationale**:
- Matches contract requirement (lines 98-99)
- Zero-padding ensures proper sorting
- Sequential counters maintain deterministic ordering

### Baseline OCR Decision

**Decision**: Do NOT implement OCR in baseline stage.

**Rationale**:
- Contract specifies "No heavy OCR dependencies for this baseline stage"
- text_content remains null as nullable field
- Can be added later when needed without breaking contract

### Synthesis Slot Placeholders

**Decision**: Leave by_synthesis_slot empty in baseline.

**Rationale**:
- Contract specifies synthesis slots populated by reading stage
- Empty dict is valid JSON structure
- Reading stage will populate when facts/synthesis are available

---

## T7 Reading Engine Decisions (2026-02-19)

### Five-Stage Architecture

**Decision**: Implement five distinct stages (D1-D5) with separate LLM calls.

**Rationale**:
- Contract specifies D1-D5 artifacts with distinct schemas (lines 76-179)
- Each stage has specific prompt template in `prompts/reading_*.md`
- Separation enables independent fallback per stage
- Enables partial pipeline completion even if later stages fail

### SiliconFlow Model Selection

**Decision**: Use same `Qwen/Qwen2.5-7B-Instruct` model as vision stage.

**Rationale**:
- Lightweight, cost-effective for structured output
- Same API integration pattern reduces complexity
- Sufficient capability for reading analysis tasks
- Configurable via `SILICONFLOW_READING_MODEL` environment variable

### JSON Template Formatting

**Decision**: Use string concatenation instead of f-strings for JSON templates.

**Implementation**:
```python
# Use concatenation to avoid f-string brace conflicts:
template = '{"page": ' + str(page) + ', "reading_order": []}'
```

**Rationale**:
- F-string interprets `{` as format specifier causing "Invalid format specifier" errors
- Non-f-string concatenation preserves literal braces in JSON
- Consistent pattern across all reading stage generators

### Asset Slot Population

**Decision**: Populate synthesis with figure/table slots from T6.5 artifacts.

**Implementation**:
- Reads `figure_table_index.jsonl` for asset metadata
- Reads `figure_table_links.json` for existing mappings
- Fills 3 figure slots and 1 table slot in synthesis JSON
- Handles null captions with fallback text

**Rationale**:
- Contract requires synthesis to reference visual evidence
- Figure/table slots enable traceable claims
- Graceful handling when assets unavailable

### Facts Batching Strategy

**Decision**: Batch paragraphs in groups of 30 for facts extraction.

**Implementation**:
```python
BATCH_SIZE = 30
for i in range(0, len(paragraphs), BATCH_SIZE):
    batch = paragraphs[i:i+BATCH_SIZE]
    # Process batch with LLM
```

**Rationale**:
- LLM context windows limit batch size
- 30 paragraphs per batch balances API efficiency with reliability
- Produces one fact per paragraph (1131 facts for test01)
- Fallback extracts one fact per paragraph deterministically

### Fallback Chain Design

**Decision**: Implement deterministic fallback for each stage.

**Fallback Outputs**:
- D1 Profile: Default paper type "empirical_study", generic problem statement
- D2 Logic: Empty argument_nodes and claim_edges
- D3 Facts: One fact per paragraph with deterministic content
- D4 Themes: Empty themes array
- D5 Synthesis: Generic summary with figure slots from T6.5

**Rationale**:
- Contract requires deterministic fallback on failure
- Fallbacks maintain schema compliance
- Enables pipeline continuation even with API failures

### Fault Event Logging

**Decision**: Log to `qa/fault_injection.json` with required keys.

**Keys**:
- `timestamp`: ISO 8601 format
- `stage`: "reading" + specific stage name
- `error_type`: "malformed_json", "api_error", "timeout"
- `error_message`: Descriptive error string
- `doc_id`: Document identifier
- `recovery_action`: "retry" or "fallback"

**Rationale**:
- Contract specifies fault injection logging requirements
- Structured format enables automated analysis
- Recovery action enables pipeline restart at failure point

---

## T8 Obsidian Renderer Decisions (2026-02-19)

### Output Format

**Decision**: Generate Obsidian Flavored Markdown with YAML frontmatter.

**Rationale**:
- Contract requires `run/<id>/obsidian/<id>.md` output (line 31)
- Obsidian markdown supports wikilinks for traceability (`[[fact_id]]`)
- YAML frontmatter provides metadata for Obsidian properties/databases
- Tables used for Evidence Index and Citations sections

### Required Sections

**Decision**: Include 6 core sections in output.

**Sections**:
1. Paper Profile - from paper_profile.json
2. Logic Graph - from logic_graph.json (nodes, edges, argument flow)
3. Themes - from themes.json with cross-theme links
4. Evidence Index - table with fact_id, category, statement, page, bbox
5. Key Evidence Lines - from synthesis.key_evidence_lines with fact_id links
6. Citations - mapped/unmapped tables with reasons

**Rationale**:
- Contract specifies required sections (line 233-251)
- Traceability maintained through wikilinks to facts
- Quality metric computed from data availability

### Figure/Table Slot Rendering

**Decision**: Support two render modes.

**Modes**:
- `content_only`: Show asset metadata (type, page, bbox) without embedding
- `full_asset_embed`: Include markdown image embed path when image exists

**Rationale**:
- Contract specifies render_mode in synthesis slots (line 178)
- content_only keeps file size manageable
- full_asset_embed enables visual assets in Obsidian

### Quality Determination

**Decision**: Compute quality from 4 signals.

**Signals**:
- has_profile: paper_type != unknown
- has_synthesis: executive_summary and key_evidence_lines present
- has_facts: any fact has non-empty quote
- has_citations: any citation has mapped_ref_key

**Quality levels**:
- high: 4/4 signals
- medium: 2-3 signals
- low: 0-1 signals

**Rationale**:
- Contract requires quality metadata in frontmatter
- Quality computed from data availability (not LLM confidence)
- Enables downstream consumers to gauge reliability

---

## Production Trial Model Configuration (2026-02-19)

### Default Model Updates

**Decision**: Update default models for production trial.

**Changes**:
- Vision: `Qwen/Qwen2.5-7B-Instruct` → `THUDM/GLM-4.1V-9B`
- Reading: `Qwen/Qwen2.5-7B-Instruct` → `Qwen/Qwen2.5-14B-Instruct`

**Rationale**:
- GLM-4.1V is preferred vision model per user specification
- Qwen2.5-14B-Instruct provides stronger reading comprehension
- Environment variables still take highest priority (SILICONFLOW_VISION_MODEL, SILICONFLOW_READING_MODEL)
- No API key hardcoding - uses SiliconFlow API with env var configuration

---

### Chinese Output Configuration

**Decision**: Configure render.py and reading prompts for Chinese-oriented output.

**Changes**:
- All section headers now include Chinese with optional English in parentheses
- Key table headers in Chinese (事实ID, 类别, 陈述, 页码, etc.)
- Fallback text includes Chinese translations alongside English
- Reading prompts updated with "Please respond in Chinese where possible" guidance

**Rationale**:
- User requested Chinese-oriented markdown output for readability
- Preserves JSON schema contracts - only user-facing text changed
- Fallback text ensures Chinese output even on API failure

**Affected sections**:
- Paper Profile → 论文档案 (Paper Profile)
- Logic Graph → 逻辑图谱 (Logic Graph)
- Nodes → 节点 (Nodes)
- Edges → 边 (Edges)
- Argument Flow → 论证流程 (Argument Flow)
- Premises → 前提 (Premises)
- Core Claims → 核心主张 (Core Claims)
- Conclusions → 结论 (Conclusions)
- Themes → 主题 (Themes)
- Cross-Theme Links → 跨主题链接 (Cross-Theme Links)
- Evidence Index → 证据索引 (Evidence Index)
- Key Evidence Lines → 关键证据线 (Key Evidence Lines)
- Claim → 论断 (Claim)
- Supporting Facts → 支撑事实 (Supporting Facts)
- Figure/Table Slots → 图表占位符 (Figure/Table Slots)
- Citations → 引用 (Citations)
- Mapped Citations → 已映射引用 (Mapped Citations)
- Unmapped Citations → 未映射引用 (Unmapped Citations)
