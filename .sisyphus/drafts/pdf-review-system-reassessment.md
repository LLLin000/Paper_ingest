# Draft: PDF Review Plan Reassessment

## Requirements (confirmed)
- Reassess existing plan quality using real PDFs in `example_pdf/`.
- Focus on structural robustness and evidence traceability.
- Add a new Text+Vision capability: extract figure/table assets independently, recognize their content, and place outputs into final summary at corresponding positions.

## Research Findings
- Corpus includes 10 PDFs with mixed publication eras/styles.
- Observed structures:
  - Modern two-column reviews with boxed sidebars and long references.
  - Figure-heavy pages with non-linear reading flow.
  - Older PDFs with dense metadata blocks and formatting artifacts.
- Text extraction risk signals seen in samples:
  - Ligature/OCR-like corruption (`cuf`, odd glyph substitutions).
  - Header/footer/page-number bleed into body stream.

## External Guidance
- Librarian: reinforce edge-case handling for columns, references variability, multilingual text, malformed links.
- Oracle: top blockers are provenance integrity, reading-order correctness, and citation-to-reference mapping completeness.

## Key Gaps to Add to Plan
- Atomic-claim provenance coverage metric and hard fail thresholds.
- Two-column/box text isolation metric and inversion thresholds.
- Citation mapping completeness/precision thresholds.
- Figure-caption association quality gate.
- Explicit truncation reporting (no silent token cuts).
- Deterministic output identity/path normalization.
- Security/resource hardening (timeout, no network egress, safe abort).
- Regression corpus with per-doc scored gates.
- Dedicated figure/table extraction + understanding pipeline with placement contract into synthesis/Obsidian output.

## Scope Boundaries
- INCLUDE: plan hardening and measurable acceptance gates.
- EXCLUDE: implementing these changes now.

## Open Questions
- Whether to patch the current plan immediately with these risk gates and release blockers.
