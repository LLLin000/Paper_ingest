2026-02-26T00:00:00Z - Outstanding caveats for Main Body leading-fragment repair

- Edge cases to watch:
  - Rare domain phrases that are 1-2 lowercase tokens and not in the protected
    set might be legitimate (e.g., some chemical abbreviations). Monitor
    false-positive reports and consider expanding protected list conservatively.
  - Documents with unusual segmentation where the first paragraph is a
    deliberate lowercase label (e.g., glossary or caption-like usage inside
    Main Body) could be trimmed incorrectly. We limited scope to Main Body
    to reduce this risk.

- Telemetry: consider adding a lightweight QA event when a fragment is
  dropped (timestamp + text) to allow sampling for false positives.

- Recommended follow-ups:
  - Add a focused unit test to assert that a telemetry event is appended when
    a Main Body suspicious fragment is dropped. Telemetry emission is
    best-effort so the test should run in an environment with qa_dir present.
  - Consider adding a small QA collector that aggregates dropped fragments for
    human review and for periodic tuning of the suspicious fragment matcher.

---

2026-02-26T17:30:00Z - Cross-column merge fix and test updates

- Fixed cross-column merge detection in `should_merge_lowercase_continuation`:
  - Added left-to-right column swap logic similar to hyphen-wrap handling
  - Relaxed x_diff threshold minimally (0.2 → 0.25) for cross-column cases where vertical wrap-back is detected
  - Root cause: prev_x_rel ~0.31 (left column), next_x_rel ~0.78 (right column), x_diff=0.46 exceeded 0.2 threshold

- Test updates (test-only):
  - `test_classify_clean_blocks_assigns_semantic_roles_and_role_based_keep_set`: Added `p1_ref` to expected kept blocks (reference_entry retained)
  - `test_run_paragraphs_writes_clean_artifacts_and_filters_nuisance`: Updated paragraph_count from 3 to 4 (reference entry paragraph included)

- Verification:
  - Paragraph-leading `^tive in improving` artifact confirmed ABSENT via grep
  - chen2024_final verification PASSED
  - Focused pytest suite: 53 passed

---

2026-02-26T14:09:14Z - Residual risks after conservative continuation merge expansion

- Residual risk: cross-page lowercase merge may occasionally join legitimate lowercase paragraph starts when layout metadata places unrelated text in bottom->top transitions.
  - Current mitigation: requires lowercase starter, non-terminal previous sentence, page delta guard (`<= 4`), and table/reference exclusions.

- Residual risk: same-row cross-gutter hyphen merge can over-join rare side-by-side column content ending with `-`.
  - Current mitigation: only enabled in hyphen-wrap path and still gated by section-cue rejection and tail checks.

- Residual risk: numeric-leading continuation trim (`^\d+, [A-Z]{2,}-\d+`) could trim uncommon valid numbered statements.
  - Current mitigation: sentence-length cap, fresh-sentence remainder requirement, table/reference exclusion.

- Follow-up suggestion: sample `chen2024_final` and one additional two-column paper for false-positive review focused on these three new guarded paths.

---

2026-02-26T14:32:53Z - Follow-up residual split fix risk notes

- New residual-risk surface: citation-tail post-repair merging at section emission stage.
  - Mitigations in place: citation-tail ending required, lowercase start required, allowlisted continuation verb required, reference/table/section guards retained.
- Known caveat: `looks_like_table_noise` can over-trigger on long narrative paragraphs; helper now only uses this guard on the next paragraph for this targeted path.
- Monitoring target: check for unintended merges where a true new paragraph starts with an allowlisted verb immediately after citation-tail endings.

---

2026-02-26T18:20:00Z - Prefer merged DOI when mapping numeric inline markers

- Root cause: When a references_merged.jsonl file already exists on disk with a DOI-backed
  record, inline numeric citation anchors (e.g. "[1]") were being mapped to local
  bib: keys derived from PDF reference text rather than to the authoritative merged DOI
  key present on disk. The mapping flow ran marker-based mapping which produced a
  pdf-local ref_key and then wrote that to cite_map instead of checking for a merged
  match by title+year for numeric markers.

- Fix summary: Enhance the merged-preference step in run_citations to, after the
  standard candidate mapping, detect numeric inline markers and attempt to resolve
  the marker -> pdf reference entry -> normalized title+year lookup against the
  existing references_merged.jsonl. If a merged record with a DOI/PMID is found, map
  the anchor to that doi: or pmid: key and prefer the merged confidence. This is a
  narrowly scoped, conservative change that does not alter other mapping heuristics.

- Verification: Added local test run for failing case; `pytest tests/test_citations_merged_mapping.py::test_mapping_prefers_merged` passes.

---

2026-02-26T15:27:34Z - Typing guards for citations mapping

- Added explicit guards/casts to satisfy static type checker (pyright/basedpyright)
- Ensured marker_to_key and author_year_to_doi only get str values after None checks
- Explicitly guarded marker_match before calling .end()
- No behavior changes; existing merged-DOI preference retained

---

2026-02-27T00:25:00Z - Conservative reference dedupe semantics to avoid false degradations

- Problem: runs with many API references and a small number of noisy PDF catalog entries
  (e.g. chen2024_final: api=523, pdf=24, merged=546) could yield misleadingly low
  raw dedupe rates because PDF entries are not bibliographically comparable.

- Fix: compute an identifier-aware dedupe metric when possible (counts only DOI/PMID
  carrying raw records and merged results). Fall back to legacy raw_total formula
  only if no identifier-bearing raw entries exist. Report both raw and identifier
  metrics in details for transparency.

- Verification: added focused unit test to ensure gate reports include dedupe and
  identifier completeness fields and that runs with missing artifacts continue to
  produce not_evaluated/degraded semantics as before.

---

2026-02-27T01:40:00Z - Follow-up: strict identifier-overlap-capacity semantics implemented

- Problem: identifier-based dedupe previously used a denominator that summed api + pdf
  identifier counts, which diluted the dedupe signal in API-dominated runs (chen2024_final).

- Change: dedupe_rate_on_identifiers now measures overlap relative to cross-source
  identifier capacity:
    - api_identifier_count = # API refs with doi/pmid
    - pdf_identifier_count = # PDF refs with doi/pmid
    - merged_with_identifier = # merged refs with doi/pmid
    - overlap_identifier_count = max(0, api_identifier_count + pdf_identifier_count - merged_with_identifier)
    - identifier_overlap_capacity = min(api_identifier_count, pdf_identifier_count)
    - dedupe_rate_on_identifiers = overlap_identifier_count / identifier_overlap_capacity (when capacity>0), clamped to [0,1]

- Supplemental details fields were added for auditability: api_identifier_count,
  pdf_identifier_count, overlap_identifier_count, identifier_overlap_capacity.

- Verification: added unit test that asserts numeric values for a small api-heavy
  scenario (api ids=2, pdf ids=1, merged ids=2) where dedupe_rate_on_identifiers==1.0.

---

2026-02-27T05:14:57Z - Empty section heading demotion in clean_document

- Root cause: section headings with no surviving body paragraphs were still emitted as markdown `###` headers, creating orphan headings in `clean_document.md`.

- Change: in `render_clean_document`, when a non-Main section has no meaningful body content after existing pruning/repair passes, the heading is now demoted to plain body text instead of emitted as `###`.
  - Information-preserving behavior: heading text is retained verbatim as narrative text.
  - Scope is narrow: only affects empty-section emission path; headings with real body paragraphs are unchanged.

- Test coverage added:
  - orphan heading is demoted (no `###` remains)
  - demoted heading text is preserved and appears in output order
  - legitimate heading with body remains a heading

---

2026-02-27T12:10:00Z - Conservative section artifact-line filtering in clean_document

- Root cause: obvious layout artifacts were still emitted in section body text when they appeared as standalone lines, especially isolated page numbers (for example `5`) and publisher banner lines beginning with `Check for updates`.

- Change: added a narrow helper in `render_clean_document` emission path that strips only:
  - lines matching strict page-number-only regex `^\d{1,3}$`
  - lines matching strict banner prefix regex `^check\s+for\s+updates\b` (case-insensitive)
  The helper is applied only to emitted section paragraphs; no broad keyword filtering was added.

- Safety posture: deterministic, conservative, and no-content-loss oriented.
  - No changes to heading demotion behavior.
  - No changes to verify/citations logic.
  - Legitimate narrative lines are preserved.

- Test coverage added in `tests/test_paragraphs_cleaning.py`:
  - `test_clean_document_drops_page_number_only_line_in_section_body`
  - `test_clean_document_drops_check_for_updates_banner_line_in_section_body`
  - `test_clean_document_keeps_legitimate_content_line_in_section_body`

- Verification:
  - Focused: `pytest tests/test_paragraphs_cleaning.py -k "drops_page_number_only_line or drops_check_for_updates_banner_line or keeps_legitimate_content_line_in_section_body"` -> 3 passed
  - Full suite: `pytest` -> 146 passed
