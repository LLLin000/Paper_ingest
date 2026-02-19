# Reading Synthesis Prompt

Generate an executive summary with traceable key evidence lines.

## Input

You will receive:
- Paper profile
- Logic graph
- All facts
- All themes
- Figure/table index and links

## Output Requirements

**STRICT JSON OUTPUT ONLY** - No prose wrapper, no markdown code fences.

Return a JSON object with this exact structure:

```json
{
  "executive_summary": "<3-5 sentence overview>",
  "key_evidence_lines": [
    {
      "line_id": "<string>",
      "statement": "<key finding or conclusion>",
      "fact_ids": [<fact_id>, ...],
      "strength": "<strong|moderate|weak>",
      "is_strong_claim": <boolean>
    }
  ],
  "figure_table_slots": [
    {
      "slot_id": "<string>",
      "position_hint": "<inline_after_claim|section_appendix|end_of_summary>",
      "asset_ids": [<asset_id>, ...],
      "render_mode": "<content_only|full_asset_embed>",
      "context_line_id": "<line_id or null>"
    }
  ]
}
```

## Key Evidence Line Rules

- Every `statement` must have ≥1 `fact_id` in `fact_ids`
- Each linked `fact_id` must exist in `reading/facts.jsonl`
- Each linked fact must have valid `quote` and `evidence_pointer`

## Strong Claim Definition

`is_strong_claim: true` when statement contains:
- Numeric data (percentages, p-values, confidence intervals)
- Causal claims ("X causes Y", "X leads to Y")
- Comparative efficacy ("X is better than Y")
- Guideline-like recommendations

**Strong claims with empty `fact_ids` will fail verification.**

## Figure/Table Slot Rules

- `asset_ids` must reference existing assets from `figures_tables/figure_table_index.jsonl`
- `context_line_id` links slot to a specific key evidence line (optional)
- Position hints:
  - `inline_after_claim`: Place immediately after referenced evidence line
  - `section_appendix`: Place at end of thematic section
  - `end_of_summary`: Place after executive summary

## Forbidden

- Unsupported synthesis claims (claims without fact linkage)
- Strong claims without evidence anchors
- Figure/table references to non-existent assets
- Silent truncation of evidence lines
