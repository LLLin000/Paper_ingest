# Reading Themes Extraction Prompt

Group related facts into themes and identify cross-theme connections.

## Input

You will receive:
- All extracted facts from `reading/facts.jsonl`
- Paper profile and logic graph

## Output Requirements

**STRICT JSON OUTPUT ONLY** - No prose wrapper, no markdown code fences.

Return a JSON object with this exact structure:

```json
{
  "themes": [
    {
      "theme_id": "<string>",
      "name": "<concise theme name>",
      "description": "<1-2 sentence summary>",
      "fact_ids": [<fact_id>, ...],
      "strength": "<strong|moderate|weak>",
      "evidence_quality": "<high|medium|low>"
    }
  ],
  "cross_theme_links": [
    {
      "from_theme": "<theme_id>",
      "to_theme": "<theme_id>",
      "relation": "<supports|contradicts|extends|qualifies>",
      "explanation": "<brief explanation>"
    }
  ],
  "contradictions": [
    {
      "theme_ids": [<theme_id>, <theme_id>],
      "description": "<nature of contradiction>",
      "resolution": "<how paper addresses it, or null>"
    }
  ]
}
```

## Theme Strength Criteria

- `strong`: ≥3 facts with statistics/results, consistent evidence
- `moderate`: 2+ facts, some quantitative support
- `weak`: 1-2 facts, primarily qualitative

## Evidence Quality Criteria

- `high`: Multiple independent sources, strong methodology
- `medium`: Some methodological concerns or limited sources
- `low`: Single source, weak methodology, or unclear provenance

## Evidence Requirements

- Every theme must have `fact_ids` pointing to actual facts
- Never create themes for inferred content
- `contradictions` may be `null` if no contradictions exist

## Forbidden

- Themes without linked facts
- Implicit inferences not grounded in extracted facts
