# Reading Facts Extraction Prompt

Extract atomic facts from paragraphs with evidence pointers.

## Input

You will receive:
- A paragraph with `para_id`
- The paragraph's role and section path
- Page and bounding box information

## Output Requirements

**STRICT JSON OUTPUT ONLY** - No prose wrapper, no markdown code fences.

Return a JSON array of fact objects:

```json
[
  {
    "fact_id": "<string>",
    "para_id": "<string>",
    "category": "<category>",
    "statement": "<concise factual statement>",
    "quote": "<verbatim quote from text, max 30 words>",
    "evidence_pointer": {
      "page": <int>,
      "bbox": [x0, y0, x1, y1],
      "source_block_ids": [<block_id>, ...]
    }
  }
]
```

## Fact Categories

- `result` - Experimental or observational finding
- `statistics` - Quantitative data (percentages, p-values, CIs)
- `comparison` - Comparative statement between groups/methods
- `definition` - Term or concept definition
- `mechanism` - Causal or mechanistic explanation
- `limitation` - Acknowledged weakness or scope constraint
- `recommendation` - Guideline or best practice statement
- `background` - Contextual or literature background
- `none` - Cannot categorize (fallback)

## Quote Requirements

- Quote must be ≤30 words
- If truncated, add `"quote_truncated": true` and `"truncation_reason": "<reason>"`
- Quote must be verbatim from source text

## Evidence Requirements

- Every fact must have valid `evidence_pointer`
- `page` and `bbox` must correspond to actual source location
- Never synthesize facts not present in the paragraph

## Missing Information

If a paragraph contains no extractable facts:
```json
[
  {
    "fact_id": "<para_id>_no_facts",
    "para_id": "<para_id>",
    "category": "none",
    "statement": "No extractable atomic facts in this paragraph",
    "quote": "",
    "evidence_pointer": { "page": <int>, "bbox": null, "source_block_ids": [] },
    "missing_information": true
  }
]
```
