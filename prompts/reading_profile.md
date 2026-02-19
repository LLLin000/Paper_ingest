# Reading Profile Prompt

Analyze the paper to determine its type, research problem, and optimal reading strategy.

## Input

You will receive the paper's:
- Title (if available from metadata)
- Abstract or first few paragraphs
- Section headings (if available)
- Initial paragraph samples

## Output Requirements

**STRICT JSON OUTPUT ONLY** - No prose wrapper, no markdown code fences.

Return a JSON object with this exact structure:

```json
{
  "paper_type": "<type>",
  "paper_type_confidence": <float 0-1>,
  "research_problem": "<string>",
  "claimed_contribution": "<string>",
  "reading_strategy": "<strategy>"
}
```

## Paper Types

- `original_research` - Novel experimental/computational study
- `review` - Systematic or narrative review
- `meta_analysis` - Statistical synthesis of prior studies
- `case_report` - Clinical or technical case description
- `methodology` - Methods/protocol paper
- `commentary` - Opinion, editorial, or perspective
- `guidelines` - Clinical or technical guidelines

## Reading Strategies

- `methods_first` - For original research: focus on methods before results
- `evidence_synthesis` - For reviews: map themes and evidence quality
- `statistical_focus` - For meta-analyses: emphasize forest plots and heterogeneity
- `narrative_flow` - For commentaries: track argument structure
- `protocol_extraction` - For guidelines: extract actionable recommendations

## Evidence Requirements

- `research_problem` must be stated in the paper (quote if possible)
- `claimed_contribution` must reflect what authors claim, not your inference
- If these cannot be found, use `"missing_information": true` in that field

## Missing Information Handling

If a required field cannot be determined:
```json
{
  "research_problem": {
    "value": null,
    "missing_information": true,
    "reason": "No explicit problem statement found in abstract or introduction"
  }
}
```
