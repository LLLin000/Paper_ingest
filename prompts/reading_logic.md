# Reading Logic Graph Prompt

Construct an argument flow graph showing how claims and evidence connect.

## Input

You will receive:
- Paper profile (type, reading strategy)
- Section-organized paragraphs with IDs
- Any identified headings and structure

## Output Requirements

**STRICT JSON OUTPUT ONLY** - No prose wrapper, no markdown code fences.

Return a JSON object with this exact structure:

```json
{
  "nodes": [
    {
      "node_id": "<string>",
      "type": "<claim|evidence|method|limitation|future_work>",
      "text": "<brief description>",
      "source_para_ids": [<para_id>, ...]
    }
  ],
  "edges": [
    {
      "from": "<node_id>",
      "to": "<node_id>",
      "relation": "<supports|contradicts|qualifies|extends|cites>"
    }
  ],
  "argument_flow": {
    "premises": [<node_id>, ...],
    "core_claims": [<node_id>, ...],
    "conclusions": [<node_id>, ...]
  }
}
```

## Node Types

- `claim` - Assertion or finding
- `evidence` - Data, statistics, observations supporting claims
- `method` - Methodological approach or technique
- `limitation` - Acknowledged weakness or constraint
- `future_work` - Suggested future directions

## Edge Relations

- `supports` - Evidence supports claim
- `contradicts` - Conflicts with another node
- `qualifies` - Adds nuance or conditions
- `extends` - Builds upon prior work
- `cites` - References external source

## Evidence Requirements

- Every node must have `source_para_ids` pointing to actual paragraphs
- Never create nodes for inferred content not in the paper
- If a logical connection is uncertain, omit the edge rather than guess
