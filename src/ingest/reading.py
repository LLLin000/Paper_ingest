"""Reading Pipeline - Paper analysis with D1-D5 artifacts.

Contract: .sisyphus/plans/pdf-blueprint-contracts.md (lines 76-179)

Output artifacts:
- reading/paper_profile.json: paper_type, research_problem, claimed_contribution, reading_strategy
- reading/logic_graph.json: nodes, edges, argument_flow
- reading/facts.jsonl: fact_id, para_id, category, statement, quote, evidence_pointer
- reading/themes.json: themes, cross_theme_links, contradictions
- reading/synthesis.json: executive_summary, key_evidence_lines, figure_table_slots
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .manifest import Manifest, load_manifest

SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

FACT_CATEGORIES = frozenset({
    "result", "statistics", "comparison", "definition",
    "mechanism", "limitation", "recommendation", "background", "none"
})

PAPER_TYPES = frozenset({
    "original_research", "review", "meta_analysis",
    "case_report", "methodology", "commentary", "guidelines"
})

READING_STRATEGIES = frozenset({
    "methods_first", "evidence_synthesis", "statistical_focus",
    "narrative_flow", "protocol_extraction"
})


@dataclass
class PaperProfile:
    paper_type: str
    paper_type_confidence: float
    research_problem: str
    claimed_contribution: str
    reading_strategy: str


@dataclass
class Fact:
    fact_id: str
    para_id: str
    category: str
    statement: str
    quote: str
    evidence_pointer: dict[str, Any]
    quote_truncated: bool = False
    truncation_reason: Optional[str] = None


@dataclass
class FaultEvent:
    stage: str
    fault: str
    retry_attempts: int
    fallback_used: bool
    status: str


@dataclass
class LLMCallEvent:
    stage: str
    step: str
    model: str
    endpoint: str
    success: bool
    error_type: str
    http_status: Optional[int]
    prompt_chars: int
    response_chars: int
    parse_success: bool
    timestamp: str
    response_preview: str


def load_paragraphs(paragraphs_path: Path) -> list[dict[str, Any]]:
    paragraphs = []
    if not paragraphs_path.exists():
        return paragraphs
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                paragraphs.append(d)
            except json.JSONDecodeError:
                continue
    return paragraphs


def load_cite_map(cite_map_path: Path) -> list[dict[str, Any]]:
    mappings = []
    if not cite_map_path.exists():
        return mappings
    with open(cite_map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                mappings.append(d)
            except json.JSONDecodeError:
                continue
    return mappings


def load_figure_table_index(index_path: Path) -> list[dict[str, Any]]:
    assets = []
    if not index_path.exists():
        return assets
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                assets.append(d)
            except json.JSONDecodeError:
                continue
    return assets


def load_figure_table_links(links_path: Path) -> dict[str, Any]:
    if not links_path.exists():
        return {"by_section": {}, "by_fact": {}, "by_synthesis_slot": {}}
    try:
        with open(links_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"by_section": {}, "by_fact": {}, "by_synthesis_slot": {}}


def call_siliconflow(prompt: str, max_tokens: int = 4000) -> tuple[str, dict[str, Any]]:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    model = os.environ.get("SILICONFLOW_READING_MODEL", SILICONFLOW_DEFAULT_MODEL)
    meta: dict[str, Any] = {
        "model": model,
        "endpoint": SILICONFLOW_ENDPOINT,
        "success": False,
        "error_type": "unknown",
        "http_status": None,
        "prompt_chars": len(prompt),
        "response_chars": 0,
        "response_preview": "",
    }
    if not api_key:
        meta["error_type"] = "missing_api_key"
        return "{}", meta
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        SILICONFLOW_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            status = getattr(resp, "status", None)
            result = json.loads(body)
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            meta["http_status"] = status
            meta["response_chars"] = len(content)
            meta["response_preview"] = content[:300]
            meta["success"] = bool(content)
            meta["error_type"] = "none" if content else "empty_content"
            return content or "{}", meta
    except urllib.error.HTTPError as e:
        meta["http_status"] = e.code
        meta["error_type"] = "http_error"
        return "{}", meta
    except urllib.error.URLError:
        meta["error_type"] = "url_error"
        return "{}", meta
    except json.JSONDecodeError:
        meta["error_type"] = "response_json_decode_error"
        return "{}", meta
    except TimeoutError:
        meta["error_type"] = "timeout"
        return "{}", meta
    except KeyError:
        meta["error_type"] = "response_schema_error"
        return "{}", meta


def append_llm_call_event(qa_dir: Path, event: LLMCallEvent) -> None:
    qa_dir.mkdir(parents=True, exist_ok=True)
    out_path = qa_dir / "llm_calls.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def generate_fact_id(para_id: str, index: int) -> str:
    return f"fact_{para_id.replace('para_', '')}_{index:03d}"


def count_words(text: str) -> int:
    return len(text.split())


def truncate_quote(quote: str, max_words: int = 30) -> tuple[str, bool, Optional[str]]:
    words = quote.split()
    if len(words) <= max_words:
        return quote, False, None
    truncated = " ".join(words[:max_words])
    return truncated, True, f"Quote truncated to {max_words} words"


def parse_paper_profile(raw: str) -> Optional[PaperProfile]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        required = ["paper_type", "paper_type_confidence", "research_problem", "claimed_contribution", "reading_strategy"]
        if not all(k in data for k in required):
            return None
        paper_type = data.get("paper_type", "")
        if paper_type not in PAPER_TYPES:
            paper_type = "original_research"
        reading_strategy = data.get("reading_strategy", "")
        if reading_strategy not in READING_STRATEGIES:
            reading_strategy = "methods_first"
        return PaperProfile(
            paper_type=paper_type,
            paper_type_confidence=float(data.get("paper_type_confidence", 0.5)),
            research_problem=str(data.get("research_problem", "")),
            claimed_contribution=str(data.get("claimed_contribution", "")),
            reading_strategy=reading_strategy,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def parse_logic_graph(raw: str) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        required = ["nodes", "edges", "argument_flow"]
        if not all(k in data for k in required):
            return None
        return data
    except json.JSONDecodeError:
        return None


def parse_facts(raw: str, para_id: str) -> list[Fact]:
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        facts = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            required = ["para_id", "category", "statement"]
            if not all(k in item for k in required):
                continue
            category = item.get("category", "none")
            if category not in FACT_CATEGORIES:
                category = "none"
            quote = str(item.get("quote", ""))
            quote_truncated = item.get("quote_truncated", False)
            truncation_reason = item.get("truncation_reason")
            if not quote_truncated and count_words(quote) > 30:
                quote, quote_truncated, truncation_reason = truncate_quote(quote)
            fact_id = item.get("fact_id")
            if not fact_id:
                fact_id = generate_fact_id(para_id, idx)
            evidence = item.get("evidence_pointer", {})
            if not isinstance(evidence, dict):
                evidence = {}
            facts.append(Fact(
                fact_id=fact_id,
                para_id=str(item.get("para_id", para_id)),
                category=category,
                statement=str(item.get("statement", "")),
                quote=quote,
                evidence_pointer=evidence,
                quote_truncated=quote_truncated,
                truncation_reason=truncation_reason,
            ))
        return facts
    except (json.JSONDecodeError, TypeError):
        return []


def parse_themes(raw: str) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        if "themes" not in data:
            return None
        return data
    except json.JSONDecodeError:
        return None


def parse_synthesis(raw: str) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        required = ["executive_summary", "key_evidence_lines"]
        if not all(k in data for k in required):
            return None
        return data
    except json.JSONDecodeError:
        return None


def build_profile_prompt(paragraphs: list[dict[str, Any]], citations: list[dict[str, Any]]) -> str:
    title = ""
    abstract = ""
    for para in paragraphs[:10]:
        text = para.get("text", "")
        if len(text) > 100 and not title:
            if para.get("role") == "Heading" or text.isupper():
                title = text[:200]
        if "abstract" in text.lower()[:50]:
            abstract = text[:500]
    headings = []
    for para in paragraphs[:20]:
        if para.get("role") == "Heading":
            headings.append(para.get("text", "")[:100])
    prompt = f"Analyze this academic paper to determine its type, research problem, and optimal reading strategy. Please provide analysis in Chinese where appropriate.\n\nTitle: {title or 'Not detected'}\nAbstract: {abstract or 'Not detected'}\n\nSection Headings:\n"
    for h in headings[:10]:
        prompt += f"- {h}\n"
    prompt += "\nFirst few paragraphs:\n"
    for para in paragraphs[:5]:
        if para.get("role") == "Body":
            text = para.get("text", "")
            if text:
                prompt += f"- {text[:200]}...\n"
    prompt += """Return STRICT JSON ONLY:

{
  "paper_type": "<type>",
  "paper_type_confidence": <float 0-1>,
  "research_problem": "<string in Chinese preferred / 中文优先>",
  "claimed_contribution": "<string in Chinese preferred / 中文优先>",
  "reading_strategy": "<strategy>"
}

Paper Types: original_research, review, meta_analysis, case_report, methodology, commentary, guidelines
Reading Strategies: methods_first, evidence_synthesis, statistical_focus, narrative_flow, protocol_extraction

Please respond in Chinese where possible.

Return JSON only:"""
    return prompt


def build_logic_prompt(paragraphs: list[dict[str, Any]], profile: PaperProfile) -> str:
    sections_text = ""
    for para in paragraphs:
        role = para.get("role", "Body")
        section = para.get("section_path", [])
        text = para.get("text", "")
        para_id = para.get("para_id", "")
        section_str = ""
        if section and isinstance(section, list):
            section_str = f" [{' > '.join(section)}]"
        sections_text += f"\n[{para_id}]{section_str} ({role}): {text[:150]}..."
    prompt = f"""Construct an argument flow graph showing how claims and evidence connect. Please provide analysis in Chinese where appropriate.

## Paper Profile
- Type: {profile.paper_type}
- Reading Strategy: {profile.reading_strategy}
- Research Problem: {profile.research_problem}

## Paragraphs with IDs
{sections_text[:4000]}

Return STRICT JSON ONLY:

"""
    prompt += """{
  "nodes": [
    {
      "node_id": "<string>",
      "type": "<claim|evidence|method|limitation|future_work>",
      "text": "<brief description in Chinese preferred / 中文优先>",
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

Please respond in Chinese where possible.

Return JSON only:"""
    return prompt


def build_facts_prompt(paragraphs: list[dict[str, Any]], start_idx: int, batch_size: int = 10) -> tuple[str, list[str]]:
    batch = paragraphs[start_idx:start_idx + batch_size]
    para_ids = []
    prompt = "Extract atomic facts from these paragraphs with evidence pointers. Please provide analysis in Chinese where appropriate.\n\n## Paragraphs\n\n"
    for para in batch:
        para_id = para.get("para_id", "")
        para_ids.append(para_id)
        role = para.get("role", "Body")
        section = para.get("section_path", [])
        section_str = ""
        if section and isinstance(section, list):
            section_str = f" [{' > '.join(section)}]"
        text = para.get("text", "")
        page_span = para.get("page_span", {})
        page = page_span.get("start", 1)
        evidence = para.get("evidence_pointer", {})
        bbox = evidence.get("bbox_union", [0, 0, 0, 0])
        prompt += f"### Para {para_id}\n"
        prompt += f"Role: {role}{section_str}\n"
        prompt += f"Page: {page}, BBox: {bbox}\n"
        prompt += f"Text: {text}\n\n"
    prompt += """Return STRICT JSON ONLY:

[
  {
    "fact_id": "<string>",
    "para_id": "<para_id>",
    "category": "<category>",
    "statement": "<concise factual statement in Chinese preferred / 中文优先>",
    "quote": "<verbatim quote, max 30 words>",
    "evidence_pointer": {
      "page": <int>,
      "bbox": [x0, y0, x1, y1],
      "source_block_ids": [<block_id>, ...]
    }
  }
]

Categories: result, statistics, comparison, definition, mechanism, limitation, recommendation, background, none

Please respond in Chinese where possible.

Return JSON only:"""
    return prompt, para_ids


def build_themes_prompt(facts: list[Fact], profile: PaperProfile, logic_graph: dict[str, Any]) -> str:
    facts_text = ""
    for fact in facts:
        facts_text += f"- [{fact.fact_id}] {fact.category}: {fact.statement[:100]}...\n"
    nodes_text = ""
    for node in logic_graph.get("nodes", [])[:20]:
        nodes_text += f"- [{node.get('node_id')}] {node.get('type')}: {node.get('text', '')[:80]}...\n"
    prompt = f"""Group related facts into themes and identify cross-theme connections. Please provide analysis in Chinese where appropriate.

## Paper Profile
- Type: {profile.paper_type}
- Research Problem: {profile.research_problem}

## Logic Graph Nodes
{nodes_text}

## Facts to Group
{facts_text[:3000]}

Return STRICT JSON ONLY:

"""
    prompt += """{
  "themes": [
    {
      "theme_id": "<string>",
      "name": "<concise theme name in Chinese preferred / 中文优先>",
      "description": "<1-2 sentence summary in Chinese preferred / 中文优先>",
      "fact_ids": [<fact_id>, ...],
      "strength": "<strong|moderate|weak>",
      "evidence_quality": "<high|medium|low>"
    }
  ],
  "cross_theme_links": [],
  "contradictions": []
}

Please respond in Chinese where possible.

Return JSON only:"""
    return prompt


def build_synthesis_prompt(profile: PaperProfile, logic_graph: dict[str, Any], facts: list[Fact], themes: dict[str, Any], assets: list[dict[str, Any]]) -> str:
    facts_text = ""
    for fact in facts[:30]:
        facts_text += f"- [{fact.fact_id}] {fact.category}: {fact.statement[:100]}...\n"
    themes_text = ""
    for theme in themes.get("themes", []):
        themes_text += f"- [{theme.get('theme_id')}] {theme.get('name')}: {theme.get('description', '')[:80]}...\n"
    assets_text = ""
    for asset in assets[:20]:
        caption = asset.get('caption_text') or 'No caption'
        assets_text += f"- [{asset.get('asset_id')}] {asset.get('asset_type')}: {caption[:60]}...\n"
    prompt = f"""Generate executive summary and key evidence lines with traceability. Please provide analysis in Chinese where appropriate.

## Paper Profile
- Type: {profile.paper_type}
- Research Problem: {profile.research_problem}
- Contribution: {profile.claimed_contribution}

## Core Claims
"""
    for node_id in logic_graph.get("argument_flow", {}).get("core_claims", [])[:5]:
        for node in logic_graph.get("nodes", []):
            if node.get("node_id") == node_id:
                prompt += f"- {node.get('text', '')[:100]}...\n"
                break
    prompt += f"""
## Facts
{facts_text}

## Themes
{themes_text}

## Available Figures/Tables
{assets_text}

Return STRICT JSON ONLY:

"""
    prompt += """{
  "executive_summary": "<3-5 sentence overview in Chinese preferred / 中文优先>",
  "key_evidence_lines": [
    {
      "line_id": "<string>",
      "statement": "<key finding in Chinese preferred / 中文优先>",
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
      "render_mode": "<content_only|full_asset_embed>"
    }
  ]
}

Please respond in Chinese where possible.

Return JSON only:"""
    return prompt


def generate_fallback_profile() -> PaperProfile:
    return PaperProfile(
        paper_type="original_research",
        paper_type_confidence=0.3,
        research_problem="无法从可用文本确定研究问题 (Unable to determine research problem from available text)",
        claimed_contribution="无法从可用文本确定声称贡献 (Unable to determine claimed contribution from available text)",
        reading_strategy="methods_first",
    )


def generate_fallback_logic_graph() -> dict[str, Any]:
    return {"nodes": [], "edges": [], "argument_flow": {"premises": [], "core_claims": [], "conclusions": []}}


def generate_fallback_facts(paragraphs: list[dict[str, Any]]) -> list[Fact]:
    facts = []
    for para in paragraphs:
        para_id = para.get("para_id", "")
        if not para_id:
            continue
        page_span = para.get("page_span", {})
        evidence = para.get("evidence_pointer", {})
        fact = Fact(
            fact_id=generate_fact_id(para_id, 0),
            para_id=para_id,
            category="none",
            statement="此段落无可提取的原子事实 (No extractable atomic facts in this paragraph)",
            quote="",
            evidence_pointer={
                "page": page_span.get("start", 1),
                "bbox": evidence.get("bbox_union", [0, 0, 0, 0]),
                "source_block_ids": evidence.get("source_block_ids", []),
            },
        )
        facts.append(fact)
    return facts


def generate_fallback_themes() -> dict[str, Any]:
    return {"themes": [], "cross_theme_links": [], "contradictions": []}


def generate_fallback_synthesis(facts: list[Fact], assets: list[dict[str, Any]]) -> dict[str, Any]:
    key_lines = []
    for fact in facts[:10]:
        if fact.category != "none" and fact.statement:
            key_lines.append({
                "line_id": f"line_{fact.fact_id}",
                "statement": fact.statement[:200],
                "fact_ids": [fact.fact_id],
                "strength": "weak",
                "is_strong_claim": False,
            })
    slots = []
    for asset in assets[:5]:
        slots.append({
            "slot_id": f"slot_{asset.get('asset_id', 'unknown')}",
            "position_hint": "end_of_summary",
            "asset_ids": [asset.get("asset_id", "")],
            "render_mode": "content_only",
        })
    return {
        "executive_summary": "由于处理失败，无法生成摘要。请参阅各个事实了解详情。(Unable to generate summary due to processing failure. See individual facts for details.)",
        "key_evidence_lines": key_lines,
        "figure_table_slots": slots,
    }


def run_reading(
    run_dir: Path,
    manifest: Optional[Manifest] = None,
    inject_malformed_json: bool = False,
) -> tuple[int, int, int, int, int]:
    if manifest is None:
        manifest = load_manifest(run_dir)
    reading_dir = run_dir / "reading"
    reading_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    paragraphs_path = run_dir / "paragraphs" / "paragraphs.jsonl"
    paragraphs = load_paragraphs(paragraphs_path)
    cite_map_path = run_dir / "citations" / "cite_map.jsonl"
    citations = load_cite_map(cite_map_path)
    figure_index_path = run_dir / "figures_tables" / "figure_table_index.jsonl"
    assets = load_figure_table_index(figure_index_path)
    figure_links_path = run_dir / "figures_tables" / "figure_table_links.json"
    figure_links = load_figure_table_links(figure_links_path)
    _ = figure_links
    fault_events: list[FaultEvent] = []
    llm_model = os.environ.get("SILICONFLOW_READING_MODEL", SILICONFLOW_DEFAULT_MODEL)

    profile_prompt = build_profile_prompt(paragraphs, citations)
    if inject_malformed_json:
        raw_profile = "{ malformed json !!! "
        profile_meta: dict[str, Any] = {
            "model": llm_model,
            "endpoint": SILICONFLOW_ENDPOINT,
            "success": False,
            "error_type": "injected_malformed_json",
            "http_status": None,
            "prompt_chars": len(profile_prompt),
            "response_chars": len(raw_profile),
            "response_preview": raw_profile[:300],
        }
    else:
        raw_profile, profile_meta = call_siliconflow(profile_prompt)
    profile = parse_paper_profile(raw_profile)
    append_llm_call_event(qa_dir, LLMCallEvent(
        stage="reading",
        step="profile",
        model=str(profile_meta.get("model", llm_model)),
        endpoint=str(profile_meta.get("endpoint", SILICONFLOW_ENDPOINT)),
        success=bool(profile_meta.get("success", False)),
        error_type=str(profile_meta.get("error_type", "unknown")),
        http_status=profile_meta.get("http_status", None),
        prompt_chars=int(profile_meta.get("prompt_chars", len(profile_prompt))),
        response_chars=int(profile_meta.get("response_chars", len(raw_profile))),
        parse_success=profile is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_preview=str(profile_meta.get("response_preview", "")),
    ))
    if profile is None:
        profile = generate_fallback_profile()
        fault_events.append(FaultEvent(stage="reading", fault="profile-parse-failure", retry_attempts=0, fallback_used=True, status="degraded"))
    profile_path = reading_dir / "paper_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump({
            "paper_type": profile.paper_type,
            "paper_type_confidence": profile.paper_type_confidence,
            "research_problem": profile.research_problem,
            "claimed_contribution": profile.claimed_contribution,
            "reading_strategy": profile.reading_strategy,
        }, f, indent=2, ensure_ascii=False)

    logic_prompt = build_logic_prompt(paragraphs, profile)
    if inject_malformed_json:
        raw_logic = "{ malformed json !!! "
        logic_meta: dict[str, Any] = {
            "model": llm_model,
            "endpoint": SILICONFLOW_ENDPOINT,
            "success": False,
            "error_type": "injected_malformed_json",
            "http_status": None,
            "prompt_chars": len(logic_prompt),
            "response_chars": len(raw_logic),
            "response_preview": raw_logic[:300],
        }
    else:
        raw_logic, logic_meta = call_siliconflow(logic_prompt)
    logic_graph = parse_logic_graph(raw_logic)
    append_llm_call_event(qa_dir, LLMCallEvent(
        stage="reading",
        step="logic",
        model=str(logic_meta.get("model", llm_model)),
        endpoint=str(logic_meta.get("endpoint", SILICONFLOW_ENDPOINT)),
        success=bool(logic_meta.get("success", False)),
        error_type=str(logic_meta.get("error_type", "unknown")),
        http_status=logic_meta.get("http_status", None),
        prompt_chars=int(logic_meta.get("prompt_chars", len(logic_prompt))),
        response_chars=int(logic_meta.get("response_chars", len(raw_logic))),
        parse_success=logic_graph is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_preview=str(logic_meta.get("response_preview", "")),
    ))
    if logic_graph is None:
        logic_graph = generate_fallback_logic_graph()
        fault_events.append(FaultEvent(stage="reading", fault="logic-parse-failure", retry_attempts=0, fallback_used=True, status="degraded"))
    logic_path = reading_dir / "logic_graph.json"
    with open(logic_path, "w", encoding="utf-8") as f:
        json.dump(logic_graph, f, indent=2, ensure_ascii=False)

    facts: list[Fact] = []
    facts_path = reading_dir / "facts.jsonl"
    batch_size = 10
    total_paragraphs = len(paragraphs)
    for start_idx in range(0, total_paragraphs, batch_size):
        facts_prompt, para_ids = build_facts_prompt(paragraphs, start_idx, batch_size)
        if inject_malformed_json and start_idx == 0:
            raw_facts = "{ malformed json !!! "
            facts_meta: dict[str, Any] = {
                "model": llm_model,
                "endpoint": SILICONFLOW_ENDPOINT,
                "success": False,
                "error_type": "injected_malformed_json",
                "http_status": None,
                "prompt_chars": len(facts_prompt),
                "response_chars": len(raw_facts),
                "response_preview": raw_facts[:300],
            }
        else:
            raw_facts, facts_meta = call_siliconflow(facts_prompt)
        batch_facts = parse_facts(raw_facts, para_ids[0] if para_ids else "")
        append_llm_call_event(qa_dir, LLMCallEvent(
            stage="reading",
            step=f"facts_batch_{start_idx:05d}",
            model=str(facts_meta.get("model", llm_model)),
            endpoint=str(facts_meta.get("endpoint", SILICONFLOW_ENDPOINT)),
            success=bool(facts_meta.get("success", False)),
            error_type=str(facts_meta.get("error_type", "unknown")),
            http_status=facts_meta.get("http_status", None),
            prompt_chars=int(facts_meta.get("prompt_chars", len(facts_prompt))),
            response_chars=int(facts_meta.get("response_chars", len(raw_facts))),
            parse_success=bool(batch_facts),
            timestamp=datetime.now(timezone.utc).isoformat(),
            response_preview=str(facts_meta.get("response_preview", "")),
        ))
        if not batch_facts:
            fault_events.append(FaultEvent(stage="reading", fault="facts-parse-failure", retry_attempts=0, fallback_used=True, status="degraded"))
        if not batch_facts:
            for para_id in para_ids:
                para = next((p for p in paragraphs if p.get("para_id") == para_id), None)
                if para:
                    page_span = para.get("page_span", {})
                    evidence = para.get("evidence_pointer", {})
                    fallback_fact = Fact(
                        fact_id=generate_fact_id(para_id, 0),
                        para_id=para_id,
                        category="none",
                        statement="此段落无可提取的原子事实 (No extractable atomic facts in this paragraph)",
                        quote="",
                        evidence_pointer={
                            "page": page_span.get("start", 1),
                            "bbox": evidence.get("bbox_union", [0, 0, 0, 0]),
                            "source_block_ids": evidence.get("source_block_ids", []),
                        },
                    )
                    batch_facts.append(fallback_fact)
        facts.extend(batch_facts)
    if not facts:
        facts = generate_fallback_facts(paragraphs)
    with open(facts_path, "w", encoding="utf-8") as f:
        for fact in facts:
            record: dict[str, Any] = {
                "fact_id": fact.fact_id,
                "para_id": fact.para_id,
                "category": fact.category,
                "statement": fact.statement,
                "quote": fact.quote,
                "evidence_pointer": fact.evidence_pointer,
            }
            if fact.quote_truncated:
                record["quote_truncated"] = True
                record["truncation_reason"] = fact.truncation_reason
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    themes_prompt = build_themes_prompt(facts, profile, logic_graph)
    if inject_malformed_json:
        raw_themes = "{ malformed json !!! "
        themes_meta: dict[str, Any] = {
            "model": llm_model,
            "endpoint": SILICONFLOW_ENDPOINT,
            "success": False,
            "error_type": "injected_malformed_json",
            "http_status": None,
            "prompt_chars": len(themes_prompt),
            "response_chars": len(raw_themes),
            "response_preview": raw_themes[:300],
        }
    else:
        raw_themes, themes_meta = call_siliconflow(themes_prompt)
    themes = parse_themes(raw_themes)
    append_llm_call_event(qa_dir, LLMCallEvent(
        stage="reading",
        step="themes",
        model=str(themes_meta.get("model", llm_model)),
        endpoint=str(themes_meta.get("endpoint", SILICONFLOW_ENDPOINT)),
        success=bool(themes_meta.get("success", False)),
        error_type=str(themes_meta.get("error_type", "unknown")),
        http_status=themes_meta.get("http_status", None),
        prompt_chars=int(themes_meta.get("prompt_chars", len(themes_prompt))),
        response_chars=int(themes_meta.get("response_chars", len(raw_themes))),
        parse_success=themes is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_preview=str(themes_meta.get("response_preview", "")),
    ))
    if themes is None:
        themes = generate_fallback_themes()
        fault_events.append(FaultEvent(stage="reading", fault="themes-parse-failure", retry_attempts=0, fallback_used=True, status="degraded"))
    themes_path = reading_dir / "themes.json"
    with open(themes_path, "w", encoding="utf-8") as f:
        json.dump(themes, f, indent=2, ensure_ascii=False)

    synthesis_prompt = build_synthesis_prompt(profile, logic_graph, facts, themes, assets)
    if inject_malformed_json:
        raw_synthesis = "{ malformed json !!! "
        synthesis_meta: dict[str, Any] = {
            "model": llm_model,
            "endpoint": SILICONFLOW_ENDPOINT,
            "success": False,
            "error_type": "injected_malformed_json",
            "http_status": None,
            "prompt_chars": len(synthesis_prompt),
            "response_chars": len(raw_synthesis),
            "response_preview": raw_synthesis[:300],
        }
    else:
        raw_synthesis, synthesis_meta = call_siliconflow(synthesis_prompt)
    synthesis = parse_synthesis(raw_synthesis)
    append_llm_call_event(qa_dir, LLMCallEvent(
        stage="reading",
        step="synthesis",
        model=str(synthesis_meta.get("model", llm_model)),
        endpoint=str(synthesis_meta.get("endpoint", SILICONFLOW_ENDPOINT)),
        success=bool(synthesis_meta.get("success", False)),
        error_type=str(synthesis_meta.get("error_type", "unknown")),
        http_status=synthesis_meta.get("http_status", None),
        prompt_chars=int(synthesis_meta.get("prompt_chars", len(synthesis_prompt))),
        response_chars=int(synthesis_meta.get("response_chars", len(raw_synthesis))),
        parse_success=synthesis is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_preview=str(synthesis_meta.get("response_preview", "")),
    ))
    if synthesis is None:
        synthesis = generate_fallback_synthesis(facts, assets)
        fault_events.append(FaultEvent(stage="reading", fault="synthesis-parse-failure", retry_attempts=0, fallback_used=True, status="degraded"))
    fact_ids_set = {f.fact_id for f in facts}
    valid_lines = []
    for line in synthesis.get("key_evidence_lines", []):
        valid_fact_ids = [fid for fid in line.get("fact_ids", []) if fid in fact_ids_set]
        if valid_fact_ids:
            line["fact_ids"] = valid_fact_ids
            valid_lines.append(line)
        elif facts:
            line["fact_ids"] = [facts[0].fact_id]
            valid_lines.append(line)
    synthesis["key_evidence_lines"] = valid_lines
    synthesis_path = reading_dir / "synthesis.json"
    with open(synthesis_path, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2, ensure_ascii=False)

    if fault_events:
        fp = qa_dir / "fault_injection.json"
        existing: list[dict[str, Any]] = []
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    existing = d if isinstance(d, list) else []
            except (json.JSONDecodeError, IOError):
                pass
        all_events = existing + [asdict(e) for e in fault_events]
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(all_events, f, indent=2, ensure_ascii=False)

    return (
        len(facts),
        len(themes.get("themes", [])),
        len(synthesis.get("key_evidence_lines", [])),
        len(synthesis.get("figure_table_slots", [])),
        len(fault_events),
    )
