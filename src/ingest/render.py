"""Obsidian Renderer - Paper-type-aware literature note generation.

Generates reader-friendly Obsidian notes from reading pipeline artifacts.
Supports paper-type-specific templates and maintains traceability via fact_ids.

Output: obsidian/{doc_id}.md
"""

import json
import re
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .manifest import Manifest


REVIEW_TYPES = frozenset({"review", "meta_analysis"})
ORIGINAL_RESEARCH_TYPES = frozenset({"original_research", "case_report", "methodology"})


def load_summary_status(run_dir: Path, doc_id: str) -> dict[str, Any]:
    qa_dir = run_dir / "qa"
    summary_status_path = qa_dir / "summary_status.json"
    reason_codes: list[str] = []

    if summary_status_path.exists():
        data = load_json_file(summary_status_path)
        if isinstance(data, dict):
            status = str(data.get("status", "")).strip().lower()
            if status in {"full", "degraded"}:
                if not isinstance(data.get("reason_codes"), list):
                    data["reason_codes"] = []
                if not isinstance(data.get("metrics"), dict):
                    data["metrics"] = {}
                data.setdefault("doc_id", doc_id)
                data.setdefault("generated_at", "")
                return data
            reason_codes.append("invalid_summary_status")
        else:
            reason_codes.append("invalid_summary_status")
    else:
        reason_codes.append("missing_summary_status")

    if not (run_dir / "reading" / "facts.jsonl").exists():
        reason_codes.append("missing_facts")
    if not (run_dir / "reading" / "themes.json").exists():
        reason_codes.append("missing_themes")
    if not (run_dir / "reading" / "synthesis.json").exists():
        reason_codes.append("missing_synthesis")

    return {
        "doc_id": doc_id,
        "status": "degraded",
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "metrics": {
            "narrative_fact_count": 0,
            "themes_count": 0,
            "synthesis_present": False,
            "narrative_evidence_ratio": 0.0,
        },
        "generated_at": "",
    }


def resolve_api_key() -> str:
    for key_name in ("SILICONFLOW_API_KEY", "SF_API_KEY", "SILICONFLOW_TOKEN"):
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    return ""


def load_json_file(file_path: Path) -> Optional[dict[str, Any]]:
    """Load a JSON file, returning None if it doesn't exist or is invalid."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning empty list if it doesn't exist."""
    results = []
    if not file_path.exists():
        return results
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass
    return results


def load_paper_profile(run_dir: Path) -> dict[str, Any]:
    """Load paper profile from reading/paper_profile.json."""
    profile_path = run_dir / "reading" / "paper_profile.json"
    data = load_json_file(profile_path)
    if data is None:
        return {
            "paper_type": "unknown",
            "paper_type_confidence": 0.0,
            "research_problem": "无法加载论文档案",
            "claimed_contribution": "无法加载论文档案",
            "reading_strategy": "methods_first",
        }
    return data


def load_logic_graph(run_dir: Path) -> dict[str, Any]:
    """Load logic graph from reading/logic_graph.json."""
    path = run_dir / "reading" / "logic_graph.json"
    data = load_json_file(path)
    if data is None:
        return {"nodes": [], "edges": [], "argument_flow": {"premises": [], "core_claims": [], "conclusions": []}}
    return data


def load_themes(run_dir: Path) -> dict[str, Any]:
    """Load themes from reading/themes.json."""
    path = run_dir / "reading" / "themes.json"
    data = load_json_file(path)
    if data is None:
        return {"themes": [], "cross_theme_links": [], "contradictions": []}
    return data


def load_synthesis(run_dir: Path) -> dict[str, Any]:
    """Load synthesis from reading/synthesis.json."""
    path = run_dir / "reading" / "synthesis.json"
    data = load_json_file(path)
    if data is None:
        return {
            "executive_summary": "无法加载综合分析",
            "key_evidence_lines": [],
            "figure_table_slots": [],
        }
    return data


def load_facts(run_dir: Path) -> list[dict[str, Any]]:
    """Load facts from reading/facts.jsonl."""
    return load_jsonl(run_dir / "reading" / "facts.jsonl")


def load_paragraphs(run_dir: Path) -> list[dict[str, Any]]:
    """Load paragraphs from paragraphs/paragraphs.jsonl."""
    return load_jsonl(run_dir / "paragraphs" / "paragraphs.jsonl")


def load_cite_map(run_dir: Path) -> list[dict[str, Any]]:
    """Load citation map from citations/cite_map.jsonl."""
    return load_jsonl(run_dir / "citations" / "cite_map.jsonl")


def load_reference_catalog(run_dir: Path) -> list[dict[str, Any]]:
    merged_path = run_dir / "refs" / "references_merged.jsonl"
    if merged_path.exists():
        return load_jsonl(merged_path)
    return load_jsonl(run_dir / "citations" / "reference_catalog.jsonl")


def load_pdf_reference_catalog(run_dir: Path) -> list[dict[str, Any]]:
    return load_jsonl(run_dir / "citations" / "reference_catalog.jsonl")


def load_figure_table_index(run_dir: Path) -> list[dict[str, Any]]:
    """Load figure/table index from figures_tables/figure_table_index.jsonl."""
    return load_jsonl(run_dir / "figures_tables" / "figure_table_index.jsonl")


def format_authors(authors: list[str], max_display: int = 3) -> str:
    """Format authors for display, using 'et al.' if too many."""
    if not authors:
        return "未知作者"
    if len(authors) <= max_display:
        return ", ".join(authors)
    return f"{authors[0]} 等 ({', '.join(authors[:max_display])} et al.)"


def format_reference_entry(ref: dict[str, Any]) -> str:
    """Format a reference catalog entry for readable display."""
    authors = format_authors(ref.get("authors", []))
    year = ref.get("year", "n.d.")
    title = str(ref.get("title", "")).strip()
    venue = ref.get("venue", "")
    raw_text = str(ref.get("raw_text", "")).strip()

    weak_title = (not title) or len(title.split()) <= 2 or title.lower() in {"无标题", "unknown", "n/a"}
    weak_authors = authors in {"未知作者", "unknown"}

    if weak_title and raw_text:
        fallback = raw_text[:220]
        if len(raw_text) > 220:
            fallback += "..."
        return fallback

    if weak_authors and raw_text:
        fallback = raw_text[:220]
        if len(raw_text) > 220:
            fallback += "..."
        return fallback

    if not title:
        title = "Untitled"
    
    parts = [f"**{authors}** ({year})"]
    if title:
        parts.append(f"*{title}*")
    if venue:
        parts.append(venue)
    
    doi = ref.get("doi")
    pmid = ref.get("pmid")
    if doi:
        parts.append(f"DOI: {doi}")
    if pmid:
        parts.append(f"PMID: {pmid}")
    
    return ". ".join(parts)


def build_citations_section(cite_map: list[dict[str, Any]], reference_catalog: list[dict[str, Any]], run_dir: Path | None = None) -> str:
    if not reference_catalog:
        return "## 引用文献 (Citations)\n\n未能提取参考文献目录。\n"
    
    ref_lookup: dict[str, dict[str, Any]] = {}
    doi_to_key: dict[str, str] = {}
    pmid_to_key: dict[str, str] = {}
    bib_to_doi: dict[str, str] = {}
    
    if run_dir:
        import re
        def normalize_title(s):
            if not s:
                return ''
            s = s.lower()
            s = re.sub(r'[^\w\s]', '', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s[:50]
        
        title_to_doi: dict[str, str] = {}
        author_year_to_doi: dict[str, str] = {}
        for ref in reference_catalog:
            title = normalize_title(ref.get("title", ""))
            doi_value = str(ref.get("doi", "") or "").strip()
            if title and doi_value:
                title_to_doi[title] = doi_value
            authors = ref.get("authors", [])
            year = ref.get("year")
            if authors and year and doi_value:
                first_author_surname = authors[0].lower().split()[-1] if authors[0] else ""
                if first_author_surname:
                    key = f"{first_author_surname}_{year}"
                    if key not in author_year_to_doi:
                        author_year_to_doi[key] = doi_value
        
        pdf_catalog = load_pdf_reference_catalog(run_dir)
        for ref in pdf_catalog:
            ref_key = ref.get("ref_key")
            if ref_key and ref_key.startswith("bib:"):
                raw = ref.get("raw_text", "")
                raw_norm = normalize_title(raw[:100])
                if raw_norm in title_to_doi:
                    bib_to_doi[ref_key] = title_to_doi[raw_norm]
                else:
                    # Try author+year match from bib key
                    match = re.match(r"bib:([a-z]+)_(\d{4})_", ref_key)
                    if match:
                        author = match.group(1)
                        year = match.group(2)
                        auth_key = f"{author}_{year}"
                        if auth_key in author_year_to_doi:
                            bib_to_doi[ref_key] = author_year_to_doi[auth_key]
    
    for ref in reference_catalog:
        ref_key = ref.get("ref_key")
        if not ref_key:
            if ref.get("doi"):
                ref_key = f"doi:{ref.get('doi')}"
            elif ref.get("pmid"):
                ref_key = f"pmid:{ref.get('pmid')}"
            elif ref.get("title"):
                ref_key = f"title:{hash(ref.get('title')) % 1000000:06x}"
        if ref_key:
            ref_lookup[ref_key] = ref
            doi_value = str(ref.get("doi", "") or "").strip()
            if doi_value:
                doi_to_key[doi_value.lower()] = ref_key
            if ref.get("pmid"):
                pmid_to_key[str(ref.get("pmid"))] = ref_key
    
    if not ref_lookup:
        return "## 引用文献 (Citations)\n\n未能提取参考文献目录。\n"
    
    mapped_refs = set()
    for cm in cite_map:
        ref_key = cm.get("mapped_ref_key")
        if ref_key:
            if ref_key in ref_lookup:
                mapped_refs.add(ref_key)
            elif ref_key.startswith("pmid:"):
                pmid = ref_key[5:]
                if pmid in pmid_to_key:
                    mapped_refs.add(pmid_to_key[pmid])
            elif ref_key.startswith("bib:"):
                doi = bib_to_doi.get(ref_key)
                if doi:
                    doi_key = f"doi:{doi}"
                    if doi_key in ref_lookup:
                        mapped_refs.add(doi_key)
    
    def sort_key(ref_key: str) -> tuple[int, int, str]:
        ref = ref_lookup.get(ref_key, {})
        year_val = ref.get("year", 0)
        if isinstance(year_val, str):
            try:
                year = int(year_val)
            except (ValueError, TypeError):
                year = 0
        else:
            year = year_val if isinstance(year_val, int) else 0
        prefix = 0 if ref_key.startswith(("doi:", "pmid:")) else 1
        return (prefix, year, ref_key)
    
    sorted_catalog = sorted(ref_lookup.keys(), key=sort_key)
    
    lines = ["## 引用文献 (Citations)\n"]
    lines.append(f"参考文献目录共 {len(sorted_catalog)} 篇；文内锚点成功映射 {len(mapped_refs)} 篇。\n")
    
    for i, ref_key in enumerate(sorted_catalog, start=1):
        ref = ref_lookup.get(ref_key, {})
        formatted = format_reference_entry(ref)
        marker = "[mapped]" if ref_key in mapped_refs else "[catalog]"
        lines.append(f"{i}. {marker} {formatted}")
    
    return "\n".join(lines)


def build_facts_lookup(facts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a lookup from fact_id to fact data."""
    return {f.get("fact_id", ""): f for f in facts if f.get("fact_id")}


def format_fact_link(fact_id: str, fact_lookup: dict[str, dict[str, Any]]) -> str:
    """Format a fact link with category badge."""
    fact = fact_lookup.get(fact_id, {})
    category = fact.get("category", "unknown")
    category_zh = {
        "result": "结果",
        "statistics": "统计",
        "comparison": "对比",
        "definition": "定义",
        "mechanism": "机制",
        "limitation": "局限",
        "recommendation": "建议",
        "background": "背景",
        "none": "其他",
    }.get(category, category)
    
    return f"[[#{fact_id}|{fact_id}]] [^^{category_zh}]"


def _fact_category_cn(category: str) -> str:
    mapping = {
        "result": "研究结果",
        "statistics": "统计证据",
        "comparison": "对比结论",
        "definition": "概念定义",
        "mechanism": "机制解释",
        "limitation": "研究局限",
        "recommendation": "实践建议",
        "background": "背景信息",
        "none": "一般证据",
    }
    return mapping.get(category, "一般证据")


def summarize_key_line_cn(
    statement: str,
    fact_ids: list[str],
    fact_lookup: dict[str, dict[str, Any]],
) -> str:
    first_fact = fact_lookup.get(fact_ids[0], {}) if fact_ids else {}
    category = str(first_fact.get("category", "none"))
    category_cn = _fact_category_cn(category)
    evidence = first_fact.get("evidence_pointer", {}) if isinstance(first_fact, dict) else {}
    page = evidence.get("page", "未知") if isinstance(evidence, dict) else "未知"

    clean = " ".join(str(statement).split())
    has_numeric = bool(re.search(r"\b\d+(?:\.\d+)?%?\b", clean))
    has_compare = bool(re.search(r"\b(vs\.?|versus|compared|higher|lower|increase|decrease)\b", clean.lower()))

    if has_numeric and has_compare:
        return f"该条属于{category_cn}，包含定量对比信息，证据页码为第{page}页。"
    if has_numeric:
        return f"该条属于{category_cn}，包含定量结果，证据页码为第{page}页。"
    if has_compare:
        return f"该条属于{category_cn}，体现组间或时序比较，证据页码为第{page}页。"
    return f"该条属于{category_cn}，用于支撑核心结论，证据页码为第{page}页。"


def normalize_statement_for_reader(statement: str, max_len: int = 220) -> str:
    text = " ".join(str(statement).replace("\u00ad", "").split())
    text = text.replace("•", "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text


def build_reader_digest(
    key_lines: list[dict[str, Any]],
    fact_lookup: dict[str, dict[str, Any]],
    max_items: int = 5,
) -> list[str]:
    digest_lines: list[str] = []
    if not key_lines:
        return digest_lines

    selected_items: list[tuple[str, str, list[str]]] = []
    for line in key_lines:
        if len(selected_items) >= max_items:
            break
        statement = normalize_statement_for_reader(str(line.get("statement", "")))
        if not statement or len(statement.split()) < 6:
            continue
        fact_ids = [str(fid) for fid in line.get("fact_ids", [])[:2]]
        narrative = summarize_key_line_cn(statement, fact_ids, fact_lookup)
        selected_items.append((statement, narrative, fact_ids))

    translations = translate_statements_to_chinese([s for s, _, _ in selected_items])

    for idx, (statement, narrative, fact_ids) in enumerate(selected_items):
        fact_links = " ".join(format_fact_link(fid, fact_lookup) for fid in fact_ids)
        translated = translations[idx] if idx < len(translations) else ""
        digest_lines.append(f"- {(translated or narrative)} {fact_links}".strip())
        digest_lines.append(f"  - 具体内容 (EN): {statement}")
    return digest_lines


def build_author_flow_sections(
    paragraphs: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    fact_lookup: dict[str, dict[str, Any]],
    max_sections: int = 18,
) -> list[str]:
    para_to_section: dict[str, str] = {}
    section_first_order: dict[str, tuple[int, int]] = {}
    section_facts: dict[str, list[dict[str, Any]]] = {}
    current_heading = "导言"

    def _is_heading_like(text: str) -> bool:
        t = " ".join(text.split()).strip()
        if len(t) < 4 or len(t) > 80:
            return False
        low = t.lower()
        if low.startswith(("fig.", "figure", "table", "check for updates")):
            return False
        if any(ch in t for ch in ".;:"):
            return False
        words = t.split()
        if len(words) > 8:
            return False
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]
        if not alpha_words:
            return False
        titleish = sum(1 for w in alpha_words if w[:1].isupper())
        return (titleish / len(alpha_words)) >= 0.6

    def _para_sort_key(para: dict[str, Any]) -> tuple[int, float, float]:
        page = int(para.get("page_span", {}).get("start", 1) or 1)
        bbox = para.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])
        if not isinstance(bbox, list) or len(bbox) < 4:
            bbox = [0, 0, 0, 0]
        return (page, float(bbox[1]), float(bbox[0]))

    ordered_paragraphs = sorted(paragraphs, key=_para_sort_key)

    for idx, para in enumerate(ordered_paragraphs):
        para_id = str(para.get("para_id", "") or "")
        if not para_id:
            continue
        role = str(para.get("role", "") or "")
        text = " ".join(str(para.get("text", "") or "").split())
        section_path = para.get("section_path")
        if isinstance(section_path, list) and section_path:
            section_name = " > ".join(str(x) for x in section_path if str(x).strip())
            if section_name:
                current_heading = section_name
        else:
            if role == "Heading" or _is_heading_like(text):
                current_heading = text[:120]
            section_name = current_heading or "未命名章节"
        page = int(para.get("page_span", {}).get("start", 1) or 1)
        para_to_section[para_id] = section_name
        if section_name not in section_first_order:
            section_first_order[section_name] = (page, idx)

    for fact in facts:
        para_id = str(fact.get("para_id", "") or "")
        section = para_to_section.get(para_id)
        if not section:
            continue
        section_facts.setdefault(section, []).append(fact)

    ordered_sections = sorted(section_first_order.items(), key=lambda kv: kv[1])
    lines: list[str] = []
    if not ordered_sections:
        return lines

    lines.append("## 按作者写作逻辑精读 (Follow Author's Flow)\n")
    lines.append(f"共识别 {len(ordered_sections)} 个章节，以下展示前 {min(max_sections, len(ordered_sections))} 个：")
    lines.append("")

    for section_name, _ in ordered_sections[:max_sections]:
        lines.append(f"### {section_name}")
        facts_in_section = section_facts.get(section_name, [])
        if not facts_in_section:
            lines.append("- 本节暂无可用结构化证据，建议回看原文段落。")
            continue

        strong = [f for f in facts_in_section if str(f.get("category", "")) in {"result", "statistics", "comparison"}]
        display_facts = strong[:3] if strong else facts_in_section[:3]
        lines.append(f"- 本节提取到 {len(facts_in_section)} 条证据，优先展示关键 {len(display_facts)} 条：")
        for fact in display_facts:
            fact_id = str(fact.get("fact_id", "") or "")
            statement = normalize_statement_for_reader(str(fact.get("statement", "")), max_len=200)
            if not statement:
                continue
            page = fact.get("evidence_pointer", {}).get("page", "?")
            link = format_fact_link(fact_id, fact_lookup) if fact_id else ""
            lines.append(f"- p.{page}: {statement} {link}".strip())
        lines.append("")

    return lines


def translate_statements_to_chinese(statements: list[str]) -> list[str]:
    if not statements:
        return []

    api_key = resolve_api_key()
    if not api_key:
        return []

    model = os.environ.get("SILICONFLOW_READING_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(statements)])
    prompt = (
        "请把下面英文证据句改写成中文读者可直接理解的结论句。"
        "除专业术语外尽量使用中文；保留关键数字和比较关系。"
        "只输出JSON数组，例如 [\"...\",\"...\"]，长度与输入一致。\n"
        f"输入:\n{numbered}"
    )

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1200,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = str(body.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
            if not content:
                return []
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            start = content.find("[")
            end = content.rfind("]")
            if start >= 0 and end >= start:
                content = content[start:end + 1]
            arr = json.loads(content)
            if not isinstance(arr, list):
                return []
            return [str(x).strip() for x in arr]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError):
        return []


def render_original_research(
    profile: dict[str, Any],
    logic_graph: dict[str, Any],
    synthesis: dict[str, Any],
    facts: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    figure_index: list[dict[str, Any]],
) -> list[str]:
    """Render content for original research papers."""
    lines = []
    fact_lookup = build_facts_lookup(facts)
    
    lines.append("## 研究问题与假设 (Research Question & Hypothesis)\n")
    research_problem = profile.get("research_problem", "未明确")
    lines.append(f"**研究问题**: {research_problem}\n")
    
    hypothesis_nodes = [
        n for n in logic_graph.get("nodes", [])
        if n.get("type") in {"claim", "hypothesis"}
    ]
    if hypothesis_nodes:
        lines.append("\n**假设**:")
        for node in hypothesis_nodes[:3]:
            node_text = node.get("text", "")[:150]
            if node_text:
                lines.append(f"- {node_text}")
    lines.append("")
    
    lines.append("## 研究方法 (Methods)\n")
    method_nodes = [
        n for n in logic_graph.get("nodes", [])
        if n.get("type") == "method"
    ]
    if method_nodes:
        for node in method_nodes[:5]:
            node_text = node.get("text", "")[:200]
            if node_text:
                lines.append(f"- {node_text}")
    else:
        lines.append("方法详情见原文。")
    lines.append("")
    
    lines.append("## 一眼看懂这篇文献 (Reader Digest)\n")
    key_lines = synthesis.get("key_evidence_lines", [])
    digest = build_reader_digest(key_lines, fact_lookup, max_items=5)
    if digest:
        lines.extend(digest)
        lines.append("")

    flow_lines = build_author_flow_sections(paragraphs, facts, fact_lookup, max_sections=16)
    if flow_lines:
        lines.extend(flow_lines)
        lines.append("")

    lines.append("## 详细证据解读 (Detailed Findings)\n")
    if key_lines:
        for i, line in enumerate(key_lines[:8], start=1):
            statement = normalize_statement_for_reader(str(line.get("statement", "")), max_len=240)
            fact_ids = line.get("fact_ids", [])
            fact_links = " ".join(
                format_fact_link(fid, fact_lookup) for fid in fact_ids[:2]
            )
            strength = line.get("strength", "unknown")
            strength_badge = {
                "strong": "🟢",
                "moderate": "🟡",
                "weak": "🔴",
            }.get(strength, "⚪")
            narrative = summarize_key_line_cn(statement, fact_ids, fact_lookup)
            lines.append(f"- {strength_badge} 结论{i}：{narrative} {fact_links}")
            if statement:
                lines.append(f"  - 具体内容 (EN): {statement}")
    else:
        lines.append("无提取的关键发现。")
    lines.append("")
    
    lines.append("## 研究局限 (Limitations)\n")
    limitation_nodes = [
        n for n in logic_graph.get("nodes", [])
        if n.get("type") == "limitation"
    ]
    if limitation_nodes:
        for node in limitation_nodes[:5]:
            node_text = node.get("text", "")[:200]
            if node_text:
                lines.append(f"- {node_text}")
    else:
        limit_facts = [f for f in facts if f.get("category") == "limitation"]
        for fact in limit_facts[:3]:
            lines.append(f"- {fact.get('statement', '')[:180]}")
    lines.append("")
    
    lines.append("## 研究意义 (Implications)\n")
    rec_facts = [f for f in facts if f.get("category") == "recommendation"]
    if rec_facts:
        for fact in rec_facts[:5]:
            lines.append(f"- {fact.get('statement', '')[:180]}")
    else:
        contribution = profile.get("claimed_contribution", "")
        if contribution:
            lines.append(f"**声称贡献**: {contribution}")
        else:
            lines.append("详见原文。")
    lines.append("")
    
    return lines


def render_review_meta_analysis(
    profile: dict[str, Any],
    synthesis: dict[str, Any],
    themes: dict[str, Any],
    facts: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    figure_index: list[dict[str, Any]],
) -> list[str]:
    """Render content for review and meta-analysis papers."""
    lines = []
    fact_lookup = build_facts_lookup(facts)
    
    lines.append("## 研究范围 (Scope)\n")
    research_problem = profile.get("research_problem", "未明确")
    lines.append(f"**综述范围**: {research_problem}\n")
    
    summary = synthesis.get("executive_summary", "")
    if summary:
        lines.append(f"\n{summary[:500]}")
    lines.append("")
    
    lines.append("## 一眼看懂这篇综述 (Reader Digest)\n")
    key_lines = synthesis.get("key_evidence_lines", [])
    digest = build_reader_digest(key_lines, fact_lookup, max_items=5)
    if digest:
        lines.extend(digest)
        lines.append("")

    flow_lines = build_author_flow_sections(paragraphs, facts, fact_lookup, max_sections=20)
    if flow_lines:
        lines.extend(flow_lines)
        lines.append("")

    lines.append("## 面向读者的证据图谱 (Reader Evidence Map)\n")
    themes_list = themes.get("themes", [])
    if themes_list:
        lines.append(f"共识别 {len(themes_list)} 个主题：\n")
        for theme in themes_list[:8]:
            theme_name = theme.get("name", "未命名")
            theme_desc = theme.get("description", "")[:100]
            fact_ids = theme.get("fact_ids", [])
            strength = theme.get("strength", "moderate")
            strength_badge = {
                "strong": "🟢",
                "moderate": "🟡",
                "weak": "🔴",
            }.get(strength, "⚪")
            lines.append(f"- **{strength_badge} {theme_name}**: {theme_desc}")
            if fact_ids:
                fact_refs = " ".join(
                    format_fact_link(fid, fact_lookup) for fid in fact_ids[:2]
                )
                first_fact = fact_lookup.get(fact_ids[0], {}) if fact_ids else {}
                statement = str(first_fact.get("statement", ""))[:140]
                lines.append(f"  - 中文解读：{summarize_key_line_cn(statement, fact_ids, fact_lookup)}")
                lines.append(f"  - 证据: {fact_refs}")
                if statement:
                    lines.append(f"  - 具体内容 (EN): {normalize_statement_for_reader(statement)}")
    else:
        lines.append("未提取到主题分析。")
    lines.append("")
    
    lines.append("## 共识与分歧 (Consensus & Disagreement)\n")
    contradictions = themes.get("contradictions", [])
    if contradictions:
        lines.append("**存在争议的领域**:\n")
        for contr in contradictions[:5]:
            lines.append(f"- {contr.get('description', '')[:150]}")
    else:
        lines.append("未发现明确分歧。")
    lines.append("")
    
    lines.append("## 实践意义 (Practical Implications)\n")
    rec_facts = [f for f in facts if f.get("category") == "recommendation"]
    if rec_facts:
        for fact in rec_facts[:8]:
            statement = fact.get("statement", "")[:180]
            fact_link = format_fact_link(fact.get("fact_id", ""), fact_lookup)
            lines.append(f"- {statement} {fact_link}")
    else:
        contribution = profile.get("claimed_contribution", "")
        if contribution:
            lines.append(f"**主要贡献**: {contribution}")
        else:
            lines.append("详见原文。")
    lines.append("")
    
    return lines


def render_generic_fallback(
    profile: dict[str, Any],
    synthesis: dict[str, Any],
    facts: list[dict[str, Any]],
) -> list[str]:
    """Render generic content for unknown paper types."""
    lines = []
    fact_lookup = build_facts_lookup(facts)
    
    lines.append("## 执行摘要 (Executive Summary)\n")
    summary = synthesis.get("executive_summary", "无摘要可用。")
    lines.append(f"{summary[:800]}\n")
    
    lines.append("## 一眼看懂核心内容 (Reader Digest)\n")
    key_lines = synthesis.get("key_evidence_lines", [])
    digest = build_reader_digest(key_lines, fact_lookup, max_items=5)
    if digest:
        lines.extend(digest)
        lines.append("")

    lines.append("## 面向读者的关键结论 (Reader Key Findings)\n")
    if key_lines:
        for i, line in enumerate(key_lines[:10], start=1):
            statement = line.get("statement", "")[:180]
            fact_ids = line.get("fact_ids", [])
            fact_links = " ".join(
                format_fact_link(fid, fact_lookup) for fid in fact_ids[:2]
            )
            narrative = summarize_key_line_cn(statement, fact_ids, fact_lookup)
            lines.append(f"- 结论{i}：{narrative} {fact_links}")
            if statement:
                lines.append(f"  - 原始证据片段 (EN): {statement}")
    else:
        lines.append("无提取的关键发现。")
    lines.append("")
    
    return lines


def render_degraded_summary(
    profile: dict[str, Any],
    synthesis: dict[str, Any],
    facts: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    summary_status: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    fact_lookup = build_facts_lookup(facts)
    reason_codes = [str(item) for item in summary_status.get("reason_codes", []) if str(item)]

    lines.append("## 摘要质量状态 (Summary Quality Status)\n")
    lines.append("- 状态: degraded")
    if reason_codes:
        lines.append(f"- 原因代码: {', '.join(reason_codes)}")
    metrics = summary_status.get("metrics", {}) if isinstance(summary_status.get("metrics", {}), dict) else {}
    lines.append(
        "- 指标: narrative_facts={0}, themes={1}, synthesis_present={2}, narrative_evidence_ratio={3}".format(
            metrics.get("narrative_fact_count", 0),
            metrics.get("themes_count", 0),
            metrics.get("synthesis_present", False),
            metrics.get("narrative_evidence_ratio", 0.0),
        )
    )
    lines.append("")

    lines.append("## 降级摘要 (Degraded Summary)\n")
    executive_summary = normalize_statement_for_reader(str(synthesis.get("executive_summary", "")), max_len=600)
    if executive_summary:
        lines.append(executive_summary)
    else:
        candidate_facts = [
            fact for fact in facts
            if str(fact.get("statement", "")).strip()
        ]
        if not candidate_facts:
            para_emitted = 0
            for para in paragraphs:
                if str(para.get("role", "")) != "Body":
                    continue
                text = normalize_statement_for_reader(str(para.get("text", "")), max_len=200)
                if text and len(text.split()) >= 8:
                    lines.append(f"- {text}")
                    para_emitted += 1
                    if para_emitted >= 4:
                        break
        else:
            preferred = [
                fact for fact in candidate_facts
                if str(fact.get("category", "")) in {"result", "statistics", "comparison"}
            ]
            chosen = preferred[:4] if preferred else candidate_facts[:4]
            for fact in chosen:
                statement = normalize_statement_for_reader(str(fact.get("statement", "")), max_len=200)
                fact_id = str(fact.get("fact_id", ""))
                fact_link = format_fact_link(fact_id, fact_lookup) if fact_id else ""
                lines.append(f"- {statement} {fact_link}".strip())
    lines.append("")

    lines.append("## 可用证据要点 (Available Evidence)\n")
    key_lines = synthesis.get("key_evidence_lines", []) if isinstance(synthesis.get("key_evidence_lines", []), list) else []
    emitted = 0
    for line in key_lines:
        if not isinstance(line, dict):
            continue
        statement = normalize_statement_for_reader(str(line.get("statement", "")), max_len=220)
        if not statement:
            continue
        fact_ids = [str(fid) for fid in line.get("fact_ids", [])[:2]]
        links = " ".join(format_fact_link(fid, fact_lookup) for fid in fact_ids if fid)
        lines.append(f"- {statement} {links}".strip())
        emitted += 1
        if emitted >= 6:
            break

    if emitted == 0:
        for fact in facts[:6]:
            statement = normalize_statement_for_reader(str(fact.get("statement", "")), max_len=220)
            if not statement:
                continue
            fact_id = str(fact.get("fact_id", ""))
            fact_link = format_fact_link(fact_id, fact_lookup) if fact_id else ""
            lines.append(f"- {statement} {fact_link}".strip())
            emitted += 1
            if emitted >= 6:
                break
    if emitted == 0:
        lines.append("- 无可用结构化证据，建议回看原文段落与附录中的原始工件。")
    lines.append("")

    contribution = str(profile.get("claimed_contribution", "")).strip()
    if contribution:
        lines.append("## 备注 (Notes)\n")
        lines.append(f"- 作者声称贡献: {normalize_statement_for_reader(contribution, max_len=260)}")
        lines.append("")

    return lines


def build_appendix(
    profile: dict[str, Any],
    logic_graph: dict[str, Any],
    themes: dict[str, Any],
    facts: list[dict[str, Any]],
    cite_map: list[dict[str, Any]],
    reference_catalog: list[dict[str, Any]],
    figure_index: list[dict[str, Any]],
) -> list[str]:
    """Build compact appendix with raw details."""
    lines = []
    lines.append("---\n")
    lines.append("## 附录 (Appendix)\n")
    
    lines.append("### 元数据 (Metadata)\n")
    lines.append(f"- 论文类型: {profile.get('paper_type', 'unknown')}")
    lines.append(f"- 类型置信度: {profile.get('paper_type_confidence', 0.0):.2f}")
    lines.append(f"- 阅读策略: {profile.get('reading_strategy', 'unknown')}")
    lines.append("")
    
    nodes = logic_graph.get("nodes", [])
    edges = logic_graph.get("edges", [])
    if nodes:
        lines.append("### 论证结构 (Argument Structure)\n")
        lines.append(f"- 节点数: {len(nodes)}")
        lines.append(f"- 边数: {len(edges)}")
        
        by_type: dict[str, int] = {}
        for n in nodes:
            t = n.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        lines.append("- 节点类型分布:")
        for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"  - {t}: {cnt}")
        lines.append("")
    
    if facts:
        lines.append("### 事实提取摘要 (Facts Extraction Summary)\n")
        lines.append(f"- 总事实数: {len(facts)}")
        
        by_category: dict[str, int] = {}
        for f in facts:
            cat = f.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
        lines.append("- 类别分布:")
        for cat, cnt in sorted(by_category.items(), key=lambda x: -x[1]):
            lines.append(f"  - {cat}: {cnt}")
        lines.append("")
    
    if figure_index:
        lines.append("### 图表索引 (Figures & Tables)\n")
        tables = [a for a in figure_index if a.get("asset_type") == "table"]
        figures = [a for a in figure_index if a.get("asset_type") == "figure"]
        lines.append(f"- 表格数: {len(tables)}")
        lines.append(f"- 图形数: {len(figures)}")
        
        if tables:
            lines.append("\n**表格**:")
            for t in tables[:5]:
                caption = str(t.get("caption_text") or "")[:60]
                asset_id = t.get("asset_id", "")
                lines.append(f"- [{asset_id}] {caption}")
        if figures:
            lines.append("\n**图形**:")
            for f in figures[:5]:
                caption = str(f.get("caption_text") or "")[:60]
                asset_id = f.get("asset_id", "")
                lines.append(f"- [{asset_id}] {caption}")
        lines.append("")
    
    if cite_map or reference_catalog:
        lines.append("### 引文统计 (Citation Statistics)\n")
        lines.append(f"- 引用锚点数: {len(cite_map)}")
        mapped = sum(1 for c in cite_map if c.get("mapped_ref_key"))
        lines.append(f"- 已映射引用: {mapped} ({mapped*100//max(1,len(cite_map))}%)")
        lines.append(f"- 参考文献数: {len(reference_catalog)}")
        lines.append("")
    
    if facts:
        lines.append("### 全部事实列表 (All Facts - Compact)\n")
        for f in facts[:30]:
            fact_id = f.get("fact_id", "")
            category = f.get("category", "")
            statement = str(f.get("statement") or "")[:80]
            lines.append(f"- [[#{fact_id}|{fact_id}]] [{category}] {statement}")
        if len(facts) > 30:
            lines.append(f"- ... 共 {len(facts)} 条事实")
        lines.append("")
    
    return lines


def run_render(
    run_dir: Path,
    manifest: Manifest,
) -> tuple[int, Path]:
    """Render Obsidian note from reading pipeline artifacts.
    
    Args:
        run_dir: Path to the run directory
        manifest: Document manifest
        
    Returns:
        Tuple of (sections_created, output_path)
    """
    obsidian_dir = run_dir / "obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    
    profile = load_paper_profile(run_dir)
    logic_graph = load_logic_graph(run_dir)
    themes = load_themes(run_dir)
    synthesis = load_synthesis(run_dir)
    facts = load_facts(run_dir)
    paragraphs = load_paragraphs(run_dir)
    cite_map = load_cite_map(run_dir)
    reference_catalog = load_reference_catalog(run_dir)
    figure_index = load_figure_table_index(run_dir)
    summary_status = load_summary_status(run_dir, manifest.doc_id)
    
    paper_type = profile.get("paper_type", "unknown")
    
    lines = []
    lines.append("---")
    lines.append(f"title: {manifest.doc_id}")
    lines.append(f"type: {paper_type}")
    lines.append(f"doc_id: {manifest.doc_id}")
    lines.append(f"source_pdf_hash: {manifest.input_pdf_sha256[:16]}...")
    lines.append(f"summary_status: {summary_status.get('status', 'degraded')}")
    lines.append("---")
    lines.append("")
    
    type_label = {
        "original_research": "原创研究 (Original Research)",
        "review": "综述 (Review)",
        "meta_analysis": "荟萃分析 (Meta-Analysis)",
        "case_report": "病例报告 (Case Report)",
        "methodology": "方法论 (Methodology)",
        "commentary": "评论 (Commentary)",
        "guidelines": "指南 (Guidelines)",
    }.get(paper_type, f"论文 ({paper_type})")
    
    lines.append(f"# {type_label}\n")
    lines.append("")
    
    research_problem = profile.get("research_problem", "")
    if research_problem and research_problem != "无法从可用文本确定研究问题 (Unable to determine research problem from available text)":
        lines.append(f"**{research_problem}**\n")
        lines.append("")
    
    if str(summary_status.get("status", "degraded")) == "degraded":
        content_lines = render_degraded_summary(
            profile=profile,
            synthesis=synthesis,
            facts=facts,
            paragraphs=paragraphs,
            summary_status=summary_status,
        )
    elif paper_type in REVIEW_TYPES:
        content_lines = render_review_meta_analysis(
            profile, synthesis, themes, facts, paragraphs, figure_index
        )
    elif paper_type in ORIGINAL_RESEARCH_TYPES:
        content_lines = render_original_research(
            profile, logic_graph, synthesis, facts, paragraphs, figure_index
        )
    else:
        content_lines = render_generic_fallback(
            profile, synthesis, facts
        )
    
    lines.extend(content_lines)
    
    citations_section = build_citations_section(cite_map, reference_catalog, run_dir)
    lines.append(citations_section)
    lines.append("")
    
    appendix_lines = build_appendix(
        profile, logic_graph, themes, facts, cite_map, reference_catalog, figure_index
    )
    lines.extend(appendix_lines)
    
    output_path = obsidian_dir / f"{manifest.doc_id}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    sections_count = sum(1 for line in lines if line.startswith("## "))
    
    return sections_count, output_path
