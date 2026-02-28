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
import re
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .contract_guard import guard_model_output, safe_json_value
from .manifest import Manifest, load_manifest
from .qa_telemetry import append_fault_events, append_jsonl_event

SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# Feature flag for Wave B narrative-only facts (default OFF)
READING_ENABLE_NARRATIVE_ONLY_FACTS = os.environ.get("READING_ENABLE_NARRATIVE_ONLY_FACTS", "0") in ("1", "true", "True")

REQUIRED_API_KEY_ENV_NAMES = ("SILICONFLOW_API_KEY", "SF_API_KEY", "SILICONFLOW_TOKEN")


def resolve_api_key() -> str:
    for key_name in REQUIRED_API_KEY_ENV_NAMES:
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    return ""


def run_preflight_check(model: str) -> tuple[bool, dict[str, Any]]:
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT).strip()
    api_key = resolve_api_key()
    if not api_key:
        return False, {
            "error_type": "missing_api_key",
            "endpoint": endpoint,
            "model": model,
            "message": (
                "Reading stage preflight failed: no SiliconFlow API credential found. "
                f"Set one of {', '.join(REQUIRED_API_KEY_ENV_NAMES)} before running reading/full stage."
            ),
        }
    if not endpoint.startswith("https://") or "/chat/completions" not in endpoint:
        return False, {
            "error_type": "invalid_endpoint",
            "endpoint": endpoint,
            "model": model,
            "message": (
                "Reading stage preflight failed: SILICONFLOW_ENDPOINT is not a valid chat-completions HTTPS endpoint. "
                f"Current value: {endpoint!r}. Expected format similar to {SILICONFLOW_ENDPOINT!r}."
            ),
        }
    return True, {
        "error_type": "none",
        "endpoint": endpoint,
        "model": model,
        "message": "ok",
    }

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

FACTS_GLOBAL_BATCH_SIZE = 80
MAX_PARALLEL_LLM_CALLS = int(os.environ.get("READING_MAX_PARALLEL_LLM_CALLS", "4"))
MAX_LOCAL_FACT_CANDIDATES = 160
NON_NARRATIVE_CLEAN_ROLES = frozenset({"nuisance", "reference_entry", "figure_caption", "table_caption"})
NARRATIVE_COMPATIBLE_CLEAN_ROLES = frozenset({"body_text", "section_heading", "main_title"})
NARRATIVE_EXCLUDED_SECTIONS = frozenset({
    "authors",
    "affiliations",
    "document metadata",
    "references",
    "figures and tables",
})
TARGET_NON_NARRATIVE_RATIO = 0.2
SUMMARY_FULL_MIN_NARRATIVE_FACTS = 12
SUMMARY_FULL_MIN_THEMES = 3
SUMMARY_FULL_MIN_NARRATIVE_EVIDENCE_RATIO = 0.8


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


def load_clean_role_by_block(blocks_path: Path) -> dict[str, str]:
    clean_role_by_block: dict[str, str] = {}
    if not blocks_path.exists():
        return clean_role_by_block
    with open(blocks_path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            try:
                payload = json.loads(row)
            except json.JSONDecodeError:
                continue
            block_id = str(payload.get("block_id", "")).strip()
            clean_role = str(payload.get("clean_role", "")).strip()
            if block_id and clean_role:
                clean_role_by_block[block_id] = clean_role
    return clean_role_by_block


def normalize_section_label(section: str) -> str:
    normalized = re.sub(r"\s+", " ", str(section).strip().lower())
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    return normalized.strip()


def resolve_paragraph_clean_roles(para: dict[str, Any], clean_role_by_block: dict[str, str]) -> set[str]:
    raw_roles = para.get("clean_roles")
    if isinstance(raw_roles, list):
        roles = {str(role).strip() for role in raw_roles if str(role).strip()}
        if roles:
            return roles

    evidence = para.get("evidence_pointer", {})
    source_ids = evidence.get("source_block_ids", []) if isinstance(evidence, dict) else []
    if not isinstance(source_ids, list):
        return set()

    roles: set[str] = set()
    for source_id in source_ids:
        clean_role = clean_role_by_block.get(str(source_id), "")
        if clean_role:
            roles.add(clean_role)
    return roles


def is_narrative_paragraph_candidate(para: dict[str, Any], clean_role_by_block: dict[str, str]) -> bool:
    role = str(para.get("role", "Body"))
    if role != "Body":
        return False

    section_path = para.get("section_path")
    if isinstance(section_path, list):
        normalized_path = [normalize_section_label(str(item)) for item in section_path if str(item).strip()]
        if any(section in NARRATIVE_EXCLUDED_SECTIONS for section in normalized_path):
            return False

    clean_roles = resolve_paragraph_clean_roles(para, clean_role_by_block)
    if clean_roles & NON_NARRATIVE_CLEAN_ROLES:
        return False
    if clean_roles:
        return bool(clean_roles & NARRATIVE_COMPATIBLE_CLEAN_ROLES)
    return True


def load_cite_anchors(cite_anchors_path: Path) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    if not cite_anchors_path.exists():
        return anchors
    with open(cite_anchors_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                anchors.append(d)
            except json.JSONDecodeError:
                continue
    return anchors


def is_noise_paragraph(para: dict[str, Any]) -> bool:
    text = str(para.get("text", "")).strip()
    if not text:
        return True
    role = str(para.get("role", "Body"))
    if role in {"HeaderFooter", "ReferenceList"}:
        return True
    low = text.lower()
    if "orcid.org" in low or "@" in text and len(text.split()) < 12:
        return True
    if "check for updates" in low:
        return True
    if re.search(r"\bdepartment of\b", low) and re.search(r"\buniversity\b", low):
        return True
    if low.count(",") >= 6 and re.search(r"\b(and|et al\.?|jr\.?|iii)\b", low):
        return True
    words = text.split()
    page_start = int(para.get("page_span", {}).get("start", 1))
    if words:
        short_token_ratio = sum(1 for w in words if len(w.strip(".,;:()[]")) <= 2) / len(words)
        if len(words) < 20 and short_token_ratio > 0.55:
            return True
        digit_tokens = sum(1 for w in words if any(ch.isdigit() for ch in w))
        title_case_tokens = sum(1 for w in words if w[:1].isupper())
        if page_start <= 2 and digit_tokens >= 3 and text.count(",") >= 3 and len(words) <= 45:
            return True
        if page_start <= 2 and len(words) <= 50 and title_case_tokens / len(words) > 0.55 and text.count(",") >= 2:
            return True
    if re.fullmatch(r"[\d\W_]+", text):
        return True
    return False


def select_analysis_paragraphs(paragraphs: list[dict[str, Any]], max_items: int = 220) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for para in paragraphs:
        if is_noise_paragraph(para):
            continue
        role = str(para.get("role", "Body"))
        text = str(para.get("text", "")).strip()
        words = len(text.split())
        if role in {"FigureCaption", "TableCaption"} and words < 8:
            continue
        if words < 5 and role != "Heading":
            continue
        score = 0.0
        score += min(70.0, float(words))
        score += float(para.get("confidence", 0.5)) * 10.0
        if role == "Heading":
            score += 8.0
        if role in {"Body", "Heading"}:
            score += 4.0
        if 18 <= words <= 180:
            score += 6.0
        para_copy = dict(para)
        para_copy["_rank_score"] = score
        candidates.append(para_copy)

    # Section-balanced selection: keep ordering while avoiding overfocus on early pages.
    by_section: dict[str, list[dict[str, Any]]] = {}
    for p in candidates:
        section_path = p.get("section_path")
        if isinstance(section_path, list) and section_path:
            key = " > ".join(str(x) for x in section_path if str(x).strip())
        else:
            key = f"page_{int(p.get('page_span', {}).get('start', 1) or 1):03d}"
        by_section.setdefault(key, []).append(p)

    for key in by_section:
        by_section[key].sort(
            key=lambda p: (
                -float(p.get("_rank_score", 0.0)),
                int(p.get("page_span", {}).get("start", 1) or 1),
                str(p.get("para_id", "")),
            )
        )

    selected: list[dict[str, Any]] = []
    section_keys = sorted(by_section.keys())
    per_section = max(3, min(12, max_items // max(1, len(section_keys))))
    for key in section_keys:
        selected.extend(by_section[key][:per_section])

    if len(selected) < max_items:
        selected_ids = {str(p.get("para_id", "")) for p in selected}
        remaining = [p for p in candidates if str(p.get("para_id", "")) not in selected_ids]
        remaining.sort(key=lambda p: -float(p.get("_rank_score", 0.0)))
        selected.extend(remaining[: max(0, max_items - len(selected))])

    selected.sort(
        key=lambda p: (
            int(p.get("page_span", {}).get("start", 1) or 1),
            float(p.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])[1] if isinstance(p.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0]), list) else 0),
            str(p.get("para_id", "")),
        )
    )
    return selected[:max_items]


def safe_json_parse(raw: str) -> Optional[Any]:
    return safe_json_value(raw)


def heuristic_statement_from_text(text: str) -> str:
    clean = " ".join(text.split())
    if not clean:
        return ""
    parts = re.split(r"(?<=[\.!?])\s+", clean)
    first = parts[0] if parts else clean
    words = first.split()
    if len(words) > 30:
        first = " ".join(words[:30])
    return first


def is_noise_statement(text: str) -> bool:
    clean = " ".join(text.split())
    if not clean:
        return True
    low = clean.lower()
    if "department of" in low or "orcid" in low or "check for updates" in low:
        return True
    words = clean.split()
    if len(words) < 5:
        return True
    digit_tokens = sum(1 for w in words if any(ch.isdigit() for ch in w))
    title_case_tokens = sum(1 for w in words if w[:1].isupper())
    if len(words) <= 45 and digit_tokens >= 3 and clean.count(",") >= 2:
        return True
    if len(words) <= 50 and title_case_tokens / len(words) > 0.6 and clean.count(",") >= 2:
        return True
    return False


def build_evidence_graph(
    paragraphs: list[dict[str, Any]],
    facts: list[Fact],
    citations: list[dict[str, Any]],
    cite_anchors: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> dict[str, Any]:
    para_lookup = {str(p.get("para_id", "")): p for p in paragraphs if p.get("para_id")}
    citation_by_anchor = {str(c.get("anchor_id", "")): c for c in citations}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for para_id, para in para_lookup.items():
        nodes.append({
            "id": f"para:{para_id}",
            "type": "paragraph",
            "page": para.get("page_span", {}).get("start", 1),
            "role": para.get("role", "Body"),
        })

    for fact in facts:
        nodes.append({
            "id": f"fact:{fact.fact_id}",
            "type": "fact",
            "category": fact.category,
            "para_id": fact.para_id,
        })
        if fact.para_id:
            edges.append({
                "from": f"fact:{fact.fact_id}",
                "to": f"para:{fact.para_id}",
                "relation": "derived_from",
            })

    for asset in assets:
        asset_id = str(asset.get("asset_id", ""))
        if not asset_id:
            continue
        nodes.append({
            "id": f"asset:{asset_id}",
            "type": "asset",
            "asset_type": asset.get("asset_type", "unknown"),
            "page": asset.get("page", 1),
        })

    for anchor in cite_anchors:
        anchor_id = str(anchor.get("anchor_id", ""))
        if not anchor_id:
            continue
        anchor_type = str(anchor.get("anchor_type", "unknown"))
        nearest_para_id = str(anchor.get("nearest_para_id", "") or "")
        nodes.append({
            "id": f"anchor:{anchor_id}",
            "type": "citation_anchor",
            "anchor_type": anchor_type,
            "page": anchor.get("page", 1),
        })
        if nearest_para_id:
            edges.append({
                "from": f"anchor:{anchor_id}",
                "to": f"para:{nearest_para_id}",
                "relation": "located_near",
            })
        mapped = citation_by_anchor.get(anchor_id, {}).get("mapped_ref_key")
        if mapped:
            edges.append({
                "from": f"anchor:{anchor_id}",
                "to": f"ref:{mapped}",
                "relation": "maps_to",
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "paragraph_nodes": len(para_lookup),
            "fact_nodes": len(facts),
            "asset_nodes": len([a for a in assets if a.get("asset_id")]),
            "citation_anchors": len(cite_anchors),
        },
    }


def ensure_synthesis_slots(
    synthesis: dict[str, Any],
    assets: list[dict[str, Any]],
    figure_links: dict[str, Any],
) -> dict[str, Any]:
    slots = synthesis.get("figure_table_slots", [])
    if not isinstance(slots, list):
        slots = []

    existing_assets = {
        aid
        for slot in slots if isinstance(slot, dict)
        for aid in slot.get("asset_ids", [])
        if isinstance(aid, str)
    }

    by_section = figure_links.get("by_section", {}) if isinstance(figure_links, dict) else {}
    if isinstance(by_section, dict):
        for section_name, asset_ids in by_section.items():
            if not isinstance(asset_ids, list):
                continue
            pending = [aid for aid in asset_ids if isinstance(aid, str) and aid not in existing_assets]
            if not pending:
                continue
            slots.append({
                "slot_id": f"slot_section_{len(slots)+1:03d}",
                "position_hint": "section_appendix",
                "asset_ids": pending,
                "render_mode": "content_only",
                "section": str(section_name),
            })
            existing_assets.update(pending)

    for asset in assets:
        aid = str(asset.get("asset_id", ""))
        if not aid or aid in existing_assets:
            continue
        slots.append({
            "slot_id": f"slot_{aid}",
            "position_hint": "end_of_summary",
            "asset_ids": [aid],
            "render_mode": "content_only",
        })
        existing_assets.add(aid)

    synthesis["figure_table_slots"] = slots
    return synthesis


def call_siliconflow(prompt: str, max_tokens: int = 4000) -> tuple[str, dict[str, Any]]:
    api_key = resolve_api_key()
    model = os.environ.get("SILICONFLOW_READING_MODEL", SILICONFLOW_DEFAULT_MODEL)
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT)
    meta: dict[str, Any] = {
        "model": model,
        "endpoint": endpoint,
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
        endpoint,
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
        code = int(e.code)
        if code == 400:
            meta["error_type"] = "request_contract_error"
        elif code in {401, 403}:
            meta["error_type"] = "auth_http_error"
        elif code in {408, 429, 500, 502, 503, 504}:
            meta["error_type"] = "transient_http_error"
        else:
            meta["error_type"] = "http_error"
        try:
            body = e.read().decode("utf-8", errors="replace")
            meta["response_chars"] = len(body)
            meta["response_preview"] = body[:300]
        except OSError:
            pass
        return "{}", meta
    except urllib.error.URLError:
        meta["error_type"] = "network_error"
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
    append_jsonl_event(qa_dir, "llm_calls.jsonl", asdict(event), "llm_calls")


def generate_fact_id(para_id: str, index: int) -> str:
    return f"fact_{para_id.replace('para_', '')}_{index:03d}"


def count_words(text: str) -> int:
    return len(text.split())


def truncate_quote(quote: str, max_words: int = 30) -> tuple[str, bool, Optional[str]]:
    words = quote.split()
    if len(words) <= max_words:
        return quote, False, None
    truncated = " ".join(words[:max_words]).rstrip() + " ..."
    return truncated, True, f"Quote truncated to {max_words} words"


def parse_paper_profile(raw: str) -> Optional[PaperProfile]:
    try:
        data = safe_json_parse(raw)
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
    except (TypeError, ValueError):
        return None


def parse_logic_graph(raw: str) -> Optional[dict[str, Any]]:
    data = safe_json_parse(raw)
    if not isinstance(data, dict):
        return None
    required = ["nodes", "edges", "argument_flow"]
    if not all(k in data for k in required):
        return None
    return data


def parse_facts(raw: str, para_id: str) -> list[Fact]:
    try:
        data = safe_json_parse(raw)
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
            parsed_fact = Fact(
                fact_id=fact_id,
                para_id=str(item.get("para_id", para_id)),
                category=category,
                statement=str(item.get("statement", "")),
                quote=quote,
                evidence_pointer=evidence,
                quote_truncated=quote_truncated,
                truncation_reason=truncation_reason,
            )
            facts.append(normalize_fact_for_truncation(parsed_fact))
        return facts
    except TypeError:
        return []


def parse_themes(raw: str) -> Optional[dict[str, Any]]:
    data = safe_json_parse(raw)
    if not isinstance(data, dict):
        return None
    if "themes" not in data:
        alt = data.get("topic_clusters") or data.get("clusters") or data.get("items")
        if isinstance(alt, list):
            data = {
                "themes": alt,
                "cross_theme_links": data.get("cross_theme_links", []),
                "contradictions": data.get("contradictions", []),
            }
        else:
            return None
    return data


def normalize_fact_for_truncation(fact: Fact) -> Fact:
    quote = str(fact.quote or "").strip()
    statement = str(fact.statement or "").strip()
    if not quote or not statement:
        return fact

    if fact.quote_truncated:
        if not quote.endswith("..."):
            fact.quote = quote.rstrip() + " ..."
        if not fact.truncation_reason:
            fact.truncation_reason = "Quote truncated by upstream model output"
        return fact

    q_len = len(quote)
    s_len = len(statement)
    if q_len < s_len and (s_len - q_len > 6):
        if quote.rstrip()[-1:] not in ".!?" and not quote.endswith("..."):
            fact.quote = quote.rstrip() + " ..."
            fact.quote_truncated = True
            fact.truncation_reason = "Detected likely truncation in model quote"
    return fact


def parse_synthesis(raw: str) -> Optional[dict[str, Any]]:
    data = safe_json_parse(raw)
    if not isinstance(data, dict):
        return None
    required = ["executive_summary", "key_evidence_lines"]
    if not all(k in data for k in required):
        return None
    return data


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


def infer_fact_candidate_category(text: str) -> str:
    low = text.lower()
    if any(token in low for token in ("recommend", "should", "guideline", "建议", "应当")):
        return "recommendation"
    if any(token in low for token in ("limitation", "future work", "不足", "局限")):
        return "limitation"
    if any(token in low for token in ("increase", "decrease", "odds", "p<", "p =", "significant", "%", "率")):
        return "statistics"
    if any(token in low for token in ("compared", "versus", "vs", "higher than", "lower than", "相比", "高于", "低于")):
        return "comparison"
    if any(token in low for token in ("mechanism", "pathway", "mediated", "机制", "通路")):
        return "mechanism"
    if any(token in low for token in ("defined", "definition", "classified", "定义", "分类")):
        return "definition"
    if any(token in low for token in ("result", "found", "showed", "conclude", "发现", "结果", "结论")):
        return "result"
    return "background"


def build_local_fact_candidates(
    paragraphs: list[dict[str, Any]],
    max_candidates: int = MAX_LOCAL_FACT_CANDIDATES,
    clean_role_by_block: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    if clean_role_by_block is None:
        clean_role_by_block = {}

    candidates: list[dict[str, Any]] = []
    for para in paragraphs:
        para_id = str(para.get("para_id", ""))
        if not para_id:
            continue
        text = str(para.get("text", "")).strip()
        statement = heuristic_statement_from_text(text)
        if not statement or is_noise_statement(statement):
            continue
        role = str(para.get("role", "Body"))
        score = 0.0
        words = len(statement.split())
        score += min(40.0, float(words))
        score += float(para.get("confidence", 0.5)) * 10.0
        if role == "Heading":
            score -= 8.0
        if role == "Body":
            score += 8.0
        if role in {"FigureCaption", "TableCaption"}:
            score -= 10.0
        if any(ch.isdigit() for ch in statement):
            score += 6.0
        if any(token in statement.lower() for token in ("significant", "increase", "decrease", "risk", "associated")):
            score += 6.0

        is_narrative = is_narrative_paragraph_candidate(para, clean_role_by_block)
        if is_narrative:
            score += 18.0
        else:
            score -= 22.0

        quote, quote_truncated, truncation_reason = truncate_quote(statement)
        evidence = para.get("evidence_pointer", {})
        if not isinstance(evidence, dict):
            evidence = {}
        page_span = para.get("page_span", {})
        if not isinstance(page_span, dict):
            page_span = {}

        candidates.append({
            "candidate_id": f"cand_{para_id}",
            "para_id": para_id,
            "category": infer_fact_candidate_category(statement),
            "statement": statement,
            "quote": quote,
            "quote_truncated": quote_truncated,
            "truncation_reason": truncation_reason,
            "evidence_pointer": {
                "page": page_span.get("start", 1),
                "bbox": evidence.get("bbox_union", [0, 0, 0, 0]),
                "source_block_ids": evidence.get("source_block_ids", []),
            },
            "_score": score,
            "_is_narrative": is_narrative,
        })

    candidates.sort(key=lambda item: float(item.get("_score", 0.0)), reverse=True)
    narrative_candidates = [item for item in candidates if bool(item.get("_is_narrative", False))]
    non_narrative_candidates = [item for item in candidates if not bool(item.get("_is_narrative", False))]

    trimmed = narrative_candidates[:max_candidates]
    if len(trimmed) >= 8:
        remaining = max_candidates - len(trimmed)
        max_non_narrative = max(1, int(len(trimmed) * (TARGET_NON_NARRATIVE_RATIO / (1.0 - TARGET_NON_NARRATIVE_RATIO))))
        if remaining > 0 and non_narrative_candidates:
            trimmed.extend(non_narrative_candidates[:min(remaining, max_non_narrative)])
    else:
        remaining = max_candidates - len(trimmed)
        if remaining > 0:
            trimmed.extend(non_narrative_candidates[:remaining])

    trimmed.sort(key=lambda item: str(item.get("para_id", "")))
    return trimmed


def build_global_facts_prompt(
    candidates: list[dict[str, Any]],
    start_idx: int,
    batch_size: int = FACTS_GLOBAL_BATCH_SIZE,
) -> tuple[str, list[str]]:
    batch = candidates[start_idx:start_idx + batch_size]
    para_ids = [str(item.get("para_id", "")) for item in batch if item.get("para_id")]
    prompt = (
        "Refine deterministic local fact candidates into contract-compliant atomic facts. "
        "Keep evidence pointers grounded to candidate fields, and preserve para_id exactly. "
        "Please provide analysis in Chinese where appropriate.\n\n"
        "## Local Candidate Facts\n\n"
    )
    for item in batch:
        prompt += (
            f"- candidate_id={item.get('candidate_id')} para_id={item.get('para_id')} "
            f"category_hint={item.get('category')} statement={item.get('statement')} "
            f"evidence={item.get('evidence_pointer')}\n"
        )

    prompt += """

Return STRICT JSON ONLY:

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

Use only para_id values present in the candidates list.

Return JSON only:"""
    return prompt, para_ids


def fallback_facts_from_candidates(candidates: list[dict[str, Any]], start_idx: int) -> list[Fact]:
    fallback_facts: list[Fact] = []
    for offset, candidate in enumerate(candidates):
        para_id = str(candidate.get("para_id", ""))
        if not para_id:
            continue
        fallback_facts.append(Fact(
            fact_id=generate_fact_id(para_id, start_idx + offset),
            para_id=para_id,
            category=str(candidate.get("category", "background")),
            statement=str(candidate.get("statement", "")),
            quote=str(candidate.get("quote", "")),
            evidence_pointer=candidate.get("evidence_pointer", {}),
            quote_truncated=bool(candidate.get("quote_truncated", False)),
            truncation_reason=(
                str(candidate.get("truncation_reason"))
                if candidate.get("truncation_reason")
                else None
            ),
        ))
    return fallback_facts


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
        para_text = str(para.get("text", "")).strip()
        heuristic_statement = heuristic_statement_from_text(para_text)
        if not heuristic_statement:
            heuristic_statement = "此段落无可提取的原子事实 (No extractable atomic facts in this paragraph)"
        if len(heuristic_statement.split()) < 6:
            continue
        fact = Fact(
            fact_id=generate_fact_id(para_id, 0),
            para_id=para_id,
            category="background" if heuristic_statement != "此段落无可提取的原子事实 (No extractable atomic facts in this paragraph)" else "none",
            statement=heuristic_statement,
            quote=heuristic_statement if len(heuristic_statement.split()) <= 30 else "",
            evidence_pointer={
                "page": page_span.get("start", 1),
                "bbox": evidence.get("bbox_union", [0, 0, 0, 0]),
                "source_block_ids": evidence.get("source_block_ids", []),
            },
        )
        facts.append(fact)
    return facts


def generate_fallback_themes(facts: list[Fact]) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    for fact in facts:
        if fact.category == "none":
            continue
        groups.setdefault(fact.category, []).append(fact.fact_id)

    themes = []
    for idx, (cat, fact_ids) in enumerate(sorted(groups.items()), start=1):
        themes.append({
            "theme_id": f"theme_{idx:03d}",
            "name": f"{cat} 主题",
            "description": f"围绕 {cat} 的证据汇总。",
            "fact_ids": fact_ids[:20],
            "strength": "moderate" if len(fact_ids) >= 3 else "weak",
            "evidence_quality": "medium" if len(fact_ids) >= 3 else "low",
        })

    return {"themes": themes, "cross_theme_links": [], "contradictions": []}


def generate_fallback_synthesis(facts: list[Fact], assets: list[dict[str, Any]]) -> dict[str, Any]:
    key_lines = []
    scored_candidates: list[tuple[float, Fact]] = []
    for fact in facts:
        stmt = str(fact.statement or "").strip()
        if not stmt or is_noise_statement(stmt):
            continue
        score = 0.0
        score += min(60.0, float(len(stmt.split())))
        if fact.category != "none":
            score += 12.0
        if any(k in stmt.lower() for k in ["risk", "increase", "decrease", "outcome", "associated", "repair", "tear"]):
            score += 10.0
        scored_candidates.append((score, fact))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    fact_candidates = [f for _, f in scored_candidates]
    if not fact_candidates:
        fact_candidates = [f for f in facts if f.statement][:8]

    for fact in fact_candidates[:8]:
        key_lines.append({
            "line_id": f"line_{fact.fact_id}",
            "statement": fact.statement[:200],
            "fact_ids": [fact.fact_id],
            "strength": "moderate" if fact.category != "none" else "weak",
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
    if fact_candidates:
        summary_lines = [f"- {f.statement[:90]}" for f in fact_candidates[:4]]
        summary_text = "自动回退摘要（基于证据段落）：\n" + "\n".join(summary_lines)
    else:
        summary_text = "由于处理失败，无法生成摘要。请参阅各个事实了解详情。(Unable to generate summary due to processing failure. See individual facts for details.)"

    return {
        "executive_summary": summary_text,
        "key_evidence_lines": key_lines,
        "figure_table_slots": slots,
    }


def load_clean_document_metrics(qa_dir: Path) -> dict[str, Any]:
    metrics_path = qa_dir / "clean_document_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def build_summary_status(
    doc_id: str,
    facts: list[Fact],
    themes: dict[str, Any],
    synthesis: dict[str, Any],
    paragraphs: list[dict[str, Any]],
    clean_role_by_block: dict[str, str],
    clean_document_metrics: dict[str, Any],
) -> dict[str, Any]:
    para_lookup = {
        str(para.get("para_id", "")): para
        for para in paragraphs
        if str(para.get("para_id", ""))
    }

    narrative_fact_ids: set[str] = set()
    for fact in facts:
        para = para_lookup.get(str(fact.para_id))
        if para and is_narrative_paragraph_candidate(para, clean_role_by_block):
            narrative_fact_ids.add(str(fact.fact_id))

    narrative_fact_count = len(narrative_fact_ids)
    themes_list = themes.get("themes", []) if isinstance(themes, dict) else []
    themes_count = len(themes_list) if isinstance(themes_list, list) else 0

    key_lines = synthesis.get("key_evidence_lines", []) if isinstance(synthesis, dict) else []
    key_lines = key_lines if isinstance(key_lines, list) else []
    executive_summary = str(synthesis.get("executive_summary", "")).strip() if isinstance(synthesis, dict) else ""
    synthesis_present = bool(executive_summary) and any(
        isinstance(line, dict) and str(line.get("statement", "")).strip()
        for line in key_lines
    )

    fact_ids_set = {str(fact.fact_id) for fact in facts}
    evidence_denominator = 0
    evidence_numerator = 0
    for line in key_lines:
        if not isinstance(line, dict):
            continue
        raw_fact_ids = line.get("fact_ids", [])
        if not isinstance(raw_fact_ids, list):
            continue
        resolved_fact_ids = [str(fid) for fid in raw_fact_ids if str(fid) in fact_ids_set]
        if not resolved_fact_ids:
            continue
        evidence_denominator += 1
        if all(fid in narrative_fact_ids for fid in resolved_fact_ids):
            evidence_numerator += 1

    narrative_evidence_ratio = (
        evidence_numerator / float(evidence_denominator)
        if evidence_denominator > 0
        else 0.0
    )

    reason_codes: list[str] = []
    if not facts:
        reason_codes.append("missing_facts")
    if themes_count == 0:
        reason_codes.append("missing_themes")
    if not synthesis_present:
        reason_codes.append("missing_synthesis")
    if narrative_fact_count < SUMMARY_FULL_MIN_NARRATIVE_FACTS:
        reason_codes.append("insufficient_narrative_facts")
    if themes_count < SUMMARY_FULL_MIN_THEMES:
        reason_codes.append("insufficient_themes")
    if evidence_denominator == 0:
        reason_codes.append("missing_key_evidence_fact_links")
    if narrative_evidence_ratio < SUMMARY_FULL_MIN_NARRATIVE_EVIDENCE_RATIO:
        reason_codes.append("low_narrative_coverage")

    ordering_confidence_low_val = _coerce_bool(clean_document_metrics.get("ordering_confidence_low"))
    section_boundary_unstable_val = _coerce_bool(clean_document_metrics.get("section_boundary_unstable"))
    if ordering_confidence_low_val is None or section_boundary_unstable_val is None:
        reason_codes.append("missing_clean_document_metrics")
    else:
        if ordering_confidence_low_val:
            reason_codes.append("ordering_confidence_low")
        if section_boundary_unstable_val:
            reason_codes.append("section_boundary_unstable")

    status = "full" if not reason_codes else "degraded"
    ordered_unique_reason_codes = list(dict.fromkeys(reason_codes))
    return {
        "doc_id": doc_id,
        "status": status,
        "reason_codes": ordered_unique_reason_codes,
        "metrics": {
            "narrative_fact_count": narrative_fact_count,
            "themes_count": themes_count,
            "synthesis_present": synthesis_present,
            "narrative_evidence_ratio": round(narrative_evidence_ratio, 6),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
    llm_model = os.environ.get("SILICONFLOW_READING_MODEL", SILICONFLOW_DEFAULT_MODEL)
    if not inject_malformed_json:
        preflight_ok, preflight_diag = run_preflight_check(llm_model)
        if not preflight_ok:
            preflight_message = str(preflight_diag.get("message", "Reading preflight failed."))
            preflight_error_type = str(preflight_diag.get("error_type", "preflight_failure"))
            append_llm_call_event(qa_dir, LLMCallEvent(
                stage="reading",
                step="preflight",
                model=str(preflight_diag.get("model", llm_model)),
                endpoint=str(preflight_diag.get("endpoint", SILICONFLOW_ENDPOINT)),
                success=False,
                error_type=preflight_error_type,
                http_status=None,
                prompt_chars=0,
                response_chars=0,
                parse_success=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
                response_preview=preflight_message[:300],
            ))
            append_fault_events(
                qa_dir,
                [
                    {
                        "stage": "reading",
                        "fault": f"preflight-{preflight_error_type}",
                        "retry_attempts": 0,
                        "fallback_used": False,
                        "status": "fail",
                        "error_type": preflight_error_type,
                    }
                ],
            )
            raise RuntimeError(preflight_message)
    paragraphs_path = run_dir / "paragraphs" / "paragraphs.jsonl"
    paragraphs = load_paragraphs(paragraphs_path)
    analysis_paragraphs = select_analysis_paragraphs(paragraphs)
    if not analysis_paragraphs:
        analysis_paragraphs = paragraphs[:220]
    cite_map_path = run_dir / "citations" / "cite_map.jsonl"
    citations = load_cite_map(cite_map_path)
    cite_anchors_path = run_dir / "citations" / "cite_anchors.jsonl"
    cite_anchors = load_cite_anchors(cite_anchors_path)
    figure_index_path = run_dir / "figures_tables" / "figure_table_index.jsonl"
    assets = load_figure_table_index(figure_index_path)
    figure_links_path = run_dir / "figures_tables" / "figure_table_links.json"
    figure_links = load_figure_table_links(figure_links_path)
    blocks_clean_path = run_dir / "text" / "blocks_clean.jsonl"
    clean_role_by_block = load_clean_role_by_block(blocks_clean_path)
    fault_events: list[FaultEvent] = []
    print(
        f"[reading] start paragraphs={len(paragraphs)} analysis={len(analysis_paragraphs)} "
        f"citations={len(citations)} assets={len(assets)} model={llm_model}",
        flush=True,
    )

    profile_prompt = build_profile_prompt(analysis_paragraphs, citations)
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
    profile_guard = guard_model_output(raw_profile, parse_paper_profile)
    profile = profile_guard.parsed
    print(
        f"[reading] profile parse_success={profile_guard.parse_success} error={profile_meta.get('error_type', 'unknown')}",
        flush=True,
    )
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
        parse_success=profile_guard.parse_success,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_preview=str(profile_meta.get("response_preview", "")),
    ))
    if profile_guard.should_fallback or profile is None:
        profile = generate_fallback_profile()
        fault_events.append(
            FaultEvent(
                stage="reading",
                fault=f"profile-{profile_guard.failure_reason.replace('_', '-')}",
                retry_attempts=0,
                fallback_used=True,
                status="degraded",
            )
        )
    profile_path = reading_dir / "paper_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump({
            "paper_type": profile.paper_type,
            "paper_type_confidence": profile.paper_type_confidence,
            "research_problem": profile.research_problem,
            "claimed_contribution": profile.claimed_contribution,
            "reading_strategy": profile.reading_strategy,
        }, f, indent=2, ensure_ascii=False)

    logic_prompt = build_logic_prompt(analysis_paragraphs, profile)
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
    logic_guard = guard_model_output(raw_logic, parse_logic_graph)
    logic_graph = logic_guard.parsed
    print(
        f"[reading] logic parse_success={logic_guard.parse_success} error={logic_meta.get('error_type', 'unknown')}",
        flush=True,
    )
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
        parse_success=logic_guard.parse_success,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_preview=str(logic_meta.get("response_preview", "")),
    ))
    if logic_guard.should_fallback or logic_graph is None:
        logic_graph = generate_fallback_logic_graph()
        fault_events.append(
            FaultEvent(
                stage="reading",
                fault=f"logic-{logic_guard.failure_reason.replace('_', '-')}",
                retry_attempts=0,
                fallback_used=True,
                status="degraded",
            )
        )
    logic_path = reading_dir / "logic_graph.json"
    with open(logic_path, "w", encoding="utf-8") as f:
        json.dump(logic_graph, f, indent=2, ensure_ascii=False)

    facts: list[Fact] = []
    facts_path = reading_dir / "facts.jsonl"
    local_candidates = build_local_fact_candidates(
        analysis_paragraphs,
        clean_role_by_block=clean_role_by_block,
    )
    global_batch_size = FACTS_GLOBAL_BATCH_SIZE
    total_candidates = len(local_candidates)
    total_batches = (total_candidates + global_batch_size - 1) // global_batch_size if total_candidates else 0
    
    if total_batches > 1 and MAX_PARALLEL_LLM_CALLS > 1:
        batch_results: dict[int, list[Fact]] = {}
        
        def process_batch(start_idx: int) -> tuple[int, list[Fact], dict[str, Any], bool, list[FaultEvent]]:
            batch_no = start_idx // global_batch_size + 1
            batch_candidates = local_candidates[start_idx:start_idx + global_batch_size]
            facts_prompt, para_ids = build_global_facts_prompt(local_candidates, start_idx, global_batch_size)
            
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
            
            batch_facts_guard = guard_model_output(
                raw_facts,
                lambda payload: parse_facts(payload, para_ids[0] if para_ids else ""),
            )
            batch_facts = batch_facts_guard.parsed or []
            
            local_faults: list[FaultEvent] = []
            if batch_facts_guard.should_fallback or not batch_facts:
                local_faults.append(
                    FaultEvent(
                        stage="reading",
                        fault=f"facts-{batch_facts_guard.failure_reason.replace('_', '-')}",
                        retry_attempts=0,
                        fallback_used=True,
                        status="degraded",
                    )
                )
                batch_facts = list(batch_facts) + fallback_facts_from_candidates(batch_candidates, start_idx)
            
            return start_idx, batch_facts, facts_meta, batch_facts_guard.parse_success, local_faults
        
        start_indices = list(range(0, total_candidates, global_batch_size))
        max_workers = min(MAX_PARALLEL_LLM_CALLS, len(start_indices))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, idx): idx for idx in start_indices}
            
            for fut in as_completed(futures):
                start_idx, batch_facts, facts_meta, parse_success, local_faults = fut.result()
                batch_no = start_idx // global_batch_size + 1
                batch_results[start_idx] = batch_facts
                fault_events.extend(local_faults)
                
                print(
                    f"[reading] facts-global batch={batch_no}/{total_batches} parse_success={parse_success} "
                    f"error={facts_meta.get('error_type', 'unknown')} candidates={len(local_candidates[start_idx:start_idx + global_batch_size])}",
                    flush=True,
                )
                
                append_llm_call_event(qa_dir, LLMCallEvent(
                    stage="reading",
                    step=f"facts_global_{start_idx:05d}",
                    model=str(facts_meta.get("model", llm_model)),
                    endpoint=str(facts_meta.get("endpoint", SILICONFLOW_ENDPOINT)),
                    success=bool(facts_meta.get("success", False)),
                    error_type=str(facts_meta.get("error_type", "unknown")),
                    http_status=facts_meta.get("http_status", None),
                    prompt_chars=int(facts_meta.get("prompt_chars", 0)),
                    response_chars=int(facts_meta.get("response_chars", 0)),
                    parse_success=parse_success,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    response_preview=str(facts_meta.get("response_preview", "")),
                ))
        
        for start_idx in start_indices:
            facts.extend(batch_results.get(start_idx, []))
    else:
        for start_idx in range(0, total_candidates, global_batch_size):
            batch_no = start_idx // global_batch_size + 1
            batch_candidates = local_candidates[start_idx:start_idx + global_batch_size]
            facts_prompt, para_ids = build_global_facts_prompt(local_candidates, start_idx, global_batch_size)
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
            batch_facts_guard = guard_model_output(
                raw_facts,
                lambda payload: parse_facts(payload, para_ids[0] if para_ids else ""),
            )
            batch_facts = batch_facts_guard.parsed or []
            print(
                f"[reading] facts-global batch={batch_no}/{total_batches} parse_success={batch_facts_guard.parse_success} "
                f"error={facts_meta.get('error_type', 'unknown')} candidates={len(batch_candidates)}",
                flush=True,
            )
            append_llm_call_event(qa_dir, LLMCallEvent(
                stage="reading",
                step=f"facts_global_{start_idx:05d}",
                model=str(facts_meta.get("model", llm_model)),
                endpoint=str(facts_meta.get("endpoint", SILICONFLOW_ENDPOINT)),
                success=bool(facts_meta.get("success", False)),
                error_type=str(facts_meta.get("error_type", "unknown")),
                http_status=facts_meta.get("http_status", None),
                prompt_chars=int(facts_meta.get("prompt_chars", len(facts_prompt))),
                response_chars=int(facts_meta.get("response_chars", len(raw_facts))),
                parse_success=batch_facts_guard.parse_success,
                timestamp=datetime.now(timezone.utc).isoformat(),
                response_preview=str(facts_meta.get("response_preview", "")),
            ))
            if batch_facts_guard.should_fallback or not batch_facts:
                fault_events.append(
                    FaultEvent(
                        stage="reading",
                        fault=f"facts-{batch_facts_guard.failure_reason.replace('_', '-')}",
                        retry_attempts=0,
                        fallback_used=True,
                        status="degraded",
                    )
                )
                batch_facts.extend(fallback_facts_from_candidates(batch_candidates, start_idx))
            facts.extend(batch_facts)
    if not facts:
        if local_candidates:
            facts = fallback_facts_from_candidates(local_candidates, 0)
        if not facts:
            facts = generate_fallback_facts(analysis_paragraphs)
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
    themes_guard = guard_model_output(raw_themes, parse_themes)
    themes = themes_guard.parsed
    print(
        f"[reading] themes parse_success={themes_guard.parse_success} error={themes_meta.get('error_type', 'unknown')}",
        flush=True,
    )
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
        parse_success=themes_guard.parse_success,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_preview=str(themes_meta.get("response_preview", "")),
    ))
    if themes_guard.should_fallback or themes is None:
        themes = generate_fallback_themes(facts)
        fault_events.append(
            FaultEvent(
                stage="reading",
                fault=f"themes-{themes_guard.failure_reason.replace('_', '-')}",
                retry_attempts=0,
                fallback_used=True,
                status="degraded",
            )
        )
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
    synthesis_guard = guard_model_output(raw_synthesis, parse_synthesis)
    synthesis = synthesis_guard.parsed
    print(
        f"[reading] synthesis parse_success={synthesis_guard.parse_success} error={synthesis_meta.get('error_type', 'unknown')}",
        flush=True,
    )
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
        parse_success=synthesis_guard.parse_success,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_preview=str(synthesis_meta.get("response_preview", "")),
    ))
    if synthesis_guard.should_fallback or synthesis is None:
        synthesis = generate_fallback_synthesis(facts, assets)
        fault_events.append(
            FaultEvent(
                stage="reading",
                fault=f"synthesis-{synthesis_guard.failure_reason.replace('_', '-')}",
                retry_attempts=0,
                fallback_used=True,
                status="degraded",
            )
        )
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
    if not valid_lines:
        for idx, fact in enumerate(facts[:5], start=1):
            valid_lines.append({
                "line_id": f"line_fallback_{idx:03d}",
                "statement": fact.statement[:180],
                "fact_ids": [fact.fact_id],
                "strength": "weak",
                "is_strong_claim": False,
            })
    synthesis["key_evidence_lines"] = valid_lines
    synthesis = ensure_synthesis_slots(synthesis, assets, figure_links)
    synthesis_path = reading_dir / "synthesis.json"
    with open(synthesis_path, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2, ensure_ascii=False)

    summary_status = build_summary_status(
        doc_id=manifest.doc_id,
        facts=facts,
        themes=themes,
        synthesis=synthesis,
        paragraphs=paragraphs,
        clean_role_by_block=clean_role_by_block,
        clean_document_metrics=load_clean_document_metrics(qa_dir),
    )
    summary_status_path = qa_dir / "summary_status.json"
    with open(summary_status_path, "w", encoding="utf-8") as f:
        json.dump(summary_status, f, indent=2, ensure_ascii=False)

    evidence_graph = build_evidence_graph(
        analysis_paragraphs,
        facts,
        citations,
        cite_anchors,
        assets,
    )
    evidence_graph_path = reading_dir / "evidence_graph.json"
    with open(evidence_graph_path, "w", encoding="utf-8") as f:
        json.dump(evidence_graph, f, indent=2, ensure_ascii=False)

    print(
        f"[reading] done facts={len(facts)} themes={len(themes.get('themes', []))} "
        f"lines={len(synthesis.get('key_evidence_lines', []))} slots={len(synthesis.get('figure_table_slots', []))} "
        f"fallbacks={len(fault_events)} summary_status={summary_status.get('status', 'degraded')}",
        flush=True,
    )

    if fault_events:
        append_fault_events(qa_dir, [asdict(e) for e in fault_events])

    return (
        len(facts),
        len(themes.get("themes", [])),
        len(synthesis.get("key_evidence_lines", [])),
        len(synthesis.get("figure_table_slots", [])),
        len(fault_events),
    )
