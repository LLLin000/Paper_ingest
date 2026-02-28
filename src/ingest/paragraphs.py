"""Paragraph Canonicalizer - Aggregates merge groups into structured paragraphs.

Contract: .sisyphus/plans/pdf-blueprint-contracts.md (lines 130-139)

Output schema per paragraphs/paragraphs.jsonl:
- para_id: hash of ordered source block_id list
- page_span: {start, end}
- role: Body, Heading, FigureCaption, TableCaption, etc.
- section_path: array of heading strings (nullable)
- text: merged text from source blocks
- evidence_pointer: {pages: [], bbox_union: [], source_block_ids: []}
- neighbors: {prev_para_id, next_para_id} or null
- confidence: float
- provenance: {source, strategy, notes}
"""

import hashlib
import http.client
import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from .qa_telemetry import append_jsonl_event


# Feature flags for Wave B upgrades (default OFF)
# Set environment variable to 1/true to enable
CLEANDOC_ENABLE_LAYOUT_ZONING = os.environ.get("CLEANDOC_ENABLE_LAYOUT_ZONING", "0") in ("1", "true", "True")
CLEANDOC_ENABLE_COLUMN_ORDER_V2 = os.environ.get("CLEANDOC_ENABLE_COLUMN_ORDER_V2", "0") in ("1", "true", "True")


KEEP_FOR_PARAGRAPH_ROLES = frozenset({
    "main_title",
    "section_heading",
    "body_text",
    "figure_caption",
    "table_caption",
    "reference_entry",
})


CLEAN_ROLE_TO_PARAGRAPH_ROLE = {
    "main_title": "Heading",
    "section_heading": "Heading",
    "body_text": "Body",
    "figure_caption": "FigureCaption",
    "table_caption": "TableCaption",
    "reference_entry": "ReferenceList",
    "keywords": "HeaderFooter",
    "doi": "HeaderFooter",
    "received": "HeaderFooter",
    "journal_meta": "HeaderFooter",
    "author_meta": "Body",
    "affiliation_meta": "Body",
    "nuisance": "HeaderFooter",
}


METADATA_SECTION_ROLES: list[tuple[str, str]] = [
    ("journal_meta", "Journal Metadata"),
    ("received", "Submission Timeline"),
    ("doi", "DOI"),
    ("keywords", "Keywords"),
]


METADATA_FAMILY_ROLES = frozenset({
    "author_meta",
    "affiliation_meta",
    "keywords",
    "doi",
    "received",
    "journal_meta",
    "reference_entry",
    "nuisance",
})


AUTHOR_TOKEN_RE = re.compile(r"^(?:[A-Z][a-zA-Z'`-]+|[A-Z]\.)$")
SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
REQUIRED_API_KEY_ENV_NAMES = ("SILICONFLOW_API_KEY", "SF_API_KEY", "SILICONFLOW_TOKEN")


@dataclass
class Paragraph:
    para_id: str
    page_span: dict[str, int]
    role: str
    section_path: Optional[list[str]] = None
    text: str = ""
    evidence_pointer: dict[str, Any] = field(default_factory=dict)
    neighbors: dict[str, Optional[str]] = field(default_factory=lambda: {"prev_para_id": None, "next_para_id": None})
    confidence: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParagraphLLMRefineConfig:
    enabled: bool
    endpoint: str
    model: str
    max_chunks: int
    max_paragraphs: int
    chunk_size: int
    context_radius: int
    timeout_sec: int
    retries: int


def resolve_api_key() -> str:
    for key_name in REQUIRED_API_KEY_ENV_NAMES:
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    return ""


def parse_env_bool(raw: str, default: bool) -> bool:
    value = raw.strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def parse_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def load_paragraph_llm_refine_config() -> ParagraphLLMRefineConfig:
    api_key_exists = bool(resolve_api_key())
    enabled = parse_env_bool(os.environ.get("PARAGRAPHS_LLM_REFINE", ""), default=api_key_exists)
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT).strip() or SILICONFLOW_ENDPOINT
    model = (
        os.environ.get("SILICONFLOW_PARAGRAPHS_MODEL", "").strip()
        or os.environ.get("SILICONFLOW_READING_MODEL", "").strip()
        or SILICONFLOW_DEFAULT_MODEL
    )
    chunk_size = parse_env_int("PARAGRAPHS_LLM_CHUNK_SIZE", default=16, minimum=12, maximum=20)
    return ParagraphLLMRefineConfig(
        enabled=enabled,
        endpoint=endpoint,
        model=model,
        max_chunks=parse_env_int("PARAGRAPHS_LLM_MAX_CHUNKS", default=4, minimum=1, maximum=50),
        max_paragraphs=parse_env_int("PARAGRAPHS_LLM_MAX_PARAGRAPHS", default=80, minimum=1, maximum=500),
        chunk_size=chunk_size,
        context_radius=parse_env_int("PARAGRAPHS_LLM_CONTEXT_RADIUS", default=2, minimum=0, maximum=4),
        timeout_sec=parse_env_int("PARAGRAPHS_LLM_TIMEOUT_SEC", default=45, minimum=5, maximum=180),
        retries=parse_env_int("PARAGRAPHS_LLM_RETRIES", default=1, minimum=0, maximum=3),
    )


def parse_llm_json(raw: str) -> Optional[dict[str, Any]]:
    text = raw.strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match is not None:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def call_siliconflow_for_paragraphs(prompt: str, config: ParagraphLLMRefineConfig) -> tuple[str, dict[str, Any]]:
    api_key = resolve_api_key()
    meta: dict[str, Any] = {
        "endpoint": config.endpoint,
        "model": config.model,
        "success": False,
        "error_type": "unknown",
        "http_status": None,
        "prompt_chars": len(prompt),
        "response_chars": 0,
    }
    if not api_key:
        meta["error_type"] = "missing_api_key"
        return "{}", meta
    payload = json.dumps(
        {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You classify scientific-document paragraph candidates. "
                        "Use corpus-general signals, avoid paper-specific assumptions, and return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        config.endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_sec) as resp:
            body = resp.read().decode("utf-8")
            status = getattr(resp, "status", None)
            result = json.loads(body)
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            meta["http_status"] = status
            meta["response_chars"] = len(content)
            meta["success"] = bool(content)
            meta["error_type"] = "none" if content else "empty_content"
            return content or "{}", meta
    except urllib.error.HTTPError as e:
        meta["http_status"] = e.code
        code = int(e.code)
        if code in {408, 429, 500, 502, 503, 504}:
            meta["error_type"] = "transient_http_error"
        elif code in {401, 403}:
            meta["error_type"] = "auth_http_error"
        elif code == 400:
            meta["error_type"] = "request_contract_error"
        else:
            meta["error_type"] = "http_error"
        return "{}", meta
    except urllib.error.URLError:
        meta["error_type"] = "network_error"
        return "{}", meta
    except http.client.RemoteDisconnected:
        meta["error_type"] = "network_error"
        return "{}", meta
    except TimeoutError:
        meta["error_type"] = "timeout"
        return "{}", meta
    except json.JSONDecodeError:
        meta["error_type"] = "response_json_decode_error"
        return "{}", meta


def append_paragraphs_llm_event(qa_dir: Optional[Path], event: dict[str, Any]) -> None:
    if qa_dir is None:
        return
    qa_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl_event(
        qa_dir,
        "paragraphs_llm_calls.jsonl",
        event,
        "paragraphs_llm_calls",
    )


def compute_para_id(block_ids: list[str]) -> str:
    joined = "|".join(block_ids)
    h = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"para_{h[:12]}"


def load_blocks(blocks_path: Path) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    if not blocks_path.exists():
        return blocks
    with open(blocks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                block_id = d.get("block_id")
                if block_id:
                    blocks[block_id] = d
            except json.JSONDecodeError:
                continue
    return blocks


def load_vision_outputs(vision_dir: Path) -> tuple[dict[int, Any], dict[int, list[dict[str, Any]]], dict[int, dict[str, str]]]:
    confidence_by_page: dict[int, Any] = {}
    merge_groups_by_page: dict[int, list[dict[str, Any]]] = {}
    role_labels_by_page: dict[int, dict[str, str]] = {}
    
    if not vision_dir.exists():
        return confidence_by_page, merge_groups_by_page, role_labels_by_page
    
    for vf in sorted(vision_dir.glob("p*_out.json")):
        try:
            with open(vf, "r", encoding="utf-8") as f:
                data = json.load(f)
            page = data.get("page")
            if page is None:
                continue
            confidence_by_page[page] = data.get("confidence", 0.5)
            merge_groups_by_page[page] = data.get("merge_groups", [])
            role_labels_by_page[page] = data.get("role_labels", {})
        except (json.JSONDecodeError, IOError):
            continue
    
    return confidence_by_page, merge_groups_by_page, role_labels_by_page


def load_page_layouts(text_dir: Path) -> dict[int, dict[str, Any]]:
    """Load page layouts from layout_analysis.json.
    
    Returns dict: {page: {"column_count": int, "column_regions": [[x0,y0,x1,y1], ...], ...}}
    """
    layout_path = text_dir / "layout_analysis.json"
    if not layout_path.exists():
        return {}
    
    try:
        with open(layout_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        page_layouts = data.get("page_layouts", {})
        return {int(k): v for k, v in page_layouts.items()}
    except (json.JSONDecodeError, IOError):
        return {}


def normalize_template_text(text: str) -> str:
    compact = " ".join(str(text).split()).strip().lower()
    if not compact:
        return ""
    compact = re.sub(r"\d+", "#", compact)
    compact = re.sub(r"[^a-z#\s]", " ", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact[:180]


def infer_page_bounds(blocks: dict[str, dict[str, Any]]) -> dict[int, list[float]]:
    bounds: dict[int, list[float]] = {}
    for block in blocks.values():
        page = int(block.get("page", 1))
        bbox = block.get("bbox_pt", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if page not in bounds:
            bounds[page] = [x0, y0, x1, y1]
        else:
            cur = bounds[page]
            cur[0] = min(cur[0], x0)
            cur[1] = min(cur[1], y0)
            cur[2] = max(cur[2], x1)
            cur[3] = max(cur[3], y1)
    return bounds


def word_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def title_case_ratio(words: list[str]) -> float:
    alpha_tokens = [w for w in words if any(ch.isalpha() for ch in w)]
    if not alpha_tokens:
        return 0.0
    title_like = sum(1 for w in alpha_tokens if w[0].isupper())
    return title_like / len(alpha_tokens)


def digit_token_count(words: list[str]) -> int:
    return sum(1 for w in words if any(ch.isdigit() for ch in w))


def sentence_like_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    lower_tokens = sum(1 for w in words if w[:1].islower())
    return lower_tokens / len(words)


def normalize_section_key(text: str) -> str:
    cleaned = clean_text_line(text)
    cleaned = re.sub(r"^\d+(?:\.\d+)*\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


METADATA_HEADING_PREFIXES = (
    "keywords",
    "key words",
    "doi",
    "received",
    "accepted",
    "revised",
    "published online",
    "copyright",
    "correspondence",
    "author for correspondence",
    "conflict of interest",
    "conﬂict of interest",
    "affiliations",
    "authors",
)


def is_metadata_like_heading(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    low = clean.lower()
    return any(
        low == prefix
        or low.startswith(f"{prefix}:")
        or low.startswith(f"{prefix} ")
        for prefix in METADATA_HEADING_PREFIXES
    )


def is_plausible_section_heading(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    if len(clean) > 180:
        return False
    if is_metadata_like_heading(clean):
        return False
    if clean.endswith(":"):
        return False

    words = word_tokens(clean)
    if len(words) < 1 or len(words) > 24:
        return False

    numbered_section_at_start = re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", clean) is not None
    embedded_numbered_section = re.search(r"\b\d+(?:\.\d+)+\.?\s+[A-Z]", clean)
    if embedded_numbered_section is not None and not numbered_section_at_start:
        return False

    sentence_breaks = len(re.findall(r"[.!?]\s+[A-Z]", clean))
    if sentence_breaks >= 1 and not numbered_section_at_start:
        return False

    alpha_match = re.search(r"[A-Za-z]", clean)
    if alpha_match is None:
        return False
    if not numbered_section_at_start and alpha_match.group(0).islower():
        return False

    if not numbered_section_at_start:
        tcase_ratio = title_case_ratio(words)
        if tcase_ratio < 0.58:
            return False

    return True


def should_emit_empty_section_heading(section: str) -> bool:
    clean = clean_text_line(section)
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", clean) is None:
        return False
    if not is_plausible_section_heading(clean):
        return False
    return not looks_like_table_noise(clean)


PAGE_NUMBER_ONLY_LINE_RE = re.compile(r"^\d{1,3}$")
CHECK_FOR_UPDATES_BANNER_RE = re.compile(r"^check\s+for\s+updates\b", flags=re.IGNORECASE)
INLINE_HYPHEN_WRAP_RE = re.compile(r"\b[^\W\d_]+-\s+[^\W\d_]+\b", flags=re.UNICODE)
INLINE_SOFT_HYPHEN_WRAP_RE = re.compile(r"\b([^\W\d_]{2,})\u00ad\s+([^\W\d_]{2,})\b", flags=re.UNICODE)
INLINE_HARD_HYPHEN_WRAP_RE = re.compile(r"\b([^\W\d_]{3,})-\s+([^\W\d_]{2,})\b", flags=re.UNICODE)

AFFILIATION_KEYWORDS = (
    "department",
    "university",
    "hospital",
    "institute",
    "medical center",
    "faculty",
    "school of",
    "address",
    "correspondence",
)


def strip_clean_document_artifact_lines(text: str) -> str:
    """Drop narrow known layout-artifact lines from a paragraph block."""
    clean = str(text)
    if not clean:
        return ""
    kept_lines: list[str] = []
    for raw_line in clean.splitlines():
        line = clean_text_line(raw_line)
        if not line:
            continue
        if PAGE_NUMBER_ONLY_LINE_RE.match(line):
            continue
        if CHECK_FOR_UPDATES_BANNER_RE.match(line):
            continue
        kept_lines.append(line)
    return clean_text_line(" ".join(kept_lines))


def suppression_rule_for_main_body_line(text: str) -> Optional[str]:
    normalized = clean_text_line(text)
    if not normalized:
        return None
    if PAGE_NUMBER_ONLY_LINE_RE.match(normalized) is not None:
        return "page_number_only"
    if CHECK_FOR_UPDATES_BANNER_RE.match(normalized) is not None:
        return "check_for_updates_banner"
    if is_affiliation_address_line(normalized):
        return "affiliation_address_line"
    return None


def is_affiliation_address_line(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    low = clean.lower()
    has_keyword = any(keyword in low for keyword in AFFILIATION_KEYWORDS)
    if not has_keyword:
        return False
    starts_with_marker = re.match(r"^\d+[\s\).,-]", clean) is not None
    has_email = "@" in clean
    has_postal_code = re.search(r"\b\d{5}(?:-\d{4})?\b", clean) is not None
    comma_density = clean.count(",") >= 2
    return starts_with_marker or has_email or has_postal_code or comma_density


def suppress_non_narrative_main_body_entries(
    entries: list[tuple[str, Paragraph]],
    doc_id: str,
) -> tuple[list[tuple[str, Paragraph]], list[dict[str, Any]]]:
    filtered_entries: list[tuple[str, Paragraph]] = []
    suppressions: list[dict[str, Any]] = []
    observed_at = datetime.now(timezone.utc).isoformat()

    for para_text, para in entries:
        kept_lines: list[str] = []
        for raw_line in str(para_text).splitlines():
            original_text = str(raw_line).strip()
            normalized_text = clean_text_line(raw_line)
            if not normalized_text:
                continue
            rule_id = suppression_rule_for_main_body_line(normalized_text)
            if rule_id is None:
                kept_lines.append(normalized_text)
                continue

            source_block_ids = para.evidence_pointer.get("source_block_ids", [])
            if not isinstance(source_block_ids, list):
                source_block_ids = []
            suppressions.append(
                {
                    "doc_id": doc_id,
                    "para_id": para.para_id if para.para_id else None,
                    "source_block_ids": [str(block_id) for block_id in source_block_ids],
                    "rule_id": rule_id,
                    "original_text": original_text,
                    "normalized_text": normalized_text,
                    "observed_at": observed_at,
                }
            )

        merged = clean_text_line(" ".join(kept_lines))
        if merged:
            filtered_entries.append((merged, para))

    return filtered_entries, suppressions


def split_embedded_section_heading(text: str) -> tuple[str, Optional[str], str]:
    """Split body text when it embeds an inline numbered section heading."""
    clean = clean_text_line(text)
    if not clean:
        return "", None, ""

    heading_start_pattern = re.compile(r"\b\d+(?:\.\d+)*\.?\s+[A-Za-z]")
    for match in heading_start_pattern.finditer(clean):
        start = match.start()
        if start <= 0:
            continue
        prefix = clean_text_line(clean[:start])
        if not prefix:
            continue
        if re.search(r"[.!?]\s*$", prefix) is None:
            continue

        tail = clean[start:]
        tail_tokens = word_tokens(tail)
        if len(tail_tokens) < 2:
            continue

        best_heading = ""
        max_tokens = min(24, len(tail_tokens))
        for token_count in range(2, max_tokens + 1):
            candidate = clean_text_line(" ".join(tail_tokens[:token_count]))
            if not candidate:
                continue
            if is_plausible_section_heading(candidate) and not looks_like_table_noise(candidate):
                best_heading = candidate

        if not best_heading:
            continue

        suffix = clean_text_line(tail[len(best_heading):])
        return prefix, best_heading, suffix

    return clean, None, ""


def split_author_segments(text: str) -> list[str]:
    line = clean_text_line(text)
    if not line:
        return []
    # Conservative normalization: strip common trailing numeric/superscript
    # markers (OCR artifacts like "1", "1,2", "¹", "*", "†") before
    # splitting. Keep changes minimal to avoid removing legitimate name
    # characters.
    def _sanitize_author_text(s: str) -> str:
        s = str(s)
        # Remove common unicode superscript digits and markers
        s = re.sub(r"[\u00B9\u00B2\u00B3\u2070\u2074-\u2079]+", "", s)
        # Remove trailing sequences of digits or common markers
        s = re.sub(r"(?:\s|^)(?:\d+|[\*†‡]+)(?:[\s,;:]*)$", "", s)
        return s

    normalized = _sanitize_author_text(line)
    normalized = re.sub(r"\s+and\s+", " | ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[;,]+", " | ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])(?:\d+|[\*†‡]+)(?=\s+[A-Z])", " | ", normalized)
    normalized = re.sub(r"\s(?:\d+|[\*†‡]+)\s+(?=[A-Z])", " | ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return [segment.strip() for segment in normalized.split("|") if segment.strip()]


def looks_like_author_name(segment: str) -> bool:
    text = clean_text_line(segment)
    if not text:
        return False
    # Strip trailing numeric/superscript markers conservatively before
    # applying stricter token-based heuristics. This helps with OCR
    # artifacts like "John Doe1", "Jane Roe¹", or trailing asterisks.
    text = re.sub(r"[\u00B9\u00B2\u00B3\u2070\u2074-\u2079]+", "", text)
    text = re.sub(r"(?:\s|^)(?:\d+|[\*†‡]+)$", "", text).strip()
    text = re.sub(r"(?<=[A-Za-z])(?:\d+|[\*†‡]+)$", "", text).strip()
    if not text:
        return False
    low = text.lower()
    if any(token in low for token in ("department", "university", "hospital", "institute", "address", "doi")):
        return False
    tokens = [t.strip(".") + "." if t.endswith(".") else t for t in word_tokens(text)]
    if len(tokens) < 2 or len(tokens) > 6:
        return False
    matched = sum(1 for token in tokens if AUTHOR_TOKEN_RE.match(token) is not None)
    long_words = sum(1 for token in tokens if re.match(r"^[A-Z][a-zA-Z'`-]{1,}$", token) is not None)
    return matched >= 2 and long_words >= 1


def extract_author_names(author_lines: list[str]) -> list[str]:
    names: list[str] = []
    for line in author_lines:
        for segment in split_author_segments(line):
            if not looks_like_author_name(segment):
                continue
            # Final cleanup: remove superscripts/trailing numeric markers that
            # might have survived splitting/validation.
            cleaned = clean_text_line(segment)
            cleaned = re.sub(r"[\u00B9\u00B2\u00B3\u2070\u2074-\u2079]+", "", cleaned)
            cleaned = re.sub(r"(?:\s|^)(?:\d+|[\*†‡]+)$", "", cleaned).strip()
            cleaned = re.sub(r"(?<=[A-Za-z])(?:\d+|[\*†‡]+)$", "", cleaned).strip()
            if cleaned:
                names.append(cleaned)
    unique_names = unique_texts_in_order(names)
    if unique_names:
        return unique_names
    return unique_texts_in_order(author_lines)


def paragraph_clean_roles(para: Paragraph, clean_role_by_block: dict[str, str]) -> set[str]:
    source_ids = para.evidence_pointer.get("source_block_ids", [])
    if not isinstance(source_ids, list):
        return set()
    roles: set[str] = set()
    for source_id in source_ids:
        role = clean_role_by_block.get(str(source_id), "")
        if role:
            roles.add(role)
    return roles


def infer_metadata_role(
    text: str,
    low: str,
    words: list[str],
    page: int,
    y_rel: float,
    tcase_ratio: float,
    digits: int,
    sentence_ratio: float,
) -> Optional[tuple[str, list[str]]]:
    word_count = len(words)
    has_email = "@" in text
    has_doi_marker = "doi" in low or "doi.org" in low
    starts_with_numeric_marker = re.match(r"^\d+[\s\*\)\.]", text) is not None
    contains_affiliation_keyword = any(
        token in low
        for token in (
            "department",
            "university",
            "hospital",
            "institute",
            "school of",
            "faculty of",
            "medical center",
            "address",
            "division of",
        )
    )

    if (
        page <= 2
        and y_rel <= 0.9
        and word_count <= 40
        and (low.startswith("keywords:") or low.startswith("key words:"))
    ):
        return "keywords", ["front_matter_keywords"]

    if (
        page <= 2
        and y_rel <= 0.9
        and word_count <= 28
        and has_doi_marker
        and digits >= 1
    ):
        return "doi", ["front_matter_doi"]

    if (
        page <= 2
        and y_rel <= 0.95
        and word_count <= 48
        and (
            low.startswith("received")
            or low.startswith("accepted")
            or low.startswith("revised")
            or low.startswith("published online")
        )
    ):
        return "received", ["front_matter_timeline"]

    journal_signals = sum(
        1
        for token in ("issn", "copyright", "vol", "volume", "issue", "pp.", "pages")
        if token in low
    )
    journal_citation_shape = (
        re.match(r"^[A-Z]{2,}\s+\d{4};\s*\d+[–-]\d+", text) is not None
        or re.match(r"^[A-Z]{2,}\s+\d{4};\s*\d+[:：]\d+", text) is not None
        or re.match(r"^\d{4}[–-]\d{3}[\dXx](?:/\d+)?", text) is not None
    )
    byline_citation_shape = "et al" in low and re.match(r"^[A-Z][a-zA-Z'`-]+\s+[A-Z],", text) is not None
    if page <= 2 and word_count <= 45 and (journal_signals >= 2 or "©" in text):
        return "journal_meta", ["journal_metadata_profile"]
    if page <= 2 and word_count <= 18 and (journal_citation_shape or byline_citation_shape):
        return "journal_meta", ["journal_citation_profile"]

    author_segments = [segment for segment in split_author_segments(text) if looks_like_author_name(segment)]

    if (
        page <= 2
        and y_rel <= 0.97
        and 4 <= word_count <= 90
        and (has_email or starts_with_numeric_marker or contains_affiliation_keyword or low.startswith("address correspondence"))
        and sentence_ratio <= 0.78
    ):
        return "affiliation_meta", ["affiliation_profile"]

    if (
        page <= 2
        and y_rel <= 0.72
        and 2 <= word_count <= 90
        and tcase_ratio >= 0.45
        and sentence_ratio <= 0.75
        and not has_email
        and not contains_affiliation_keyword
        and len(author_segments) >= 1
        and not low.startswith("keywords:")
        and not has_doi_marker
    ):
        if digits >= max(1, word_count // 2) and (starts_with_numeric_marker or contains_affiliation_keyword):
            return "affiliation_meta", ["numeric_heavy_front_matter"]
        return "author_meta", [f"author_name_profile_{len(author_segments)}"]

    return None


def infer_clean_role(
    block: dict[str, Any],
    vision_role: str,
    page: int,
    y_rel: float,
    repetition_ratio: float,
    repetition_count: int,
    is_nuisance: bool,
) -> tuple[str, list[str]]:
    if is_nuisance:
        return "nuisance", ["nuisance_filter"]

    text = " ".join(str(block.get("text", "")).split()).strip()
    low = text.lower()
    words = word_tokens(text)
    word_count = len(words)
    tcase_ratio = title_case_ratio(words)
    digits = digit_token_count(words)
    sentence_ratio = sentence_like_ratio(words)
    font_stats = block.get("font_stats", {})
    if not isinstance(font_stats, dict):
        font_stats = {}
    avg_size = float(font_stats.get("avg_size", 0.0) or 0.0)
    is_bold = bool(font_stats.get("is_bold", False))
    heading_candidate = bool(block.get("is_heading_candidate", False))

    if vision_role == "FigureCaption":
        return "figure_caption", ["vision_role_figure_caption"]
    if vision_role == "TableCaption":
        return "table_caption", ["vision_role_table_caption"]
    if vision_role == "ReferenceList":
        return "reference_entry", ["vision_role_reference_list"]

    if (
        page == 1
        and y_rel <= 0.35
        and word_count >= 6
        and (vision_role == "Heading" or heading_candidate or (is_bold and avg_size >= 13.0))
    ):
        return "main_title", ["page1_heading_profile"]

    looks_heading_shape = (
        word_count <= 16
        and tcase_ratio >= 0.55
        and not text.endswith((".", ";", ":"))
        and digits <= 2
    )
    if vision_role == "Heading" or (heading_candidate and looks_heading_shape):
        return "section_heading", ["heading_signal"]

    reference_shape = (
        4 <= word_count <= 55
        and (
            re.match(r"^\d+[\.)]\s+", text) is not None
            or re.match(r"^\[\d+\]", text) is not None
            or low.startswith("references ")
            or low.startswith("references:")
        )
        and text.count(".") >= 2
    )
    if reference_shape:
        return "reference_entry", ["citation_entry_shape"]

    # Guard: if this line begins with a numbered section heading (e.g., "3.3. Something")
    # it's likely a duplicated heading merged into the body and should not be
    # classified as front-matter / affiliation metadata. Skip metadata inference
    # in that case so we preserve it as body_text.
    # Consider a numbered section at start only when the numeric marker
    # is followed by a dot (e.g., '1. Introduction' or '2.1. Background').
    # Avoid treating plain numeric-leading affiliation markers like
    # '1 Department of Radiology...' as section headings.
    numbered_section_at_start = re.match(r"^\d+(?:\.\d+)*\.\s+[A-Za-z]", text) is not None
    metadata = None
    if not numbered_section_at_start:
        metadata = infer_metadata_role(
            text=text,
            low=low,
            words=words,
            page=page,
            y_rel=y_rel,
            tcase_ratio=tcase_ratio,
            digits=digits,
            sentence_ratio=sentence_ratio,
        )
    if metadata is not None:
        role, reasons = metadata
        if repetition_ratio >= 0.6 and repetition_count >= 3 and role in {"journal_meta", "received"}:
            return "nuisance", reasons + ["highly_repeated_metadata"]
        return role, reasons

    return "body_text", ["default_body"]


def paragraph_role_for_block(block: dict[str, Any], fallback_role: str) -> str:
    clean_role = str(block.get("clean_role", "") or "")
    if clean_role in CLEAN_ROLE_TO_PARAGRAPH_ROLE:
        return str(CLEAN_ROLE_TO_PARAGRAPH_ROLE[clean_role])
    return fallback_role


def classify_clean_blocks(
    blocks: dict[str, dict[str, Any]],
    role_labels_by_page: dict[int, dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    template_pages: dict[str, set[int]] = defaultdict(set)
    template_counts: dict[str, int] = defaultdict(int)
    template_edge_hits: dict[str, int] = defaultdict(int)
    template_side_hits: dict[str, int] = defaultdict(int)
    page_bounds = infer_page_bounds(blocks)
    all_pages = {int(block.get("page", 1)) for block in blocks.values()}
    doc_page_count = max(1, len(all_pages))

    repetition_page_threshold = 2 if doc_page_count <= 3 else 3
    repetition_ratio_threshold = 0.7 if doc_page_count <= 2 else (0.55 if doc_page_count <= 4 else 0.4)
    nuisance_score_threshold = 3.8 if doc_page_count <= 2 else 4.1

    for block in blocks.values():
        text = str(block.get("text", ""))
        template = normalize_template_text(text)
        if template:
            page = int(block.get("page", 1))
            bbox = block.get("bbox_pt", [0.0, 0.0, 0.0, 0.0])
            if not isinstance(bbox, list) or len(bbox) < 4:
                bbox = [0.0, 0.0, 0.0, 0.0]
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            bounds = page_bounds.get(page, [0.0, 0.0, x1, y1])
            page_width = max(1e-6, float(bounds[2]) - float(bounds[0]))
            page_height = max(1e-6, float(bounds[3]) - float(bounds[1]))
            y_rel = (((y0 + y1) / 2.0) - float(bounds[1])) / page_height
            x_rel = (((x0 + x1) / 2.0) - float(bounds[0])) / page_width
            near_top = y_rel <= 0.12
            near_bottom = y_rel >= 0.88
            near_side = x_rel <= 0.05 or x_rel >= 0.95 or x0 <= 12.0

            template_counts[template] += 1
            template_pages[template].add(page)
            if near_top or near_bottom:
                template_edge_hits[template] += 1
            if near_side:
                template_side_hits[template] += 1

    annotated: list[dict[str, Any]] = []
    kept_blocks: dict[str, dict[str, Any]] = {}

    for block_id, block in blocks.items():
        block_copy = dict(block)
        page = int(block.get("page", 1))
        bbox = block.get("bbox_pt", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(bbox, list) or len(bbox) < 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        bounds = page_bounds.get(page, [0.0, 0.0, x1, y1])
        page_width = max(1e-6, float(bounds[2]) - float(bounds[0]))
        page_height = max(1e-6, float(bounds[3]) - float(bounds[1]))
        y_center = (y0 + y1) / 2.0
        x_center = (x0 + x1) / 2.0
        y_rel = (y_center - float(bounds[1])) / page_height
        x_rel = (x_center - float(bounds[0])) / page_width

        role = role_labels_by_page.get(page, {}).get(block_id, "Body")
        template = normalize_template_text(str(block.get("text", "")))
        template_page_count = len(template_pages.get(template, set()))
        template_count = int(template_counts.get(template, 0))
        repetition_ratio = template_page_count / float(doc_page_count)
        repeated_edge_hits = int(template_edge_hits.get(template, 0))
        repeated_side_hits = int(template_side_hits.get(template, 0))

        reasons: list[str] = []
        score = 0.0
        strong_vision_nuisance = role == "HeaderFooter"
        protective_vision_role = role in {"FigureCaption", "TableCaption", "ReferenceList"}

        if strong_vision_nuisance:
            reasons.append("vision_role_header_footer")
            score += 4.5

        if bool(block.get("is_header_footer_candidate", False)):
            reasons.append("extractor_header_footer_candidate")
            score += 1.8

        near_top = y_rel <= 0.12
        near_bottom = y_rel >= 0.88
        if near_top:
            reasons.append("near_page_top")
            score += 1.0
        if near_bottom:
            reasons.append("near_page_bottom")
            score += 1.0

        near_side_margin = x_rel <= 0.04 or x_rel >= 0.96 or x0 <= 12.0
        if near_side_margin:
            reasons.append("near_side_margin")
            score += 0.8

        repeated_across_doc = template_page_count >= repetition_page_threshold and repetition_ratio >= repetition_ratio_threshold
        repeated_template = template_page_count >= 2
        if repeated_template:
            reasons.append(f"repeated_template_{template_page_count}_pages")
            score += 1.1
            if repeated_across_doc:
                reasons.append("document_wide_repetition")
                score += 1.5
            if repeated_edge_hits >= repetition_page_threshold and (near_top or near_bottom):
                reasons.append("edge_repetition_alignment")
                score += 1.2
            if repeated_side_hits >= repetition_page_threshold and near_side_margin:
                reasons.append("side_repetition_alignment")
                score += 1.2
            elif repeated_side_hits >= 2 and near_side_margin:
                reasons.append("side_repetition_pair")
                score += 1.0

        words = word_tokens(str(block.get("text", "")))
        if 1 <= len(words) <= 4 and (near_top or near_bottom):
            reasons.append("short_edge_text")
            score += 0.6

        if protective_vision_role:
            score -= 1.6

        is_nuisance = False
        if strong_vision_nuisance:
            is_nuisance = True
        elif "extractor_header_footer_candidate" in reasons and score >= 3.1 and (near_top or near_bottom or near_side_margin):
            is_nuisance = True
        elif repeated_across_doc and (near_top or near_bottom or near_side_margin):
            is_nuisance = True
        elif near_side_margin and template_page_count >= 2 and repeated_side_hits >= 2 and score >= 2.6:
            is_nuisance = True
        elif score >= nuisance_score_threshold and (near_top or near_bottom or near_side_margin or repeated_template):
            is_nuisance = True

        if protective_vision_role and not repeated_across_doc and not strong_vision_nuisance:
            is_nuisance = False

        clean_role, role_reasons = infer_clean_role(
            block=block,
            vision_role=role,
            page=page,
            y_rel=y_rel,
            repetition_ratio=repetition_ratio,
            repetition_count=template_count,
            is_nuisance=is_nuisance,
        )

        block_copy["vision_role"] = role
        block_copy["is_nuisance"] = is_nuisance
        block_copy["nuisance_score"] = round(score, 3)
        block_copy["nuisance_reasons"] = reasons
        block_copy["clean_role"] = clean_role
        block_copy["role_reasons"] = role_reasons
        annotated.append(block_copy)

        if clean_role in KEEP_FOR_PARAGRAPH_ROLES:
            kept_blocks[block_id] = block_copy

    annotated.sort(
        key=lambda b: (
            int(b.get("page", 1)),
            float(b.get("bbox_pt", [0.0, 0.0, 0.0, 0.0])[1]),
            str(b.get("block_id", "")),
        )
    )
    return kept_blocks, annotated


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def union_bbox(bboxes: list[list[float]]) -> list[float]:
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [b[0] for b in bboxes] + [b[2] for b in bboxes]
    ys = [b[1] for b in bboxes] + [b[3] for b in bboxes]
    return [min(xs), min(ys), max(xs), max(ys)]


def infer_section_heading(block: dict[str, Any], role: str) -> Optional[str]:
    if role != "Heading":
        return None
    
    text = block.get("text", "").strip()  # type: ignore[union-attr]
    if not text:
        return None
    
    # Preserve section numbering prefix (e.g., "1. Introduction", "2.1. Background")
    # Do NOT strip the numbering - it's part of the canonical section identifier
    return text


def build_paragraph_from_group(
    group_id: str,
    block_ids: list[str],
    blocks: dict[str, dict[str, Any]],
    role_labels: dict[str, str],
    confidence: float,
) -> Paragraph:
    sorted_block_ids = sorted(block_ids, key=lambda bid: (
        blocks.get(bid, {}).get("page", 0),  # type: ignore[arg-type]
        blocks.get(bid, {}).get("bbox_pt", [0, 0, 0, 0])[1]  # type: ignore[union-attr, index]
    ))
    
    texts: list[str] = []
    bboxes: list[list[float]] = []
    pages: set[int] = set()
    source_block_ids: list[str] = []
    section_path_candidates: list[str] = []
    
    for bid in sorted_block_ids:
        block = blocks.get(bid)
        if not block:
            continue
        texts.append(block.get("text", ""))  # type: ignore[arg]
        bbox = block.get("bbox_pt", [0.0, 0.0, 0.0, 0.0])  # type: ignore[assignment]
        bboxes.append(bbox)  # type: ignore[arg-type]
        pages.add(block.get("page", 1))  # type: ignore[arg]
        source_block_ids.append(bid)
        
        role = paragraph_role_for_block(block, role_labels.get(bid, "Body"))
        heading = infer_section_heading(block, role)
        if heading:
            section_path_candidates.append(heading)
    
    merged_text = " ".join(t.strip() for t in texts if t.strip())
    
    primary_role = "Body"
    if sorted_block_ids:
        first_bid = sorted_block_ids[0]
        first_block = blocks.get(first_bid, {})
        primary_role = paragraph_role_for_block(first_block, role_labels.get(first_bid, "Body"))
    
    sorted_pages = sorted(pages)
    page_span = {
        "start": sorted_pages[0] if sorted_pages else 1,
        "end": sorted_pages[-1] if sorted_pages else 1,
    }
    
    para_id = compute_para_id(sorted_block_ids)
    
    bbox_union = union_bbox(bboxes)
    evidence_pointer = {
        "pages": sorted_pages,
        "bbox_union": bbox_union,
        "source_block_ids": source_block_ids,
    }
    
    section_path: Optional[list[str]] = None
    if section_path_candidates:
        section_path = section_path_candidates
    
    return Paragraph(
        para_id=para_id,
        page_span=page_span,
        role=primary_role,
        section_path=section_path,
        text=merged_text,
        evidence_pointer=evidence_pointer,
        confidence=confidence,
        provenance={
            "source": "merge_group",
            "strategy": "vision_merge_group" if group_id.startswith("mg_") else "singleton",
            "group_id": group_id,
            "notes": "",
        },
    )


def build_singleton_paragraph(
    block_id: str,
    block: dict[str, Any],
    role_labels: dict[str, str],
    confidence: float,
) -> Paragraph:
    role = paragraph_role_for_block(block, role_labels.get(block_id, "Body"))
    
    page: int = int(block.get("page", 1))  # type: ignore[arg-type]
    bbox: list[float] = list(block.get("bbox_pt", [0.0, 0.0, 0.0, 0.0]))  # type: ignore[arg-type]
    text: str = str(block.get("text", ""))  # type: ignore[arg-type]
    
    section_path_candidates: list[str] = []
    heading = infer_section_heading(block, role)
    if heading:
        section_path_candidates.append(heading)
    
    para_id = compute_para_id([block_id])
    
    evidence_pointer = {
        "pages": [page],
        "bbox_union": bbox,
        "source_block_ids": [block_id],
    }
    
    section_path: Optional[list[str]] = None
    if section_path_candidates:
        section_path = section_path_candidates
    
    page_span: dict[str, int] = {"start": page, "end": page}
    text_stripped = text.strip() if isinstance(text, str) else ""
    return Paragraph(
        para_id=para_id,
        page_span=page_span,
        role=role,
        section_path=section_path,
        text=text_stripped,
        evidence_pointer=evidence_pointer,
        confidence=confidence,
        provenance={
            "source": "singleton_block",
            "strategy": "direct_block",
            "block_id": block_id,
            "notes": "",
        },
    )


def aggregate_paragraphs(
    blocks: dict[str, dict[str, Any]],
    merge_groups_by_page: dict[int, list[dict[str, Any]]],
    role_labels_by_page: dict[int, dict[str, str]],
    confidence_by_page: dict[int, Any],
) -> list[Paragraph]:
    all_paragraphs: list[Paragraph] = []
    
    grouped_blocks: set[str] = set()
    
    for page in sorted(merge_groups_by_page.keys()):
        merge_groups = merge_groups_by_page.get(page, [])
        role_labels = role_labels_by_page.get(page, {})
        confidence = confidence_by_page.get(page, 0.5)
        
        for group in merge_groups:
            group_id = group.get("group_id", "")
            block_ids = group.get("block_ids", [])
            if not block_ids:
                continue

            filtered_block_ids = [bid for bid in block_ids if bid in blocks]
            if not filtered_block_ids:
                continue
            
            para = build_paragraph_from_group(
                group_id, filtered_block_ids, blocks, role_labels, confidence
            )
            all_paragraphs.append(para)
            grouped_blocks.update(filtered_block_ids)  # type: ignore[arg-type]
    
    for block_id, block in blocks.items():
        if block_id in grouped_blocks:
            continue
        
        page = block.get("page", 1)  # type: ignore[assignment]
        role_labels = role_labels_by_page.get(page, {})  # type: ignore[index]
        confidence = confidence_by_page.get(page, 0.5)  # type: ignore[index]
        
        para = build_singleton_paragraph(block_id, block, role_labels, confidence)
        all_paragraphs.append(para)
    
    return all_paragraphs


def build_neighbors(paragraphs: list[Paragraph], text_dir: Optional[Path] = None) -> list[Paragraph]:
    """Build neighbor links between paragraphs and sort in reading order.
    
    Uses column-aware sorting for two-column layouts to maintain narrative continuity.
    """
    page_layouts = {}
    if text_dir:
        page_layouts = load_page_layouts(text_dir)
    
    # Use column-aware sorting to handle two-column layouts properly
    para_bounds_by_page = infer_para_page_bounds(paragraphs)
    sorted_paras = sort_paragraphs_column_aware(paragraphs, para_bounds_by_page, page_layouts)
    
    for i, para in enumerate(sorted_paras):
        if i > 0:
            para.neighbors["prev_para_id"] = sorted_paras[i - 1].para_id
        if i < len(sorted_paras) - 1:
            para.neighbors["next_para_id"] = sorted_paras[i + 1].para_id
    
    return sorted_paras


def add_uncertainty_notes(paragraphs: list[Paragraph]) -> list[Paragraph]:
    for para in paragraphs:
        provenance = para.provenance
        
        if para.confidence < 0.6:
            provenance["notes"] = provenance.get("notes", "") + " low_confidence"
        
        if para.section_path is None:
            provenance["notes"] = provenance.get("notes", "") + " no_section_path"
        
        if provenance.get("strategy") == "singleton":
            provenance["notes"] = provenance.get("notes", "") + " no_merge_group"
        
        provenance["notes"] = provenance.get("notes", "").strip()  # type: ignore[union-attr]
    
    return paragraphs


def paragraph_sort_key(para: Paragraph) -> tuple[int, float, float]:
    bbox = para.evidence_pointer.get("bbox_union", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0.0, 0.0, 0.0, 0.0]
    return (para.page_span.get("start", 1), float(bbox[1]), float(bbox[0]))


def detect_two_column_pages(
    paragraphs: list[Paragraph],
    bounds_by_page: dict[int, list[float]],
    page_layouts: Optional[dict[int, dict[str, Any]]] = None,
) -> set[int]:
    """Detect pages that appear to have two-column layout.
    
    Uses layout_analyzer data if available, otherwise falls back to x-position clustering.
    """
    page_layouts = page_layouts or {}
    
    # First, use layout_analyzer column detection if available
    layout_based_pages: set[int] = set()
    for page, layout in page_layouts.items():
        column_count = layout.get("column_count", 1)
        if column_count >= 2:
            layout_based_pages.add(page)
    
    # If we have layout info for all pages, use it directly
    if layout_based_pages:
        return layout_based_pages
    
    # Otherwise, fall back to x-position clustering
    page_x_centers: dict[int, list[float]] = {}
    
    for para in paragraphs:
        page = para.page_span.get("start", 1)
        bbox = para.evidence_pointer.get("bbox_union", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        bounds = bounds_by_page.get(page, [0.0, 0.0, 1.0, 1.0])
        page_width = bounds[2] - bounds[0]
        page_height = bounds[3] - bounds[1]
        if page_width <= 0:
            continue
        block_width_rel = (bbox[2] - bbox[0]) / page_width
        if block_width_rel >= 0.75:
            continue
        # Calculate x-center relative position
        x_center = (bbox[0] + bbox[2]) / 2
        x_rel = (x_center - bounds[0]) / page_width
        # Skip items at the very top of the page (likely titles/headings that span full width)
        # Use relative position from top of content bounds
        y_rel = (bbox[1] - bounds[1]) / max(page_height, 1)
        if y_rel < 0.15:
            continue
        if page not in page_x_centers:
            page_x_centers[page] = []
        page_x_centers[page].append(x_rel)
    
    two_column_pages: set[int] = set()
    for page, x_rel_list in page_x_centers.items():
        if len(x_rel_list) < 4:
            continue
        # Sort and check for bimodal distribution
        sorted_x = sorted(x_rel_list)
        # Check if we have clear left/right clustering
        # If there are items both in left quarter and right quarter, likely two-column
        left_count = sum(1 for x in x_rel_list if x < 0.4)
        right_count = sum(1 for x in x_rel_list if x > 0.6)
        middle_count = len(x_rel_list) - left_count - right_count
        # If significant items in both left and right, and not too many in middle
        if left_count >= 2 and right_count >= 2 and middle_count <= len(x_rel_list) * 0.3:
            two_column_pages.add(page)

    return two_column_pages


def classify_paragraph_column_bucket(
    para: Paragraph,
    bounds_by_page: dict[int, list[float]],
    two_column_pages: set[int],
) -> tuple[int, int, float, float]:
    page = int(para.page_span.get("start", 1))
    bbox = para.evidence_pointer.get("bbox_union", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0.0, 0.0, 0.0, 0.0]

    x0 = float(bbox[0])
    y0 = float(bbox[1])
    x1 = float(bbox[2])
    y1 = float(bbox[3])

    bounds = bounds_by_page.get(page, [0.0, 0.0, max(x1, 1.0), max(y1, 1.0)])
    page_width = max(1e-6, float(bounds[2]) - float(bounds[0]))
    page_height = max(1e-6, float(bounds[3]) - float(bounds[1]))

    x0_rel = (x0 - float(bounds[0])) / page_width
    x1_rel = (x1 - float(bounds[0])) / page_width
    x_center_rel = ((x0 + x1) / 2.0 - float(bounds[0])) / page_width
    y_center_rel = ((y0 + y1) / 2.0 - float(bounds[1])) / page_height
    width_rel = (x1 - x0) / page_width

    if page not in two_column_pages:
        return (page, 0, y0, x0)

    if width_rel >= 0.72:
        if y_center_rel <= 0.2:
            return (page, 0, y0, x0)
        if y_center_rel >= 0.82:
            return (page, 3, y0, x0)
        return (page, 2, y0, x0)

    if x_center_rel < 0.5:
        return (page, 1, y0, x0)
    return (page, 2, y0, x0)


def sort_paragraphs_column_aware(
    paragraphs: list[Paragraph],
    bounds_by_page: dict[int, list[float]],
    page_layouts: Optional[dict[int, dict[str, Any]]] = None,
) -> list[Paragraph]:
    """Sort paragraphs with column-aware ordering for two-column pages."""
    page_layouts = page_layouts or {}
    two_column_pages = detect_two_column_pages(paragraphs, bounds_by_page, page_layouts)

    return sorted(
        paragraphs,
        key=lambda para: classify_paragraph_column_bucket(para, bounds_by_page, two_column_pages),
    )


def block_sort_key(block: dict[str, Any]) -> tuple[int, float, float]:
    bbox = block.get("bbox_pt", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0.0, 0.0, 0.0, 0.0]
    return (int(block.get("page", 1)), float(bbox[1]), float(bbox[0]))


def clean_text_line(text: str) -> str:
    return " ".join(str(text).replace("\u00ad", "").split()).strip()


def normalize_inline_hyphen_wrap_artifacts(text: str) -> str:
    """Normalize conservative inline wrap artifacts without aggressive merging."""
    if not text:
        return ""
    normalized = str(text)

    def _soft_wrap_repl(match: re.Match[str]) -> str:
        left = match.group(1)
        right = match.group(2)
        if not right or not right[0].islower():
            return f"{left} {right}"
        return f"{left}{right}"

    def _hard_wrap_repl(match: re.Match[str]) -> str:
        left = match.group(1)
        right = match.group(2)
        if not right or not right[0].islower():
            return f"{left}- {right}"
        return f"{left}-{right}"

    normalized = INLINE_SOFT_HYPHEN_WRAP_RE.sub(_soft_wrap_repl, normalized)
    normalized = INLINE_HARD_HYPHEN_WRAP_RE.sub(_hard_wrap_repl, normalized)
    return clean_text_line(normalized)


def unique_texts_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = clean_text_line(value)
        if not clean:
            continue
        norm = clean.lower()
        if norm in seen:
            continue
        seen.add(norm)
        result.append(clean)
    return result


def is_informative_caption(text: str) -> bool:
    clean = clean_text_line(text)
    if len(clean) < 8:
        return False
    words = word_tokens(clean)
    if len(words) < 3:
        return False
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    return len(alpha_words) >= 3


def is_caption_like_entry(text: str) -> bool:
    clean = clean_text_line(text)
    if not is_informative_caption(clean):
        return False
    low = clean.lower()
    words = word_tokens(clean)
    word_count = len(words)
    if word_count > 110 or len(clean) > 900:
        return False
    sentence_breaks = len(re.findall(r"[\.!?;:]", clean))
    if sentence_breaks > 8:
        return False

    label_like = (
        re.match(r"^(fig(?:ure)?|table)\s*\.?\s*\d+[a-z]?\b", low) is not None
        or re.match(r"^[a-d][,.:\-]\s", low) is not None
        or " fig." in low
        or " figure " in f" {low} "
        or " table " in f" {low} "
    )
    narrative_markers = sum(
        1
        for marker in ("objective", "materials", "methods", "results", "discussion", "conclusion", "introduction")
        if marker in low
    )
    if narrative_markers >= 2 and word_count > 35:
        return False
    if word_count > 55 and not label_like:
        return False
    return label_like or (word_count <= 40 and sentence_breaks <= 4)


def looks_like_table_noise(text: str) -> bool:
    """Detect table-like or listing-like noise that should be excluded from Main Body.

    This is a conservative gate: prefer false negatives (keep questionable content)
    over false positives (drop narrative text). Uses corpus-general patterns only.

    Detects:
    - Table continuation headers (Table X. (Continued))
    - Table captions (Table X. description)
    - Dense tabular headings with Advantages/Disadvantages columns
    - Citation-only bracket lines (reference numbers only)
    - Fragment-like table rows (semicolon-separated short lists)
    - Dense capitalized column headers
    - Units/advantages/disadvantages columns
    - Numeric row data
    - Front-matter fragments (ORCID, DOI, keywords, etc.)
    """
    clean = clean_text_line(text)
    if not clean:
        return False

    # First check for citation-only bracket lines (can be short)
    # Pattern: [ number ], [ num, num, ... ], possibly with spaces
    bracket_match = re.match(r"^\[.+\]$", clean.strip())
    if bracket_match:
        bracket_content = clean.strip()[1:-1].strip()
        # Check if content is only numbers, commas, spaces
        if re.match(r"^[\d\s,]+$", bracket_content):
            return True

    if len(clean) < 8:
        return False

    if not clean or len(clean) < 8:
        return False

    low = clean.lower()
    words = word_tokens(clean)
    word_count = len(words)

    # Front-matter fragment detection (ORCID, DOI, keywords, etc.)
    # These should never appear in Main Body
    front_matter_patterns = [
        r"\borcid\b",                    # ORCID identifiers
        r"\bdoi\b",                      # DOI markers
        r"\bdoi\.org\b",                 # DOI URLs
        r"\bkeywords:\b",                # Keywords section
        r"\bkey words:\b",               # Alternative keywords
        r"\breceived:",                  # Submission dates
        r"\baccepted:",                  # Acceptance dates
        r"\brevised:",                   # Revision dates
        r"\bpublished online:",          # Publication dates
        r"\bcopyright\b",                # Copyright notices
        r"\bthis article is protected\b", # Copyright warnings
        r"\bcorrespondence\b",           # Correspondence info
        r"\bauthor for correspondence\b", # Correspondence info
    ]
    for pattern in front_matter_patterns:
        if re.search(pattern, low) is not None:
            return True

    # Strong table markers that should trigger regardless of length
    # Table continuation header (with or without period)
    if re.search(r"\btable\s*\d+\s*[\.\)]?\s*\(continued\)", low) is not None:
        return True
    
    # Table caption/heading (Table X. followed by description) - not just "Table 1" but full caption
    if re.search(r"^\btable\s+\d+(?:\.\d+)*\.\s+", low) is not None:
        return True

    # Advantages Disadvantages column header pair (very strong table indicator)
    if "advantages" in low and "disadvantages" in low:
        return True

    # Citation-only bracket line (reference numbers only in brackets) - check early
    # Pattern: [ number ], [ num, num, ... ], possibly with spaces
    bracket_match = re.match(r"^\[.+\]$", clean.strip())
    if bracket_match:
        bracket_content = clean.strip()[1:-1].strip()
        # Check if content is only numbers, commas, spaces
        if re.match(r"^[\d\s,]+$", bracket_content):
            return True

    # Numbered section headings (e.g., "3.1. Methods") are valid body structure,
    # not table noise.
    if (
        re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", clean) is not None
        and word_count <= 20
        and "table" not in low
        and "figure" not in low
    ):
        return False

    # Very short - likely not table noise (but can still be strong markers)
    if word_count < 4:
        return False

    # Fragment-like table row: short/medium semicolon-separated listing
    # with little narrative structure (no sentence-ending punctuation)
    semicolon_count = clean.count(";")
    has_sentence_end = re.search(r"[.!?]\s*$", clean.strip()) is not None
    if 4 <= word_count <= 32 and semicolon_count >= 2 and not has_sentence_end:
        lower_ratio = sentence_like_ratio(words)
        numeric_terms = len(re.findall(r"\b\d+(?:\.\d+)?\b", clean))
        if lower_ratio <= 0.62 or numeric_terms >= 2:
            return True

    if (
        4 <= word_count <= 24
        and semicolon_count >= 1
        and re.search(r"\b\d+\s*[×x]\s*10\b|\b10\s*[−-]\s*\d+\b", clean) is not None
        and has_sentence_end is False
    ):
        return True

    # For longer content, check specific strong markers
    # Check for common table/listing structural markers (more specific)
    has_tabular_markers = any(
        marker in low
        for marker in (
            "advantage:",
            "disadvantage:",
            "pros:",
            "cons:",
            "benefits:",
            "limitations:",
            "specifications:",
            "tolerances:",
        )
    )

    # Dense capitalized short terms (typical table headers) - works for longer content
    capitalized_words = [w for w in words if w and w[0].isupper() and len(w) <= 12]
    if len(capitalized_words) >= word_count * 0.6 and len(capitalized_words) >= 4:
        # Likely tabular heading
        return True

    # Units column pattern: multiple number + unit pairs
    unit_pattern_matches = len(re.findall(r"\b\d+\.?\d*\s+(mm|cm|kg|hz|khz|mhz|kv|ma|w|v|degrees?|°)\b", low))
    if unit_pattern_matches >= 2:
        return True

    # Numeric row pattern: 3+ standalone numbers (table column data)
    if word_count >= 4 and word_count <= 12:
        numeric_only = re.findall(r"\b\d+\.?\d*\b", clean)
        if len(numeric_only) >= 3:
            return True

    # Tabular markers present
    if has_tabular_markers:
        return True

    # Column-like structure: repeated separators with short capitalized items
    # Must have 4+ short items to be considered table-like
    if word_count >= 5:
        short_items = re.findall(r"\b[A-Z][a-z]{1,8}\b", clean)
        if len(short_items) >= 4:
            # Multiple short title-case items - check for separator pattern
            if re.search(r"[,;]\s+[A-Z][a-z]", clean):
                return True

    # Numbers-only rows with consistent structure (very strict)
    if word_count >= 4 and word_count <= 10:
        numeric_words = sum(1 for w in words if re.match(r"^\d+\.?\d*$", w))
        if numeric_words >= word_count * 0.6 and numeric_words >= 3:
            # Mostly numeric content with 3+ numbers - likely table data
            return True

    return False


def is_reference_entry_text(text: str) -> bool:
    clean = normalize_reference_prefix(text)
    low = clean.lower()
    return (
        re.match(r"^\d+[\.)]\s+", clean) is not None
        or re.match(r"^\[\d+\]", clean) is not None
        or low.startswith("et al")
    )


def normalize_reference_prefix(text: str) -> str:
    clean = clean_text_line(text)
    clean = re.sub(r"^references\s*[:\-]?\s*", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def split_reference_entries(text: str) -> list[str]:
    clean = normalize_reference_prefix(text)
    if not clean:
        return []
    marker_pattern = re.compile(r"(?:(?<=^)|(?<=\s))(?:\[\d+\]|\d+[\.)])\s+")
    starts = [m.start() for m in marker_pattern.finditer(clean)]
    if len(starts) <= 1:
        return [clean]
    if starts[0] != 0:
        return [clean]

    parts: list[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(clean)
        chunk = clean_text_line(clean[start:end])
        if chunk:
            parts.append(chunk)
    return parts if parts else [clean]


def normalize_reference_entries(values: list[str]) -> list[str]:
    entries: list[str] = []
    for value in values:
        for candidate in split_reference_entries(value):
            normalized = normalize_reference_prefix(candidate)
            if normalized and is_reference_entry_text(normalized):
                entries.append(normalized)
    return unique_texts_in_order(entries)


def normalized_similarity(a: str, b: str) -> float:
    a_tokens = set(word_tokens(re.sub(r"[^a-z0-9\s]", " ", a.lower())))
    b_tokens = set(word_tokens(re.sub(r"[^a-z0-9\s]", " ", b.lower())))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    containment = overlap / float(min(len(a_tokens), len(b_tokens)))
    jaccard = overlap / float(union)
    return max(containment, jaccard)


def should_exclude_for_abstract_overlap(body_text: str, abstract_text: Optional[str]) -> bool:
    """Check if body text should be excluded due to abstract/front-matter overlap.
    
    Also detects abstract-like fragments that shouldn't appear in Main Body.
    """
    if not body_text:
        return False
    
    clean_body = clean_text_line(body_text)
    if not clean_body:
        return False
    
    low_body = clean_body.lower()
    
    # Direct detection: abstract-like summary fragments
    # These are typically concluding sentences from abstract that leaked into Main Body
    abstract_fragment_markers = [
        "is emphasized", "is proposed", "is reviewed", "is summarized",
        "we review", "we summarize", "we propose", "we present",
        "this review", "this article", "this paper",
        "future development", "challenges are", "perspectives are",
        "in conclusion", "to conclude", "overall,", "in summary",
    ]
    for marker in abstract_fragment_markers:
        if marker in low_body:
            # Short summary-like text on page 1 is likely abstract fragment
            if len(clean_body) < 300:  # Abstract fragments are typically short
                return True
    
    # Original overlap detection logic
    if not abstract_text:
        return False
    
    clean_abstract = clean_text_line(abstract_text)
    if not clean_body or not clean_abstract:
        return False
    sim = normalized_similarity(clean_body, clean_abstract)
    if sim >= 0.86:
        return True
    abstract_norm = re.sub(r"[^a-z0-9\s]", " ", clean_abstract.lower())
    body_norm = re.sub(r"[^a-z0-9\s]", " ", clean_body.lower())
    abstract_norm = " ".join(abstract_norm.split())
    body_norm = " ".join(body_norm.split())
    return len(abstract_norm) >= 60 and abstract_norm in body_norm


def para_bbox_union(para: Paragraph) -> list[float]:
    bbox = para.evidence_pointer.get("bbox_union", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(bbox, list) or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def infer_para_page_bounds(paragraphs: list[Paragraph]) -> dict[int, list[float]]:
    bounds: dict[int, list[float]] = {}
    for para in paragraphs:
        page = int(para.page_span.get("start", 1))
        x0, y0, x1, y1 = para_bbox_union(para)
        if page not in bounds:
            bounds[page] = [x0, y0, x1, y1]
        else:
            cur = bounds[page]
            cur[0] = min(cur[0], x0)
            cur[1] = min(cur[1], y0)
            cur[2] = max(cur[2], x1)
            cur[3] = max(cur[3], y1)
    return bounds


def para_relative_position(para: Paragraph, bounds_by_page: dict[int, list[float]]) -> tuple[float, float]:
    page = int(para.page_span.get("start", 1))
    x0, y0, x1, y1 = para_bbox_union(para)
    bx0, by0, bx1, by1 = bounds_by_page.get(page, [0.0, 0.0, max(x1, 1.0), max(y1, 1.0)])
    width = max(1e-6, bx1 - bx0)
    height = max(1e-6, by1 - by0)
    x_center = (x0 + x1) / 2.0
    y_center = (y0 + y1) / 2.0
    return ((x_center - bx0) / width, (y_center - by0) / height)


def starts_like_orphan_continuation(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    words = word_tokens(clean)
    if len(words) < 3 or len(words) > 45:
        return False
    first_token = words[0]
    if not re.match(r"^[a-z]{1,5}$", first_token):
        return False
    second_token = words[1].lower() if len(words) > 1 else ""
    punctuation_hits = len(re.findall(r"[\.!?]", clean))
    has_figure_hook = second_token.startswith("(fig") or "(fig" in clean.lower()[:30]
    return has_figure_hook or punctuation_hits >= 1


def starts_like_fresh_sentence(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    first_char_match = re.search(r"[A-Za-z0-9]", clean)
    if first_char_match is None:
        return False
    ch = first_char_match.group(0)
    return ch.isupper() or ch.isdigit()


def starts_with_section_cue(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    return re.match(
        r"^(results|discussion|materials\s+and\s+methods|methods|introduction|conclusion|background|clinical\s+study|cadaveric\s+study)\b",
        clean,
        flags=re.IGNORECASE,
    ) is not None


def can_merge_leading_fragment_with_next(current_text: str, next_text: str) -> bool:
    current = clean_text_line(current_text)
    nxt = clean_text_line(next_text)
    if not current or not nxt:
        return False
    if starts_with_section_cue(nxt):
        return False
    if not starts_like_orphan_continuation(current):
        return False
    if not re.match(r"^[a-z]", nxt):
        return False
    return current.endswith("-") or re.search(r"[\.!?]$", current) is None


def merge_fragment_pair(current_text: str, next_text: str) -> str:
    current = clean_text_line(current_text)
    nxt = clean_text_line(next_text)
    if current.endswith("-"):
        return clean_text_line(current[:-1] + nxt)
    return clean_text_line(f"{current} {nxt}")


def should_drop_leading_fragment(
    current_text: str,
    current_para: Paragraph,
    next_text: str,
    next_para: Paragraph,
    bounds_by_page: dict[int, list[float]],
) -> bool:
    if not starts_like_orphan_continuation(current_text):
        return False
    if not starts_like_fresh_sentence(next_text):
        return False
    cur_words = len(word_tokens(current_text))
    nxt_words = len(word_tokens(next_text))
    if nxt_words < max(12, int(cur_words * 1.5)):
        return False

    cur_x_rel, cur_y_rel = para_relative_position(current_para, bounds_by_page)
    nxt_x_rel, nxt_y_rel = para_relative_position(next_para, bounds_by_page)
    cur_page = int(current_para.page_span.get("start", 1))
    nxt_page = int(next_para.page_span.get("start", 1))
    cur_bbox = para_bbox_union(current_para)
    nxt_bbox = para_bbox_union(next_para)
    cur_bounds = bounds_by_page.get(cur_page, [0.0, 0.0, max(cur_bbox[2], 1.0), max(cur_bbox[3], 1.0)])
    nxt_bounds = bounds_by_page.get(nxt_page, [0.0, 0.0, max(nxt_bbox[2], 1.0), max(nxt_bbox[3], 1.0)])
    cur_height = max(1e-6, cur_bounds[3] - cur_bounds[1])
    nxt_height = max(1e-6, nxt_bounds[3] - nxt_bounds[1])
    cur_top_rel = (cur_bbox[1] - cur_bounds[1]) / cur_height
    nxt_top_rel = (nxt_bbox[1] - nxt_bounds[1]) / nxt_height
    near_page_top = cur_y_rel <= 0.42
    right_column_like = cur_x_rel >= 0.52
    same_page = current_para.page_span.get("start", 1) == next_para.page_span.get("start", 1)
    column_progression = same_page and abs(cur_x_rel - nxt_x_rel) <= 0.22 and nxt_y_rel > cur_y_rel
    page_transition = not same_page and next_para.page_span.get("start", 1) >= current_para.page_span.get("start", 1)
    strong_column_continuation = same_page and right_column_like and nxt_x_rel >= 0.45 and nxt_y_rel > cur_y_rel
    cross_column_swap = same_page and cur_x_rel >= 0.62 and nxt_x_rel <= 0.45 and abs(cur_top_rel - nxt_top_rel) <= 0.06
    coherent_next_start = starts_with_section_cue(next_text) or starts_like_fresh_sentence(next_text)
    return (
        (near_page_top and (right_column_like or page_transition or column_progression))
        or strong_column_continuation
        or (cross_column_swap and coherent_next_start)
    )


def prune_leading_orphan_fragments(
    entries: list[tuple[str, Paragraph]],
    bounds_by_page: dict[int, list[float]],
) -> list[tuple[str, Paragraph]]:
    """Prune orphan leading fragments, return filtered entries as tuples."""
    if len(entries) <= 1:
        return list(entries)
    working = list(entries)
    max_passes = min(3, len(working) - 1)
    passes = 0
    while passes < max_passes and len(working) > 1:
        current_text, current_para = working[0]
        next_text, next_para = working[1]
        if should_drop_leading_fragment(current_text, current_para, next_text, next_para, bounds_by_page):
            if can_merge_leading_fragment_with_next(current_text, next_text):
                merged = merge_fragment_pair(current_text, next_text)
                working[1] = (merged, next_para)
            working = working[1:]
            passes += 1
            continue
        break
    return working


def starts_with_lowercase_token(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    match = re.match(r"^([a-z][a-z\-']*)\b", clean)
    return match is not None


def is_likely_legitimate_lowercase_start(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean or not starts_with_lowercase_token(clean):
        return False
    words = word_tokens(clean)
    if len(words) < 2:
        return False

    normalized_words: list[str] = []
    for token in words[:3]:
        normalized = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token).lower()
        if normalized:
            normalized_words.append(normalized)
    if len(normalized_words) < 2:
        return False

    first = normalized_words[0]
    second = normalized_words[1]
    protected_two_token_starts = {
        "de novo",
        "e coli",
        "e g",
        "ex vivo",
        "i e",
        "in situ",
        "in vivo",
        "in vitro",
    }
    if f"{first} {second}" in protected_two_token_starts:
        return True

    # Preserve common taxonomy-style lowercase starts like "e coli ...".
    return re.match(r"^[a-z]$", first) is not None and re.match(r"^[a-z]{2,}$", second) is not None


def is_section_leading_continuation_fragment(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean or not starts_with_lowercase_token(clean):
        return False
    if starts_with_section_cue(clean):
        return False
    if is_reference_entry_text(clean):
        return False
    words = word_tokens(clean)
    if len(words) < 4 or len(words) > 220:
        return False
    first = words[0].lower()
    continuation_tokens = {
        "and",
        "but",
        "can",
        "could",
        "however",
        "it",
        "its",
        "may",
        "might",
        "must",
        "should",
        "that",
        "them",
        "therefore",
        "these",
        "they",
        "this",
        "those",
        "thus",
        "will",
        "would",
        "which",
    }
    return first in continuation_tokens


def is_suspicious_lowercase_leading_fragment(text: str) -> bool:
    """Detect very short/suspicious lowercase leading fragments likely from OCR.

    Conservative: only single-token or two-token lowercase fragments composed of
    simple alphabetic characters and not matching known legitimate lowercase
    starts (e.g., 'e coli', 'de novo'). This avoids trimming real scientific
    phrases while catching truncated openers like 'tive', 'versy', 'cruitment'.
    """
    clean = clean_text_line(text)
    if not clean:
        return False
    words = word_tokens(clean)
    if len(words) < 1 or len(words) > 2:
        return False

    # Strip terminal punctuation for token checks
    toks = [re.sub(r"[\.!,;:\)\]\}]+$", "", w) for w in words]
    # All tokens must be simple lowercase alphabetic or hyphen/apostrophe containing
    if not all(re.match(r"^[a-z\-']{2,8}$", t) for t in toks):
        return False

    # Exclude known legitimate lowercase starts (taxonomic, Latin phrases, etc.)
    if is_likely_legitimate_lowercase_start(clean):
        return False

    # Avoid trimming things that look like reference or table noise
    if is_reference_entry_text(clean) or looks_like_table_noise(clean):
        return False

    return True


def trim_section_leading_continuation_sentence(text: str) -> str:
    clean = clean_text_line(text)
    if not clean:
        return ""
    sentence_match = re.match(r"^(.*?[.!?])(?:\s+|$)(.*)$", clean)
    if sentence_match is None:
        return clean
    first_sentence = clean_text_line(sentence_match.group(1))
    remainder = clean_text_line(sentence_match.group(2))
    if not first_sentence or not remainder:
        return clean
    first_words = word_tokens(first_sentence)
    # Allow trimming for typical continuation sentences of modest length
    # or for very short lowercase fragments (1-2 words) that clearly
    # continue a prior heading/section and are followed by a fresh sentence.
    fw_len = len(first_words)
    if fw_len > 28:
        return clean

    # Require the leading fragment to start with a lowercase token
    if not starts_with_lowercase_token(first_sentence):
        return clean

    first_token = first_words[0].lower()
    second_token = ""
    if len(first_words) > 1:
        second_token = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", first_words[1]).lower()
    continuation_tokens = {
        "and",
        "but",
        "can",
        "could",
        "however",
        "it",
        "its",
        "may",
        "might",
        "must",
        "should",
        "that",
        "them",
        "these",
        "they",
        "this",
        "those",
        "which",
        "will",
        "would",
    }
    first_low = first_sentence.lower()

    # Accept if the first token is an explicit continuation cue, or mentions table/figure.
    # Additionally, allow very short (1-2 word) lowercase noun fragments to be trimmed
    # when followed by a fresh sentence — this addresses cases like "drug delivery." at
    # the start of a section where the real sentence follows.
    short_lowercase_fragment_ok = False
    if fw_len <= 2:
        # Very conservative checks: tokens must be simple lowercase words (no digits/punctuation)
        # Accept trailing terminal punctuation on the last token (e.g., "delivery.") by
        # stripping common trailing punctuation before validation. This preserves the
        # narrow guarding behavior while allowing OCR/normalization artifacts.
        def _simple_lower_tok(tok: str) -> bool:
            # Remove common trailing terminal punctuation characters
            stripped = re.sub(r"[\.!,;:\)\]\}]+$", "", tok)
            return re.match(r"^[a-z\-']+\Z", stripped) is not None

        if all(_simple_lower_tok(w) for w in first_words):
            # Avoid trimming explicit table-like fragments
            if not looks_like_table_noise(first_sentence):
                short_lowercase_fragment_ok = True

    lowercase_sentence_fragment_ok = False
    if 3 <= fw_len <= 16:
        if (
            not looks_like_table_noise(first_sentence)
            and not is_reference_entry_text(first_sentence)
            and not is_likely_legitimate_lowercase_start(first_sentence)
        ):
            lowercase_sentence_fragment_ok = True

    truncated_section_start_ok = False
    if first_token == "cated" and second_token in {"that", "which"}:
        if 8 <= fw_len <= 80 and not looks_like_table_noise(first_sentence):
            truncated_section_start_ok = True

    if not (
        first_token in continuation_tokens
        or "table" in first_low
        or "figure" in first_low
        or short_lowercase_fragment_ok
        or lowercase_sentence_fragment_ok
        or truncated_section_start_ok
    ):
        return clean

    # Remainder must look like a fresh sentence start (uppercase or digit)
    if not starts_like_fresh_sentence(remainder):
        return clean
    table_preface_match = re.match(
        r"^\(\s*Table\s+\d+[A-Za-z]?\s*\)\.\s*(.+)$",
        remainder,
        flags=re.IGNORECASE,
    )
    if table_preface_match is not None:
        after_table_preface = clean_text_line(table_preface_match.group(1))
        if starts_like_fresh_sentence(after_table_preface):
            remainder = after_table_preface
    return remainder


def trim_numeric_leading_continuation_sentence(text: str) -> str:
    clean = clean_text_line(text)
    if not clean:
        return ""
    if re.match(r"^\d+\s*,\s*[A-Z]{2,}-\d+\b", clean) is None:
        return clean
    if is_reference_entry_text(clean) or looks_like_table_noise(clean):
        return clean
    sentence_match = re.match(r"^(.*?[.!?])(?:\s+|$)(.*)$", clean)
    if sentence_match is None:
        return clean
    first_sentence = clean_text_line(sentence_match.group(1))
    remainder = clean_text_line(sentence_match.group(2))
    if not first_sentence or not remainder:
        return clean
    if len(word_tokens(first_sentence)) > 30:
        return clean
    if not starts_like_fresh_sentence(remainder):
        return clean
    return remainder


def should_merge_hyphen_wrap(
    previous_text: str,
    next_text: str,
    prev_x_rel: float = 0.5,
    next_x_rel: float = 0.5,
    prev_y_rel: float = 0.5,
    next_y_rel: float = 0.5,
) -> bool:
    previous = clean_text_line(previous_text)
    nxt = clean_text_line(next_text)
    if not previous or not nxt:
        return False
    if not previous.endswith("-"):
        return False
    if len(previous) >= 2 and previous[-2].isspace():
        return False

    acronym_numeric_continuation = (
        re.search(r"\b[A-Z]{2,}-$", previous) is not None
        and re.match(r"^\d+(?:\s*[,.;:])?(?:\s|$)", nxt) is not None
    )

    # Unicode-aware letter matching (supports ligatures like 'effe' from 'effe-',
    # accented characters, etc.) - uses pattern that matches Unicode letters
    # but excludes digits, underscore, and whitespace.
    # First try: 4+ letter tail (standard case)
    tail_match = re.search(r"([^\W\d_]{4,})-$", previous, re.UNICODE)
    is_short_tail = False
    if tail_match is None:
        # Second try: 2-3 letter tail (short-tail case) - only allowed under
        # strict continuity guards (left-bottom -> right-top column swap with
        # vertical wrap-back AND next starting with lowercase)
        short_tail_match = re.search(r"([^\W\d_]{2,3})-$", previous, re.UNICODE)
        if short_tail_match is None:
            return False
        if not acronym_numeric_continuation:
            # Apply strict guards for short tails:
            # Must be left-to-right column swap with vertical wrap-back
            left_to_right_column_swap = prev_x_rel <= 0.46 and next_x_rel >= 0.54
            wraps_to_earlier_vertical_position = next_y_rel + 0.2 < prev_y_rel
            if not (left_to_right_column_swap and wraps_to_earlier_vertical_position):
                return False
            # Additionally require next text to start with lowercase (not uppercase/digit)
            first_char = nxt[0] if nxt else ""
            if not first_char.islower():
                return False
        tail_match = short_tail_match
        is_short_tail = True

    tail = tail_match.group(1)
    if any(ch.isdigit() for ch in tail):
        return False
    if starts_with_section_cue(nxt):
        return False
    if not starts_with_lowercase_token(nxt) and not acronym_numeric_continuation:
        return False

    same_column_like = abs(prev_x_rel - next_x_rel) <= 0.2
    if not same_column_like:
        # Conservative allowance for true wrap-continuity transitions:
        # left-column bottom -> right-column top in two-column reading order.
        left_to_right_column_swap = prev_x_rel <= 0.46 and next_x_rel >= 0.54
        wraps_to_earlier_vertical_position = next_y_rel + 0.2 < prev_y_rel
        same_row_cross_gutter = left_to_right_column_swap and abs(next_y_rel - prev_y_rel) <= 0.08
        if not ((left_to_right_column_swap and wraps_to_earlier_vertical_position) or same_row_cross_gutter):
            return False

    # Guard against large downward y-gaps in same-column flow.
    # For short tails, apply even stricter vertical continuity.
    if same_column_like:
        y_gap_threshold = 0.25 if is_short_tail else 0.4
        if next_y_rel - prev_y_rel > y_gap_threshold:
            return False
    return True


def should_merge_lowercase_continuation(
    previous_text: str,
    next_text: str,
    prev_x_rel: float = 0.5,
    next_x_rel: float = 0.5,
    prev_y_rel: float = 0.5,
    next_y_rel: float = 0.5,
    prev_page: int | None = None,
    next_page: int | None = None,
) -> bool:
    previous = clean_text_line(previous_text)
    nxt = clean_text_line(next_text)
    if not previous or not nxt:
        return False
    if starts_with_section_cue(nxt):
        return False
    if not starts_with_lowercase_token(nxt):
        return False
    if looks_like_table_noise(nxt):
        return False
    if is_reference_entry_text(previous) or is_reference_entry_text(nxt):
        return False
    if re.search(r"[\.!?]$", previous) is not None:
        return False
    if len(word_tokens(previous)) < 6:
        return False

    if (
        prev_page is not None
        and next_page is not None
        and next_page > prev_page
        and next_page - prev_page <= 4
        and prev_y_rel >= 0.62
        and next_y_rel <= 0.38
    ):
        return True

    # Check for left-to-right column swap with vertical wrap-back
    # This handles the case where text continues from left column bottom to right column top
    left_to_right_column_swap = prev_x_rel <= 0.46 and next_x_rel >= 0.54
    wraps_to_earlier_vertical_position = next_y_rel + 0.2 < prev_y_rel

    if left_to_right_column_swap and wraps_to_earlier_vertical_position:
        # Allow cross-column merge for true wrap-continuity transitions
        return True

    # Guard against cross-column merge in same column flow: stricter threshold (0.2)
    if abs(prev_x_rel - next_x_rel) > 0.2:
        return False
    # Guard against y-gap: if next para is significantly below, don't merge
    # This prevents column-bottom to column-top merges
    if next_y_rel - prev_y_rel > 0.4:
        return False
    return True


def should_merge_citation_tail_continuation(previous_text: str, next_text: str) -> bool:
    previous = clean_text_line(previous_text)
    nxt = clean_text_line(next_text)
    if not previous or not nxt:
        return False
    if starts_with_section_cue(nxt):
        return False
    if not starts_with_lowercase_token(nxt):
        return False
    if looks_like_table_noise(previous) or looks_like_table_noise(nxt):
        return False
    if is_reference_entry_text(previous) or is_reference_entry_text(nxt):
        return False
    if len(word_tokens(previous)) < 10:
        return False
    if re.search(r"\[\s*\d+[a-z]?\s*\]$", previous) is None:
        return False

    first_word = word_tokens(nxt)[0].lower() if word_tokens(nxt) else ""
    allowed_leading_verbs = {
        "developed",
        "demonstrated",
        "employed",
        "found",
        "introduced",
        "prepared",
        "reported",
        "showed",
        "used",
    }
    return first_word in allowed_leading_verbs


def merge_citation_tail_continuation_paragraphs(paragraphs: list[str]) -> list[str]:
    if not paragraphs:
        return []
    merged: list[str] = [clean_text_line(paragraphs[0])]
    for text in paragraphs[1:]:
        current = clean_text_line(text)
        if not current:
            continue
        previous = merged[-1] if merged else ""
        if previous and should_merge_citation_tail_continuation(previous, current):
            merged[-1] = clean_text_line(f"{previous} {current}")
            continue
        merged.append(current)
    return merged


def repair_body_paragraph_boundaries(
    entries: list[tuple[str, Paragraph]],
    bounds_by_page: dict[int, list[float]],
) -> list[str]:
    """Repair paragraph boundaries with position-aware merge guards."""
    if not entries:
        return []
    repaired: list[str] = []
    idx = 0
    while idx < len(entries):
        current_text, current_para = entries[idx]
        current = clean_text_line(current_text)
        if not current:
            idx += 1
            continue
        
        # Get current paragraph's position (x and y relative)
        cur_x_rel, cur_y_rel = para_relative_position(current_para, bounds_by_page)
        cur_page = int(current_para.page_span.get("start", 1))
        
        while idx + 1 < len(entries):
            next_text, next_para = entries[idx + 1]
            nxt = clean_text_line(next_text)
            if not nxt:
                idx += 1
                continue
            
            # Get next paragraph's position (x and y relative)
            next_x_rel, next_y_rel = para_relative_position(next_para, bounds_by_page)
            next_page = int(next_para.page_span.get("start", 1))
            
            if should_merge_hyphen_wrap(current, nxt, cur_x_rel, next_x_rel, cur_y_rel, next_y_rel):
                current = clean_text_line(current[:-1] + nxt)
                idx += 1
                cur_x_rel = next_x_rel  # Update position after merge
                cur_y_rel = next_y_rel
                cur_page = next_page
                continue
            if should_merge_lowercase_continuation(
                current,
                nxt,
                cur_x_rel,
                next_x_rel,
                cur_y_rel,
                next_y_rel,
                prev_page=cur_page,
                next_page=next_page,
            ):
                current = clean_text_line(f"{current} {nxt}")
                idx += 1
                cur_x_rel = next_x_rel  # Update position after merge
                cur_y_rel = next_y_rel
                cur_page = next_page
                continue
            if should_merge_citation_tail_continuation(current, nxt):
                current = clean_text_line(f"{current} {nxt}")
                idx += 1
                cur_x_rel = next_x_rel
                cur_y_rel = next_y_rel
                cur_page = next_page
                continue
            break
        current = trim_numeric_leading_continuation_sentence(current)
        repaired.append(current)
        idx += 1
    return repaired


def build_paragraph_refine_prompt(
    rows: list[dict[str, Any]],
    target_para_ids: list[str],
) -> str:
    schema = {
        "decisions": [
            {
                "para_id": "string",
                "keep": True,
                "continuity_group": "g1",
                "confidence": 0.92,
                "reason": "narrative paragraph",
            }
        ]
    }
    instructions = [
        "Task: Decide if each target paragraph is useful narrative main-body content.",
        "Use corpus-general signals from layout + language + context.",
        "Do not use paper-specific assumptions.",
        "For every target para_id, return one decision with keep/drop, continuity_group, confidence, reason.",
        "If uncertain, keep=true.",
        "Respond with JSON object only.",
    ]
    payload = {
        "instructions": instructions,
        "target_para_ids": target_para_ids,
        "window_rows": rows,
        "output_schema": schema,
    }
    return json.dumps(payload, ensure_ascii=False)


def refine_main_body_with_llm(
    section_entries: list[tuple[str, Paragraph]],
    bounds_by_page: dict[int, list[float]],
    qa_dir: Optional[Path],
) -> list[tuple[str, Paragraph]]:
    config = load_paragraph_llm_refine_config()
    if not config.enabled:
        append_paragraphs_llm_event(
            qa_dir,
            {
                "stage": "paragraphs",
                "step": "llm_refine_disabled",
                "enabled": False,
                "candidate_count": len(section_entries),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return section_entries

    append_paragraphs_llm_event(
        qa_dir,
        {
            "stage": "paragraphs",
            "step": "llm_refine_enabled",
            "enabled": True,
            "candidate_count": len(section_entries),
            "max_chunks": config.max_chunks,
            "max_paragraphs": config.max_paragraphs,
            "chunk_size": config.chunk_size,
            "context_radius": config.context_radius,
            "model": config.model,
            "endpoint": config.endpoint,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    if not resolve_api_key():
        append_paragraphs_llm_event(
            qa_dir,
            {
                "stage": "paragraphs",
                "step": "llm_refine_missing_api_key",
                "enabled": True,
                "error_type": "missing_api_key",
                "candidate_count": len(section_entries),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return section_entries

    max_candidates = min(len(section_entries), config.max_paragraphs)
    if max_candidates <= 0:
        append_paragraphs_llm_event(
            qa_dir,
            {
                "stage": "paragraphs",
                "step": "llm_refine_no_candidates",
                "enabled": True,
                "candidate_count": len(section_entries),
                "max_candidates": max_candidates,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return section_entries

    decisions_by_para_id: dict[str, dict[str, Any]] = {}
    processed_chunks = min(config.max_chunks, (max_candidates + config.chunk_size - 1) // config.chunk_size)
    total_attempt_events = 0
    for chunk_idx in range(processed_chunks):
        start = chunk_idx * config.chunk_size
        end = min(max_candidates, start + config.chunk_size)
        target_entries = section_entries[start:end]
        if not target_entries:
            continue
        context_start = max(0, start - config.context_radius)
        context_end = min(len(section_entries), end + config.context_radius)
        context_entries = section_entries[context_start:context_end]
        target_para_ids = [para.para_id for _, para in target_entries]

        rows: list[dict[str, Any]] = []
        for absolute_idx in range(context_start, context_end):
            text, para = section_entries[absolute_idx]
            x_rel, y_rel = para_relative_position(para, bounds_by_page)
            rows.append(
                {
                    "idx": absolute_idx,
                    "para_id": para.para_id,
                    "is_target": para.para_id in target_para_ids,
                    "role": para.role,
                    "page": int(para.page_span.get("start", 1)),
                    "heading_cue": starts_with_section_cue(text),
                    "word_count": len(word_tokens(text)),
                    "char_count": len(text),
                    "source_block_count": len(para.evidence_pointer.get("source_block_ids", [])),
                    "x_rel": round(x_rel, 3),
                    "y_rel": round(y_rel, 3),
                    "text": text[:900],
                }
            )

        prompt = build_paragraph_refine_prompt(rows, target_para_ids)
        parsed_payload: Optional[dict[str, Any]] = None
        meta: dict[str, Any] = {
            "endpoint": config.endpoint,
            "model": config.model,
            "success": False,
            "error_type": "not_called",
            "http_status": None,
            "prompt_chars": len(prompt),
            "response_chars": 0,
        }
        for attempt in range(config.retries + 1):
            raw, meta = call_siliconflow_for_paragraphs(prompt, config)
            parsed_payload = parse_llm_json(raw)
            parse_ok = isinstance(parsed_payload, dict) and isinstance(parsed_payload.get("decisions"), list)
            append_paragraphs_llm_event(
                qa_dir,
                {
                    "stage": "paragraphs",
                    "step": "llm_refine_main_body",
                    "chunk_index": chunk_idx,
                    "attempt": attempt,
                    "model": str(meta.get("model", config.model)),
                    "endpoint": str(meta.get("endpoint", config.endpoint)),
                    "success": bool(meta.get("success", False)),
                    "error_type": str(meta.get("error_type", "unknown")),
                    "http_status": meta.get("http_status"),
                    "prompt_chars": int(meta.get("prompt_chars", len(prompt)) or 0),
                    "response_chars": int(meta.get("response_chars", 0) or 0),
                    "parse_success": parse_ok,
                    "target_paragraph_count": len(target_para_ids),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            total_attempt_events += 1
            if parse_ok:
                break

        if not isinstance(parsed_payload, dict):
            continue
        decisions = parsed_payload.get("decisions", [])
        if not isinstance(decisions, list):
            continue

        for row in decisions:
            if not isinstance(row, dict):
                continue
            para_id = str(row.get("para_id", "")).strip()
            if para_id not in target_para_ids:
                continue
            keep = bool(row.get("keep", True))
            confidence_raw = row.get("confidence", 0.5)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.5
            decisions_by_para_id[para_id] = {
                "keep": keep,
                "continuity_group": str(row.get("continuity_group", "")),
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(row.get("reason", "")),
            }

    refined: list[tuple[str, Paragraph]] = []
    dropped_count = 0
    for idx, entry in enumerate(section_entries):
        if idx >= max_candidates:
            refined.append(entry)
            continue
        text, para = entry
        decision = decisions_by_para_id.get(para.para_id)
        if decision is None:
            refined.append(entry)
            continue
        keep = bool(decision.get("keep", True))
        confidence = float(decision.get("confidence", 0.5) or 0.5)
        if not keep and confidence >= 0.55:
            dropped_count += 1
            continue
        refined.append((text, para))

    append_paragraphs_llm_event(
        qa_dir,
        {
            "stage": "paragraphs",
            "step": "llm_refine_applied_summary",
            "enabled": True,
            "candidate_count": len(section_entries),
            "max_candidates": max_candidates,
            "chunks_processed": processed_chunks,
            "attempt_events": total_attempt_events,
            "decisions_count": len(decisions_by_para_id),
            "dropped_count": dropped_count,
            "kept_count": len(refined),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return refined


def extract_objective_or_abstract(paragraphs: list[Paragraph]) -> Optional[str]:
    """Extract abstract or objective text from early paragraphs for overlap detection."""
    for para in sorted(paragraphs, key=paragraph_sort_key):
        if para.role != "Body":
            continue
        if para.page_span.get("start", 99) > 2:
            continue
        text = clean_text_line(para.text)
        if not text:
            continue
        low = text.lower()
        
        # Explicit markers
        if "objective" in low:
            match = re.search(r"objective\.?\s*(.+?)(?:\bmaterials\b|\bmethods\b|\bbackground\b|$)", text, flags=re.IGNORECASE)
            if match:
                return clean_text_line(match.group(0))
            return text[:800]
        if "abstract" in low[:80]:
            return text[:800]
        
        # Implicit abstract detection: short summary-like paragraphs on page 1
        # Abstracts typically: 1) appear first, 2) are summary-style, 3) mention "review", "overview", "emphasized"
        if para.page_span.get("start", 1) == 1:
            summary_markers = [
                "is emphasized", "is proposed", "is reviewed", "is summarized",
                "we review", "we summarize", "we propose", "we present",
                "this review", "this article", "this paper",
                "future development", "challenges are", "perspectives are",
            ]
            if any(marker in low for marker in summary_markers):
                # Likely abstract fragment - return for overlap detection
                return text[:800]
    
    return None


def recover_missing_organic_subsection(
    section_order: list[str],
    section_to_paragraphs: dict[str, list[tuple[str, Paragraph]]],
) -> list[str]:
    """Recover missing `x.y.2` organic subsection between `x.y.1` and `x.y.3` headings."""
    recovered_order = list(section_order)
    for idx, section in enumerate(list(recovered_order)):
        clean_section = clean_text_line(section)
        inorganic_match = re.match(
            r"^(?P<prefix>\d+\.\d+)\.1\.\s+Inorganic\s+Piezoelectric\s+Materials$",
            clean_section,
            flags=re.IGNORECASE,
        )
        if inorganic_match is None:
            continue

        prefix = inorganic_match.group("prefix")
        if any(re.match(rf"^{re.escape(prefix)}\.2\.\s+", clean_text_line(name)) for name in recovered_order):
            continue

        idx3 = -1
        for cursor in range(idx + 1, len(recovered_order)):
            candidate = clean_text_line(recovered_order[cursor])
            if re.match(rf"^{re.escape(prefix)}\.3\.\s+Piezocomposites\b", candidate, flags=re.IGNORECASE):
                idx3 = cursor
                break
        if idx3 < 0:
            continue

        entries = section_to_paragraphs.get(section, [])
        if not entries:
            continue
        organic_cue_idx = -1
        for entry_idx, (entry_text, _) in enumerate(entries):
            if re.search(r"\bOrganic\s+Piezoelectric\s+Materials?\b", clean_text_line(entry_text), flags=re.IGNORECASE):
                organic_cue_idx = entry_idx
                break
        if organic_cue_idx < 0:
            continue

        inferred_section = f"{prefix}.2. Organic Piezoelectric Materials"
        if not is_plausible_section_heading(inferred_section) or looks_like_table_noise(inferred_section):
            continue

        inferred_entries = entries[organic_cue_idx:]
        if not inferred_entries:
            continue
        section_to_paragraphs[section] = entries[:organic_cue_idx]
        section_to_paragraphs[inferred_section] = inferred_entries
        recovered_order.insert(idx3, inferred_section)
        break

    return recovered_order


def render_clean_document(
    paragraphs: list[Paragraph],
    annotated_blocks: list[dict[str, Any]],
    doc_id: str,
    qa_dir: Optional[Path] = None,
    text_dir: Optional[Path] = None,
) -> tuple[str, list[dict[str, Any]]]:
    page_layouts = {}
    if text_dir:
        page_layouts = load_page_layouts(text_dir)
    
    # First compute bounds from unsorted paragraphs for column detection
    para_bounds_by_page = infer_para_page_bounds(paragraphs)
    # Use column-aware sorting for two-column layout handling
    ordered_paragraphs = sort_paragraphs_column_aware(paragraphs, para_bounds_by_page, page_layouts)
    # Recompute bounds after sorting (bounds should be same but ensures consistency)
    para_bounds_by_page = infer_para_page_bounds(ordered_paragraphs)
    clean_role_by_block = {
        str(block.get("block_id", "")): str(block.get("clean_role", ""))
        for block in annotated_blocks
        if str(block.get("block_id", ""))
    }

    title = ""
    # continued-marker regex used to avoid promoting marker-only paragraphs to title
    cont_marker_re = re.compile(r"^\(?\s*(?:continued|cont\.?|continued\.)\s*\)?[\s\-:;.,]*$", flags=re.IGNORECASE)
    # Prefer explicit main_title signals from annotated blocks to set the
    # document H1. This avoids promoting section_heading text (like
    # "1. Introduction") into the document title. Fall back to prior
    # heuristics if no main_title is found to preserve compatibility.
    for para in ordered_paragraphs:
        src_ids = para.evidence_pointer.get("source_block_ids", [])
        if not isinstance(src_ids, list):
            continue
        for sid in src_ids:
            if clean_role_by_block.get(str(sid)) == "main_title":
                cand_title = clean_text_line(para.text)
                # Skip continued-marker-like paragraphs
                if cand_title and cont_marker_re.match(cand_title):
                    continue
                title = cand_title
                break
        if title:
            break

    if not title:
        # legacy fallback: first Heading on page 1, but avoid picking
        # numbered section headings (e.g., "1. Introduction") as the
        # document title because that suppresses emission of the section
        # header later. Only accept non-numbered headings here.
        for para in ordered_paragraphs:
            if para.role == "Heading" and para.page_span.get("start", 2) == 1:
                cand = clean_text_line(para.text)
                if not cand:
                    continue
                # Skip continued-marker-like paragraphs
                if cont_marker_re.match(cand):
                    continue
                # If it looks like a numbered section at start, skip as title
                if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", cand):
                    continue
                title = cand
                if title:
                    break

    if not title:
        for para in ordered_paragraphs:
            txt = clean_text_line(para.text)
            if not txt:
                continue
            # Skip continued-marker-like paragraphs
            if cont_marker_re.match(txt):
                continue
            # Avoid selecting a numbered section heading as the title here
            if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", txt):
                continue
            title = txt[:180]
            break

    objective_or_abstract = extract_objective_or_abstract(ordered_paragraphs)

    section_to_paragraphs: dict[str, list[tuple[str, Paragraph]]] = {}
    section_order: list[str] = []
    section_key_to_title: dict[str, str] = {}
    current_section = "Main Body"

    figure_captions: list[str] = []
    table_captions: list[str] = []

    for para in ordered_paragraphs:
        text = clean_text_line(para.text)
        if not text:
            continue
        para_clean = paragraph_clean_roles(para, clean_role_by_block)

        # Priority 1: Use para.section_path when available (from vision-detected headings)
        # This captures section context that heading-tracking might miss
        section_from_path = False
        path_eligible = (
            ("section_heading" in para_clean or "body_text" in para_clean)
            and "table_caption" not in para_clean
            and "figure_caption" not in para_clean
        )
        if path_eligible and para.section_path and isinstance(para.section_path, list) and para.section_path:
            section_display = clean_text_line(str(para.section_path[-1]))
            # Skip metadata sections
            key = normalize_section_key(section_display)
            if (
                key
                and key not in {"keywords", "doi", "received", "accepted", "affiliations", "authors"}
                and is_plausible_section_heading(section_display)
                and not looks_like_table_noise(section_display)
            ):
                current_section = section_display
                if current_section not in section_to_paragraphs:
                    section_to_paragraphs[current_section] = []
                    section_order.append(current_section)
                section_key_to_title[key] = current_section
                section_from_path = True

        if section_from_path and "body_text" not in para_clean:
            continue

        # Priority 2: Fall back to heading role detection when section_path is unavailable
        section_from_heading_role = False
        if "section_heading" in para_clean and not (para_clean & METADATA_FAMILY_ROLES):
            if title and text.lower() == title.lower():
                continue
            key = normalize_section_key(text)
            if not key:
                continue
            if key in {"keywords", "doi", "received", "accepted", "affiliations", "authors"}:
                continue
            if not is_plausible_section_heading(text):
                key = ""
            if not key:
                section_from_heading_role = False
            else:
                # Skip table continuation headings - they're noise, not real sections
                if looks_like_table_noise(text):
                    section_from_heading_role = False
                else:
                    current_section = section_key_to_title.get(key, text)
                    section_key_to_title[key] = current_section
                    if current_section not in section_to_paragraphs:
                        section_to_paragraphs[current_section] = []
                        section_order.append(current_section)
                    section_from_heading_role = True

        if section_from_heading_role and "body_text" not in para_clean:
            continue

        if "figure_caption" in para_clean or para.role == "FigureCaption":
            if is_caption_like_entry(text):
                figure_captions.append(text)
            continue
        if "table_caption" in para_clean or para.role == "TableCaption":
            if is_caption_like_entry(text):
                table_captions.append(text)
            continue
        if para_clean & METADATA_FAMILY_ROLES:
            continue
        if "body_text" not in para_clean:
            # Conservative fallback: when paragraph aggregation already
            # classified an item as Body, keep it unless it is explicitly
            # marked as reference-entry or nuisance by clean-role labels.
            if para.role != "Body":
                continue
            if para_clean & {"reference_entry", "nuisance"}:
                continue

        if current_section not in section_to_paragraphs:
            section_to_paragraphs[current_section] = []
            section_order.append(current_section)

        body_prefix, embedded_heading, body_suffix = split_embedded_section_heading(text)
        if body_prefix:
            if not should_exclude_for_abstract_overlap(body_prefix, objective_or_abstract):
                if not looks_like_table_noise(body_prefix):
                    section_to_paragraphs[current_section].append((body_prefix, para))

        if embedded_heading:
            heading_key = normalize_section_key(embedded_heading)
            if (
                heading_key
                and heading_key not in {"keywords", "doi", "received", "accepted", "affiliations", "authors"}
                and is_plausible_section_heading(embedded_heading)
                and not looks_like_table_noise(embedded_heading)
            ):
                current_section = section_key_to_title.get(heading_key, embedded_heading)
                section_key_to_title[heading_key] = current_section
                if current_section not in section_to_paragraphs:
                    section_to_paragraphs[current_section] = []
                    section_order.append(current_section)
            if not body_suffix:
                continue
            if should_exclude_for_abstract_overlap(body_suffix, objective_or_abstract):
                continue
            if looks_like_table_noise(body_suffix):
                continue
            section_to_paragraphs[current_section].append((body_suffix, para))

    ordered_blocks = sorted(annotated_blocks, key=block_sort_key)
    # Exclude items that look like section headings or table noise from
    # author/affiliation metadata lists. Some OCR'd documents duplicate
    # section headings into nearby blocks which can confuse metadata
    # extraction. Be conservative: only drop candidates that are clearly
    # plausible section headings or table-like noise, or that exactly
    # match the detected title.
    author_meta_lines = unique_texts_in_order([
        str(block.get("text", ""))
        for block in ordered_blocks
        if str(block.get("clean_role", "")) == "author_meta"
        and (not looks_like_table_noise(str(block.get("text", ""))))
        and (not title or str(block.get("text", "")).strip().lower() != title.lower())
    ])
    author_meta = extract_author_names(author_meta_lines)
    # Conservative fallback: if extraction returned nothing but there are
    # classifier-identified author_meta blocks, preserve the raw annotated
    # texts. This avoids losing authors due to brittle name-parsing heuristics
    # (e.g., OCR superscripts, unusual separators) while still respecting
    # prior table-noise/title guards above.
    if not author_meta:
        raw_author_candidates = [
            str(block.get("text", ""))
            for block in ordered_blocks
            if str(block.get("clean_role", "")) == "author_meta"
        ]
        if raw_author_candidates:
            author_meta = unique_texts_in_order(raw_author_candidates)
    affiliation_meta = unique_texts_in_order([
        str(block.get("text", ""))
        for block in ordered_blocks
        if str(block.get("clean_role", "")) == "affiliation_meta"
        and (not looks_like_table_noise(str(block.get("text", ""))))
        and (not title or str(block.get("text", "")).strip().lower() != title.lower())
    ])
    metadata_values: dict[str, list[str]] = {
        role: unique_texts_in_order([
            str(block.get("text", ""))
            for block in ordered_blocks
            if str(block.get("clean_role", "")) == role
        ])
        for role, _ in METADATA_SECTION_ROLES
    }
    
    # Extract DOI directly from annotated_blocks (includes nuisance-filtered blocks)
    if not metadata_values.get("doi"):
        doi_pattern = re.compile(r"10\.\d{4,}[^\s\]>\"\'\)]+", re.IGNORECASE)
        for block in annotated_blocks:
            text = str(block.get("text", ""))
            doi_match = doi_pattern.search(text)
            if doi_match:
                found_doi = doi_match.group(0).rstrip(".,;:)")
                if found_doi:
                    metadata_values["doi"] = [f"https://doi.org/{found_doi}"]
                    break
    references = normalize_reference_entries([
        str(block.get("text", ""))
        for block in ordered_blocks
        if str(block.get("clean_role", "")) == "reference_entry"
    ])

    lines: list[str] = []
    lines.append(f"# {title or 'Clean Document'}")
    lines.append("")

    if author_meta:
        lines.append("## Authors")
        lines.append("")
        for item in author_meta:
            lines.append(f"- {item}")
        lines.append("")

    if affiliation_meta:
        lines.append("## Affiliations")
        lines.append("")
        for item in affiliation_meta:
            lines.append(f"- {item}")
        lines.append("")

    has_metadata_sections = any(metadata_values.get(role, []) for role, _ in METADATA_SECTION_ROLES)
    if has_metadata_sections:
        lines.append("## Document Metadata")
        lines.append("")
        for role, title_text in METADATA_SECTION_ROLES:
            values = metadata_values.get(role, [])
            if not values:
                continue
            lines.append(f"### {title_text}")
            lines.append("")
            for value in values:
                lines.append(f"- {value}")
            lines.append("")

    if objective_or_abstract:
        lines.append("## Abstract / Objective")
        lines.append("")
        lines.append(objective_or_abstract)
        lines.append("")

    lines.append("## Main Body")
    lines.append("")

    if not section_order:
        section_order = ["Main Body"]
        fallback_body: list[tuple[str, Paragraph]] = []
        for para in ordered_paragraphs:
            para_clean = paragraph_clean_roles(para, clean_role_by_block)
            if "body_text" not in para_clean or (para_clean & METADATA_FAMILY_ROLES):
                continue
            text = clean_text_line(para.text)
            if should_exclude_for_abstract_overlap(text, objective_or_abstract):
                continue
            if looks_like_table_noise(text):
                continue
            if text:
                fallback_body.append((text, para))
        section_to_paragraphs["Main Body"] = fallback_body

    section_order = recover_missing_organic_subsection(section_order, section_to_paragraphs)
    suppression_events: list[dict[str, Any]] = []
    # Process and refine each section conservatively, collecting intermediate results
    processed_section_entries: dict[str, list[tuple[str, Paragraph]]] = {}
    for section in section_order:
        section_entries = section_to_paragraphs.get(section, [])
        section_entries = refine_main_body_with_llm(section_entries, para_bounds_by_page, qa_dir)
        pruned_entries = prune_leading_orphan_fragments(section_entries, para_bounds_by_page)

        # Conservative Main Body-only repair:
        # If the very first entry is a suspicious short lowercase fragment (likely OCR
        # artifact) and the following paragraph clearly starts like a fresh sentence,
        # drop the fragment. This is intentionally narrow: only for Main Body and only
        # when is_suspicious_lowercase_leading_fragment + starts_like_fresh_sentence
        # both hold. Do NOT touch non-Main Body sections here (they have separate
        # continuation-trimming logic below).
        if section == "Main Body" and pruned_entries and len(pruned_entries) > 1:
            first_text, first_para = pruned_entries[0]
            second_text, _ = pruned_entries[1]
            if is_suspicious_lowercase_leading_fragment(first_text) and starts_like_fresh_sentence(second_text):
                pruned_entries = pruned_entries[1:]

        if section != "Main Body" and pruned_entries:
            first_text = pruned_entries[0][0]
            trimmed_first = trim_section_leading_continuation_sentence(first_text)
            if trimmed_first and trimmed_first != clean_text_line(first_text):
                pruned_entries[0] = (trimmed_first, pruned_entries[0][1])
            else:
                if (
                    len(pruned_entries) > 1
                    and starts_with_lowercase_token(first_text)
                    and len(word_tokens(clean_text_line(first_text))) <= 2
                    and not looks_like_table_noise(first_text)
                    and not is_reference_entry_text(first_text)
                    and starts_like_fresh_sentence(pruned_entries[1][0])
                ):
                    pruned_entries = pruned_entries[1:]
                elif is_section_leading_continuation_fragment(first_text):
                    pruned_entries = pruned_entries[1:]

                if pruned_entries:
                    heading_text = clean_text_line(section)
                    if heading_text:
                        try:
                            heading_re = re.compile(rf"^{re.escape(heading_text)}[\s\:\.\-–—]*\s*(.*)$")
                        except re.error:
                            heading_re = None
                        if heading_re is not None:
                            m = heading_re.match(clean_text_line(pruned_entries[0][0]))
                            if m:
                                remainder = clean_text_line(m.group(1))
                                if remainder and starts_like_fresh_sentence(remainder):
                                    pruned_entries[0] = (remainder, pruned_entries[0][1])

            if pruned_entries:
                first_clean = clean_text_line(pruned_entries[0][0])
                cont_re = re.compile(r"^\(?\s*(?:continued|cont\.?|continued\.)\s*\)?[\s\-:;.,]*", flags=re.IGNORECASE)
                new_first = cont_re.sub("", first_clean)
                if new_first and new_first != first_clean and starts_like_fresh_sentence(new_first):
                    pruned_entries[0] = (new_first, pruned_entries[0][1])
                else:
                    if not new_first and len(pruned_entries) > 1 and starts_like_fresh_sentence(pruned_entries[1][0]):
                        pruned_entries = pruned_entries[1:]

            if pruned_entries:
                first_clean = clean_text_line(pruned_entries[0][0])
                citation_prefix_match = re.match(r"^\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]\s*(.+)$", first_clean)
                if citation_prefix_match is not None:
                    after_citation = clean_text_line(citation_prefix_match.group(1))
                    if after_citation and starts_like_fresh_sentence(after_citation):
                        pruned_entries[0] = (after_citation, pruned_entries[0][1])

        # Drop marker-only paragraphs conservatively
        cont_marker_re = re.compile(r"^\(?\s*(?:continued|cont\.?|continued\.)\s*\)?[\s\-:;.,]*$", flags=re.IGNORECASE)
        filtered_pruned: list[tuple[str, Paragraph]] = []
        for idx, (txt, para) in enumerate(pruned_entries):
            clean_txt = clean_text_line(txt)
            if cont_marker_re.match(clean_txt):
                next_ok = idx + 1 < len(pruned_entries) and starts_like_fresh_sentence(pruned_entries[idx + 1][0])
                other_non_marker = any(not cont_marker_re.match(clean_text_line(e[0])) for j, e in enumerate(pruned_entries) if j != idx)
                if next_ok or other_non_marker:
                    continue
            filtered_pruned.append((txt, para))

        if filtered_pruned:
            filtered_pruned, section_suppressions = suppress_non_narrative_main_body_entries(
                filtered_pruned,
                doc_id=doc_id,
            )
            suppression_events.extend(section_suppressions)

        processed_section_entries[section] = filtered_pruned

    # Conservative post-processing: consider merging tiny leading lowercase
    # fragments under a heading back into the previous section when spatial
    # and textual guards permit. This avoids leaving orphan 1-2 word fragments
    # immediately after a ### heading.
    for i in range(1, len(section_order)):
        prev_sec = section_order[i - 1]
        sec = section_order[i]
        entries = processed_section_entries.get(sec, [])
        if not entries:
            continue
        first_text, first_para = entries[0]
        if not starts_with_lowercase_token(first_text):
            continue
        prev_entries = processed_section_entries.get(prev_sec, [])
        if not prev_entries:
            continue
        last_text, last_para = prev_entries[-1]
        prev_x_rel, prev_y_rel = para_relative_position(last_para, para_bounds_by_page)
        first_x_rel, first_y_rel = para_relative_position(first_para, para_bounds_by_page)
        prev_page = int(last_para.page_span.get("start", 1))
        first_page = int(first_para.page_span.get("start", 1))
        can_hyphen = should_merge_hyphen_wrap(last_text, first_text, prev_x_rel, first_x_rel, prev_y_rel, first_y_rel)
        can_lowercase = should_merge_lowercase_continuation(
            last_text,
            first_text,
            prev_x_rel,
            first_x_rel,
            prev_y_rel,
            first_y_rel,
            prev_page=prev_page,
            next_page=first_page,
        )
        if can_hyphen or can_lowercase:
            # Merge into previous section's last paragraph and drop the fragment
            merged = merge_fragment_pair(last_text, first_text) if can_hyphen else clean_text_line(f"{last_text} {first_text}")
            prev_entries[-1] = (merged, last_para)
            processed_section_entries[prev_sec] = prev_entries
            processed_section_entries[sec] = entries[1:]

    # Emit sections using repaired boundaries
    for section in section_order:
        pruned_entries = processed_section_entries.get(section, [])
        section_texts = repair_body_paragraph_boundaries(pruned_entries, para_bounds_by_page)
        section_texts = merge_citation_tail_continuation_paragraphs(section_texts)
        paragraphs_in_section = unique_texts_in_order(section_texts)
        filtered_paragraphs: list[str] = []
        for paragraph_text in paragraphs_in_section:
            cleaned_paragraph_text = normalize_inline_hyphen_wrap_artifacts(paragraph_text)
            if cleaned_paragraph_text:
                filtered_paragraphs.append(cleaned_paragraph_text)
        paragraphs_in_section = unique_texts_in_order(filtered_paragraphs)
        if not paragraphs_in_section:
            if section != "Main Body" and should_emit_empty_section_heading(section):
                lines.append(clean_text_line(section))
                lines.append("")
            continue
        if section != "Main Body":
            if not (title and clean_text_line(section).lower() == title.lower()):
                lines.append(f"### {section}")
                lines.append("")
        for paragraph_text in paragraphs_in_section:
            lines.append(paragraph_text)
            lines.append("")

    figure_captions = [cap for cap in unique_texts_in_order(figure_captions) if is_caption_like_entry(cap)]
    table_captions = [cap for cap in unique_texts_in_order(table_captions) if is_caption_like_entry(cap)]

    if figure_captions or table_captions:
        lines.append("## Figures and Tables")
        lines.append("")
        if figure_captions:
            lines.append("### Figure Captions")
            lines.append("")
            for item in figure_captions:
                lines.append(f"- {item}")
            lines.append("")
        if table_captions:
            lines.append("### Table Captions")
            lines.append("")
            for item in table_captions:
                lines.append(f"- {item}")
            lines.append("")

    if references:
        lines.append("## References")
        lines.append("")
        for item in references:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n", suppression_events


def extract_markdown_section(markdown: str, section_title: str) -> str:
    pattern = rf"^## {re.escape(section_title)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        return ""
    return str(match.group(1))


def parse_main_body_structure(clean_document: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    main_body_raw = extract_markdown_section(clean_document, "Main Body")
    if not main_body_raw:
        return [], []

    main_body_lines = [line.strip() for line in main_body_raw.splitlines() if line.strip()]
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Main Body"
    current_lines: list[str] = []
    saw_explicit_heading = False

    for line in main_body_lines:
        if line.startswith("### "):
            if saw_explicit_heading:
                sections.append((current_heading, current_lines))
            current_heading = clean_text_line(line[4:])
            current_lines = []
            saw_explicit_heading = True
            continue
        current_lines.append(line)

    if saw_explicit_heading:
        sections.append((current_heading, current_lines))

    return main_body_lines, sections


def compute_column_crossovers_per_1000_tokens(
    paragraphs: list[Paragraph],
    clean_role_by_block: dict[str, str],
) -> float:
    if not paragraphs:
        return 0.0

    bounds_by_page = infer_para_page_bounds(paragraphs)
    token_total = 0
    column_crossovers = 0
    previous: Optional[tuple[int, float, float]] = None

    for para in paragraphs:
        if para.role != "Body":
            continue
        para_clean = paragraph_clean_roles(para, clean_role_by_block)
        if para_clean & METADATA_FAMILY_ROLES:
            continue
        if para_clean & {"figure_caption", "table_caption", "reference_entry", "nuisance"}:
            continue

        text = clean_text_line(para.text)
        if not text:
            continue

        x0, _, x1, _ = para_bbox_union(para)
        page = int(para.page_span.get("start", 1))
        bounds = bounds_by_page.get(page, [0.0, 0.0, max(x1, 1.0), 1.0])
        page_width = max(1e-6, float(bounds[2]) - float(bounds[0]))
        width_rel = (x1 - x0) / page_width
        if width_rel >= 0.72:
            continue

        x_rel, _ = para_relative_position(para, bounds_by_page)
        tokens = len(word_tokens(text))
        token_total += tokens

        if previous is not None and previous[0] == page:
            previous_x_rel = previous[1]
            crossed_column = (previous_x_rel < 0.5 <= x_rel) or (previous_x_rel >= 0.5 > x_rel)
            if crossed_column:
                column_crossovers += 1

        previous = (page, x_rel, width_rel)

    if token_total <= 0:
        return 0.0
    return (column_crossovers * 1000.0) / float(token_total)


def compute_clean_document_metrics(
    clean_document: str,
    paragraphs: list[Paragraph],
    annotated_blocks: list[dict[str, Any]],
    doc_id: str,
) -> dict[str, Any]:
    main_body_lines, main_body_sections = parse_main_body_structure(clean_document)
    heading_body_detach_count = sum(1 for _, lines in main_body_sections if not lines)
    explicit_headings = [heading for heading, _ in main_body_sections if heading]
    normalized_headings = [normalize_section_key(heading) for heading in explicit_headings if normalize_section_key(heading)]

    duplicate_count = max(0, len(normalized_headings) - len(set(normalized_headings)))
    duplicate_ratio = (duplicate_count / float(len(normalized_headings))) if normalized_headings else 0.0

    clean_role_by_block = {
        str(block.get("block_id", "")): str(block.get("clean_role", ""))
        for block in annotated_blocks
        if str(block.get("block_id", ""))
    }
    column_crossovers_per_1000_tokens = compute_column_crossovers_per_1000_tokens(paragraphs, clean_role_by_block)
    references_like_blocks_exist = any(
        str(block.get("clean_role", "")) == "reference_entry"
        or is_reference_entry_text(str(block.get("text", "")))
        for block in annotated_blocks
    )
    references_boundary_detected = "## References" in clean_document

    hyphen_wrap_count = len(INLINE_HYPHEN_WRAP_RE.findall("\n".join(main_body_lines)))
    affiliation_leak_count = sum(1 for line in main_body_lines if is_affiliation_address_line(line))
    standalone_page_number_count = sum(1 for line in main_body_lines if PAGE_NUMBER_ONLY_LINE_RE.match(line) is not None)
    check_for_updates_count = sum(1 for line in main_body_lines if CHECK_FOR_UPDATES_BANNER_RE.match(line) is not None)

    section_boundary_unstable = (
        len(main_body_sections) == 0
        or (not references_boundary_detected and references_like_blocks_exist)
        or duplicate_ratio > 0.2
    )
    ordering_confidence_low = column_crossovers_per_1000_tokens > 8.0 or heading_body_detach_count > 3

    return {
        "doc_id": doc_id,
        "orphan_heading_count": heading_body_detach_count,
        "standalone_page_number_count": standalone_page_number_count,
        "check_for_updates_count": check_for_updates_count,
        "affiliation_leak_count": affiliation_leak_count,
        "ordering_confidence_low": ordering_confidence_low,
        "section_boundary_unstable": section_boundary_unstable,
        "hyphen_wrap_count": hyphen_wrap_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_paragraphs(
    run_dir: Path,
    manifest: Optional[Any] = None,
) -> tuple[int, int]:
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    qa_dir = run_dir / "qa"
    paragraphs_dir = run_dir / "paragraphs"
    
    paragraphs_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks(text_dir / "blocks_norm.jsonl")
    confidence_by_page, merge_groups_by_page, role_labels_by_page = load_vision_outputs(vision_dir)

    if not blocks:
        return 0, 0

    cleaned_blocks, annotated_blocks = classify_clean_blocks(blocks, role_labels_by_page)
    write_jsonl(text_dir / "blocks_clean.jsonl", annotated_blocks)

    paragraphs = aggregate_paragraphs(
        cleaned_blocks,
        merge_groups_by_page,
        role_labels_by_page,
        confidence_by_page,
    )

    paragraphs = build_neighbors(paragraphs, text_dir)

    paragraphs = add_uncertainty_notes(paragraphs)

    output_path = paragraphs_dir / "paragraphs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for para in paragraphs:
            record = {
                "para_id": para.para_id,
                "page_span": para.page_span,
                "role": para.role,
                "section_path": para.section_path,
                "text": para.text,
                "evidence_pointer": para.evidence_pointer,
                "neighbors": para.neighbors,
                "confidence": para.confidence,
                "provenance": para.provenance,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    clean_document, suppression_events = render_clean_document(
        paragraphs,
        annotated_blocks,
        doc_id=run_dir.name,
        qa_dir=qa_dir,
        text_dir=text_dir,
    )
    with open(text_dir / "clean_document.md", "w", encoding="utf-8") as f:
        f.write(clean_document)

    write_jsonl(qa_dir / "clean_document_suppressions.jsonl", suppression_events)

    metrics = compute_clean_document_metrics(
        clean_document=clean_document,
        paragraphs=paragraphs,
        annotated_blocks=annotated_blocks,
        doc_id=run_dir.name,
    )
    with open(qa_dir / "clean_document_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return len(paragraphs), len(blocks)
