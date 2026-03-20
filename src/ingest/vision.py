"""Vision Structure Corrector with SiliconFlow API and fallback chain.

Contract: .sisyphus/plans/pdf-blueprint-contracts.md (lines 46-74, 276-308)

Output schema per pXXX_out.json:
- page, reading_order, merge_groups, role_labels, confidence, fallback_used
"""

import json
import os
import re
import base64
import io
import unicodedata
import urllib.request
import urllib.error
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .contract_guard import guard_model_output, safe_json_value
from .manifest import Manifest, load_manifest
from .paragraphs import (
    clean_text_line,
    is_common_unnumbered_section_heading,
    looks_like_document_title_candidate,
    split_embedded_section_heading,
)
from .qa_telemetry import append_fault_events, append_jsonl_event
from .vision_faults import FaultEvent, VisionLLMCallEvent
from .vision_runtime import (
    VisionImageDataUrlCache,
    VisionRequestBudget,
    VisionRuntimeContext,
    append_vision_llm_call_event,
    append_vision_runtime_event,
    resolve_vision_request_budget,
)


ROLE_LABELS = frozenset({
    "Body", "Heading", "FigureCaption", "TableCaption",
    "Footnote", "ReferenceList", "Sidebar", "HeaderFooter",
})

CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.|Table)\b", re.IGNORECASE)
DOI_LINE_RE = re.compile(r"^(?:https?://doi\.org/\S+|10\.\d{4,}\S*)$", re.IGNORECASE)
FRONT_MATTER_LABEL_RE = re.compile(
    r"^(?:article|review|research article|brief communication|perspective)$",
    re.IGNORECASE,
)
REGION_HEADING_LEAD_STOPWORDS = frozenset({
    "a",
    "an",
    "after",
    "among",
    "based",
    "by",
    "furthermore",
    "here",
    "however",
    "in",
    "meanwhile",
    "moreover",
    "next",
    "our",
    "the",
    "then",
    "there",
    "therefore",
    "these",
    "this",
    "to",
    "we",
})
MICROBLOCK_STOPWORDS = frozenset({
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "was",
    "were",
    "with",
})

SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SILICONFLOW_FALLBACK_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
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
                "Vision stage preflight failed: no SiliconFlow API credential found. "
                f"Set one of {', '.join(REQUIRED_API_KEY_ENV_NAMES)} before running vision/full stage."
            ),
        }
    if not endpoint.startswith("https://") or "/chat/completions" not in endpoint:
        return False, {
            "error_type": "invalid_endpoint",
            "endpoint": endpoint,
            "model": model,
            "message": (
                "Vision stage preflight failed: SILICONFLOW_ENDPOINT is not a valid chat-completions HTTPS endpoint. "
                f"Current value: {endpoint!r}. Expected format similar to {SILICONFLOW_ENDPOINT!r}."
            ),
        }
    return True, {
        "error_type": "none",
        "endpoint": endpoint,
        "model": model,
        "message": "ok",
    }


def is_request_contract_error(error_type: str, http_status: Any) -> bool:
    error_l = str(error_type or "").lower()
    status_num = int(http_status) if isinstance(http_status, int) else None
    return status_num == 400 or "request_contract" in error_l


def is_transient_retryable_error(error_type: str, http_status: Any) -> bool:
    error_l = str(error_type or "").lower()
    status_num = int(http_status) if isinstance(http_status, int) else None
    if is_request_contract_error(error_l, status_num):
        return False
    if "timeout" in error_l or "url_error" in error_l or "network_error" in error_l:
        return True
    if "transient_http_error" in error_l:
        return True
    return status_num in {408, 429, 500, 502, 503, 504}


@dataclass
class BlockCandidate:
    block_id: str
    text: str
    bbox_pt: list[float]
    bbox_px: list[int]
    column_guess: int
    is_heading_candidate: bool
    is_header_footer_candidate: bool
    source_block_ids: list[str] = field(default_factory=list)
    source_line_span: list[int] = field(default_factory=list)
    source_kind: str = "raw"


@dataclass
class VisionOutput:
    page: int
    reading_order: list[str]
    merge_groups: list[dict[str, Any]]
    role_labels: dict[str, str]
    confidence: float
    fallback_used: bool
    source: str = "model"
    embedded_headings: list[dict[str, Any]] = field(default_factory=list)
    embedded_heading_reviewed_block_ids: list[str] = field(default_factory=list)
    mixed_group_reviews: list[dict[str, Any]] = field(default_factory=list)
    mixed_group_reviewed_block_ids: list[str] = field(default_factory=list)


def classify_page_layout_mode(blocks: list["BlockCandidate"]) -> str:
    if not blocks:
        return "direct"

    text_lengths = [len(clean_text_line(block.text)) for block in blocks]
    avg_text_len = sum(text_lengths) / max(1, len(text_lengths))
    short_blocks = sum(1 for length in text_lengths if length <= 24)
    short_ratio = short_blocks / max(1, len(blocks))
    small_area_blocks = 0
    graphic_microblocks = 0
    for block in blocks:
        width_pt = max(0.0, float(block.bbox_pt[2]) - float(block.bbox_pt[0]))
        height_pt = max(0.0, float(block.bbox_pt[3]) - float(block.bbox_pt[1]))
        if width_pt * height_pt <= 3500.0:
            small_area_blocks += 1
        clean = clean_text_line(block.text)
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?", clean)
        numeric_tokens = sum(1 for word in words if re.fullmatch(r"\d+(?:\.\d+)?", word) is not None)
        short_tokens = sum(1 for word in words if len(word) <= 3)
        if (
            (len(words) <= 4 and short_tokens >= max(1, len(words) - 1))
            or (numeric_tokens >= 1 and len(words) <= 6)
            or (
                len(words) <= 6
                and re.search(r"\b(?:vs\.?|p-?value|avg|sil|width|hypoxia|inflammation)\b", clean, flags=re.IGNORECASE)
                is not None
            )
        ):
            graphic_microblocks += 1
    small_area_ratio = small_area_blocks / max(1, len(blocks))
    graphic_ratio = graphic_microblocks / max(1, len(blocks))

    if len(blocks) >= 90 and avg_text_len <= 40 and short_ratio >= 0.6:
        return "hierarchical"
    if len(blocks) >= 80 and avg_text_len <= 45 and (
        short_ratio >= 0.55 or graphic_ratio >= 0.4 or small_area_ratio >= 0.45
    ):
        return "hierarchical"
    return "direct"


def classify_microblock_cluster_coherence(cluster: dict[str, Any]) -> str:
    texts = cluster.get("texts", [])
    if not isinstance(texts, list):
        return "uncertain"

    cleaned = [clean_text_line(str(text)) for text in texts if clean_text_line(str(text))]
    if not cleaned:
        return "uncertain"

    joined = clean_text_line(" ".join(cleaned))
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?", joined)
    if not words:
        return "uncertain"

    short_tokens = sum(1 for word in words if len(word) <= 3)
    numeric_tokens = sum(1 for word in words if re.fullmatch(r"\d+(?:\.\d+)?", word) is not None)
    stopword_tokens = sum(1 for word in words if word.lower() in MICROBLOCK_STOPWORDS)
    punctuation_sentences = joined.count(".") + joined.count(";") + joined.count(":")

    short_ratio = short_tokens / max(1, len(words))
    numeric_ratio = numeric_tokens / max(1, len(words))
    stopword_ratio = stopword_tokens / max(1, len(words))

    if len(cleaned) >= 5 and short_ratio >= 0.5 and stopword_ratio <= 0.2:
        return "non_narrative_graphic"
    if numeric_ratio >= 0.25 and stopword_ratio <= 0.2:
        return "non_narrative_graphic"
    if punctuation_sentences >= 1 and stopword_ratio >= 0.12 and len(words) >= 10:
        return "narrative"
    if len(cleaned) <= 3 and len(words) >= 10 and stopword_ratio >= 0.12:
        return "narrative"
    return "uncertain"


def pt_to_px(bbox_pt: list[float], dpi: int, scale: float) -> list[int]:
    zoom = dpi / 72.0 * scale
    return [int(c * zoom) for c in bbox_pt]


def load_blocks(blocks_path: Path, dpi: int, scale: float) -> dict[int, list[BlockCandidate]]:
    result: dict[int, list[BlockCandidate]] = {}
    if not blocks_path.exists():
        return result
    with open(blocks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if not all(k in d for k in ["block_id", "page", "bbox_pt", "text"]):
                    continue
                bbox_pt = [float(x) for x in d["bbox_pt"][:4]]
                b = BlockCandidate(
                    block_id=d["block_id"],
                    text=d.get("text", ""),
                    bbox_pt=bbox_pt,
                    bbox_px=pt_to_px(bbox_pt, dpi, scale),
                    column_guess=d.get("column_guess", 1),
                    is_heading_candidate=d.get("is_heading_candidate", False),
                    is_header_footer_candidate=d.get("is_header_footer_candidate", False),
                    source_block_ids=[str(d["block_id"])],
                )
                result.setdefault(d["page"], []).append(b)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return result


def load_document_layout_profile(profile_path: Path) -> dict[str, Any]:
    if not profile_path.exists():
        return {}
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_layout_analysis_hints(layout_path: Path) -> dict[int, list[list[str]]]:
    if not layout_path.exists():
        return {}
    try:
        with open(layout_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    hints = data.get("paragraph_regrouping_hints", {})
    if not isinstance(hints, dict):
        return {}
    return {
        int(page): [
            [str(block_id).strip() for block_id in group if str(block_id).strip()]
            for group in groups
            if isinstance(group, list)
        ]
        for page, groups in hints.items()
        if isinstance(groups, list)
    }


def load_block_line_records(text_dir: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    lines_path = text_dir / "block_lines.jsonl"
    if not lines_path.exists():
        return records
    with open(lines_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            block_id = str(record.get("block_id", "")).strip()
            lines = record.get("lines", [])
            if not block_id or not isinstance(lines, list):
                continue
            valid_lines = [line for line in lines if isinstance(line, dict)]
            if valid_lines:
                records[block_id] = valid_lines
    return records


def _union_bbox_pt(bboxes: list[list[float]]) -> list[float]:
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def _union_bbox_px(bboxes: list[list[int]]) -> list[int]:
    if not bboxes:
        return [0, 0, 0, 0]
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def _line_font_value(line: dict[str, Any], key: str, default: Any) -> Any:
    font_stats = line.get("font_stats", {})
    if isinstance(font_stats, dict) and key in font_stats:
        return font_stats.get(key, default)
    return line.get(key, default)


def _clean_block_line_records(line_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for line in line_records:
        if not isinstance(line, dict):
            continue
        text = clean_text_line(str(line.get("text", "")))
        bbox_pt = line.get("bbox_pt", [])
        if not text or not isinstance(bbox_pt, list) or len(bbox_pt) < 4:
            continue
        try:
            bbox_clean = [float(value) for value in bbox_pt[:4]]
        except (TypeError, ValueError):
            continue
        cleaned.append(
            {
                "line_index": int(line.get("line_index", len(cleaned))),
                "text": text,
                "bbox_pt": bbox_clean,
                "font_stats": {
                    "avg_size": float(_line_font_value(line, "avg_size", 0.0) or 0.0),
                    "is_bold": bool(_line_font_value(line, "is_bold", False)),
                    "dominant_font": str(_line_font_value(line, "dominant_font", "") or ""),
                },
            }
        )
    return cleaned


def _line_matches_heading_profile(line: dict[str, Any], document_profile: Optional[dict[str, Any]]) -> bool:
    if not isinstance(document_profile, dict):
        return False
    heading_profile = document_profile.get("heading_font_profile", {})
    if not isinstance(heading_profile, dict):
        return False
    avg_size = float(line.get("font_stats", {}).get("avg_size", 0.0) or 0.0)
    is_bold = bool(line.get("font_stats", {}).get("is_bold", False))
    dominant_font = str(line.get("font_stats", {}).get("dominant_font", "") or "")
    profile_size = float(heading_profile.get("avg_size", 0.0) or 0.0)
    profile_fonts = {
        str(value).strip()
        for value in heading_profile.get("dominant_fonts", [])
        if str(value).strip()
    }
    bold_ratio = float(heading_profile.get("bold_ratio", 0.0) or 0.0)
    size_match = profile_size > 0 and avg_size >= max(profile_size * 0.88, profile_size - 1.0)
    font_match = bool(profile_fonts) and dominant_font in profile_fonts
    bold_match = bold_ratio >= 0.5 and is_bold
    return size_match and (font_match or bold_match)


def _line_matches_body_profile(line: dict[str, Any], document_profile: Optional[dict[str, Any]]) -> bool:
    if not isinstance(document_profile, dict):
        return False
    body_profile = document_profile.get("body_font_profile", {})
    if not isinstance(body_profile, dict):
        return False
    avg_size = float(line.get("font_stats", {}).get("avg_size", 0.0) or 0.0)
    dominant_font = str(line.get("font_stats", {}).get("dominant_font", "") or "")
    profile_size = float(body_profile.get("avg_size", 0.0) or 0.0)
    profile_fonts = {
        str(value).strip()
        for value in body_profile.get("dominant_fonts", [])
        if str(value).strip()
    }
    size_match = profile_size > 0 and abs(avg_size - profile_size) <= max(0.8, profile_size * 0.12)
    font_match = bool(profile_fonts) and dominant_font in profile_fonts
    return size_match and (font_match or not profile_fonts)


def _line_text_looks_like_heading(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    if CAPTION_RE.match(clean):
        return False
    if clean.endswith((".", "?", "!", ";", ",")):
        return False
    words = clean.split()
    if not words or len(words) > 20 or len(clean) > 160:
        return False
    if len(words) == 1 and len(words[0]) <= 3 and not is_common_unnumbered_section_heading(words[0]):
        return False
    lowercase_words = sum(1 for word in words if word[:1].islower())
    return lowercase_words <= max(2, len(words) // 3)


def _looks_like_compact_graphic_label_cluster(block: BlockCandidate, lines: list[dict[str, Any]]) -> bool:
    if len(lines) < 2 or len(lines) > 4:
        return False
    width_pt = max(0.0, float(block.bbox_pt[2]) - float(block.bbox_pt[0]))
    if width_pt > 90.0:
        return False
    cleaned_texts = [clean_text_line(str(line.get("text", ""))) for line in lines]
    cleaned_texts = [text for text in cleaned_texts if text]
    if not cleaned_texts:
        return False
    joined = clean_text_line(" ".join(cleaned_texts))
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?", joined)
    if len(words) > 7 or len(joined) > 48:
        return False
    same_font = len(
        {
            str(line.get("font_stats", {}).get("dominant_font", "") or "")
            for line in lines
            if str(line.get("font_stats", {}).get("dominant_font", "") or "")
        }
    ) <= 1
    all_bold = all(bool(line.get("font_stats", {}).get("is_bold", False)) for line in lines)
    first_words = words[:1]
    if not first_words or len(first_words[0]) > 3:
        return False
    short_tokens = sum(1 for word in words if len(word) <= 3)
    return same_font and all_bold and short_tokens >= max(1, len(words) // 2)


def _same_heading_line_style(line: dict[str, Any], anchor: dict[str, Any]) -> bool:
    line_stats = line.get("font_stats", {})
    anchor_stats = anchor.get("font_stats", {})
    line_size = float(line_stats.get("avg_size", 0.0) or 0.0)
    anchor_size = float(anchor_stats.get("avg_size", 0.0) or 0.0)
    if anchor_size <= 0 or line_size <= 0:
        return False
    if abs(line_size - anchor_size) > max(0.8, anchor_size * 0.12):
        return False
    line_font = str(line_stats.get("dominant_font", "") or "")
    anchor_font = str(anchor_stats.get("dominant_font", "") or "")
    if line_font and anchor_font and line_font != anchor_font:
        return False
    if bool(line_stats.get("is_bold", False)) != bool(anchor_stats.get("is_bold", False)):
        return False
    return True


def _line_text_looks_like_front_matter_label(text: str) -> bool:
    return FRONT_MATTER_LABEL_RE.match(clean_text_line(text)) is not None


def _line_text_looks_like_doi(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean:
        return False
    return DOI_LINE_RE.match(clean) is not None


def _looks_like_page1_author_name_block(lines: list[dict[str, Any]]) -> bool:
    if len(lines) < 2:
        return False
    first_text = clean_text_line(str(lines[0].get("text", "")))
    second_text = clean_text_line(str(lines[1].get("text", "")))
    if not first_text or not second_text:
        return False
    if _line_text_looks_like_front_matter_label(first_text) or _line_text_looks_like_doi(first_text):
        return False
    first_words = first_text.split()
    if not (2 <= len(first_words) <= 4):
        return False
    if any(any(ch.isdigit() for ch in word) for word in first_words):
        return False
    if any(not word[:1].isupper() for word in first_words):
        return False
    digit_count = sum(1 for ch in second_text if ch.isdigit())
    comma_count = second_text.count(",")
    return digit_count >= 4 and comma_count >= 2


def _build_line_split_block(
    block: BlockCandidate,
    lines: list[dict[str, Any]],
    suffix: str,
    is_heading_candidate: bool,
    source_kind: str = "line_split",
) -> BlockCandidate:
    bbox_pt = _union_bbox_pt([list(line["bbox_pt"]) for line in lines])
    block_width_pt = max(1.0, float(block.bbox_pt[2]) - float(block.bbox_pt[0]))
    block_height_pt = max(1.0, float(block.bbox_pt[3]) - float(block.bbox_pt[1]))
    scale_x = (float(block.bbox_px[2]) - float(block.bbox_px[0])) / block_width_pt
    scale_y = (float(block.bbox_px[3]) - float(block.bbox_px[1])) / block_height_pt
    bbox_px = [
        int(round(float(block.bbox_px[0]) + (float(bbox_pt[0]) - float(block.bbox_pt[0])) * scale_x)),
        int(round(float(block.bbox_px[1]) + (float(bbox_pt[1]) - float(block.bbox_pt[1])) * scale_y)),
        int(round(float(block.bbox_px[0]) + (float(bbox_pt[2]) - float(block.bbox_pt[0])) * scale_x)),
        int(round(float(block.bbox_px[1]) + (float(bbox_pt[3]) - float(block.bbox_pt[1])) * scale_y)),
    ]
    return BlockCandidate(
        block_id=f"{block.block_id}{suffix}",
        text=" ".join(line["text"] for line in lines if line.get("text")),
        bbox_pt=bbox_pt,
        bbox_px=bbox_px,
        column_guess=block.column_guess,
        is_heading_candidate=is_heading_candidate,
        is_header_footer_candidate=False,
        source_block_ids=list(block.source_block_ids) or [block.block_id],
        source_line_span=[int(lines[0].get("line_index", 0)), int(lines[-1].get("line_index", 0))],
        source_kind=source_kind,
    )


def split_block_for_vision_by_lines(
    block: BlockCandidate,
    line_records: list[dict[str, Any]],
    page: Optional[int] = None,
    document_profile: Optional[dict[str, Any]] = None,
) -> list[BlockCandidate]:
    lines = _clean_block_line_records(line_records)
    if len(lines) < 2:
        return [block]
    if _looks_like_compact_graphic_label_cluster(block, lines):
        return [block]

    if page == 1:
        if _looks_like_page1_author_name_block(lines):
            return [block]
        article_idx = 0 if _line_text_looks_like_front_matter_label(lines[0].get("text", "")) else None
        doi_idx = next(
            (idx for idx, line in enumerate(lines[:3]) if _line_text_looks_like_doi(line.get("text", ""))),
            None,
        )
        if doi_idx is not None and doi_idx < len(lines) - 1:
            title_start = doi_idx + 1
            title_lines = lines[title_start:]
            title_text = " ".join(
                clean_text_line(str(line.get("text", "")))
                for line in title_lines
                if clean_text_line(str(line.get("text", "")))
            )
            title_sizes = [
                float(line.get("font_stats", {}).get("avg_size", 0.0) or 0.0)
                for line in title_lines
                if float(line.get("font_stats", {}).get("avg_size", 0.0) or 0.0) > 0
            ]
            lead_lines = lines[:title_start]
            lead_sizes = [
                float(line.get("font_stats", {}).get("avg_size", 0.0) or 0.0)
                for line in lead_lines
                if float(line.get("font_stats", {}).get("avg_size", 0.0) or 0.0) > 0
            ]
            title_avg = sum(title_sizes) / len(title_sizes) if title_sizes else 0.0
            lead_avg = sum(lead_sizes) / len(lead_sizes) if lead_sizes else 0.0
            title_lead = title_lines[0] if title_lines else None
            title_typography_signal = bool(
                title_lead and (
                    _line_matches_heading_profile(title_lead, document_profile)
                    or (lead_avg > 0 and title_avg >= lead_avg * 1.2)
                    or any(_line_is_bold(line) for line in title_lines)
                )
            )
            if title_text and title_typography_signal and looks_like_document_title_candidate(title_text):
                split_blocks: list[BlockCandidate] = []
                split_index = 0
                if article_idx == 0:
                    split_blocks.append(
                        _build_line_split_block(
                            block,
                            [lines[0]],
                            f"__fm{split_index}",
                            False,
                            source_kind="front_matter_split",
                        )
                    )
                    split_index += 1
                split_blocks.append(
                    _build_line_split_block(
                        block,
                        [lines[doi_idx]],
                        f"__fm{split_index}",
                        False,
                        source_kind="front_matter_split",
                    )
                )
                split_index += 1
                split_blocks.append(
                    _build_line_split_block(
                        block,
                        title_lines,
                        f"__fm{split_index}",
                        True,
                        source_kind="front_matter_split",
                    )
                )
                return split_blocks

    first_line = lines[0]
    second_line = lines[1]
    first_size = float(first_line.get("font_stats", {}).get("avg_size", 0.0) or 0.0)
    second_size = float(second_line.get("font_stats", {}).get("avg_size", 0.0) or 0.0)
    first_bold = bool(first_line.get("font_stats", {}).get("is_bold", False))
    second_bold = bool(second_line.get("font_stats", {}).get("is_bold", False))
    first_font = str(first_line.get("font_stats", {}).get("dominant_font", "") or "")
    second_font = str(second_line.get("font_stats", {}).get("dominant_font", "") or "")

    typography_signal = (
        first_size >= max(second_size * 1.12, second_size + 0.6)
        or (first_bold and not second_bold)
        or (first_font and second_font and first_font != second_font)
        or _line_matches_heading_profile(first_line, document_profile)
    )
    if not (_line_text_looks_like_heading(first_line.get("text", "")) and typography_signal):
        return [block]

    if _line_matches_body_profile(first_line, document_profile) and not _line_matches_heading_profile(first_line, document_profile):
        return [block]

    heading_lines = [first_line]
    body_start = 1
    for idx in range(1, len(lines)):
        candidate = lines[idx]
        if _line_text_looks_like_heading(candidate.get("text", "")) and _same_heading_line_style(candidate, first_line):
            heading_lines.append(candidate)
            body_start = idx + 1
            continue
        break

    if body_start <= 0 or body_start >= len(lines):
        return [block]

    body_lines = lines[body_start:]
    if not body_lines:
        return [block]

    return [
        _build_line_split_block(block, heading_lines, "__nl0", True),
        _build_line_split_block(block, body_lines, "__nl1", False),
    ]


def normalize_blocks_for_vision_page(
    page: int,
    blocks: list[BlockCandidate],
    document_profile: Optional[dict[str, Any]] = None,
    paragraph_regrouping_hints_by_page: Optional[dict[int, list[list[str]]]] = None,
    block_lines_by_block: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> tuple[list[BlockCandidate], dict[str, str]]:
    header_band = document_profile.get("header_band_pt") if isinstance(document_profile, dict) else None
    footer_band = document_profile.get("footer_band_pt") if isinstance(document_profile, dict) else None
    hidden_role_labels: dict[str, str] = {}
    visible_blocks: list[BlockCandidate] = []
    for block in blocks:
        if _bbox_pt_intersects_band(block.bbox_pt, header_band) or _bbox_pt_intersects_band(block.bbox_pt, footer_band):
            for source_block_id in (block.source_block_ids or [block.block_id]):
                hidden_role_labels[str(source_block_id)] = "HeaderFooter"
            continue
        source_ids = list(block.source_block_ids) if block.source_block_ids else [block.block_id]
        normalized_block = BlockCandidate(
            block_id=block.block_id,
            text=block.text,
            bbox_pt=list(block.bbox_pt),
            bbox_px=list(block.bbox_px),
            column_guess=block.column_guess,
            is_heading_candidate=block.is_heading_candidate,
            is_header_footer_candidate=block.is_header_footer_candidate,
            source_block_ids=source_ids,
        )
        visible_blocks.extend(
            split_block_for_vision_by_lines(
                normalized_block,
                (block_lines_by_block or {}).get(block.block_id, []),
                page=page,
                document_profile=document_profile,
            )
        )

    if not visible_blocks:
        return [], hidden_role_labels

    regrouping_hints = (paragraph_regrouping_hints_by_page or {}).get(page, [])
    block_by_id = {block.block_id: block for block in visible_blocks}
    grouped_ids: set[str] = set()
    normalized_blocks: list[BlockCandidate] = []

    for hint_idx, group in enumerate(regrouping_hints):
        hinted_ids = [str(block_id).strip() for block_id in group if str(block_id).strip()]
        hinted_ids = list(dict.fromkeys(hinted_ids))
        if len(hinted_ids) < 2:
            continue
        if any(block_id not in block_by_id for block_id in hinted_ids):
            continue
        if any(block_id in grouped_ids for block_id in hinted_ids):
            continue
        grouped_blocks = sorted(
            [block_by_id[block_id] for block_id in hinted_ids],
            key=lambda block: (block.bbox_pt[1], block.bbox_pt[0]),
        )
        merged_source_ids: list[str] = []
        for grouped_block in grouped_blocks:
            merged_source_ids.extend(grouped_block.source_block_ids or [grouped_block.block_id])
        normalized_blocks.append(
            BlockCandidate(
                block_id=f"p{page:03d}_rg{hint_idx:03d}",
                text=" ".join(clean_text_line(block.text) for block in grouped_blocks if clean_text_line(block.text)),
                bbox_pt=_union_bbox_pt([block.bbox_pt for block in grouped_blocks]),
                bbox_px=_union_bbox_px([block.bbox_px for block in grouped_blocks]),
                column_guess=grouped_blocks[0].column_guess,
                is_heading_candidate=any(block.is_heading_candidate for block in grouped_blocks),
                is_header_footer_candidate=False,
                source_block_ids=merged_source_ids,
            )
        )
        grouped_ids.update(hinted_ids)

    for block in sorted(visible_blocks, key=lambda item: (item.bbox_pt[1], item.bbox_pt[0])):
        if block.block_id in grouped_ids:
            continue
        normalized_blocks.append(block)

    return normalized_blocks, hidden_role_labels


def expand_vision_output_to_raw_blocks(
    output: VisionOutput,
    normalized_blocks: list[BlockCandidate],
    raw_blocks: list[BlockCandidate],
    hidden_role_labels: Optional[dict[str, str]] = None,
) -> VisionOutput:
    normalized_by_id = {block.block_id: block for block in normalized_blocks}
    raw_ids = {block.block_id for block in raw_blocks}

    def expand_block_id(block_id: str) -> list[str]:
        block = normalized_by_id.get(block_id)
        if block is None:
            return [block_id] if block_id in raw_ids else []
        source_ids = block.source_block_ids or [block.block_id]
        return [source_id for source_id in source_ids if source_id in raw_ids]

    expanded_reading_order: list[str] = []
    for block_id in output.reading_order:
        expanded_reading_order.extend(expand_block_id(block_id))

    expanded_merge_groups: list[dict[str, Any]] = []
    for group in output.merge_groups:
        expanded_block_ids: list[str] = []
        for block_id in group.get("block_ids", []):
            expanded_block_ids.extend(expand_block_id(str(block_id)))
        expanded_block_ids = list(dict.fromkeys(expanded_block_ids))
        expanded_merge_groups.append(
            {
                **group,
                "block_ids": expanded_block_ids,
            }
        )

    role_candidates_by_raw: dict[str, list[str]] = {}
    front_matter_source_blocks: set[str] = set()
    for block_id, role in output.role_labels.items():
        normalized_block = normalized_by_id.get(block_id)
        if normalized_block is not None and normalized_block.source_kind == "front_matter_split":
            for source_block_id in expand_block_id(block_id):
                front_matter_source_blocks.add(source_block_id)
            continue
        for source_block_id in expand_block_id(block_id):
            role_candidates_by_raw.setdefault(source_block_id, []).append(str(role))

    def collapse_role_candidates(roles: list[str]) -> str:
        unique_roles = list(dict.fromkeys(role for role in roles if role))
        if not unique_roles:
            return "Body"
        if len(unique_roles) == 1:
            return unique_roles[0]
        if "Body" in unique_roles:
            return "Body"
        for preferred in ("FigureCaption", "TableCaption", "ReferenceList", "Footnote", "Sidebar", "Heading"):
            if preferred in unique_roles:
                return preferred
        return unique_roles[0]

    expanded_role_labels: dict[str, str] = {
        raw_block_id: collapse_role_candidates(roles)
        for raw_block_id, roles in role_candidates_by_raw.items()
    }
    for raw_block_id in front_matter_source_blocks:
        expanded_role_labels.setdefault(raw_block_id, "Body")
    if hidden_role_labels:
        expanded_role_labels.update(hidden_role_labels)

    def remap_review_block_id(item: dict[str, Any]) -> dict[str, Any]:
        block_id = str(item.get("block_id", "")).strip()
        expanded_ids = expand_block_id(block_id)
        if not expanded_ids:
            return item
        remapped = dict(item)
        remapped["block_id"] = expanded_ids[0]
        return remapped

    remapped_embedded_headings = [remap_review_block_id(item) for item in output.embedded_headings]
    line_split_embedded_headings: list[dict[str, Any]] = []
    seen_line_split_hints: set[tuple[str, str]] = {
        (str(item.get("block_id", "")).strip(), clean_text_line(str(item.get("heading_text", ""))))
        for item in remapped_embedded_headings
        if isinstance(item, dict)
    }
    for block_id, role in output.role_labels.items():
        if role != "Heading":
            continue
        block = normalized_by_id.get(block_id)
        if block is None:
            continue
        if block.source_kind == "front_matter_split":
            source_ids = expand_block_id(block_id)
            if len(source_ids) != 1:
                continue
            source_block_id = source_ids[0]
            heading_text = clean_text_line(block.text)
            if not heading_text or not looks_like_document_title_candidate(heading_text):
                continue
            dedupe_key = (source_block_id, heading_text)
            if dedupe_key in seen_line_split_hints:
                continue
            line_split_embedded_headings.append(
                {
                    "block_id": source_block_id,
                    "heading_text": heading_text,
                    "confidence": 1.0,
                }
            )
            seen_line_split_hints.add(dedupe_key)
            continue
        is_line_split_block = (
            block.source_kind == "line_split"
            or (len(block.source_block_ids) == 1 and block.block_id.startswith(f"{block.source_block_ids[0]}__nl"))
        )
        if not is_line_split_block:
            continue
        source_ids = expand_block_id(block_id)
        if len(source_ids) != 1:
            continue
        source_block_id = source_ids[0]
        heading_text = clean_text_line(block.text)
        if not heading_text:
            continue
        dedupe_key = (source_block_id, heading_text)
        if dedupe_key in seen_line_split_hints:
            continue
        line_split_embedded_headings.append(
            {
                "block_id": source_block_id,
                "heading_text": heading_text,
                "confidence": 1.0,
            }
        )
        seen_line_split_hints.add(dedupe_key)
    remapped_embedded_headings.extend(line_split_embedded_headings)
    remapped_mixed_reviews = [remap_review_block_id(item) for item in output.mixed_group_reviews]
    remapped_reviewed_block_ids = [
        expand_block_id(block_id)[0]
        for block_id in output.embedded_heading_reviewed_block_ids
        if expand_block_id(block_id)
    ]
    remapped_mixed_reviewed_block_ids = [
        expand_block_id(block_id)[0]
        for block_id in output.mixed_group_reviewed_block_ids
        if expand_block_id(block_id)
    ]

    return VisionOutput(
        page=output.page,
        reading_order=list(dict.fromkeys(expanded_reading_order)),
        merge_groups=expanded_merge_groups,
        role_labels=expanded_role_labels,
        confidence=output.confidence,
        fallback_used=output.fallback_used,
        source=output.source,
        embedded_headings=remapped_embedded_headings,
        embedded_heading_reviewed_block_ids=list(dict.fromkeys(remapped_reviewed_block_ids)),
        mixed_group_reviews=remapped_mixed_reviews,
        mixed_group_reviewed_block_ids=list(dict.fromkeys(remapped_mixed_reviewed_block_ids)),
    )


def build_input_pkg(
    page: int,
    blocks: list[BlockCandidate],
    pages_dir: Path,
    text_limit: int = 220,
    max_blocks: Optional[int] = None,
    layout_mode: str = "direct",
) -> dict[str, Any]:
    selected_blocks = blocks
    if max_blocks is not None and max_blocks > 0:
        selected_blocks = sorted(blocks, key=lambda b: (b.column_guess, b.bbox_pt[1], b.bbox_pt[0]))[:max_blocks]
    return {
        "page": page,
        "image_path": str(pages_dir / f"p{page:03d}.png"),
        "blocks": [
            {
                "block_id": b.block_id,
                "text": b.text[:text_limit],
                "bbox_pt": b.bbox_pt,
                "bbox_px": b.bbox_px,
                "column_guess": b.column_guess,
            }
            for b in selected_blocks
        ],
        "constraints": {
            "total_blocks": len(blocks),
            "selected_blocks": len(selected_blocks),
            "text_limit": text_limit,
            "layout_mode": layout_mode,
        },
    }


def encode_image_data_url(image_path: Path, max_side: int = 1400) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def encode_pil_image_data_url(image: Image.Image, max_side: int = 1400) -> str:
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def crop_region_data_url(
    image_path: Path,
    bbox_px: list[int],
    margin_px: int,
    max_side: int,
) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
        x0 = max(0, int(bbox_px[0]) - margin_px)
        y0 = max(0, int(bbox_px[1]) - margin_px)
        x1 = min(width, int(bbox_px[2]) + margin_px)
        y1 = min(height, int(bbox_px[3]) + margin_px)
        if x1 <= x0 or y1 <= y0:
            return encode_pil_image_data_url(img, max_side=max_side)
        cropped = img.crop((x0, y0, x1, y1))
    return encode_pil_image_data_url(cropped, max_side=max_side)


def parse_model_json(raw: str) -> Optional[dict[str, Any]]:
    data = safe_json_value(raw)
    if not isinstance(data, dict):
        return None
    req = ["page", "reading_order", "merge_groups", "role_labels", "confidence"]
    if not all(k in data for k in req):
        return None
    if not isinstance(data["reading_order"], list):
        return None
    if not all(isinstance(x, str) for x in data["reading_order"]):
        return None
    if not isinstance(data["merge_groups"], list):
        return None
    for mg in data["merge_groups"]:
        if not isinstance(mg, dict) or "group_id" not in mg or "block_ids" not in mg:
            return None
    if not isinstance(data["role_labels"], dict):
        return None
    if not isinstance(data["confidence"], (int, float)):
        return None
    return data


def parse_coarse_layout_json(raw: str) -> Optional[dict[str, Any]]:
    data = safe_json_value(raw)
    if not isinstance(data, dict):
        return None
    required = {
        "page",
        "text_regions",
        "caption_regions",
        "figure_regions",
        "table_regions",
        "header_footer_regions",
        "confidence",
    }
    if not required.issubset(data.keys()):
        return None
    return data


def validate_coarse_layout_output(parsed: dict[str, Any], page: int) -> tuple[bool, str]:
    if int(parsed.get("page", -1)) != page:
        return False, "page_mismatch"

    for key in (
        "text_regions",
        "caption_regions",
        "figure_regions",
        "table_regions",
        "header_footer_regions",
    ):
        regions = parsed.get(key)
        if not isinstance(regions, list):
            return False, f"missing_region_array:{key}"
        for item in regions:
            if not isinstance(item, dict):
                return False, f"invalid_region_item:{key}"
            region_id = str(item.get("region_id", "")).strip()
            bbox = item.get("bbox_px")
            if not region_id:
                return False, f"missing_region_id:{key}"
            if not isinstance(bbox, list) or len(bbox) < 4:
                return False, f"invalid_region_bbox:{key}"

    if not isinstance(parsed.get("confidence"), (int, float)):
        return False, "invalid_confidence"
    return True, "ok"


def parse_region_heading_review_json(raw: str) -> Optional[dict[str, Any]]:
    data = safe_json_value(raw)
    if not isinstance(data, dict):
        return None
    if "page" not in data or "block_id" not in data or "accepted_headings" not in data:
        return None
    if not isinstance(data["accepted_headings"], list):
        return None
    if "reviewed" in data and not isinstance(data["reviewed"], bool):
        return None
    return data


def normalize_heading_match_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    ligature_map = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
    }
    for source, target in ligature_map.items():
        normalized = normalized.replace(source, target)
    return clean_text_line(normalized).lower()


def validate_region_heading_review_output(
    parsed: dict[str, Any],
    page: int,
    block: BlockCandidate,
    candidate_headings: list[str],
) -> tuple[bool, str]:
    if int(parsed.get("page", -1)) != page:
        return False, "page_mismatch"
    if str(parsed.get("block_id", "")) != block.block_id:
        return False, "block_mismatch"
    allowed = {normalize_heading_match_text(item) for item in candidate_headings}
    for item in parsed.get("accepted_headings", []):
        if not isinstance(item, dict):
            return False, "accepted_heading_not_object"
        heading_text = clean_text_line(str(item.get("heading_text", "")))
        if not heading_text:
            return False, "missing_heading_text"
        if normalize_heading_match_text(heading_text) not in allowed:
            return False, "accepted_heading_not_in_candidates"
        confidence = item.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            return False, "invalid_heading_confidence"
    return True, "ok"


def extract_embedded_heading_candidates(text: str, max_candidates: int = 4) -> list[str]:
    remaining = clean_text_line(text)
    candidates: list[str] = []
    seen: set[str] = set()
    steps = 0
    while remaining and steps < max(2, max_candidates * 2) and len(candidates) < max_candidates:
        steps += 1
        _, heading, suffix = split_embedded_section_heading(remaining)
        if not heading:
            break
        clean_heading = clean_text_line(heading)
        if clean_heading:
            heading_words = [word for word in clean_heading.split() if word]
            variants: list[str] = []
            if 3 <= len(heading_words) <= 4:
                first_word = re.sub(r"^[^A-Za-z]+|[^A-Za-z-]+$", "", heading_words[0]).lower()
                if first_word and first_word not in REGION_HEADING_LEAD_STOPWORDS:
                    variants.append(" ".join(heading_words[:2]))
            variants.append(clean_heading)
            for variant in variants:
                norm = variant.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                candidates.append(variant)
                if len(candidates) >= max_candidates:
                    break
        next_remaining = clean_text_line(suffix)
        if not next_remaining or next_remaining == remaining:
            break
        remaining = next_remaining
    return candidates


def looks_like_mixed_caption_tail_candidate(text: str) -> bool:
    clean = clean_text_line(text)
    if not clean or not clean[:1].islower():
        return False
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?", clean)
    if len(words) < 8:
        return False
    low = clean.lower()
    caption_tail_markers = (
        "each point represents",
        "points of different colors",
        "scale bar",
        "source data",
        "representative",
        "upregulated genes",
        "downregulated genes",
        "fold change",
    )
    return any(marker in low for marker in caption_tail_markers)


def parse_region_mixed_review_json(raw: Any) -> Optional[dict[str, Any]]:
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(data, dict):
        return None
    if "page" not in data or "block_id" not in data or "decision" not in data:
        return None
    if "reviewed" in data and not isinstance(data["reviewed"], bool):
        return None
    return data


def validate_region_mixed_review_output(
    parsed: dict[str, Any],
    page: int,
    block: BlockCandidate,
) -> tuple[bool, str]:
    if int(parsed.get("page", -1)) != page:
        return False, "page_mismatch"
    if str(parsed.get("block_id", "")).strip() != block.block_id:
        return False, "block_mismatch"
    decision = str(parsed.get("decision", "")).strip()
    if decision not in {"body", "heading", "caption_tail", "graphic_text"}:
        return False, "invalid_decision"
    caption_kind = str(parsed.get("caption_kind", "") or "").strip()
    if caption_kind and caption_kind not in {"figure", "table"}:
        return False, "invalid_caption_kind"
    confidence = parsed.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        return False, "invalid_confidence"
    return True, "ok"


def build_region_heading_review_prompt(
    page: int,
    block: BlockCandidate,
    candidate_headings: list[str],
) -> str:
    payload = {
        "page": page,
        "block_id": block.block_id,
        "bbox_px": block.bbox_px,
        "block_text": clean_text_line(block.text),
        "candidate_headings": candidate_headings,
    }
    return f"""Perform section-heading recovery for one ambiguous scientific PDF block and return STRICT JSON ONLY.

Inspect the crop and confirm which candidate strings are true visible section headings at the start of this block.
Reject body sentences, caption tails, axis labels, legend fragments, and follow-up clauses.

Input:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Output JSON schema:
{{"page": int, "block_id": str, "accepted_headings": [{{"heading_text": str, "confidence": float}}], "reviewed": bool}}

Rules:
- `heading_text` must be chosen from `candidate_headings` exactly.
- Preserve document order.
- If none are true headings, return `"accepted_headings": []` and `"reviewed": true`.

Return JSON only:"""


def build_region_mixed_review_prompt(
    page: int,
    block: BlockCandidate,
    nearby_caption_block: BlockCandidate,
    caption_kind: str,
) -> str:
    payload = {
        "page": page,
        "block_id": block.block_id,
        "bbox_px": block.bbox_px,
        "block_text": clean_text_line(block.text),
        "nearby_caption_block_id": nearby_caption_block.block_id,
        "nearby_caption_kind": caption_kind,
        "nearby_caption_text": clean_text_line(nearby_caption_block.text),
    }
    return f"""Perform local layout review for one ambiguous scientific PDF block and return STRICT JSON ONLY.

Decide whether the ambiguous block is normal body text, a true heading, a caption tail that belongs to the nearby caption, or graphic/legend text.

Input:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Output JSON schema:
{{"page": int, "block_id": str, "decision": "body|heading|caption_tail|graphic_text", "caption_kind": "figure|table", "confidence": float, "reviewed": true}}

Rules:
- Use `caption_tail` only when the ambiguous block clearly continues the nearby caption.
- Use `graphic_text` for legend/axis/panel text that should not enter main body.
- Preserve the provided `block_id`.

Return JSON only:"""


def collect_region_heading_review_candidates(
    blocks: list[BlockCandidate],
    output: VisionOutput,
    max_blocks: int,
    max_candidates: int,
) -> list[tuple[BlockCandidate, list[str]]]:
    if max_blocks <= 0:
        return []

    block_by_id = {block.block_id: block for block in blocks}
    ordered_block_ids = output.reading_order or [block.block_id for block in blocks]
    scored_candidates: list[tuple[int, int, int, BlockCandidate, list[str]]] = []
    for order_index, block_id in enumerate(ordered_block_ids):
        block = block_by_id.get(block_id)
        if block is None:
            continue
        role_label = output.role_labels.get(block_id, "Body")
        if role_label not in {"Body", "Heading"}:
            continue
        if block.is_header_footer_candidate:
            continue
        candidate_headings = extract_embedded_heading_candidates(block.text, max_candidates=max_candidates)
        if not candidate_headings:
            continue
        block_text = clean_text_line(block.text)
        if role_label == "Heading":
            word_count = len(block_text.split())
            if len(block_text) < 80 or word_count < 8:
                continue
        first_heading = candidate_headings[0]
        first_offset = max(0, block_text.lower().find(first_heading.lower()))
        role_priority = 0 if role_label == "Heading" else 1
        scored_candidates.append((role_priority, first_offset, -len(candidate_headings), order_index, block, candidate_headings))

    scored_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [(block, candidate_headings) for _, _, _, _, block, candidate_headings in scored_candidates[:max_blocks]]


def collect_region_mixed_review_candidates(
    blocks: list[BlockCandidate],
    output: VisionOutput,
    max_blocks: int,
) -> list[tuple[BlockCandidate, BlockCandidate, str]]:
    if max_blocks <= 0:
        return []

    block_by_id = {block.block_id: block for block in blocks}
    caption_blocks: list[tuple[BlockCandidate, str]] = []
    for block_id, role in output.role_labels.items():
        if role not in {"FigureCaption", "TableCaption"}:
            continue
        block = block_by_id.get(block_id)
        if block is None:
            continue
        caption_blocks.append((block, "figure" if role == "FigureCaption" else "table"))

    if not caption_blocks:
        return []

    scored_candidates: list[tuple[float, int, BlockCandidate, BlockCandidate, str]] = []
    ordered_block_ids = output.reading_order or [block.block_id for block in blocks]
    for order_index, block_id in enumerate(ordered_block_ids):
        if output.role_labels.get(block_id, "Body") != "Body":
            continue
        block = block_by_id.get(block_id)
        if block is None or block.is_header_footer_candidate:
            continue
        if not looks_like_mixed_caption_tail_candidate(block.text):
            continue

        best_match: Optional[tuple[BlockCandidate, str]] = None
        best_score: Optional[float] = None
        for caption_block, caption_kind in caption_blocks:
            vertical_gap = max(0.0, float(block.bbox_px[1]) - float(caption_block.bbox_px[3]))
            if vertical_gap > 260.0:
                continue
            overlap = max(
                0.0,
                min(float(block.bbox_px[2]), float(caption_block.bbox_px[2]))
                - max(float(block.bbox_px[0]), float(caption_block.bbox_px[0])),
            )
            min_width = max(
                1.0,
                min(
                    float(block.bbox_px[2]) - float(block.bbox_px[0]),
                    float(caption_block.bbox_px[2]) - float(caption_block.bbox_px[0]),
                ),
            )
            overlap_ratio = overlap / min_width
            right_column_continuation = float(block.bbox_px[0]) >= float(caption_block.bbox_px[2]) - 24.0
            if overlap_ratio < 0.05 and not right_column_continuation:
                continue
            score = vertical_gap - overlap_ratio * 80.0
            if best_score is None or score < best_score:
                best_score = score
                best_match = (caption_block, caption_kind)

        if best_match is None or best_score is None:
            continue
        scored_candidates.append((best_score, order_index, block, best_match[0], best_match[1]))

    scored_candidates.sort(key=lambda item: (item[0], item[1]))
    return [(block, caption_block, caption_kind) for _, _, block, caption_block, caption_kind in scored_candidates[:max_blocks]]


def review_embedded_heading_blocks(
    page: int,
    blocks: list[BlockCandidate],
    output: VisionOutput,
    page_image_path: Path,
    qa_dir: Path,
    runtime: VisionRuntimeContext,
) -> tuple[list[dict[str, Any]], list[str]]:
    enabled = os.environ.get("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return [], []

    max_blocks = max(0, int(os.environ.get("SILICONFLOW_VISION_HEADING_REVIEW_MAX_BLOCKS", "4")))
    if max_blocks == 0:
        return [], []

    crop_margin_px = max(0, int(os.environ.get("SILICONFLOW_VISION_HEADING_REVIEW_MARGIN_PX", "48")))
    crop_max_side = max(500, int(os.environ.get("SILICONFLOW_VISION_HEADING_REVIEW_IMAGE_MAX_SIDE", "900")))
    timeout_seconds = max(5, int(os.environ.get("SILICONFLOW_VISION_HEADING_REVIEW_TIMEOUT", "20")))
    max_candidates = max(1, int(os.environ.get("SILICONFLOW_VISION_HEADING_REVIEW_MAX_CANDIDATES", "4")))
    model_name = (
        os.environ.get("SILICONFLOW_VISION_HEADING_REVIEW_MODEL", "").strip()
        or os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL).strip()
    )
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT)

    embedded_headings: list[dict[str, Any]] = []
    reviewed_block_ids: list[str] = []

    review_candidates = collect_region_heading_review_candidates(
        blocks,
        output,
        max_blocks=max_blocks,
        max_candidates=max_candidates,
    )
    for block, candidate_headings in review_candidates:
        prompt = build_region_heading_review_prompt(page, block, candidate_headings)
        budget_allowed, budget_before, budget_after, budget_remaining_after = runtime.budget.try_consume()
        if not budget_allowed:
            budget_limit, _, _ = runtime.budget.snapshot()
            append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
                stage="vision_region_heading",
                page=page,
                attempt=1,
                model=model_name,
                endpoint=endpoint,
                success=False,
                parse_success=False,
                validation_success=False,
                error_type="budget_exhausted",
                http_status=None,
                prompt_chars=len(prompt),
                response_chars=0,
                response_preview="vision region-heading review budget exhausted",
                timestamp=datetime.now(timezone.utc).isoformat(),
                budget_limit=budget_limit,
                budget_consumed_before=budget_before,
                budget_consumed_after=budget_after,
                budget_remaining_after=budget_remaining_after,
                budget_exhausted=True,
                cache_hit=None,
                cache_miss=None,
                cache_key=None,
            ))
            continue

        try:
            image_data_url = crop_region_data_url(
                page_image_path,
                block.bbox_px,
                margin_px=crop_margin_px,
                max_side=crop_max_side,
            )
        except (OSError, ValueError):
            budget_limit, _, _ = runtime.budget.snapshot()
            append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
                stage="vision_region_heading",
                page=page,
                attempt=1,
                model=model_name,
                endpoint=endpoint,
                success=False,
                parse_success=False,
                validation_success=False,
                error_type="image_read_error",
                http_status=None,
                prompt_chars=len(prompt),
                response_chars=0,
                response_preview="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                budget_limit=budget_limit,
                budget_consumed_before=budget_before,
                budget_consumed_after=budget_after,
                budget_remaining_after=budget_remaining_after,
                budget_exhausted=False,
                cache_hit=None,
                cache_miss=None,
                cache_key=None,
            ))
            continue

        raw, meta = call_siliconflow(
            prompt,
            image_data_url=image_data_url,
            image_max_side=crop_max_side,
            model_override=model_name,
            timeout_seconds=timeout_seconds,
        )
        meta["budget_limit"] = runtime.budget.snapshot()[0]
        meta["budget_consumed_before"] = budget_before
        meta["budget_consumed_after"] = budget_after
        meta["budget_remaining_after"] = budget_remaining_after
        meta["budget_exhausted"] = False
        meta["cache_hit"] = None
        meta["cache_miss"] = None
        meta["cache_key"] = None

        guard = guard_model_output(
            raw,
            parse_region_heading_review_json,
            validator=lambda payload: validate_region_heading_review_output(payload, page, block, candidate_headings),
        )
        event_error_type = str(meta.get("error_type", "unknown"))
        if guard.should_fallback and event_error_type in {"none", "empty_content"}:
            event_error_type = guard.failure_reason
        append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
            stage="vision_region_heading",
            page=page,
            attempt=1,
            model=str(meta.get("model", model_name)),
            endpoint=str(meta.get("endpoint", endpoint)),
            success=bool(meta.get("success", False)),
            parse_success=guard.parse_success,
            validation_success=guard.validation_success,
            error_type=event_error_type,
            http_status=meta.get("http_status", None),
            prompt_chars=int(meta.get("prompt_chars", len(prompt))),
            response_chars=int(meta.get("response_chars", len(raw))),
            response_preview=str(meta.get("response_preview", "")),
            timestamp=datetime.now(timezone.utc).isoformat(),
            budget_limit=meta.get("budget_limit"),
            budget_consumed_before=meta.get("budget_consumed_before"),
            budget_consumed_after=meta.get("budget_consumed_after"),
            budget_remaining_after=meta.get("budget_remaining_after"),
            budget_exhausted=bool(meta.get("budget_exhausted", False)),
            cache_hit=None,
            cache_miss=None,
            cache_key=None,
        ))
        if guard.parsed is None or not guard.validation_success:
            continue

        if bool(guard.parsed.get("reviewed", True)):
            reviewed_block_ids.append(block.block_id)

        for item in guard.parsed.get("accepted_headings", []):
            heading_text = clean_text_line(str(item.get("heading_text", "")))
            if not heading_text:
                continue
            embedded_headings.append(
                {
                    "block_id": block.block_id,
                    "heading_text": heading_text,
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                }
            )

    reviewed_block_ids = list(dict.fromkeys(reviewed_block_ids))
    reviewed_block_ids.sort()
    return embedded_headings, reviewed_block_ids


def review_mixed_group_blocks(
    page: int,
    blocks: list[BlockCandidate],
    output: VisionOutput,
    page_image_path: Path,
    qa_dir: Path,
    runtime: VisionRuntimeContext,
) -> tuple[list[dict[str, Any]], list[str]]:
    enabled = os.environ.get("SILICONFLOW_VISION_MIXED_REVIEW_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return [], []

    max_blocks = max(0, int(os.environ.get("SILICONFLOW_VISION_MIXED_REVIEW_MAX_BLOCKS", "2")))
    if max_blocks == 0:
        return [], []

    crop_margin_px = max(0, int(os.environ.get("SILICONFLOW_VISION_MIXED_REVIEW_MARGIN_PX", "48")))
    crop_max_side = max(500, int(os.environ.get("SILICONFLOW_VISION_MIXED_REVIEW_IMAGE_MAX_SIDE", "900")))
    timeout_seconds = max(5, int(os.environ.get("SILICONFLOW_VISION_MIXED_REVIEW_TIMEOUT", "20")))
    model_name = (
        os.environ.get("SILICONFLOW_VISION_MIXED_REVIEW_MODEL", "").strip()
        or os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL).strip()
    )
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT)

    reviews: list[dict[str, Any]] = []
    reviewed_block_ids: list[str] = []

    for block, caption_block, caption_kind in collect_region_mixed_review_candidates(blocks, output, max_blocks):
        prompt = build_region_mixed_review_prompt(page, block, caption_block, caption_kind)
        budget_allowed, budget_before, budget_after, budget_remaining_after = runtime.budget.try_consume()
        if not budget_allowed:
            continue

        crop_bbox_px = [
            min(block.bbox_px[0], caption_block.bbox_px[0]),
            min(block.bbox_px[1], caption_block.bbox_px[1]),
            max(block.bbox_px[2], caption_block.bbox_px[2]),
            max(block.bbox_px[3], caption_block.bbox_px[3]),
        ]
        try:
            image_data_url = crop_region_data_url(
                page_image_path,
                crop_bbox_px,
                margin_px=crop_margin_px,
                max_side=crop_max_side,
            )
        except (OSError, ValueError):
            continue

        raw, meta = call_siliconflow(
            prompt,
            image_data_url=image_data_url,
            image_max_side=crop_max_side,
            model_override=model_name,
            timeout_seconds=timeout_seconds,
        )
        meta["budget_limit"] = runtime.budget.snapshot()[0]
        meta["budget_consumed_before"] = budget_before
        meta["budget_consumed_after"] = budget_after
        meta["budget_remaining_after"] = budget_remaining_after
        meta["budget_exhausted"] = False

        guard = guard_model_output(
            raw,
            parse_region_mixed_review_json,
            validator=lambda payload: validate_region_mixed_review_output(payload, page, block),
        )
        event_error_type = str(meta.get("error_type", "unknown"))
        if guard.should_fallback and event_error_type in {"none", "empty_content"}:
            event_error_type = guard.failure_reason
        append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
            stage="vision_region_mixed",
            page=page,
            attempt=1,
            model=str(meta.get("model", model_name)),
            endpoint=str(meta.get("endpoint", endpoint)),
            success=bool(meta.get("success", False)),
            parse_success=guard.parse_success,
            validation_success=guard.validation_success,
            error_type=event_error_type,
            http_status=meta.get("http_status", None),
            prompt_chars=int(meta.get("prompt_chars", len(prompt))),
            response_chars=int(meta.get("response_chars", len(raw))),
            response_preview=str(meta.get("response_preview", "")),
            timestamp=datetime.now(timezone.utc).isoformat(),
            budget_limit=meta.get("budget_limit"),
            budget_consumed_before=meta.get("budget_consumed_before"),
            budget_consumed_after=meta.get("budget_consumed_after"),
            budget_remaining_after=meta.get("budget_remaining_after"),
            budget_exhausted=bool(meta.get("budget_exhausted", False)),
        ))
        if guard.parsed is None or not guard.validation_success:
            continue

        if bool(guard.parsed.get("reviewed", True)):
            reviewed_block_ids.append(block.block_id)
        reviews.append(
            {
                "block_id": block.block_id,
                "decision": str(guard.parsed.get("decision", "")),
                "caption_kind": str(guard.parsed.get("caption_kind", caption_kind) or caption_kind),
                "confidence": float(guard.parsed.get("confidence", 0.0) or 0.0),
            }
        )

    reviewed_block_ids = list(dict.fromkeys(reviewed_block_ids))
    reviewed_block_ids.sort()
    return reviews, reviewed_block_ids


def detect_role(block: BlockCandidate) -> str:
    txt = block.text.strip()
    if CAPTION_RE.match(txt):
        return "FigureCaption" if txt.lower().startswith(("fig", "figure")) else "TableCaption"
    if block.is_heading_candidate:
        return "Heading"
    if block.is_header_footer_candidate:
        return "HeaderFooter"
    if re.match(r"^\d+\.\s+", txt) or re.match(r"^\[[\d,-]+\]", txt):
        if len(txt) > 50:
            return "ReferenceList"
    return "Body"


def fallback_reading_order(blocks: list[BlockCandidate]) -> list[str]:
    return [b.block_id for b in sorted(blocks, key=lambda x: (x.column_guess, x.bbox_pt[1], x.bbox_pt[0]))]


def fallback_merge_groups(blocks: list[BlockCandidate]) -> list[dict[str, Any]]:
    sorted_blocks = sorted(blocks, key=lambda x: (x.column_guess, x.bbox_pt[1]))
    groups: list[dict[str, Any]] = []
    cur: list[str] = []
    prev: Optional[BlockCandidate] = None
    gid = 0
    GAP = 20.0
    for b in sorted_blocks:
        role = detect_role(b)
        merge = False
        if prev and detect_role(prev) == role == "Body" and b.column_guess == prev.column_guess:
            gap = b.bbox_pt[1] - prev.bbox_pt[3]
            if 0 <= gap <= GAP:
                merge = True
        if merge:
            cur.append(b.block_id)
        else:
            if len(cur) > 1:
                groups.append({"group_id": f"mg_{gid}", "block_ids": cur})
                gid += 1
            cur = [b.block_id]
        prev = b
    if len(cur) > 1:
        groups.append({"group_id": f"mg_{gid}", "block_ids": cur})
    return groups


def fallback_role_labels(blocks: list[BlockCandidate]) -> dict[str, str]:
    return {b.block_id: detect_role(b) for b in blocks}


def generate_fallback(page: int, blocks: list[BlockCandidate]) -> VisionOutput:
    return VisionOutput(
        page=page,
        reading_order=fallback_reading_order(blocks),
        merge_groups=fallback_merge_groups(blocks),
        role_labels=fallback_role_labels(blocks),
        confidence=0.5,
        fallback_used=True,
        source="fallback",
    )


def call_siliconflow(
    prompt: str,
    image_path: Optional[Path] = None,
    image_data_url: Optional[str] = None,
    image_max_side: int = 1400,
    model_override: Optional[str] = None,
    timeout_seconds: int = 60,
) -> tuple[Any, dict[str, Any]]:
    api_key = resolve_api_key()
    model = (model_override or os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL)).strip()
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

    message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_data_url is not None:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url},
        })
    elif image_path is not None:
        try:
            image_data_url = encode_image_data_url(image_path, max_side=image_max_side)
            message_content.append({
                "type": "image_url",
                "image_url": {"url": image_data_url},
            })
        except (OSError, ValueError):
            meta["error_type"] = "image_read_error"
            return "{}", meta

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": message_content}],
        "temperature": 0.0,
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
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", None)
            result = json.loads(resp.read().decode("utf-8"))
            raw_content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            response_text: str
            if isinstance(raw_content, str):
                response_text = raw_content
            elif isinstance(raw_content, list):
                text_parts = [
                    str(part.get("text", ""))
                    for part in raw_content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                response_text = "\n".join(p for p in text_parts if p).strip() or "{}"
            else:
                response_text = "{}"
            meta["http_status"] = status
            meta["response_chars"] = len(response_text)
            meta["response_preview"] = response_text[:300]
            meta["success"] = bool(response_text and response_text != "{}")
            meta["error_type"] = "none" if meta["success"] else "empty_content"
            return response_text, meta
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
    except Exception as e:  # defensive: keep telemetry informative
        meta["error_type"] = f"exception:{type(e).__name__}"
        meta["response_preview"] = str(e)[:300]
        return "{}", meta


def build_prompt(input_pkg: dict[str, Any]) -> str:
    return f"""Analyze this document page and return STRICT JSON ONLY (no markdown, no prose).

Input:
{json.dumps(input_pkg, indent=2, ensure_ascii=False)}

Output JSON schema:
{{"page": int, "reading_order": ["block_id", ...], "merge_groups": [{{"group_id": str, "block_ids": [...]}}], "role_labels": {{"block_id": "role"}}, "confidence": float}}

Role enum: Body, Heading, FigureCaption, TableCaption, Footnote, ReferenceList, Sidebar, HeaderFooter

Return JSON only:"""


def build_coarse_layout_prompt(input_pkg: dict[str, Any]) -> str:
    return f"""Perform coarse page-region segmentation for this scientific PDF page and return STRICT JSON ONLY.

Goal:
- Identify the main text regions, caption regions, figure regions, table regions, and header/footer regions.
- This is coarse page-region segmentation, not final paragraph ordering.

Input:
{json.dumps(input_pkg, indent=2, ensure_ascii=False)}

Output JSON schema:
{{"page": int, "text_regions": [{{"region_id": str, "bbox_px": [x0, y0, x1, y1]}}], "caption_regions": [{{"region_id": str, "bbox_px": [x0, y0, x1, y1]}}], "figure_regions": [{{"region_id": str, "bbox_px": [x0, y0, x1, y1]}}], "table_regions": [{{"region_id": str, "bbox_px": [x0, y0, x1, y1]}}], "header_footer_regions": [{{"region_id": str, "bbox_px": [x0, y0, x1, y1]}}], "confidence": float}}

Rules:
- Return only large coarse regions, not per-word or per-block labels.
- Use pixel coordinates.
- Return JSON only."""


def _coarse_seed_score(block: BlockCandidate) -> tuple[int, int, int]:
    cleaned = clean_text_line(block.text)
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?", cleaned)
    alpha_tokens = sum(1 for word in words if re.search(r"[A-Za-z]", word))
    numeric_tokens = sum(1 for word in words if re.fullmatch(r"\d+(?:\.\d+)?", word) is not None)
    score = min(len(cleaned), 160)
    if re.match(r"^(fig(?:ure)?\.?|table)\b", cleaned, flags=re.IGNORECASE):
        score += 120
    if alpha_tokens >= 4:
        score += 60
    if block.is_heading_candidate:
        score += 40
    if block.is_header_footer_candidate:
        score += 20
    if numeric_tokens >= max(2, len(words) - 1):
        score -= 80
    return (score, alpha_tokens, len(cleaned))


def select_coarse_layout_seed_blocks(
    blocks: list[BlockCandidate],
    max_blocks: int,
) -> list[BlockCandidate]:
    if max_blocks <= 0 or len(blocks) <= max_blocks:
        return list(blocks)

    ranked = sorted(
        blocks,
        key=lambda block: (
            _coarse_seed_score(block),
            -block.bbox_px[1],
            -block.bbox_px[0],
        ),
        reverse=True,
    )

    selected: list[BlockCandidate] = []
    occupied_cells: set[tuple[int, int]] = set()
    grid_size = 800
    for block in ranked:
        cx, cy = _block_center(block)
        cell = (int(cx // grid_size), int(cy // grid_size))
        if cell in occupied_cells and len(selected) < max_blocks // 2:
            continue
        occupied_cells.add(cell)
        selected.append(block)
        if len(selected) >= max_blocks:
            break

    if len(selected) < max_blocks:
        selected_ids = {block.block_id for block in selected}
        for block in ranked:
            if block.block_id in selected_ids:
                continue
            selected.append(block)
            selected_ids.add(block.block_id)
            if len(selected) >= max_blocks:
                break

    return selected


def maybe_run_coarse_layout_pass(
    page: int,
    blocks: list[BlockCandidate],
    pages_dir: Path,
    vision_dir: Path,
    qa_dir: Path,
    runtime: VisionRuntimeContext,
    layout_mode: str,
) -> Optional[dict[str, Any]]:
    if layout_mode != "hierarchical":
        return None

    coarse_text_limit = int(os.environ.get("SILICONFLOW_VISION_COARSE_TEXT_LIMIT", "80"))
    coarse_max_blocks = int(os.environ.get("SILICONFLOW_VISION_COARSE_MAX_BLOCKS", "24"))
    coarse_image_max_side = int(os.environ.get("SILICONFLOW_VISION_COARSE_IMAGE_MAX_SIDE", "1100"))
    coarse_timeout = int(os.environ.get("SILICONFLOW_VISION_COARSE_TIMEOUT", "20"))
    model_name = (
        os.environ.get("SILICONFLOW_VISION_COARSE_MODEL", "").strip()
        or os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL).strip()
    )
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT)
    page_image_path = pages_dir / f"p{page:03d}.png"
    seed_blocks = select_coarse_layout_seed_blocks(blocks, max(8, coarse_max_blocks))

    coarse_input = build_input_pkg(
        page,
        seed_blocks,
        pages_dir,
        text_limit=max(30, coarse_text_limit),
        max_blocks=max(8, coarse_max_blocks),
        layout_mode="coarse_regions",
    )
    prompt = build_coarse_layout_prompt(coarse_input)

    budget_allowed, budget_before, budget_after, budget_remaining_after = runtime.budget.try_consume()
    if not budget_allowed:
        return None

    try:
        image_data_url, _, _ = runtime.image_cache.get_or_encode(
            page_image_path,
            max_side=max(700, coarse_image_max_side),
        )
    except (OSError, ValueError):
        return None

    raw, meta = call_siliconflow(
        prompt,
        image_data_url=image_data_url,
        image_max_side=max(700, coarse_image_max_side),
        model_override=model_name,
        timeout_seconds=max(5, coarse_timeout),
    )
    meta["budget_limit"] = runtime.budget.snapshot()[0]
    meta["budget_consumed_before"] = budget_before
    meta["budget_consumed_after"] = budget_after
    meta["budget_remaining_after"] = budget_remaining_after
    meta["budget_exhausted"] = False

    guard = guard_model_output(
        raw,
        parse_coarse_layout_json,
        validator=lambda payload: validate_coarse_layout_output(payload, page),
    )
    event_error_type = str(meta.get("error_type", "unknown"))
    if guard.should_fallback and event_error_type in {"none", "empty_content"}:
        event_error_type = guard.failure_reason
    append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
        stage="vision_coarse",
        page=page,
        attempt=1,
        model=str(meta.get("model", model_name)),
        endpoint=str(meta.get("endpoint", endpoint)),
        success=bool(meta.get("success", False)),
        parse_success=guard.parse_success,
        validation_success=guard.validation_success,
        error_type=event_error_type,
        http_status=meta.get("http_status", None),
        prompt_chars=int(meta.get("prompt_chars", len(prompt))),
        response_chars=int(meta.get("response_chars", len(raw))),
        response_preview=str(meta.get("response_preview", "")),
        timestamp=datetime.now(timezone.utc).isoformat(),
        budget_limit=meta.get("budget_limit"),
        budget_consumed_before=meta.get("budget_consumed_before"),
        budget_consumed_after=meta.get("budget_consumed_after"),
        budget_remaining_after=meta.get("budget_remaining_after"),
        budget_exhausted=bool(meta.get("budget_exhausted", False)),
        cache_hit=None,
        cache_miss=None,
        cache_key=None,
    ))
    if guard.parsed is None or not guard.validation_success:
        return None

    regions_path = vision_dir / f"p{page:03d}_regions.json"
    with open(regions_path, "w", encoding="utf-8") as f:
        json.dump(guard.parsed, f, indent=2, ensure_ascii=False)
    return guard.parsed


def _point_in_bbox(x: float, y: float, bbox: list[int]) -> bool:
    if len(bbox) < 4:
        return False
    return float(bbox[0]) <= x <= float(bbox[2]) and float(bbox[1]) <= y <= float(bbox[3])


def _bbox_area(bbox: list[int]) -> float:
    if len(bbox) < 4:
        return 0.0
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return width * height


def _bbox_overlap_area(left: list[int], right: list[int]) -> float:
    if len(left) < 4 or len(right) < 4:
        return 0.0
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def project_blocks_to_layout_regions(
    blocks: list[BlockCandidate],
    coarse_layout: dict[str, Any],
) -> dict[str, list[str]]:
    projected = {
        "text_block_ids": [],
        "caption_block_ids": [],
        "figure_block_ids": [],
        "table_block_ids": [],
        "header_footer_block_ids": [],
        "unassigned_block_ids": [],
    }

    text_regions = coarse_layout.get("text_regions", [])
    caption_regions = coarse_layout.get("caption_regions", [])
    figure_regions = coarse_layout.get("figure_regions", [])
    table_regions = coarse_layout.get("table_regions", [])
    header_footer_regions = coarse_layout.get("header_footer_regions", [])

    for block in blocks:
        x0, y0, x1, y1 = block.bbox_px
        cx = (float(x0) + float(x1)) / 2.0
        cy = (float(y0) + float(y1)) / 2.0

        assigned_key: Optional[str] = None
        for key, regions in (
            ("header_footer_block_ids", header_footer_regions),
            ("text_block_ids", text_regions),
            ("caption_block_ids", caption_regions),
            ("figure_block_ids", figure_regions),
            ("table_block_ids", table_regions),
        ):
            if any(
                isinstance(region, dict) and _point_in_bbox(cx, cy, list(region.get("bbox_px", [])))
                for region in regions
            ):
                assigned_key = key
                break

        if assigned_key is None:
            best_overlap_key: Optional[str] = None
            best_overlap_ratio = 0.0
            block_area = _bbox_area(block.bbox_px)
            for key, regions in (
                ("header_footer_block_ids", header_footer_regions),
                ("text_block_ids", text_regions),
                ("caption_block_ids", caption_regions),
                ("figure_block_ids", figure_regions),
                ("table_block_ids", table_regions),
            ):
                for region in regions:
                    if not isinstance(region, dict):
                        continue
                    overlap_area = _bbox_overlap_area(block.bbox_px, list(region.get("bbox_px", [])))
                    if overlap_area <= 0.0:
                        continue
                    overlap_ratio = overlap_area / max(1.0, block_area)
                    if overlap_ratio > best_overlap_ratio:
                        best_overlap_ratio = overlap_ratio
                        best_overlap_key = key
            if best_overlap_key is not None and best_overlap_ratio >= 0.05:
                assigned_key = best_overlap_key

        if assigned_key is None:
            projected["unassigned_block_ids"].append(block.block_id)
        else:
            projected[assigned_key].append(block.block_id)

    return projected


def _block_center(block: BlockCandidate) -> tuple[float, float]:
    x0, y0, x1, y1 = block.bbox_px
    return ((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0)


def _bbox_pt_intersects_band(bbox_pt: list[float], band: Any) -> bool:
    if not isinstance(band, list) or len(band) < 4:
        return False
    try:
        x0, y0, x1, y1 = [float(value) for value in band[:4]]
    except (TypeError, ValueError):
        return False
    if x1 <= x0 or y1 <= y0:
        return False
    return not (
        bbox_pt[2] <= x0
        or bbox_pt[0] >= x1
        or bbox_pt[3] <= y0
        or bbox_pt[1] >= y1
    )


def _filter_block_ids_by_document_profile(
    blocks: list[BlockCandidate],
    block_ids: list[str],
    document_profile: Optional[dict[str, Any]],
) -> list[str]:
    if not document_profile or not block_ids:
        return block_ids
    header_band = document_profile.get("header_band_pt")
    footer_band = document_profile.get("footer_band_pt")
    block_by_id = {block.block_id: block for block in blocks}
    kept: list[str] = []
    for block_id in block_ids:
        block = block_by_id.get(block_id)
        if block is None:
            continue
        if _bbox_pt_intersects_band(block.bbox_pt, header_band):
            continue
        if _bbox_pt_intersects_band(block.bbox_pt, footer_band):
            continue
        kept.append(block_id)
    return kept


def _blocks_are_cluster_neighbors(
    left: BlockCandidate,
    right: BlockCandidate,
    max_dx_px: int = 360,
    max_dy_px: int = 220,
) -> bool:
    lx, ly = _block_center(left)
    rx, ry = _block_center(right)
    return abs(lx - rx) <= float(max_dx_px) and abs(ly - ry) <= float(max_dy_px)


def _cluster_block_ids_by_spatial_proximity(
    blocks: list[BlockCandidate],
    block_ids: list[str],
) -> list[list[BlockCandidate]]:
    if not block_ids:
        return []

    block_by_id = {block.block_id: block for block in blocks}
    candidate_blocks = [block_by_id[block_id] for block_id in block_ids if block_id in block_by_id]
    if not candidate_blocks:
        return []

    ordered_blocks = sorted(candidate_blocks, key=lambda block: (_block_center(block)[0], _block_center(block)[1]))
    clusters: list[list[BlockCandidate]] = []
    for block in ordered_blocks:
        matched_cluster: Optional[list[BlockCandidate]] = None
        for cluster in clusters:
            if any(_blocks_are_cluster_neighbors(member, block) for member in cluster):
                matched_cluster = cluster
                break
        if matched_cluster is None:
            clusters.append([block])
        else:
            matched_cluster.append(block)
    return clusters


def prune_non_narrative_text_block_ids(
    blocks: list[BlockCandidate],
    text_block_ids: list[str],
) -> list[str]:
    if len(text_block_ids) <= 1:
        return text_block_ids

    clusters = _cluster_block_ids_by_spatial_proximity(blocks, text_block_ids)
    if not clusters:
        return []

    dropped_ids: set[str] = set()
    for cluster_index, cluster in enumerate(clusters):
        cluster_texts = [
            block.text
            for block in sorted(cluster, key=lambda item: (item.bbox_px[1], item.bbox_px[0]))
            if clean_text_line(block.text)
        ]
        if not cluster_texts:
            continue
        label = classify_microblock_cluster_coherence(
            {
                "region_id": f"text_cluster_{cluster_index}",
                "texts": cluster_texts,
            }
        )
        if label == "non_narrative_graphic":
            dropped_ids.update(block.block_id for block in cluster)

    return [block_id for block_id in text_block_ids if block_id not in dropped_ids]


def select_hierarchical_fine_layout_block_ids(
    blocks: list[BlockCandidate],
    projected_regions: dict[str, list[str]],
    document_profile: Optional[dict[str, Any]] = None,
) -> list[str]:
    eligible_text_ids = _filter_block_ids_by_document_profile(
        blocks,
        projected_regions.get("text_block_ids", []),
        document_profile,
    )
    filtered_text_ids = prune_non_narrative_text_block_ids(blocks, eligible_text_ids)
    caption_block_ids = _filter_block_ids_by_document_profile(
        blocks,
        projected_regions.get("caption_block_ids", []),
        document_profile,
    )
    if filtered_text_ids:
        return filtered_text_ids + caption_block_ids
    if eligible_text_ids:
        return eligible_text_ids + caption_block_ids
    return caption_block_ids


def build_coarse_region_fallback_output(
    page: int,
    blocks: list[BlockCandidate],
    projected_regions: dict[str, list[str]],
) -> VisionOutput:
    role_labels = fallback_role_labels(blocks)
    for block_id in projected_regions.get("header_footer_block_ids", []):
        role_labels[block_id] = "HeaderFooter"
    for block_id in projected_regions.get("caption_block_ids", []):
        role_labels[block_id] = "FigureCaption"
    for block_id in projected_regions.get("figure_block_ids", []):
        role_labels[block_id] = "Sidebar"
    for block_id in projected_regions.get("table_block_ids", []):
        role_labels[block_id] = "Sidebar"

    for cluster_index, cluster in enumerate(
        _cluster_block_ids_by_spatial_proximity(blocks, projected_regions.get("unassigned_block_ids", []))
    ):
        cluster_texts = [
            block.text
            for block in sorted(cluster, key=lambda item: (item.bbox_px[1], item.bbox_px[0]))
            if clean_text_line(block.text)
        ]
        if not cluster_texts:
            continue
        label = classify_microblock_cluster_coherence(
            {
                "region_id": f"unassigned_cluster_{cluster_index}",
                "texts": cluster_texts,
            }
        )
        if label == "non_narrative_graphic":
            for block in cluster:
                role_labels[block.block_id] = "Sidebar"

    return VisionOutput(
        page=page,
        reading_order=fallback_reading_order(blocks),
        merge_groups=fallback_merge_groups(blocks),
        role_labels=role_labels,
        confidence=0.35,
        fallback_used=True,
        source="coarse_fallback",
    )


def validate_model_output(parsed: dict[str, Any], expected_page: int, blocks: list[BlockCandidate]) -> tuple[bool, str]:
    known_ids = {b.block_id for b in blocks}
    if not known_ids:
        return False, "no_input_blocks"

    if int(parsed.get("page", -1)) != expected_page:
        return False, "page_mismatch"

    reading_order = parsed.get("reading_order", [])
    if not isinstance(reading_order, list) or not reading_order:
        return False, "empty_reading_order"
    ro_valid = [bid for bid in reading_order if isinstance(bid, str) and bid in known_ids]
    if not ro_valid:
        return False, "reading_order_unknown_ids"
    coverage = len(set(ro_valid)) / max(1, len(known_ids))
    if coverage < 0.35:
        return False, "reading_order_low_coverage"

    role_labels = parsed.get("role_labels", {})
    if not isinstance(role_labels, dict) or not role_labels:
        return False, "empty_role_labels"
    role_valid_count = 0
    for bid, role in role_labels.items():
        if isinstance(bid, str) and bid in known_ids and isinstance(role, str) and role in ROLE_LABELS:
            role_valid_count += 1
    if role_valid_count == 0:
        return False, "invalid_role_labels"

    return True, "ok"


def process_page(
    page: int,
    blocks: list[BlockCandidate],
    pages_dir: Path,
    vision_dir: Path,
    qa_dir: Path,
    inject_malformed: bool,
    runtime: VisionRuntimeContext,
    primary_model_override: Optional[str] = None,
    retry_model_override: Optional[str] = None,
    document_profile: Optional[dict[str, Any]] = None,
    paragraph_regrouping_hints_by_page: Optional[dict[int, list[list[str]]]] = None,
    block_lines_by_block: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> tuple[VisionOutput, Optional[FaultEvent]]:
    raw_blocks = blocks
    normalized_blocks, hidden_role_labels = normalize_blocks_for_vision_page(
        page=page,
        blocks=raw_blocks,
        document_profile=document_profile,
        paragraph_regrouping_hints_by_page=paragraph_regrouping_hints_by_page,
        block_lines_by_block=block_lines_by_block,
    )
    vision_input_blocks = normalized_blocks or raw_blocks

    layout_mode = classify_page_layout_mode(vision_input_blocks)
    coarse_layout = maybe_run_coarse_layout_pass(
        page=page,
        blocks=vision_input_blocks,
        pages_dir=pages_dir,
        vision_dir=vision_dir,
        qa_dir=qa_dir,
        runtime=runtime,
        layout_mode=layout_mode,
    )
    layout_blocks = vision_input_blocks
    projected_regions: Optional[dict[str, list[str]]] = None
    if layout_mode == "hierarchical" and coarse_layout is not None:
        projected_regions = project_blocks_to_layout_regions(vision_input_blocks, coarse_layout)
        fine_layout_ids = select_hierarchical_fine_layout_block_ids(
            vision_input_blocks,
            projected_regions,
            document_profile=document_profile,
        )
        if fine_layout_ids:
            allowed = set(fine_layout_ids)
            layout_blocks = [block for block in vision_input_blocks if block.block_id in allowed]
        else:
            layout_blocks = []
    elif layout_mode == "hierarchical" and coarse_layout is None:
        fallback_output = generate_fallback(page, vision_input_blocks)
        fallback_output.source = "hierarchical_fallback"
        fallback_output = expand_vision_output_to_raw_blocks(
            fallback_output,
            vision_input_blocks,
            raw_blocks,
            hidden_role_labels=hidden_role_labels,
        )
        output_path = vision_dir / f"p{page:03d}_out.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "page": fallback_output.page,
                    "reading_order": fallback_output.reading_order,
                    "merge_groups": fallback_output.merge_groups,
                    "role_labels": fallback_output.role_labels,
                    "confidence": fallback_output.confidence,
                    "fallback_used": fallback_output.fallback_used,
                    "source": fallback_output.source,
                    "embedded_headings": fallback_output.embedded_headings,
                    "embedded_heading_reviewed_block_ids": fallback_output.embedded_heading_reviewed_block_ids,
                    "mixed_group_reviews": fallback_output.mixed_group_reviews,
                    "mixed_group_reviewed_block_ids": fallback_output.mixed_group_reviewed_block_ids,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        fault = FaultEvent(
            stage="vision",
            fault="coarse-layout-unavailable",
            page=page,
            retry_attempts=0,
            fallback_used=True,
            status="degraded",
        )
        return fallback_output, fault
    primary_text_limit = int(os.environ.get("SILICONFLOW_VISION_PRIMARY_TEXT_LIMIT", "180"))
    primary_max_blocks = int(os.environ.get("SILICONFLOW_VISION_PRIMARY_MAX_BLOCKS", "280"))
    if layout_mode == "hierarchical":
        primary_text_limit = int(os.environ.get("SILICONFLOW_VISION_DENSE_TEXT_LIMIT", str(primary_text_limit)))
        primary_max_blocks = int(os.environ.get("SILICONFLOW_VISION_DENSE_MAX_BLOCKS", str(primary_max_blocks)))
    min_primary_text_limit = 40 if layout_mode == "hierarchical" else 80
    min_primary_max_blocks = 20 if layout_mode == "hierarchical" else 80
    primary_image_max_side = int(os.environ.get("SILICONFLOW_VISION_PRIMARY_IMAGE_MAX_SIDE", "1300"))
    input_pkg = build_input_pkg(
        page,
        layout_blocks,
        pages_dir,
        text_limit=max(min_primary_text_limit, primary_text_limit),
        max_blocks=max(min_primary_max_blocks, primary_max_blocks),
        layout_mode=layout_mode,
    )
    selected_block_ids = {
        str(item.get("block_id", "")).strip()
        for item in input_pkg.get("blocks", [])
        if isinstance(item, dict) and str(item.get("block_id", "")).strip()
    }
    validation_blocks = [block for block in layout_blocks if block.block_id in selected_block_ids] or layout_blocks
    input_path = vision_dir / f"p{page:03d}_in.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_pkg, f, indent=2, ensure_ascii=False)

    if layout_mode == "hierarchical" and coarse_layout is not None and projected_regions is not None and not layout_blocks:
        coarse_output = build_coarse_region_fallback_output(page, vision_input_blocks, projected_regions)
        coarse_output = expand_vision_output_to_raw_blocks(
            coarse_output,
            vision_input_blocks,
            raw_blocks,
            hidden_role_labels=hidden_role_labels,
        )
        output_path = vision_dir / f"p{page:03d}_out.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "page": coarse_output.page,
                    "reading_order": coarse_output.reading_order,
                    "merge_groups": coarse_output.merge_groups,
                    "role_labels": coarse_output.role_labels,
                    "confidence": coarse_output.confidence,
                    "fallback_used": coarse_output.fallback_used,
                    "source": coarse_output.source,
                    "embedded_headings": coarse_output.embedded_headings,
                    "embedded_heading_reviewed_block_ids": coarse_output.embedded_heading_reviewed_block_ids,
                    "mixed_group_reviews": coarse_output.mixed_group_reviews,
                    "mixed_group_reviewed_block_ids": coarse_output.mixed_group_reviewed_block_ids,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return coarse_output, None

    fault: Optional[FaultEvent] = None
    retry_attempts = 0
    prompt = build_prompt(input_pkg)
    model_name = (primary_model_override or os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL)).strip()
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT)
    page_image_path = pages_dir / f"p{page:03d}.png"

    if inject_malformed:
        budget_limit, budget_consumed, budget_remaining = runtime.budget.snapshot()
        raw = "{ malformed json !!! "
        meta: dict[str, Any] = {
            "model": model_name,
            "endpoint": endpoint,
            "success": False,
            "error_type": "injected_malformed_json",
            "http_status": None,
            "prompt_chars": len(prompt),
            "response_chars": len(raw),
            "response_preview": raw[:300],
            "budget_limit": budget_limit,
            "budget_consumed_before": budget_consumed,
            "budget_consumed_after": budget_consumed,
            "budget_remaining_after": budget_remaining,
            "budget_exhausted": False,
            "cache_hit": None,
            "cache_miss": None,
            "cache_key": None,
        }
    else:
        primary_timeout = int(os.environ.get("SILICONFLOW_VISION_PRIMARY_TIMEOUT", "20"))
        budget_allowed, budget_before, budget_after, budget_remaining_after = runtime.budget.try_consume()
        if not budget_allowed:
            raw = "{}"
            budget_limit, _, _ = runtime.budget.snapshot()
            meta = {
                "model": model_name,
                "endpoint": endpoint,
                "success": False,
                "error_type": "budget_exhausted",
                "http_status": None,
                "prompt_chars": len(prompt),
                "response_chars": len(raw),
                "response_preview": "vision request budget exhausted",
                "budget_limit": budget_limit,
                "budget_consumed_before": budget_before,
                "budget_consumed_after": budget_after,
                "budget_remaining_after": budget_remaining_after,
                "budget_exhausted": True,
                "cache_hit": None,
                "cache_miss": None,
                "cache_key": None,
            }
        else:
            try:
                image_data_url, cache_hit, cache_key = runtime.image_cache.get_or_encode(
                    page_image_path,
                    max_side=max(800, primary_image_max_side),
                )
            except (OSError, ValueError):
                raw = "{}"
                budget_limit, _, _ = runtime.budget.snapshot()
                meta = {
                    "model": model_name,
                    "endpoint": endpoint,
                    "success": False,
                    "error_type": "image_read_error",
                    "http_status": None,
                    "prompt_chars": len(prompt),
                    "response_chars": len(raw),
                    "response_preview": "",
                    "budget_limit": budget_limit,
                    "budget_consumed_before": budget_before,
                    "budget_consumed_after": budget_after,
                    "budget_remaining_after": budget_remaining_after,
                    "budget_exhausted": False,
                    "cache_hit": None,
                    "cache_miss": None,
                    "cache_key": None,
                }
            else:
                raw, meta = call_siliconflow(
                    prompt,
                    image_data_url=image_data_url,
                    image_max_side=max(800, primary_image_max_side),
                    model_override=model_name,
                    timeout_seconds=max(5, primary_timeout),
                )
                meta["budget_limit"] = runtime.budget.snapshot()[0]
                meta["budget_consumed_before"] = budget_before
                meta["budget_consumed_after"] = budget_after
                meta["budget_remaining_after"] = budget_remaining_after
                meta["budget_exhausted"] = False
                meta["cache_hit"] = cache_hit
                meta["cache_miss"] = not cache_hit
                meta["cache_key"] = cache_key

    initial_guard = guard_model_output(
        raw,
        parse_model_json,
        validator=lambda payload: validate_model_output(payload, page, validation_blocks),
    )
    parsed = initial_guard.parsed
    validation_success = initial_guard.validation_success
    validation_reason = initial_guard.failure_reason

    event_error_type = str(meta.get("error_type", "unknown"))
    if initial_guard.should_fallback and event_error_type in {"none", "empty_content"}:
        event_error_type = validation_reason

    append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
        stage="vision",
        page=page,
        attempt=1,
        model=str(meta.get("model", SILICONFLOW_DEFAULT_MODEL)),
        endpoint=str(meta.get("endpoint", SILICONFLOW_ENDPOINT)),
        success=bool(meta.get("success", False)),
        parse_success=initial_guard.parse_success,
        validation_success=validation_success,
        error_type=event_error_type,
        http_status=meta.get("http_status", None),
        prompt_chars=int(meta.get("prompt_chars", len(prompt))),
        response_chars=int(meta.get("response_chars", len(raw))),
        response_preview=str(meta.get("response_preview", "")),
        timestamp=datetime.now(timezone.utc).isoformat(),
        budget_limit=meta.get("budget_limit"),
        budget_consumed_before=meta.get("budget_consumed_before"),
        budget_consumed_after=meta.get("budget_consumed_after"),
        budget_remaining_after=meta.get("budget_remaining_after"),
        budget_exhausted=bool(meta.get("budget_exhausted", False)),
        cache_hit=meta.get("cache_hit"),
        cache_miss=meta.get("cache_miss"),
        cache_key=meta.get("cache_key"),
    ))

    if initial_guard.should_fallback:
        if not inject_malformed:
            retry_attempts = 0
            retry_model = (retry_model_override or os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL)).strip()
            first_error_type = str(meta.get("error_type", "unknown"))
            first_http_status = meta.get("http_status", None)
            if first_error_type in {"none", "empty_content"}:
                first_error_type = validation_reason

            retry_blockers = {"missing_api_key", "invalid_endpoint", "auth_http_error", "budget_exhausted"}
            if first_error_type in retry_blockers:
                should_retry = False
            else:
                should_retry = (
                    (parsed is not None and not validation_success)
                    or validation_reason == "parse_failure"
                    or not is_request_contract_error(first_error_type, first_http_status)
                )

            if should_retry:
                retry_attempts = 1
                transient_retry = is_transient_retryable_error(first_error_type, first_http_status)
                if transient_retry:
                    retry_backoff_seconds = float(os.environ.get("SILICONFLOW_VISION_RETRY_BACKOFF_SECONDS", "1.5"))
                    if retry_backoff_seconds > 0:
                        time.sleep(min(10.0, retry_backoff_seconds))
                    retry_model = (
                        retry_model_override
                        or os.environ.get("SILICONFLOW_VISION_FALLBACK_MODEL", SILICONFLOW_FALLBACK_MODEL)
                    ).strip()
                retry_input_pkg = build_input_pkg(
                    page,
                    layout_blocks,
                    pages_dir,
                    text_limit=120,
                    max_blocks=220,
                    layout_mode=layout_mode,
                )
                retry_selected_block_ids = {
                    str(item.get("block_id", "")).strip()
                    for item in retry_input_pkg.get("blocks", [])
                    if isinstance(item, dict) and str(item.get("block_id", "")).strip()
                }
                retry_validation_blocks = [
                    block for block in layout_blocks if block.block_id in retry_selected_block_ids
                ] or layout_blocks
                retry_prompt = build_prompt(retry_input_pkg)
                retry_timeout = int(os.environ.get("SILICONFLOW_VISION_RETRY_TIMEOUT", "45"))
                budget_allowed, budget_before, budget_after, budget_remaining_after = runtime.budget.try_consume()
                if not budget_allowed:
                    raw = "{}"
                    budget_limit, _, _ = runtime.budget.snapshot()
                    meta = {
                        "model": retry_model,
                        "endpoint": endpoint,
                        "success": False,
                        "error_type": "budget_exhausted",
                        "http_status": None,
                        "prompt_chars": len(retry_prompt),
                        "response_chars": len(raw),
                        "response_preview": "vision retry budget exhausted",
                        "budget_limit": budget_limit,
                        "budget_consumed_before": budget_before,
                        "budget_consumed_after": budget_after,
                        "budget_remaining_after": budget_remaining_after,
                        "budget_exhausted": True,
                        "cache_hit": None,
                        "cache_miss": None,
                        "cache_key": None,
                    }
                else:
                    try:
                        image_data_url, cache_hit, cache_key = runtime.image_cache.get_or_encode(
                            page_image_path,
                            max_side=1100,
                        )
                    except (OSError, ValueError):
                        raw = "{}"
                        budget_limit, _, _ = runtime.budget.snapshot()
                        meta = {
                            "model": retry_model,
                            "endpoint": endpoint,
                            "success": False,
                            "error_type": "image_read_error",
                            "http_status": None,
                            "prompt_chars": len(retry_prompt),
                            "response_chars": len(raw),
                            "response_preview": "",
                            "budget_limit": budget_limit,
                            "budget_consumed_before": budget_before,
                            "budget_consumed_after": budget_after,
                            "budget_remaining_after": budget_remaining_after,
                            "budget_exhausted": False,
                            "cache_hit": None,
                            "cache_miss": None,
                            "cache_key": None,
                        }
                    else:
                        raw, meta = call_siliconflow(
                            retry_prompt,
                            image_data_url=image_data_url,
                            image_max_side=1100,
                            model_override=retry_model,
                            timeout_seconds=max(10, retry_timeout),
                        )
                        meta["budget_limit"] = runtime.budget.snapshot()[0]
                        meta["budget_consumed_before"] = budget_before
                        meta["budget_consumed_after"] = budget_after
                        meta["budget_remaining_after"] = budget_remaining_after
                        meta["budget_exhausted"] = False
                        meta["cache_hit"] = cache_hit
                        meta["cache_miss"] = not cache_hit
                        meta["cache_key"] = cache_key
                retry_guard = guard_model_output(
                    raw,
                    parse_model_json,
                    validator=lambda payload: validate_model_output(payload, page, retry_validation_blocks),
                )
                parsed = retry_guard.parsed
                validation_success = retry_guard.validation_success
                validation_reason = retry_guard.failure_reason

                event_error_type = str(meta.get("error_type", "unknown"))
                if retry_guard.should_fallback and event_error_type in {"none", "empty_content"}:
                    event_error_type = validation_reason

                append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
                    stage="vision",
                    page=page,
                    attempt=2,
                    model=str(meta.get("model", SILICONFLOW_DEFAULT_MODEL)),
                    endpoint=str(meta.get("endpoint", SILICONFLOW_ENDPOINT)),
                    success=bool(meta.get("success", False)),
                    parse_success=retry_guard.parse_success,
                    validation_success=validation_success,
                    error_type=event_error_type,
                    http_status=meta.get("http_status", None),
                    prompt_chars=int(meta.get("prompt_chars", len(retry_prompt))),
                    response_chars=int(meta.get("response_chars", len(raw))),
                    response_preview=str(meta.get("response_preview", "")),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    budget_limit=meta.get("budget_limit"),
                    budget_consumed_before=meta.get("budget_consumed_before"),
                    budget_consumed_after=meta.get("budget_consumed_after"),
                    budget_remaining_after=meta.get("budget_remaining_after"),
                    budget_exhausted=bool(meta.get("budget_exhausted", False)),
                    cache_hit=meta.get("cache_hit"),
                    cache_miss=meta.get("cache_miss"),
                    cache_key=meta.get("cache_key"),
                ))

    if parsed is None or not validation_success:
        output = generate_fallback(page, vision_input_blocks)
        if not inject_malformed and str(meta.get("error_type", "")) == "budget_exhausted":
            fault_label = "budget-exhausted"
        else:
            fault_label = "malformed-json" if inject_malformed else f"invalid-model-output:{validation_reason}"
        fault = FaultEvent(
            stage="vision",
            fault=fault_label,
            page=page,
            retry_attempts=retry_attempts,
            fallback_used=True,
            status="degraded",
        )
    else:
        output = VisionOutput(
            page=parsed["page"],
            reading_order=parsed["reading_order"],
            merge_groups=parsed["merge_groups"],
            role_labels=parsed["role_labels"],
            confidence=float(parsed["confidence"]),
            fallback_used=parsed.get("fallback_used", False),
            source="model",
        )

    embedded_headings, reviewed_block_ids = review_embedded_heading_blocks(
        page=page,
        blocks=layout_blocks,
        output=output,
        page_image_path=page_image_path,
        qa_dir=qa_dir,
        runtime=runtime,
    )
    output.embedded_headings = embedded_headings
    output.embedded_heading_reviewed_block_ids = reviewed_block_ids
    mixed_group_reviews, mixed_reviewed_block_ids = review_mixed_group_blocks(
        page=page,
        blocks=layout_blocks,
        output=output,
        page_image_path=page_image_path,
        qa_dir=qa_dir,
        runtime=runtime,
    )
    output.mixed_group_reviews = mixed_group_reviews
    output.mixed_group_reviewed_block_ids = mixed_reviewed_block_ids

    output = expand_vision_output_to_raw_blocks(
        output,
        vision_input_blocks,
        raw_blocks,
        hidden_role_labels=hidden_role_labels,
    )

    out_dict = {
        "page": output.page,
        "reading_order": output.reading_order,
        "merge_groups": output.merge_groups,
        "role_labels": output.role_labels,
        "confidence": output.confidence,
        "fallback_used": output.fallback_used,
        "source": output.source,
        "embedded_headings": output.embedded_headings,
        "embedded_heading_reviewed_block_ids": output.embedded_heading_reviewed_block_ids,
        "mixed_group_reviews": output.mixed_group_reviews,
        "mixed_group_reviewed_block_ids": output.mixed_group_reviewed_block_ids,
    }
    out_path = vision_dir / f"p{page:03d}_out.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)

    return output, fault


def run_vision(
    run_dir: Path,
    manifest: Optional[Manifest] = None,
    inject_malformed_json: bool = False,
) -> tuple[int, int]:
    if manifest is None:
        manifest = load_manifest(run_dir)

    dpi = manifest.render_config.dpi
    scale = manifest.render_config.scale

    pages_dir = run_dir / "pages"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    qa_dir = run_dir / "qa"

    vision_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    llm_model = os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL).strip()
    if not inject_malformed_json:
        preflight_ok, preflight_diag = run_preflight_check(llm_model)
        if not preflight_ok:
            preflight_message = str(preflight_diag.get("message", "Vision preflight failed."))
            preflight_error_type = str(preflight_diag.get("error_type", "preflight_failure"))
            append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
                stage="vision",
                page=0,
                attempt=0,
                model=str(preflight_diag.get("model", llm_model)),
                endpoint=str(preflight_diag.get("endpoint", SILICONFLOW_ENDPOINT)),
                success=False,
                parse_success=False,
                validation_success=False,
                error_type=preflight_error_type,
                http_status=None,
                prompt_chars=0,
                response_chars=0,
                response_preview=preflight_message[:300],
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            append_fault_events(
                qa_dir,
                [
                    {
                        "stage": "vision",
                        "fault": f"preflight-{preflight_error_type}",
                        "page": 0,
                        "retry_attempts": 0,
                        "fallback_used": False,
                        "status": "fail",
                        "error_type": preflight_error_type,
                    }
                ],
            )
            raise RuntimeError(preflight_message)

    blocks_by_page = load_blocks(text_dir / "blocks_norm.jsonl", dpi, scale)
    document_profile = load_document_layout_profile(text_dir / "document_layout_profile.json")
    paragraph_regrouping_hints_by_page = load_layout_analysis_hints(text_dir / "layout_analysis.json")
    block_lines_by_block = load_block_line_records(text_dir)

    if not blocks_by_page:
        return 0, 0

    faults: list[FaultEvent] = []
    pages_done = 0
    blocks_done = 0
    total_pages = len(blocks_by_page)
    runtime = VisionRuntimeContext(
        budget=VisionRequestBudget(resolve_vision_request_budget()),
        image_cache=VisionImageDataUrlCache(encode_fn=encode_image_data_url),
    )

    page_fallback_retries = max(0, int(os.environ.get("SILICONFLOW_VISION_PAGE_FALLBACK_RETRIES", "2")))
    consecutive_fallback_switch_threshold = max(
        0,
        int(os.environ.get("SILICONFLOW_VISION_CONSEC_FALLBACK_SWITCH_THRESHOLD", "2")),
    )
    consecutive_fallback_model = (
        os.environ.get("SILICONFLOW_VISION_CONSEC_FALLBACK_MODEL", "").strip()
        or os.environ.get("SILICONFLOW_VISION_FALLBACK_MODEL", SILICONFLOW_FALLBACK_MODEL).strip()
    )

    max_workers = int(os.environ.get("SILICONFLOW_VISION_MAX_WORKERS", "2"))
    max_workers = max(1, min(4, max_workers))
    if consecutive_fallback_switch_threshold > 0:
        max_workers = 1

    def process_page_with_recovery(
        page: int,
        blocks: list[BlockCandidate],
        consecutive_fallback_pages_before: int,
    ) -> tuple[VisionOutput, Optional[FaultEvent], int]:
        attempts_used = 0
        use_alt_model = (
            consecutive_fallback_switch_threshold > 0
            and consecutive_fallback_pages_before >= consecutive_fallback_switch_threshold
        )
        output: VisionOutput
        fault: Optional[FaultEvent]
        while True:
            primary_override = consecutive_fallback_model if use_alt_model else None
            retry_override = consecutive_fallback_model if use_alt_model else None
            output, fault = process_page(
                page,
                blocks,
                pages_dir,
                vision_dir,
                qa_dir,
                inject_malformed_json,
                runtime,
                primary_model_override=primary_override,
                retry_model_override=retry_override,
                document_profile=document_profile,
                paragraph_regrouping_hints_by_page=paragraph_regrouping_hints_by_page,
                block_lines_by_block=block_lines_by_block,
            )
            should_retry_page = (
                not inject_malformed_json
                and output.fallback_used
                and output.source != "coarse_fallback"
                and attempts_used < page_fallback_retries
                and (fault is None or fault.fault != "budget-exhausted")
            )
            if not should_retry_page:
                return output, fault, attempts_used
            attempts_used += 1
            if (
                consecutive_fallback_switch_threshold > 0
                and consecutive_fallback_pages_before + attempts_used >= consecutive_fallback_switch_threshold
            ):
                use_alt_model = True

    if max_workers == 1:
        consecutive_fallback_pages = 0
        for index, page in enumerate(sorted(blocks_by_page.keys()), start=1):
            page_start = time.time()
            blocks = blocks_by_page[page]
            output, fault, page_retries = process_page_with_recovery(page, blocks, consecutive_fallback_pages)
            if fault:
                faults.append(fault)
            pages_done += 1
            blocks_done += len(blocks)
            if output.fallback_used:
                consecutive_fallback_pages += 1
            else:
                consecutive_fallback_pages = 0
            elapsed = time.time() - page_start
            print(
                f"[vision] {index}/{total_pages} page={page} blocks={len(blocks)} "
                f"source={output.source} fallback={output.fallback_used} "
                f"page_retries={page_retries} elapsed={elapsed:.1f}s",
                flush=True,
            )
    else:
        def process_page_timed(page: int, blocks: list[BlockCandidate]) -> tuple[VisionOutput, Optional[FaultEvent], float]:
            started = time.time()
            output, fault, _ = process_page_with_recovery(page, blocks, 0)
            return output, fault, time.time() - started

        ordered_pages = sorted(blocks_by_page.keys())
        future_map: dict[Any, tuple[int, list[BlockCandidate]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for page in ordered_pages:
                blocks = blocks_by_page[page]
                fut = executor.submit(process_page_timed, page, blocks)
                future_map[fut] = (page, blocks)

            for done_count, fut in enumerate(as_completed(future_map), start=1):
                page, blocks = future_map[fut]
                output, fault, elapsed = fut.result()
                if fault:
                    faults.append(fault)
                pages_done += 1
                blocks_done += len(blocks)
                print(
                    f"[vision] {done_count}/{total_pages} page={page} blocks={len(blocks)} "
                    f"source={output.source} fallback={output.fallback_used} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if faults:
        append_fault_events(qa_dir, [asdict(e) for e in faults])

    append_vision_runtime_event(qa_dir, runtime, pages_done=pages_done, total_pages=total_pages)

    return pages_done, blocks_done
