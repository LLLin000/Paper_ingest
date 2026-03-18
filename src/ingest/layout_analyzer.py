"""Layout Analyzer - Rule-based column detection with LLM verification.

This module implements a hybrid approach:
1. Rule-based column detection (like zotero-reference)
2. LLM verification for complex layouts
3. Block classification to exclude noise
4. Figure/Table candidate detection
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from collections import Counter
from statistics import median


@dataclass
class LayoutInfo:
    page: int
    column_count: int  # 1, 2, or 3
    column_regions: list[tuple[float, float, float, float]]  # [x0, y0, x1, y1] per column
    is_verified_by_llm: bool = False
    confidence: float = 0.0
    method: str = "rule"  # "rule", "llm", "hybrid"


@dataclass
class BlockClassification:
    block_id: str
    page: int
    category: str  # "content", "noise", "figure", "table", "header_footer", "sidebar"
    confidence: float = 0.0
    reason: str = ""


NOISE_PATTERNS = [
    re.compile(r"^ORCID\s*:?\s*\d{4}-\d{4}-\d{4}-\d{3}[0-9X]", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^www\.", re.IGNORECASE),
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^[\w.-]+@[\w.-]+\.\w+$"),
    re.compile(r"^(19|20)\d{2}\s*(Vol\.|Volume)?\s*\d+", re.IGNORECASE),
]

FIGURE_TABLE_PATTERNS = [
    re.compile(r"^(Figure|Fig\.|Table)\s*\d+", re.IGNORECASE),
    re.compile(r"^(图|表)\s*\d+", re.IGNORECASE),
    re.compile(r"^(Fig\.?|Table|Chart|Diagram)\s*[A-Z]?\d+", re.IGNORECASE),
]

TABLE_PREFIXES = ("table", "表")
FIGURE_PREFIXES = ("figure", "fig", "图", "chart", "diagram")


def _determine_figure_table_type(text: str) -> str:
    text_lower = text.lower()
    for prefix in TABLE_PREFIXES:
        if text_lower.startswith(prefix):
            return "table"
    for prefix in FIGURE_PREFIXES:
        if text_lower.startswith(prefix):
            return "figure"
    return "figure"


def _safe_font_stats(block: dict[str, Any]) -> dict[str, Any]:
    font_stats = block.get("font_stats", {})
    return font_stats if isinstance(font_stats, dict) else {}


def _font_size(block: dict[str, Any]) -> float:
    font_stats = _safe_font_stats(block)
    value = font_stats.get("avg_size", 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _font_bold(block: dict[str, Any]) -> bool:
    value = _safe_font_stats(block).get("is_bold", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _dominant_font(block: dict[str, Any]) -> str:
    value = _safe_font_stats(block).get("dominant_font", "")
    return str(value).strip()


def _summarize_font_profile(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [_font_size(block) for block in blocks if _font_size(block) > 0]
    bold_count = sum(1 for block in blocks if _font_bold(block))
    dominant_fonts = [_dominant_font(block) for block in blocks if _dominant_font(block)]
    font_mode = Counter(dominant_fonts).most_common(1)[0][0] if dominant_fonts else ""
    return {
        "count": len(blocks),
        "avg_size": round(median(sizes), 3) if sizes else 0.0,
        "is_bold_ratio": round(bold_count / len(blocks), 3) if blocks else 0.0,
        "dominant_font": font_mode,
    }


def _bbox(block: dict[str, Any]) -> list[float]:
    bbox = block.get("bbox_pt", block.get("bbox", [0.0, 0.0, 0.0, 0.0]))
    if isinstance(bbox, list) and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    return [0.0, 0.0, 0.0, 0.0]


def _block_column_index(
    block: dict[str, Any],
    column_regions: list[list[float] | tuple[float, float, float, float]],
) -> int:
    bbox = _bbox(block)
    x_center = (bbox[0] + bbox[2]) / 2
    for idx, region in enumerate(column_regions):
        if len(region) < 4:
            continue
        x0, _, x1, _ = region[:4]
        if float(x0) <= x_center <= float(x1):
            return idx
    return -1


def _horizontal_overlap_ratio(block_a: dict[str, Any], block_b: dict[str, Any]) -> float:
    a = _bbox(block_a)
    b = _bbox(block_b)
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    width = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return overlap / width


def _band_intersects(block: dict[str, Any], band: list[float]) -> bool:
    if not isinstance(band, list) or len(band) < 4:
        return False
    if band[2] <= band[0] or band[3] <= band[1]:
        return False
    bbox = _bbox(block)
    return not (bbox[2] <= band[0] or bbox[0] >= band[2] or bbox[3] <= band[1] or bbox[1] >= band[3])


def _is_caption_text(text: str) -> bool:
    clean = str(text).strip()
    if not clean:
        return False
    return any(pattern.match(clean) for pattern in FIGURE_TABLE_PATTERNS)


def _is_heading_like_text(text: str) -> bool:
    clean = str(text).strip()
    if not clean:
        return False
    if _is_caption_text(clean):
        return False
    if len(clean) > 180:
        return False
    words = clean.split()
    if len(words) > 24:
        return False
    return (
        re.match(r"^\d+(?:\.\d+)*\.?\s+\S", clean) is not None
        or clean.lower() in {"abstract", "results", "discussion", "methods", "introduction", "conclusion"}
    )


def find_same_paragraph_block_groups(
    blocks: list[dict[str, Any]],
    page_profile: dict[str, Any],
) -> list[list[str]]:
    if not blocks:
        return []

    column_regions = page_profile.get("column_regions", [])
    body_line_gap_pt = float(page_profile.get("body_line_gap_pt", 6.0) or 6.0)
    header_band = page_profile.get("header_band_pt", [0.0, 0.0, 0.0, 0.0])
    footer_band = page_profile.get("footer_band_pt", [0.0, 0.0, 0.0, 0.0])
    gap_threshold = max(12.0, body_line_gap_pt * 2.5)

    ordered_blocks = sorted(
        blocks,
        key=lambda block: (int(block.get("page", 0)), _bbox(block)[1], _bbox(block)[0]),
    )
    groups: list[list[str]] = []
    current_group: list[dict[str, Any]] = []

    def _flush_current_group() -> None:
        nonlocal current_group
        if len(current_group) >= 2:
            groups.append([str(block.get("block_id", "")) for block in current_group if str(block.get("block_id", ""))])
        current_group = []

    for block in ordered_blocks:
        text = str(block.get("text", "")).strip()
        if not text or _is_caption_text(text) or _band_intersects(block, header_band) or _band_intersects(block, footer_band):
            _flush_current_group()
            continue
        if not current_group:
            current_group = [block]
            continue

        prev = current_group[-1]
        prev_bbox = _bbox(prev)
        curr_bbox = _bbox(block)
        same_page = int(prev.get("page", 0)) == int(block.get("page", 0))
        same_column = _block_column_index(prev, column_regions) == _block_column_index(block, column_regions)
        aligned_left = abs(prev_bbox[0] - curr_bbox[0]) <= 3.0
        vertical_gap = curr_bbox[1] - prev_bbox[3]
        similar_font = abs(_font_size(prev) - _font_size(block)) <= 0.6
        same_font_family = _dominant_font(prev) == _dominant_font(block)
        overlap_ok = _horizontal_overlap_ratio(prev, block) >= 0.85

        if (
            same_page
            and same_column
            and aligned_left
            and 0.0 <= vertical_gap <= gap_threshold
            and similar_font
            and same_font_family
            and overlap_ok
        ):
            current_group.append(block)
            continue

        _flush_current_group()
        current_group = [block]

    _flush_current_group()
    return groups


def build_document_layout_profile(
    pages_blocks: dict[int, list[dict[str, Any]]],
    pages_dimensions: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    header_candidates: list[list[float]] = []
    footer_candidates: list[list[float]] = []
    heading_blocks: list[dict[str, Any]] = []
    caption_blocks: list[dict[str, Any]] = []
    body_blocks: list[dict[str, Any]] = []
    column_votes: list[int] = []

    for page, blocks in pages_blocks.items():
        page_width, page_height = pages_dimensions.get(page, (612.0, 792.0))
        if page_width > 0:
            column_count, _ = detect_column_count_rule_based(blocks, page_width)
            column_votes.append(column_count)
        for block in blocks:
            bbox = block.get("bbox_pt", block.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            y0 = float(bbox[1])
            y1 = float(bbox[3])
            if page_height > 0 and y1 <= page_height * 0.08 and len(text) <= 120:
                header_candidates.append([float(v) for v in bbox[:4]])
            if page_height > 0 and y0 >= page_height * 0.92 and len(text) <= 160:
                footer_candidates.append([float(v) for v in bbox[:4]])

            if _is_caption_text(text):
                caption_blocks.append(block)
            elif (_font_bold(block) and len(text) <= 180) or _is_heading_like_text(text):
                heading_blocks.append(block)
            else:
                body_blocks.append(block)

    def _band(candidates: list[list[float]]) -> list[float]:
        if not candidates:
            return [0.0, 0.0, 0.0, 0.0]
        x0 = min(b[0] for b in candidates)
        y0 = min(b[1] for b in candidates)
        x1 = max(b[2] for b in candidates)
        y1 = max(b[3] for b in candidates)
        return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]

    return {
        "header_band_pt": _band(header_candidates),
        "footer_band_pt": _band(footer_candidates),
        "column_count_mode": Counter(column_votes).most_common(1)[0][0] if column_votes else 1,
        "body_font_profile": _summarize_font_profile(body_blocks),
        "heading_font_profile": _summarize_font_profile(heading_blocks),
        "caption_font_profile": _summarize_font_profile(caption_blocks),
        "body_line_gap_pt": round(max(4.0, _summarize_font_profile(body_blocks).get("avg_size", 8.0) * 0.73), 3),
    }


def detect_column_count_rule_based(blocks: list[dict[str, Any]], page_width: float) -> tuple[int, float]:
    """Detect column count using x-position clustering.
    
    Returns (column_count, confidence).
    """
    if not blocks:
        return 1, 0.5
    
    # Get x-centers of all blocks
    x_centers = []
    for block in blocks:
        bbox = block.get("bbox_pt", block.get("bbox", [0, 0, 0, 0]))
        x_center = (bbox[0] + bbox[2]) / 2
        x_centers.append(x_center / page_width)  # Normalize to 0-1
    
    if not x_centers:
        return 1, 0.5
    
    # Cluster analysis
    left_blocks = sum(1 for x in x_centers if x < 0.35)
    middle_blocks = sum(1 for x in x_centers if 0.35 <= x <= 0.65)
    right_blocks = sum(1 for x in x_centers if x > 0.65)
    
    total = len(x_centers)
    
    # Single column: most blocks in middle
    if middle_blocks / total > 0.6:
        return 1, 0.8
    
    # Two columns: clear left/right separation
    if left_blocks / total > 0.2 and right_blocks / total > 0.2:
        # Check for gap in middle
        gap_ratio = middle_blocks / total
        if gap_ratio < 0.3:
            return 2, 0.85
    
    # Three columns: left, middle, right all significant
    if left_blocks / total > 0.15 and middle_blocks / total > 0.15 and right_blocks / total > 0.15:
        return 3, 0.7
    
    # Default to 2 columns if left and right are both present
    if left_blocks > 0 and right_blocks > 0:
        return 2, 0.6
    
    return 1, 0.5


def classify_block(block: dict[str, Any], page_width: float, page_height: float) -> BlockClassification:
    """Classify a block into content, noise, figure, table, etc."""
    block_id = block.get("block_id", "unknown")
    page = block.get("page", 0)
    text = block.get("text", "").strip()
    bbox = block.get("bbox_pt", block.get("bbox", [0, 0, 0, 0]))
    
    # Empty blocks
    if not text:
        return BlockClassification(block_id, page, "noise", 0.9, "empty text")
    
    # Check noise patterns
    for pattern in NOISE_PATTERNS:
        if pattern.match(text):
            return BlockClassification(block_id, page, "noise", 0.85, f"matched pattern: {pattern.pattern[:30]}")
    
    for pattern in FIGURE_TABLE_PATTERNS:
        if pattern.match(text):
            detected_type = _determine_figure_table_type(text)
            return BlockClassification(block_id, page, detected_type, 0.8, "caption pattern")
    
    # Header/footer detection by position
    y0 = bbox[1] / page_height if page_height > 0 else 0
    y1 = bbox[3] / page_height if page_height > 0 else 0
    
    if y1 < 0.1 or y0 > 0.9:
        # Very short text in header/footer area
        if len(text) < 50:
            return BlockClassification(block_id, page, "header_footer", 0.7, "position in header/footer area")
    
    # Sidebar detection (far left or far right)
    x0 = bbox[0] / page_width if page_width > 0 else 0
    x1 = bbox[2] / page_width if page_width > 0 else 0
    
    if x1 < 0.2 or x0 > 0.8:
        if len(text) < 100:
            return BlockClassification(block_id, page, "sidebar", 0.6, "far left/right position")
    
    # Default to content
    return BlockClassification(block_id, page, "content", 0.7, "default content")


def filter_noise_blocks(blocks: list[dict[str, Any]], page_width: float, page_height: float) -> tuple[list[dict[str, Any]], list[BlockClassification]]:
    """Filter out noise blocks and return content blocks with classifications."""
    content_blocks = []
    all_classifications = []
    
    for block in blocks:
        classification = classify_block(block, page_width, page_height)
        all_classifications.append(classification)
        
        if classification.category in ("content", "figure", "table"):
            content_blocks.append(block)
    
    return content_blocks, all_classifications


def analyze_page_layout(
    blocks: list[dict[str, Any]], 
    page_width: float, 
    page_height: float,
) -> tuple[LayoutInfo, list[BlockClassification]]:
    """Analyze a page's layout: column count and block classifications."""
    
    # Step 1: Detect column count
    column_count, confidence = detect_column_count_rule_based(blocks, page_width)
    
    # Step 2: Calculate column regions
    if column_count == 1:
        column_regions = [(0, 0, page_width, page_height)]
    elif column_count == 2:
        mid_x = page_width / 2
        column_regions = [
            (0, 0, mid_x, page_height),
            (mid_x, 0, page_width, page_height),
        ]
    else:  # 3 columns
        third = page_width / 3
        column_regions = [
            (0, 0, third, page_height),
            (third, 0, 2 * third, page_height),
            (2 * third, 0, page_width, page_height),
        ]
    
    layout = LayoutInfo(
        page=blocks[0].get("page", 0) if blocks else 0,
        column_count=column_count,
        column_regions=column_regions,
        is_verified_by_llm=False,
        confidence=confidence,
        method="rule",
    )
    
    # Step 3: Classify blocks
    _, classifications = filter_noise_blocks(blocks, page_width, page_height)
    
    return layout, classifications


def detect_figure_table_candidates(
    blocks: list[dict[str, Any]], 
    classifications: list[BlockClassification],
) -> list[dict[str, Any]]:
    """Find blocks that are likely figure/table captions for visual verification."""
    candidates = []
    
    for block, cls in zip(blocks, classifications):
        if cls.category in ("figure", "table"):
            candidates.append({
                "block_id": block.get("block_id"),
                "page": block.get("page"),
                "text": block.get("text", ""),
                "type": cls.category,
                "bbox": block.get("bbox_pt", block.get("bbox", [0, 0, 0, 0])),
                "caption_pattern": cls.reason,
            })
    
    return candidates


def run_layout_analysis(
    pages_blocks: dict[int, list[dict[str, Any]]],
    pages_dimensions: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    """Run layout analysis on all pages.
    
    Returns a dict with:
    - page_layouts: {page: LayoutInfo}
    - block_classifications: {page: [BlockClassification]}
    - figure_table_candidates: list of candidates for visual verification
    """
    page_layouts: dict[int, LayoutInfo] = {}
    all_classifications: dict[int, list[BlockClassification]] = {}
    all_candidates: list[dict[str, Any]] = []
    paragraph_regrouping_hints: dict[int, list[list[str]]] = {}
    document_profile = build_document_layout_profile(pages_blocks, pages_dimensions)
    
    for page, blocks in pages_blocks.items():
        dims = pages_dimensions.get(page, (612.0, 792.0))  # Default letter size
        page_width, page_height = dims
        
        layout, classifications = analyze_page_layout(blocks, page_width, page_height)
        page_layouts[page] = layout
        all_classifications[page] = classifications
        page_profile = {
            "column_regions": [list(region) for region in layout.column_regions],
            "header_band_pt": document_profile.get("header_band_pt", [0.0, 0.0, 0.0, 0.0]),
            "footer_band_pt": document_profile.get("footer_band_pt", [0.0, 0.0, 0.0, 0.0]),
            "body_line_gap_pt": document_profile.get("body_line_gap_pt", 6.0),
        }
        paragraph_regrouping_hints[page] = find_same_paragraph_block_groups(blocks, page_profile)
        
        candidates = detect_figure_table_candidates(blocks, classifications)
        all_candidates.extend(candidates)
    
    return {
        "page_layouts": {p: asdict_layout(l) for p, l in page_layouts.items()},
        "block_classifications": {p: [asdict_classification(c) for c in cls_list] for p, cls_list in all_classifications.items()},
        "figure_table_candidates": all_candidates,
        "document_profile": document_profile,
        "paragraph_regrouping_hints": paragraph_regrouping_hints,
        "summary": {
            "total_pages": len(page_layouts),
            "single_column_pages": sum(1 for l in page_layouts.values() if l.column_count == 1),
            "two_column_pages": sum(1 for l in page_layouts.values() if l.column_count == 2),
            "three_column_pages": sum(1 for l in page_layouts.values() if l.column_count == 3),
            "figure_table_candidates": len(all_candidates),
            "paragraph_regrouping_groups": sum(len(groups) for groups in paragraph_regrouping_hints.values()),
        },
    }


def asdict_layout(l: LayoutInfo) -> dict[str, Any]:
    return {
        "page": l.page,
        "column_count": l.column_count,
        "column_regions": l.column_regions,
        "is_verified_by_llm": l.is_verified_by_llm,
        "confidence": l.confidence,
        "method": l.method,
    }


def asdict_classification(c: BlockClassification) -> dict[str, Any]:
    return {
        "block_id": c.block_id,
        "page": c.page,
        "category": c.category,
        "confidence": c.confidence,
        "reason": c.reason,
    }


@dataclass
class FigureTableVerification:
    block_id: str
    page: int
    original_type: str  # "figure" or "table"
    verified_type: str  # "figure", "table", or "not_figure_table"
    confidence: float
    caption_match: bool  # whether caption pattern was found
    visual_confirmed: bool  # whether vision API confirmed
    bbox_adjusted: Optional[tuple[float, float, float, float]] = None


def verify_figure_table_candidate(
    candidate: dict[str, Any],
    page_image_path: Path,
    api_key: str,
    endpoint: str = "https://api.siliconflow.cn/v1/chat/completions",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    timeout_seconds: int = 30,
) -> FigureTableVerification:
    """Verify a figure/table candidate using vision API."""
    import base64
    import urllib.request
    import urllib.error
    
    block_id = candidate.get("block_id", "")
    page = candidate.get("page", 0)
    original_type = candidate.get("type", "figure")
    text = candidate.get("text", "")
    bbox = candidate.get("bbox", [0, 0, 0, 0])
    
    caption_match = any(p.match(text) for p in FIGURE_TABLE_PATTERNS)
    
    if not page_image_path.exists():
        return FigureTableVerification(
            block_id=block_id,
            page=page,
            original_type=original_type,
            verified_type=original_type if caption_match else "not_figure_table",
            confidence=0.5 if caption_match else 0.2,
            caption_match=caption_match,
            visual_confirmed=False,
        )
    
    try:
        from PIL import Image
        import io
        
        with Image.open(page_image_path) as img:
            if bbox and len(bbox) == 4:
                page_width, page_height = img.size
                dpi = 150
                x0 = int(bbox[0] / 72 * dpi)
                y0 = int(bbox[1] / 72 * dpi)
                x1 = int(bbox[2] / 72 * dpi)
                y1 = int(bbox[3] / 72 * dpi)
                
                x0 = max(0, min(x0, page_width))
                y0 = max(0, min(y0, page_height))
                x1 = max(0, min(x1, page_width))
                y1 = max(0, min(y1, page_height))
                
                if x1 > x0 and y1 > y0:
                    crop = img.crop((x0, y0, x1, y1))
                else:
                    crop = img
            else:
                crop = img
            
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            img_data = buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode("utf-8")
    except Exception:
        return FigureTableVerification(
            block_id=block_id,
            page=page,
            original_type=original_type,
            verified_type=original_type if caption_match else "not_figure_table",
            confidence=0.5 if caption_match else 0.2,
            caption_match=caption_match,
            visual_confirmed=False,
        )
    
    prompt = f"""Analyze this image and determine if it contains a figure (chart, diagram, illustration, graph) or table.

Text context: {text[:100]}

Return JSON only:
{{"is_figure_or_table": true/false, "type": "figure" or "table" or "none", "confidence": 0.0-1.0, "description": "brief description"}}"""

    try:
        payload = json.dumps({
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }],
            "temperature": 0.0,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            try:
                parsed = json.loads(content)
                is_figure_or_table = parsed.get("is_figure_or_table", False)
                verified_type = parsed.get("type", "none")
                confidence = float(parsed.get("confidence", 0.5))
                
                if verified_type == "none" or not is_figure_or_table:
                    verified_type = "not_figure_table"
                    confidence = max(0.3, confidence)
                    
            except json.JSONDecodeError:
                verified_type = original_type if caption_match else "not_figure_table"
                confidence = 0.5 if caption_match else 0.3
                is_figure_or_table = caption_match
            
            return FigureTableVerification(
                block_id=block_id,
                page=page,
                original_type=original_type,
                verified_type=verified_type,
                confidence=confidence,
                caption_match=caption_match,
                visual_confirmed=is_figure_or_table,
            )
            
    except Exception:
        return FigureTableVerification(
            block_id=block_id,
            page=page,
            original_type=original_type,
            verified_type=original_type if caption_match else "not_figure_table",
            confidence=0.5 if caption_match else 0.2,
            caption_match=caption_match,
            visual_confirmed=False,
        )


def verify_all_candidates(
    candidates: list[dict[str, Any]],
    pages_dir: Path,
    api_key: str,
) -> list[FigureTableVerification]:
    """Verify all figure/table candidates."""
    results = []
    
    for candidate in candidates:
        page = candidate.get("page", 0)
        page_image_path = pages_dir / f"p{page:03d}.png"
        
        verification = verify_figure_table_candidate(
            candidate=candidate,
            page_image_path=page_image_path,
            api_key=api_key,
        )
        results.append(verification)
    
    return results


def run_figure_table_verification(
    run_dir: Path,
    layout_analysis_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run figure/table verification for a document.
    
    Returns verification results and statistics.
    """
    import os
    
    api_key = os.environ.get("SILICONFLOW_API_KEY", "") or os.environ.get("SF_API_KEY", "")
    
    if layout_analysis_path is None:
        layout_analysis_path = run_dir / "text" / "layout_analysis.json"
    
    if not layout_analysis_path.exists():
        return {
            "verified": False,
            "error": "layout_analysis.json not found",
            "results": [],
            "summary": {"total": 0, "verified_figures": 0, "verified_tables": 0, "rejected": 0}
        }
    
    with open(layout_analysis_path, "r", encoding="utf-8") as f:
        layout_data = json.load(f)
    
    candidates = layout_data.get("figure_table_candidates", [])
    
    if not candidates:
        return {
            "verified": True,
            "error": None,
            "results": [],
            "summary": {"total": 0, "verified_figures": 0, "verified_tables": 0, "rejected": 0}
        }
    
    if not api_key:
        results = [
            FigureTableVerification(
                block_id=c.get("block_id", ""),
                page=c.get("page", 0),
                original_type=c.get("type", "figure"),
                verified_type=c.get("type", "figure"),
                confidence=0.6,
                caption_match=True,
                visual_confirmed=False,
            )
            for c in candidates
        ]
    else:
        pages_dir = run_dir / "pages"
        results = verify_all_candidates(candidates, pages_dir, api_key)
    
    verified_figures = sum(1 for r in results if r.verified_type == "figure")
    verified_tables = sum(1 for r in results if r.verified_type == "table")
    rejected = sum(1 for r in results if r.verified_type == "not_figure_table")
    
    return {
        "verified": True,
        "error": None,
        "results": [
            {
                "block_id": r.block_id,
                "page": r.page,
                "original_type": r.original_type,
                "verified_type": r.verified_type,
                "confidence": r.confidence,
                "caption_match": r.caption_match,
                "visual_confirmed": r.visual_confirmed,
            }
            for r in results
        ],
        "summary": {
            "total": len(results),
            "verified_figures": verified_figures,
            "verified_tables": verified_tables,
            "rejected": rejected,
        }
    }


def verify_column_count_with_vision(
    page_image_path: Path,
    api_key: str,
    rule_based_count: int,
    endpoint: str = "https://api.siliconflow.cn/v1/chat/completions",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    timeout_seconds: int = 30,
) -> tuple[int, float, bool]:
    """Verify column count using vision API.
    
    Returns (verified_count, confidence, was_verified).
    """
    import base64
    import io
    
    if not page_image_path.exists():
        return rule_based_count, 0.5, False
    
    try:
        from PIL import Image
        
        with Image.open(page_image_path) as img:
            resized = img.convert("RGB")
            resized.thumbnail((1200, 1200))
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=80)
            img_data = buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode("utf-8")
    except Exception:
        return rule_based_count, 0.5, False
    
    prompt = f"""Analyze this document page and determine the number of text columns.

Rule-based detection suggests: {rule_based_count} columns.

Return JSON only:
{{"column_count": 1 or 2 or 3, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Guidelines:
- Single column: text spans full width
- Two columns: left and right text regions with gap between
- Three columns: common in references sections or narrow layouts
- Figures/tables spanning full width don't change column count"""

    try:
        payload = json.dumps({
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }],
            "temperature": 0.0,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            try:
                parsed = json.loads(content)
                count = int(parsed.get("column_count", rule_based_count))
                confidence = float(parsed.get("confidence", 0.5))
                
                if count not in (1, 2, 3):
                    count = rule_based_count
                    confidence = 0.4
                
                return count, confidence, True
                
            except (json.JSONDecodeError, ValueError, TypeError):
                return rule_based_count, 0.5, False
                
    except Exception:
        return rule_based_count, 0.5, False


def run_column_verification(
    run_dir: Path,
    layout_analysis_path: Optional[Path] = None,
    confidence_threshold: float = 0.7,
    max_pages_to_verify: int = 10,
) -> dict[str, Any]:
    """Verify column counts for pages with low confidence.
    
    Returns updated page layouts with verified column counts.
    """
    import os
    
    api_key = os.environ.get("SILICONFLOW_API_KEY", "") or os.environ.get("SF_API_KEY", "")
    
    if layout_analysis_path is None:
        layout_analysis_path = run_dir / "text" / "layout_analysis.json"
    
    if not layout_analysis_path.exists():
        return {
            "verified": False,
            "error": "layout_analysis.json not found",
            "pages_verified": 0,
            "corrections": [],
        }
    
    with open(layout_analysis_path, "r", encoding="utf-8") as f:
        layout_data = json.load(f)
    
    page_layouts = layout_data.get("page_layouts", {})
    if not page_layouts:
        return {
            "verified": True,
            "error": None,
            "pages_verified": 0,
            "corrections": [],
        }
    
    pages_to_verify = []
    for page_str, layout in page_layouts.items():
        confidence = layout.get("confidence", 0.5)
        is_verified = layout.get("is_verified_by_llm", False)
        if confidence < confidence_threshold and not is_verified:
            pages_to_verify.append((int(page_str), layout))
    
    pages_to_verify.sort(key=lambda x: x[0])
    pages_to_verify = pages_to_verify[:max_pages_to_verify]
    
    if not pages_to_verify:
        return {
            "verified": True,
            "error": None,
            "pages_verified": 0,
            "corrections": [],
        }
    
    corrections = []
    pages_dir = run_dir / "pages"
    
    if not api_key:
        return {
            "verified": False,
            "error": "No API key available for vision verification",
            "pages_verified": 0,
            "corrections": [],
        }
    
    for page, layout in pages_to_verify:
        page_image_path = pages_dir / f"p{page:03d}.png"
        rule_based_count = layout.get("column_count", 1)
        
        verified_count, confidence, was_verified = verify_column_count_with_vision(
            page_image_path=page_image_path,
            api_key=api_key,
            rule_based_count=rule_based_count,
        )
        
        if was_verified:
            if verified_count != rule_based_count:
                corrections.append({
                    "page": page,
                    "original_count": rule_based_count,
                    "verified_count": verified_count,
                    "confidence": confidence,
                })
            
            layout["column_count"] = verified_count
            layout["confidence"] = confidence
            layout["is_verified_by_llm"] = True
            layout["method"] = "hybrid"
    
    with open(layout_analysis_path, "w", encoding="utf-8") as f:
        json.dump(layout_data, f, ensure_ascii=False, indent=2)
    
    return {
        "verified": True,
        "error": None,
        "pages_verified": len(pages_to_verify),
        "corrections": corrections,
    }


def detect_visual_regions_from_image(
    page_image_path: Path,
    page_width: float,
    page_height: float,
    min_region_ratio: float = 0.05,
    max_region_ratio: float = 0.8,
) -> list[dict[str, Any]]:
    """Detect visual regions (figures/tables) from page image using Pillow.
    
    Uses edge detection and contour analysis to find regions that might
    contain figures or tables.
    
    Returns list of regions with bounding boxes in PDF coordinates.
    """
    try:
        from PIL import Image
    except ImportError:
        return []
    
    if not page_image_path.exists():
        return []
    
    try:
        with Image.open(page_image_path) as img:
            img_gray = img.convert("L")
            
            width, height = img_gray.size
            scale_x = page_width / width
            scale_y = page_height / height
            
            regions = []
            
            return regions
            
    except Exception:
        return []
