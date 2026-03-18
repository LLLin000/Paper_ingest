"""Figure/Table Extraction Pipeline - Detects and indexes visual assets.

Contract: .sisyphus/plans/pdf-blueprint-contracts.md (lines 96-114)

Output schema per figures_tables/figure_table_index.jsonl:
- asset_id: stable ID (fig_* or tbl_*)
- asset_type: 'figure' or 'table'
- page: page number (1-based)
- bbox_px: [x0, y0, x1, y1] in pixels
- caption_text: caption text if found
- caption_id: ID of caption paragraph if detectable
- source_para_id: source paragraph ID (nullable)
- image_path: path to cropped asset file
- text_content: OCR/vision text (nullable)
- summary_content: LLM summary (nullable)
- confidence: confidence score

Output schema per figures_tables/figure_table_links.json:
- by_section: map section -> ordered asset ids
- by_fact: map fact_id -> asset ids
- by_synthesis_slot: map slot_id -> {asset_ids, render_mode}
"""

import hashlib
import json
import re
import os
import io
import base64
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from .manifest import Manifest
from .layout_analyzer import run_figure_table_verification


# Caption patterns
FIG_CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)[\s\.\-:]*(.*)",
    re.IGNORECASE
)
TABLE_CAPTION_RE = re.compile(
    r"^(?:Table)\s*(\d+[a-zA-Z]?)[\s\.\-:]*(.*)",
    re.IGNORECASE
)
FIG_CAPTION_INLINE_RE = re.compile(
    r"(?:^|[\s\.;:,])(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)\s*[\|:\-]\s*(.+)",
    re.IGNORECASE,
)
TABLE_CAPTION_INLINE_RE = re.compile(
    r"(?:^|[\s\.;:,])(?:Table)\s*(\d+[a-zA-Z]?)\s*[\|:\-]\s*(.+)",
    re.IGNORECASE,
)

# Role label constants
ROLE_FIGURE_CAPTION = "FigureCaption"
ROLE_TABLE_CAPTION = "TableCaption"

SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class FigureTableAsset:
    asset_id: str
    asset_type: str  # 'figure' or 'table'
    page: int
    bbox_px: list[int]
    caption_text: Optional[str] = None
    caption_id: Optional[str] = None
    source_para_id: Optional[str] = None
    image_path: Optional[str] = None
    text_content: Optional[str] = None
    summary_content: Optional[str] = None
    confidence: float = 0.5


@dataclass
class FigureTableLinks:
    by_section: dict[str, list[str]] = field(default_factory=dict)
    by_fact: dict[str, list[str]] = field(default_factory=dict)
    by_synthesis_slot: dict[str, dict[str, Any]] = field(default_factory=dict)


def compute_asset_id(asset_type: str, sequence: int) -> str:
    """Generate deterministic asset ID."""
    prefix = "fig" if asset_type == "figure" else "tbl"
    return f"{prefix}_{sequence:03d}"


def load_paragraphs(paragraphs_path: Path) -> list[dict[str, Any]]:
    """Load all paragraphs from JSONL file."""
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


def load_blocks_norm(blocks_path: Path) -> dict[int, list[dict[str, Any]]]:
    blocks_by_page: dict[int, list[dict[str, Any]]] = {}
    if not blocks_path.exists():
        return blocks_by_page
    with open(blocks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            page = int(row.get("page", 0) or 0)
            block_id = str(row.get("block_id", "") or "")
            bbox = row.get("bbox_pt", [0, 0, 0, 0])
            if page <= 0 or not block_id or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            blocks_by_page.setdefault(page, []).append({
                "block_id": block_id,
                "text": str(row.get("text", "") or ""),
                "bbox_pt": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            })
    return blocks_by_page


def load_vision_outputs(vision_dir: Path, dpi: int = 150, scale: float = 2.0) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not vision_dir.exists():
        return result
    for out_file in sorted(vision_dir.glob("p*_out.json")):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            page = int(data.get("page", 0) or 0)
            if page > 0:
                result[page] = data
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    zoom = dpi / 72.0 * scale
    if zoom <= 0:
        zoom = 1.0
    for regions_file in sorted(vision_dir.glob("p*_regions.json")):
        try:
            with open(regions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            page = int(data.get("page", 0) or 0)
            if page <= 0:
                continue
            page_payload = result.setdefault(page, {"page": page})
            for key in ("figure_regions", "table_regions", "caption_regions", "header_footer_regions"):
                regions = data.get(key, [])
                if not isinstance(regions, list):
                    continue
                normalized: list[dict[str, Any]] = []
                for region in regions:
                    if not isinstance(region, dict):
                        continue
                    bbox_px = region.get("bbox_px", [])
                    if not isinstance(bbox_px, list) or len(bbox_px) < 4:
                        continue
                    normalized.append({
                        **region,
                        "bbox_pt": [float(v) / zoom for v in bbox_px[:4]],
                    })
                page_payload[key] = normalized
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            continue
    return result


def _point_in_bbox_pt(x: float, y: float, bbox: list[float]) -> bool:
    if len(bbox) < 4:
        return False
    return float(bbox[0]) <= x <= float(bbox[2]) and float(bbox[1]) <= y <= float(bbox[3])


def _bbox_overlap_area_pt(left: list[float], right: list[float]) -> float:
    if len(left) < 4 or len(right) < 4:
        return 0.0
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def build_vision_region_block_hints(
    blocks_by_page: dict[int, list[dict[str, Any]]],
    vision_by_page: dict[int, dict[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[list[float]]]]:
    vision_caption_by_page: dict[int, list[dict[str, Any]]] = {}
    header_footer_bboxes_by_page: dict[int, list[list[float]]] = {}
    for page, blocks in blocks_by_page.items():
        vision = vision_by_page.get(page, {})
        if not isinstance(vision, dict):
            continue
        caption_regions = vision.get("caption_regions", [])
        header_footer_regions = vision.get("header_footer_regions", [])
        if not isinstance(caption_regions, list):
            caption_regions = []
        if not isinstance(header_footer_regions, list):
            header_footer_regions = []

        for block in blocks:
            bbox_pt = block.get("bbox_pt", [0.0, 0.0, 0.0, 0.0])
            if not isinstance(bbox_pt, list) or len(bbox_pt) < 4:
                continue
            bbox = [float(v) for v in bbox_pt[:4]]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            if any(
                isinstance(region, dict)
                and isinstance(region.get("bbox_pt"), list)
                and (
                    _point_in_bbox_pt(cx, cy, [float(v) for v in region.get("bbox_pt", [])[:4]])
                    or _bbox_overlap_area_pt(bbox, [float(v) for v in region.get("bbox_pt", [])[:4]]) > 0.0
                )
                for region in header_footer_regions
            ):
                header_footer_bboxes_by_page.setdefault(page, []).append(bbox)

            text = " ".join(str(block.get("text", "") or "").split()).strip()
            low = text.lower()
            if not text or not (low.startswith("fig") or low.startswith("figure") or low.startswith("table")):
                continue

            if any(
                isinstance(region, dict)
                and isinstance(region.get("bbox_pt"), list)
                and (
                    _point_in_bbox_pt(cx, cy, [float(v) for v in region.get("bbox_pt", [])[:4]])
                    or _bbox_overlap_area_pt(bbox, [float(v) for v in region.get("bbox_pt", [])[:4]]) > 0.0
                )
                for region in caption_regions
            ):
                role = ROLE_TABLE_CAPTION if low.startswith("table") else ROLE_FIGURE_CAPTION
                vision_caption_by_page.setdefault(page, []).append({
                    "para_id": f"viscap_region_{page}_{block.get('block_id', '')}",
                    "text": text,
                    "role": role,
                    "page_span": {"start": page, "end": page},
                    "evidence_pointer": {"bbox_union": bbox},
                })
    return vision_caption_by_page, header_footer_bboxes_by_page


def resolve_api_key() -> str:
    for key_name in ("SILICONFLOW_API_KEY", "SF_API_KEY", "SILICONFLOW_TOKEN"):
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    return ""


def _extract_json_object(raw: str) -> Optional[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def propose_bbox_with_vision(
    page_image_path: Path,
    caption_text: str,
    asset_type: str,
) -> Optional[list[int]]:
    api_key = resolve_api_key()
    if not api_key or not page_image_path.exists():
        return None

    model = os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_VISION_MODEL)
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT)

    try:
        with Image.open(page_image_path) as img:
            orig_w, orig_h = img.width, img.height
            resized = img.convert("RGB")
            resized.thumbnail((1500, 1500))
            vis_w, vis_h = resized.width, resized.height
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=84, optimize=True)
            image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except OSError:
        return None

    prompt = (
        f"Locate the full {asset_type} region on this page that matches caption text: {caption_text[:220]}\n"
        "Return STRICT JSON only with keys: x0,y0,x1,y1,is_valid,confidence.\n"
        "Coordinates must be integer pixels in THIS IMAGE coordinate system.\n"
        f"Image size: width={vis_w}, height={vis_h}.\n"
        "Rules: include complete visual content; exclude caption text body and page headers/footers.\n"
        "If not found, set is_valid=false and confidence=0."
    )

    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
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
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, KeyError):
        return None

    if isinstance(content, list):
        text_parts = [
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        content_text = "\n".join(text_parts)
    else:
        content_text = str(content)

    obj = _extract_json_object(content_text)
    if not obj:
        return None

    if not bool(obj.get("is_valid", False)):
        return None
    conf = float(obj.get("confidence", 0.0) or 0.0)
    if conf < 0.35:
        return None

    try:
        x0 = int(obj.get("x0", 0))
        y0 = int(obj.get("y0", 0))
        x1 = int(obj.get("x1", 0))
        y1 = int(obj.get("y1", 0))
    except (TypeError, ValueError):
        return None

    if x1 <= x0 or y1 <= y0:
        return None

    sx = orig_w / max(1, vis_w)
    sy = orig_h / max(1, vis_h)
    mapped = [
        max(0, min(orig_w - 1, int(round(x0 * sx)))),
        max(0, min(orig_h - 1, int(round(y0 * sy)))),
        max(1, min(orig_w, int(round(x1 * sx)))),
        max(1, min(orig_h, int(round(y1 * sy)))),
    ]
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        return None
    return mapped


def crop_quality_metrics(image_path: Path) -> Optional[dict[str, float]]:
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            hist = gray.histogram()
            total = float(sum(hist) or 1)
            whiteish = float(sum(hist[245:])) / total
            entropy = float(gray.entropy())
            return {
                "white_ratio": whiteish,
                "entropy": entropy,
                "width": float(img.width),
                "height": float(img.height),
            }
    except OSError:
        return None


def is_crop_plausible(asset_type: str, image_path: Path) -> bool:
    m = crop_quality_metrics(image_path)
    if m is None:
        return False
    width = m["width"]
    height = m["height"]
    if width < 120 or height < 90:
        return False
    if asset_type == "figure":
        if m["white_ratio"] > 0.96 and m["entropy"] < 1.2:
            return False
    return True


def inflate_bbox_px(bbox: list[int], img_w: int, img_h: int, margin_ratio: float) -> list[int]:
    x0, y0, x1, y1 = bbox
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    mx = int(round(w * margin_ratio))
    my = int(round(h * margin_ratio))
    return [
        max(0, x0 - mx),
        max(0, y0 - my),
        min(img_w, x1 + mx),
        min(img_h, y1 + my),
    ]


def union_bbox_px(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def constrain_bbox_by_caption_context(
    crop_bbox: list[int],
    caption_bbox_px: list[int],
    page_img_w: int,
    page_img_h: int,
    asset_type: str,
) -> list[int]:
    x0, y0, x1, y1 = crop_bbox
    cx0, cy0, cx1, cy1 = caption_bbox_px

    # Side-caption layout (caption on right narrow column) -> keep left content band.
    side_caption = cx0 > int(page_img_w * 0.55) and (cx1 - cx0) < int(page_img_w * 0.45)
    if side_caption:
        x1 = min(x1, max(0, cx0 - 8))

    if asset_type == "figure":
        include_legend = os.environ.get("FIGURE_INCLUDE_LEGEND", "1").strip() != "0"
        # Keep figure primarily above caption line; optionally include legend/caption block.
        if not side_caption:
            if include_legend:
                # Include the caption/legend region to preserve interpretation context.
                y1 = max(y1, min(page_img_h, cy1 + 28))
                caption_w = max(1, cx1 - cx0)
                if caption_w < int(page_img_w * 0.75):
                    x0 = max(0, min(x0, cx0 - 20))
                    x1 = min(page_img_w, max(x1, cx1 + 20))
            else:
                y1 = min(y1, max(0, cy0 - 6))
            # Avoid swallowing far-above page body text when caption is near lower area.
            y0 = max(y0, cy0 - int(page_img_h * 0.28))
        y0 = max(0, y0)
    else:
        # Tables often have narrow captions but wide content; do not clamp width to caption span.
        y0 = max(y0, max(0, cy1 + 2))
        x0 = max(0, min(x0, cx0 - int(page_img_w * 0.18)))
        x1 = min(page_img_w, max(x1, cx1 + int(page_img_w * 0.18)))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(page_img_w, x1)
    y1 = min(page_img_h, y1)

    if x1 - x0 < 60 or y1 - y0 < 60:
        return crop_bbox
    return [x0, y0, x1, y1]


def expand_figure_bbox_with_nearby_blocks(
    crop_bbox_px: list[int],
    blocks_on_page: list[dict[str, Any]],
    role_labels: dict[str, str],
    dpi: int,
    scale: float,
    page_img_w: int,
    page_img_h: int,
) -> list[int]:
    x0, y0, x1, y1 = crop_bbox_px
    expanded = [x0, y0, x1, y1]

    def _is_short_label(text: str) -> bool:
        t = " ".join(text.split()).strip()
        if not t:
            return False
        if len(t) > 64:
            return False
        if t.count(".") > 2:
            return False
        words = t.split(" ")
        if len(words) > 10:
            return False
        return True

    horiz_pad = 180
    top_allow = 360
    bottom_allow = 120

    for block in blocks_on_page:
        b_id = str(block.get("block_id", "") or "")
        role = str(role_labels.get(b_id, "") or "")
        if role == "HeaderFooter":
            continue

        b_pt = block.get("bbox_pt", [0, 0, 0, 0])
        if not isinstance(b_pt, list) or len(b_pt) < 4:
            continue
        b_px = pt_to_px_bbox([float(b_pt[0]), float(b_pt[1]), float(b_pt[2]), float(b_pt[3])], dpi, scale)
        bx0, by0, bx1, by1 = b_px
        if bx1 <= bx0 or by1 <= by0:
            continue

        text = str(block.get("text", "") or "")
        if not _is_short_label(text):
            continue

        in_horiz_band = bx1 >= (expanded[0] - horiz_pad) and bx0 <= (expanded[2] + horiz_pad)
        near_top = by1 >= (expanded[1] - top_allow) and by0 <= expanded[1]
        near_inside = by0 <= expanded[3] and by1 >= expanded[1]
        near_bottom = by0 <= (expanded[3] + bottom_allow) and by1 >= expanded[3]

        if in_horiz_band and (near_top or near_inside or near_bottom):
            expanded = union_bbox_px(expanded, b_px)

    expanded = [
        max(0, expanded[0]),
        max(0, expanded[1]),
        min(page_img_w, expanded[2]),
        min(page_img_h, expanded[3]),
    ]
    return expanded


def _moving_avg(values: list[float], window: int) -> list[float]:
    if not values or window <= 1:
        return values[:]
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _pixel_to_gray(pixel: Any) -> int:
    if isinstance(pixel, int):
        return pixel
    if isinstance(pixel, float):
        return max(0, min(255, round(pixel)))
    if isinstance(pixel, tuple) and len(pixel) > 0:
        first = pixel[0]
        if isinstance(first, int):
            return first
        if isinstance(first, float):
            return max(0, min(255, round(first)))
    return 255


def auto_trim_bbox_by_density(
    page_image_path: Path,
    bbox: list[int],
    asset_type: str,
) -> list[int]:
    if asset_type == "figure":
        # Preserve complete in-figure labels; avoid aggressive auto-trim cuts.
        return bbox
    try:
        with Image.open(page_image_path) as img:
            img_w = _pixel_to_gray(getattr(img, "width", 0))
            img_h = _pixel_to_gray(getattr(img, "height", 0))
            if img_w <= 1 or img_h <= 1:
                return bbox
            x0, y0, x1, y1 = bbox
            x0 = max(0, min(img_w - 1, x0))
            y0 = max(0, min(img_h - 1, y0))
            x1 = max(x0 + 1, min(img_w, x1))
            y1 = max(y0 + 1, min(img_h, y1))
            roi = img.crop((x0, y0, x1, y1)).convert("L")
            w, h = roi.size
            if w < 120 or h < 120:
                return [x0, y0, x1, y1]

            step = 2
            row_ratio: list[float] = []
            for yy in range(0, h, step):
                cnt = 0
                total = 0
                for xx in range(0, w, step):
                    total += 1
                    g = _pixel_to_gray(roi.getpixel((xx, yy)))
                    if g < 240:
                        cnt += 1
                row_ratio.append(cnt / max(1, total))

            col_ratio: list[float] = []
            for xx in range(0, w, step):
                cnt = 0
                total = 0
                for yy in range(0, h, step):
                    total += 1
                    g = _pixel_to_gray(roi.getpixel((xx, yy)))
                    if g < 240:
                        cnt += 1
                col_ratio.append(cnt / max(1, total))

            row_s = _moving_avg(row_ratio, 12 if asset_type == "figure" else 8)
            col_s = _moving_avg(col_ratio, 10 if asset_type == "figure" else 6)
            row_thr = 0.055 if asset_type == "figure" else 0.03
            col_thr = 0.022 if asset_type == "figure" else 0.018

            row_idx = [i for i, v in enumerate(row_s) if v >= row_thr]
            col_idx = [i for i, v in enumerate(col_s) if v >= col_thr]
            if not row_idx or not col_idx:
                return [x0, y0, x1, y1]

            top = max(0, row_idx[0] * step - 12)
            bottom = min(h, row_idx[-1] * step + 12)
            left = max(0, col_idx[0] * step - 12)
            right = min(w, col_idx[-1] * step + 12)

            if right - left < 80 or bottom - top < 80:
                return [x0, y0, x1, y1]

            old_area = (x1 - x0) * (y1 - y0)
            new_area = (right - left) * (bottom - top)
            if new_area < old_area * 0.35:
                return [x0, y0, x1, y1]

            return [x0 + left, y0 + top, x0 + right, y0 + bottom]
    except OSError:
        return bbox


def table_block_likelihood_score(text: str) -> float:
    clean = _clean_line(text)
    if not clean:
        return 0.0

    words = re.findall(r"[A-Za-z0-9]+(?:[\-/][A-Za-z0-9]+)?", clean)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    low = clean.lower()
    sentence_like_end = re.search(r"[.!?]\s*$", clean) is not None
    numeric_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", clean))
    semicolons = clean.count(";")

    descriptor_hits = sum(
        1
        for marker in (
            "high",
            "low",
            "poor",
            "limited",
            "excellent",
            "good",
            "mechanical",
            "biocompatibility",
            "conductivity",
            "surface area",
            "resistance",
            "stability",
        )
        if marker in low
    )
    header_hits = sum(
        1
        for marker in (
            "advantages",
            "disadvantages",
            "applications",
            "references",
            "coefficient",
            "type",
        )
        if marker in low
    )

    score = 0.0
    if semicolons >= 1 and word_count <= 80:
        score += 1.6
    if numeric_count >= 1 and descriptor_hits >= 1:
        score += 1.2
    if re.search(r"\bd\d{1,2}\s*=", low) is not None:
        score += 1.2
    if header_hits >= 2 and word_count <= 24 and not sentence_like_end:
        score += 1.0
    if "table" in low and "continued" in low:
        score += 0.8

    if sentence_like_end and semicolons == 0 and word_count >= 24:
        score -= 1.5

    return score


def _is_table_watermark_block(text: str) -> bool:
    low = _clean_line(text).lower()
    if not low:
        return True
    if "downloaded from" in low:
        return True
    if "adv. funct. mater." in low:
        return True
    if re.search(r"\(\d+\s+of\s+\d+\)", low):
        return True
    if "www." in low and any(token in low for token in ("wiley", "afm-journal", "advancedsciencenews")):
        return True
    return False


def _cluster_table_candidate_blocks(
    seed_index: int,
    candidates: list[dict[str, Any]],
    max_vertical_gap: float,
) -> tuple[set[int], list[float]]:
    used: set[int] = {seed_index}
    bx0, by0, bx1, by1 = candidates[seed_index]["bbox"]
    ux0, uy0, ux1, uy1 = bx0, by0, bx1, by1

    changed = True
    while changed:
        changed = False
        for idx, candidate in enumerate(candidates):
            if idx in used:
                continue
            cbx0, cby0, cbx1, cby1 = candidate["bbox"]
            if cby0 > uy1 + max_vertical_gap:
                continue
            if cby1 < uy0 - 12.0:
                continue
            if cbx1 < ux0 - 120.0 or cbx0 > ux1 + 120.0:
                continue
            used.add(idx)
            ux0 = min(ux0, cbx0)
            uy0 = min(uy0, cby0)
            ux1 = max(ux1, cbx1)
            uy1 = max(uy1, cby1)
            changed = True

    return used, [ux0, uy0, ux1, uy1]


def propose_table_bbox_from_blocks(
    caption_para: dict[str, Any],
    blocks_on_page: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> Optional[list[float]]:
    if not caption_para or not blocks_on_page:
        return None

    caption_bbox = caption_para.get("evidence_pointer", {}).get("bbox_union", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(caption_bbox, list) or len(caption_bbox) < 4:
        return None

    try:
        cx0 = float(caption_bbox[0])
        cy0 = float(caption_bbox[1])
        cx1 = float(caption_bbox[2])
        cy1 = float(caption_bbox[3])
    except (TypeError, ValueError):
        return None

    candidates: list[dict[str, Any]] = []
    for block in blocks_on_page:
        bbox_pt = block.get("bbox_pt", [0, 0, 0, 0])
        if not isinstance(bbox_pt, list) or len(bbox_pt) < 4:
            continue
        try:
            bx0 = float(bbox_pt[0])
            by0 = float(bbox_pt[1])
            bx1 = float(bbox_pt[2])
            by1 = float(bbox_pt[3])
        except (TypeError, ValueError):
            continue
        if bx1 <= bx0 or by1 <= by0:
            continue

        bw = bx1 - bx0
        bh = by1 - by0
        if bw < max(12.0, page_width * 0.03) and bh > page_height * 0.4:
            continue
        if bx0 > page_width * 0.93 and bw < page_width * 0.06:
            continue
        if by1 <= cy0 - 12:
            continue
        if by0 >= page_height * 0.97:
            continue

        block_text = _clean_line(str(block.get("text", "") or ""))
        if _is_table_watermark_block(block_text):
            continue

        low_text = block_text.lower()
        score = table_block_likelihood_score(block_text)
        word_count = len(re.findall(r"[A-Za-z0-9]+(?:[\-/][A-Za-z0-9]+)?", block_text))
        sentence_like_end = re.search(r"[.!?]\s*$", block_text) is not None
        numeric_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", block_text))
        descriptor_hits = sum(
            1
            for marker in (
                "high",
                "low",
                "poor",
                "limited",
                "excellent",
                "good",
                "mechanical",
                "surface area",
                "conductivity",
            )
            if marker in low_text
        )
        row_pattern = (
            (";" in block_text and word_count <= 64)
            or re.search(r"\bd\d{1,2}\s*=", low_text) is not None
            or (any(marker in low_text for marker in ("advantages", "disadvantages", "applications", "references", "type", "coefficient")) and word_count <= 24 and not sentence_like_end)
            or (numeric_count >= 1 and descriptor_hits >= 2 and word_count <= 28 and not sentence_like_end)
        )

        if score >= 1.0 and (row_pattern or score >= 1.6):
            candidates.append(
                {
                    "bbox": [bx0, by0, bx1, by1],
                    "text": block_text,
                    "score": score,
                }
            )

    if len(candidates) < 2:
        return None

    seed_upper_bound = cy1 + max(120.0, page_height * 0.24)
    seed_indices = [
        idx
        for idx, candidate in enumerate(candidates)
        if candidate["bbox"][0] < page_width * 0.94
        and candidate["bbox"][1] <= seed_upper_bound
        and candidate["bbox"][3] >= cy1 - 4.0
    ]
    if not seed_indices:
        seed_indices = [
            idx
            for idx, candidate in enumerate(candidates)
            if candidate["bbox"][0] < page_width * 0.94 and candidate["bbox"][3] >= cy1 - 4.0
        ]
    if not seed_indices:
        return None

    seed_index = min(
        seed_indices,
        key=lambda idx: (abs(candidates[idx]["bbox"][1] - cy1), -candidates[idx]["score"]),
    )

    used, union_bbox = _cluster_table_candidate_blocks(seed_index, candidates, max_vertical_gap=26.0)
    ux0, uy0, ux1, uy1 = union_bbox

    # Continue downward using left-anchored rows to recover split table bodies
    # while avoiding right-column narrative blocks.
    expanded = True
    while expanded:
        expanded = False
        for idx, candidate in sorted(enumerate(candidates), key=lambda item: item[1]["bbox"][1]):
            if idx in used:
                continue
            bx0, by0, bx1, by1 = candidate["bbox"]
            if by0 < uy0 - 12.0:
                continue
            gap = by0 - uy1
            if gap < 0.0 or gap > 96.0:
                continue
            if bx0 > ux0 + page_width * 0.20:
                continue
            if bx1 < ux0 + page_width * 0.25:
                continue
            used.add(idx)
            ux0 = min(ux0, bx0)
            uy0 = min(uy0, by0)
            ux1 = max(ux1, bx1)
            uy1 = max(uy1, by1)
            expanded = True

    cluster_candidates = [candidates[idx] for idx in sorted(used)]

    if len(cluster_candidates) < 2:
        return None
    if (ux1 - ux0) < page_width * 0.35 and len(cluster_candidates) < 4:
        return None
    if min(candidate["bbox"][1] for candidate in cluster_candidates) > cy1 + max(90.0, page_height * 0.15):
        return None

    for block in blocks_on_page:
        bbox_pt = block.get("bbox_pt", [0, 0, 0, 0])
        if not isinstance(bbox_pt, list) or len(bbox_pt) < 4:
            continue
        try:
            bx0 = float(bbox_pt[0])
            by0 = float(bbox_pt[1])
            bx1 = float(bbox_pt[2])
            by1 = float(bbox_pt[3])
        except (TypeError, ValueError):
            continue
        if bx1 <= bx0 or by1 <= by0:
            continue
        if by0 > uy1 + 14.0 or by1 < uy0 - 10.0:
            continue
        if bx1 < ux0 - 80.0 or bx0 > ux1 + 80.0:
            continue

        block_text = _clean_line(str(block.get("text", "") or ""))
        if _is_table_watermark_block(block_text):
            continue

        low_text = block_text.lower()
        score = table_block_likelihood_score(block_text)
        word_count = len(re.findall(r"[A-Za-z0-9]+(?:[\-/][A-Za-z0-9]+)?", block_text))
        sentence_like_end = re.search(r"[.!?]\s*$", block_text) is not None
        numeric_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", block_text))
        descriptor_hits = sum(
            1
            for marker in ("high", "low", "poor", "limited", "excellent", "good", "mechanical", "surface area", "conductivity")
            if marker in low_text
        )
        row_pattern = (
            (";" in block_text and word_count <= 64)
            or re.search(r"\bd\d{1,2}\s*=", low_text) is not None
            or (any(marker in low_text for marker in ("advantages", "disadvantages", "applications", "references", "type", "coefficient")) and word_count <= 24 and not sentence_like_end)
            or (numeric_count >= 1 and descriptor_hits >= 2 and word_count <= 28 and not sentence_like_end)
        )
        if score >= 0.85 and row_pattern:
            ux0 = min(ux0, bx0)
            uy0 = min(uy0, by0)
            ux1 = max(ux1, bx1)
            uy1 = max(uy1, by1)

    x0 = max(0.0, min(ux0 - 80.0, cx0 - 24.0))
    y0 = max(cy1 + 1.0, uy0 - 6.0)
    x1 = min(page_width, ux1 + 12.0)
    y1 = min(page_height, uy1 + 10.0)

    if x1 - x0 < 120.0 or y1 - y0 < 80.0:
        return None
    return [x0, y0, x1, y1]

def is_figure_caption(text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Check if text is a figure caption. Returns (is_caption, figure_num, remainder)."""
    normalized = " ".join(str(text).split()).strip()
    match = FIG_CAPTION_RE.match(normalized)
    if match:
        fig_num = match.group(1)
        remainder = match.group(2).strip() if match.group(2) else ""
        return True, fig_num, remainder
    inline_match = FIG_CAPTION_INLINE_RE.search(normalized)
    if inline_match:
        fig_num = inline_match.group(1)
        remainder = inline_match.group(2).strip() if inline_match.group(2) else ""
        return True, fig_num, remainder
    return False, None, None


def is_table_caption(text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Check if text is a table caption. Returns (is_caption, table_num, remainder)."""
    normalized = " ".join(str(text).split()).strip()
    match = TABLE_CAPTION_RE.match(normalized)
    if match:
        table_num = match.group(1)
        remainder = match.group(2).strip() if match.group(2) else ""
        return True, table_num, remainder
    inline_match = TABLE_CAPTION_INLINE_RE.search(normalized)
    if inline_match:
        table_num = inline_match.group(1)
        remainder = inline_match.group(2).strip() if inline_match.group(2) else ""
        return True, table_num, remainder
    return False, None, None


def _normalize_caption_remainder(remainder: str) -> str:
    clean = " ".join(str(remainder).split()).strip()
    if not clean:
        return ""
    clean = re.sub(r"^[\|:\-–—\.;,\s]+", "", clean)
    clean = re.sub(r"\s*\|\s*\|+\s*", " | ", clean)
    clean = clean.strip(" |")
    return clean


def canonicalize_caption_text(text: Optional[str], asset_type: str) -> Optional[str]:
    raw = " ".join(str(text or "").split()).strip()
    if not raw:
        return None
    if asset_type == "figure":
        is_cap, num, remainder = is_figure_caption(raw)
        if is_cap:
            remainder_clean = _normalize_caption_remainder(remainder or "")
            if remainder_clean:
                return f"Fig. {num} | {remainder_clean}"
            return f"Fig. {num}"
    if asset_type == "table":
        is_cap, num, remainder = is_table_caption(raw)
        if is_cap:
            remainder_clean = _normalize_caption_remainder(remainder or "")
            if remainder_clean:
                return f"Table {num} | {remainder_clean}"
            return f"Table {num}"
    return raw

def caption_num_to_index(num_text: Optional[str]) -> Optional[int]:
    if not num_text:
        return None
    m = re.match(r"^(\d+)", str(num_text).strip())
    if not m:
        return None
    try:
        value = int(m.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def find_caption_paragraphs(paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find paragraphs that are likely captions."""
    captions = []
    for para in paragraphs:
        text = para.get("text", "")

        # Check for Figure/Table captions
        is_fig, _, _ = is_figure_caption(text)
        is_tbl, _, _ = is_table_caption(text)

        if is_fig or is_tbl:
            captions.append(para)
            continue

        # Also check if role is explicitly set
        role = para.get("role", "")
        if role in (ROLE_FIGURE_CAPTION, ROLE_TABLE_CAPTION):
            captions.append(para)

    return captions


def _asset_preference_key(asset: FigureTableAsset) -> tuple[int, int, int, float]:
    return (
        1 if asset.caption_id else 0,
        1 if asset.caption_text else 0,
        1 if asset.image_path else 0,
        float(asset.confidence),
    )


def deduplicate_assets(assets: list[FigureTableAsset]) -> list[FigureTableAsset]:
    """Remove duplicate assets while preferring entries with stronger caption linkage."""
    if not assets:
        return []

    by_asset_id: dict[str, FigureTableAsset] = {}
    ordered_ids: list[str] = []
    for asset in assets:
        existing = by_asset_id.get(asset.asset_id)
        if existing is None:
            by_asset_id[asset.asset_id] = asset
            ordered_ids.append(asset.asset_id)
            continue
        if _asset_preference_key(asset) >= _asset_preference_key(existing):
            by_asset_id[asset.asset_id] = asset

    collapsed = [by_asset_id[asset_id] for asset_id in ordered_ids if asset_id in by_asset_id]

    deduped: list[FigureTableAsset] = []
    by_semantic_key: dict[tuple[str, int, str], FigureTableAsset] = {}
    semantic_order: list[tuple[str, int, str]] = []
    for asset in collapsed:
        caption_norm = (
            str(asset.caption_id).strip().lower()
            if asset.caption_id
            else (asset.caption_text or "")[:160].strip().lower()
        )
        key = (asset.asset_type, int(asset.page), caption_norm)
        existing = by_semantic_key.get(key)
        if existing is None:
            by_semantic_key[key] = asset
            semantic_order.append(key)
            continue
        if _asset_preference_key(asset) > _asset_preference_key(existing):
            by_semantic_key[key] = asset

    for key in semantic_order:
        kept = by_semantic_key.get(key)
        if kept is not None:
            deduped.append(kept)
    return deduped


def _clean_line(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _is_table_caption_summary_noise(text: str) -> bool:
    low = _clean_line(text).lower()
    if not low:
        return True
    if re.fullmatch(r"table\s+\d+[a-z]?\s*(?:\|\s*)*(?:\(?continued\)?\.?\s*)?", low):
        return True
    token_count = len(re.findall(r"[a-z]+", low))
    if re.match(r"^table\s+\d+[a-z]?\b", low) and "continued" in low and token_count <= 6:
        return True
    return False


def _table_caption_summaries(assets: list[FigureTableAsset], max_items: int = 2) -> list[str]:
    captions: list[str] = []
    seen: set[str] = set()
    for asset in assets:
        if asset.asset_type != "table":
            continue
        caption = canonicalize_caption_text(asset.caption_text, "table")
        if not caption:
            caption = asset.asset_id.replace("_", " ").upper()
        normalized = _clean_line(caption)
        if not normalized:
            continue
        if _is_table_caption_summary_noise(normalized):
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        captions.append(normalized)
        if len(captions) >= max_items:
            break
    return captions

def _build_table_placeholder_text(assets: list[FigureTableAsset]) -> str:
    summaries = _table_caption_summaries(assets)
    if summaries:
        joined = "; ".join(summaries)
        return (
            "[TABLE_PLACEHOLDER] Table-like content omitted from Main Body. "
            f"See processed tables: {joined} (## Figures and Tables)."
        )
    return (
        "[TABLE_PLACEHOLDER] Table-like content omitted from Main Body. "
        "See processed table assets in ## Figures and Tables."
    )


def _is_table_artifact_paragraph(text: str) -> bool:
    clean = _clean_line(text)
    if not clean:
        return False
    words = re.findall(r"[A-Za-z0-9]+(?:[''’\-/][A-Za-z0-9]+)?", clean)
    word_count = len(words)
    if word_count == 0:
        return False

    if re.fullmatch(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", clean):
        return True

    semicolon_heavy = (
        clean.count(";") >= 1
        and word_count <= 34
        and re.search(r"[.!?]$", clean) is None
    )

    short_item = (
        word_count <= 6
        and re.match(
            r"^[A-Z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,5}(?:\s*\([^)]+\))?$",
            clean,
        )
        is not None
    )

    dense_numeric = (
        len(re.findall(r"\b\d+(?:\.\d+)?\b", clean)) >= 3
        and word_count <= 24
    )

    headerish = (
        word_count <= 10
        and sum(1 for token in words if token[:1].isupper()) >= max(2, int(word_count * 0.6))
        and "." not in clean
        and ";" not in clean
    )

    citation_fragment = (
        "[" in clean
        and "]" in clean
        and word_count <= 18
        and re.search(r"[.!?]$", clean) is None
    )

    return semicolon_heavy or short_item or dense_numeric or headerish or citation_fragment


def _looks_like_mixed_table_prefix(text: str) -> bool:
    clean = _clean_line(text)
    if not clean:
        return False
    words = re.findall(r"[A-Za-z0-9]+(?:[''’\-/][A-Za-z0-9]+)?", clean)
    word_count = len(words)
    if word_count < 8:
        return False

    low = clean.lower()
    descriptor_hits = sum(
        1
        for marker in (
            "high",
            "low",
            "poor",
            "limited",
            "excellent",
            "good",
            "mechanical",
            "surface area",
            "biodegradability",
            "conductivity",
            "properties",
            "applications",
            "references",
        )
        if marker in low
    )
    has_number = re.search(r"\b\d+(?:\.\d+)?\b", clean) is not None
    has_table_token = (
        re.search(r"\bd\d{1,2}\s*=", low) is not None
        or re.search(r"\b\d+(?:\.\d+)?\s*[×x]\s*10\b", clean) is not None
        or any(
            token in low
            for token in (
                "aunps",
                "agnps",
                "mxene",
                "pvdf",
                "phb",
                "piezoelectric biomaterials type",
            )
        )
    )
    return descriptor_hits >= 2 and has_number and has_table_token and ";" in clean



def _is_table_row_fragment_near_placeholder(text: str) -> bool:
    clean = _clean_line(text)
    if not clean:
        return False
    if re.search(r"[.!?]\s*$", clean) is not None:
        return False

    words = re.findall(r"[A-Za-z0-9]+(?:[\-/][A-Za-z0-9]+)?", clean)
    word_count = len(words)
    if word_count < 8 or word_count > 140:
        return False

    low = clean.lower()
    descriptor_hits = sum(
        1
        for marker in (
            "high",
            "low",
            "poor",
            "limited",
            "excellent",
            "good",
            "mechanical",
            "surface area",
            "biodegradability",
            "conductivity",
            "properties",
            "applications",
            "references",
            "resistance",
            "stability",
        )
        if marker in low
    )
    if descriptor_hits < 2:
        return False

    has_number = re.search(r"\b\d+(?:\.\d+)?\b", clean) is not None
    if not has_number:
        return False

    has_table_token = (
        re.search(r"\bd\d{1,2}\s*=", low) is not None
        or re.search(r"\b\d+(?:\.\d+)?\s*[×x]\s*10\b", clean) is not None
        or any(token in low for token in ("aunps", "agnps", "mxene", "pvdf", "phb"))
    )
    uppercase_short = sum(1 for token in words if token[:1].isupper() and len(token) <= 12)
    return has_table_token or clean.count(";") >= 1 or uppercase_short >= 3

def _trim_mixed_table_prefix(paragraph_text: str) -> str:
    clean = _clean_line(paragraph_text)
    if not _looks_like_mixed_table_prefix(clean):
        return clean

    low = clean.lower()
    marker = re.search(
        r"\b(notably|with the ongoing|following the successful|however|in addition|in summary|overall|studies have|research has|it has been|these findings)\b",
        low,
    )
    if marker is None or marker.start() < 30:
        return clean

    trimmed = clean[marker.start() :].lstrip(" ,;:-")
    if len(trimmed) < 30:
        return clean
    return trimmed


def _trim_mixed_table_prefixes_in_main_body(clean_document: str) -> str:
    lines = clean_document.splitlines()
    main_start = -1
    main_end = len(lines)
    for idx, line in enumerate(lines):
        if line.strip() == "## Main Body":
            main_start = idx
            continue
        if main_start >= 0 and idx > main_start and line.startswith("## "):
            main_end = idx
            break
    if main_start < 0:
        return clean_document

    paragraphs: list[tuple[int, int, str, bool]] = []
    cursor = main_start + 1
    while cursor < main_end:
        if not lines[cursor].strip():
            cursor += 1
            continue
        if lines[cursor].lstrip().startswith("### "):
            paragraphs.append((cursor, cursor, lines[cursor].strip(), True))
            cursor += 1
            continue

        start_line = cursor
        chunk_lines: list[str] = []
        while cursor < main_end and lines[cursor].strip() and not lines[cursor].lstrip().startswith("### "):
            chunk_lines.append(lines[cursor].strip())
            cursor += 1
        end_line = cursor - 1
        paragraph_text = _clean_line(" ".join(chunk_lines))
        if paragraph_text:
            paragraphs.append((start_line, end_line, paragraph_text, False))

    updates: list[tuple[int, int, str]] = []
    for idx, (start_line, end_line, paragraph_text, is_heading) in enumerate(paragraphs):
        if is_heading:
            continue
        if "[TABLE_PLACEHOLDER]" in paragraph_text:
            continue

        near_placeholder = False
        for back in (1, 2):
            prev_idx = idx - back
            if prev_idx < 0:
                break
            if "[TABLE_PLACEHOLDER]" in paragraphs[prev_idx][2]:
                near_placeholder = True
                break
        if not near_placeholder:
            continue

        trimmed = _trim_mixed_table_prefix(paragraph_text)
        if trimmed != paragraph_text:
            updates.append((start_line, end_line, trimmed))
            continue

        if _is_table_row_fragment_near_placeholder(paragraph_text):
            updates.append((start_line, end_line, ""))

    if not updates:
        return clean_document

    for start_line, end_line, trimmed in sorted(updates, key=lambda item: item[0], reverse=True):
        if trimmed:
            lines[start_line : end_line + 1] = [trimmed]
        else:
            lines[start_line : end_line + 1] = []

    return "\n".join(lines).rstrip() + "\n"

def suppress_table_artifacts_in_clean_document(
    clean_document: str,
    assets: list[FigureTableAsset],
) -> tuple[str, int]:
    if not clean_document:
        return clean_document, 0
    table_assets = [asset for asset in assets if asset.asset_type == "table"]
    if not table_assets:
        return clean_document, 0

    lines = clean_document.splitlines()
    main_start = -1
    main_end = len(lines)
    for idx, line in enumerate(lines):
        if line.strip() == "## Main Body":
            main_start = idx
            continue
        if main_start >= 0 and idx > main_start and line.startswith("## "):
            main_end = idx
            break
    if main_start < 0:
        return clean_document, 0

    def _word_count(text: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]+(?:[''’\-/][A-Za-z0-9]+)?", _clean_line(text)))

    def _has_table_signal(text: str) -> bool:
        clean = _clean_line(text)
        return (";" in clean) or (re.search(r"\b\d+(?:\.\d+)?\b", clean) is not None)

    def _is_bridge_paragraph(text: str) -> bool:
        clean = _clean_line(text)
        if not clean or re.search(r"[.!?]$", clean) is not None:
            return False
        word_count = _word_count(clean)
        if word_count <= 6:
            return True
        return ("[" in clean and "]" in clean and word_count <= 12)

    paragraphs: list[tuple[int, int, str]] = []
    cursor = main_start + 1
    while cursor < main_end:
        if not lines[cursor].strip():
            cursor += 1
            continue
        if lines[cursor].lstrip().startswith("### "):
            cursor += 1
            continue
        start_line = cursor
        chunk_lines: list[str] = []
        while cursor < main_end and lines[cursor].strip() and not lines[cursor].lstrip().startswith("### "):
            chunk_lines.append(lines[cursor].strip())
            cursor += 1
        end_line = cursor - 1
        paragraph_text = _clean_line(" ".join(chunk_lines))
        if paragraph_text:
            paragraphs.append((start_line, end_line, paragraph_text))

    artifact_indices = [
        idx for idx, (_, _, paragraph_text) in enumerate(paragraphs)
        if _is_table_artifact_paragraph(paragraph_text)
    ]
    if not artifact_indices:
        return clean_document, 0

    runs: list[list[int]] = []
    current_run: list[int] = []
    for idx in artifact_indices:
        if not current_run or idx == current_run[-1] + 1:
            current_run.append(idx)
            continue
        runs.append(current_run)
        current_run = [idx]
    if current_run:
        runs.append(current_run)

    primary_run_indices: list[int] = []
    for run_idx, run in enumerate(runs):
        has_table_signals = any(_has_table_signal(paragraphs[run_item_idx][2]) for run_item_idx in run)
        if not has_table_signals:
            continue
        if len(run) >= 2:
            primary_run_indices.append(run_idx)
            continue
        if _word_count(paragraphs[run[0]][2]) <= 14:
            primary_run_indices.append(run_idx)

    if not primary_run_indices:
        return clean_document, 0

    selected_run_indices = set(primary_run_indices)
    for run_idx in primary_run_indices:
        prev_idx = run_idx - 1
        while prev_idx >= 0:
            gap = runs[prev_idx + 1][0] - runs[prev_idx][-1] - 1
            if gap > 1 or len(runs[prev_idx]) < 2:
                break
            selected_run_indices.add(prev_idx)
            prev_idx -= 1

        next_idx = run_idx + 1
        while next_idx < len(runs):
            gap = runs[next_idx][0] - runs[next_idx - 1][-1] - 1
            if gap > 1 or len(runs[next_idx]) < 2:
                break
            selected_run_indices.add(next_idx)
            next_idx += 1

    selected_spans = sorted((runs[idx][0], runs[idx][-1]) for idx in selected_run_indices)

    merged_spans: list[tuple[int, int]] = []
    for span_start, span_end in selected_spans:
        if not merged_spans:
            merged_spans.append((span_start, span_end))
            continue
        prev_start, prev_end = merged_spans[-1]
        gap = span_start - prev_end - 1
        if gap <= 0:
            merged_spans[-1] = (prev_start, max(prev_end, span_end))
            continue
        if gap <= 2:
            bridge_ok = all(
                _is_bridge_paragraph(paragraphs[bridge_idx][2])
                for bridge_idx in range(prev_end + 1, span_start)
            )
            if bridge_ok:
                merged_spans[-1] = (prev_start, span_end)
                continue
        merged_spans.append((span_start, span_end))

    if not merged_spans:
        return clean_document, 0

    max_regions = max(1, len(table_assets))
    if len(merged_spans) > max_regions:
        capped_spans = merged_spans[:]
        while len(capped_spans) > max_regions:
            best_gap = 10**9
            best_idx = -1
            for idx in range(len(capped_spans) - 1):
                gap = capped_spans[idx + 1][0] - capped_spans[idx][1] - 1
                if gap < best_gap:
                    best_gap = gap
                    best_idx = idx
            if best_idx < 0 or best_gap > 6:
                break
            left_start, left_end = capped_spans[best_idx]
            right_start, right_end = capped_spans[best_idx + 1]
            capped_spans[best_idx:best_idx + 2] = [(left_start, right_end)]

        if len(capped_spans) > max_regions:
            ranked = sorted(
                (
                    span_end - span_start + 1,
                    idx,
                    (span_start, span_end),
                )
                for idx, (span_start, span_end) in enumerate(capped_spans)
            )
            keep_indices = {idx for _, idx, _ in ranked[-max_regions:]}
            capped_spans = [span for idx, span in enumerate(capped_spans) if idx in keep_indices]
            capped_spans.sort(key=lambda span: span[0])

        merged_spans = capped_spans

    placeholder = _build_table_placeholder_text(table_assets)
    rewritten: list[str] = []
    line_cursor = 0
    for run_start_idx, run_end_idx in merged_spans:
        start_line = paragraphs[run_start_idx][0]
        end_line = paragraphs[run_end_idx][1]
        rewritten.extend(lines[line_cursor:start_line])

        residual_paragraphs: list[str] = []
        for para_idx in range(run_start_idx, run_end_idx + 1):
            paragraph_text = paragraphs[para_idx][2]
            trimmed = _trim_mixed_table_prefix(paragraph_text)
            if trimmed != paragraph_text:
                residual_paragraphs.append(trimmed)

        if rewritten and rewritten[-1].strip():
            rewritten.append("")
        rewritten.append(placeholder)
        rewritten.append("")
        for residual in residual_paragraphs:
            rewritten.append(residual)
            rewritten.append("")

        line_cursor = end_line + 1

    rewritten.extend(lines[line_cursor:])
    new_doc = "\n".join(rewritten).rstrip() + "\n"
    new_doc = _trim_mixed_table_prefixes_in_main_body(new_doc)
    return new_doc, len(merged_spans)


def rewrite_clean_document_after_asset_detection(run_dir: Path, assets: list[FigureTableAsset]) -> int:
    clean_path = run_dir / "text" / "clean_document.md"
    if not clean_path.exists():
        return 0
    try:
        original = clean_path.read_text(encoding="utf-8")
    except OSError:
        return 0

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(original, assets)
    if replaced_regions <= 0 or rewritten == original:
        return 0

    try:
        clean_path.write_text(rewritten, encoding="utf-8")
    except OSError:
        return 0
    return replaced_regions

def extract_images_from_pdf(pdf_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Extract image metadata from PDF pages.
    
    Returns dict mapping page_num (1-based) to list of image info dicts.
    Each image info contains: xref, bbox (in points), width, height.
    """
    images_by_page: dict[int, list[dict[str, Any]]] = {}
    
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return images_by_page
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_num_1based = page_num + 1
        
        # Get images from page
        img_list = page.get_images()
        
        page_images = []
        for img_index, img in enumerate(img_list):
            xref = img[0]
            
            # Get image rectangle
            try:
                img_rect = page.get_image_rects(xref)
                if img_rect:
                    bbox = [img_rect[0].x0, img_rect[0].y0, img_rect[0].x1, img_rect[0].y1]
                else:
                    bbox = [0, 0, 0, 0]
            except Exception:
                bbox = [0, 0, 0, 0]
            
            # Try to get dimensions from the image
            try:
                base_image = doc.extract_image(xref)
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
            except Exception:
                width, height = 0, 0
            
            page_images.append({
                "xref": xref,
                "index": img_index,
                "bbox": bbox,
                "width": width,
                "height": height,
            })
        
        if page_images:
            images_by_page[page_num_1based] = page_images
    
    doc.close()
    return images_by_page


def crop_image_from_page(
    page_image_path: Path,
    bbox_px: list[int],
    output_path: Path,
    padding: int = 10,
) -> bool:
    """Crop a region from a page image and save it."""
    try:
        with Image.open(page_image_path) as img:
            # Add padding but stay within image bounds
            x0 = max(0, bbox_px[0] - padding)
            y0 = max(0, bbox_px[1] - padding)
            x1 = min(img.width, bbox_px[2] + padding)
            y1 = min(img.height, bbox_px[3] + padding)
            
            # Ensure valid crop region
            if x1 <= x0 or y1 <= y0:
                return False
            
            cropped = img.crop((x0, y0, x1, y1))
            cropped.save(output_path, format="PNG")
            return True
    except Exception:
        return False


def append_legend_to_image(image_path: Path, legend_text: str) -> bool:
    text = " ".join((legend_text or "").split()).strip()
    if not text:
        return False
    try:
        with Image.open(image_path) as img:
            base = img.convert("RGB")
            w, h = base.size
            if w < 120 or h < 120:
                return False

            font = ImageFont.load_default()
            max_chars = max(60, min(180, w // 8))
            words = text.split(" ")
            lines: list[str] = []
            cur = ""
            for word in words:
                nxt = (cur + " " + word).strip()
                if len(nxt) > max_chars:
                    if cur:
                        lines.append(cur)
                    cur = word
                else:
                    cur = nxt
                if len(lines) >= 4:
                    break
            if cur and len(lines) < 4:
                lines.append(cur)
            if len(words) > 0 and len(lines) >= 4:
                lines[-1] = (lines[-1][: max(0, max_chars - 3)] + "...")

            strip_h = 16 * max(1, len(lines)) + 18
            out = Image.new("RGB", (w, h + strip_h), "white")
            out.paste(base, (0, 0))

            draw = ImageDraw.Draw(out)
            y = h + 8
            for line in lines:
                draw.text((8, y), line, fill=(0, 0, 0), font=font)
                y += 16

            out.save(image_path, format="PNG")
            return True
    except OSError:
        return False


def estimate_bbox_from_caption(
    caption_para: dict[str, Any],
    other_paras: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    is_figure: bool,
    prefer_above: bool = True,
) -> Optional[list[float]]:
    """Estimate the bounding box of a figure/table based on caption position.
    
    For figures: typically the figure is ABOVE the caption
    For tables: typically the table is ABOVE or BELOW the caption
    """
    caption_bbox = caption_para.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])
    if not caption_bbox or caption_bbox == [0, 0, 0, 0]:
        return None
    
    caption_x0, caption_y0, caption_x1, caption_y1 = [float(x) for x in caption_bbox[:4]]
    caption_width = max(1.0, caption_x1 - caption_x0)
    # Use near full-page span for synthetic crops to avoid missing wide multipanel assets.
    x0 = max(0.0, page_width * 0.02)
    x1 = min(page_width, page_width * 0.98)
    est_width = max(1.0, x1 - x0)

    if is_figure:
        # Side-caption layout: caption on right column, figure/algorithm on left side.
        if caption_x0 > page_width * 0.55 and caption_width < page_width * 0.45:
            sx0 = max(0.0, page_width * 0.03)
            sx1 = min(page_width, caption_x0 - page_width * 0.04)
            sy0 = max(0.0, caption_y0 - page_height * 0.08)
            sy1 = min(page_height, caption_y1 + page_height * 0.28)
            if sx1 - sx0 >= 60.0 and sy1 - sy0 >= 60.0:
                return [sx0, sy0, sx1, sy1]

        # Figures are usually above caption and often span both columns.
        est_height = min(page_height * 0.55, max(180.0, est_width * 0.32))
        if prefer_above:
            y1 = max(0.0, caption_y0 - 6.0)
            y0 = max(0.0, y1 - est_height)
        else:
            y0 = min(page_height - 5.0, caption_y1 + 6.0)
            y1 = min(page_height, y0 + est_height)
        if y1 - y0 < 40.0:
            return None
        return [x0, y0, x1, y1]

    # Tables are usually below caption line in review papers.
    est_height = min(page_height * 0.52, max(160.0, est_width * 0.28))
    if prefer_above:
        y0 = min(page_height - 5.0, caption_y1 + 6.0)
        y1 = min(page_height, y0 + est_height)
    else:
        y1 = max(0.0, caption_y0 - 6.0)
        y0 = max(0.0, y1 - est_height)
    if y1 - y0 < 40.0:
        # fallback to above-caption region if below region is too small
        y1 = max(0.0, caption_y0 - 6.0)
        y0 = max(0.0, y1 - est_height)
    if y1 - y0 < 40.0:
        return None
    return [x0, y0, x1, y1]


def load_render_config(run_dir: Path) -> dict[str, Any]:
    """Load render config from manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {"dpi": 150, "scale": 2.0}
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    return manifest.get("render_config", {"dpi": 150, "scale": 2.0})


def load_verified_figure_table_candidates(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load verified figure/table candidates from layout analysis.
    
    Returns a dict mapping block_id to verification result.
    Only returns candidates that were verified as actual figures or tables.
    """
    layout_path = run_dir / "text" / "layout_analysis.json"
    if not layout_path.exists():
        return {}
    
    verification = run_figure_table_verification(run_dir, layout_path)
    if not verification.get("verified", False):
        return {}
    
    results = verification.get("results", [])
    verified_by_block_id: dict[str, dict[str, Any]] = {}
    
    for r in results:
        verified_type = r.get("verified_type", "")
        if verified_type in ("figure", "table"):
            block_id = r.get("block_id", "")
            if block_id:
                verified_by_block_id[block_id] = {
                    "type": verified_type,
                    "confidence": r.get("confidence", 0.5),
                    "caption_match": r.get("caption_match", False),
                    "visual_confirmed": r.get("visual_confirmed", False),
                }
    
    return verified_by_block_id


from typing import Any, Optional, Sequence


def pt_to_px_bbox(bbox_pt: Sequence[float | int], dpi: int, scale: float) -> list[int]:
    """Convert PDF point bbox to pixel bbox."""
    zoom = dpi / 72.0 * scale
    return [int(float(c) * zoom) for c in bbox_pt]


def find_related_caption_for_image(
    image_bbox: list[float],
    caption_paras: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Find the caption paragraph most likely associated with an image."""
    if not caption_paras:
        return None
    
    image_y1 = image_bbox[3]  # Bottom of image
    
    best_caption = None
    best_distance = float('inf')
    
    image_x0, _, image_x1, _ = image_bbox
    image_width = max(1.0, image_x1 - image_x0)

    for caption in caption_paras:
        caption_bbox = caption.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])
        if not caption_bbox:
            continue
        
        caption_y0 = caption_bbox[1]  # Top of caption
        
        # For figures: caption is usually below the image
        # Distance = caption_y0 - image_y1 (should be small positive)
        distance = caption_y0 - image_y1
        
        caption_x0, _, caption_x1, _ = caption_bbox
        overlap = max(0.0, min(image_x1, caption_x1) - max(image_x0, caption_x0))
        overlap_ratio = overlap / image_width

        # Accept if caption is slightly below or above image and roughly aligned in x-axis.
        if -50 <= distance < 220 and overlap_ratio >= 0.15 and abs(distance) < best_distance:
            best_distance = abs(distance)
            best_caption = caption
    
    return best_caption


def bbox_intersection_area(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    return inter_w * inter_h


def bbox_area(a: list[float]) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def is_probable_header_footer_artifact(
    image_bbox: list[float],
    header_footer_bboxes: list[list[float]],
) -> bool:
    if not header_footer_bboxes:
        return False
    area = bbox_area(image_bbox)
    if area <= 0:
        return False
    overlap = 0.0
    for hf in header_footer_bboxes:
        overlap += bbox_intersection_area(image_bbox, hf)
    return (overlap / area) >= 0.55


def is_likely_body_asset(
    bbox_pt: list[float],
    page_width: float,
    page_height: float,
    caption_para: Optional[dict[str, Any]],
) -> bool:
    x0, y0, x1, y1 = bbox_pt
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    area = width * height
    page_area = max(1.0, page_width * page_height)
    area_ratio = area / page_area

    if caption_para is not None:
        if width < 35 or height < 25:
            return False
        if area_ratio < 0.0006:
            return False
    else:
        if width < 70 or height < 55:
            return False
        if area_ratio < 0.004:
            return False

    center_y_ratio = ((y0 + y1) * 0.5) / max(1.0, page_height)
    in_margin = center_y_ratio < 0.1 or center_y_ratio > 0.92

    # Allow margin assets only if a likely caption is present on the same page.
    if in_margin and caption_para is None:
        return False

    return True


def run_figures_tables(
    run_dir: Path,
    manifest: Optional[Manifest] = None,
) -> tuple[int, int]:
    """Run figure/table extraction pipeline.
    
    Returns (assets_created, images_cropped).
    """
    # Load manifest
    if manifest is None:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        else:
            return 0, 0
    else:
        manifest_data = manifest.model_dump() if hasattr(manifest, 'model_dump') else {}
    
    pdf_path = Path(manifest_data.get("input_pdf_path", ""))
    if not pdf_path.exists():
        return 0, 0
    
    # Create output directory
    figures_tables_dir = run_dir / "figures_tables"
    figures_tables_dir.mkdir(parents=True, exist_ok=True)
    
    assets_dir = figures_tables_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for stale in assets_dir.glob("*.png"):
        try:
            stale.unlink()
        except OSError:
            pass
    
    # Load config
    render_config = load_render_config(run_dir)
    dpi = render_config.get("dpi", 150)
    scale = render_config.get("scale", 2.0)
    
    # Load paragraphs
    paragraphs_path = run_dir / "paragraphs" / "paragraphs.jsonl"
    paragraphs = load_paragraphs(paragraphs_path)
    
    # Find caption paragraphs
    caption_paras = find_caption_paragraphs(paragraphs)
    
    verified_candidates = load_verified_figure_table_candidates(run_dir)

    # Load vision role labels to strengthen caption/header-footer decisions.
    blocks_by_page = load_blocks_norm(run_dir / "text" / "blocks_norm.jsonl")
    vision_by_page = load_vision_outputs(run_dir / "vision", dpi=dpi, scale=scale)
    vision_caption_by_page: dict[int, list[dict[str, Any]]] = {}
    header_footer_bboxes_by_page: dict[int, list[list[float]]] = {}
    for page, blocks in blocks_by_page.items():
        vision = vision_by_page.get(page, {})
        role_labels = vision.get("role_labels", {}) if isinstance(vision, dict) else {}
        if not isinstance(role_labels, dict):
            role_labels = {}
        for block in blocks:
            bid = str(block.get("block_id", "") or "")
            role = str(role_labels.get(bid, "") or "")
            bbox_pt = block.get("bbox_pt", [0, 0, 0, 0])
            if role in {ROLE_FIGURE_CAPTION, ROLE_TABLE_CAPTION}:
                vision_caption_by_page.setdefault(page, []).append({
                    "para_id": f"viscap_{page}_{bid}",
                    "text": str(block.get("text", "") or ""),
                    "role": role,
                    "page_span": {"start": page, "end": page},
                    "evidence_pointer": {"bbox_union": bbox_pt},
                })
            if role == "HeaderFooter":
                header_footer_bboxes_by_page.setdefault(page, []).append([
                    float(bbox_pt[0]), float(bbox_pt[1]), float(bbox_pt[2]), float(bbox_pt[3])
                ])
    region_caption_hints, region_header_footer_hints = build_vision_region_block_hints(blocks_by_page, vision_by_page)
    for page, hints in region_caption_hints.items():
        vision_caption_by_page.setdefault(page, []).extend(hints)
    for page, bboxes in region_header_footer_hints.items():
        header_footer_bboxes_by_page.setdefault(page, []).extend(bboxes)
    
    # Extract images from PDF
    pdf_images = extract_images_from_pdf(pdf_path)
    
    # Get page dimensions
    page_dims: dict[int, tuple[float, float]] = {}
    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) > 0:
            for i in range(len(doc)):
                rect = doc[i].rect
                page_dims[i + 1] = (rect.width, rect.height)
        doc.close()
    except Exception:
        page_dims = {}
    
    assets: list[FigureTableAsset] = []
    figure_count = 0
    table_count = 0
    images_cropped = 0
    
    # Process images from PDF
    for page_num, page_images in pdf_images.items():
        page_image_path = run_dir / "pages" / f"p{page_num:03d}.png"
        if not page_image_path.exists():
            continue

        page_width, page_height = page_dims.get(page_num, (612.0, 792.0))
        caption_paras_on_page = [
            p for p in caption_paras
            if int(p.get("page_span", {}).get("start", 0) or 0) == page_num
        ]
        vision_caption_candidates = vision_caption_by_page.get(page_num, [])
        header_footer_bboxes = header_footer_bboxes_by_page.get(page_num, [])
        
        for img_info in page_images:
            bbox_pt = img_info.get("bbox", [0, 0, 0, 0])
            if bbox_pt == [0, 0, 0, 0] or bbox_pt[2] <= bbox_pt[0] or bbox_pt[3] <= bbox_pt[1]:
                continue
            
            # Find associated caption
            caption_para = find_related_caption_for_image(bbox_pt, caption_paras_on_page)
            if caption_para is None and vision_caption_candidates:
                caption_para = find_related_caption_for_image(bbox_pt, vision_caption_candidates)

            if caption_para is None and is_probable_header_footer_artifact(bbox_pt, header_footer_bboxes):
                continue

            if not is_likely_body_asset(bbox_pt, page_width, page_height, caption_para):
                continue
            
            # Determine asset type from caption
            asset_type = "figure"
            verification_boost = 0.0
            if caption_para:
                text = caption_para.get("text", "")
                para_id = caption_para.get("para_id", "")
                if is_table_caption(text)[0]:
                    asset_type = "table"
                if para_id and para_id.startswith("viscap_"):
                    parts = para_id.split("_")
                    if len(parts) >= 3:
                        block_id = "_".join(parts[-2:])
                        vc = verified_candidates.get(block_id, {})
                        if vc:
                            if vc.get("type") in ("figure", "table"):
                                asset_type = vc["type"]
                            verification_boost = vc.get("confidence", 0.0) * 0.2
            else:
                width = img_info.get("width", 0)
                height = img_info.get("height", 0)
                if height > 0 and width / height < 2:
                    asset_type = "figure"
                else:
                    asset_type = "table"
            
            if asset_type == "table":
                table_hint_bbox_pt: Optional[list[float]] = None
                if caption_para is not None:
                    table_hint_bbox_pt = propose_table_bbox_from_blocks(
                        caption_para,
                        blocks_by_page.get(page_num, []),
                        page_width,
                        page_height,
                    )
                if table_hint_bbox_pt is None:
                    continue
                bbox_pt = table_hint_bbox_pt

            # Generate asset ID
            caption_index: Optional[int] = None
            if caption_para:
                ctext = str(caption_para.get("text", "") or "")
                if asset_type == "figure":
                    caption_index = caption_num_to_index(is_figure_caption(ctext)[1])
                else:
                    caption_index = caption_num_to_index(is_table_caption(ctext)[1])

            if asset_type == "figure":
                if caption_index is not None:
                    asset_id = compute_asset_id("figure", caption_index)
                    figure_count = max(figure_count, caption_index)
                else:
                    figure_count += 1
                    asset_id = compute_asset_id("figure", figure_count)
            else:
                if caption_index is not None:
                    asset_id = compute_asset_id("table", caption_index)
                    table_count = max(table_count, caption_index)
                else:
                    table_count += 1
                    asset_id = compute_asset_id("table", table_count)
            
            # Convert bbox to pixels
            bbox_px = pt_to_px_bbox(bbox_pt, dpi, scale)
            
            # Extract caption text
            caption_text = None
            caption_id = None
            if caption_para:
                caption_text = canonicalize_caption_text(caption_para.get("text", ""), asset_type)
                caption_id = caption_para.get("para_id")
            
            # Crop image
            image_path = assets_dir / f"{asset_id}.png"
            if crop_image_from_page(page_image_path, bbox_px, image_path):
                if is_crop_plausible(asset_type, image_path):
                    if asset_type == "figure" and caption_text and os.environ.get("FIGURE_APPEND_LEGEND", "1").strip() != "0":
                        append_legend_to_image(image_path, caption_text)
                    images_cropped += 1
                    image_path_str = str(image_path.relative_to(run_dir))
                else:
                    try:
                        image_path.unlink()
                    except OSError:
                        pass
                    image_path_str = None
            else:
                image_path_str = None

            # Reject obvious non-figure/table artifacts
            if image_path_str is None:
                continue
            
            asset = FigureTableAsset(
                asset_id=asset_id,
                asset_type=asset_type,
                page=page_num,
                bbox_px=bbox_px,
                caption_text=caption_text,
                caption_id=caption_id,
                source_para_id=caption_id,
                image_path=image_path_str,
                text_content=None,
                summary_content=None,
                confidence=min(1.0, (0.6 if caption_para else 0.4) + verification_boost),
            )
            assets.append(asset)
    
    # Always synthesize missing assets from caption paragraphs that were not matched.
    used_caption_ids = {
        str(a.caption_id) for a in assets
        if a.caption_id
    }
    for caption_para in caption_paras:
        caption_id = str(caption_para.get("para_id", "") or "")
        if caption_id and caption_id in used_caption_ids:
            continue

        text = str(caption_para.get("text", "") or "")
        page_span = caption_para.get("page_span", {})
        page = int(page_span.get("start", 1) or 1)

        is_fig, fig_num_text, _ = is_figure_caption(text)
        is_tbl, tbl_num_text, _ = is_table_caption(text)

        if is_fig:
            asset_type = "figure"
            caption_index = caption_num_to_index(fig_num_text)
            if caption_index is not None:
                asset_id = compute_asset_id("figure", caption_index)
                figure_count = max(figure_count, caption_index)
            else:
                figure_count += 1
                asset_id = compute_asset_id("figure", figure_count)
        elif is_tbl:
            asset_type = "table"
            caption_index = caption_num_to_index(tbl_num_text)
            if caption_index is not None:
                asset_id = compute_asset_id("table", caption_index)
                table_count = max(table_count, caption_index)
            else:
                table_count += 1
                asset_id = compute_asset_id("table", table_count)
        else:
            continue

        bbox_pt = caption_para.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])
        if bbox_pt == [0, 0, 0, 0]:
            bbox_pt = [100, 100, 500, 500]

        caption_bbox_px = pt_to_px_bbox(bbox_pt, dpi, scale)
        page_image_path = run_dir / "pages" / f"p{page:03d}.png"
        image_path = assets_dir / f"{asset_id}.png"

        page_width, page_height = page_dims.get(int(page), (612.0, 792.0))

        table_bbox_pt: Optional[list[float]] = None
        if asset_type == "table":
            table_bbox_pt = propose_table_bbox_from_blocks(
                caption_para,
                blocks_by_page.get(page, []),
                page_width,
                page_height,
            )

        canonical_caption = canonicalize_caption_text(text, asset_type)
        if asset_type == "table" and table_bbox_pt is None and _is_table_caption_summary_noise(canonical_caption):
            continue

        if table_bbox_pt is not None:
            crop_bbox = pt_to_px_bbox(table_bbox_pt, dpi, scale)
        else:
            est_bbox_pt = estimate_bbox_from_caption(
                caption_para, paragraphs, page_width, page_height, is_fig, prefer_above=True
            )
            if est_bbox_pt:
                crop_bbox = pt_to_px_bbox(est_bbox_pt, dpi, scale)
            else:
                crop_bbox = [int(c) for c in caption_bbox_px]

        vision_bbox = None if table_bbox_pt is not None else propose_bbox_with_vision(page_image_path, text, asset_type)
        if vision_bbox is not None:
            try:
                with Image.open(page_image_path) as page_img:
                    if asset_type == "figure":
                        crop_bbox = inflate_bbox_px(vision_bbox, page_img.width, page_img.height, 0.08)
                    else:
                        merged_bbox = union_bbox_px(crop_bbox, vision_bbox)
                        crop_bbox = inflate_bbox_px(merged_bbox, page_img.width, page_img.height, 0.28)
                    crop_bbox = constrain_bbox_by_caption_context(
                        crop_bbox,
                        caption_bbox_px,
                        page_img.width,
                        page_img.height,
                        asset_type,
                    )
                    if asset_type == "figure":
                        role_labels = vision_by_page.get(page, {}).get("role_labels", {}) if isinstance(vision_by_page.get(page, {}), dict) else {}
                        if not isinstance(role_labels, dict):
                            role_labels = {}
                        crop_bbox = expand_figure_bbox_with_nearby_blocks(
                            crop_bbox,
                            blocks_by_page.get(page, []),
                            role_labels,
                            dpi,
                            scale,
                            page_img.width,
                            page_img.height,
                        )
            except OSError:
                crop_bbox = union_bbox_px(crop_bbox, vision_bbox)
        else:
            try:
                with Image.open(page_image_path) as page_img:
                    if not (asset_type == "table" and table_bbox_pt is not None):
                        crop_bbox = constrain_bbox_by_caption_context(
                            crop_bbox,
                            caption_bbox_px,
                            page_img.width,
                            page_img.height,
                            asset_type,
                        )
                    if asset_type == "figure":
                        role_labels = vision_by_page.get(page, {}).get("role_labels", {}) if isinstance(vision_by_page.get(page, {}), dict) else {}
                        if not isinstance(role_labels, dict):
                            role_labels = {}
                        crop_bbox = expand_figure_bbox_with_nearby_blocks(
                            crop_bbox,
                            blocks_by_page.get(page, []),
                            role_labels,
                            dpi,
                            scale,
                            page_img.width,
                            page_img.height,
                        )
            except OSError:
                pass

        if not (asset_type == "table" and table_bbox_pt is not None):
            crop_bbox = auto_trim_bbox_by_density(page_image_path, crop_bbox, asset_type)

        image_path_str = None
        if crop_image_from_page(page_image_path, crop_bbox, image_path):
            plausible = is_crop_plausible(asset_type, image_path)
            if not plausible:
                alt_est_bbox_pt = estimate_bbox_from_caption(
                    caption_para,
                    paragraphs,
                    page_width,
                    page_height,
                    is_fig,
                    prefer_above=False,
                )
                if alt_est_bbox_pt is not None:
                    alt_bbox = pt_to_px_bbox(alt_est_bbox_pt, dpi, scale)
                    if crop_image_from_page(page_image_path, alt_bbox, image_path) and is_crop_plausible(asset_type, image_path):
                        crop_bbox = alt_bbox
                        plausible = True
            if not plausible:
                try:
                    image_path.unlink()
                except OSError:
                    pass
            if plausible:
                if asset_type == "figure" and text and os.environ.get("FIGURE_APPEND_LEGEND", "1").strip() != "0":
                    append_legend_to_image(image_path, text)
                images_cropped += 1
                image_path_str = str(image_path.relative_to(run_dir))

        asset = FigureTableAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            page=page,
            bbox_px=crop_bbox,
            caption_text=canonical_caption,
            caption_id=caption_id or None,
            source_para_id=caption_id or None,
            image_path=image_path_str,
            text_content=None,
            summary_content=None,
            confidence=0.5,
        )
        assets.append(asset)
    
    assets = deduplicate_assets(assets)

    # Build section mapping from paragraphs
    section_assets: dict[str, list[str]] = {}
    for asset in assets:
        if asset.source_para_id:
            # Find the section for this para
            for para in paragraphs:
                if para.get("para_id") == asset.source_para_id:
                    section = para.get("section_path")
                    if section and isinstance(section, list):
                        section_key = " > ".join(section)
                    else:
                        section_key = "Unknown"
                    
                    if section_key not in section_assets:
                        section_assets[section_key] = []
                    section_assets[section_key].append(asset.asset_id)
                    break
    
    # Write figure_table_index.jsonl
    index_path = figures_tables_dir / "figure_table_index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for asset in assets:
            record = {
                "asset_id": asset.asset_id,
                "asset_type": asset.asset_type,
                "page": asset.page,
                "bbox_px": asset.bbox_px,
                "caption_text": asset.caption_text,
                "caption_id": asset.caption_id,
                "source_para_id": asset.source_para_id,
                "image_path": asset.image_path,
                "text_content": asset.text_content,
                "summary_content": asset.summary_content,
                "confidence": asset.confidence,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Write figure_table_links.json
    links = FigureTableLinks(
        by_section=section_assets,
        by_fact={},  # Will be populated by reading stage
        by_synthesis_slot={},  # Will be populated by reading stage
    )
    
    links_path = figures_tables_dir / "figure_table_links.json"
    with open(links_path, "w", encoding="utf-8") as f:
        json.dump({
            "by_section": links.by_section,
            "by_fact": links.by_fact,
            "by_synthesis_slot": links.by_synthesis_slot,
        }, f, indent=2)

    rewrite_clean_document_after_asset_detection(run_dir, assets)
    
    return len(assets), images_cropped






















