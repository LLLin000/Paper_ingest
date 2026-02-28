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


def load_vision_outputs(vision_dir: Path) -> dict[int, dict[str, Any]]:
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
    return result


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
        # Tables generally start near/under caption; avoid huge unrelated header capture.
        y0 = max(y0, max(0, cy1 + 2))
        caption_w = max(1, cx1 - cx0)
        if caption_w < int(page_img_w * 0.75):
            x0 = max(x0, max(0, cx0 - 40))
            x1 = min(x1, min(page_img_w, cx1 + 40))

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


def is_figure_caption(text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Check if text is a figure caption. Returns (is_caption, figure_num, remainder)."""
    match = FIG_CAPTION_RE.match(text.strip())
    if match:
        fig_num = match.group(1)
        remainder = match.group(2).strip() if match.group(2) else ""
        return True, fig_num, remainder
    return False, None, None


def is_table_caption(text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Check if text is a table caption. Returns (is_caption, table_num, remainder)."""
    match = TABLE_CAPTION_RE.match(text.strip())
    if match:
        table_num = match.group(1)
        remainder = match.group(2).strip() if match.group(2) else ""
        return True, table_num, remainder
    return False, None, None


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
        is_fig, fig_num, fig_remainder = is_figure_caption(text)
        is_tbl, tbl_num, tbl_remainder = is_table_caption(text)
        
        if is_fig or is_tbl:
            captions.append(para)
            continue
        
        # Also check if role is explicitly set
        role = para.get("role", "")
        if role in (ROLE_FIGURE_CAPTION, ROLE_TABLE_CAPTION):
            captions.append(para)
    
    return captions


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
    vision_by_page = load_vision_outputs(run_dir / "vision")
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
                caption_text = caption_para.get("text", "")
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
        est_bbox_pt = estimate_bbox_from_caption(
            caption_para, paragraphs, page_width, page_height, is_fig, prefer_above=True
        )
        if est_bbox_pt:
            crop_bbox = pt_to_px_bbox(est_bbox_pt, dpi, scale)
        else:
            crop_bbox = [int(c) for c in caption_bbox_px]

        vision_bbox = propose_bbox_with_vision(page_image_path, text, asset_type)
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
            caption_text=text,
            caption_id=caption_id or None,
            source_para_id=caption_id or None,
            image_path=image_path_str,
            text_content=None,
            summary_content=None,
            confidence=0.5,
        )
        assets.append(asset)
    
    # Deduplicate assets by caption identity / page.
    dedup_assets: list[FigureTableAsset] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for a in assets:
        caption_norm = (a.caption_id or (a.caption_text or "")[:120]).strip().lower()
        key = (a.asset_type, int(a.page), caption_norm)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        dedup_assets.append(a)
    assets = dedup_assets

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
    
    return len(assets), images_cropped
