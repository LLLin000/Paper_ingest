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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pymupdf
from PIL import Image

from .manifest import Manifest


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


def estimate_bbox_from_caption(
    caption_para: dict[str, Any],
    other_paras: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    is_figure: bool,
) -> Optional[list[float]]:
    """Estimate the bounding box of a figure/table based on caption position.
    
    For figures: typically the figure is ABOVE the caption
    For tables: typically the table is ABOVE or BELOW the caption
    """
    caption_bbox = caption_para.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])
    if not caption_bbox or caption_bbox == [0, 0, 0, 0]:
        return None
    
    caption_y1 = caption_bbox[3]  # Bottom of caption
    
    # Estimate figure/table position (usually above caption)
    # Assume max height of 400pt for figure, 500pt for table
    max_height = 500 if is_figure else 600
    estimated_y0 = max(0, caption_y1 - max_height)
    
    # For figures, typically they're above the caption
    # For tables, they can be above or below
    # We'll use a reasonable default width (full column or page width)
    estimated_width = min(page_width - caption_bbox[0], 450)
    
    return [
        caption_bbox[0],  # Same left edge
        estimated_y0,
        caption_bbox[0] + estimated_width,
        caption_y1 - 5,  # Slightly above caption
    ]


def load_render_config(run_dir: Path) -> dict[str, Any]:
    """Load render config from manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {"dpi": 150, "scale": 2.0}
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    return manifest.get("render_config", {"dpi": 150, "scale": 2.0})


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
    
    for caption in caption_paras:
        caption_bbox = caption.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])
        if not caption_bbox:
            continue
        
        caption_y0 = caption_bbox[1]  # Top of caption
        
        # For figures: caption is usually below the image
        # Distance = caption_y0 - image_y1 (should be small positive)
        distance = caption_y0 - image_y1
        
        # Accept if caption is slightly below or above image
        if -50 <= distance < 200 and abs(distance) < best_distance:
            best_distance = abs(distance)
            best_caption = caption
    
    return best_caption


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
    
    # Load config
    render_config = load_render_config(run_dir)
    dpi = render_config.get("dpi", 150)
    scale = render_config.get("scale", 2.0)
    
    # Load paragraphs
    paragraphs_path = run_dir / "paragraphs" / "paragraphs.jsonl"
    paragraphs = load_paragraphs(paragraphs_path)
    
    # Find caption paragraphs
    caption_paras = find_caption_paragraphs(paragraphs)
    
    # Extract images from PDF
    pdf_images = extract_images_from_pdf(pdf_path)
    
    # Get page dimensions from first page
    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) > 0:
            first_page = doc[0]
            page_width = first_page.rect.width
            page_height = first_page.rect.height
        else:
            page_width, page_height = 612, 792  # Letter size
        doc.close()
    except Exception:
        page_width, page_height = 612, 792
    
    assets: list[FigureTableAsset] = []
    figure_count = 0
    table_count = 0
    images_cropped = 0
    
    # Process images from PDF
    for page_num, page_images in pdf_images.items():
        page_image_path = run_dir / "pages" / f"p{page_num:03d}.png"
        if not page_image_path.exists():
            continue
        
        for img_info in page_images:
            bbox_pt = img_info.get("bbox", [0, 0, 0, 0])
            if bbox_pt == [0, 0, 0, 0] or bbox_pt[2] <= bbox_pt[0] or bbox_pt[3] <= bbox_pt[1]:
                continue
            
            # Find associated caption
            caption_para = find_related_caption_for_image(bbox_pt, caption_paras)
            
            # Determine asset type from caption
            asset_type = "figure"
            if caption_para:
                text = caption_para.get("text", "")
                if is_table_caption(text)[0]:
                    asset_type = "table"
            else:
                # Use heuristics based on image dimensions
                width = img_info.get("width", 0)
                height = img_info.get("height", 0)
                if height > 0 and width / height < 2:
                    # Taller images are more likely figures
                    asset_type = "figure"
                else:
                    asset_type = "table"
            
            # Generate asset ID
            if asset_type == "figure":
                figure_count += 1
                asset_id = compute_asset_id("figure", figure_count)
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
                images_cropped += 1
                image_path_str = str(image_path.relative_to(run_dir))
            else:
                image_path_str = None
            
            asset = FigureTableAsset(
                asset_id=asset_id,
                asset_type=asset_type,
                page=page_num,
                bbox_px=bbox_px,
                caption_text=caption_text,
                caption_id=caption_id,
                source_para_id=caption_id,
                image_path=image_path_str,
                text_content=None,  # No OCR in baseline
                summary_content=None,  # No LLM summary in baseline
                confidence=0.6 if caption_para else 0.4,
            )
            assets.append(asset)
    
    # If no images found from PDF, create entries from caption paragraphs only
    if not assets:
        # Use caption paragraphs to create synthetic entries
        for caption_para in caption_paras:
            text = caption_para.get("text", "")
            page_span = caption_para.get("page_span", {})
            page = page_span.get("start", 1)
            
            is_fig, _, _ = is_figure_caption(text)
            is_tbl, _, _ = is_table_caption(text)
            
            if is_fig:
                asset_type = "figure"
                figure_count += 1
                asset_id = compute_asset_id("figure", figure_count)
            elif is_tbl:
                asset_type = "table"
                table_count += 1
                asset_id = compute_asset_id("table", table_count)
            else:
                continue
            
            # Get bbox from paragraph evidence
            bbox_pt = caption_para.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])
            if bbox_pt == [0, 0, 0, 0]:
                bbox_pt = [100, 100, 500, 500]  # Default
            
            bbox_px = pt_to_px_bbox(bbox_pt, dpi, scale)
            
            # Try to crop image from page
            page_image_path = run_dir / "pages" / f"p{page:03d}.png"
            image_path = assets_dir / f"{asset_id}.png"
            
            # For captions without images, create a placeholder region
            # Estimate the figure/table location
            est_bbox_pt = estimate_bbox_from_caption(
                caption_para, paragraphs, page_width, page_height, is_fig
            )
            if est_bbox_pt:
                crop_bbox = pt_to_px_bbox(est_bbox_pt, dpi, scale)
            else:
                crop_bbox = [int(c) for c in bbox_px]
            
            if crop_image_from_page(page_image_path, crop_bbox, image_path):
                images_cropped += 1
                image_path_str = str(image_path.relative_to(run_dir))
            else:
                image_path_str = None
            
            asset = FigureTableAsset(
                asset_id=asset_id,
                asset_type=asset_type,
                page=page,
                bbox_px=bbox_px,
                caption_text=text,
                caption_id=caption_para.get("para_id"),
                source_para_id=caption_para.get("para_id"),
                image_path=image_path_str,
                text_content=None,
                summary_content=None,
                confidence=0.5,
            )
            assets.append(asset)
    
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
