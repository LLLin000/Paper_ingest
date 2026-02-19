"""Overlay generation for visual debugging.

Contract reference: .sisyphus/plans/pdf-blueprint-contracts.md

Coordinate Rules (lines 186-189):
- PDF-native coordinates use points (pt) with origin from PyMuPDF convention.
- Rendered coordinates use pixels (px) at fixed dpi and scale.
- Any bbox_px must be derived from bbox_pt * scale.

Overlay stage produces annotated page images with block bboxes and IDs.
Output: run/<id>/pages/pXXX_annot.png for each page.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .manifest import Manifest, load_manifest


@dataclass
class Block:
    """Block data loaded from blocks_raw.jsonl."""
    block_id: str
    page: int
    bbox_pt: list[float]
    text: str
    font_stats: dict[str, Any]


def pt_to_px(bbox_pt: list[float], dpi: int, scale: float) -> list[int]:
    """Convert PDF point coordinates to pixel coordinates.
    
    Per contract (lines 186-189):
    - bbox_px derived from bbox_pt * scale
    - zoom = dpi / 72.0 * scale (same as render_page_to_png)
    
    Args:
        bbox_pt: [x0, y0, x1, y1] in PDF points
        dpi: DPI used for rendering
        scale: Scale factor used for rendering
        
    Returns:
        [x0, y0, x1, y1] in pixels (integers for PIL)
    """
    zoom = dpi / 72.0 * scale
    return [int(coord * zoom) for coord in bbox_pt]


def load_blocks(blocks_path: Path) -> list[Block]:
    """Load blocks from blocks_raw.jsonl.
    
    Args:
        blocks_path: Path to blocks_raw.jsonl
        
    Returns:
        List of Block objects
    """
    blocks = []
    if not blocks_path.exists():
        warnings.warn(f"Blocks file not found: {blocks_path}")
        return blocks
    
    with open(blocks_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                required_fields = ["block_id", "page", "bbox_pt", "text"]
                if not all(k in data for k in required_fields):
                    warnings.warn(f"Line {line_num}: missing required fields, skipping")
                    continue
                if not isinstance(data["bbox_pt"], list) or len(data["bbox_pt"]) < 4:
                    warnings.warn(f"Line {line_num}: invalid bbox_pt format, skipping")
                    continue
                    
                block = Block(
                    block_id=data["block_id"],
                    page=data["page"],
                    bbox_pt=[float(x) for x in data["bbox_pt"][:4]],
                    text=data.get("text", ""),
                    font_stats=data.get("font_stats", {}),
                )
                blocks.append(block)
            except json.JSONDecodeError as e:
                warnings.warn(f"Line {line_num}: JSON decode error: {e}, skipping")
            except (KeyError, TypeError, ValueError) as e:
                warnings.warn(f"Line {line_num}: data error: {e}, skipping")
    
    return blocks


def group_blocks_by_page(blocks: list[Block]) -> dict[int, list[Block]]:
    """Group blocks by page number.
    
    Args:
        blocks: List of Block objects
        
    Returns:
        Dict mapping page number to list of blocks on that page
    """
    pages: dict[int, list[Block]] = {}
    for block in blocks:
        if block.page not in pages:
            pages[block.page] = []
        pages[block.page].append(block)
    return pages


def _load_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font for overlay labels, with fallbacks."""
    font_paths = [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_block_overlay(
    image: Image.Image,
    blocks: list[Block],
    dpi: int,
    scale: float,
) -> Image.Image:
    """Draw block bounding boxes and IDs on an image.
    
    Args:
        image: PIL Image to annotate
        blocks: List of blocks on this page
        dpi: DPI used for rendering
        scale: Scale factor used for rendering
        
    Returns:
        Annotated image (copy, original not modified)
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(12)
    
    for block in blocks:
        bbox_px = pt_to_px(block.bbox_pt, dpi, scale)
        x0, y0, x1, y1 = bbox_px
        
        if x0 < 0 or y0 < 0 or x1 > image.width or y1 > image.height:
            warnings.warn(
                f"Block {block.block_id} bbox [{x0},{y0},{x1},{y1}] "
                f"exceeds image bounds [{image.width}x{image.height}], clipping"
            )
            x0 = max(0, min(x0, image.width))
            y0 = max(0, min(y0, image.height))
            x1 = max(0, min(x1, image.width))
            y1 = max(0, min(y1, image.height))
        
        draw.rectangle([x0, y0, x1, y1], outline="green", width=2)
        
        label = block.block_id
        text_bbox = draw.textbbox((x0, y0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        label_y = y0 - text_height - 2
        if label_y < 0:
            label_y = y0 + 2
        
        draw.rectangle(
            [x0, label_y, x0 + text_width + 4, label_y + text_height + 2],
            fill="white",
            outline="green",
        )
        draw.text((x0 + 2, label_y + 1), label, fill="green", font=font)
    
    return annotated


def run_overlay(
    run_dir: Path,
    manifest: Manifest | None = None,
) -> tuple[int, int]:
    """Run overlay generation for a document.
    
    Args:
        run_dir: Path to run directory (run/<doc_id>)
        manifest: Optional Manifest object (loaded from run_dir if not provided)
        
    Returns:
        Tuple of (pages_processed, blocks_drawn)
    """
    if manifest is None:
        manifest = load_manifest(run_dir)
    
    dpi = manifest.render_config.dpi
    scale = manifest.render_config.scale
    
    pages_dir = run_dir / "pages"
    text_dir = run_dir / "text"
    blocks_path = text_dir / "blocks_raw.jsonl"
    blocks = load_blocks(blocks_path)
    
    if not blocks:
        warnings.warn(f"No blocks found in {blocks_path}")
        return 0, 0
    
    blocks_by_page = group_blocks_by_page(blocks)
    
    pages_processed = 0
    blocks_drawn = 0
    
    for page_num, page_blocks in sorted(blocks_by_page.items()):
        page_path = pages_dir / f"p{page_num:03d}.png"
        if not page_path.exists():
            warnings.warn(f"Page image not found: {page_path}, skipping")
            continue
        
        try:
            image = Image.open(page_path)
        except Exception as e:
            warnings.warn(f"Failed to load image {page_path}: {e}, skipping")
            continue
        
        annotated = draw_block_overlay(image, page_blocks, dpi, scale)
        annot_path = pages_dir / f"p{page_num:03d}_annot.png"
        annotated.save(annot_path, "PNG")
        
        pages_processed += 1
        blocks_drawn += len(page_blocks)
    
    return pages_processed, blocks_drawn
