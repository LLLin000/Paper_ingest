"""Text/Layout Extractor stage implementation.

Contract reference: .sisyphus/plans/pdf-blueprint-contracts.md

blocks_raw.jsonl (lines 116-122):
- block_id: stable within document (p{page}_b{index})
- page: 1-based
- bbox_pt: [x0,y0,x1,y1] in PDF points
- text: string
- font_stats: object (may be partial)

blocks_norm.jsonl (lines 124-128):
- All fields from blocks_raw plus:
- is_heading_candidate
- is_header_footer_candidate
- column_guess
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
from collections import Counter

import pymupdf

from .manifest import Manifest, load_manifest


@dataclass
class FontStats:
    sizes: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    is_bold: bool = False
    is_italic: bool = False
    avg_size: float = 0.0
    dominant_font: str = ""


@dataclass
class RawBlock:
    block_id: str
    page: int
    bbox_pt: list[float]
    text: str
    font_stats: dict[str, Any]


@dataclass
class NormBlock(RawBlock):
    is_heading_candidate: bool = False
    is_header_footer_candidate: bool = False
    column_guess: int = 1


@dataclass
class FaultInjectionEvent:
    stage: str
    fault: str
    retry_attempts: int
    fallback_used: bool
    status: str


def render_page_to_png(
    page: pymupdf.Page,
    output_path: Path,
    dpi: int = 150,
    scale: float = 2.0,
) -> None:
    zoom = dpi / 72.0 * scale
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(output_path))


def extract_font_stats(span: dict[str, Any]) -> FontStats:
    stats = FontStats()
    size = span.get("size", 0.0)
    font_name = span.get("font", "")
    stats.sizes = [size] if isinstance(size, (int, float)) else [0.0]
    stats.names = [font_name] if isinstance(font_name, str) else [""]
    stats.avg_size = float(size) if isinstance(size, (int, float)) else 0.0
    stats.dominant_font = font_name if isinstance(font_name, str) else ""
    if isinstance(font_name, str):
        font_lower = font_name.lower()
        stats.is_bold = "bold" in font_lower or "heavy" in font_lower or "black" in font_lower
        stats.is_italic = "italic" in font_lower or "oblique" in font_lower
    return stats


def extract_blocks_from_page(
    page: pymupdf.Page,
    page_num: int,
    block_index_offset: int = 0,
) -> list[RawBlock]:
    blocks = []
    text_dict = page.get_text("dict")
    if not isinstance(text_dict, dict):
        return blocks
    
    block_idx = block_index_offset
    page_blocks = text_dict.get("blocks", [])
    if not isinstance(page_blocks, list):
        return blocks

    for block in page_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type", 0) != 0:
            continue

        lines = block.get("lines", [])
        if not isinstance(lines, list):
            continue

        for line in lines:
            if not isinstance(line, dict):
                continue
            bbox = line.get("bbox", (0, 0, 0, 0))
            text_parts = []
            all_sizes = []
            all_fonts = []
            has_bold = False
            has_italic = False

            spans = line.get("spans", [])
            if not isinstance(spans, list):
                continue

            for span in spans:
                if not isinstance(span, dict):
                    continue
                text = span.get("text", "")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
                    size = span.get("size", 0.0)
                    if isinstance(size, (int, float)):
                        all_sizes.append(float(size))
                    font = span.get("font", "")
                    if isinstance(font, str):
                        all_fonts.append(font)
                        font_lower = font.lower()
                        if "bold" in font_lower or "heavy" in font_lower:
                            has_bold = True
                        if "italic" in font_lower or "oblique" in font_lower:
                            has_italic = True

            combined_text = " ".join(text_parts)
            if not combined_text.strip():
                continue

            font_stats: dict[str, Any] = {
                "sizes": all_sizes,
                "names": list(set(all_fonts)),
                "avg_size": sum(all_sizes) / len(all_sizes) if all_sizes else 0.0,
                "is_bold": has_bold,
                "is_italic": has_italic,
                "dominant_font": Counter(all_fonts).most_common(1)[0][0] if all_fonts else "",
            }

            if isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
                bbox_list = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            else:
                bbox_list = [0.0, 0.0, 0.0, 0.0]

            raw_block = RawBlock(
                block_id=f"p{page_num}_b{block_idx}",
                page=page_num,
                bbox_pt=bbox_list,
                text=combined_text,
                font_stats=font_stats,
            )
            blocks.append(raw_block)
            block_idx += 1

    return blocks


def compute_heading_candidate(
    block: RawBlock,
    avg_body_size: float,
) -> bool:
    font_stats = block.font_stats
    avg_size = font_stats.get("avg_size", 0.0)
    is_bold = font_stats.get("is_bold", False)
    text = block.text.strip()
    is_larger = isinstance(avg_size, (int, float)) and float(avg_size) > avg_body_size * 1.1
    is_short = len(text) < 150
    return is_larger and is_short and bool(is_bold)


def compute_header_footer_candidate(
    block: RawBlock,
    page_height: float,
) -> bool:
    bbox = block.bbox_pt
    y0, y1 = bbox[1], bbox[3]
    text = block.text.strip()
    top_threshold = page_height * 0.10
    bottom_threshold = page_height * 0.90
    in_top = y0 < top_threshold
    in_bottom = y1 > bottom_threshold
    is_short = len(text) < 100
    return (in_top or in_bottom) and is_short


def compute_column_guess(block: RawBlock, page_width: float) -> int:
    bbox = block.bbox_pt
    x_center = (bbox[0] + bbox[2]) / 2
    return 1 if x_center < page_width / 2 else 2


def compute_average_body_size(blocks: list[RawBlock]) -> float:
    sizes = []
    for block in blocks:
        avg_size = block.font_stats.get("avg_size", 0.0)
        if isinstance(avg_size, (int, float)) and avg_size > 0:
            sizes.append(float(avg_size))
    return sum(sizes) / len(sizes) if sizes else 12.0


def normalize_block(
    block: RawBlock,
    page_width: float,
    page_height: float,
    avg_body_size: float,
) -> NormBlock:
    return NormBlock(
        block_id=block.block_id,
        page=block.page,
        bbox_pt=block.bbox_pt,
        text=block.text,
        font_stats=block.font_stats,
        is_heading_candidate=compute_heading_candidate(block, avg_body_size),
        is_header_footer_candidate=compute_header_footer_candidate(block, page_height),
        column_guess=compute_column_guess(block, page_width),
    )


def run_extractor(
    run_dir: Path,
    manifest: Optional[Manifest] = None,
    inject_missing_font_stats: bool = False,
) -> tuple[int, int]:
    if manifest is None:
        manifest = load_manifest(run_dir)

    pdf_path = Path(manifest.input_pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    dpi = manifest.render_config.dpi
    scale = manifest.render_config.scale
    pages_dir = run_dir / "pages"
    text_dir = run_dir / "text"
    qa_dir = run_dir / "qa"
    pages_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    fault_events: list[FaultInjectionEvent] = []
    doc = pymupdf.open(str(pdf_path))

    try:
        all_raw_blocks: list[RawBlock] = []
        all_norm_blocks: list[NormBlock] = []
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            png_path = pages_dir / f"p{page_num:03d}.png"
            render_page_to_png(page, png_path, dpi=dpi, scale=scale)
            blocks = extract_blocks_from_page(page, page_num)
            all_raw_blocks.extend(blocks)

        avg_body_size = compute_average_body_size(all_raw_blocks)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            page_blocks = [b for b in all_raw_blocks if b.page == page_num]

            for block in page_blocks:
                if inject_missing_font_stats:
                    block.font_stats = {}
                    fault_events.append(FaultInjectionEvent(
                        stage="extractor",
                        fault="inject-missing-font-stats",
                        retry_attempts=0,
                        fallback_used=True,
                        status="degraded",
                    ))

                norm_block = normalize_block(
                    block, page_width, page_height, avg_body_size
                )
                all_norm_blocks.append(norm_block)

        raw_path = text_dir / "blocks_raw.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for block in all_raw_blocks:
                f.write(json.dumps(asdict(block), ensure_ascii=False) + "\n")

        norm_path = text_dir / "blocks_norm.jsonl"
        with open(norm_path, "w", encoding="utf-8") as f:
            for block in all_norm_blocks:
                f.write(json.dumps(asdict(block), ensure_ascii=False) + "\n")

        if fault_events:
            qa_dir.mkdir(parents=True, exist_ok=True)
            fault_path = qa_dir / "fault_injection.json"
            existing_events = []
            if fault_path.exists():
                try:
                    with open(fault_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        existing_events = data if isinstance(data, list) else []
                except (json.JSONDecodeError, IOError):
                    existing_events = []

            all_events = existing_events + [asdict(e) for e in fault_events]
            with open(fault_path, "w", encoding="utf-8") as f:
                json.dump(all_events, f, indent=2, ensure_ascii=False)

        return total_pages, len(all_raw_blocks)

    finally:
        doc.close()
