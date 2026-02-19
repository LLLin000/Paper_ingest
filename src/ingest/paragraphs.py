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
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


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
            confidence_by_page[page] = data.get("confidence", 0.5)  # type: ignore[assignment]
            merge_groups_by_page[page] = data.get("merge_groups", [])  # type: ignore[assignment]
            role_labels_by_page[page] = data.get("role_labels", {})  # type: ignore[assignment]
        except (json.JSONDecodeError, IOError):
            continue
    
    return confidence_by_page, merge_groups_by_page, role_labels_by_page


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
    
    text = re.sub(r"^\d+(\.\d+)*\s*", "", text).strip()
    
    return text if text else None


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
        
        role = role_labels.get(bid, "Body")
        heading = infer_section_heading(block, role)
        if heading:
            section_path_candidates.append(heading)
    
    merged_text = " ".join(t.strip() for t in texts if t.strip())
    
    primary_role = "Body"
    if sorted_block_ids:
        first_bid = sorted_block_ids[0]
        primary_role = role_labels.get(first_bid, "Body")
    
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
    role = role_labels.get(block_id, "Body")
    
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
            
            para = build_paragraph_from_group(
                group_id, block_ids, blocks, role_labels, confidence
            )
            all_paragraphs.append(para)
            grouped_blocks.update(block_ids)  # type: ignore[arg-type]
    
    for block_id, block in blocks.items():
        if block_id in grouped_blocks:
            continue
        
        page = block.get("page", 1)  # type: ignore[assignment]
        role_labels = role_labels_by_page.get(page, {})  # type: ignore[index]
        confidence = confidence_by_page.get(page, 0.5)  # type: ignore[index]
        
        role = role_labels.get(block_id, "Body")
        if role == "HeaderFooter":
            continue
        
        para = build_singleton_paragraph(block_id, block, role_labels, confidence)
        all_paragraphs.append(para)
    
    return all_paragraphs


def build_neighbors(paragraphs: list[Paragraph]) -> list[Paragraph]:
    def sort_key(p: Paragraph) -> tuple[int, float]:  # type: ignore[return-value]
        bbox = p.evidence_pointer.get("bbox_union", [0, 0, 0, 0])  # type: ignore[union-attr]
        return (p.page_span["start"], float(bbox[1]))  # type: ignore[index]
    
    sorted_paras = sorted(paragraphs, key=sort_key)
    
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


def run_paragraphs(
    run_dir: Path,
    manifest: Optional[Any] = None,
) -> tuple[int, int]:
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    paragraphs_dir = run_dir / "paragraphs"
    
    paragraphs_dir.mkdir(parents=True, exist_ok=True)
    
    blocks = load_blocks(text_dir / "blocks_norm.jsonl")
    confidence_by_page, merge_groups_by_page, role_labels_by_page = load_vision_outputs(vision_dir)
    
    if not blocks:
        return 0, 0
    
    paragraphs = aggregate_paragraphs(
        blocks,
        merge_groups_by_page,
        role_labels_by_page,
        confidence_by_page,
    )
    
    paragraphs = build_neighbors(paragraphs)
    
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
    
    return len(paragraphs), len(blocks)
