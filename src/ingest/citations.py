"""Citation Mapper - Extracts PDF links and maps them to references.

Contract: .sisyphus/plans/pdf-blueprint-contracts.md (lines 141-156)

Output schema per citations/cite_anchors.jsonl:
- anchor_id: stable ID for the citation anchor
- page: page number (1-based)
- anchor_bbox: [x0, y0, x1, y1] in PDF points
- anchor_text: text of the link if available
- nearest_para_id: para_id of nearest paragraph
- link_type: 'internal' or 'external'

Output schema per citations/cite_map.jsonl:
- anchor_id: reference to cite_anchors record
- mapped_ref_key: normalized reference key (doi:, pmid:, bib:) or null
- strategy_used: 'internal_dest', 'regex', 'fuzzy', 'none'
- confidence: float 0.0-1.0
- reason: required when unmapped
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pymupdf


# Strategy enum values
STRATEGY_INTERNAL_DEST = "internal_dest"
STRATEGY_REGEX = "regex"
STRATEGY_FUZZY = "fuzzy"
STRATEGY_NONE = "none"

# Regex patterns for reference extraction
DOI_PATTERN = re.compile(
    r"\b(10\.\d{4,}(?:\.\d+)*/[^\s\]>\"\']+)\b",
    re.IGNORECASE
)
PMID_PATTERN = re.compile(r"\b(PMID[:\s]*)?(\d{7,9})\b", re.IGNORECASE)


@dataclass
class CiteAnchor:
    anchor_id: str
    page: int
    anchor_bbox: list[float]
    anchor_text: Optional[str]
    nearest_para_id: Optional[str]
    link_type: str


@dataclass
class CiteMap:
    anchor_id: str
    mapped_ref_key: Optional[str]
    strategy_used: str
    confidence: float
    reason: Optional[str] = None


def compute_anchor_id(page: int, link_index: int, dest: Optional[str] = None) -> str:
    key = f"p{page}_i{link_index}_{dest or 'nodest'}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"anchor_{h[:12]}"


def load_paragraphs(paragraphs_path: Path) -> dict[str, dict[str, Any]]:
    paragraphs: dict[str, dict[str, Any]] = {}
    if not paragraphs_path.exists():
        return paragraphs
    
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                para_id = d.get("para_id")
                if para_id:
                    paragraphs[para_id] = d
            except json.JSONDecodeError:
                continue
    return paragraphs


def find_nearest_para(
    anchor_bbox: list[float],
    page: int,
    paragraphs: dict[str, dict[str, Any]],
) -> Optional[str]:
    if not paragraphs:
        return None
    
    same_page_paras = [
        (para_id, para) for para_id, para in paragraphs.items()
        if para.get("page_span", {}).get("start", 999) <= page <= para.get("page_span", {}).get("end", 0)
    ]
    
    if not same_page_paras:
        same_page_paras = list(paragraphs.items())
    
    anchor_y = anchor_bbox[1]
    
    best_para_id: Optional[str] = None
    best_distance = float('inf')
    
    for para_id, para in same_page_paras:
        bbox = para.get("evidence_pointer", {}).get("bbox_union", [0, 0, 0, 0])
        para_y = bbox[1]
        distance = abs(para_y - anchor_y)
        
        if distance < best_distance:
            best_distance = distance
            best_para_id = para_id
    
    return best_para_id


def is_reference_paragraph(para: dict[str, Any]) -> bool:
    role = para.get("role", "")
    if role == "ReferenceList":
        return True
    
    text = para.get("text", "")
    if re.match(r'^\[\d+\]', text):
        return True
    if re.match(r'^\d+\.\s', text):
        return True
    if len(text) > 30 and any(x in text.lower() for x in ["doi:", "pmid:", "http"]):
        return True
    
    return False


def extract_reference_key(text: str) -> Optional[tuple[str, str, float]]:
    doi_match = DOI_PATTERN.search(text)
    if doi_match:
        doi = doi_match.group(1).lower()
        doi = doi.rstrip('.,;:)')
        if '&' in doi:
            doi = doi.split('&')[0]
        if '?' in doi:
            doi = doi.split('?')[0]
        return (f"doi:{doi}", STRATEGY_REGEX, 0.95)
    
    pmid_match = PMID_PATTERN.search(text)
    if pmid_match:
        pmid = pmid_match.group(2)
        return (f"pmid:{pmid}", STRATEGY_REGEX, 0.90)
    
    return None


def fuzzy_match_reference(
    anchor_text: str,
    reference_paras: dict[str, dict[str, Any]],
) -> Optional[tuple[str, float]]:
    if not anchor_text or not reference_paras:
        return None
    
    anchor_lower = anchor_text.lower().strip()
    candidates: list[tuple[str, float]] = []
    
    for para_id, para in reference_paras.items():
        para_text = para.get("text", "")
        
        if anchor_lower in para_text.lower():
            candidates.append((para_id, 0.85))
            continue
        
        ref_key = extract_reference_key(para_text)
        if ref_key:
            candidates.append((ref_key[0], 0.70))
    
    if not candidates:
        return None
    
    best = max(candidates, key=lambda x: x[1])
    return best


def normalize_reference_key(text: str) -> Optional[str]:
    ref_result = extract_reference_key(text)
    if ref_result:
        return ref_result[0]
    
    text_lower = text.lower()
    author_match = re.match(r'^([a-z]+)', text_lower)
    year_match = re.search(r'(\d{4})', text)
    
    if author_match and year_match:
        author = author_match.group(1)
        year = year_match.group(1)
        title_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"bib:{author}_{year}_{title_hash}"
    
    return None


def map_citation_to_reference(
    anchor: CiteAnchor,
    paragraphs: dict[str, dict[str, Any]],
    reference_paras: dict[str, dict[str, Any]],
) -> CiteMap:
    anchor_id = anchor.anchor_id
    anchor_text = anchor.anchor_text or ""
    link_type = anchor.link_type
    
    # Strategy 1: Internal destination
    if link_type == "internal":
        anchor_page = anchor.page
        
        for para_id, para in reference_paras.items():
            para_page = para.get("page_span", {}).get("start", 0)
            if abs(para_page - anchor_page) <= 2:
                if anchor_text:
                    para_text = para.get("text", "")
                    if anchor_text.strip() in para_text or para_text.strip() in anchor_text:
                        ref_key = normalize_reference_key(para_text)
                        if ref_key:
                            return CiteMap(
                                anchor_id=anchor_id,
                                mapped_ref_key=ref_key,
                                strategy_used=STRATEGY_INTERNAL_DEST,
                                confidence=0.80,
                            )
    
    # Strategy 2: Regex extraction
    if anchor_text:
        ref_key = normalize_reference_key(anchor_text)
        if ref_key:
            conf = 0.90 if ref_key.startswith(("doi:", "pmid:")) else 0.60
            return CiteMap(
                anchor_id=anchor_id,
                mapped_ref_key=ref_key,
                strategy_used=STRATEGY_REGEX,
                confidence=conf,
            )
    
    # Strategy 3: Fuzzy match
    if anchor_text and reference_paras:
        fuzzy_result = fuzzy_match_reference(anchor_text, reference_paras)
        if fuzzy_result:
            ref_key, conf = fuzzy_result
            return CiteMap(
                anchor_id=anchor_id,
                mapped_ref_key=ref_key,
                strategy_used=STRATEGY_FUZZY,
                confidence=conf,
            )
    
    # Strategy 4: None - could not map
    reason = "no matching reference found"
    if not anchor_text:
        reason = "empty anchor_text"
    elif not reference_paras:
        reason = "no reference paragraphs available"
    
    return CiteMap(
        anchor_id=anchor_id,
        mapped_ref_key=None,
        strategy_used=STRATEGY_NONE,
        confidence=0.0,
        reason=reason,
    )


def extract_links_from_pdf(pdf_path: Path) -> list[CiteAnchor]:
    anchors: list[CiteAnchor] = []
    
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return anchors
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_num_1based = page_num + 1
        
        links = page.get_links()
        
        for link_idx, link in enumerate(links):
            kind = link.get("kind", -1)
            
            # Link kind values: 1=URI, 2=URL (external), 3=Internal (goto), 4=Gotob
            if kind == 3:
                link_type = "internal"
            elif kind in (1, 2):
                link_type = "external"
            elif kind == 4:
                link_type = "external"
            else:
                link_type = "external"
            
            dest = link.get("dest") or link.get("uri", "")
            
            # Get bbox - PyMuPDF uses 'from' key for link rect
            rect = link.get("from")
            if rect:
                bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            
            anchor_text: Optional[str] = None
            if link_type == "external" and dest:
                anchor_text = dest
            
            anchor_id = compute_anchor_id(page_num_1based, link_idx, str(dest))
            
            anchors.append(CiteAnchor(
                anchor_id=anchor_id,
                page=page_num_1based,
                anchor_bbox=bbox,
                anchor_text=anchor_text,
                nearest_para_id=None,
                link_type=link_type,
            ))
    
    doc.close()
    return anchors


def run_citations(
    run_dir: Path,
    manifest: Optional[Any] = None,
) -> tuple[int, int]:
    # Load manifest to get PDF path
    if manifest is None:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        else:
            return 0, 0
    else:
        # manifest is a Pydantic model - convert to dict
        manifest_data = manifest.model_dump()
    
    pdf_path = Path(manifest_data.get("input_pdf_path", ""))
    if not pdf_path.exists():
        return 0, 0
    
    citations_dir = run_dir / "citations"
    citations_dir.mkdir(parents=True, exist_ok=True)
    
    paragraphs_path = run_dir / "paragraphs" / "paragraphs.jsonl"
    paragraphs = load_paragraphs(paragraphs_path)
    
    anchors = extract_links_from_pdf(pdf_path)
    
    for anchor in anchors:
        anchor.nearest_para_id = find_nearest_para(
            anchor.anchor_bbox,
            anchor.page,
            paragraphs,
        )
    
    reference_paras = {
        para_id: para for para_id, para in paragraphs.items()
        if is_reference_paragraph(para)
    }
    
    cite_maps: list[CiteMap] = []
    for anchor in anchors:
        mapping = map_citation_to_reference(anchor, paragraphs, reference_paras)
        cite_maps.append(mapping)
    
    anchors_path = citations_dir / "cite_anchors.jsonl"
    with open(anchors_path, "w", encoding="utf-8") as f:
        for anchor in anchors:
            record = {
                "anchor_id": anchor.anchor_id,
                "page": anchor.page,
                "anchor_bbox": anchor.anchor_bbox,
                "anchor_text": anchor.anchor_text,
                "nearest_para_id": anchor.nearest_para_id,
                "link_type": anchor.link_type,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    map_path = citations_dir / "cite_map.jsonl"
    with open(map_path, "w", encoding="utf-8") as f:
        for mapping in cite_maps:
            record = {
                "anchor_id": mapping.anchor_id,
                "mapped_ref_key": mapping.mapped_ref_key,
                "strategy_used": mapping.strategy_used,
                "confidence": mapping.confidence,
            }
            if mapping.reason:
                record["reason"] = mapping.reason
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return len(anchors), len(cite_maps)
