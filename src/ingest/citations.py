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
INLINE_BRACKET_PATTERN = re.compile(r"\[(\d{1,3}(?:\s*[-,;]\s*\d{1,3})*)\]")
INLINE_PAREN_PATTERN = re.compile(r"\((\d{1,3}(?:\s*[-,;]\s*\d{1,3})*)\)")
REF_MARKER_PATTERN = re.compile(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})[\.)])")
REF_SEGMENT_MARKER_PATTERN = re.compile(r"(?:^|\s)(?:\[(\d{1,3})\]|(\d{1,3})[\.)])\s+")


@dataclass
class CiteAnchor:
    anchor_id: str
    page: int
    anchor_bbox: list[float]
    anchor_text: Optional[str]
    nearest_para_id: Optional[str]
    link_type: str
    anchor_type: str


@dataclass
class CiteMap:
    anchor_id: str
    mapped_ref_key: Optional[str]
    strategy_used: str
    confidence: float
    reason: Optional[str] = None


@dataclass
class ReferenceEntry:
    para_id: str
    page: int
    marker: Optional[int]
    ref_key: Optional[str]
    text: str


def parse_reference_metadata(text: str) -> dict[str, Any]:
    clean = re.sub(r"^\s*(?:\[\d+\]|\d+[\.)])\s*", "", text).strip()
    doi_match = DOI_PATTERN.search(clean)
    pmid_match = PMID_PATTERN.search(clean)
    doi = doi_match.group(1).lower().rstrip(".,;:)") if doi_match else None
    pmid = pmid_match.group(2) if pmid_match else None

    year_matches = list(re.finditer(r"\b(19\d{2}|20\d{2})\b", clean))
    year = int(year_matches[-1].group(1)) if year_matches else None

    normalized = " ".join(clean.split())
    if doi:
        normalized = re.sub(re.escape(doi), "", normalized, flags=re.IGNORECASE)
    if pmid:
        normalized = re.sub(r"\bPMID[:\s]*" + re.escape(pmid) + r"\b", "", normalized, flags=re.IGNORECASE)
    normalized = " ".join(normalized.split())

    chunks = [c.strip(" .") for c in re.split(r"\.\s+", normalized) if c.strip(" .")]

    def is_author_like(chunk: str) -> bool:
        if "et al" in chunk.lower():
            return True
        initial_hits = len(re.findall(r"\b[A-Z]\.", chunk))
        comma_hits = chunk.count(",")
        words = chunk.split()
        if words and initial_hits >= 2:
            return True
        if words and comma_hits >= 2 and len(words) <= 20:
            return True
        return False

    title = ""
    title_idx = -1
    for idx, chunk in enumerate(chunks):
        if is_author_like(chunk):
            continue
        if len(chunk.split()) < 4:
            continue
        title = chunk
        title_idx = idx
        break

    if not title and chunks:
        title = chunks[0]
        title_idx = 0

    if title_idx > 0:
        authors_raw = ". ".join(chunks[:title_idx])
    elif chunks:
        authors_raw = chunks[0]
    else:
        authors_raw = normalized

    venue = chunks[title_idx + 1] if (title_idx >= 0 and title_idx + 1 < len(chunks)) else ""

    raw_authors = [
        a.strip(" ,")
        for a in re.split(r",\s*(?=[A-Z][a-z]|[A-Z]\.)|\s+and\s+|\s*&\s*", authors_raw)
        if a.strip(" ,")
    ]
    authors: list[str] = []
    for item in raw_authors:
        if item.lower() in {"et al", "et al."}:
            continue
        if len(item) <= 2:
            continue
        if re.fullmatch(r"[A-Z]\.?", item):
            continue
        authors.append(item)
    if len(authors) > 10:
        authors = authors[:10]

    if len(title.split()) <= 2:
        title = normalized[:180]
    if not venue and len(chunks) >= 2 and title_idx >= 0:
        tail = chunks[title_idx + 1:] if title_idx + 1 < len(chunks) else []
        if tail:
            venue = tail[0]

    return {
        "authors": authors,
        "title": title,
        "year": year,
        "venue": venue,
        "doi": doi,
        "pmid": pmid,
        "raw_text": clean,
    }


def build_reference_catalog(reference_entries: list[ReferenceEntry]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in reference_entries:
        if not entry.ref_key or entry.ref_key in seen:
            continue
        meta = parse_reference_metadata(entry.text)
        catalog.append({
            "ref_key": entry.ref_key,
            "source_para_id": entry.para_id,
            "page": entry.page,
            "marker": entry.marker,
            "authors": meta.get("authors", []),
            "title": meta.get("title", ""),
            "year": meta.get("year"),
            "venue": meta.get("venue", ""),
            "doi": meta.get("doi"),
            "pmid": meta.get("pmid"),
            "raw_text": meta.get("raw_text", entry.text),
            "confidence": 0.65 if entry.marker is not None else 0.55,
        })
        seen.add(entry.ref_key)
    return sorted(catalog, key=lambda x: str(x.get("ref_key", "")))


def compute_anchor_id(page: int, link_index: int, dest: Optional[str] = None) -> str:
    key = f"p{page}_i{link_index}_{dest or 'nodest'}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"anchor_{h[:12]}"


def compute_anchor_id_from_text(page: int, para_id: str, marker_text: str) -> str:
    key = f"p{page}_{para_id}_{marker_text}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"anchor_{h[:12]}"


def classify_anchor_type(anchor_text: Optional[str], link_type: str) -> str:
    text = (anchor_text or "").strip().lower()
    if text:
        if re.fullmatch(r"[\[\(]?\d+(?:\s*[-,;]\s*\d+)*[\]\)]?", text):
            return "citation_marker"
        if DOI_PATTERN.search(text) or PMID_PATTERN.search(text):
            return "citation_marker"
        if text.startswith("http") or "doi.org/" in text or "pubmed" in text:
            return "resource_link"
    if link_type == "internal":
        return "citation_marker"
    if link_type == "external":
        return "resource_link"
    return "structural_link"


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


def expand_marker_text(marker_text: str) -> list[int]:
    values: list[int] = []
    clean = marker_text.replace(" ", "")
    for part in re.split(r"[,;]", clean):
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                start = int(bounds[0])
                end = int(bounds[1])
                if start <= end and (end - start) <= 30:
                    values.extend(list(range(start, end + 1)))
            continue
        if part.isdigit():
            values.append(int(part))
    dedup = sorted(set(v for v in values if v > 0))
    return dedup


def extract_inline_anchors(paragraphs: dict[str, dict[str, Any]]) -> list[CiteAnchor]:
    anchors: list[CiteAnchor] = []
    ordered = sorted(
        paragraphs.items(),
        key=lambda x: (x[1].get("page_span", {}).get("start", 0), x[0]),
    )
    for para_id, para in ordered:
        if is_reference_paragraph(para):
            continue
        text = str(para.get("text", ""))
        if not text:
            continue
        page = int(para.get("page_span", {}).get("start", 1))
        bbox = para.get("evidence_pointer", {}).get("bbox_union", [0.0, 0.0, 0.0, 0.0])
        if not (isinstance(bbox, list) and len(bbox) >= 4):
            bbox = [0.0, 0.0, 0.0, 0.0]
        for pattern in (INLINE_BRACKET_PATTERN, INLINE_PAREN_PATTERN):
            for match in pattern.finditer(text):
                marker_text = match.group(1)
                for marker_num in expand_marker_text(marker_text):
                    normalized_marker = f"[{marker_num}]"
                    anchors.append(CiteAnchor(
                        anchor_id=compute_anchor_id_from_text(page, para_id, normalized_marker),
                        page=page,
                        anchor_bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        anchor_text=normalized_marker,
                        nearest_para_id=para_id,
                        link_type="internal",
                        anchor_type="citation_marker",
                    ))
    return anchors


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

    section_path = para.get("section_path") or []
    if isinstance(section_path, list):
        section_text = " ".join(str(x).lower() for x in section_path)
        if "reference" in section_text or "bibliograph" in section_text:
            return True
    
    text = str(para.get("text", ""))
    text_lower = text.lower()
    if re.match(r'^\[\d+\]', text):
        return True
    if re.match(r'^\d+\.\s', text):
        return True
    if len(text) > 30 and any(x in text_lower for x in ["doi:", "pmid:", "http"]):
        return True

    marker_hits = len(REF_SEGMENT_MARKER_PATTERN.findall(text))
    if marker_hits >= 5:
        return True
    if "references" in text_lower and marker_hits >= 2:
        return True
    
    return False


def build_reference_entries(reference_paras: dict[str, dict[str, Any]]) -> tuple[list[ReferenceEntry], dict[int, str]]:
    entries: list[ReferenceEntry] = []
    marker_to_key: dict[int, str] = {}
    ordered = sorted(
        reference_paras.items(),
        key=lambda x: (x[1].get("page_span", {}).get("start", 0), x[0]),
    )

    for para_id, para in ordered:
        text = str(para.get("text", "")).strip()
        if not text:
            continue
        page = int(para.get("page_span", {}).get("start", 1))

        segment_matches = list(REF_SEGMENT_MARKER_PATTERN.finditer(text))
        if len(segment_matches) >= 2:
            for i, match in enumerate(segment_matches):
                marker_str = match.group(1) or match.group(2)
                if not marker_str or not marker_str.isdigit():
                    continue
                marker_val = int(marker_str)
                seg_start = match.end()
                seg_end = segment_matches[i + 1].start() if i + 1 < len(segment_matches) else len(text)
                seg_text = text[seg_start:seg_end].strip()
                if not seg_text:
                    continue
                seg_key = normalize_reference_key(seg_text)
                entry = ReferenceEntry(
                    para_id=para_id,
                    page=page,
                    marker=marker_val,
                    ref_key=seg_key,
                    text=seg_text,
                )
                entries.append(entry)
                if seg_key and marker_val not in marker_to_key:
                    marker_to_key[marker_val] = seg_key
            continue

        marker = None
        marker_match = REF_MARKER_PATTERN.match(text)
        if marker_match:
            marker_str = marker_match.group(1) or marker_match.group(2)
            if marker_str and marker_str.isdigit():
                marker = int(marker_str)

        ref_key = normalize_reference_key(text)
        entry = ReferenceEntry(
            para_id=para_id,
            page=page,
            marker=marker,
            ref_key=ref_key,
            text=text,
        )
        entries.append(entry)

        if marker is not None and ref_key and marker not in marker_to_key:
            marker_to_key[marker] = ref_key

    return entries, marker_to_key


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
    reference_entries: list[ReferenceEntry],
) -> Optional[tuple[str, float]]:
    if not anchor_text or not reference_entries:
        return None

    anchor_lower = anchor_text.lower().strip()
    if re.fullmatch(r"[\[\(]?\d+(?:\s*[-,;]\s*\d+)*[\]\)]?", anchor_lower):
        return None

    anchor_tokens = {
        t for t in re.findall(r"[a-z0-9]{4,}", anchor_lower)
        if t not in {"http", "https", "www", "doi", "pmid"}
    }
    if not anchor_tokens:
        return None

    candidates: list[tuple[str, float]] = []

    for entry in reference_entries:
        para_text = entry.text
        if not entry.ref_key:
            continue

        para_lower = para_text.lower()
        if anchor_lower in para_lower and len(anchor_lower) >= 8:
            candidates.append((entry.ref_key, 0.88))
            continue

        para_tokens = set(re.findall(r"[a-z0-9]{4,}", para_lower))
        if not para_tokens:
            continue
        overlap = len(anchor_tokens & para_tokens)
        if overlap == 0:
            continue
        score = overlap / max(1, len(anchor_tokens))
        if score >= 0.35:
            candidates.append((entry.ref_key, min(0.84, 0.55 + score * 0.35)))
    
    if not candidates:
        return None
    
    best = max(candidates, key=lambda x: x[1])
    return best


def normalize_reference_key(text: str) -> Optional[str]:
    ref_result = extract_reference_key(text)
    if ref_result:
        return ref_result[0]

    text_clean = re.sub(r"^\s*(?:\[\d+\]|\d+[\.)])\s*", "", text).strip()
    text_lower = text_clean.lower()
    author_match = re.match(r'^([a-z]+)', text_lower)
    year_match = re.search(r'(\d{4})', text_clean)
    
    if author_match and year_match:
        author = author_match.group(1)
        year = year_match.group(1)
        title_hash = hashlib.md5(text_clean.encode()).hexdigest()[:8]
        return f"bib:{author}_{year}_{title_hash}"
    
    return None


def map_citation_to_reference(
    anchor: CiteAnchor,
    paragraphs: dict[str, dict[str, Any]],
    reference_paras: dict[str, dict[str, Any]],
    reference_entries: list[ReferenceEntry],
    marker_to_key: dict[int, str],
) -> CiteMap:
    anchor_id = anchor.anchor_id
    anchor_text = anchor.anchor_text or ""
    link_type = anchor.link_type
    anchor_type = anchor.anchor_type

    if anchor_type != "citation_marker":
        return CiteMap(
            anchor_id=anchor_id,
            mapped_ref_key=None,
            strategy_used=STRATEGY_NONE,
            confidence=0.0,
            reason=f"non-citation anchor type: {anchor_type}",
        )
    
    # Strategy 1: Regex extraction from anchor text (DOI/PMID)
    if anchor_text:
        ref_key = normalize_reference_key(anchor_text)
        if ref_key:
            conf = 0.92 if ref_key.startswith(("doi:", "pmid:")) else 0.62
            return CiteMap(
                anchor_id=anchor_id,
                mapped_ref_key=ref_key,
                strategy_used=STRATEGY_REGEX,
                confidence=conf,
            )

    # Strategy 2: Marker-based reference mapping ([12], [3-5])
    marker_text = anchor_text.strip() if anchor_text else ""
    marker_payload = marker_text.strip("[]() ")
    marker_values = expand_marker_text(marker_payload) if marker_payload else []
    if marker_values and marker_to_key:
        for marker in marker_values:
            mapped = marker_to_key.get(marker)
            if mapped:
                return CiteMap(
                    anchor_id=anchor_id,
                    mapped_ref_key=mapped,
                    strategy_used=STRATEGY_INTERNAL_DEST,
                    confidence=0.86,
                )
        return CiteMap(
            anchor_id=anchor_id,
            mapped_ref_key=None,
            strategy_used=STRATEGY_NONE,
            confidence=0.0,
            reason="marker not found in reference index",
        )
    if marker_values and not marker_to_key:
        return CiteMap(
            anchor_id=anchor_id,
            mapped_ref_key=None,
            strategy_used=STRATEGY_NONE,
            confidence=0.0,
            reason="reference marker index unavailable",
        )

    # Strategy 3: Internal destination / nearest reference paragraph
    if link_type == "internal":
        nearest_para_id = anchor.nearest_para_id
        if nearest_para_id and nearest_para_id in reference_paras:
            para_text = str(reference_paras[nearest_para_id].get("text", ""))
            ref_key = normalize_reference_key(para_text)
            if ref_key:
                return CiteMap(
                    anchor_id=anchor_id,
                    mapped_ref_key=ref_key,
                    strategy_used=STRATEGY_INTERNAL_DEST,
                    confidence=0.82,
                )

        anchor_page = anchor.page
        for entry in reference_entries:
            para_page = entry.page
            if abs(para_page - anchor_page) <= 2:
                if anchor_text and entry.ref_key:
                    para_text = entry.text
                    if anchor_text.strip() in para_text or para_text.strip() in anchor_text:
                            return CiteMap(
                                anchor_id=anchor_id,
                                mapped_ref_key=entry.ref_key,
                                strategy_used=STRATEGY_INTERNAL_DEST,
                                confidence=0.80,
                            )

    # Strategy 4: Fuzzy match
    if anchor_text and reference_entries:
        fuzzy_result = fuzzy_match_reference(anchor_text, reference_entries)
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
    elif not reference_entries:
        reason = "no reference entries available"
    elif marker_values and marker_to_key:
        reason = "marker not found in reference index"
    elif link_type == "external":
        reason = "external link not mappable to bibliography"

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
            if rect:
                try:
                    extracted = page.get_textbox(rect)
                    if isinstance(extracted, str) and extracted.strip():
                        anchor_text = extracted.strip()
                except Exception:
                    anchor_text = None
            if not anchor_text and link_type == "external" and dest:
                anchor_text = str(dest)
            
            anchor_id = compute_anchor_id(page_num_1based, link_idx, str(dest))
            
            anchors.append(CiteAnchor(
                anchor_id=anchor_id,
                page=page_num_1based,
                anchor_bbox=bbox,
                anchor_text=anchor_text,
                nearest_para_id=None,
                link_type=link_type,
                anchor_type=classify_anchor_type(anchor_text, link_type),
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
    
    link_anchors = extract_links_from_pdf(pdf_path)
    inline_anchors = extract_inline_anchors(paragraphs)
    dedup: dict[tuple[int, str, str, str], CiteAnchor] = {}
    for anchor in link_anchors + inline_anchors:
        key = (
            int(anchor.page),
            str(anchor.anchor_text or ""),
            str(anchor.link_type),
            str(anchor.anchor_type),
        )
        if key not in dedup:
            dedup[key] = anchor
    anchors = sorted(dedup.values(), key=lambda a: (a.page, a.anchor_id))
    
    for anchor in anchors:
        if not anchor.nearest_para_id:
            anchor.nearest_para_id = find_nearest_para(
                anchor.anchor_bbox,
                anchor.page,
                paragraphs,
            )
    
    reference_paras = {
        para_id: para for para_id, para in paragraphs.items()
        if is_reference_paragraph(para)
    }
    reference_entries, marker_to_key = build_reference_entries(reference_paras)
    
    cite_maps: list[CiteMap] = []
    for anchor in anchors:
        mapping = map_citation_to_reference(
            anchor,
            paragraphs,
            reference_paras,
            reference_entries,
            marker_to_key,
        )
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
                "anchor_type": anchor.anchor_type,
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

    catalog_path = citations_dir / "reference_catalog.jsonl"
    reference_catalog = build_reference_catalog(reference_entries)
    with open(catalog_path, "w", encoding="utf-8") as f:
        for ref in reference_catalog:
            f.write(json.dumps(ref, ensure_ascii=False) + "\n")
    
    return len(anchors), len(cite_maps)
