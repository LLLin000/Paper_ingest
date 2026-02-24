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

# pyright: reportOptionalIterable=false
# basedpyright: reportOptionalIterable=false

import hashlib
import json
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pymupdf

from .reference_providers_impl import build_doc_identity
from .reference_providers import collect_api_references


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
API_REFERENCE_FIELDS = (
    "title",
    "authors",
    "year",
    "doi",
    "pmid",
    "arxiv",
    "venue",
    "url",
    "source",
    "confidence",
    "source_chain",
    "filled_fields",
)


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


@dataclass
class ParagraphSpatialIndex:
    by_page_entries: dict[int, list[tuple[float, str]]]
    by_page_y: dict[int, list[float]]
    para_bbox: dict[str, list[float]]


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


def is_unbracketed_single_digit(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    if any(ch in candidate for ch in "[]()"):
        return False
    return bool(re.fullmatch(r"\d", candidate))


def should_demote_anchor_to_structural_link(anchor: CiteAnchor, marker_to_key: dict[int, str]) -> bool:
    if anchor.anchor_type != "citation_marker":
        return False
    marker_text = (anchor.anchor_text or "").strip()
    if not is_unbracketed_single_digit(marker_text):
        return False
    marker_values = expand_marker_text(marker_text)
    if not marker_values:
        return False
    return all(marker_to_key.get(marker) is None for marker in marker_values)


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


def build_paragraph_spatial_index(paragraphs: dict[str, dict[str, Any]]) -> ParagraphSpatialIndex:
    by_page_entries: dict[int, list[tuple[float, str]]] = {}
    by_page_y: dict[int, list[float]] = {}
    para_bbox: dict[str, list[float]] = {}

    for para_id, para in paragraphs.items():
        raw_bbox = para.get("evidence_pointer", {}).get("bbox_union", [0.0, 0.0, 0.0, 0.0])
        if isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
            bbox = [float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
        para_bbox[para_id] = bbox

        page_span = para.get("page_span", {})
        start_page = int(page_span.get("start", 1))
        end_page = int(page_span.get("end", start_page))
        if end_page < start_page:
            end_page = start_page

        y0 = bbox[1]
        for page_num in range(start_page, end_page + 1):
            by_page_entries.setdefault(page_num, []).append((y0, para_id))

    for page_num, entries in by_page_entries.items():
        sorted_entries = sorted(entries, key=lambda x: (x[0], x[1]))
        by_page_entries[page_num] = sorted_entries
        by_page_y[page_num] = [y for y, _ in sorted_entries]

    return ParagraphSpatialIndex(
        by_page_entries=by_page_entries,
        by_page_y=by_page_y,
        para_bbox=para_bbox,
    )


def _find_nearest_on_page(
    index: ParagraphSpatialIndex,
    page: int,
    anchor_bbox: list[float],
) -> Optional[str]:
    entries = index.by_page_entries.get(page)
    y_values = index.by_page_y.get(page)
    if not entries or not y_values:
        return None

    anchor_y = float(anchor_bbox[1])
    anchor_x_center = (float(anchor_bbox[0]) + float(anchor_bbox[2])) / 2.0

    insertion = bisect_left(y_values, anchor_y)
    if len(entries) <= 8:
        candidate_slice = entries
    else:
        left = max(0, insertion - 4)
        right = min(len(entries), insertion + 4)
        candidate_slice = entries[left:right]

    best_para_id: Optional[str] = None
    best_score: tuple[float, float, str] = (float("inf"), float("inf"), "")
    for _, para_id in candidate_slice:
        bbox = index.para_bbox.get(para_id, [0.0, 0.0, 0.0, 0.0])
        para_y = float(bbox[1])
        para_x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
        score = (abs(para_y - anchor_y), abs(para_x_center - anchor_x_center), para_id)
        if score < best_score:
            best_score = score
            best_para_id = para_id
    return best_para_id


def find_nearest_para(
    anchor_bbox: list[float],
    page: int,
    paragraphs: dict[str, dict[str, Any]],
    paragraph_index: Optional[ParagraphSpatialIndex] = None,
) -> Optional[str]:
    if not paragraphs:
        return None

    index = paragraph_index or build_paragraph_spatial_index(paragraphs)
    nearest_same_page = _find_nearest_on_page(index, page, anchor_bbox)
    if nearest_same_page is not None:
        return nearest_same_page

    candidate_pages = sorted(index.by_page_entries.keys(), key=lambda p: abs(p - page))
    for candidate_page in candidate_pages:
        nearest = _find_nearest_on_page(index, candidate_page, anchor_bbox)
        if nearest is not None:
            return nearest

    return None


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


def normalize_api_reference_record(raw: dict[str, Any]) -> dict[str, Any]:
    authors_raw = raw.get("authors")
    authors: list[str] = []
    if isinstance(authors_raw, list):
        authors = [str(item).strip() for item in authors_raw if str(item).strip()]

    year_raw = raw.get("year")
    year = year_raw if isinstance(year_raw, int) else None

    confidence_raw = raw.get("confidence")
    confidence_value = 0.0
    if isinstance(confidence_raw, (int, float)):
        try:
            confidence_value = float(confidence_raw)
        except Exception:
            confidence_value = 0.0
    elif isinstance(confidence_raw, str):
        try:
            confidence_value = float(confidence_raw.strip())
        except Exception:
            confidence_value = 0.0
    confidence = max(0.0, min(1.0, confidence_value))

    source_chain_norm: list[str] = []
    source_chain_raw = raw.get("source_chain")
    if isinstance(source_chain_raw, list):
        idx = 0
        while idx < len(source_chain_raw):
            item = source_chain_raw[idx]
            token = str(item).strip()
            if token:
                source_chain_norm.append(token)
            idx += 1

    filled_fields_norm: list[str] = []
    filled_fields_raw = raw.get("filled_fields")
    if isinstance(filled_fields_raw, list):
        idx = 0
        while idx < len(filled_fields_raw):
            item = filled_fields_raw[idx]
            token = str(item).strip()
            if token:
                filled_fields_norm.append(token)
            idx += 1

    normalized = {
        "title": str(raw.get("title") or "").strip(),
        "authors": authors,
        "year": year,
        "doi": str(raw.get("doi") or "").strip().lower() or None,
        "pmid": str(raw.get("pmid") or "").strip() or None,
        "arxiv": str(raw.get("arxiv") or "").strip() or None,
        "venue": str(raw.get("venue") or "").strip(),
        "url": str(raw.get("url") or "").strip() or None,
        "source": str(raw.get("source") or "unknown").strip() or "unknown",
        "confidence": round(confidence, 3),
        "source_chain": source_chain_norm,
        "filled_fields": filled_fields_norm,
    }
    return {field: normalized[field] for field in API_REFERENCE_FIELDS}


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
    paragraph_index = build_paragraph_spatial_index(paragraphs)
    
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
                paragraph_index,
            )
    
    reference_paras = {
        para_id: para for para_id, para in paragraphs.items()
        if is_reference_paragraph(para)
    }
    reference_entries, marker_to_key = build_reference_entries(reference_paras)

    for anchor in anchors:
        if should_demote_anchor_to_structural_link(anchor, marker_to_key):
            anchor.anchor_type = "structural_link"

    refs_dir = run_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    doc_identity = build_doc_identity(run_dir, manifest_data)
    try:
        api_references, provider_status = collect_api_references(doc_identity)
    except Exception as e:
        api_references = []
        provider_status = [{
            "provider": "reference_collection",
            "status": "error",
            "reason": f"collection_exception:{type(e).__name__}",
            "records": 0,
        }]
    doc_identity["provider_status"] = provider_status

    doc_identity_path = refs_dir / "doc_identity.json"
    with open(doc_identity_path, "w", encoding="utf-8") as f:
        json.dump(doc_identity, f, ensure_ascii=False, indent=2)

    references_api_path = refs_dir / "references_api.jsonl"
    with open(references_api_path, "w", encoding="utf-8") as f:
        for ref in api_references:
            record = normalize_api_reference_record(ref)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Stage R3: Merge API + PDF references into a deduped merged set
    merged_path = refs_dir / "references_merged.jsonl"

    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not path.exists():
            return out
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        return out

    # Normalization helpers for dedupe keys
    def _normalized_title_key(title: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())

    def _record_key(rec: dict[str, Any]) -> str:
        doi = rec.get("doi") or rec.get("DOI") or None
        if isinstance(doi, str) and doi.strip():
            d = doi.strip().lower().rstrip('.,;:)')
            return f"doi:{d}"
        pmid = rec.get("pmid")
        if isinstance(pmid, str) and pmid.strip():
            return f"pmid:{pmid.strip()}"
        # fallback to normalized title+year
        title = rec.get("title") or rec.get("raw_text") or ""
        norm = _normalized_title_key(str(title))
        year = rec.get("year") or ""
        return f"title:{norm}|year:{year}"

    api_list = _load_jsonl(references_api_path)
    # build pdf_list from in-memory reference entries produced earlier in this run
    # so we don't depend on a file that is written later in the pipeline
    pdf_list = build_reference_catalog(reference_entries)

    merged_map: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def _ingest(rec: dict[str, Any], kind: str) -> None:
        # kind: 'api' or 'pdf'
        key = _record_key(rec)
        # prepare a normalized candidate record with common fields
        # normalize year defensively to avoid Optional[str].isdigit() calls
        year_raw = rec.get("year")
        year_val: Optional[int] = None
        if isinstance(year_raw, int):
            year_val = year_raw
        elif isinstance(year_raw, str):
            yr = year_raw.strip()
            if yr.isdigit():
                try:
                    year_val = int(yr)
                except Exception:
                    year_val = None

        cand = {
            "title": rec.get("title") or rec.get("raw_text") or "",
            "authors": rec.get("authors") or [],
            "year": year_val,
            "doi": (rec.get("doi") or rec.get("DOI") or None) if rec.get("doi") or rec.get("DOI") else None,
            "pmid": rec.get("pmid") or None,
            "arxiv": rec.get("arxiv") or None,
            "venue": rec.get("venue") or "",
            "url": rec.get("url") or None,
            "raw_text": rec.get("raw_text") or None,
        }
        # source attribution entry
        src = None
        if kind == "api":
            src = str(rec.get("source") or "api")
        else:
            # pdf catalog entries don't have 'source' field - use marker/page info
            src = f"pdf:para:{rec.get('source_para_id') or rec.get('para_id') or 'unknown'}"

        confidence_raw = rec.get("confidence")
        confidence = 0.0
        if isinstance(confidence_raw, (int, float)):
            try:
                confidence = float(confidence_raw)
            except Exception:
                confidence = 0.0
        elif isinstance(confidence_raw, str):
            try:
                confidence = float(confidence_raw.strip())
            except Exception:
                confidence = 0.0

        src_entry = {"provider": src, "confidence": round(max(0.0, min(1.0, confidence)), 3), "kind": kind}

        if key not in merged_map:
            merged_map[key] = {
                **cand,
                "sources": [src_entry],
                "confidence": src_entry["confidence"],
            }
            order.append(key)
            return

        # Merge into existing: prefer non-empty fields, and keep max confidence
        existing = merged_map[key]
        # Update simple fields preferring existing if present, else candidate
        for fld in ("title", "authors", "year", "doi", "pmid", "arxiv", "venue", "url", "raw_text"):
            if (existing.get(fld) in (None, "", [])) and cand.get(fld):
                existing[fld] = cand.get(fld)
        # append source entry
        existing.setdefault("sources", []).append(src_entry)
        # update confidence to max
        existing_conf_raw = existing.get("confidence")
        existing_conf = 0.0
        if isinstance(existing_conf_raw, (int, float)):
            try:
                existing_conf = float(existing_conf_raw)
            except Exception:
                existing_conf = 0.0
        elif isinstance(existing_conf_raw, str):
            try:
                existing_conf = float(existing_conf_raw.strip())
            except Exception:
                existing_conf = 0.0
        # ensure both operands to round() are float to satisfy typing
        try:
            chosen_conf = max(float(existing_conf), float(src_entry.get("confidence", 0.0)))
        except Exception:
            chosen_conf = max(existing_conf, 0.0)
        existing["confidence"] = round(chosen_conf, 3)

    # ingest api first then pdf (pdf may complement missing fields)
    for r in api_list:
        _ingest(r, "api")
    for r in pdf_list:
        _ingest(r, "pdf")

    # If a merged refs artifact already exists on disk (from earlier stage/run), prefer it
    merged_refs_lookup: dict[str, dict[str, Any]] = {}
    existing_merged = _load_jsonl(merged_path)
    use_disk_merged = False
    if existing_merged:
        use_disk_merged = True
        for r in existing_merged:
            if r.get("doi"):
                merged_refs_lookup[f"doi:{r.get('doi')}"] = r
            if r.get("pmid"):
                merged_refs_lookup[f"pmid:{r.get('pmid')}"] = r
            title_norm = _normalized_title_key(r.get("title") or r.get("raw_text") or "")
            year = r.get("year") or ""
            merged_refs_lookup[f"title:{title_norm}|year:{year}"] = r
    else:
        for k in order:
            rec = merged_map[k]
            if rec.get("doi"):
                merged_refs_lookup[f"doi:{rec.get('doi')}"] = rec
            if rec.get("pmid"):
                merged_refs_lookup[f"pmid:{rec.get('pmid')}"] = rec
            title_norm = _normalized_title_key(rec.get("title") or rec.get("raw_text") or "")
            year = rec.get("year") or ""
            merged_refs_lookup[f"title:{title_norm}|year:{year}"] = rec

    # Map citations preferring merged references when possible
    cite_maps: list[CiteMap] = []
    for anchor in anchors:
        # run the existing mapping logic first to get candidate
        candidate = map_citation_to_reference(
            anchor,
            paragraphs,
            reference_paras,
            reference_entries,
            marker_to_key,
        )

        mapped_key = candidate.mapped_ref_key
        chosen = candidate

        # If merged refs available, attempt to prefer them
        if merged_refs_lookup:
            # 1) If anchor text directly contains DOI/PMID that exists in merged refs
            if anchor.anchor_text and not mapped_key:
                ext = extract_reference_key(anchor.anchor_text)
                if ext and ext[0] in merged_refs_lookup:
                    k = ext[0]
                    # prefer merged key
                    chosen = CiteMap(anchor_id=anchor.anchor_id, mapped_ref_key=k, strategy_used=STRATEGY_REGEX, confidence=ext[2])
                    cite_maps.append(chosen)
                    continue

            # 2) If candidate mapped_key already matches merged ref, keep
            if mapped_key and mapped_key in merged_refs_lookup:
                cite_maps.append(candidate)
                continue

            # 3) If candidate mapped_key refers to a PDF key, try to find a merged match by title+year
            if mapped_key:
                # locate the reference entry for this ref_key
                match_entry = None
                for entry in reference_entries:
                    if entry.ref_key == mapped_key:
                        match_entry = entry
                        break
                if match_entry:
                    meta = parse_reference_metadata(match_entry.text)
                    # remove year tokens from title before normalizing to improve matching
                    raw_title = meta.get("title") or meta.get("raw_text") or ""
                    title_no_year = re.sub(r"\b(19|20)\d{2}\b", "", raw_title)
                    title_norm = _normalized_title_key(title_no_year)
                    year = meta.get("year") or ""
                    lookup_key = f"title:{title_norm}|year:{year}"
                    if lookup_key in merged_refs_lookup:
                        # map to merged key (prefer DOI/PMID if available)
                        merged_rec = merged_refs_lookup[lookup_key]
                        new_key = None
                        if merged_rec.get("doi"):
                            new_key = f"doi:{merged_rec.get('doi')}"
                        elif merged_rec.get("pmid"):
                            new_key = f"pmid:{merged_rec.get('pmid')}"
                        else:
                            new_key = lookup_key
                        new_conf = max(candidate.confidence, float(merged_rec.get("confidence") or 0.0))
                        chosen = CiteMap(anchor_id=anchor.anchor_id, mapped_ref_key=new_key, strategy_used=candidate.strategy_used, confidence=round(new_conf, 3))
                        cite_maps.append(chosen)
                        continue

        # Default: keep candidate
        cite_maps.append(candidate)

    # Write anchors and mapping outputs (contract unchanged)
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

    # write merged out in stable order
    with open(merged_path, "w", encoding="utf-8") as f:
        for k in order:
            rec = merged_map[k]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(anchors), len(cite_maps)
