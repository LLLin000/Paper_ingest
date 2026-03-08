# pyright: reportOptionalMemberAccess=false
# basedpyright: reportOptionalMemberAccess=false
"""Reference providers and doc identity bootstrap for citations stage."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from difflib import SequenceMatcher
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


DOI_PATTERN = re.compile(r"\b(10\.\d{4,}(?:\.\d+)*/[^\s\]>\"']+)\b", re.IGNORECASE)
ARXIV_ID_PATTERN = re.compile(r"\b(?:arxiv\s*[:/]\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)

# Heuristic to strip leading author-like prefixes from unstructured titles in
# Crossref reference entries. We conservatively match sequences that look like
# author surnames and/or initials, joined by commas/and/&/semicolons and
# optionally followed by 'et al.' and an optional year. Examples:
#  - "Davies, M. R.; Smith, J. et al. - Title"
#  - "M. R. Davies & J. Smith - Title"
#  - "Davies MR 1999: Title"
# The regex intentionally remains conservative to avoid stripping legitimate
# title content.
# token that is either an initial sequence like 'M.' or 'M. R.' or a capitalized
# surname with at least one lowercase letter (to avoid matching ALLCAPS words like 'DNA')
# initial sequence like 'M.' or 'M. R.'
INITIAL_SEQ = r"(?:[A-Z]\.(?:\s*[A-Z]\.)*)"
# capitalized name with at least one lowercase to avoid ALLCAPS acronyms
CAP_NAME = r"(?:[A-Z][a-z][A-Za-z'`-]*)"
# allow either: initials optionally followed by a surname, or one-or-more capitalized names
AUTHOR_PREFIX_NAME = rf"(?:{INITIAL_SEQ}(?:\s+{CAP_NAME})?|{CAP_NAME}(?:\s+{CAP_NAME})*)"
# sequence: either a separated list of names (comma/and/&/;) optionally with 'et al.'
# and optional year; OR an initials sequence optionally followed by a capitalized name.
AUTHOR_PREFIX_SEQ = rf"(?:(?:{AUTHOR_PREFIX_NAME}(?:\s*(?:,|;|and|&)\s*{AUTHOR_PREFIX_NAME})+)(?:\s*,?\s*(?:et\s+al\.?))?(?:\s*[\(\,]?\d{{4}}[\)]?)?|(?:{INITIAL_SEQ}(?:\s+{CAP_NAME})?))"
# match at start followed by common separators (dash, colon, em-dash, dot, semicolon)
AUTHOR_PREFIX_RE = re.compile(rf"^\s*{AUTHOR_PREFIX_SEQ}[\s:\-\—\.\;]+", re.IGNORECASE)
SURNAME_INITIAL_START_RE = re.compile(r"^[A-Z][a-zA-Z'`-]+,\s*[A-Z](?:\.|\s)")
AUTHOR_SURNAME_INITIAL_RE = re.compile(
    r"\b([A-Z][a-zA-Z'`-]+,\s*(?:[A-Z]\.?\,?\s*){1,4}(?:[A-Z][a-zA-Z'`-]+)?)"
)
STRICT_AUTHOR_TOKEN_RE = r"[A-Z][A-Za-z'-]+,\s*(?:[A-Z]\.\s*){1,4}"
STRICT_AUTHOR_PREFIX_RE = re.compile(
    rf"^\s*(?:{STRICT_AUTHOR_TOKEN_RE}(?:\s*(?:,|;|&|and)\s*{STRICT_AUTHOR_TOKEN_RE})*)(?:\s*(?:,|;)?\s*et\s+al\.?)?\s*(?P<rest>.+)$",
    re.IGNORECASE,
)
FALLBACK_ET_AL_PREFIX_RE = re.compile(
    r"^\s*[A-Z][A-Za-z'-]+,\s*(?:[A-Z]\.\s*){1,4}(?:\s*(?:,|;|&|and)\s*[A-Z][A-Za-z'-]+,\s*(?:[A-Z]\.\s*){1,4})*\s*et\s+al\.?\s*(?P<rest>.+)$",
    re.IGNORECASE,
)


@dataclass
class NormalizedReference:
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    pmid: str | None
    arxiv: str | None
    venue: str
    url: str | None
    source: str
    confidence: float
    source_chain: list[str] = field(default_factory=list)
    filled_fields: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "pmid": self.pmid,
            "arxiv": self.arxiv,
            "venue": self.venue,
            "url": self.url,
            "source": self.source,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
            "source_chain": self.source_chain,
            "filled_fields": self.filled_fields,
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _extract_from_clean_document(clean_document_path: Path) -> dict[str, Any]:
    if not clean_document_path.exists():
        return {}

    try:
        content = clean_document_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    title = ""
    authors: list[str] = []
    doi = None
    arxiv = None

    lines = content.splitlines()
    for line in lines:
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            break

    in_authors = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## Authors":
            in_authors = True
            continue
        if in_authors and stripped.startswith("## "):
            break
        if in_authors and stripped.startswith("- "):
            author = stripped[2:].strip()
            if author:
                authors.append(author)

    doi_match = DOI_PATTERN.search(content)
    if doi_match:
        doi = doi_match.group(1).lower().rstrip(".,;:)")

    arxiv_match = ARXIV_ID_PATTERN.search(content)
    if arxiv_match:
        arxiv = arxiv_match.group(1)

    return {
        "title": title,
        "authors": authors,
        "doi": doi,
        "arxiv": arxiv,
        "source": "clean_document",
    }


def _extract_doi_from_blocks(blocks_dir: Path) -> Optional[str]:
    """Extract DOI from blocks_norm.jsonl (includes metadata blocks filtered from clean_document)."""
    blocks_path = blocks_dir / "blocks_norm.jsonl"
    if not blocks_path.exists():
        return None
    
    try:
        with open(blocks_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    block = json.loads(line)
                    text = str(block.get("text", ""))
                    doi_match = DOI_PATTERN.search(text)
                    if doi_match:
                        return doi_match.group(1).lower().rstrip(".,;:)")
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return None


def build_doc_identity(run_dir: Path, manifest_data: dict[str, Any]) -> dict[str, Any]:
    clean_doc = _extract_from_clean_document(run_dir / "text" / "clean_document.md")
    profile = _read_json(run_dir / "reading" / "paper_profile.json")

    title = str(clean_doc.get("title") or "").strip()
    if not title:
        title = str(profile.get("research_problem") or "").strip()

    authors = clean_doc.get("authors") if isinstance(clean_doc.get("authors"), list) else []
    if not isinstance(authors, list):
        authors = []
    authors = [str(x).strip() for x in authors if str(x).strip()]

    # Try to extract DOI from clean_document first, then from blocks
    doi = clean_doc.get("doi")
    if not doi:
        doi = _extract_doi_from_blocks(run_dir / "text")
    arxiv = clean_doc.get("arxiv")

    if not title:
        input_pdf = str(manifest_data.get("input_pdf_path") or "")
        if input_pdf:
            title = Path(input_pdf).stem

    title_source = "clean_document" if clean_doc.get("title") else (
        "reading_profile" if profile.get("research_problem") else "manifest_fallback"
    )

    identity = {
        "doc_id": manifest_data.get("doc_id"),
        "doi": doi,
        "arxiv": arxiv,
        "title": title,
        "authors": authors,
        "identifier_source": clean_doc.get("source", "none"),
        "title_source": title_source,
        "status": "ok" if any([doi, arxiv, title]) else "insufficient_identity",
    }

    # Optional: attempt Zotero sqlite lookup if configured to enrich identity.
    # This is a best-effort read-only enrichment and never hard-fails.
    zotero_path = os.environ.get("ZOTERO_SQLITE_PATH")
    if doi and zotero_path and os.path.exists(zotero_path):
        try:
            uri = f"file:{urllib.parse.quote(zotero_path)}?mode=ro&immutable=1"
            conn = sqlite3.connect(uri, uri=True, timeout=2)
            cur = conn.cursor()

            cur.execute(
                """
                SELECT i.itemID, i.key
                FROM items i
                JOIN itemData d ON d.itemID = i.itemID
                JOIN fields f ON f.fieldID = d.fieldID
                JOIN itemDataValues v ON v.valueID = d.valueID
                WHERE f.fieldName = 'DOI' AND lower(v.value) = lower(?)
                LIMIT 1
                """,
                (str(doi),),
            )
            item_row = cur.fetchone()

            if item_row and isinstance(item_row[0], int):
                item_id = int(item_row[0])
                item_key = str(item_row[1]) if len(item_row) > 1 and item_row[1] is not None else None

                cur.execute(
                    """
                    SELECT f.fieldName, v.value
                    FROM itemData d
                    JOIN fields f ON f.fieldID = d.fieldID
                    JOIN itemDataValues v ON v.valueID = d.valueID
                    WHERE d.itemID = ?
                    """,
                    (item_id,),
                )
                field_rows = cur.fetchall()
                field_map: dict[str, str] = {}
                for field_name, value in field_rows:
                    if isinstance(field_name, str) and isinstance(value, str):
                        field_map[field_name] = value

                cur.execute(
                    """
                    SELECT c.firstName, c.lastName, ic.orderIndex
                    FROM itemCreators ic
                    JOIN creators c ON c.creatorID = ic.creatorID
                    WHERE ic.itemID = ?
                    ORDER BY ic.orderIndex
                    """,
                    (item_id,),
                )
                author_rows = cur.fetchall()
                z_authors: list[str] = []
                for first_name, last_name, _ in author_rows:
                    fn = str(first_name).strip() if isinstance(first_name, str) else ""
                    ln = str(last_name).strip() if isinstance(last_name, str) else ""
                    full = " ".join(x for x in [fn, ln] if x).strip()
                    if full:
                        z_authors.append(full)

                z_title = field_map.get("title", "").strip()
                z_doi = field_map.get("DOI", "").strip().lower()

                if z_title:
                    identity["title"] = z_title
                    identity["title_source"] = "zotero_sqlite"
                if z_authors:
                    identity["authors"] = z_authors
                if z_doi:
                    identity["doi"] = z_doi

                identity["identifier_source"] = "zotero_sqlite"
                if item_key:
                    identity["zotero_item_key"] = item_key

            conn.close()
        except Exception:
            pass

    return identity


def _request_json(url: str, timeout_sec: float = 5.0) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "pdf-ingest/0.1.0 (reference-prototype)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
            if isinstance(data, (dict, list)):
                return data, None
            return None, "invalid_json_payload"
    except urllib.error.HTTPError as e:
        return None, f"http_{e.code}"
    except urllib.error.URLError as e:
        return None, f"network_error:{e.reason}"
    except TimeoutError:
        return None, "timeout"
    except Exception as e:
        return None, f"request_error:{type(e).__name__}"


def _request_text(url: str, timeout_sec: float = 5.0) -> tuple[str | None, str | None]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "pdf-ingest/0.1.0 (reference-prototype)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            return response.read().decode("utf-8", errors="replace"), None
    except urllib.error.HTTPError as e:
        return None, f"http_{e.code}"
    except urllib.error.URLError as e:
        return None, f"network_error:{e.reason}"
    except TimeoutError:
        return None, "timeout"
    except Exception as e:
        return None, f"request_error:{type(e).__name__}"


def _normalize_author_name(author_obj: dict[str, Any]) -> str:
    given = str(author_obj.get("given") or "").strip()
    family = str(author_obj.get("family") or "").strip()
    if given and family:
        return f"{given} {family}".strip()
    return family or given


def _is_probable_venue_fragment(segment: str) -> bool:
    seg = segment.strip()
    if not seg:
        return True
    words = [w.strip(".,;:()") for w in seg.split() if w.strip(".,;:()")]
    if re.search(r"\b\d+\s*[-:]\s*\d+\b", seg) and len(words) <= 10:
        return True
    if re.search(r"\b(?:vol|no|pp|issue|issn|doi|journal|surg|radiol|orthop)\b", seg, re.IGNORECASE):
        return True
    if words:
        abbrev_words = 0
        for w in words:
            if len(w) <= 4 and w[0].isupper():
                abbrev_words += 1
        if abbrev_words >= max(2, int(len(words) * 0.6)):
            return True
    if re.search(r"\b(?:19|20)\d{2}\b", seg):
        if re.search(r"\b(?:journal|surg|radiol|orthop|arthroscopy|clin|rev|res|tech)\b", seg, re.IGNORECASE):
            return True
        if re.search(r"\b\d+\s*,\s*\d+\b", seg):
            return True
    return False


def _split_reference_sentences(text: str) -> list[str]:
    raw = re.sub(r"\s+", " ", text).strip()
    if not raw:
        return []
    out: list[str] = []
    start = 0
    n = len(raw)
    idx = 0
    while idx < n:
        ch = raw[idx]
        if ch in ".!?":
            token_start = idx - 1
            while token_start >= 0 and raw[token_start].isalpha():
                token_start -= 1
            token = raw[token_start + 1:idx]
            next_idx = idx + 1
            while next_idx < n and raw[next_idx].isspace():
                next_idx += 1
            if next_idx < n:
                next_ch = raw[next_idx]
                is_initial = len(token) == 1 and token.isupper()
                if not is_initial and (next_ch.isupper() or next_ch.isdigit()):
                    sent = raw[start:idx + 1].strip()
                    if sent:
                        out.append(sent)
                    start = next_idx
                    idx = next_idx
                    continue
        idx += 1
    tail = raw[start:].strip()
    if tail:
        out.append(tail)
    return out


def _is_probable_author_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return False
    if SURNAME_INITIAL_START_RE.match(s):
        if re.search(r"\bet\s+al\.?\b", s, re.IGNORECASE):
            return True
        if len(AUTHOR_SURNAME_INITIAL_RE.findall(s)) >= 2:
            return True
    return bool(
        re.search(r",\s*[A-Z](?:\.|\s)", s)
        or re.search(r"\b[A-Z]\.\s*[A-Z]\.", s)
        or re.search(r"\s&\s", s)
    )


def _rewrite_author_prefixed_title(title: str) -> str:
    raw = title.strip()
    if not raw:
        return title
    if not SURNAME_INITIAL_START_RE.match(raw):
        return title

    m = STRICT_AUTHOR_PREFIX_RE.match(raw)
    if not m:
        m = FALLBACK_ET_AL_PREFIX_RE.match(raw)
        if not m:
            return title

    rest = str(m.group("rest") or "").strip()
    if not rest:
        return title
    rest = re.sub(r"^et\s+al\.?\s+", "", rest, flags=re.IGNORECASE)

    sentences = _split_reference_sentences(rest)
    if len(sentences) >= 2 and _is_probable_author_sentence(sentences[0]):
        candidate = sentences[1].strip(" \t-:;,.()")
        if candidate:
            if not _is_probable_venue_fragment(candidate):
                if not SURNAME_INITIAL_START_RE.match(candidate):
                    words = [w for w in candidate.split() if re.search(r"[A-Za-z]", w)]
                    letter_count = len(re.findall(r"[A-Za-z]", candidate))
                    first_char = candidate[0]
                    if (first_char.isupper() or first_char.isdigit()) and len(words) >= 2 and letter_count >= 5:
                        return candidate
                    first_token_match = re.match(r"^([A-Z][A-Za-z'\-]{5,})\b", candidate)
                    if len(words) == 1 and first_token_match:
                        next_tail = ". ".join(sentences[2:6]).strip()
                        if next_tail and _is_probable_venue_fragment(next_tail):
                            return candidate

    segments = [seg.strip(" \t-:;,.()") for seg in re.split(r"\.\s+", rest) if seg.strip()]
    if not segments:
        return title

    for idx, seg in enumerate(segments):
        candidate = seg.strip()
        if not candidate:
            continue
        if SURNAME_INITIAL_START_RE.match(candidate):
            continue
        if _is_probable_venue_fragment(candidate):
            continue

        token_match = re.match(r"^([A-Z][A-Za-z'-]+)", candidate)
        if not candidate or not (candidate[0].isupper() or candidate[0].isdigit()):
            continue

        words = [w for w in candidate.split() if re.search(r"[A-Za-z]", w)]
        if len(re.findall(r"[A-Za-z]", candidate)) < 5:
            continue

        if len(words) >= 2:
            return candidate

        first_token_match = re.match(r"^([A-Z][A-Za-z'\-]{5,})\b", candidate)
        if len(words) == 1 and first_token_match:
            next_tail = ". ".join(segments[idx + 1:idx + 5]).strip()
            if next_tail and _is_probable_venue_fragment(next_tail):
                return candidate

    return title


def _normalize_crossref_item(item: dict[str, Any], confidence: float) -> NormalizedReference:
    title_value = item.get("title")
    title = ""
    if isinstance(title_value, list) and title_value:
        title = str(title_value[0]).strip()
    elif isinstance(title_value, str):
        title = title_value.strip()

    author_value = item.get("author")
    authors: list[str] = []
    if isinstance(author_value, list):
        for a in author_value:
            if isinstance(a, dict):
                name = _normalize_author_name(a)
                if name:
                    authors.append(name)

    year = None
    issued = item.get("issued")
    if isinstance(issued, dict):
        date_parts = issued.get("date-parts")
        if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            first = date_parts[0][0]
            if isinstance(first, int):
                year = first

    venue = ""
    container = item.get("container-title")
    if isinstance(container, list) and container:
        venue = str(container[0]).strip()
    elif isinstance(container, str):
        venue = container.strip()

    doi = item.get("DOI")
    doi_str = str(doi).lower().strip() if doi else None

    return NormalizedReference(
        title=title,
        authors=authors,
        year=year,
        doi=doi_str,
        pmid=None,
        arxiv=None,
        venue=venue,
        url=str(item.get("URL") or "").strip() or None,
        source="crossref",
        confidence=confidence,
    )


def _normalize_pubmed_summary_item(item: dict[str, Any], confidence: float) -> NormalizedReference:
    title = str(item.get("title") or "").strip()
    authors: list[str] = []
    authors_raw = item.get("authors")
    if isinstance(authors_raw, list):
        for entry in authors_raw:
            if isinstance(entry, dict):
                name = str(entry.get("name") or "").strip()
                if name:
                    authors.append(name)

    year = None
    pubdate = str(item.get("pubdate") or "").strip()
    m = re.search(r"\b(19\d{2}|20\d{2})\b", pubdate)
    if m:
        try:
            year = int(m.group(1))
        except Exception:
            year = None

    doi = None
    articleids = item.get("articleids")
    if isinstance(articleids, list):
        for articleid in articleids:
            if isinstance(articleid, dict) and str(articleid.get("idtype") or "").lower() == "doi":
                doi_value = str(articleid.get("value") or "").strip().lower()
                if doi_value:
                    doi = doi_value
                    break

    pmid = str(item.get("uid") or "").strip() or None
    venue = str(item.get("source") or "").strip()

    return NormalizedReference(
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        pmid=pmid,
        arxiv=None,
        venue=venue,
        url=(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None),
        source="pubmed",
        confidence=confidence,
    )


def _first_author_token(record: NormalizedReference) -> str:
    if not record.authors:
        return ""
    first_author = str(record.authors[0]).strip()
    if not first_author:
        return ""
    for token in re.split(r"\s+|,", first_author):
        clean = re.sub(r"[^A-Za-z]", "", token)
        if len(clean) >= 3:
            return clean
    return ""


def _title_similarity_score(left: str, right: str) -> float:
    lkey = _normalized_title_key(left)
    rkey = _normalized_title_key(right)
    if not lkey or not rkey:
        return 0.0
    if lkey == rkey:
        return 1.0
    return SequenceMatcher(None, lkey, rkey).ratio()


def _title_year_match_is_strong(record: NormalizedReference, candidate: NormalizedReference) -> bool:
    if not _field_is_filled(record.title) or not _field_is_filled(candidate.title):
        return False
    score = _title_similarity_score(record.title, candidate.title)
    if score < 0.9:
        return False
    if record.year and candidate.year and abs(record.year - candidate.year) > 1:
        return False
    return True


def _backfill_fields_from_reference(candidate: NormalizedReference) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field_name in ["title", "authors", "year", "venue", "url", "doi", "pmid"]:
        value = getattr(candidate, field_name)
        if _field_is_filled(value):
            fields[field_name] = value
    return fields


def _pubmed_backfill_record(record: NormalizedReference) -> tuple[dict[str, Any], dict[str, Any]]:
    doi = str(record.doi or "").strip().lower()
    has_doi = bool(doi)

    title = str(record.title or "").strip()
    if not has_doi and not title:
        return {}, {"provider": "pubmed_backfill", "status": "skipped", "reason": "missing_doi_and_title", "records": 0}

    term = ""
    retmax = 1
    if has_doi:
        term = f"{doi}[AID] OR {doi}[DOI]"
        retmax = 1
    else:
        author_token = _first_author_token(record)
        title_clause = f'"{title}"[Title]'
        term_parts = [title_clause]
        if isinstance(record.year, int):
            term_parts.append(f"{record.year}[DP]")
        if author_token:
            term_parts.append(f"{author_token}[Author]")
        term = " AND ".join(term_parts)
        retmax = 5

    search_query = urllib.parse.urlencode({"db": "pubmed", "retmode": "json", "retmax": retmax, "term": term})
    search_payload, search_error = _request_json(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_query}",
    )
    if search_error:
        return {}, {"provider": "pubmed_backfill", "status": "error", "reason": search_error, "records": 0}
    if not isinstance(search_payload, dict):
        return {}, {"provider": "pubmed_backfill", "status": "error", "reason": "invalid_payload", "records": 0}

    search_result = search_payload.get("esearchresult")
    if not isinstance(search_result, dict):
        return {}, {"provider": "pubmed_backfill", "status": "error", "reason": "missing_esearchresult", "records": 0}

    idlist = search_result.get("idlist")
    if not isinstance(idlist, list) or not idlist:
        return {}, {"provider": "pubmed_backfill", "status": "ok", "reason": "no_match", "records": 0}

    ids = [str(pmid).strip() for pmid in idlist if str(pmid).strip()]
    if not ids:
        return {}, {"provider": "pubmed_backfill", "status": "ok", "reason": "no_match", "records": 0}

    summary_query = urllib.parse.urlencode({"db": "pubmed", "retmode": "json", "id": ",".join(ids)})
    summary_payload, summary_error = _request_json(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{summary_query}",
    )
    if summary_error:
        return {}, {"provider": "pubmed_backfill", "status": "error", "reason": summary_error, "records": 0}
    if not isinstance(summary_payload, dict):
        return {}, {"provider": "pubmed_backfill", "status": "error", "reason": "invalid_summary_payload", "records": 0}

    summary_result = summary_payload.get("result")
    if not isinstance(summary_result, dict):
        return {}, {"provider": "pubmed_backfill", "status": "error", "reason": "missing_summary_result", "records": 0}

    chosen: NormalizedReference | None = None
    for pmid in ids:
        item = summary_result.get(pmid)
        if not isinstance(item, dict):
            continue
        normalized = _normalize_pubmed_summary_item(item, confidence=record.confidence)
        if has_doi:
            chosen = normalized
            break
        if _title_year_match_is_strong(record, normalized):
            chosen = normalized
            break

    if chosen is None:
        if has_doi:
            return {}, {"provider": "pubmed_backfill", "status": "ok", "reason": "no_summary_item", "records": 0}
        return {}, {"provider": "pubmed_backfill", "status": "ok", "reason": "uncertain_title_match", "records": 0}

    fields = _backfill_fields_from_reference(chosen)
    reason = "filled" if fields else "empty_record"
    return fields, {
        "provider": "pubmed_backfill",
        "status": "ok",
        "reason": reason,
        "records": 1 if fields else 0,
    }


def _openalex_backfill_record(record: NormalizedReference) -> tuple[dict[str, Any], dict[str, Any]]:
    doi = str(record.doi or "").strip().lower()
    title = str(record.title or "").strip()
    if not doi and not title:
        return {}, {"provider": "openalex_backfill", "status": "skipped", "reason": "missing_doi_and_title", "records": 0}

    candidates: list[NormalizedReference] = []
    if doi:
        doi_url = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
        payload, error = _request_json(f"https://api.openalex.org/works/{doi_url}")
        if error:
            return {}, {"provider": "openalex_backfill", "status": "error", "reason": error, "records": 0}
        if not isinstance(payload, dict):
            return {}, {"provider": "openalex_backfill", "status": "error", "reason": "invalid_payload", "records": 0}
        candidates.append(_normalize_openalex_item(payload, confidence=record.confidence))
    else:
        query = urllib.parse.urlencode({"search": title, "per-page": 5})
        payload, error = _request_json(f"https://api.openalex.org/works?{query}")
        if error:
            return {}, {"provider": "openalex_backfill", "status": "error", "reason": error, "records": 0}
        if not isinstance(payload, dict):
            return {}, {"provider": "openalex_backfill", "status": "error", "reason": "invalid_payload", "records": 0}
        results = payload.get("results")
        if isinstance(results, list):
            candidates = [_normalize_openalex_item(item, confidence=record.confidence) for item in results if isinstance(item, dict)]

    chosen: NormalizedReference | None = None
    if doi and candidates:
        chosen = candidates[0]
    elif candidates:
        for candidate in candidates:
            if _title_year_match_is_strong(record, candidate):
                chosen = candidate
                break

    if chosen is None:
        reason = "uncertain_title_match" if not doi else "no_match"
        return {}, {"provider": "openalex_backfill", "status": "ok", "reason": reason, "records": 0}

    fields = _backfill_fields_from_reference(chosen)
    reason = "filled" if fields else "empty_record"
    return fields, {"provider": "openalex_backfill", "status": "ok", "reason": reason, "records": 1 if fields else 0}


def _crossref_backfill_record(record: NormalizedReference) -> tuple[dict[str, Any], dict[str, Any]]:
    doi = str(record.doi or "").strip().lower()
    title = str(record.title or "").strip()
    if not doi and not title:
        return {}, {"provider": "crossref_backfill", "status": "skipped", "reason": "missing_doi_and_title", "records": 0}

    candidates: list[NormalizedReference] = []
    if doi:
        encoded = urllib.parse.quote(doi, safe="")
        payload, error = _request_json(f"https://api.crossref.org/works/{encoded}")
        if error:
            return {}, {"provider": "crossref_backfill", "status": "error", "reason": error, "records": 0}
        if not isinstance(payload, dict):
            return {}, {"provider": "crossref_backfill", "status": "error", "reason": "invalid_payload", "records": 0}
        message = payload.get("message")
        if not isinstance(message, dict):
            return {}, {"provider": "crossref_backfill", "status": "error", "reason": "missing_message", "records": 0}
        candidates.append(_normalize_crossref_item(message, confidence=record.confidence))
    else:
        query = urllib.parse.urlencode({"query.title": title, "rows": 5})
        payload, error = _request_json(f"https://api.crossref.org/works?{query}")
        if error:
            return {}, {"provider": "crossref_backfill", "status": "error", "reason": error, "records": 0}
        if not isinstance(payload, dict):
            return {}, {"provider": "crossref_backfill", "status": "error", "reason": "invalid_payload", "records": 0}
        message = payload.get("message")
        if not isinstance(message, dict):
            return {}, {"provider": "crossref_backfill", "status": "error", "reason": "missing_message", "records": 0}
        items = message.get("items")
        if isinstance(items, list):
            candidates = [_normalize_crossref_item(item, confidence=record.confidence) for item in items if isinstance(item, dict)]

    chosen: NormalizedReference | None = None
    if doi and candidates:
        chosen = candidates[0]
    elif candidates:
        for candidate in candidates:
            if _title_year_match_is_strong(record, candidate):
                chosen = candidate
                break

    if chosen is None:
        reason = "uncertain_title_match" if not doi else "no_match"
        return {}, {"provider": "crossref_backfill", "status": "ok", "reason": reason, "records": 0}

    fields = _backfill_fields_from_reference(chosen)
    reason = "filled" if fields else "empty_record"
    return fields, {"provider": "crossref_backfill", "status": "ok", "reason": reason, "records": 1 if fields else 0}


def _normalize_openalex_item(item: dict[str, Any], confidence: float) -> NormalizedReference:
    title = str(item.get("display_name") or "").strip()
    authors: list[str] = []
    authorships = item.get("authorships")
    if isinstance(authorships, list):
        for entry in authorships:
            if isinstance(entry, dict):
                author_obj = entry.get("author")
                if isinstance(author_obj, dict):
                    name = str(author_obj.get("display_name") or "").strip()
                    if name:
                        authors.append(name)

    year_raw = item.get("publication_year")
    year = year_raw if isinstance(year_raw, int) else None

    doi = None
    ids = item.get("ids")
    if isinstance(ids, dict):
        doi_url = str(ids.get("doi") or "").strip()
        if doi_url:
            doi = doi_url.removeprefix("https://doi.org/").lower()

    pmid = None
    if isinstance(ids, dict):
        pmid_url = str(ids.get("pmid") or "").strip()
        if pmid_url:
            pmid = pmid_url.rstrip("/").split("/")[-1] or None

    venue = ""
    primary_location = item.get("primary_location")
    if isinstance(primary_location, dict):
        source_obj = primary_location.get("source")
        if isinstance(source_obj, dict):
            venue = str(source_obj.get("display_name") or "").strip()

    url = None
    if isinstance(primary_location, dict):
        url = str(primary_location.get("landing_page_url") or "").strip() or None

    return NormalizedReference(
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        pmid=pmid,
        arxiv=None,
        venue=venue,
        url=url,
        source="openalex",
        confidence=confidence,
    )


def _normalize_crossref_reference_entry(ref: dict[str, Any], confidence: float = 0.85) -> NormalizedReference:
    """Normalize a Crossref 'reference' entry (bibliography) into NormalizedReference.

    This is best-effort: Crossref reference items can be inconsistent. We parse common fields
    and fall back to unstructured text where necessary.
    """
    # Crossref reference entries may have keys like "article-title", "DOI", "author",
    # "year", "journal-title", "unstructured", "article-title", "volume", etc.
    def _split_unstructured_sentences(text: str) -> list[str]:
        raw = re.sub(r"\s+", " ", text).strip()
        if not raw:
            return []
        out: list[str] = []
        start = 0
        n = len(raw)
        idx = 0
        while idx < n:
            ch = raw[idx]
            if ch in ".!?":
                token_start = idx - 1
                while token_start >= 0 and raw[token_start].isalpha():
                    token_start -= 1
                token = raw[token_start + 1:idx]
                next_idx = idx + 1
                while next_idx < n and raw[next_idx].isspace():
                    next_idx += 1
                if next_idx < n:
                    next_ch = raw[next_idx]
                    is_initial = len(token) == 1 and token.isupper()
                    if not is_initial and (next_ch.isupper() or next_ch.isdigit()):
                        sent = raw[start:idx + 1].strip()
                        if sent:
                            out.append(sent)
                        start = next_idx
                        idx = next_idx
                        continue
            idx += 1
        tail = raw[start:].strip()
        if tail:
            out.append(tail)
        return out

    def _contains_long_word_sequence(text: str) -> bool:
        words = [w for w in re.findall(r"[A-Za-z]+", text)]
        run = 0
        for w in words:
            if len(w) >= 4 and not (w[0].isupper() and w[1:].islower()):
                run += 1
                if run >= 4:
                    return True
            else:
                run = 0
        return False

    def _is_probable_author_block(text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        has_marker = bool(
            re.search(r"\bet\s+al\.?\b", t, re.IGNORECASE)
            or re.search(r",\s*[A-Z](?:\.|\s)", t)
            or re.search(r"\b(?:and|&)\b", t)
            or re.search(r";", t)
            or AUTHOR_SURNAME_INITIAL_RE.search(t)
        )
        if not has_marker:
            return False
        if _contains_long_word_sequence(t):
            return False
        return True

    def _extract_unstructured_title(unstructured: str) -> str | None:
        sentences = _split_unstructured_sentences(unstructured)
        if len(sentences) < 2:
            return None
        first = sentences[0].strip()
        second = sentences[1].strip(" \t-:;,.()")
        if not _is_probable_author_block(first):
            return None
        if len(re.findall(r"[A-Za-z]", second)) < 5:
            return None
        if len(second.split()) < 5:
            return None
        if SURNAME_INITIAL_START_RE.match(second):
            return None
        return second or None

    def _extract_unstructured_authors(unstructured: str) -> list[str]:
        sentences = _split_unstructured_sentences(unstructured)
        if not sentences:
            return []
        first = sentences[0].strip()
        if not _is_probable_author_block(first):
            return []

        matches = [m.strip(" ,.;") for m in AUTHOR_SURNAME_INITIAL_RE.findall(first)]
        cleaned = [m for m in matches if SURNAME_INITIAL_START_RE.match(m)]
        # Conservative: only return when we have at least one clearly parsed name.
        return cleaned

    title = ""
    unstructured_value = ""
    if isinstance(ref.get("unstructured"), str) and ref.get("unstructured"):
        unstructured_value = str(ref.get("unstructured")).strip()

    has_structured_title = False
    if isinstance(ref.get("article-title"), str) and ref.get("article-title"):
        title = str(ref.get("article-title")).strip()
        has_structured_title = bool(title)
    elif isinstance(ref.get("title"), str) and ref.get("title"):
        title = str(ref.get("title")).strip()
        has_structured_title = bool(title)
    elif unstructured_value:
        # unstructured often contains the full reference text
        title = unstructured_value

    if (not has_structured_title) and unstructured_value:
        parsed_title = _extract_unstructured_title(unstructured_value)
        if parsed_title:
            title = parsed_title

    authors: list[str] = []
    author_raw = ref.get("author") or ref.get("authors")
    if isinstance(author_raw, str):
        # comma-separated names
        parts = [p.strip() for p in re.split(r"[,;]", author_raw) if p.strip()]
        authors = parts
    elif isinstance(author_raw, list):
        for a in author_raw:
            if isinstance(a, str):
                authors.append(a.strip())
            elif isinstance(a, dict):
                # sometimes author dicts have 'given'/'family'
                authors.append(_normalize_author_name(a))

    if not authors and unstructured_value:
        parsed_authors = _extract_unstructured_authors(unstructured_value)
        if parsed_authors:
            authors = parsed_authors

    year = None
    if "year" in ref and isinstance(ref.get("year"), (int, str)):
        yr = ref.get("year")
        try:
            year = int(str(yr))
        except Exception:
            year = None

    doi = None
    if ref.get("DOI"):
        doi = str(ref.get("DOI")).lower().strip()
    elif ref.get("doi"):
        doi = str(ref.get("doi")).lower().strip()

    venue = ""
    if ref.get("journal-title"):
        venue = str(ref.get("journal-title")).strip()
    elif ref.get("container-title"):
        venue = str(ref.get("container-title")).strip()

    url = None
    if ref.get("URL"):
        url = str(ref.get("URL")).strip()

    # If title is still empty, attempt to use 'article-title' or 'unstructured'
    if not title:
        title = str(ref.get("unstructured") or "").strip()

    # Sanitize title: remove leading author-like prefixes that sometimes are
    # embedded in Crossref unstructured reference strings, e.g.
    # "Davies, M. R. et al. Some interesting title..."
    if isinstance(title, str) and title:
        def _strip_author_prefix(s: str) -> str:
            # Try to split on a common separator between author list and title
            m = re.match(r"^(.+?)\s*[\-\—:\;]\s*(.+)$", s)
            if not m:
                return s
            prefix, rest = m.group(1).strip(), m.group(2).strip()
            # remove trailing comma/period from prefix
            prefix = prefix.rstrip(' ,.')
            # split author-like tokens by commas/semicolons/and/&
            tokens = re.split(r"\s*(?:,|;|\band\b|&)\s*", prefix)
            if not tokens:
                return s
            # Require some author-like separators or initials present to consider stripping.
            # This avoids stripping legitimate title starts like "DNA-binding proteins: ..."
            if not re.search(r"[\.,;\&]|\band\b|\.", prefix):
                return s
            # heuristics for tokens: either initials (e.g. 'M.' or 'M. R.') or
            # capitalized surname with at least one lowercase (avoid ALLCAPS)
            def is_initials(tok: str) -> bool:
                return bool(re.fullmatch(r"(?:[A-Z]\.){1,4}(?:\s*[A-Z]\.)*", tok.strip()))

            def is_cap_name(tok: str) -> bool:
                return bool(re.fullmatch(r"[A-Z][a-z][A-Za-z'`-]*?(?:\s+[A-Z][a-z][A-Za-z'`-]*?)*", tok.strip()))

            def is_initials_with_surname(tok: str) -> bool:
                # e.g. 'M. R. Davies' or 'M.R. Davies'
                return bool(re.fullmatch(r"(?:[A-Z]\.(?:\s*[A-Z]\.)*)(?:\s+[A-Z][a-z][A-Za-z'`-]*)", tok.strip()))

            matches = 0
            for tok in tokens:
                if is_initials(tok) or is_cap_name(tok) or is_initials_with_surname(tok):
                    matches += 1
            # Require at least one token matching author pattern and reasonable count
            if matches >= 1 and len(tokens) <= 6:
                # allow trailing 'et al.' in prefix; if present, strip it
                if re.search(r"et\s+al\.?$", prefix, re.IGNORECASE):
                    return rest
                # also allow a trailing year in prefix
                if re.search(r"\b\d{4}$", prefix):
                    return rest
                # otherwise, only strip when prefix tokens mostly look like authors
                if matches >= 1 and matches >= len(tokens) / 2:
                    return rest
            return s

        cleaned = _strip_author_prefix(title)
        # Only accept the cleaned title if it shortens the prefix and leaves
        # a reasonable title body (>= 3 words). Additional safeguards:
        # - cleaned title must start with uppercase letter or digit
        # - avoid cleaned starts that are obvious conjunction/fragments like
        #   'and', 'or', 'term', 'row' which indicate we truncated too far
        def _clean_accept(cand: str, orig: str) -> bool:
            if not cand or len(cand.split()) < 3:
                return False
            if len(cand) >= len(orig):
                return False
            first = cand.strip()[0]
            # accept if starts with uppercase letter or digit
            if not (first.isupper() or first.isdigit()):
                return False
            low_start = cand.strip().lower().split()[0]
            if low_start in {"and", "or", "term", "row", "the", "a", "an"}:
                return False
            return True

        if _clean_accept(cleaned, title):
            title = cleaned

        def _extract_second_pass_title(s: str) -> str | None:
            candidate = s.strip()
            if not candidate:
                return None
            if not (SURNAME_INITIAL_START_RE.match(candidate) or AUTHOR_SURNAME_INITIAL_RE.match(candidate)):
                return None
            segments = [seg.strip(" \t-:;,.()") for seg in re.split(r"\.\s+", candidate) if seg.strip()]
            if len(segments) < 2:
                return None
            first = segments[0]
            second = segments[1]
            if not _is_probable_author_block(first):
                return None
            if len(second.split()) < 5:
                return None
            if len(re.findall(r"[A-Za-z]", second)) < 5:
                return None
            if SURNAME_INITIAL_START_RE.match(second):
                return None
            if _is_probable_venue_fragment(second):
                return None
            return second or None

        second_pass_title = _extract_second_pass_title(title)
        if second_pass_title:
            title = second_pass_title

    return NormalizedReference(
        title=title,
        authors=authors,
        year=year,
        doi=doi or None,
        pmid=None,
        arxiv=None,
        venue=venue,
        url=url,
        source="crossref",
        confidence=confidence,
    )


def _fetch_pubmed(identity: dict[str, Any]) -> tuple[list[NormalizedReference], dict[str, Any]]:
    doi = str(identity.get("doi") or "").strip()
    title = str(identity.get("title") or "").strip()

    if not doi and not title:
        return [], {"provider": "pubmed", "status": "skipped", "reason": "missing_doi_and_title", "records": 0}

    # Resolve the seed PubMed record first (prefer DOI, fallback to title).
    term = f"{doi}[AID] OR {doi}[DOI]" if doi else title
    query = urllib.parse.urlencode({"db": "pubmed", "retmode": "json", "retmax": 5, "term": term})
    search_payload, search_error = _request_json(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{query}",
    )
    if search_error:
        return [], {"provider": "pubmed", "status": "error", "reason": search_error, "records": 0}
    if not isinstance(search_payload, dict):
        return [], {"provider": "pubmed", "status": "error", "reason": "invalid_payload", "records": 0}

    result = search_payload.get("esearchresult")
    if not isinstance(result, dict):
        return [], {"provider": "pubmed", "status": "error", "reason": "missing_esearchresult", "records": 0}
    ids = result.get("idlist")
    if not isinstance(ids, list) or not ids:
        return [], {"provider": "pubmed", "status": "ok", "reason": "no_match", "records": 0}

    seed_ids = [str(i).strip() for i in ids if str(i).strip()]
    id_csv = ",".join(seed_ids)
    if not id_csv:
        return [], {"provider": "pubmed", "status": "ok", "reason": "no_match", "records": 0}

    # Try to fetch bibliography references via NCBI ELink from the primary PMID.
    primary_id = seed_ids[0]
    elink_query = urllib.parse.urlencode(
        {
            "dbfrom": "pubmed",
            "db": "pubmed",
            "retmode": "json",
            "linkname": "pubmed_pubmed_refs",
            "id": primary_id,
        }
    )
    elink_payload, elink_error = _request_json(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?{elink_query}",
    )

    linked_ids: list[str] = []
    if not elink_error and isinstance(elink_payload, dict):
        linksets = elink_payload.get("linksets")
        if isinstance(linksets, list):
            for linkset in linksets:
                if not isinstance(linkset, dict):
                    continue
                linksetdbs = linkset.get("linksetdbs")
                if not isinstance(linksetdbs, list):
                    continue
                for db in linksetdbs:
                    if not isinstance(db, dict):
                        continue
                    links = db.get("links")
                    if not isinstance(links, list):
                        continue
                    for rid in links:
                        rid_str = str(rid).strip()
                        if rid_str and rid_str not in linked_ids:
                            linked_ids.append(rid_str)

    # Bound PubMed reference fan-out for runtime stability.
    try:
        refs_max = int(os.environ.get("PUBMED_REFS_MAX", "400"))
    except Exception:
        refs_max = 400
    if refs_max < 1:
        refs_max = 1
    if linked_ids:
        linked_ids = linked_ids[:refs_max]

    # Prefer returning referenced bibliography records when available.
    target_ids = linked_ids if linked_ids else seed_ids
    target_csv = ",".join(target_ids)

    summary_query = urllib.parse.urlencode({"db": "pubmed", "retmode": "json", "id": target_csv})
    summary_payload, summary_error = _request_json(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{summary_query}",
    )
    if summary_error:
        return [], {"provider": "pubmed", "status": "error", "reason": summary_error, "records": 0}
    if not isinstance(summary_payload, dict):
        return [], {"provider": "pubmed", "status": "error", "reason": "invalid_summary_payload", "records": 0}

    summary_result = summary_payload.get("result")
    if not isinstance(summary_result, dict):
        return [], {"provider": "pubmed", "status": "error", "reason": "missing_summary_result", "records": 0}

    refs: list[NormalizedReference] = []
    uids = summary_result.get("uids")
    if isinstance(uids, list):
        for uid in uids:
            item = summary_result.get(str(uid))
            if isinstance(item, dict):
                ref = _normalize_pubmed_summary_item(item, confidence=0.78)
                if ref.title or ref.doi or ref.pmid:
                    refs.append(ref)

    reason = "references" if linked_ids else "seed_match"
    return refs, {"provider": "pubmed", "status": "ok", "reason": reason, "records": len(refs)}


def _fetch_openalex(identity: dict[str, Any]) -> tuple[list[NormalizedReference], dict[str, Any]]:
    doi = str(identity.get("doi") or "").strip()
    title = str(identity.get("title") or "").strip()

    if not doi and not title:
        return [], {"provider": "openalex", "status": "skipped", "reason": "missing_doi_and_title", "records": 0}

    payload: dict[str, Any] | list[Any] | None
    error: str | None
    if doi:
        doi_norm = doi.lower()
        doi_url = urllib.parse.quote(f"https://doi.org/{doi_norm}", safe="")
        payload, error = _request_json(f"https://api.openalex.org/works/{doi_url}")
        if error:
            return [], {"provider": "openalex", "status": "error", "reason": error, "records": 0}
        if not isinstance(payload, dict):
            return [], {"provider": "openalex", "status": "error", "reason": "invalid_payload", "records": 0}
        ref = _normalize_openalex_item(payload, confidence=0.82)
        if ref.title or ref.doi or ref.pmid:
            return [ref], {"provider": "openalex", "status": "ok", "reason": None, "records": 1}
        return [], {"provider": "openalex", "status": "ok", "reason": "empty_record", "records": 0}

    query = urllib.parse.urlencode({"search": title, "per-page": 10})
    payload, error = _request_json(f"https://api.openalex.org/works?{query}")
    if error:
        return [], {"provider": "openalex", "status": "error", "reason": error, "records": 0}
    if not isinstance(payload, dict):
        return [], {"provider": "openalex", "status": "error", "reason": "invalid_payload", "records": 0}

    results = payload.get("results")
    if not isinstance(results, list):
        return [], {"provider": "openalex", "status": "ok", "reason": None, "records": 0}
    refs = [_normalize_openalex_item(item, confidence=0.68) for item in results if isinstance(item, dict)]
    refs = [r for r in refs if r.title or r.doi or r.pmid]
    return refs, {"provider": "openalex", "status": "ok", "reason": None, "records": len(refs)}


def _fetch_crossref(identity: dict[str, Any]) -> tuple[list[NormalizedReference], dict[str, Any]]:
    doi = str(identity.get("doi") or "").strip()
    title = str(identity.get("title") or "").strip()

    if not doi and not title:
        return [], {"provider": "crossref", "status": "skipped", "reason": "missing_doi_and_title", "records": 0}

    if doi:
        encoded = urllib.parse.quote(doi, safe="")
        payload, error = _request_json(f"https://api.crossref.org/works/{encoded}")
        if error:
            return [], {"provider": "crossref", "status": "error", "reason": error, "records": 0}
        if not isinstance(payload, dict):
            return [], {"provider": "crossref", "status": "error", "reason": "invalid_payload", "records": 0}
        message = payload.get("message")
        if not isinstance(message, dict):
            return [], {"provider": "crossref", "status": "error", "reason": "missing_message", "records": 0}
        # If Crossref provides a 'reference' list (the paper's bibliography), prefer
        # converting those into NormalizedReference records. Otherwise fall back
        # to returning the paper itself as a single record.
        references_list = message.get("reference") or message.get("references")
        if isinstance(references_list, list) and references_list:
            refs: list[NormalizedReference] = []
            for r in references_list:
                if isinstance(r, dict):
                    try:
                        nr = _normalize_crossref_reference_entry(r, confidence=0.85)
                        # Keep entries that have at least a DOI or a title.
                        if nr.title or nr.doi or nr.pmid or nr.arxiv:
                            refs.append(nr)
                    except Exception:
                        # best-effort: skip problematic items
                        continue

            # If we have refs, attempt bounded hydration for sparse DOI-only entries.
            if refs:
                try:
                    max_hydrates = int(os.environ.get("CROSSREF_HYDRATE_MAX", "80"))
                except Exception:
                    max_hydrates = 80
                hydrate_count = 0
                hydrate_cache: dict[str, tuple[dict[str, Any] | list[Any] | None, str | None]] = {}

                for rec in refs:
                    # Stop hydrating when the budget is exhausted
                    if hydrate_count >= max_hydrates:
                        break
                    try:
                        def _is_noisy_title(t: str) -> bool:
                            if not t:
                                return True
                            # If original matches the author-prefix heuristic, noisy
                            if AUTHOR_PREFIX_RE.search(t):
                                return True
                            if SURNAME_INITIAL_START_RE.match(t):
                                return True
                            low = t.lower().strip()
                            if low.startswith("et al"):
                                return True
                            # very short titles are likely noise
                            if len(t.split()) < 3:
                                return True
                            # leading initials like "R." or "M. R."
                            if re.match(r"^[A-Z](?:\.\s?){1,3}", t):
                                return True
                            return False

                        needs_fields = not (rec.title and rec.authors and rec.year and rec.venue and rec.url)
                        title_is_noisy = _is_noisy_title(rec.title)
                        if rec.doi and (needs_fields or title_is_noisy):
                            doi_norm = str(rec.doi).lower().strip()
                            if doi_norm in hydrate_cache:
                                payload_obj, payload_err = hydrate_cache[doi_norm]
                            else:
                                encoded_ref = urllib.parse.quote(doi_norm, safe="")
                                payload_obj, payload_err = _request_json(f"https://api.crossref.org/works/{encoded_ref}")
                                hydrate_cache[doi_norm] = (payload_obj, payload_err)
                                hydrate_count += 1

                            if payload_err:
                                # hydration failed for this DOI - keep original sparse rec
                                continue
                            if not isinstance(payload_obj, dict):
                                continue
                            message_obj = payload_obj.get("message")
                            if not isinstance(message_obj, dict):
                                continue
                            # normalize hydrated item
                            try:
                                hydrated = _normalize_crossref_item(message_obj, confidence=0.92)
                            except Exception:
                                continue

                            # Merge hydrated fields into rec. Prefer to fill missing
                            # fields, but also allow replacing a noisy title when the
                            # hydrated title appears cleaner.
                            filled = False

                            def _is_clean_hydrated_title(t: str) -> bool:
                                if not t:
                                    return False
                                if SURNAME_INITIAL_START_RE.match(t):
                                    return False
                                return not _is_noisy_title(t)

                            if hydrated.title:
                                if not rec.title:
                                    rec.title = hydrated.title
                                    filled = True
                                else:
                                    try:
                                        if _is_noisy_title(rec.title) and _is_clean_hydrated_title(hydrated.title):
                                            rec.title = hydrated.title
                                            filled = True
                                    except Exception:
                                        pass

                            if (not rec.authors or len(rec.authors) == 0) and hydrated.authors:
                                rec.authors = hydrated.authors
                                filled = True
                            if (not rec.year) and hydrated.year:
                                rec.year = hydrated.year
                                filled = True
                            if (not rec.venue) and hydrated.venue:
                                rec.venue = hydrated.venue
                                filled = True
                            if (not rec.url) and hydrated.url:
                                rec.url = hydrated.url
                                filled = True

                            if filled:
                                try:
                                    rec.confidence = max(rec.confidence, min(hydrated.confidence, rec.confidence + 0.08))
                                except Exception:
                                    pass
                    except Exception:
                        # ensure provider never fails due to hydration
                        continue

                for rec in refs:
                    try:
                        if rec.title:
                            rec.title = _rewrite_author_prefixed_title(rec.title)
                    except Exception:
                        continue

                refs, batch_status = _hydrate_refs_via_openalex_batch(refs, max_batch=200)

                return refs, {"provider": "crossref", "status": "ok", "reason": "references", "records": len(refs)}

        # Fallback: normalize the work item itself (the paper)
        refs = [_normalize_crossref_item(message, confidence=0.93)]
        return refs, {"provider": "crossref", "status": "ok", "reason": "no_references_fallback", "records": len(refs)}

    query = urllib.parse.urlencode({"query.title": title, "rows": 10})
    payload, error = _request_json(f"https://api.crossref.org/works?{query}")
    if error:
        return [], {"provider": "crossref", "status": "error", "reason": error, "records": 0}
    if not isinstance(payload, dict):
        return [], {"provider": "crossref", "status": "error", "reason": "invalid_payload", "records": 0}
    message = payload.get("message")
    if not isinstance(message, dict):
        return [], {"provider": "crossref", "status": "error", "reason": "missing_message", "records": 0}
    items = message.get("items")
    if not isinstance(items, list):
        return [], {"provider": "crossref", "status": "ok", "reason": None, "records": 0}
    refs = [_normalize_crossref_item(item, confidence=0.72) for item in items if isinstance(item, dict)]
    refs = [r for r in refs if r.title or r.doi or r.pmid or r.arxiv]
    return refs, {"provider": "crossref", "status": "ok", "reason": None, "records": len(refs)}


def _hydrate_refs_via_openalex_batch(
    refs: list[NormalizedReference],
    max_batch: int = 200,
) -> tuple[list[NormalizedReference], dict[str, Any]]:
    dois_to_hydrate = [
        r for r in refs
        if r.doi and (not r.title or not r.authors)
    ][:500]
    
    if not dois_to_hydrate:
        return refs, {"provider": "openalex_batch", "status": "skipped", "reason": "no_hydration_needed", "records": 0}
    
    batch_size = 25
    results: list[dict[str, Any]] = []
    
    for i in range(0, len(dois_to_hydrate), batch_size):
        batch = dois_to_hydrate[i:i + batch_size]
        doi_list = [r.doi for r in batch if r.doi]
        
        if not doi_list:
            continue
        
        pipe_dois = "|".join(doi_list)
        url = f"https://api.openalex.org/works?filter=doi:{urllib.parse.quote(pipe_dois, safe='')}"
        
        payload, error = _request_json(url)
        if error:
            continue
        if not isinstance(payload, dict):
            continue
        
        batch_results = payload.get("results", [])
        results.extend(batch_results)
    
    doi_to_result: dict[str, dict[str, Any]] = {}
    for item in results:
        doi = item.get("doi", "")
        if doi:
            doi = doi.lower().replace("https://doi.org/", "")
            doi_to_result[doi] = item
    
    hydrated_count = 0
    for rec in refs:
        if rec.doi:
            doi_key = rec.doi.lower()
            if doi_key in doi_to_result:
                item = doi_to_result[doi_key]
                if not rec.title and item.get("title"):
                    rec.title = item["title"]
                    hydrated_count += 1
                if (not rec.authors or len(rec.authors) == 0):
                    authors = []
                    authorships = item.get("authorships", [])
                    if authorships and isinstance(authorships, list):
                        for a in authorships:
                            if isinstance(a, dict):
                                author_info = a.get("author", {})
                                if isinstance(author_info, dict):
                                    name = author_info.get("display_name") or ""
                                    if name:
                                        authors.append(name)
                    if authors:
                        rec.authors = authors
                        hydrated_count += 1
                if not rec.year and item.get("publication_year"):
                    rec.year = item["publication_year"]
                if not rec.venue:
                    primary = item.get("primary_location", {})
                    source = primary.get("source") or {}
                    if isinstance(source, dict):
                        rec.venue = source.get("display_name") or ""
    
    return refs, {"provider": "openalex_batch", "status": "ok", "reason": None, "records": hydrated_count}


def _normalize_semantic_scholar_item(item: dict[str, Any], confidence: float) -> NormalizedReference:
    authors: list[str] = []
    authors_raw = item["authors"] if "authors" in item else []
    if isinstance(authors_raw, list):
        for author_entry in authors_raw:
            if isinstance(author_entry, dict):
                name_raw = author_entry["name"] if "name" in author_entry else ""
                name = str(name_raw).strip()
                if name:
                    authors.append(name)

    raw_external_ids: Any = item["externalIds"] if "externalIds" in item else None
    doi_raw = ""
    pmid_raw = ""
    arxiv_raw = ""
    if isinstance(raw_external_ids, dict):
        external_ids = cast(dict[str, Any], raw_external_ids)
        doi_raw = str(external_ids["DOI"] if "DOI" in external_ids else "")
        pmid_raw = str(external_ids["PubMed"] if "PubMed" in external_ids else "")
        arxiv_raw = str(external_ids["ArXiv"] if "ArXiv" in external_ids else "")

    doi = doi_raw.strip().lower() or None
    pmid = pmid_raw.strip() or None
    arxiv = arxiv_raw.strip() or None

    year_raw = item["year"] if "year" in item else None
    year_val = year_raw if isinstance(year_raw, int) else None

    title_raw = item["title"] if "title" in item else ""
    venue_raw = item["venue"] if "venue" in item else ""
    url_raw = item["url"] if "url" in item else ""

    return NormalizedReference(
        title=str(title_raw).strip(),
        authors=authors,
        year=year_val,
        doi=doi,
        pmid=pmid,
        arxiv=arxiv,
        venue=str(venue_raw).strip(),
        url=str(url_raw).strip() or None,
        source="semantic_scholar",
        confidence=confidence,
    )


def _fetch_semantic_scholar(identity: dict[str, Any]) -> tuple[list[NormalizedReference], dict[str, Any]]:
    doi = str(identity.get("doi") or "").strip()
    title = str(identity.get("title") or "").strip()
    fields = "title,authors,year,externalIds,venue,url"

    if not doi and not title:
        return [], {
            "provider": "semantic_scholar",
            "status": "skipped",
            "reason": "missing_doi_and_title",
            "records": 0,
        }

    if doi:
        payload, error = _request_json(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi, safe='')}?fields={fields}"
        )
        if error:
            return [], {"provider": "semantic_scholar", "status": "error", "reason": error, "records": 0}
        if not isinstance(payload, dict):
            return [], {"provider": "semantic_scholar", "status": "error", "reason": "invalid_payload", "records": 0}
        ref = _normalize_semantic_scholar_item(payload, confidence=0.91)
        if not ref.title:
            return [], {"provider": "semantic_scholar", "status": "ok", "reason": None, "records": 0}
        return [ref], {"provider": "semantic_scholar", "status": "ok", "reason": None, "records": 1}

    query = urllib.parse.urlencode({"query": title, "limit": 10, "fields": fields})
    payload, error = _request_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{query}")
    if error:
        return [], {"provider": "semantic_scholar", "status": "error", "reason": error, "records": 0}
    if not isinstance(payload, dict):
        return [], {"provider": "semantic_scholar", "status": "error", "reason": "invalid_payload", "records": 0}
    items = payload.get("data")
    if not isinstance(items, list):
        return [], {"provider": "semantic_scholar", "status": "ok", "reason": None, "records": 0}
    refs = [_normalize_semantic_scholar_item(item, confidence=0.7) for item in items if isinstance(item, dict)]
    refs = [r for r in refs if r.title]
    return refs, {"provider": "semantic_scholar", "status": "ok", "reason": None, "records": len(refs)}


def _fetch_arxiv(identity: dict[str, Any]) -> tuple[list[NormalizedReference], dict[str, Any]]:
    arxiv_id = str(identity.get("arxiv") or "").strip()
    if not arxiv_id:
        return [], {"provider": "arxiv", "status": "skipped", "reason": "missing_arxiv_id", "records": 0}

    query = urllib.parse.urlencode({"id_list": arxiv_id})
    xml_text, error = _request_text(f"https://export.arxiv.org/api/query?{query}")
    if error:
        return [], {"provider": "arxiv", "status": "error", "reason": error, "records": 0}
    if not xml_text:
        return [], {"provider": "arxiv", "status": "error", "reason": "empty_payload", "records": 0}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], {"provider": "arxiv", "status": "error", "reason": "xml_parse_error", "records": 0}

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    entries = root.findall("atom:entry", ns)
    refs: list[NormalizedReference] = []
    for entry in entries:
        title = str(entry.findtext("atom:title", default="", namespaces=ns)).strip()
        authors = [
            str(name.text or "").strip()
            for name in entry.findall("atom:author/atom:name", ns)
            if isinstance(name.text, str) and name.text.strip()
        ]
        published = str(entry.findtext("atom:published", default="", namespaces=ns)).strip()
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
        url = str(entry.findtext("atom:id", default="", namespaces=ns)).strip() or None
        doi = str(entry.findtext("arxiv:doi", default="", namespaces=ns)).strip().lower() or None
        venue = str(entry.findtext("arxiv:journal_ref", default="", namespaces=ns)).strip()
        refs.append(
            NormalizedReference(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                pmid=None,
                arxiv=arxiv_id,
                venue=venue,
                url=url,
                source="arxiv",
                confidence=0.92,
            )
        )

    refs = [r for r in refs if r.title]
    return refs, {"provider": "arxiv", "status": "ok", "reason": None, "records": len(refs)}


def _normalized_title_key(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return " ".join(cleaned.split())


def _dedupe_references(records: list[NormalizedReference]) -> list[NormalizedReference]:
    merged: dict[str, NormalizedReference] = {}
    order: list[str] = []

    for record in records:
        key = ""
        if record.doi:
            key = f"doi:{record.doi.lower()}"
        elif record.pmid:
            key = f"pmid:{record.pmid}"
        elif record.arxiv:
            key = f"arxiv:{record.arxiv.lower()}"
        else:
            key = f"title:{_normalized_title_key(record.title)}|year:{record.year or ''}"
        if key not in merged:
            merged[key] = record
            order.append(key)
            continue
        if record.confidence > merged[key].confidence:
            merged[key] = record

    deduped = [merged[k] for k in order]
    return sorted(deduped, key=lambda r: (-r.confidence, r.title.lower(), r.source))


def _reference_key(record: NormalizedReference) -> str:
    if record.doi:
        return f"doi:{record.doi.lower()}"
    if record.pmid:
        return f"pmid:{record.pmid}"
    if record.arxiv:
        return f"arxiv:{record.arxiv.lower()}"
    return f"title:{_normalized_title_key(record.title)}|year:{record.year or ''}"


def _field_is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def _reference_filled_fields(record: NormalizedReference) -> list[str]:
    field_names = ["title", "authors", "year", "doi", "pmid", "arxiv", "venue", "url"]
    return [name for name in field_names if _field_is_filled(getattr(record, name, None))]


def collect_api_references(doc_identity: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provider_chain: list[tuple[str, Any]] = [
        ("pubmed", _fetch_pubmed),
        ("crossref", _fetch_crossref),
        ("openalex", _fetch_openalex),
    ]
    if str(doc_identity.get("arxiv") or "").strip():
        provider_chain.append(("arxiv", _fetch_arxiv))

    collected: list[tuple[str, NormalizedReference]] = []
    statuses: list[dict[str, Any]] = []

    for provider_name, provider_call in provider_chain:
        try:
            refs, status = provider_call(doc_identity)
            for ref in refs:
                collected.append((provider_name, ref))
            if isinstance(status, dict):
                statuses.append({**status, "provider": provider_name})
            else:
                statuses.append({"provider": provider_name, "status": "error", "reason": "invalid_status", "records": 0})
        except Exception as e:
            statuses.append({
                "provider": provider_name,
                "status": "error",
                "reason": f"provider_exception:{type(e).__name__}",
                "records": 0,
            })

    merged: dict[str, NormalizedReference] = {}
    trace_chain: dict[str, list[str]] = {}
    for provider_name, rec in collected:
        key = _reference_key(rec)
        if key not in merged:
            rec.source_chain = [provider_name]
            rec.filled_fields = _reference_filled_fields(rec)
            merged[key] = rec
            trace_chain[key] = [provider_name]
            continue

        existing = merged[key]
        chain = trace_chain[key]
        if provider_name not in chain:
            chain.append(provider_name)

        for field_name in ["title", "authors", "year", "doi", "pmid", "arxiv", "venue", "url"]:
            current_val = getattr(existing, field_name)
            incoming_val = getattr(rec, field_name)
            if not _field_is_filled(current_val) and _field_is_filled(incoming_val):
                setattr(existing, field_name, incoming_val)

        if rec.confidence > existing.confidence:
            existing.source = rec.source
            existing.confidence = rec.confidence

        existing.source_chain = chain
        existing.filled_fields = _reference_filled_fields(existing)

    fill_fields = ["title", "authors", "year", "venue", "url", "doi", "pmid"]
    sparse_check_fields = ["authors", "year", "venue", "url"]
    sparse_count = sum(
        1
        for rec in merged.values()
        if any(not _field_is_filled(getattr(rec, field_name, None)) for field_name in sparse_check_fields)
    )

    has_doi_based_references = False
    for status in statuses:
        if status.get("provider") == "crossref" and status.get("reason") == "references":
            if status.get("records", 0) > 0:
                has_doi_based_references = True
                break

    backfill_max = 180
    env_max_raw = os.environ.get("REFERENCE_BACKFILL_MAX")
    if env_max_raw is None:
        env_max_raw = os.environ.get("PUBMED_BACKFILL_MAX")

    if env_max_raw is not None:
        try:
            backfill_max = int(str(env_max_raw))
        except Exception:
            backfill_max = 180
    else:
        if has_doi_based_references:
            backfill_max = 0
        else:
            backfill_max = max(180, min(1000, sparse_count + 140))

    if backfill_max < 0:
        backfill_max = 0
    if backfill_max > 1000:
        backfill_max = 1000

    total_attempts = 0
    provider_stats: dict[str, dict[str, int]] = {
        "pubmed_backfill": {"attempted": 0, "records": 0, "filled_fields": 0},
        "openalex_backfill": {"attempted": 0, "records": 0, "filled_fields": 0},
        "crossref_backfill": {"attempted": 0, "records": 0, "filled_fields": 0},
    }
    backfill_cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def _cache_key(provider_tag: str, record: NormalizedReference) -> str:
        doi = str(record.doi or "").strip().lower()
        if doi:
            return f"{provider_tag}:doi:{doi}"
        title = _normalized_title_key(record.title)
        year = str(record.year) if isinstance(record.year, int) else ""
        author_token = _first_author_token(record).lower()
        return f"{provider_tag}:title:{title}|year:{year}|author:{author_token}"

    backfill_chain: list[tuple[str, Any]] = [
        ("pubmed_backfill", _pubmed_backfill_record),
        ("openalex_backfill", _openalex_backfill_record),
        ("crossref_backfill", _crossref_backfill_record),
    ]

    for rec in merged.values():
        if not any(not _field_is_filled(getattr(rec, field_name, None)) for field_name in sparse_check_fields):
            continue

        for provider_tag, backfill_call in backfill_chain:
            if total_attempts >= backfill_max:
                break
            if not any(not _field_is_filled(getattr(rec, field_name, None)) for field_name in sparse_check_fields):
                break

            key = _cache_key(provider_tag, rec)
            if key in backfill_cache:
                backfill_fields, _ = backfill_cache[key]
            else:
                provider_stats[provider_tag]["attempted"] += 1
                total_attempts += 1
                try:
                    backfill_fields, backfill_status = backfill_call(rec)
                except Exception as e:
                    backfill_fields, backfill_status = {}, {
                        "provider": provider_tag,
                        "status": "error",
                        "reason": f"provider_exception:{type(e).__name__}",
                        "records": 0,
                    }
                backfill_cache[key] = (backfill_fields, backfill_status)

            newly_filled: list[str] = []
            for field_name in fill_fields:
                current_val = getattr(rec, field_name)
                incoming_val = backfill_fields.get(field_name)
                if not _field_is_filled(current_val) and _field_is_filled(incoming_val):
                    setattr(rec, field_name, incoming_val)
                    newly_filled.append(field_name)

            if newly_filled:
                if provider_tag not in rec.source_chain:
                    rec.source_chain.append(provider_tag)
                for field_name in newly_filled:
                    if field_name not in rec.filled_fields:
                        rec.filled_fields.append(field_name)
                provider_stats[provider_tag]["records"] += 1
                provider_stats[provider_tag]["filled_fields"] += len(newly_filled)

        if total_attempts >= backfill_max:
            break

    for provider_tag in ["pubmed_backfill", "openalex_backfill", "crossref_backfill"]:
        statuses.append(
            {
                "provider": provider_tag,
                "status": "ok",
                "reason": None,
                "records": provider_stats[provider_tag]["records"],
                "attempted": provider_stats[provider_tag]["attempted"],
                "filled_fields": provider_stats[provider_tag]["filled_fields"],
                "cache_size": len([k for k in backfill_cache if k.startswith(f"{provider_tag}:")]),
                "max": backfill_max,
            }
        )

    deduped = sorted(merged.values(), key=lambda r: (-r.confidence, r.title.lower(), r.source))
    normalized = [record.to_record() for record in deduped]
    return normalized, statuses
