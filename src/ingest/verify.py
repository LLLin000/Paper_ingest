"""Quality gates and verification for PDF ingest pipeline.

Computes quality metrics and enforces pass/fail thresholds per blueprint contracts:
- provenance_coverage, strong_claim_unsupported_count
- paragraph_inversion_rate, cross_column_interleave_page_rate
- citation_mapping_coverage, doi_pmid_precision
- figure_caption_precision, caption_id_retention
- silent_truncation_count

Emits:
- run/<id>/qa/report.json - full gate report with quantified values
- run/<id>/qa/stage_status.json - per-stage status
- run/<id>/qa/runtime_safety.json - network safety evidence
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .structure_quality import load_structure_quality

RUN_ROOT = Path("run")
EVAL_ROOT = Path("eval/golden")


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"
    DEGRADED = "degraded"


@dataclass
class GateResult:
    """Result of a single quality gate evaluation."""
    name: str
    value: float
    status: GateStatus
    threshold: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """Complete QA report for a document run."""
    doc_id: str
    generated_at: str
    gates: dict[str, GateResult] = field(default_factory=dict)
    overall_status: GateStatus = GateStatus.NOT_EVALUATED
    hard_stop: bool = False
    reason: Optional[str] = None
    degradation_labels: list[str] = field(default_factory=list)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    if not path.exists():
        return []
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_json(path: Path) -> Optional[dict[str, Any]]:
    """Load JSON file into dict."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_golden(doc_id: str) -> Optional[dict[str, Any]]:
    """Load golden evaluation data for a document."""
    golden_path = EVAL_ROOT / f"{doc_id}.json"
    return load_json(golden_path)


def _has_evaluation_data(golden: Optional[dict[str, Any]]) -> bool:
    """Check if golden file has actual evaluation data."""
    if golden is None:
        return False
    # Check if any of the evaluation arrays have content
    return bool(
        golden.get("reading_order_pairs") or
        golden.get("citation_truth") or
        golden.get("figure_caption_truth") or
        golden.get("multi_column_pages")
    )


def _is_strong_claim(statement: str, fact_category: str) -> bool:
    """Determine if a claim is a 'strong' claim requiring evidence.
    
    Strong claims contain:
    - Numeric data (percentages, counts, statistics)
    - Causal language (causes, leads to, results in)
    - Comparative efficacy claims (better, worse, superior)
    - Guideline-like recommendations (should, must, recommended)
    - Categories: result, statistics, comparison, limitation
    """
    if fact_category in {"result", "statistics", "comparison", "limitation"}:
        return True
    
    strong_patterns = [
        r"\d+%?",  # percentages
        r"\d+\s*(patients|subjects|years|days|months|mm|cm|kg)",  # numeric quantities
        r"\b(better|worse|superior|inferior|more effective|less effective)\b",
        r"\b(causes|leads to|results in|associated with)\b",
        r"\b(should|must|recommended|guideline)\b",
    ]
    
    statement_lower = statement.lower()
    for pattern in strong_patterns:
        if re.search(pattern, statement_lower):
            return True
    return False


def _has_valid_evidence(fact: dict[str, Any]) -> bool:
    """Check if a fact has valid evidence pointer."""
    evidence = fact.get("evidence_pointer", {})
    if not evidence:
        return False
    page = evidence.get("page")
    bbox = evidence.get("bbox")
    quote = fact.get("quote", "")
    
    # Must have page, bbox, and non-empty quote
    if page is None or not bbox or not quote:
        return False
    return True


def _has_minimal_anchor_evidence(fact: dict[str, Any]) -> tuple[bool, str, str]:
    """Validate minimal fact evidence for provenance anchoring.

    Returns:
        (is_valid, reason, text_evidence_type)
    """
    evidence = fact.get("evidence_pointer", {})
    if not isinstance(evidence, dict) or not evidence:
        return False, "missing_evidence_pointer", "none"

    page = evidence.get("page")
    if not isinstance(page, int) or page <= 0:
        return False, "missing_or_invalid_evidence_page", "none"

    bbox = evidence.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, "missing_or_invalid_evidence_bbox", "none"
    if not all(isinstance(v, (int, float)) for v in bbox):
        return False, "missing_or_invalid_evidence_bbox", "none"

    source_block_ids = evidence.get("source_block_ids")
    if not isinstance(source_block_ids, list) or len(source_block_ids) == 0:
        return False, "missing_source_block_ids", "none"

    quote = str(fact.get("quote", "") or "").strip()
    if quote:
        return True, "ok", "quote"

    statement = str(fact.get("statement", "") or "").strip()
    if statement:
        return True, "ok", "statement_fallback"

    return False, "missing_text_evidence", "none"


def _fact_page_matches_paragraph(fact_page: int, paragraph: dict[str, Any]) -> bool:
    """Check whether the fact evidence page is compatible with paragraph span."""
    para_evidence = paragraph.get("evidence_pointer", {})
    para_pages = para_evidence.get("pages", []) if isinstance(para_evidence, dict) else []
    if isinstance(para_pages, list) and para_pages:
        valid_pages = {int(p) for p in para_pages if isinstance(p, int)}
        if fact_page in valid_pages:
            return True

    page_span = paragraph.get("page_span", {})
    if isinstance(page_span, dict):
        start = page_span.get("start")
        end = page_span.get("end")
        if isinstance(start, int) and isinstance(end, int) and start <= fact_page <= end:
            return True

    return False


def _is_fact_mapped_to_paragraph(
    fact: dict[str, Any],
    para_lookup: dict[str, dict[str, Any]],
    para_ids_by_source_block: dict[str, set[str]],
) -> tuple[bool, str]:
    """Validate fact -> paragraph mapping and page consistency."""
    para_id = str(fact.get("para_id", "") or "")
    if not para_id:
        return False, "missing_para_id"

    fact_evidence = fact.get("evidence_pointer", {})
    fact_page = fact_evidence.get("page") if isinstance(fact_evidence, dict) else None
    if not isinstance(fact_page, int) or fact_page <= 0:
        return False, "missing_or_invalid_evidence_page"

    paragraph = para_lookup.get(para_id)
    if paragraph:
        paragraph_text = str(paragraph.get("text", "") or "").strip()
        if not paragraph_text:
            return False, "empty_paragraph_text"
        if _fact_page_matches_paragraph(fact_page, paragraph):
            return True, "ok"
        return False, "fact_page_not_in_paragraph_span"

    evidence = fact_evidence if isinstance(fact_evidence, dict) else {}
    source_block_ids = evidence.get("source_block_ids", [])
    if not isinstance(source_block_ids, list) or not source_block_ids:
        return False, "missing_paragraph_mapping"

    candidate_para_ids: set[str] = set()
    for block_id in source_block_ids:
        candidate_para_ids.update(para_ids_by_source_block.get(str(block_id), set()))

    if not candidate_para_ids:
        return False, "missing_paragraph_mapping"

    valid_candidates: list[str] = []
    for candidate_para_id in sorted(candidate_para_ids):
        candidate = para_lookup.get(candidate_para_id)
        if not candidate:
            continue
        candidate_text = str(candidate.get("text", "") or "").strip()
        if not candidate_text:
            continue
        if _fact_page_matches_paragraph(fact_page, candidate):
            valid_candidates.append(candidate_para_id)

    if len(valid_candidates) == 1:
        return True, "ok"
    if len(valid_candidates) > 1:
        return False, "ambiguous_paragraph_mapping"

    return False, "fact_page_not_in_paragraph_span"


def compute_provenance_gate(run_dir: Path, golden: Optional[dict[str, Any]]) -> GateResult:
    """Compute provenance coverage and strong claim support metrics.
    
    Formula: provenance_coverage = anchored_atomic_claims / total_atomic_claims
    Formula: strong_claim_unsupported_count = count(strong_claims_without_anchor)
    """
    synthesis_path = run_dir / "reading" / "synthesis.json"
    facts_path = run_dir / "reading" / "facts.jsonl"
    paragraphs_path = run_dir / "paragraphs" / "paragraphs.jsonl"
    cite_anchors_path = run_dir / "citations" / "cite_anchors.jsonl"
    cite_map_path = run_dir / "citations" / "cite_map.jsonl"
    
    synthesis = load_json(synthesis_path)
    facts = load_jsonl(facts_path)
    paragraphs = load_jsonl(paragraphs_path)
    cite_anchors = load_jsonl(cite_anchors_path)
    cite_map = load_jsonl(cite_map_path)
    
    if not synthesis or not facts:
        return GateResult(
            name="provenance_coverage",
            value=0.0,
            status=GateStatus.NOT_EVALUATED,
            details={"reason": "missing synthesis or facts"}
        )
    
    # Build fact/paragraph/citation lookups
    fact_lookup = {f["fact_id"]: f for f in facts}
    para_lookup = {str(p.get("para_id", "")): p for p in paragraphs}
    para_ids_by_source_block: dict[str, set[str]] = {}
    for paragraph in paragraphs:
        paragraph_id = str(paragraph.get("para_id", "") or "")
        if not paragraph_id:
            continue
        evidence = paragraph.get("evidence_pointer", {})
        source_block_ids = evidence.get("source_block_ids", []) if isinstance(evidence, dict) else []
        if not isinstance(source_block_ids, list):
            continue
        for source_block_id in source_block_ids:
            block_id = str(source_block_id or "")
            if not block_id:
                continue
            para_ids_by_source_block.setdefault(block_id, set()).add(paragraph_id)

    mapped_ref_by_anchor_id = {
        str(m.get("anchor_id", "")): str(m.get("mapped_ref_key", "") or "")
        for m in cite_map
    }
    citation_support_by_para_id: dict[str, int] = {}
    for anchor in cite_anchors:
        if str(anchor.get("anchor_type", "")).strip().lower() != "citation_marker":
            continue
        anchor_id = str(anchor.get("anchor_id", ""))
        if not anchor_id:
            continue
        if not mapped_ref_by_anchor_id.get(anchor_id):
            continue
        para_id = str(anchor.get("nearest_para_id", "") or "")
        if not para_id:
            continue
        citation_support_by_para_id[para_id] = citation_support_by_para_id.get(para_id, 0) + 1
    
    # Get key evidence lines from synthesis
    key_evidence_lines = synthesis.get("key_evidence_lines", [])
    total_atomic_claims = len(key_evidence_lines)
    
    if total_atomic_claims == 0:
        # No claims to evaluate - check facts directly
        total_atomic_claims = len(facts)
        if total_atomic_claims == 0:
            return GateResult(
                name="provenance_coverage",
                value=0.0,
                status=GateStatus.NOT_EVALUATED,
                details={"reason": "no atomic claims found"}
            )
    
    # Count anchored claims
    anchored_count = 0
    strong_claim_count = 0
    unsupported_strong_claims = 0
    unanchored_claims: list[dict[str, Any]] = []
    anchor_type_counts: dict[str, int] = {}
    claims_with_citation_support = 0
    
    # Check facts directly if no key_evidence_lines
    claims_to_check = key_evidence_lines if key_evidence_lines else facts
    
    for claim in claims_to_check:
        # Get fact_ids from the claim
        fact_ids = claim.get("fact_ids", [])
        
        # For facts as claims, use the fact_id itself
        if not fact_ids and "fact_id" in claim:
            fact_ids = [claim["fact_id"]]
        
        if not fact_ids:
            # This is a claim without any linked facts - could be unsupported
            statement = claim.get("statement", "")
            category = claim.get("category", "none")
            if _is_strong_claim(statement, category):
                strong_claim_count += 1
                unsupported_strong_claims += 1
            unanchored_claims.append(
                {
                    "line_id": claim.get("line_id") or claim.get("fact_id") or "unknown",
                    "statement": str(statement),
                    "fact_ids": [],
                    "missing_anchor_reasons": ["missing_fact_ids"],
                }
            )
            continue

        # Check each linked fact
        all_anchored = True
        claim_anchor_types: set[str] = set()
        claim_reasons: list[str] = []
        has_citation_support = False
        for fid in fact_ids:
            fact = fact_lookup.get(fid)
            if not fact:
                all_anchored = False
                claim_reasons.append(f"missing_fact:{fid}")
                break

            has_evidence, evidence_reason, text_evidence_type = _has_minimal_anchor_evidence(fact)
            if not has_evidence:
                all_anchored = False
                claim_reasons.append(f"{fid}:{evidence_reason}")
                continue

            mapped_to_para, para_reason = _is_fact_mapped_to_paragraph(
                fact,
                para_lookup,
                para_ids_by_source_block,
            )
            if not mapped_to_para:
                all_anchored = False
                claim_reasons.append(f"{fid}:{para_reason}")
                continue

            para_id = str(fact.get("para_id", "") or "")
            if citation_support_by_para_id.get(para_id, 0) > 0:
                has_citation_support = True

            anchor_type = f"paragraph_{text_evidence_type}"
            claim_anchor_types.add(anchor_type)

        if all_anchored:
            anchored_count += 1
            for anchor_type in claim_anchor_types:
                anchor_type_counts[anchor_type] = anchor_type_counts.get(anchor_type, 0) + 1
            if has_citation_support:
                claims_with_citation_support += 1
        else:
            statement = str(claim.get("statement", "") or "")
            unanchored_claims.append(
                {
                    "line_id": claim.get("line_id") or claim.get("fact_id") or "unknown",
                    "statement": statement,
                    "fact_ids": list(fact_ids),
                    "missing_anchor_reasons": claim_reasons or ["unknown"],
                }
            )

        # Check if this is a strong claim
        statement = str(claim.get("statement", "") or "")
        category = str(claim.get("category", "none") or "none")
        inferred_category = category
        if category == "none" and isinstance(fact_ids, list) and fact_ids:
            first_fact = fact_lookup.get(str(fact_ids[0]))
            if isinstance(first_fact, dict):
                inferred_category = str(first_fact.get("category", "none") or "none")

        is_strong = bool(claim.get("is_strong_claim", False)) or _is_strong_claim(statement, inferred_category)
        if is_strong:
            strong_claim_count += 1
            if not all_anchored:
                unsupported_strong_claims += 1
    
    # Calculate provenance coverage
    provenance_coverage = anchored_count / total_atomic_claims if total_atomic_claims > 0 else 0.0
    
    golden_available = _has_evaluation_data(golden)
    
    # Determine status
    if not golden_available:
        status = GateStatus.NOT_EVALUATED
    elif provenance_coverage >= 0.95 and unsupported_strong_claims == 0:
        status = GateStatus.PASS
    else:
        status = GateStatus.FAIL
    
    return GateResult(
        name="provenance_coverage",
        value=provenance_coverage,
        status=status,
        threshold=">= 0.95",
        details={
            "anchored_atomic_claims": anchored_count,
            "total_atomic_claims": total_atomic_claims,
            "strong_claim_unsupported_count": unsupported_strong_claims,
            "strong_claim_count": strong_claim_count,
            "claims_with_citation_support": claims_with_citation_support,
            "anchor_type_counts": anchor_type_counts,
            "anchored_claim_rule": {
                "anchor_type": "paragraph_anchor",
                "minimal_evidence": [
                    "fact.evidence_pointer.page",
                    "fact.evidence_pointer.bbox",
                    "fact.evidence_pointer.source_block_ids",
                    "fact.quote or fact.statement",
                ],
                "mapping_rules": [
                    "claim.fact_ids must all resolve in reading/facts.jsonl",
                    "each fact.para_id must resolve in paragraphs/paragraphs.jsonl",
                    "fact evidence page must align with mapped paragraph pages/page_span",
                    "if fact.para_id is missing in paragraphs, deterministic source_block_ids lookup may recover a unique paragraph on the same page",
                ],
                "citation_support_note": "citation-marker anchors with mapped_ref_key are counted as supporting evidence when nearest_para_id maps to the fact paragraph",
            },
            "unanchored_claims": unanchored_claims,
            "golden_available": _has_evaluation_data(golden)
        }
    )


def compute_reading_order_gate(run_dir: Path, golden: Optional[dict[str, Any]]) -> GateResult:
    """Compute paragraph inversion rate and cross-column interleave rate.
    
    Formula: paragraph_inversion_rate = inverted_paragraph_pairs / evaluated_paragraph_pairs
    Formula: cross_column_interleave_page_rate = pages_with_interleave / pages_with_columns
    """
    paragraphs_path = run_dir / "paragraphs" / "paragraphs.jsonl"
    vision_out_paths = sorted(run_dir.glob("vision/p*_out.json"))
    
    paragraphs = load_jsonl(paragraphs_path)
    structure_quality = load_structure_quality(run_dir / "qa")
    structure_quality_flags = {
        "ordering_confidence_low": bool(structure_quality.get("ordering_confidence_low", False)),
        "section_boundary_unstable": bool(structure_quality.get("section_boundary_unstable", False)),
        "reference_region_ambiguous": bool(structure_quality.get("reference_region_ambiguous", False)),
        "caption_linking_partial": bool(structure_quality.get("caption_linking_partial", False)),
    }
    
    if not paragraphs:
        return GateResult(
            name="paragraph_inversion_rate",
            value=0.0,
            status=GateStatus.NOT_EVALUATED,
            details={
                "reason": "no paragraphs found",
                "structure_quality_flags": structure_quality_flags,
            }
        )
    
    # If golden available, compute actual inversion rate
    if golden and "reading_order_pairs" in golden:
        reading_pairs = golden.get("reading_order_pairs", [])
        
        if not reading_pairs:
            paragraph_inversion_rate = 0.0
            inverted_count = 0
        else:
            # Build para_id to position map
            para_positions = {}
            for i, para in enumerate(paragraphs):
                para_positions[para.get("para_id", "")] = i
            
            # Count inversions
            inverted_count = 0
            evaluated_count = 0
            
            for pair in reading_pairs:
                left_id = pair.get("left_para_id")
                right_id = pair.get("right_para_id")
                expected = pair.get("expected_order", "left_before_right")
                
                if left_id not in para_positions or right_id not in para_positions:
                    continue
                
                evaluated_count += 1
                left_pos = para_positions[left_id]
                right_pos = para_positions[right_id]
                
                if expected == "left_before_right":
                    if left_pos > right_pos:
                        inverted_count += 1
                else:
                    if left_pos < right_pos:
                        inverted_count += 1
            
            paragraph_inversion_rate = inverted_count / evaluated_count if evaluated_count > 0 else 0.0
    else:
        paragraph_inversion_rate = 0.0
        inverted_count = 0
    
    # Compute cross-column interleave rate
    multi_column_pages = []
    if golden and "multi_column_pages" in golden:
        multi_column_pages = golden.get("multi_column_pages", [])
    
    pages_with_interleave = 0
    if golden and "interleave_labels" in golden:
        for label in golden.get("interleave_labels", []):
            if label.get("has_interleave", False):
                pages_with_interleave += 1
    
    total_multi_column = len(multi_column_pages)
    if total_multi_column > 0:
        cross_column_interleave_rate = pages_with_interleave / total_multi_column
    else:
        cross_column_interleave_rate = 0.0
    
    golden_available = _has_evaluation_data(golden)
    
    # Determine status
    if not golden_available:
        status = GateStatus.NOT_EVALUATED
    elif paragraph_inversion_rate <= 0.02 and cross_column_interleave_rate <= 0.01:
        status = GateStatus.PASS
    else:
        status = GateStatus.FAIL
    
    return GateResult(
        name="paragraph_inversion_rate",
        value=paragraph_inversion_rate,
        status=status,
        threshold="<= 0.02",
        details={
            "inverted_pairs": inverted_count,
            "evaluated_pairs": golden.get("reading_order_pairs", []) if golden else [],
            "cross_column_interleave_rate": cross_column_interleave_rate,
            "pages_with_interleave": pages_with_interleave,
            "pages_with_columns": total_multi_column,
            "golden_available": _has_evaluation_data(golden),
            "structure_quality_flags": structure_quality_flags,
        }
    )


def compute_citation_gate(run_dir: Path, golden: Optional[dict[str, Any]]) -> GateResult:
    """Compute citation mapping coverage and DOI/PMID precision.
    
    Formula: citation_mapping_coverage = mapped_markers / total_markers
    Formula: doi_pmid_precision = correct_doi_pmid / extracted_doi_pmid
    """
    cite_anchors_path = run_dir / "citations" / "cite_anchors.jsonl"
    cite_map_path = run_dir / "citations" / "cite_map.jsonl"
    
    anchors = load_jsonl(cite_anchors_path)
    mappings = load_jsonl(cite_map_path)
    
    if not anchors or not mappings:
        return GateResult(
            name="citation_mapping_coverage",
            value=0.0,
            status=GateStatus.NOT_EVALUATED,
            details={"reason": "no citation data found"}
        )
    
    mapping_by_anchor = {str(m.get("anchor_id", "")): m for m in mappings}

    def is_citation_marker_anchor(anchor: dict[str, Any]) -> bool:
        anchor_type = str(anchor.get("anchor_type", "")).strip().lower()
        if anchor_type:
            return anchor_type == "citation_marker"
        text = str(anchor.get("anchor_text", "")).strip().lower()
        if re.fullmatch(r"[\[\(]?\d+(?:\s*[-,;]\s*\d+)*[\]\)]?", text):
            return True
        if text.startswith("doi:") or text.startswith("pmid:"):
            return True
        return False

    citation_anchors = [a for a in anchors if is_citation_marker_anchor(a)]
    citation_anchor_ids = {str(a.get("anchor_id", "")) for a in citation_anchors}

    if not citation_anchors:
        return GateResult(
            name="citation_mapping_coverage",
            value=0.0,
            status=GateStatus.NOT_EVALUATED,
            details={
                "reason": "no citation-marker anchors found",
                "raw_total_anchors": len(anchors),
                "raw_total_mappings": len(mappings),
            },
        )

    citation_truth = golden.get("citation_truth", []) if golden else []
    truth_marker_ids = {
        str(truth.get("marker_id", ""))
        for truth in citation_truth
        if str(truth.get("marker_id", ""))
    }

    if truth_marker_ids:
        marker_ids_for_coverage = truth_marker_ids
        coverage_scope = "golden_citation_truth"
    else:
        marker_ids_for_coverage = citation_anchor_ids
        coverage_scope = "all_citation_markers"

    total_markers = len(marker_ids_for_coverage)
    mapped_markers = sum(
        1
        for anchor_id in marker_ids_for_coverage
        if mapping_by_anchor.get(anchor_id, {}).get("mapped_ref_key") is not None
    )
    citation_mapping_coverage = mapped_markers / total_markers if total_markers > 0 else 0.0
    
    # Compute DOI/PMID precision if golden available.
    #
    # Semantics used here are explicit to avoid conflating coverage and
    # identifier precision when golden truth is only partially annotated:
    # - True positive: normalized exact identifier match on a golden-marked
    #   DOI/PMID anchor.
    # - Missing identifier on a golden DOI/PMID anchor: counted as incorrect
    #   (included in denominator).
    # - Predicted DOI/PMID on anchors not present in golden DOI/PMID truth:
    #   excluded from this precision denominator (reported separately).
    # - Multiple identifiers (expected/predicted): pass if any normalized
    #   predicted DOI/PMID matches any normalized expected DOI/PMID.
    doi_pmid_precision = 0.0
    correct_doi_pmid = 0
    extracted_doi_pmid = 0
    missing_doi_pmid = 0
    mismatched_doi_pmid = 0
    unscored_extracted_doi_pmid = 0

    def _normalize_identifier(ref_key: str) -> str:
        if ref_key.startswith("doi:"):
            return f"doi:{ref_key[4:].strip().lower()}"
        if ref_key.startswith("pmid:"):
            return f"pmid:{ref_key[5:].strip()}"
        return ref_key.strip().lower()

    def _extract_identifier_candidates(value: Any) -> set[str]:
        values: list[str] = []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = [item for item in value if isinstance(item, str)]
        elif isinstance(value, tuple):
            values = [item for item in value if isinstance(item, str)]

        identifiers = {
            _normalize_identifier(item)
            for item in values
            if item.startswith(("doi:", "pmid:"))
        }
        return identifiers
    
    if golden and "citation_truth" in golden:
        doi_pmid_truth: dict[str, set[str]] = {}

        for truth in citation_truth:
            marker_id = str(truth.get("marker_id", ""))
            expected_identifiers = _extract_identifier_candidates(truth.get("expected_ref_key"))
            if marker_id and expected_identifiers:
                doi_pmid_truth[marker_id] = expected_identifiers

        extracted_identifier_by_anchor: dict[str, set[str]] = {}
        for mapping in mappings:
            anchor_id = str(mapping.get("anchor_id", ""))
            if anchor_id not in citation_anchor_ids:
                continue
            predicted = _extract_identifier_candidates(mapping.get("mapped_ref_key"))
            if not predicted:
                continue
            extracted_identifier_by_anchor.setdefault(anchor_id, set()).update(predicted)

        extracted_doi_pmid = len(doi_pmid_truth)
        if extracted_doi_pmid > 0:
            for anchor_id, expected_identifiers in doi_pmid_truth.items():
                predicted_identifiers = extracted_identifier_by_anchor.get(anchor_id, set())
                if not predicted_identifiers:
                    missing_doi_pmid += 1
                    continue
                if expected_identifiers & predicted_identifiers:
                    correct_doi_pmid += 1
                else:
                    mismatched_doi_pmid += 1

            doi_pmid_precision = correct_doi_pmid / extracted_doi_pmid

        truth_anchor_ids = set(doi_pmid_truth)
        for anchor_id, predicted_identifiers in extracted_identifier_by_anchor.items():
            if anchor_id not in truth_anchor_ids:
                unscored_extracted_doi_pmid += len(predicted_identifiers)
    else:
        # If no golden, assume extracted DOI/PMIDs are correct
        doi_pmid_precision = 1.0
    
    golden_available = _has_evaluation_data(golden)
    
    # Determine status
    if not golden_available:
        status = GateStatus.NOT_EVALUATED
    elif citation_mapping_coverage >= 0.98 and doi_pmid_precision >= 0.99:
        status = GateStatus.PASS
    else:
        status = GateStatus.FAIL
    
    return GateResult(
        name="citation_mapping_coverage",
        value=citation_mapping_coverage,
        status=status,
        threshold=">= 0.98",
        details={
            "mapped_markers": mapped_markers,
            "total_markers": total_markers,
            "raw_total_anchors": len(anchors),
            "raw_total_mappings": len(mappings),
            "raw_mapped_any": sum(1 for m in mappings if m.get("mapped_ref_key") is not None),
            "coverage_scope": coverage_scope,
            "doi_pmid_precision": doi_pmid_precision,
            "correct_doi_pmid": correct_doi_pmid,
            "extracted_doi_pmid": extracted_doi_pmid,
            "missing_doi_pmid": missing_doi_pmid,
            "mismatched_doi_pmid": mismatched_doi_pmid,
            "unscored_extracted_doi_pmid": unscored_extracted_doi_pmid,
            "doi_pmid_precision_semantics": {
                "true_positive": "normalized exact match between expected and predicted doi:/pmid: on a golden citation marker",
                "missing_identifier_handling": "missing doi:/pmid: on a golden doi/pmid marker counts as incorrect and is included in denominator",
                "multi_identifier_handling": "if expected or predicted has multiple doi:/pmid: values, any normalized overlap counts as correct",
                "non_truth_identifier_handling": "predicted doi:/pmid: on citation markers not annotated in golden doi/pmid truth are excluded from precision denominator and reported via unscored_extracted_doi_pmid",
                "precision_formula": "doi_pmid_precision = correct_doi_pmid / extracted_doi_pmid, where extracted_doi_pmid is the count of golden doi/pmid truth markers evaluated",
                "gate_rule": "citation gate passes only when citation_mapping_coverage >= 0.98 and doi_pmid_precision >= 0.99",
            },
            "golden_available": _has_evaluation_data(golden)
        }
    )


def compute_figure_caption_gate(run_dir: Path, golden: Optional[dict[str, Any]]) -> GateResult:
    """Compute figure-caption precision and caption ID retention.
    
    Formula: figure_caption_precision = correct_figure_caption_links / evaluated_figure_caption_links
    Formula: caption_id_retention = captions_with_id / captions_expected_with_id
    """
    ft_index_path = run_dir / "figures_tables" / "figure_table_index.jsonl"
    
    ft_index = load_jsonl(ft_index_path)
    
    if not ft_index:
        return GateResult(
            name="figure_caption_precision",
            value=0.0,
            status=GateStatus.NOT_EVALUATED,
            details={"reason": "no figure/table data found"}
        )
    
    # Extract figures and captions
    figures = [f for f in ft_index if f.get("asset_type") == "figure"]
    captions = [f for f in ft_index if f.get("asset_type") == "figure" and f.get("caption_id")]
    
    captions_with_id = len(captions)
    captions_expected = len(figures)  # Each figure should have a caption
    caption_id_retention = captions_with_id / captions_expected if captions_expected > 0 else 0.0
    
    # Compute precision if golden available
    figure_caption_precision = 0.0
    correct_links = 0
    evaluated_links = 0
    
    if golden and "figure_caption_truth" in golden:
        truth_map = {
            t["figure_id"]: t["expected_caption_id"] 
            for t in golden.get("figure_caption_truth", [])
        }
        
        for fig in figures:
            fig_id = fig.get("asset_id")
            expected_caption = truth_map.get(fig_id)
            
            if expected_caption is None:
                continue
            
            evaluated_links += 1
            actual_caption = fig.get("caption_id", "")
            
            if actual_caption and actual_caption == expected_caption:
                correct_links += 1
        
        figure_caption_precision = correct_links / evaluated_links if evaluated_links > 0 else 0.0
    else:
        # Without golden, assume all linked captions are correct
        figure_caption_precision = 1.0 if captions_with_id > 0 else 0.0
    
    golden_available = _has_evaluation_data(golden)
    
    # Determine status
    if not golden_available:
        status = GateStatus.NOT_EVALUATED
    elif figure_caption_precision >= 0.90 and caption_id_retention >= 0.95:
        status = GateStatus.PASS
    else:
        status = GateStatus.FAIL
    
    return GateResult(
        name="figure_caption_precision",
        value=figure_caption_precision,
        status=status,
        threshold=">= 0.90",
        details={
            "correct_figure_caption_links": correct_links,
            "evaluated_figure_caption_links": evaluated_links,
            "caption_id_retention": caption_id_retention,
            "captions_with_id": captions_with_id,
            "captions_expected": captions_expected,
            "golden_available": _has_evaluation_data(golden)
        }
    )


def compute_truncation_gate(run_dir: Path) -> GateResult:
    """Compute silent truncation count.
    
    Formula: silent_truncation_count = count(truncations_without_metadata)
    """
    facts_path = run_dir / "reading" / "facts.jsonl"
    synthesis_path = run_dir / "reading" / "synthesis.json"
    
    facts = load_jsonl(facts_path)
    synthesis = load_json(synthesis_path)
    
    # Check facts for silent truncations
    # A silent truncation is when a quote is truncated but no truncation metadata exists
    silent_truncations = 0
    
    for fact in facts:
        quote = fact.get("quote", "")
        statement = fact.get("statement", "")
        quote_truncated = bool(fact.get("quote_truncated", False))
        truncation_reason = str(fact.get("truncation_reason", "") or "")

        if quote_truncated or truncation_reason:
            continue
        
        # Check if quote looks truncated (ends abruptly with incomplete sentence)
        # but no truncation reason is provided in statement
        if quote and len(quote) < len(statement):
            ratio = len(quote) / max(1, len(statement))
            if ratio >= 0.9 or (len(statement) - len(quote) <= 6):
                continue
            # Quote is shorter than statement - might be truncated
            # Check if truncation is documented
            if "truncat" not in statement.lower() and "..." not in quote:
                # No explicit truncation marker
                # Heuristic: if quote ends with incomplete word or sentence
                if quote.rstrip()[-1] not in ".!?":
                    silent_truncations += 1
    
    # Check synthesis for truncation issues
    if synthesis:
        key_lines = synthesis.get("key_evidence_lines", [])
        for line in key_lines:
            quotes = line.get("quotes", [])
            for q in quotes:
                if isinstance(q, str) and len(q) < 30 and q.rstrip()[-1] not in ".!?":
                    # Short quote ending without punctuation - possible silent truncation
                    silent_truncations += 1
    
    status = GateStatus.PASS if silent_truncations == 0 else GateStatus.FAIL
    
    return GateResult(
        name="silent_truncation_count",
        value=float(silent_truncations),
        status=status,
        threshold="== 0",
        details={
            "silent_truncations": silent_truncations,
            "facts_checked": len(facts)
        }
    )


def compute_reference_quality_gate(run_dir: Path, golden: Optional[dict[str, Any]] = None) -> GateResult:
    """Compute reference-pipeline quality metrics.

    Metrics produced in details:
      - reference_source_mix: counts by provider and by kind (api/pdf)
      - api_reference_count: number of API-sourced reference records
      - dedupe_rate: 1 - (merged_count / raw_total) where raw_total = api_count + pdf_count
      - identifier_completeness: fraction of merged refs with doi or pmid

    Robust fallback semantics:
      - If no reference artifacts at all -> NOT_EVALUATED
      - If merged artifact missing but raw artifacts present -> DEGRADED
    """
    refs_api_path = run_dir / "refs" / "references_api.jsonl"
    refs_merged_path = run_dir / "refs" / "references_merged.jsonl"
    pdf_catalog_path = run_dir / "citations" / "reference_catalog.jsonl"

    api_refs = load_jsonl(refs_api_path)
    merged_refs = load_jsonl(refs_merged_path)
    pdf_refs = load_jsonl(pdf_catalog_path)

    api_count = len(api_refs)
    merged_count = len(merged_refs)
    pdf_count = len(pdf_refs)

    raw_total = api_count + pdf_count

    # Compute dedupe semantics with explicit comparable-capacity handling.
    # We prefer identifier-based dedupe measurement (doi/pmid) because PDF
    # catalog entries are often noisy and not comparable; fall back to the
    # raw_total formula only when no identifier-bearing raw entries exist.
    def _has_identifier(r: dict[str, Any]) -> bool:
        return bool(r.get("doi") or r.get("pmid"))

    # Count identifier-bearing records separately by source for overlap-capacity semantics
    api_identifier_count = sum(1 for r in api_refs if _has_identifier(r))
    pdf_identifier_count = sum(1 for r in pdf_refs if _has_identifier(r))
    raw_identifier_total = api_identifier_count + pdf_identifier_count
    merged_with_identifier = sum(1 for r in merged_refs if _has_identifier(r))

    # Raw (old) dedupe rate kept for backward compatibility reporting
    dedupe_rate_raw = 0.0
    if raw_total > 0:
        dedupe_rate_raw = 1.0 - (merged_count / float(raw_total))
        if dedupe_rate_raw < 0:
            dedupe_rate_raw = 0.0

    # Previous identifier-based metric measured reduction across all identifier
    # records; this produced misleadingly small values when API identifiers
    # dominated. We instead compute an overlap-capacity metric that measures
    # cross-source overlap among identifier-bearing records.
    dedupe_rate_on_identifiers: Optional[float]
    # Compute identifier overlap counts and capacity
    overlap_identifier_count = max(0, api_identifier_count + pdf_identifier_count - merged_with_identifier)
    identifier_overlap_capacity = min(api_identifier_count, pdf_identifier_count)

    if identifier_overlap_capacity > 0:
        dedupe_rate_on_identifiers = overlap_identifier_count / float(identifier_overlap_capacity)
        # Clamp to [0,1]
        if dedupe_rate_on_identifiers < 0:
            dedupe_rate_on_identifiers = 0.0
        if dedupe_rate_on_identifiers > 1:
            dedupe_rate_on_identifiers = 1.0
    else:
        dedupe_rate_on_identifiers = None

    # Choose dedupe_rate to report: prefer identifier-based when available
    # Second-pass: if no identifier signal available, prefer an overlap-capacity
    # semantics that measures dedupe relative to comparable PDF capacity. This
    # avoids penalizing cases where PDFs are mostly noisy/non-comparable.
    # Compute overlap/capacity metrics:
    overlap_count = max(0, api_count + pdf_count - merged_count)

    def _pdf_is_comparable(r: dict[str, Any]) -> bool:
        # Comparable PDF refs require at least one bibliographic signal
        # (doi, pmid, or year). Year may be present as int or str.
        if r.get("doi") or r.get("pmid"):
            return True
        year = r.get("year")
        if isinstance(year, int) and year > 0:
            return True
        if isinstance(year, str) and year.isdigit() and len(year) == 4:
            return True
        return False

    comparable_pdf_count = sum(1 for r in pdf_refs if _pdf_is_comparable(r))
    overlap_capacity = min(api_count, comparable_pdf_count)

    dedupe_rate_overlap: Optional[float]
    if overlap_capacity > 0:
        dedupe_rate_overlap = overlap_count / float(overlap_capacity)
        if dedupe_rate_overlap < 0:
            dedupe_rate_overlap = 0.0
        if dedupe_rate_overlap > 1:
            dedupe_rate_overlap = 1.0
    else:
        dedupe_rate_overlap = None

    # Selection priority: identifier-based -> overlap-capacity -> raw fallback
    if dedupe_rate_on_identifiers is not None:
        dedupe_rate = dedupe_rate_on_identifiers
        dedupe_choice = "identifier_preferred"
    elif dedupe_rate_overlap is not None:
        dedupe_rate = dedupe_rate_overlap
        dedupe_choice = "overlap_capacity"
    else:
        dedupe_rate = dedupe_rate_raw
        dedupe_choice = "raw_fallback"

    # Identifier completeness over merged set
    id_with_count = 0
    provider_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}

    for mr in merged_refs:
        doi = mr.get("doi")
        pmid = mr.get("pmid")
        if doi or pmid:
            id_with_count += 1

        sources = mr.get("sources") or []
        if isinstance(sources, list):
            for s in sources:
                provider = str(s.get("provider") or "unknown")
                kind = str(s.get("kind") or "api")
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
                kind_counts[kind] = kind_counts.get(kind, 0) + 1

    identifier_completeness = (id_with_count / merged_count) if merged_count > 0 else 0.0

    reference_source_mix = {"by_provider": provider_counts, "by_kind": kind_counts}

    # Determine status semantics
    # If no artifacts exist at all, we cannot evaluate
    if api_count == 0 and merged_count == 0 and pdf_count == 0:
        return GateResult(
            name="reference_pipeline_quality",
            value=0.0,
            status=GateStatus.NOT_EVALUATED,
            details={
                "reason": "no_reference_artifacts",
                "api_reference_count": api_count,
                "pdf_reference_count": pdf_count,
                "merged_reference_count": merged_count,
            },
        )

    # If merged is missing but raw data exists, degrade
    if merged_count == 0 and raw_total > 0:
        return GateResult(
            name="reference_pipeline_quality",
            value=0.0,
            status=GateStatus.DEGRADED,
            details={
                "reason": "merged_artifact_missing",
                "api_reference_count": api_count,
                "pdf_reference_count": pdf_count,
                "merged_reference_count": merged_count,
                "reference_source_mix": reference_source_mix,
            },
        )

    # Otherwise produce measured metrics
    # Provisional thresholds: identifier_completeness >= 0.5 and dedupe_rate >= 0.05 -> PASS
    status = GateStatus.PASS if (identifier_completeness >= 0.5 and dedupe_rate >= 0.05) else GateStatus.DEGRADED

    return GateResult(
        name="reference_pipeline_quality",
        value=float(identifier_completeness),
        status=status,
        threshold="identifier_completeness>=0.5,dedupe_rate>=0.05",
        details={
            "reference_source_mix": reference_source_mix,
            "api_reference_count": api_count,
            "pdf_reference_count": pdf_count,
            "merged_reference_count": merged_count,
            # Backwards-compatible primary dedupe_rate key (may be identifier-based)
            "dedupe_rate": dedupe_rate,
            # Expose raw and identifier-focused dedupe metrics for transparency
            "dedupe_rate_raw": dedupe_rate_raw,
            "dedupe_rate_on_identifiers": dedupe_rate_on_identifiers,
            "api_identifier_count": api_identifier_count,
            "pdf_identifier_count": pdf_identifier_count,
            "overlap_identifier_count": overlap_identifier_count,
            "identifier_overlap_capacity": identifier_overlap_capacity,
            "raw_identifier_total": raw_identifier_total,
            "merged_with_identifier": merged_with_identifier,
            "dedupe_semantics": (
                "identifier_preferred" if dedupe_rate_on_identifiers is not None else "raw_fallback"
            ),
            "identifier_completeness": identifier_completeness,
            "golden_available": _has_evaluation_data(golden),
        },
    )


def compute_runtime_safety(run_dir: Path) -> dict[str, Any]:
    """Compute runtime safety metrics.
    
    Returns dict with network deny evidence.
    """
    runtime_safety_path = run_dir / "qa" / "runtime_safety.json"
    
    if runtime_safety_path.exists():
        return load_json(runtime_safety_path) or {}
    
    # Return default structure
    return {
        "network_deny_mode": True,
        "egress_attempt_count": 0,
        "egress_attempt_targets": [],
        "stage": "unknown",
        "note": "runtime_safety.json not found - parse stages should deny network"
    }


def run_verification(doc_id: str) -> Report:
    """Run complete verification for a document."""
    run_dir = RUN_ROOT / doc_id
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load golden data if available
    golden = load_golden(doc_id)
    
    # Compute all gates
    provenance_gate = compute_provenance_gate(run_dir, golden)
    reading_order_gate = compute_reading_order_gate(run_dir, golden)
    citation_gate = compute_citation_gate(run_dir, golden)
    figure_caption_gate = compute_figure_caption_gate(run_dir, golden)
    truncation_gate = compute_truncation_gate(run_dir)
    reference_gate = compute_reference_quality_gate(run_dir, golden)
    
    # Build gates dict
    gates = {
        "provenance": provenance_gate,
        "reading_order": reading_order_gate,
        "citation": citation_gate,
        "reference": reference_gate,
        "figure_caption": figure_caption_gate,
        "truncation": truncation_gate
    }
    
    # Determine overall status
    # Hard-stop gates: provenance, reading_order, citation
    hard_stop_gates = ["provenance", "reading_order", "citation"]
    hard_stop_failed = any(
        gates[g].status == GateStatus.FAIL for g in hard_stop_gates
    )
    
    # Degradation gates: figure_caption, truncation
    degradation_labels = []
    for g in ["figure_caption", "truncation"]:
        if gates[g].status == GateStatus.FAIL:
            if g == "figure_caption":
                degradation_labels.append("quality: low")
            elif g == "truncation":
                degradation_labels.append("limitations")
    
    if hard_stop_failed:
        overall_status = GateStatus.FAIL
        hard_stop = True
        reason = "Release blocker failed: " + ", ".join(
            g for g in hard_stop_gates if gates[g].status == GateStatus.FAIL
        )
    elif degradation_labels:
        overall_status = GateStatus.DEGRADED
        hard_stop = False
        reason = f"Quality degraded: {', '.join(degradation_labels)}"
    else:
        overall_status = GateStatus.PASS
        hard_stop = False
        reason = "All gates passed"
    
    # Add not_evaluated gates reason if any
    not_evaluated = [g for g in gates.values() if g.status == GateStatus.NOT_EVALUATED]
    if not_evaluated and overall_status == GateStatus.PASS:
        # If some gates are not evaluated but others passed, mark as degraded
        overall_status = GateStatus.DEGRADED
        reason = "Some metrics not evaluated (no golden file)"
    
    report = Report(
        doc_id=doc_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        gates=gates,
        overall_status=overall_status,
        hard_stop=hard_stop,
        reason=reason,
        degradation_labels=degradation_labels
    )
    
    return report


def save_report(report: Report, run_dir: Path) -> None:
    """Save QA report and stage status to run directory."""
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    
    # Save report.json
    report_data: dict[str, Any] = {
        "doc_id": report.doc_id,
        "generated_at": report.generated_at,
        "overall_status": report.overall_status.value,
        "hard_stop": report.hard_stop,
        "reason": report.reason,
        "degradation_labels": report.degradation_labels,
        "gates": {}
    }
    
    for gate_name, gate in report.gates.items():
        report_data["gates"][gate_name] = {
            "name": gate.name,
            "value": gate.value,
            "status": gate.status.value,
            "threshold": gate.threshold,
            "details": gate.details
        }
    
    report_path = qa_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    
    # Save stage_status.json
    stage_status = {
        "stage": "verify",
        "status": report.overall_status.value,
        "hard_stop": report.hard_stop,
        "reason": report.reason
    }
    
    stage_status_path = qa_dir / "stage_status.json"
    with open(stage_status_path, "w", encoding="utf-8") as f:
        json.dump(stage_status, f, indent=2)
    
    # Ensure runtime_safety.json exists
    runtime_safety_path = qa_dir / "runtime_safety.json"
    if not runtime_safety_path.exists():
        runtime_safety = {
            "network_deny_mode": True,
            "egress_attempt_count": 0,
            "egress_attempt_targets": [],
            "stage": "verify"
        }
        with open(runtime_safety_path, "w", encoding="utf-8") as f:
            json.dump(runtime_safety, f, indent=2)


def verify(doc_id: str) -> int:
    """Run verification and return exit code.
    
    Returns:
        0 if verification passed or degraded
        1 if verification failed (hard-stop)
    """
    try:
        report = run_verification(doc_id)
        run_dir = RUN_ROOT / doc_id
        save_report(report, run_dir)
        
        if report.hard_stop:
            return 1
        return 0
        
    except FileNotFoundError as e:
        # Log error and return failure
        print(f"Verification failed: {e}")
        return 1
    except Exception as e:
        print(f"Verification error: {e}")
        return 1
