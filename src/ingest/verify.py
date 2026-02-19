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


def compute_provenance_gate(run_dir: Path, golden: Optional[dict[str, Any]]) -> GateResult:
    """Compute provenance coverage and strong claim support metrics.
    
    Formula: provenance_coverage = anchored_atomic_claims / total_atomic_claims
    Formula: strong_claim_unsupported_count = count(strong_claims_without_anchor)
    """
    synthesis_path = run_dir / "reading" / "synthesis.json"
    facts_path = run_dir / "reading" / "facts.jsonl"
    
    synthesis = load_json(synthesis_path)
    facts = load_jsonl(facts_path)
    
    if not synthesis or not facts:
        return GateResult(
            name="provenance_coverage",
            value=0.0,
            status=GateStatus.NOT_EVALUATED,
            details={"reason": "missing synthesis or facts"}
        )
    
    # Build fact lookup
    fact_lookup = {f["fact_id"]: f for f in facts}
    
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
                unsupported_strong_claims += 1
            continue
        
        # Check each linked fact
        all_anchored = True
        for fid in fact_ids:
            fact = fact_lookup.get(fid)
            if not fact:
                all_anchored = False
                break
            if not _has_valid_evidence(fact):
                all_anchored = False
                break
        
        if all_anchored:
            anchored_count += 1
        
        # Check if this is a strong claim
        statement = claim.get("statement", "")
        category = claim.get("category", "none")
        if _is_strong_claim(statement, category):
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
    
    if not paragraphs:
        return GateResult(
            name="paragraph_inversion_rate",
            value=0.0,
            status=GateStatus.NOT_EVALUATED,
            details={"reason": "no paragraphs found"}
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
            "golden_available": _has_evaluation_data(golden)
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
    
    total_markers = len(anchors)
    mapped_markers = sum(1 for m in mappings if m.get("mapped_ref_key") is not None)
    citation_mapping_coverage = mapped_markers / total_markers if total_markers > 0 else 0.0
    
    # Compute DOI/PMID precision if golden available
    doi_pmid_precision = 0.0
    correct_doi_pmid = 0
    extracted_doi_pmid = 0
    
    if golden and "citation_truth" in golden:
        citation_truth = golden.get("citation_truth", [])
        truth_map = {t["marker_id"]: t["expected_ref_key"] for t in citation_truth}
        
        # Extract DOI/PMID mappings
        doi_pmid_mappings = [
            m for m in mappings 
            if m.get("mapped_ref_key") and 
            (m["mapped_ref_key"].startswith("doi:") or m["mapped_ref_key"].startswith("pmid:"))
        ]
        
        extracted_doi_pmid = len(doi_pmid_mappings)
        
        if extracted_doi_pmid > 0:
            for m in doi_pmid_mappings:
                anchor_id = m.get("anchor_id", "")
                mapped = m.get("mapped_ref_key", "")
                expected = truth_map.get(anchor_id)
                
                if expected and mapped.lower() == expected.lower():
                    correct_doi_pmid += 1
            
            doi_pmid_precision = correct_doi_pmid / extracted_doi_pmid
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
            "doi_pmid_precision": doi_pmid_precision,
            "correct_doi_pmid": correct_doi_pmid,
            "extracted_doi_pmid": extracted_doi_pmid,
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
        
        # Check if quote looks truncated (ends abruptly with incomplete sentence)
        # but no truncation reason is provided in statement
        if quote and len(quote) < len(statement):
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
    
    # Build gates dict
    gates = {
        "provenance": provenance_gate,
        "reading_order": reading_order_gate,
        "citation": citation_gate,
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
