"""Structure-layer quality signals shared across stages."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def derive_structure_quality_flags(
    vision_dir: Path,
    clean_document_metrics: dict[str, Any],
) -> dict[str, bool]:
    reference_region_ambiguous = False
    caption_linking_partial = False

    for path in sorted(vision_dir.glob("p*_out.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue

        role_labels = payload.get("role_labels", {})
        if not isinstance(role_labels, dict):
            role_labels = {}
        role_values = {str(role) for role in role_labels.values() if isinstance(role, str)}
        fallback_used = bool(payload.get("fallback_used", False))

        if fallback_used and "ReferenceList" in role_values:
            reference_region_ambiguous = True
        if fallback_used and role_values.intersection({"FigureCaption", "TableCaption"}):
            caption_linking_partial = True

    return {
        "ordering_confidence_low": _safe_bool(clean_document_metrics.get("ordering_confidence_low")),
        "section_boundary_unstable": _safe_bool(clean_document_metrics.get("section_boundary_unstable")),
        "reference_region_ambiguous": reference_region_ambiguous,
        "caption_linking_partial": caption_linking_partial,
    }


def build_structure_quality_artifact(
    *,
    doc_id: str,
    parser_backend: str,
    vision_dir: Path,
    clean_document_metrics: dict[str, Any],
) -> dict[str, Any]:
    flags = derive_structure_quality_flags(
        vision_dir=vision_dir,
        clean_document_metrics=clean_document_metrics,
    )
    return {
        "doc_id": doc_id,
        "parser_backend": str(parser_backend or "builtin"),
        **flags,
        "generated_at": _iso_now(),
    }


def load_structure_quality(qa_dir: Path) -> dict[str, Any]:
    artifact_path = qa_dir / "structure_quality.json"
    if artifact_path.exists():
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        if isinstance(payload, dict):
            return payload

    metrics_path = qa_dir / "clean_document_metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        if isinstance(payload, dict):
            return {
                "ordering_confidence_low": _safe_bool(payload.get("ordering_confidence_low")),
                "section_boundary_unstable": _safe_bool(payload.get("section_boundary_unstable")),
                "reference_region_ambiguous": False,
                "caption_linking_partial": False,
            }

    return {}
