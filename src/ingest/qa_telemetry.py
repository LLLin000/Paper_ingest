"""QA telemetry normalization and trend summaries."""

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_context(qa_dir: Path) -> tuple[str, str]:
    attempt_id = os.environ.get("INGEST_QA_ATTEMPT_ID", "").strip()
    attempt_started_at = os.environ.get("INGEST_QA_ATTEMPT_STARTED_AT", "").strip()
    if not attempt_id:
        attempt_id = f"attempt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        os.environ["INGEST_QA_ATTEMPT_ID"] = attempt_id
    if not attempt_started_at:
        attempt_started_at = _iso_now()
        os.environ["INGEST_QA_ATTEMPT_STARTED_AT"] = attempt_started_at
    qa_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = qa_dir / "run_attempt.json"
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "attempt_id": attempt_id,
                "attempt_started_at": attempt_started_at,
                "updated_at": _iso_now(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return attempt_id, attempt_started_at


def _classify(error_type: str, http_status: Any, fault: str, success: Any) -> tuple[str, str, str]:
    if bool(success) and str(error_type or "none") in {"", "none"}:
        return "none", "none", "not_applicable"

    error_l = str(error_type or "").lower()
    fault_l = str(fault or "").lower()
    status_num = int(http_status) if isinstance(http_status, int) else None

    if "missing_api_key" in error_l or "credential" in error_l or "auth" in error_l or "token" in error_l:
        return "config/auth", "missing_credentials", "non_retryable"
    if "invalid_endpoint" in error_l:
        return "config/auth", "invalid_configuration", "non_retryable"
    if "missing-api-key" in fault_l or "auth" in fault_l:
        return "config/auth", "missing_credentials", "non_retryable"
    if status_num == 400 or "request_contract" in error_l:
        return "request-contract", "request_contract", "non_retryable"
    if status_num in {408, 429, 500, 502, 503, 504} or "transient_http_error" in error_l:
        return "timeout/retry", "transient_timeout_or_network", "retryable"
    if "timeout" in error_l or "timeout" in fault_l or "url_error" in error_l or "network_error" in error_l:
        return "timeout/retry", "transient_timeout_or_network", "retryable"
    if "parse" in error_l or "schema" in error_l or "validation" in error_l or "parse" in fault_l:
        return "parse/validation", "parse_or_validation", "non_retryable"
    if "invalid-model-output" in fault_l:
        return "parse/validation", "parse_or_validation", "non_retryable"
    return "parse/validation", "unknown", "unknown"


def enrich_event(qa_dir: Path, event: dict[str, Any], source: str) -> dict[str, Any]:
    attempt_id, attempt_started_at = _attempt_context(qa_dir)
    payload = dict(event)
    payload.setdefault("observed_at", _iso_now())
    payload["attempt_id"] = attempt_id
    payload["attempt_started_at"] = attempt_started_at
    payload["telemetry_source"] = source
    category, root_cause_class, retryability = _classify(
        str(payload.get("error_type", "")),
        payload.get("http_status"),
        str(payload.get("fault", "")),
        payload.get("success"),
    )
    payload["category"] = category
    payload["root_cause_class"] = root_cause_class
    payload["retryability"] = retryability
    return payload


def append_jsonl_event(qa_dir: Path, file_name: str, event: dict[str, Any], source: str) -> None:
    qa_dir.mkdir(parents=True, exist_ok=True)
    payload = enrich_event(qa_dir, event, source)
    out_path = qa_dir / file_name
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    refresh_fault_summary(qa_dir)


def append_fault_events(qa_dir: Path, events: list[dict[str, Any]], source: str = "fault_injection") -> None:
    if not events:
        return
    qa_dir.mkdir(parents=True, exist_ok=True)
    fault_path = qa_dir / "fault_injection.json"
    existing: list[dict[str, Any]] = []
    if fault_path.exists():
        try:
            with open(fault_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing = data
        except (json.JSONDecodeError, OSError):
            existing = []
    enriched = [enrich_event(qa_dir, e, source) for e in events]
    with open(fault_path, "w", encoding="utf-8") as f:
        json.dump(existing + enriched, f, indent=2, ensure_ascii=False)
    refresh_fault_summary(qa_dir)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def _is_failure_event(event: dict[str, Any]) -> bool:
    if str(event.get("status", "")).lower() == "degraded":
        return True
    if "success" in event and not bool(event.get("success")):
        return True
    err = str(event.get("error_type", "")).strip().lower()
    return err not in {"", "none"}


def _dedupe_key(event: dict[str, Any]) -> str:
    return "|".join(
        [
            str(event.get("stage", "")),
            str(event.get("telemetry_source", "")),
            str(event.get("fault", "")),
            str(event.get("step", "")),
            str(event.get("page", "")),
            str(event.get("error_type", "")),
            str(event.get("http_status", "")),
            str(event.get("category", "")),
            str(event.get("root_cause_class", "")),
            str(event.get("retryability", "")),
        ]
    )


def refresh_fault_summary(qa_dir: Path) -> None:
    attempt_id = os.environ.get("INGEST_QA_ATTEMPT_ID", "").strip()
    if not attempt_id:
        return

    all_events: list[dict[str, Any]] = []
    fault_path = qa_dir / "fault_injection.json"
    if fault_path.exists():
        try:
            with open(fault_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_events.extend([d for d in data if isinstance(d, dict)])
        except (json.JSONDecodeError, OSError):
            pass
    all_events.extend(_read_jsonl(qa_dir / "llm_calls.jsonl"))
    all_events.extend(_read_jsonl(qa_dir / "vision_llm_calls.jsonl"))

    attempt_events = [e for e in all_events if str(e.get("attempt_id", "")) == attempt_id]
    attempt_failures = [e for e in attempt_events if _is_failure_event(e)]

    deduped: dict[str, dict[str, Any]] = {}
    for event in attempt_failures:
        key = _dedupe_key(event)
        if key not in deduped:
            deduped[key] = event

    deduped_events = list(deduped.values())
    deduped_events.sort(
        key=lambda e: (
            str(e.get("category", "")),
            str(e.get("stage", "")),
            str(e.get("fault", "")),
            str(e.get("step", "")),
            int(e.get("page", 0) or 0),
        )
    )

    by_category = Counter(str(e.get("category", "unknown")) for e in deduped_events)
    by_root_cause_class = Counter(str(e.get("root_cause_class", "unknown")) for e in deduped_events)
    by_retryability = Counter(str(e.get("retryability", "unknown")) for e in deduped_events)

    observed = [str(e.get("observed_at", "")) for e in attempt_failures if str(e.get("observed_at", ""))]
    summary = {
        "attempt_id": attempt_id,
        "attempt_started_at": os.environ.get("INGEST_QA_ATTEMPT_STARTED_AT", ""),
        "window_event_count": len(attempt_events),
        "window_failure_count_raw": len(attempt_failures),
        "window_failure_count_deduped": len(deduped_events),
        "window_failure_counts_by_category": dict(sorted(by_category.items())),
        "window_failure_counts_by_root_cause_class": dict(sorted(by_root_cause_class.items())),
        "window_failure_counts_by_retryability": dict(sorted(by_retryability.items())),
        "window_observed_at_min": min(observed) if observed else None,
        "window_observed_at_max": max(observed) if observed else None,
        "dedupe_policy": "dedupe by stage/source/fault+step+page/error/http/category/root_cause/retryability",
        "failures_deduped": deduped_events,
    }
    out_path = qa_dir / "fault_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
