"""Vision Structure Corrector with SiliconFlow API and fallback chain.

Contract: .sisyphus/plans/pdf-blueprint-contracts.md (lines 46-74, 276-308)

Output schema per pXXX_out.json:
- page, reading_order, merge_groups, role_labels, confidence, fallback_used
"""

import json
import os
import re
import base64
import io
import urllib.request
import urllib.error
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .manifest import Manifest, load_manifest


ROLE_LABELS = frozenset({
    "Body", "Heading", "FigureCaption", "TableCaption",
    "Footnote", "ReferenceList", "Sidebar", "HeaderFooter",
})

CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.|Table)\b", re.IGNORECASE)

SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SILICONFLOW_FALLBACK_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
VISION_LLM_LOG_LOCK = threading.Lock()


def resolve_api_key() -> str:
    for key_name in ("SILICONFLOW_API_KEY", "SF_API_KEY", "SILICONFLOW_TOKEN"):
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    return ""


@dataclass
class BlockCandidate:
    block_id: str
    text: str
    bbox_pt: list[float]
    bbox_px: list[int]
    column_guess: int
    is_heading_candidate: bool
    is_header_footer_candidate: bool


@dataclass
class VisionOutput:
    page: int
    reading_order: list[str]
    merge_groups: list[dict[str, Any]]
    role_labels: dict[str, str]
    confidence: float
    fallback_used: bool
    source: str = "model"


@dataclass
class FaultEvent:
    stage: str
    fault: str
    page: int
    retry_attempts: int
    fallback_used: bool
    status: str


@dataclass
class VisionLLMCallEvent:
    stage: str
    page: int
    attempt: int
    model: str
    endpoint: str
    success: bool
    parse_success: bool
    validation_success: bool
    error_type: str
    http_status: Optional[int]
    prompt_chars: int
    response_chars: int
    response_preview: str
    timestamp: str


def pt_to_px(bbox_pt: list[float], dpi: int, scale: float) -> list[int]:
    zoom = dpi / 72.0 * scale
    return [int(c * zoom) for c in bbox_pt]


def load_blocks(blocks_path: Path, dpi: int, scale: float) -> dict[int, list[BlockCandidate]]:
    result: dict[int, list[BlockCandidate]] = {}
    if not blocks_path.exists():
        return result
    with open(blocks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if not all(k in d for k in ["block_id", "page", "bbox_pt", "text"]):
                    continue
                bbox_pt = [float(x) for x in d["bbox_pt"][:4]]
                b = BlockCandidate(
                    block_id=d["block_id"],
                    text=d.get("text", ""),
                    bbox_pt=bbox_pt,
                    bbox_px=pt_to_px(bbox_pt, dpi, scale),
                    column_guess=d.get("column_guess", 1),
                    is_heading_candidate=d.get("is_heading_candidate", False),
                    is_header_footer_candidate=d.get("is_header_footer_candidate", False),
                )
                result.setdefault(d["page"], []).append(b)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return result


def build_input_pkg(
    page: int,
    blocks: list[BlockCandidate],
    pages_dir: Path,
    text_limit: int = 220,
    max_blocks: Optional[int] = None,
) -> dict[str, Any]:
    selected_blocks = blocks
    if max_blocks is not None and max_blocks > 0:
        selected_blocks = sorted(blocks, key=lambda b: (b.column_guess, b.bbox_pt[1], b.bbox_pt[0]))[:max_blocks]
    return {
        "page": page,
        "image_path": str(pages_dir / f"p{page:03d}.png"),
        "blocks": [
            {
                "block_id": b.block_id,
                "text": b.text[:text_limit],
                "bbox_pt": b.bbox_pt,
                "bbox_px": b.bbox_px,
                "column_guess": b.column_guess,
            }
            for b in selected_blocks
        ],
        "constraints": {
            "total_blocks": len(blocks),
            "selected_blocks": len(selected_blocks),
            "text_limit": text_limit,
        },
    }


def encode_image_data_url(image_path: Path, max_side: int = 1400) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_model_json(raw: str) -> Optional[dict[str, Any]]:
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        start_obj = raw.find("{")
        if start_obj >= 0:
            raw = raw[start_obj:]
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        req = ["page", "reading_order", "merge_groups", "role_labels", "confidence"]
        if not all(k in data for k in req):
            return None
        if not isinstance(data["reading_order"], list):
            return None
        if not all(isinstance(x, str) for x in data["reading_order"]):
            return None
        if not isinstance(data["merge_groups"], list):
            return None
        for mg in data["merge_groups"]:
            if not isinstance(mg, dict) or "group_id" not in mg or "block_ids" not in mg:
                return None
        if not isinstance(data["role_labels"], dict):
            return None
        if not isinstance(data["confidence"], (int, float)):
            return None
        return data
    except json.JSONDecodeError:
        return None


def detect_role(block: BlockCandidate) -> str:
    txt = block.text.strip()
    if CAPTION_RE.match(txt):
        return "FigureCaption" if txt.lower().startswith(("fig", "figure")) else "TableCaption"
    if block.is_heading_candidate:
        return "Heading"
    if block.is_header_footer_candidate:
        return "HeaderFooter"
    if re.match(r"^\d+\.\s+", txt) or re.match(r"^\[[\d,-]+\]", txt):
        if len(txt) > 50:
            return "ReferenceList"
    return "Body"


def fallback_reading_order(blocks: list[BlockCandidate]) -> list[str]:
    return [b.block_id for b in sorted(blocks, key=lambda x: (x.column_guess, x.bbox_pt[1], x.bbox_pt[0]))]


def fallback_merge_groups(blocks: list[BlockCandidate]) -> list[dict[str, Any]]:
    sorted_blocks = sorted(blocks, key=lambda x: (x.column_guess, x.bbox_pt[1]))
    groups: list[dict[str, Any]] = []
    cur: list[str] = []
    prev: Optional[BlockCandidate] = None
    gid = 0
    GAP = 20.0
    for b in sorted_blocks:
        role = detect_role(b)
        merge = False
        if prev and detect_role(prev) == role == "Body" and b.column_guess == prev.column_guess:
            gap = b.bbox_pt[1] - prev.bbox_pt[3]
            if 0 <= gap <= GAP:
                merge = True
        if merge:
            cur.append(b.block_id)
        else:
            if len(cur) > 1:
                groups.append({"group_id": f"mg_{gid}", "block_ids": cur})
                gid += 1
            cur = [b.block_id]
        prev = b
    if len(cur) > 1:
        groups.append({"group_id": f"mg_{gid}", "block_ids": cur})
    return groups


def fallback_role_labels(blocks: list[BlockCandidate]) -> dict[str, str]:
    return {b.block_id: detect_role(b) for b in blocks}


def generate_fallback(page: int, blocks: list[BlockCandidate]) -> VisionOutput:
    return VisionOutput(
        page=page,
        reading_order=fallback_reading_order(blocks),
        merge_groups=fallback_merge_groups(blocks),
        role_labels=fallback_role_labels(blocks),
        confidence=0.5,
        fallback_used=True,
        source="fallback",
    )


def call_siliconflow(
    prompt: str,
    image_path: Optional[Path] = None,
    image_max_side: int = 1400,
    model_override: Optional[str] = None,
    timeout_seconds: int = 60,
) -> tuple[Any, dict[str, Any]]:
    api_key = resolve_api_key()
    model = (model_override or os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL)).strip()
    endpoint = os.environ.get("SILICONFLOW_ENDPOINT", SILICONFLOW_ENDPOINT)
    meta: dict[str, Any] = {
        "model": model,
        "endpoint": endpoint,
        "success": False,
        "error_type": "unknown",
        "http_status": None,
        "prompt_chars": len(prompt),
        "response_chars": 0,
        "response_preview": "",
    }
    if not api_key:
        meta["error_type"] = "missing_api_key"
        return "{}", meta

    message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_path is not None:
        try:
            image_data_url = encode_image_data_url(image_path, max_side=image_max_side)
            message_content.append({
                "type": "image_url",
                "image_url": {"url": image_data_url},
            })
        except (OSError, ValueError):
            meta["error_type"] = "image_read_error"
            return "{}", meta

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": message_content}],
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", None)
            result = json.loads(resp.read().decode("utf-8"))
            raw_content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            response_text: str
            if isinstance(raw_content, str):
                response_text = raw_content
            elif isinstance(raw_content, list):
                text_parts = [
                    str(part.get("text", ""))
                    for part in raw_content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                response_text = "\n".join(p for p in text_parts if p).strip() or "{}"
            else:
                response_text = "{}"
            meta["http_status"] = status
            meta["response_chars"] = len(response_text)
            meta["response_preview"] = response_text[:300]
            meta["success"] = bool(response_text and response_text != "{}")
            meta["error_type"] = "none" if meta["success"] else "empty_content"
            return response_text, meta
    except urllib.error.HTTPError as e:
        meta["http_status"] = e.code
        meta["error_type"] = "http_error"
        try:
            body = e.read().decode("utf-8", errors="replace")
            meta["response_chars"] = len(body)
            meta["response_preview"] = body[:300]
        except OSError:
            pass
        return "{}", meta
    except urllib.error.URLError:
        meta["error_type"] = "url_error"
        return "{}", meta
    except json.JSONDecodeError:
        meta["error_type"] = "response_json_decode_error"
        return "{}", meta
    except TimeoutError:
        meta["error_type"] = "timeout"
        return "{}", meta
    except KeyError:
        meta["error_type"] = "response_schema_error"
        return "{}", meta
    except Exception as e:  # defensive: keep telemetry informative
        meta["error_type"] = f"exception:{type(e).__name__}"
        meta["response_preview"] = str(e)[:300]
        return "{}", meta


def build_prompt(input_pkg: dict[str, Any]) -> str:
    return f"""Analyze this document page and return STRICT JSON ONLY (no markdown, no prose).

Input:
{json.dumps(input_pkg, indent=2, ensure_ascii=False)}

Output JSON schema:
{{"page": int, "reading_order": ["block_id", ...], "merge_groups": [{{"group_id": str, "block_ids": [...]}}], "role_labels": {{"block_id": "role"}}, "confidence": float}}

Role enum: Body, Heading, FigureCaption, TableCaption, Footnote, ReferenceList, Sidebar, HeaderFooter

Return JSON only:"""


def validate_model_output(parsed: dict[str, Any], expected_page: int, blocks: list[BlockCandidate]) -> tuple[bool, str]:
    known_ids = {b.block_id for b in blocks}
    if not known_ids:
        return False, "no_input_blocks"

    if int(parsed.get("page", -1)) != expected_page:
        return False, "page_mismatch"

    reading_order = parsed.get("reading_order", [])
    if not isinstance(reading_order, list) or not reading_order:
        return False, "empty_reading_order"
    ro_valid = [bid for bid in reading_order if isinstance(bid, str) and bid in known_ids]
    if not ro_valid:
        return False, "reading_order_unknown_ids"
    coverage = len(set(ro_valid)) / max(1, len(known_ids))
    if coverage < 0.35:
        return False, "reading_order_low_coverage"

    role_labels = parsed.get("role_labels", {})
    if not isinstance(role_labels, dict) or not role_labels:
        return False, "empty_role_labels"
    role_valid_count = 0
    for bid, role in role_labels.items():
        if isinstance(bid, str) and bid in known_ids and isinstance(role, str) and role in ROLE_LABELS:
            role_valid_count += 1
    if role_valid_count == 0:
        return False, "invalid_role_labels"

    return True, "ok"


def append_vision_llm_call_event(qa_dir: Path, event: VisionLLMCallEvent) -> None:
    qa_dir.mkdir(parents=True, exist_ok=True)
    out_path = qa_dir / "vision_llm_calls.jsonl"
    with VISION_LLM_LOG_LOCK:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def process_page(
    page: int,
    blocks: list[BlockCandidate],
    pages_dir: Path,
    vision_dir: Path,
    qa_dir: Path,
    inject_malformed: bool,
) -> tuple[VisionOutput, Optional[FaultEvent]]:
    primary_text_limit = int(os.environ.get("SILICONFLOW_VISION_PRIMARY_TEXT_LIMIT", "180"))
    primary_max_blocks = int(os.environ.get("SILICONFLOW_VISION_PRIMARY_MAX_BLOCKS", "280"))
    primary_image_max_side = int(os.environ.get("SILICONFLOW_VISION_PRIMARY_IMAGE_MAX_SIDE", "1300"))
    input_pkg = build_input_pkg(
        page,
        blocks,
        pages_dir,
        text_limit=max(80, primary_text_limit),
        max_blocks=max(80, primary_max_blocks),
    )
    input_path = vision_dir / f"p{page:03d}_in.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_pkg, f, indent=2, ensure_ascii=False)

    fault: Optional[FaultEvent] = None
    retry_attempts = 0
    prompt = build_prompt(input_pkg)

    if inject_malformed:
        raw = "{ malformed json !!! "
        meta: dict[str, Any] = {
            "model": os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL),
            "endpoint": SILICONFLOW_ENDPOINT,
            "success": False,
            "error_type": "injected_malformed_json",
            "http_status": None,
            "prompt_chars": len(prompt),
            "response_chars": len(raw),
            "response_preview": raw[:300],
        }
    else:
        primary_timeout = int(os.environ.get("SILICONFLOW_VISION_PRIMARY_TIMEOUT", "20"))
        raw, meta = call_siliconflow(
            prompt,
            pages_dir / f"p{page:03d}.png",
            image_max_side=max(800, primary_image_max_side),
            timeout_seconds=max(5, primary_timeout),
        )

    parsed = parse_model_json(raw)
    validation_success = False
    validation_reason = "parse_failure"
    if parsed is not None:
        validation_success, validation_reason = validate_model_output(parsed, page, blocks)

    event_error_type = str(meta.get("error_type", "unknown"))
    if parsed is None and event_error_type in {"none", "empty_content"}:
        event_error_type = "parse_failure"
    elif parsed is not None and not validation_success:
        event_error_type = validation_reason

    append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
        stage="vision",
        page=page,
        attempt=1,
        model=str(meta.get("model", SILICONFLOW_DEFAULT_MODEL)),
        endpoint=str(meta.get("endpoint", SILICONFLOW_ENDPOINT)),
        success=bool(meta.get("success", False)),
        parse_success=parsed is not None,
        validation_success=validation_success,
        error_type=event_error_type,
        http_status=meta.get("http_status", None),
        prompt_chars=int(meta.get("prompt_chars", len(prompt))),
        response_chars=int(meta.get("response_chars", len(raw))),
        response_preview=str(meta.get("response_preview", "")),
        timestamp=datetime.now(timezone.utc).isoformat(),
    ))

    if parsed is None or not validation_success:
        retry_attempts = 1
        if not inject_malformed:
            retry_model = os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL).strip()
            first_error_type = str(meta.get("error_type", "unknown"))
            if first_error_type in {"timeout", "url_error", "http_error", "response_json_decode_error", "response_schema_error"}:
                retry_model = os.environ.get("SILICONFLOW_VISION_FALLBACK_MODEL", SILICONFLOW_FALLBACK_MODEL).strip()
            retry_input_pkg = build_input_pkg(
                page,
                blocks,
                pages_dir,
                text_limit=120,
                max_blocks=220,
            )
            retry_prompt = build_prompt(retry_input_pkg)
            retry_timeout = int(os.environ.get("SILICONFLOW_VISION_RETRY_TIMEOUT", "45"))
            raw, meta = call_siliconflow(
                retry_prompt,
                pages_dir / f"p{page:03d}.png",
                image_max_side=1100,
                model_override=retry_model,
                timeout_seconds=max(10, retry_timeout),
            )
            parsed = parse_model_json(raw)
            validation_success = False
            validation_reason = "parse_failure"
            if parsed is not None:
                validation_success, validation_reason = validate_model_output(parsed, page, blocks)

            event_error_type = str(meta.get("error_type", "unknown"))
            if parsed is None and event_error_type in {"none", "empty_content"}:
                event_error_type = "parse_failure"
            elif parsed is not None and not validation_success:
                event_error_type = validation_reason

            append_vision_llm_call_event(qa_dir, VisionLLMCallEvent(
                stage="vision",
                page=page,
                attempt=2,
                model=str(meta.get("model", SILICONFLOW_DEFAULT_MODEL)),
                endpoint=str(meta.get("endpoint", SILICONFLOW_ENDPOINT)),
                success=bool(meta.get("success", False)),
                parse_success=parsed is not None,
                validation_success=validation_success,
                error_type=event_error_type,
                http_status=meta.get("http_status", None),
                prompt_chars=int(meta.get("prompt_chars", len(retry_prompt))),
                response_chars=int(meta.get("response_chars", len(raw))),
                response_preview=str(meta.get("response_preview", "")),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

    if parsed is None or not validation_success:
        output = generate_fallback(page, blocks)
        fault = FaultEvent(
            stage="vision",
            fault="malformed-json" if inject_malformed else f"invalid-model-output:{validation_reason}",
            page=page,
            retry_attempts=retry_attempts,
            fallback_used=True,
            status="degraded",
        )
    else:
        output = VisionOutput(
            page=parsed["page"],
            reading_order=parsed["reading_order"],
            merge_groups=parsed["merge_groups"],
            role_labels=parsed["role_labels"],
            confidence=float(parsed["confidence"]),
            fallback_used=parsed.get("fallback_used", False),
            source="model",
        )

    out_dict = {
        "page": output.page,
        "reading_order": output.reading_order,
        "merge_groups": output.merge_groups,
        "role_labels": output.role_labels,
        "confidence": output.confidence,
        "fallback_used": output.fallback_used,
        "source": output.source,
    }
    out_path = vision_dir / f"p{page:03d}_out.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)

    return output, fault


def run_vision(
    run_dir: Path,
    manifest: Optional[Manifest] = None,
    inject_malformed_json: bool = False,
) -> tuple[int, int]:
    if manifest is None:
        manifest = load_manifest(run_dir)

    dpi = manifest.render_config.dpi
    scale = manifest.render_config.scale

    pages_dir = run_dir / "pages"
    text_dir = run_dir / "text"
    vision_dir = run_dir / "vision"
    qa_dir = run_dir / "qa"

    vision_dir.mkdir(parents=True, exist_ok=True)

    blocks_by_page = load_blocks(text_dir / "blocks_norm.jsonl", dpi, scale)

    if not blocks_by_page:
        return 0, 0

    faults: list[FaultEvent] = []
    pages_done = 0
    blocks_done = 0
    total_pages = len(blocks_by_page)

    max_workers = int(os.environ.get("SILICONFLOW_VISION_MAX_WORKERS", "2"))
    max_workers = max(1, min(4, max_workers))

    if max_workers == 1:
        for index, page in enumerate(sorted(blocks_by_page.keys()), start=1):
            page_start = time.time()
            blocks = blocks_by_page[page]
            output, fault = process_page(page, blocks, pages_dir, vision_dir, qa_dir, inject_malformed_json)
            if fault:
                faults.append(fault)
            pages_done += 1
            blocks_done += len(blocks)
            elapsed = time.time() - page_start
            print(
                f"[vision] {index}/{total_pages} page={page} blocks={len(blocks)} "
                f"source={output.source} fallback={output.fallback_used} elapsed={elapsed:.1f}s",
                flush=True,
            )
    else:
        def process_page_timed(page: int, blocks: list[BlockCandidate]) -> tuple[VisionOutput, Optional[FaultEvent], float]:
            started = time.time()
            output, fault = process_page(page, blocks, pages_dir, vision_dir, qa_dir, inject_malformed_json)
            return output, fault, time.time() - started

        ordered_pages = sorted(blocks_by_page.keys())
        future_map: dict[Any, tuple[int, list[BlockCandidate]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for page in ordered_pages:
                blocks = blocks_by_page[page]
                fut = executor.submit(process_page_timed, page, blocks)
                future_map[fut] = (page, blocks)

            for done_count, fut in enumerate(as_completed(future_map), start=1):
                page, blocks = future_map[fut]
                output, fault, elapsed = fut.result()
                if fault:
                    faults.append(fault)
                pages_done += 1
                blocks_done += len(blocks)
                print(
                    f"[vision] {done_count}/{total_pages} page={page} blocks={len(blocks)} "
                    f"source={output.source} fallback={output.fallback_used} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if faults:
        qa_dir.mkdir(parents=True, exist_ok=True)
        fp = qa_dir / "fault_injection.json"
        existing: list[dict[str, Any]] = []
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    existing = d if isinstance(d, list) else []
            except (json.JSONDecodeError, IOError):
                pass
        all_events = existing + [asdict(e) for e in faults]
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(all_events, f, indent=2, ensure_ascii=False)

    return pages_done, blocks_done
