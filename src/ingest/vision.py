"""Vision Structure Corrector with SiliconFlow API and fallback chain.

Contract: .sisyphus/plans/pdf-blueprint-contracts.md (lines 46-74, 276-308)

Output schema per pXXX_out.json:
- page, reading_order, merge_groups, role_labels, confidence, fallback_used
"""

import json
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from .manifest import Manifest, load_manifest


ROLE_LABELS = frozenset({
    "Body", "Heading", "FigureCaption", "TableCaption",
    "Footnote", "ReferenceList", "Sidebar", "HeaderFooter",
})

CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.|Table)\b", re.IGNORECASE)

SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_DEFAULT_MODEL = "THUDM/GLM-4.1V-9B"


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


def build_input_pkg(page: int, blocks: list[BlockCandidate], pages_dir: Path) -> dict[str, Any]:
    return {
        "page": page,
        "image_path": str(pages_dir / f"p{page:03d}.png"),
        "blocks": [
            {
                "block_id": b.block_id,
                "text": b.text,
                "bbox_pt": b.bbox_pt,
                "bbox_px": b.bbox_px,
                "column_guess": b.column_guess,
            }
            for b in blocks
        ],
        "constraints": {"total_blocks": len(blocks)},
    }


def parse_model_json(raw: str) -> Optional[dict[str, Any]]:
    try:
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


def call_siliconflow(prompt: str) -> str:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    model = os.environ.get("SILICONFLOW_VISION_MODEL", SILICONFLOW_DEFAULT_MODEL)
    if not api_key:
        return json.dumps({
            "page": 0,
            "reading_order": [],
            "merge_groups": [],
            "role_labels": {},
            "confidence": 0.0,
            "error": "SILICONFLOW_API_KEY not set",
        })
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        SILICONFLOW_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, TimeoutError):
        return "{}"


def build_prompt(input_pkg: dict[str, Any]) -> str:
    return f"""Analyze this document page and return STRICT JSON ONLY (no markdown, no prose).

Input:
{json.dumps(input_pkg, indent=2, ensure_ascii=False)}

Output JSON schema:
{{"page": int, "reading_order": ["block_id", ...], "merge_groups": [{{"group_id": str, "block_ids": [...]}}], "role_labels": {{"block_id": "role"}}, "confidence": float}}

Role enum: Body, Heading, FigureCaption, TableCaption, Footnote, ReferenceList, Sidebar, HeaderFooter

Return JSON only:"""


def process_page(
    page: int,
    blocks: list[BlockCandidate],
    pages_dir: Path,
    vision_dir: Path,
    inject_malformed: bool,
) -> tuple[VisionOutput, Optional[FaultEvent]]:
    input_pkg = build_input_pkg(page, blocks, pages_dir)
    input_path = vision_dir / f"p{page:03d}_in.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_pkg, f, indent=2, ensure_ascii=False)

    fault: Optional[FaultEvent] = None
    retry_attempts = 0
    prompt = build_prompt(input_pkg)

    if inject_malformed:
        raw = "{ malformed json !!! "
    else:
        raw = call_siliconflow(prompt)

    parsed = parse_model_json(raw)

    if parsed is None:
        retry_attempts = 1
        if not inject_malformed:
            raw = call_siliconflow(prompt)
            parsed = parse_model_json(raw)

    if parsed is None:
        output = generate_fallback(page, blocks)
        fault = FaultEvent(
            stage="vision",
            fault="malformed-json" if inject_malformed else "parse-failure",
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

    for page in sorted(blocks_by_page.keys()):
        blocks = blocks_by_page[page]
        _, fault = process_page(page, blocks, pages_dir, vision_dir, inject_malformed_json)
        if fault:
            faults.append(fault)
        pages_done += 1
        blocks_done += len(blocks)

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
