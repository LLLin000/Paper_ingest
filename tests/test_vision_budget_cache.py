import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ingest.manifest import Manifest
from ingest.vision import run_vision


def _make_manifest() -> Manifest:
    return Manifest(
        doc_id="vision_task6_test",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="a" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _write_blocks(run_dir: Path, pages: int) -> None:
    text_dir = run_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = text_dir / "blocks_norm.jsonl"
    with open(blocks_path, "w", encoding="utf-8") as f:
        for page in range(1, pages + 1):
            row = {
                "block_id": f"p{page:03d}_b001",
                "page": page,
                "bbox_pt": [10.0, 20.0, 200.0, 80.0],
                "text": f"Figure {page}. Caption text",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_pages(run_dir: Path, pages: int) -> None:
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for page in range(1, pages + 1):
        image_path = pages_dir / f"p{page:03d}.png"
        Image.new("RGB", (240, 180), color=(255, 255, 255)).save(image_path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_page_and_block(prompt: str) -> tuple[int, str]:
    page_match = re.search(r'"page"\s*:\s*(\d+)', prompt)
    block_match = re.search(r'"block_id"\s*:\s*"([^"]+)"', prompt)
    assert page_match is not None
    assert block_match is not None
    return int(page_match.group(1)), block_match.group(1)


def test_run_vision_budget_exhaustion_reduces_calls_and_emits_telemetry(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_budget"
    _write_pages(run_dir, pages=30)
    _write_blocks(run_dir, pages=30)

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "8")

    calls: list[int] = []

    def _fake_call(prompt: str, **_kwargs):
        page, block_id = _parse_page_and_block(prompt)
        calls.append(page)
        raw = json.dumps({
            "page": page,
            "reading_order": [block_id],
            "merge_groups": [],
            "role_labels": {block_id: "Body"},
            "confidence": 0.95,
        })
        meta = {
            "model": "fake-vision",
            "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
            "success": True,
            "error_type": "none",
            "http_status": 200,
            "prompt_chars": len(prompt),
            "response_chars": len(raw),
            "response_preview": raw[:300],
        }
        return raw, meta

    monkeypatch.setattr("ingest.vision.call_siliconflow", _fake_call)

    pages_done, _ = run_vision(run_dir, manifest=_make_manifest())

    assert pages_done == 30
    assert len(calls) == 8

    page30_out = json.loads((run_dir / "vision" / "p030_out.json").read_text(encoding="utf-8"))
    assert page30_out["fallback_used"] is True
    assert page30_out["source"] == "fallback"

    llm_events = _read_jsonl(run_dir / "qa" / "vision_llm_calls.jsonl")
    exhausted = [e for e in llm_events if e.get("error_type") == "budget_exhausted"]
    assert len(exhausted) == 22
    assert all(e.get("budget_limit") == 8 for e in exhausted)
    assert all(e.get("budget_remaining_after") == 0 for e in exhausted)

    runtime_events = _read_jsonl(run_dir / "qa" / "vision_runtime.jsonl")
    assert runtime_events[-1].get("budget_consumed") == 8
    assert runtime_events[-1].get("budget_remaining") == 0


def test_run_vision_uses_image_encoding_cache_for_retry_attempts(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_cache"
    _write_pages(run_dir, pages=1)
    _write_blocks(run_dir, pages=1)

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "5")
    monkeypatch.setenv("SILICONFLOW_VISION_PRIMARY_IMAGE_MAX_SIDE", "1100")
    monkeypatch.setenv("SILICONFLOW_VISION_RETRY_BACKOFF_SECONDS", "0")

    encode_calls: list[tuple[str, int]] = []

    def _fake_encode(image_path: Path, max_side: int = 1400) -> str:
        encode_calls.append((image_path.name, max_side))
        return f"data:image/jpeg;base64,stub-{image_path.name}-{max_side}"

    attempts = {"count": 0}

    def _fake_call(prompt: str, **_kwargs):
        page, block_id = _parse_page_and_block(prompt)
        attempts["count"] += 1
        if attempts["count"] == 1:
            raw = "{}"
            meta = {
                "model": "fake-vision",
                "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
                "success": True,
                "error_type": "none",
                "http_status": 200,
                "prompt_chars": len(prompt),
                "response_chars": len(raw),
                "response_preview": raw,
            }
            return raw, meta
        raw = json.dumps({
            "page": page,
            "reading_order": [block_id],
            "merge_groups": [],
            "role_labels": {block_id: "Body"},
            "confidence": 0.88,
        })
        meta = {
            "model": "fake-vision",
            "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
            "success": True,
            "error_type": "none",
            "http_status": 200,
            "prompt_chars": len(prompt),
            "response_chars": len(raw),
            "response_preview": raw[:300],
        }
        return raw, meta

    monkeypatch.setattr("ingest.vision.encode_image_data_url", _fake_encode)
    monkeypatch.setattr("ingest.vision.call_siliconflow", _fake_call)

    pages_done, _ = run_vision(run_dir, manifest=_make_manifest())

    assert pages_done == 1
    assert attempts["count"] == 2
    assert len(encode_calls) == 1

    llm_events = _read_jsonl(run_dir / "qa" / "vision_llm_calls.jsonl")
    assert len(llm_events) == 2
    assert llm_events[0].get("cache_miss") is True
    assert llm_events[1].get("cache_hit") is True

    runtime_events = _read_jsonl(run_dir / "qa" / "vision_runtime.jsonl")
    assert runtime_events[-1].get("cache_hits") == 1
    assert runtime_events[-1].get("cache_misses") == 1
