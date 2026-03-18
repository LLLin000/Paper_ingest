import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ingest.manifest import Manifest
from ingest.vision import run_vision


def _make_manifest() -> Manifest:
    return Manifest(
        doc_id="vision_fallback_retry_test",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="c" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _write_pages(run_dir: Path, pages: int) -> None:
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for page in range(1, pages + 1):
        Image.new("RGB", (900, 1200), color=(255, 255, 255)).save(pages_dir / f"p{page:03d}.png")


def _write_blocks(run_dir: Path, pages: int) -> None:
    text_dir = run_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    with open(text_dir / "blocks_norm.jsonl", "w", encoding="utf-8") as f:
        for page in range(1, pages + 1):
            row = {
                "block_id": f"p{page:03d}_b001",
                "page": page,
                "bbox_pt": [36.0, 72.0, 280.0, 200.0],
                "text": f"Body text for page {page}.",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_vision_retries_page_after_fallback_and_recovers(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_page_retry"
    _write_pages(run_dir, pages=1)
    _write_blocks(run_dir, pages=1)

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_PAGE_FALLBACK_RETRIES", "2")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")

    calls: list[int] = []

    def _fake_call(_prompt: str, **_kwargs):
        calls.append(1)
        if len(calls) <= 2:
            return "{}", {
                "model": "fake-vision",
                "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
                "success": False,
                "error_type": "empty_content",
                "http_status": 200,
                "prompt_chars": 10,
                "response_chars": 2,
                "response_preview": "{}",
            }
        raw = json.dumps(
            {
                "page": 1,
                "reading_order": ["p001_b001"],
                "merge_groups": [],
                "role_labels": {"p001_b001": "Body"},
                "confidence": 0.93,
            }
        )
        return raw, {
            "model": "fake-vision",
            "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
            "success": True,
            "error_type": "none",
            "http_status": 200,
            "prompt_chars": 10,
            "response_chars": len(raw),
            "response_preview": raw[:200],
        }

    monkeypatch.setattr("ingest.vision.call_siliconflow", _fake_call)

    pages_done, _ = run_vision(run_dir, manifest=_make_manifest())

    assert pages_done == 1
    out = json.loads((run_dir / "vision" / "p001_out.json").read_text(encoding="utf-8"))
    assert out["fallback_used"] is False
    assert out["source"] == "model"
    assert len(calls) == 3


def test_run_vision_switches_to_alternate_model_after_consecutive_fallbacks(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_alt_model_after_streak"
    _write_pages(run_dir, pages=3)
    _write_blocks(run_dir, pages=3)

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MODEL", "primary-model")
    monkeypatch.setenv("SILICONFLOW_VISION_FALLBACK_MODEL", "retry-model")
    monkeypatch.setenv("SILICONFLOW_VISION_CONSEC_FALLBACK_MODEL", "alt-model")
    monkeypatch.setenv("SILICONFLOW_VISION_CONSEC_FALLBACK_SWITCH_THRESHOLD", "2")
    monkeypatch.setenv("SILICONFLOW_VISION_PAGE_FALLBACK_RETRIES", "0")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "20")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")

    model_overrides: list[str | None] = []

    def _fake_call(prompt: str, **kwargs):
        model_override = kwargs.get("model_override")
        model_overrides.append(model_override)
        if model_override == "alt-model" and '"page": 3' in prompt:
            raw = json.dumps(
                {
                    "page": 3,
                    "reading_order": ["p003_b001"],
                    "merge_groups": [],
                    "role_labels": {"p003_b001": "Body"},
                    "confidence": 0.95,
                }
            )
            return raw, {
                "model": "alt-model",
                "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
                "success": True,
                "error_type": "none",
                "http_status": 200,
                "prompt_chars": 10,
                "response_chars": len(raw),
                "response_preview": raw[:200],
            }
        return "{}", {
            "model": model_override or "primary-model",
            "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
            "success": False,
            "error_type": "empty_content",
            "http_status": 200,
            "prompt_chars": 10,
            "response_chars": 2,
            "response_preview": "{}",
        }

    monkeypatch.setattr("ingest.vision.call_siliconflow", _fake_call)

    pages_done, _ = run_vision(run_dir, manifest=_make_manifest())

    assert pages_done == 3
    page1 = json.loads((run_dir / "vision" / "p001_out.json").read_text(encoding="utf-8"))
    page2 = json.loads((run_dir / "vision" / "p002_out.json").read_text(encoding="utf-8"))
    page3 = json.loads((run_dir / "vision" / "p003_out.json").read_text(encoding="utf-8"))
    assert page1["source"] == "fallback"
    assert page2["source"] == "fallback"
    assert page3["source"] == "model"
    assert "alt-model" in model_overrides
