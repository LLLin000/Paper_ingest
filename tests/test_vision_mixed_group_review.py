import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ingest.manifest import Manifest
from ingest.vision import run_vision


def _make_manifest() -> Manifest:
    return Manifest(
        doc_id="vision_mixed_group_review_test",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="c" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _write_blocks(run_dir: Path, rows: list[dict[str, object]]) -> None:
    text_dir = run_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = text_dir / "blocks_norm.jsonl"
    with open(blocks_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_page(run_dir: Path, page: int = 1) -> None:
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    image_path = pages_dir / f"p{page:03d}.png"
    Image.new("RGB", (900, 1200), color=(255, 255, 255)).save(image_path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_run_vision_writes_mixed_group_caption_tail_reviews(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_mixed_group_review"
    _write_page(run_dir, page=1)
    _write_blocks(
        run_dir,
        rows=[
            {
                "block_id": "p001_b001",
                "page": 1,
                "bbox_pt": [36.0, 520.0, 282.0, 620.0],
                "text": "Fig. 3 | Key pathogenesis and key regulatory genes of the three distinct rotator cuff tendinopathy subtypes.",
            },
            {
                "block_id": "p001_b002",
                "page": 1,
                "bbox_pt": [306.0, 640.0, 560.0, 710.0],
                "text": (
                    "key drivers for each subtype plotted in its location in the network. "
                    "Each point represents a gene. Points of different colors represent "
                    "different gene co-expression modules."
                ),
            },
        ],
    )

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "8")
    monkeypatch.setenv("SILICONFLOW_VISION_MIXED_REVIEW_ENABLED", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_MIXED_REVIEW_MAX_BLOCKS", "2")

    calls: list[str] = []

    def _fake_call(prompt: str, **_kwargs):
        calls.append(prompt)
        block_ids = re.findall(r'"block_id"\s*:\s*"([^"]+)"', prompt)
        if '"decision"' in prompt and "caption_tail" in prompt:
            assert block_ids
            block_id = block_ids[0]
            raw = json.dumps(
                {
                    "page": 1,
                    "block_id": block_id,
                    "decision": "caption_tail",
                    "caption_kind": "figure",
                    "confidence": 0.97,
                    "reviewed": True,
                }
            )
            meta = {
                "model": "fake-vision-mixed",
                "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
                "success": True,
                "error_type": "none",
                "http_status": 200,
                "prompt_chars": len(prompt),
                "response_chars": len(raw),
                "response_preview": raw[:300],
            }
            return raw, meta

        unique_block_ids = list(dict.fromkeys(block_ids))
        raw = json.dumps(
            {
                "page": 1,
                "reading_order": unique_block_ids,
                "merge_groups": [{"group_id": "mg_0", "block_ids": unique_block_ids}],
                "role_labels": {"p001_b001": "FigureCaption", "p001_b002": "Body"},
                "confidence": 0.92,
            }
        )
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

    assert pages_done == 1
    out = json.loads((run_dir / "vision" / "p001_out.json").read_text(encoding="utf-8"))
    assert out["mixed_group_reviewed_block_ids"] == ["p001_b002"]
    assert out["mixed_group_reviews"] == [
        {
            "block_id": "p001_b002",
            "decision": "caption_tail",
            "caption_kind": "figure",
            "confidence": 0.97,
        }
    ]

    llm_events = _read_jsonl(run_dir / "qa" / "vision_llm_calls.jsonl")
    mixed_events = [event for event in llm_events if event.get("stage") == "vision_region_mixed"]
    assert len(mixed_events) == 1
    assert len(calls) == 2

