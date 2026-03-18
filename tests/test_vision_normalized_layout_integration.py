import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ingest.manifest import Manifest
from ingest.vision import run_vision


def _make_manifest() -> Manifest:
    return Manifest(
        doc_id="vision_normalized_layout_test",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="f" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def test_run_vision_uses_normalized_layout_blocks_before_direct_vision(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_normalized_layout"
    pages_dir = run_dir / "pages"
    text_dir = run_dir / "text"
    pages_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (900, 1200), color=(255, 255, 255)).save(pages_dir / "p001.png")

    rows = [
        {
            "block_id": "header_1",
            "page": 1,
            "bbox_pt": [30.0, 10.0, 560.0, 24.0],
            "text": "Nature Communications | Article",
        },
        {
            "block_id": "b1",
            "page": 1,
            "bbox_pt": [40.0, 100.0, 290.0, 145.0],
            "text": "Paragraph part one",
        },
        {
            "block_id": "b2",
            "page": 1,
            "bbox_pt": [40.0, 148.0, 290.0, 188.0],
            "text": "Paragraph part two",
        },
        {
            "block_id": "b3",
            "page": 1,
            "bbox_pt": [310.0, 100.0, 500.0, 128.0],
            "text": "Results",
            "is_heading_candidate": True,
        },
    ]
    with open(text_dir / "blocks_norm.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (text_dir / "document_layout_profile.json").write_text(
        json.dumps(
            {
                "header_band_pt": [0.0, 0.0, 595.0, 30.0],
                "footer_band_pt": [0.0, 760.0, 595.0, 792.0],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (text_dir / "layout_analysis.json").write_text(
        json.dumps(
            {
                "page_layouts": {"1": {"page": 1, "column_count": 2, "column_regions": [[0, 0, 300, 792], [300, 0, 595, 792]]}},
                "paragraph_regrouping_hints": {"1": [["b1", "b2"]]},
                "document_profile": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")

    def _fake_call(prompt: str, **_kwargs):
        block_ids = list(dict.fromkeys(re.findall(r'"block_id"\s*:\s*"([^"]+)"', prompt)))
        raw = json.dumps(
            {
                "page": 1,
                "reading_order": block_ids,
                "merge_groups": [{"group_id": "g1", "block_ids": block_ids}],
                "role_labels": {block_ids[0]: "Body", block_ids[1]: "Heading"},
                "confidence": 0.95,
            }
        )
        return raw, {
            "model": "fake-vision",
            "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
            "success": True,
            "error_type": "none",
            "http_status": 200,
            "prompt_chars": len(prompt),
            "response_chars": len(raw),
            "response_preview": raw[:200],
        }

    monkeypatch.setattr("ingest.vision.call_siliconflow", _fake_call)

    pages_done, _ = run_vision(run_dir, manifest=_make_manifest())

    assert pages_done == 1
    in_payload = json.loads((run_dir / "vision" / "p001_in.json").read_text(encoding="utf-8"))
    assert [block["block_id"] for block in in_payload["blocks"]] == ["p001_rg000", "b3"]

    out_payload = json.loads((run_dir / "vision" / "p001_out.json").read_text(encoding="utf-8"))
    assert out_payload["reading_order"] == ["b1", "b2", "b3"]
    assert out_payload["role_labels"] == {
        "b1": "Body",
        "b2": "Body",
        "b3": "Heading",
        "header_1": "HeaderFooter",
    }
