import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ingest.manifest import Manifest
from ingest.vision import BlockCandidate, classify_page_layout_mode, run_vision


def _block(block_id: str, text: str) -> BlockCandidate:
    return BlockCandidate(
        block_id=block_id,
        text=text,
        bbox_pt=[0.0, 0.0, 10.0, 10.0],
        bbox_px=[0, 0, 10, 10],
        column_guess=1,
        is_heading_candidate=False,
        is_header_footer_candidate=False,
    )


def test_classify_page_layout_mode_flags_dense_microblock_page_as_hierarchical() -> None:
    blocks = [_block(f"b{i}", "0 5 10") for i in range(120)]

    mode = classify_page_layout_mode(blocks)

    assert mode == "hierarchical"


def test_classify_page_layout_mode_keeps_sparse_narrative_page_direct() -> None:
    blocks = [
        _block(
            "b1",
            "The incidence of large bone and articular cartilage defects caused by traumatic injury is increasing worldwide.",
        ),
        _block(
            "b2",
            "Endogenous bioelectrical phenomenon has been well recognized to play an important role in bone and cartilage homeostasis.",
        ),
        _block(
            "b3",
            "Studies have reported that electrical stimulation can effectively regulate various biological processes.",
        ),
    ]

    mode = classify_page_layout_mode(blocks)

    assert mode == "direct"


def test_classify_page_layout_mode_flags_graphic_dense_page_below_legacy_block_threshold() -> None:
    tiny_graphic_texts = [
        "Hw",
        "Iw",
        "Ir",
        "0.8 0.9",
        "1.0 0.8",
        "2 3 4 5 6 7",
        "Avg sil width",
        "Hypoxia",
        "Inflammation",
        "white tendon",
        "red tendon",
        "Hw vs. N",
        "Iw vs. N",
        "Ir vs. N",
        "-log10 (P-Value)",
        "Upregulated genes",
        "Downregulated genes",
    ]
    blocks = [_block(f"b{i}", tiny_graphic_texts[i % len(tiny_graphic_texts)]) for i in range(88)]

    mode = classify_page_layout_mode(blocks)

    assert mode == "hierarchical"


def _make_manifest() -> Manifest:
    return Manifest(
        doc_id="vision_dense_page_mode_test",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="d" * 64,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _write_dense_run(run_dir: Path, block_count: int) -> None:
    pages_dir = run_dir / "pages"
    text_dir = run_dir / "text"
    pages_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (900, 1200), color=(255, 255, 255)).save(pages_dir / "p001.png")
    with open(text_dir / "blocks_norm.jsonl", "w", encoding="utf-8") as f:
        for idx in range(block_count):
            row = {
                "block_id": f"p001_b{idx:03d}",
                "page": 1,
                "bbox_pt": [36.0, float(40 + idx * 3), 96.0, float(52 + idx * 3)],
                "text": "0 5 10",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_vision_uses_smaller_input_budget_for_dense_pages(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_dense_input_budget"
    _write_dense_run(run_dir, block_count=120)

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")
    monkeypatch.setenv("SILICONFLOW_VISION_DENSE_MAX_BLOCKS", "40")
    monkeypatch.setenv("SILICONFLOW_VISION_DENSE_TEXT_LIMIT", "60")

    def _fake_call(prompt: str, **_kwargs):
        if "coarse page-region segmentation" in prompt:
            raw = json.dumps(
                {
                    "page": 1,
                    "text_regions": [{"region_id": "text_1", "bbox_px": [0, 0, 900, 1200]}],
                    "caption_regions": [],
                    "figure_regions": [],
                    "table_regions": [],
                    "header_footer_regions": [],
                    "confidence": 0.9,
                }
            )
            return raw, {
                "model": "fake-vision-coarse",
                "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
                "success": True,
                "error_type": "none",
                "http_status": 200,
                "prompt_chars": len(prompt),
                "response_chars": len(raw),
                "response_preview": raw[:200],
            }
        block_ids = list(dict.fromkeys(re.findall(r'"block_id"\s*:\s*"([^"]+)"', prompt)))
        raw = json.dumps(
            {
                "page": 1,
                "reading_order": block_ids,
                "merge_groups": [],
                "role_labels": {block_id: "Body" for block_id in block_ids},
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
    assert in_payload["constraints"]["layout_mode"] == "hierarchical"
    assert in_payload["constraints"]["selected_blocks"] == 40
    assert in_payload["constraints"]["text_limit"] == 60
