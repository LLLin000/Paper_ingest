import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ingest.manifest import Manifest
from ingest.vision import run_vision


def _make_manifest() -> Manifest:
    return Manifest(
        doc_id="vision_hierarchical_flow_test",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="e" * 64,
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


def test_run_vision_writes_coarse_region_artifact_for_dense_page(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_hierarchical_regions"
    _write_dense_run(run_dir, block_count=120)

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")
    monkeypatch.setenv("SILICONFLOW_VISION_DENSE_MAX_BLOCKS", "40")
    monkeypatch.setenv("SILICONFLOW_VISION_DENSE_TEXT_LIMIT", "60")

    coarse_calls: list[str] = []
    direct_calls: list[str] = []

    def _fake_call(prompt: str, **_kwargs):
        if "coarse page-region segmentation" in prompt:
            coarse_calls.append(prompt)
            raw = json.dumps(
                {
                    "page": 1,
                    "text_regions": [],
                    "caption_regions": [],
                    "figure_regions": [{"region_id": "fig_1", "bbox_px": [0, 0, 900, 1200]}],
                    "table_regions": [],
                    "header_footer_regions": [],
                    "confidence": 0.88,
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

        direct_calls.append(prompt)
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
    assert len(coarse_calls) == 1
    assert len(direct_calls) == 0

    coarse_path = run_dir / "vision" / "p001_regions.json"
    assert coarse_path.exists()
    payload = json.loads(coarse_path.read_text(encoding="utf-8"))
    assert payload["page"] == 1
    assert payload["figure_regions"] == [{"region_id": "fig_1", "bbox_px": [0, 0, 900, 1200]}]

    out_path = run_dir / "vision" / "p001_out.json"
    out_payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert out_payload["source"] == "coarse_fallback"
    assert out_payload["fallback_used"] is True


def test_hierarchical_dense_page_scopes_fine_layout_to_text_and_caption_regions(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_hierarchical_scope"
    _write_dense_run(run_dir, block_count=120)

    text_dir = run_dir / "text"
    with open(text_dir / "blocks_norm.jsonl", "w", encoding="utf-8") as f:
        rows = [
            {
                "block_id": "p001_b000",
                "page": 1,
                "bbox_pt": [36.0, 40.0, 260.0, 80.0],
                "text": "Results To better understand the heterogeneity of tendinopathy.",
            },
            {
                "block_id": "p001_b001",
                "page": 1,
                "bbox_pt": [36.0, 90.0, 260.0, 120.0],
                "text": "Fig. 1 | Summary of the distinct subtypes.",
            },
        ]
        for idx in range(2, 120):
            rows.append(
                {
                    "block_id": f"p001_b{idx:03d}",
                    "page": 1,
                    "bbox_pt": [500.0, float(40 + idx * 5), 520.0, float(52 + idx * 5)],
                    "text": "0 5 10",
                }
            )
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")
    monkeypatch.setenv("SILICONFLOW_VISION_DENSE_MAX_BLOCKS", "40")
    monkeypatch.setenv("SILICONFLOW_VISION_DENSE_TEXT_LIMIT", "60")

    direct_block_sets: list[list[str]] = []

    def _fake_call(prompt: str, **_kwargs):
        if "coarse page-region segmentation" in prompt:
            raw = json.dumps(
                {
                    "page": 1,
                    "text_regions": [{"region_id": "text_1", "bbox_px": [0, 0, 800, 340]}],
                    "caption_regions": [{"region_id": "cap_1", "bbox_px": [0, 340, 800, 520]}],
                    "figure_regions": [{"region_id": "fig_1", "bbox_px": [1800, 0, 2400, 2800]}],
                    "table_regions": [],
                    "header_footer_regions": [],
                    "confidence": 0.88,
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

        block_ids = list(dict.fromkeys(re.findall(r'"block_id"\s*:\s*"(p\d{3}_b\d+)"', prompt)))
        direct_block_sets.append(block_ids)
        raw = json.dumps(
            {
                "page": 1,
                "reading_order": block_ids,
                "merge_groups": [],
                "role_labels": {block_id: ("FigureCaption" if block_id == "p001_b001" else "Body") for block_id in block_ids},
                "confidence": 0.95,
            }
        )
        return raw, {
            "model": "fake-vision-direct",
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
    assert len(direct_block_sets) == 1
    assert direct_block_sets[0] == ["p001_b000", "p001_b001"]


def test_hierarchical_dense_page_prunes_non_narrative_microblocks_from_wide_text_region(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run" / "vision_hierarchical_text_prune"
    _write_dense_run(run_dir, block_count=120)

    text_dir = run_dir / "text"
    with open(text_dir / "blocks_norm.jsonl", "w", encoding="utf-8") as f:
        rows = [
            {
                "block_id": "p001_b000",
                "page": 1,
                "bbox_pt": [36.0, 40.0, 260.0, 80.0],
                "text": "Results To better understand the heterogeneity of tendinopathy.",
            },
            {
                "block_id": "p001_b001",
                "page": 1,
                "bbox_pt": [36.0, 90.0, 260.0, 120.0],
                "text": "Fig. 1 | Summary of the distinct subtypes.",
            },
        ]
        for idx in range(2, 120):
            rows.append(
                {
                    "block_id": f"p001_b{idx:03d}",
                    "page": 1,
                    "bbox_pt": [500.0, float(40 + idx * 5), 520.0, float(52 + idx * 5)],
                    "text": "0 5 10",
                }
            )
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")
    monkeypatch.setenv("SILICONFLOW_VISION_DENSE_MAX_BLOCKS", "40")
    monkeypatch.setenv("SILICONFLOW_VISION_DENSE_TEXT_LIMIT", "60")

    direct_block_sets: list[list[str]] = []

    def _fake_call(prompt: str, **_kwargs):
        if "coarse page-region segmentation" in prompt:
            raw = json.dumps(
                {
                    "page": 1,
                    "text_regions": [{"region_id": "text_1", "bbox_px": [0, 0, 2400, 340]}],
                    "caption_regions": [{"region_id": "cap_1", "bbox_px": [0, 340, 800, 520]}],
                    "figure_regions": [],
                    "table_regions": [],
                    "header_footer_regions": [],
                    "confidence": 0.88,
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

        block_ids = list(dict.fromkeys(re.findall(r'"block_id"\s*:\s*"(p\d{3}_b\d+)"', prompt)))
        direct_block_sets.append(block_ids)
        raw = json.dumps(
            {
                "page": 1,
                "reading_order": block_ids,
                "merge_groups": [],
                "role_labels": {block_id: ("FigureCaption" if block_id == "p001_b001" else "Body") for block_id in block_ids},
                "confidence": 0.95,
            }
        )
        return raw, {
            "model": "fake-vision-direct",
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
    assert len(direct_block_sets) == 1
    assert direct_block_sets[0] == ["p001_b000", "p001_b001"]


def test_hierarchical_dense_page_uses_coarse_fallback_when_no_fine_layout_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run" / "vision_hierarchical_coarse_fallback"
    _write_dense_run(run_dir, block_count=120)

    text_dir = run_dir / "text"
    with open(text_dir / "blocks_norm.jsonl", "w", encoding="utf-8") as f:
        rows = [
            {
                "block_id": "p001_b000",
                "page": 1,
                "bbox_pt": [10.0, 5.0, 120.0, 15.0],
                "text": "Nature Communications",
            },
        ]
        for idx in range(1, 120):
            rows.append(
                {
                    "block_id": f"p001_b{idx:03d}",
                    "page": 1,
                    "bbox_pt": [500.0, float(40 + idx * 5), 520.0, float(52 + idx * 5)],
                    "text": "0 5 10",
                }
            )
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")

    call_stages: list[str] = []

    def _fake_call(prompt: str, **_kwargs):
        if "coarse page-region segmentation" in prompt:
            call_stages.append("coarse")
            raw = json.dumps(
                {
                    "page": 1,
                    "text_regions": [],
                    "caption_regions": [],
                    "figure_regions": [{"region_id": "fig_1", "bbox_px": [1700, 0, 2400, 2800]}],
                    "table_regions": [],
                    "header_footer_regions": [{"region_id": "hf_1", "bbox_px": [0, 0, 800, 80]}],
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

        call_stages.append("direct")
        raise AssertionError("direct vision call should be skipped when no fine layout candidates survive")

    monkeypatch.setattr("ingest.vision.call_siliconflow", _fake_call)

    pages_done, _ = run_vision(run_dir, manifest=_make_manifest())

    assert pages_done == 1
    assert call_stages == ["coarse"]

    out_path = run_dir / "vision" / "p001_out.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source"] == "coarse_fallback"
    assert payload["fallback_used"] is True
    assert payload["role_labels"]["p001_b000"] == "HeaderFooter"
    assert payload["role_labels"]["p001_b001"] == "Sidebar"


def test_hierarchical_dense_page_coarse_prompt_uses_seed_block_subset(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_hierarchical_coarse_seed"
    _write_dense_run(run_dir, block_count=120)

    text_dir = run_dir / "text"
    with open(text_dir / "blocks_norm.jsonl", "w", encoding="utf-8") as f:
        rows = [
            {
                "block_id": "p001_b000",
                "page": 1,
                "bbox_pt": [10.0, 5.0, 120.0, 15.0],
                "text": "Nature Communications",
            },
            {
                "block_id": "p001_b001",
                "page": 1,
                "bbox_pt": [20.0, 80.0, 260.0, 120.0],
                "text": "Results To better understand the heterogeneity of tendinopathy.",
            },
            {
                "block_id": "p001_b002",
                "page": 1,
                "bbox_pt": [20.0, 130.0, 260.0, 160.0],
                "text": "Fig. 1 | Classification of distinct tendinopathy subtypes.",
            },
        ]
        for idx in range(3, 120):
            rows.append(
                {
                    "block_id": f"p001_b{idx:03d}",
                    "page": 1,
                    "bbox_pt": [500.0, float(40 + idx * 5), 520.0, float(52 + idx * 5)],
                    "text": "0 5 10",
                }
            )
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")
    monkeypatch.setenv("SILICONFLOW_VISION_COARSE_MAX_BLOCKS", "12")

    coarse_block_sets: list[list[str]] = []

    def _fake_call(prompt: str, **_kwargs):
        if "coarse page-region segmentation" in prompt:
            block_ids = list(dict.fromkeys(re.findall(r'"block_id"\s*:\s*"(p\d{3}_b\d+)"', prompt)))
            coarse_block_sets.append(block_ids)
            raw = json.dumps(
                {
                    "page": 1,
                    "text_regions": [],
                    "caption_regions": [],
                    "figure_regions": [{"region_id": "fig_1", "bbox_px": [1700, 0, 2400, 2800]}],
                    "table_regions": [],
                    "header_footer_regions": [{"region_id": "hf_1", "bbox_px": [0, 0, 800, 80]}],
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

        raise AssertionError("direct vision call should be skipped for this coarse-fallback test")

    monkeypatch.setattr("ingest.vision.call_siliconflow", _fake_call)

    pages_done, _ = run_vision(run_dir, manifest=_make_manifest())

    assert pages_done == 1
    assert len(coarse_block_sets) == 1
    assert "p001_b000" in coarse_block_sets[0]
    assert "p001_b001" in coarse_block_sets[0]
    assert "p001_b002" in coarse_block_sets[0]
    assert len(coarse_block_sets[0]) <= 12


def test_hierarchical_dense_page_skips_direct_vision_when_coarse_layout_times_out(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run" / "vision_hierarchical_coarse_timeout"
    _write_dense_run(run_dir, block_count=120)

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "10")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "0")
    monkeypatch.setenv("SILICONFLOW_VISION_PAGE_FALLBACK_RETRIES", "0")

    call_stages: list[str] = []

    def _fake_call(prompt: str, **_kwargs):
        if "coarse page-region segmentation" in prompt:
            call_stages.append("coarse")
            return "{}", {
                "model": "fake-vision-coarse",
                "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
                "success": False,
                "error_type": "timeout",
                "http_status": None,
                "prompt_chars": len(prompt),
                "response_chars": 2,
                "response_preview": "{}",
            }
        call_stages.append("direct")
        raise AssertionError("direct vision call should be skipped when coarse layout is unavailable")

    monkeypatch.setattr("ingest.vision.call_siliconflow", _fake_call)

    pages_done, _ = run_vision(run_dir, manifest=_make_manifest())

    assert pages_done == 1
    assert call_stages == ["coarse"]

    out_path = run_dir / "vision" / "p001_out.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source"] == "hierarchical_fallback"
    assert payload["fallback_used"] is True
