import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ingest.manifest import Manifest
from ingest.vision import (
    BlockCandidate,
    VisionOutput,
    collect_region_heading_review_candidates,
    run_vision,
    validate_region_heading_review_output,
)


def _make_manifest() -> Manifest:
    return Manifest(
        doc_id="vision_region_heading_test",
        input_pdf_path="dummy.pdf",
        input_pdf_sha256="b" * 64,
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


def test_run_vision_writes_embedded_heading_hints_for_reviewed_blocks(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run" / "vision_region_heading"
    _write_page(run_dir, page=1)
    _write_blocks(
        run_dir,
        rows=[
            {
                "block_id": "p001_b001",
                "page": 1,
                "bbox_pt": [36.0, 160.0, 282.0, 430.0],
                "text": (
                    "Results Identification of three distinct subtypes of tendinopathy based on "
                    "transcriptome profiles and clinical features To better understand the "
                    "heterogeneity of tendinopathy and identify potential subtypes, we collected "
                    "clinical data. Principal compo- nent analysis demonstrated a clear distinction. "
                    "Therefore, based on the RNA-seq data, tendinopathy can be classified into two "
                    "distinct molecular subtypes."
                ),
            },
            {
                "block_id": "p001_b002",
                "page": 1,
                "bbox_pt": [36.0, 460.0, 282.0, 620.0],
                "text": (
                    "RNA sequence Total RNA was extracted from the tissue using TRIzol Reagent "
                    "according to the manufacturer instructions. After quantification by TBS380, "
                    "Paired-end libraries were sequenced with the Illumina HiSeq PE 2X150bp read length."
                ),
            },
        ],
    )

    monkeypatch.setenv("SILICONFLOW_API_KEY", "token")
    monkeypatch.setenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
    monkeypatch.setenv("SILICONFLOW_VISION_MAX_WORKERS", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_REQUEST_BUDGET", "8")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_ENABLED", "1")
    monkeypatch.setenv("SILICONFLOW_VISION_HEADING_REVIEW_MAX_BLOCKS", "2")

    calls: list[str] = []

    def _fake_call(prompt: str, **_kwargs):
        calls.append(prompt)
        block_ids = re.findall(r'"block_id"\s*:\s*"([^"]+)"', prompt)
        if "section-heading recovery" in prompt:
            assert block_ids
            block_id = block_ids[0]
            accepted_map = {
                "p001_b001": [
                    "Results",
                    "Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features",
                ],
                "p001_b002": [
                    "RNA sequence",
                ],
            }
            raw = json.dumps(
                {
                    "page": 1,
                    "block_id": block_id,
                    "accepted_headings": [
                        {"heading_text": heading, "confidence": 0.96}
                        for heading in accepted_map.get(block_id, [])
                    ],
                    "reviewed": True,
                }
            )
            meta = {
                "model": "fake-vision-region",
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
                "role_labels": {block_id: "Body" for block_id in unique_block_ids},
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
    assert out["embedded_heading_reviewed_block_ids"] == ["p001_b001", "p001_b002"]
    assert {(item["block_id"], item["heading_text"]) for item in out["embedded_headings"]} == {
        ("p001_b001", "Results"),
        ("p001_b001", "Identification of three distinct subtypes of tendinopathy based on transcriptome profiles and clinical features"),
        ("p001_b002", "RNA sequence"),
    }

    llm_events = _read_jsonl(run_dir / "qa" / "vision_llm_calls.jsonl")
    region_events = [event for event in llm_events if event.get("stage") == "vision_region_heading"]
    assert len(region_events) == 2
    assert len(calls) == 3


def test_validate_region_heading_review_accepts_ligature_normalized_headings() -> None:
    block = BlockCandidate(
        block_id="p2_b2",
        text="",
        bbox_pt=[0.0, 0.0, 100.0, 100.0],
        bbox_px=[0, 0, 100, 100],
        column_guess=1,
        is_heading_candidate=False,
        is_header_footer_candidate=False,
    )
    parsed = {
        "page": 2,
        "block_id": "p2_b2",
        "accepted_headings": [
            {"heading_text": "Results", "confidence": 1.0},
            {
                "heading_text": (
                    "Identification of three distinct subtypes of tendinopathy "
                    "based on transcriptome profiles and clinical features"
                ),
                "confidence": 1.0,
            },
        ],
        "reviewed": True,
    }

    ok, reason = validate_region_heading_review_output(
        parsed,
        page=2,
        block=block,
        candidate_headings=[
            "Results",
            "Identiﬁcation of three distinct subtypes of tendinopathy based on transcriptome proﬁles and clinical features",
        ],
    )

    assert ok is True
    assert reason == "ok"


def test_collect_region_heading_review_candidates_prioritizes_start_aligned_blocks() -> None:
    output = VisionOutput(
        page=2,
        reading_order=["late_body", "real_heading"],
        merge_groups=[],
        role_labels={"late_body": "Body", "real_heading": "Body"},
        confidence=0.9,
        fallback_used=False,
    )
    blocks = [
        BlockCandidate(
            block_id="late_body",
            text=(
                "Body paragraph before the heading. Therefore, based on the RNA-seq data, "
                "tendinopathy can be classified into two distinct molecular subtypes."
            ),
            bbox_pt=[0.0, 0.0, 100.0, 100.0],
            bbox_px=[0, 0, 100, 100],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
        ),
        BlockCandidate(
            block_id="real_heading",
            text=(
                "Results Identification of three distinct subtypes of tendinopathy based on "
                "transcriptome profiles and clinical features To better understand heterogeneity."
            ),
            bbox_pt=[0.0, 120.0, 100.0, 220.0],
            bbox_px=[0, 120, 100, 220],
            column_guess=1,
            is_heading_candidate=False,
            is_header_footer_candidate=False,
        ),
    ]

    candidates = collect_region_heading_review_candidates(blocks, output, max_blocks=1, max_candidates=4)

    assert len(candidates) == 1
    assert candidates[0][0].block_id == "real_heading"
    assert candidates[0][1][0] == "Results"


def test_collect_region_heading_review_candidates_includes_overlong_heading_blocks() -> None:
    output = VisionOutput(
        page=2,
        reading_order=["plain_heading", "mixed_heading"],
        merge_groups=[],
        role_labels={"plain_heading": "Heading", "mixed_heading": "Heading"},
        confidence=0.9,
        fallback_used=False,
    )
    blocks = [
        BlockCandidate(
            block_id="plain_heading",
            text="Discussion",
            bbox_pt=[0.0, 0.0, 100.0, 20.0],
            bbox_px=[0, 0, 100, 20],
            column_guess=1,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
        ),
        BlockCandidate(
            block_id="mixed_heading",
            text=(
                "Results Identification of three distinct subtypes of tendinopathy based on "
                "transcriptome profiles and clinical features To better understand heterogeneity, "
                "we collected clinical data from diseased tendons."
            ),
            bbox_pt=[0.0, 40.0, 100.0, 180.0],
            bbox_px=[0, 40, 100, 180],
            column_guess=1,
            is_heading_candidate=True,
            is_header_footer_candidate=False,
        ),
    ]

    candidates = collect_region_heading_review_candidates(blocks, output, max_blocks=2, max_candidates=4)

    assert [block.block_id for block, _ in candidates] == ["mixed_heading"]
    assert candidates[0][1][0] == "Results"
