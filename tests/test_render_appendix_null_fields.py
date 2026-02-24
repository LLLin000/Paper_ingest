# pyright: reportMissingTypeStubs=false

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingest.render import build_appendix


def test_build_appendix_handles_null_caption_and_statement_fields() -> None:
    appendix = build_appendix(
        profile={
            "paper_type": "original_research",
            "paper_type_confidence": 0.9,
            "reading_strategy": "default",
        },
        logic_graph={"nodes": [], "edges": []},
        themes={"themes": []},
        facts=[
            {
                "fact_id": "fact_1",
                "category": "background",
                "statement": None,
            }
        ],
        cite_map=[],
        reference_catalog=[],
        figure_index=[
            {
                "asset_id": "fig_001",
                "asset_type": "figure",
                "caption_text": None,
            }
        ],
    )

    assert any("[fig_001]" in line for line in appendix)
    assert any("[[#fact_1|fact_1]] [background]" in line for line in appendix)
