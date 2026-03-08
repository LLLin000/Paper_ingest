from ingest.figures_tables import FigureTableAsset, suppress_table_artifacts_in_clean_document


def test_suppress_table_artifacts_replaces_leaked_table_region_with_placeholder() -> None:
    clean_document = """# Demo\n\n## Main Body\n\n### 4.1. Carbon-Based Nanomaterials\n\nNarrative paragraph before table noise.\n\nNanodiamonds (NDs)\n\nCarbon Quantum Dots (CQD)\n\nGraphene\n\nHigh cost; Limited solubility; Easy to degrade [297]\n\nTransmission performance is affected by the environment; High cost; Low dispersion\n\nNarrative paragraph after table noise.\n\n## Figures and Tables\n\n### Table Captions\n\n- Table 1. Summary of representative conductive biomaterials.\n"""
    assets = [
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=10,
            bbox_px=[10, 10, 100, 100],
            caption_text="Table 1 | Summary of representative conductive biomaterials",
            caption_id="para_tbl_1",
            source_para_id="para_tbl_1",
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.9,
        )
    ]

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(clean_document, assets)

    assert replaced_regions >= 1
    assert "[TABLE_PLACEHOLDER]" in rewritten
    assert "Nanodiamonds (NDs)" not in rewritten
    assert "Narrative paragraph before table noise." in rewritten
    assert "Narrative paragraph after table noise." in rewritten


def test_suppress_table_artifacts_skips_when_no_table_assets() -> None:
    clean_document = """# Demo\n\n## Main Body\n\nNanodiamonds (NDs)\n\nGraphene\n"""

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(clean_document, [])

    assert replaced_regions == 0
    assert rewritten == clean_document


def test_suppress_table_artifacts_handles_title_case_labels_with_lowercase_words() -> None:
    clean_document = """# Demo

## Main Body

### 4.1. Carbon-Based Nanomaterials

Narrative paragraph before table noise.

Nanodiamonds (NDs)

Carbon Quantum Dots (CQD)

Graphene

Conductive polymers

High cost; Limited solubility; Easy to degrade [297]

Transmission performance is affected by the environment; High cost; Low dispersion

Narrative paragraph after table noise.

## Figures and Tables
"""
    assets = [
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=10,
            bbox_px=[10, 10, 100, 100],
            caption_text="Table 1 | Summary of representative conductive biomaterials",
            caption_id="para_tbl_1",
            source_para_id="para_tbl_1",
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.9,
        )
    ]

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(clean_document, assets)

    assert replaced_regions >= 1
    assert "[TABLE_PLACEHOLDER]" in rewritten
    assert "Nanodiamonds (NDs)" not in rewritten
    assert "Conductive polymers" not in rewritten


def test_suppress_table_artifacts_handles_numeric_property_rows() -> None:
    clean_document = """# Demo

## Main Body

Narrative paragraph before table noise.

High cost; Complex processing; Fast enzymatic hydrolysis speed

Silk ﬁbroin d 14 = 1.5–5 pC N − 1 Good biodegradability and biodegradability Outstanding mechanical property

Poor mechanical strength; Low molding efficiency

Narrative paragraph after table noise.

## Figures and Tables
"""
    assets = [
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=10,
            bbox_px=[10, 10, 100, 100],
            caption_text="Table 2 | Piezoelectric biomaterials",
            caption_id="para_tbl_2",
            source_para_id="para_tbl_2",
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.9,
        )
    ]

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(clean_document, assets)

    assert replaced_regions >= 1
    assert "[TABLE_PLACEHOLDER]" in rewritten
    assert "Silk ﬁbroin" not in rewritten
    assert "Narrative paragraph before table noise." in rewritten
    assert "Narrative paragraph after table noise." in rewritten

def test_table_placeholder_skips_continued_caption_noise_in_summary() -> None:
    clean_document = """# Demo

## Main Body

Nanodiamonds (NDs)

Graphene

High cost; Limited solubility; Easy to degrade [297]

## Figures and Tables
"""
    assets = [
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=10,
            bbox_px=[10, 10, 100, 100],
            caption_text="Table 1 | | (Continued).",
            caption_id="para_tbl_1",
            source_para_id="para_tbl_1",
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.9,
        )
    ]

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(clean_document, assets)

    assert replaced_regions >= 1
    assert "[TABLE_PLACEHOLDER]" in rewritten
    assert "(Continued)" not in rewritten
    assert "See processed table assets in ## Figures and Tables." in rewritten

def test_suppress_table_artifacts_bridges_compact_numeric_row_between_detected_runs() -> None:
    clean_document = """# Demo

## Main Body

Narrative paragraph before table noise.

High cost; Limited solubility; Easy to degrade [297]

AuNPs 2.86 × 10 2 High specific surface area High mechanical properties Unique optoelectronic properties

Poor mechanical strength; Low molding efficiency

Narrative paragraph after table noise.

## Figures and Tables
"""
    assets = [
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=10,
            bbox_px=[10, 10, 100, 100],
            caption_text="Table 1 | Summary of representative conductive biomaterials",
            caption_id="para_tbl_1",
            source_para_id="para_tbl_1",
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.9,
        )
    ]

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(clean_document, assets)

    assert replaced_regions >= 1
    assert "[TABLE_PLACEHOLDER]" in rewritten
    assert "AuNPs 2.86" not in rewritten
    assert "Narrative paragraph before table noise." in rewritten
    assert "Narrative paragraph after table noise." in rewritten

def test_suppress_table_artifacts_trims_mixed_table_prefix_and_keeps_narrative_suffix() -> None:
    clean_document = """# Demo

## Main Body

Narrative paragraph before table noise.

High cost; Limited solubility; Easy to degrade [297]

PHB d33 = 3.25 pC N−1 Good biodegradability; Outstanding biocompatibility Notably, BaTiO3 has favorable biocompatibility and supports bone regeneration.

Poor mechanical strength; Low molding efficiency

Narrative paragraph after table noise.

## Figures and Tables
"""
    assets = [
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=10,
            bbox_px=[10, 10, 100, 100],
            caption_text="Table 2 | Piezoelectric biomaterials",
            caption_id="para_tbl_2",
            source_para_id="para_tbl_2",
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.9,
        )
    ]

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(clean_document, assets)

    assert replaced_regions >= 1
    assert "[TABLE_PLACEHOLDER]" in rewritten
    assert "PHB d33 = 3.25" not in rewritten
    assert "Notably, BaTiO3 has favorable biocompatibility and supports bone regeneration." in rewritten
    assert "Narrative paragraph before table noise." in rewritten
    assert "Narrative paragraph after table noise." in rewritten

def test_suppress_table_artifacts_limits_placeholder_regions_to_table_asset_count() -> None:
    clean_document = """# Demo

## Main Body

Narrative start.

High cost; Limited solubility; Easy to degrade [297]

Graphene

Narrative bridge.

Silk fibroin d14 = 1.5-5 pC N-1 Good biodegradability; Outstanding mechanical property

Poor mechanical strength; Low molding efficiency

Narrative bridge 2.

PVDF d33 = 28 pC N-1 High mechanical strength; Excellent impact resistance

Low dipole moment Osteogenic differentiation; ECM formation

Narrative end.

## Figures and Tables
"""
    assets = [
        FigureTableAsset(
            asset_id="tbl_001",
            asset_type="table",
            page=10,
            bbox_px=[10, 10, 100, 100],
            caption_text="Table 1 | Summary of representative conductive biomaterials",
            caption_id="para_tbl_1",
            source_para_id="para_tbl_1",
            image_path="figures_tables/assets/tbl_001.png",
            confidence=0.9,
        )
    ]

    rewritten, replaced_regions = suppress_table_artifacts_in_clean_document(clean_document, assets)

    assert replaced_regions <= len(assets)
    assert rewritten.count("[TABLE_PLACEHOLDER]") <= len(assets)
    assert "Narrative start." in rewritten
    assert "Narrative end." in rewritten
