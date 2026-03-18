from ingest.vision import classify_microblock_cluster_coherence


def test_classify_microblock_cluster_marks_axis_legend_group_non_narrative() -> None:
    cluster = {
        "region_id": "fig_1_cluster_0",
        "texts": ["0", "5", "10", "Hw", "Iw", "Ir", "Inflammation", "Log2 of FC", "Up", "Down"],
    }

    label = classify_microblock_cluster_coherence(cluster)

    assert label == "non_narrative_graphic"


def test_classify_microblock_cluster_keeps_sentence_group_narrative() -> None:
    cluster = {
        "region_id": "text_1_cluster_0",
        "texts": [
            "RNA sequence",
            "Total RNA was extracted from the tissue using TRIzol Reagent",
            "according to the manufacturer's instructions.",
        ],
    }

    label = classify_microblock_cluster_coherence(cluster)

    assert label == "narrative"
