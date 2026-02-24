"""DAG-like orchestration helpers for full pipeline execution."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

StageJob = Callable[[], Any]

# Explicit full-run dependency graph for orchestration and documentation.
FULL_PIPELINE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "extractor": (),
    "overlay": ("extractor",),
    "vision": ("extractor",),
    "paragraphs": ("vision",),
    "citations": ("paragraphs",),
    "figures_tables": ("paragraphs", "vision"),
    "reading": ("citations", "figures_tables"),
    "render": ("reading",),
}

# Levelized DAG plan that allows safe parallel execution by level.
FULL_PIPELINE_LEVELS: tuple[tuple[str, ...], ...] = (
    ("extractor",),
    ("overlay", "vision"),
    ("paragraphs",),
    ("citations", "figures_tables"),
    ("reading",),
    ("render",),
)


def execute_levelized_dag(
    jobs: dict[str, StageJob],
    levels: tuple[tuple[str, ...], ...] = FULL_PIPELINE_LEVELS,
    max_workers: int = 2,
) -> dict[str, Any]:
    """Execute levelized DAG jobs with deterministic result ordering."""
    results: dict[str, Any] = {}

    for level in levels:
        if len(level) == 1:
            stage_name = level[0]
            results[stage_name] = jobs[stage_name]()
            continue

        with ThreadPoolExecutor(max_workers=min(max_workers, len(level))) as executor:
            futures: dict[str, Future[Any]] = {
                stage_name: executor.submit(jobs[stage_name])
                for stage_name in level
            }
            for stage_name in level:
                results[stage_name] = futures[stage_name].result()

    return results
