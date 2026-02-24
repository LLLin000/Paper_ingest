"""Provider abstraction layer for external reference collectors.

This module exposes a lightweight provider registry and helper functions
so higher-level code can enumerate and invoke concrete providers without
depending on implementation details in reference_providers_impl.

The concrete provider implementations remain in reference_providers_impl
for ease of testing; this layer simply wires them into a stable list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from . import reference_providers_impl as impl


@dataclass
class Provider:
    name: str
    fetch_name: str


# Default provider registry (ordered): PubMed, Crossref, OpenAlex, arXiv (conditional)
PROVIDERS: List[Provider] = [
    Provider(name="pubmed", fetch_name="_fetch_pubmed"),
    Provider(name="crossref", fetch_name="_fetch_crossref"),
    Provider(name="openalex", fetch_name="_fetch_openalex"),
    Provider(name="arxiv", fetch_name="_fetch_arxiv"),
]


def get_provider_names() -> List[str]:
    return [p.name for p in PROVIDERS]


def collect_api_references(doc_identity: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect references using implementation-level chained orchestration."""
    return impl.collect_api_references(doc_identity)
