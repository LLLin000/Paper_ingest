"""Parser backend abstraction seam for structure extraction stages.

This provides a stable place to swap the front-half parser implementation in the
future without changing downstream evidence and rendering stages.
"""

from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_PARSER_BACKEND = "builtin"
_KNOWN_ALIASES = {"", "default", "self", "builtin", "internal"}


class ParserBackend(Protocol):
    """Minimal parser backend contract for run metadata and stage wiring."""

    name: str

    def metadata(self) -> dict[str, Any]:
        """Return serializable backend metadata for artifacts and manifests."""


@dataclass(frozen=True)
class BuiltinParserBackend:
    """Current in-repo parser chain backed by extractor + vision + paragraphs."""

    name: str = DEFAULT_PARSER_BACKEND

    def metadata(self) -> dict[str, Any]:
        return {
            "parser_backend": self.name,
            "parser_backend_kind": "in_repo",
            "structure_pipeline": "extractor->vision->paragraphs",
        }


def normalize_parser_backend_name(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _KNOWN_ALIASES:
        return DEFAULT_PARSER_BACKEND
    return DEFAULT_PARSER_BACKEND


def resolve_parser_backend(value: str | None) -> ParserBackend:
    """Resolve a configured backend name to the current implementation.

    Unknown names intentionally fall back to the builtin backend for now while
    preserving a stable seam for future adapters such as MinerU.
    """

    _ = value
    return BuiltinParserBackend(name=normalize_parser_backend_name(value))
