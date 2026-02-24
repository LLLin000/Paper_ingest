"""Shared parse/repair/fail guard for model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class GuardDecision(Generic[T]):
    parsed: T | None
    parse_success: bool
    validation_success: bool
    should_fallback: bool
    failure_reason: str


def safe_json_value(raw: str) -> Any | None:
    """Parse JSON with light repair for code-fence/prose wrapped outputs."""
    text = raw.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [idx for idx in (start_obj, start_arr) if idx >= 0]
    if starts:
        text = text[min(starts) :]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def guard_model_output(
    raw: str,
    parser: Callable[[str], T | None],
    validator: Callable[[T], tuple[bool, str]] | None = None,
) -> GuardDecision[T]:
    """Centralized parse/repair/fail policy for model outputs.

    - parse fails -> fallback with parse_failure
    - parse succeeds and validator fails -> fallback with validation reason
    - otherwise accepted
    """
    parsed = parser(raw)
    if parsed is None:
        return GuardDecision(
            parsed=None,
            parse_success=False,
            validation_success=False,
            should_fallback=True,
            failure_reason="parse_failure",
        )

    if validator is not None:
        valid, reason = validator(parsed)
        if not valid:
            return GuardDecision(
                parsed=parsed,
                parse_success=True,
                validation_success=False,
                should_fallback=True,
                failure_reason=reason,
            )

    return GuardDecision(
        parsed=parsed,
        parse_success=True,
        validation_success=True,
        should_fallback=False,
        failure_reason="ok",
    )
