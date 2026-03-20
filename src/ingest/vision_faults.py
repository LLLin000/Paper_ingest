"""Shared dataclasses for vision-stage faults and telemetry."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FaultEvent:
    stage: str
    fault: str
    page: int
    retry_attempts: int
    fallback_used: bool
    status: str


@dataclass
class VisionLLMCallEvent:
    stage: str
    page: int
    attempt: int
    model: str
    endpoint: str
    success: bool
    parse_success: bool
    validation_success: bool
    error_type: str
    http_status: Optional[int]
    prompt_chars: int
    response_chars: int
    response_preview: str
    timestamp: str
    budget_limit: Optional[int] = None
    budget_consumed_before: Optional[int] = None
    budget_consumed_after: Optional[int] = None
    budget_remaining_after: Optional[int] = None
    budget_exhausted: bool = False
    cache_hit: Optional[bool] = None
    cache_miss: Optional[bool] = None
    cache_key: Optional[str] = None
