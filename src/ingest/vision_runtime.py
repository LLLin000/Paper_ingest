"""Runtime/cache/telemetry helpers for the vision stage."""

import hashlib
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .qa_telemetry import append_jsonl_event
from .vision_faults import VisionLLMCallEvent

VISION_LLM_LOG_LOCK = threading.Lock()


class VisionRequestBudget:
    def __init__(self, limit: Optional[int]) -> None:
        self.limit = limit if limit is not None and limit >= 0 else None
        self._consumed = 0
        self._lock = threading.Lock()

    def try_consume(self) -> tuple[bool, int, int, Optional[int]]:
        with self._lock:
            consumed_before = self._consumed
            if self.limit is None:
                self._consumed += 1
                return True, consumed_before, self._consumed, None
            if self._consumed >= self.limit:
                return False, consumed_before, self._consumed, max(0, self.limit - self._consumed)
            self._consumed += 1
            remaining = max(0, self.limit - self._consumed)
            return True, consumed_before, self._consumed, remaining

    def snapshot(self) -> tuple[Optional[int], int, Optional[int]]:
        with self._lock:
            if self.limit is None:
                return None, self._consumed, None
            return self.limit, self._consumed, max(0, self.limit - self._consumed)


class VisionImageDataUrlCache:
    def __init__(self, encode_fn: Optional[Callable[[Path, int], str]] = None) -> None:
        self._cache: dict[str, str] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
        self._encode_fn = encode_fn

    def _cache_key(self, image_path: Path, max_side: int) -> str:
        image_bytes = image_path.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        return f"{image_hash}:{max_side}"

    def get_or_encode(self, image_path: Path, max_side: int) -> tuple[str, bool, str]:
        key = self._cache_key(image_path, max_side)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                return cached, True, key

        if self._encode_fn is None:
            raise RuntimeError("VisionImageDataUrlCache requires an encode function")
        encoded = self._encode_fn(image_path, max_side)
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._hits += 1
                return existing, True, key
            self._cache[key] = encoded
            self._misses += 1
            return encoded, False, key

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return self._hits, self._misses, len(self._cache)


@dataclass
class VisionRuntimeContext:
    budget: VisionRequestBudget
    image_cache: VisionImageDataUrlCache


def append_vision_llm_call_event(qa_dir: Path, event: VisionLLMCallEvent) -> None:
    payload = asdict(event)
    with VISION_LLM_LOG_LOCK:
        append_jsonl_event(qa_dir, "vision_llm_calls.jsonl", payload, "vision_llm_calls")


def resolve_vision_request_budget() -> Optional[int]:
    raw_limit = os.environ.get("SILICONFLOW_VISION_REQUEST_BUDGET", "0").strip()
    try:
        parsed = int(raw_limit)
    except ValueError:
        parsed = 0
    if parsed <= 0:
        return None
    return parsed


def append_vision_runtime_event(qa_dir: Path, runtime: VisionRuntimeContext, pages_done: int, total_pages: int) -> None:
    budget_limit, budget_consumed, budget_remaining = runtime.budget.snapshot()
    cache_hits, cache_misses, cache_entries = runtime.image_cache.snapshot()
    payload = {
        "stage": "vision",
        "pages_done": pages_done,
        "total_pages": total_pages,
        "budget_limit": budget_limit,
        "budget_consumed": budget_consumed,
        "budget_remaining": budget_remaining,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_entries": cache_entries,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl_event(qa_dir, "vision_runtime.jsonl", payload, "vision_runtime")
