"""
ReAgent Provider Telemetry  (Phase 4 — observability)

Captures a sanitized per-attempt telemetry record for every live provider call.
Never stores API keys, raw responses, or any secrets.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# How token counts were obtained
TOKEN_SOURCE_PROVIDER = "PROVIDER_REPORTED"   # SDK usage object present
TOKEN_SOURCE_ESTIMATED = "ESTIMATED"           # Counted locally; SDK gave nothing
TOKEN_SOURCE_UNKNOWN = "UNKNOWN"               # Neither available


@dataclass
class ProviderTelemetryEvent:
    """
    Sanitized record of a single live provider attempt.

    Design invariants:
    - No secret values, API keys, or raw model responses are stored.
    - ``canonical_output_hash`` is a SHA-256 digest of the serialized decision
      (or empty string if the call failed or output was None).
    - ``is_live`` distinguishes a real network call (True) from an injected
      test-client call (False).
    - All fields have safe defaults so partial construction never raises.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    utc_timestamp: float = field(default_factory=time.time)
    run_id: str = ""
    provider_name: str = "nebius_nemotron"
    model_id: str = ""
    endpoint_host: str = ""          # Only the hostname, no auth params
    http_status_category: str = ""   # e.g. "2xx", "5xx", "timeout", "error"
    latency_ms: int = 0
    input_tokens_reported: Optional[int] = None
    output_tokens_reported: Optional[int] = None
    token_source: str = TOKEN_SOURCE_UNKNOWN
    schema_valid: bool = False
    grounding_valid: Optional[bool] = None    # None = not checked (e.g. key absent)
    canonical_output_hash: str = ""           # SHA-256 of serialized StructuredDecision
    is_live: bool = False
    failure_reason: Optional[str] = None      # None = success

    def to_dict(self) -> dict:
        """Return a plain dict safe for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def compute_canonical_hash(decision_dict: Optional[dict]) -> str:
    """
    Compute a deterministic SHA-256 hex digest of a decision dict.
    Returns empty string if ``decision_dict`` is None.
    """
    if decision_dict is None:
        return ""
    canonical = json.dumps(decision_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
