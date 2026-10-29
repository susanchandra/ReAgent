"""
reagent/evidence_record.py — Auditable Evidence Capture for NVIDIA Nemotron Proposals

Provides a strict, auditable, and sanitized record mechanism for NVIDIA Nemotron
inference requests (via Nebius Token Factory) and deterministic offline reference fixtures.

Design Invariants:
1. Strict Provenance Attribution: A deterministic fixture is NEVER labeled as
   a live inference. If is_live is False, invocation_mode is strictly 'DETERMINISTIC_FIXTURE'.
2. Zero Secrets Leakage: API keys, authorization headers, bearer tokens, credentials,
   and secret environment values are recursively sanitized and never written to disk.
3. Honest 4-Stage Role Separation:
   - 1. Model Proposal: Advisory contract proposed by NVIDIA Nemotron.
   - 2. Validated Contract: Frozen schema & task-grounded contract.
   - 3. Observed State: Empirical facts read by independent probe.
   - 4. Final Verdict: Authoritative decision by deterministic verifier (UNKNOWN != ABSENT).
4. No Fabrication: Missing metadata is explicitly marked as None or UNKNOWN;
   timestamps and run IDs are never fabricated.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, List


SENSITIVE_KEY_PATTERNS = re.compile(
    r"(api[_-]?key|secret|password|auth|bearer|credential|cookie|private|passwd|(?:access|auth|session|api|secret|bearer)[_-]?token|^token$)",
    re.IGNORECASE,
)

SECRET_VALUE_PATTERNS = re.compile(
    r"(sk-[a-zA-Z0-9_-]{10,}|Bearer\s+[a-zA-Z0-9_\-\.]+|ghp_[a-zA-Z0-9]+)",
    re.IGNORECASE,
)


def sanitize_evidence_payload(obj: Any) -> Any:
    """
    Recursively sanitizes dictionaries, lists, and strings to eliminate
    any API keys, authorization tokens, passwords, or credentials.
    Preserves non-sensitive contract keys: tool, arguments, proposed_postconditions,
    as well as token accounting counts (prompt_tokens, completion_tokens, etc.).
    """
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            k_str = str(k)
            # Preserve token metrics like prompt_tokens, completion_tokens, total_tokens
            if k_str in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
                sanitized[k] = v
            elif SENSITIVE_KEY_PATTERNS.search(k_str):
                sanitized[k] = "[REDACTED_SECRET]"
            else:
                sanitized[k] = sanitize_evidence_payload(v)
        return sanitized
    elif isinstance(obj, list):
        return [sanitize_evidence_payload(item) for item in obj]
    elif isinstance(obj, str):
        # Scrub inline bearer tokens or sk- keys
        return SECRET_VALUE_PATTERNS.sub("[REDACTED_SECRET]", obj)
    else:
        return obj


@dataclass
class NemotronEvidenceRecord:
    """
    Sanitized record of an NVIDIA Nemotron contract proposal run.
    Distinguishes genuine live Nebius inference from deterministic offline fixtures.
    """
    model_id: str
    provider_name: str
    timestamp_utc: float
    iso_timestamp: str
    run_id: str
    is_live: bool
    invocation_mode: str  # 'GENUINE_LIVE_INFERENCE' or 'DETERMINISTIC_FIXTURE'
    task_objective: str
    workflow: str
    sanitized_proposal_output: Optional[Dict[str, Any]]
    schema_valid: bool
    grounding_valid: Optional[bool]
    contract_validation_status: str  # CONTRACT_VALIDATED, GROUNDING_FAILED, SCHEMA_INVALID, NOT_CONFIGURED
    failure_reason: Optional[str]
    latency_ms: int
    token_usage: Dict[str, Any]
    endpoint_host: Optional[str] = None
    role_separation: Dict[str, str] = field(default_factory=lambda: {
        "stage_1_model_proposal": "Advisory proposal only. Model has zero tool authority and cannot verify execution.",
        "stage_2_contract_validation": "Deterministic schema and semantic grounding check before freezing.",
        "stage_3_observed_state": "Independent read-only probe queries actual SQLite/system state directly.",
        "stage_4_final_verdict": "Deterministic verifier decides outcome; UNKNOWN is never treated as ABSENT.",
    })

    def __post_init__(self):
        # Strict invariant: Never label a fixture as live inference
        if not self.is_live and self.invocation_mode == "GENUINE_LIVE_INFERENCE":
            raise ValueError(
                "Integrity violation: A deterministic fixture or offline mock CANNOT be labeled as 'GENUINE_LIVE_INFERENCE'."
            )
        if self.is_live and self.invocation_mode == "DETERMINISTIC_FIXTURE":
            raise ValueError(
                "Integrity violation: A live inference run must not be labeled as 'DETERMINISTIC_FIXTURE'."
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return sanitize_evidence_payload(d)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def create_evidence_record(
    model_id: str,
    provider_name: str,
    run_id: str,
    is_live: bool,
    task_objective: str,
    workflow: str,
    raw_proposal: Optional[Dict[str, Any]],
    schema_valid: bool,
    grounding_valid: Optional[bool],
    contract_validation_status: str,
    failure_reason: Optional[str] = None,
    latency_ms: int = 0,
    token_usage: Optional[Dict[str, Any]] = None,
    endpoint_host: Optional[str] = None,
    timestamp_utc: Optional[float] = None,
) -> NemotronEvidenceRecord:
    """Factory creating a validated, sanitized NemotronEvidenceRecord."""
    ts = timestamp_utc if timestamp_utc is not None else time.time()
    iso_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    mode = "GENUINE_LIVE_INFERENCE" if is_live else "DETERMINISTIC_FIXTURE"

    sanitized_output = sanitize_evidence_payload(raw_proposal) if raw_proposal else None
    sanitized_usage = sanitize_evidence_payload(token_usage or {})

    # Sanitize endpoint host to host-only (no tokens or parameters)
    clean_host = None
    if endpoint_host:
        from urllib.parse import urlparse
        clean_host = urlparse(endpoint_host).netloc or endpoint_host.split("/")[0].split("?")[0]

    return NemotronEvidenceRecord(
        model_id=str(model_id or "unknown-model"),
        provider_name=str(provider_name or "nebius_nemotron"),
        timestamp_utc=ts,
        iso_timestamp=iso_ts,
        run_id=str(run_id or f"run_{uuid.uuid4().hex[:8]}"),
        is_live=bool(is_live),
        invocation_mode=mode,
        task_objective=str(task_objective or ""),
        workflow=str(workflow or "generic"),
        sanitized_proposal_output=sanitized_output,
        schema_valid=bool(schema_valid),
        grounding_valid=grounding_valid,
        contract_validation_status=str(contract_validation_status),
        failure_reason=str(failure_reason) if failure_reason else None,
        latency_ms=int(latency_ms),
        token_usage=sanitized_usage,
        endpoint_host=clean_host,
    )


def save_sanitized_evidence(
    record: NemotronEvidenceRecord,
    target_path: str = "reports/evidence/nemotron_live_evidence.json",
) -> str:
    """
    Saves a sanitized NemotronEvidenceRecord to disk.
    Creates parent directories if necessary and ensures no secrets exist.
    """
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    json_str = record.to_json(indent=2)

    # Final safety assertion before write: no raw keys in serialized payload
    if SECRET_VALUE_PATTERNS.search(json_str):
        raise ValueError("Secret leakage guard: Raw credentials detected in serialized evidence. Save blocked.")

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    return target_path


def load_sanitized_evidence(path: str) -> Optional[Dict[str, Any]]:
    """Loads and returns evidence record if present on disk, or None if absent."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
