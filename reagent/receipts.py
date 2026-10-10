"""
ReAgent Safe Audit Receipts (Milestone 4D revision)

Provides structured, allowlisted receipt persistence for execution results,
failure diagnoses, and probe runs.
Ensures zero secret leakage: raw exception messages, authorization headers,
bearer tokens, request/response bodies, argument values, file contents,
and model reasoning are strictly excluded.
"""

import os
import json
import uuid
import logging
from typing import Dict, Any, Optional, List, Union
from reagent.models import get_utc_now

logger = logging.getLogger(__name__)

# Strict allowlist of top-level receipt fields
ALLOWED_RECEIPT_KEYS = {
    "receipt_id",
    "timestamp",
    "task_id",
    "probe_type",
    "model_id",
    "provenance",
    "tool_name",
    "argument_keys",
    "finish_reason",
    "proposed_postconditions",
    "per_predicate_results",
    "status",
    "verified",
    "recovery_count",
    "token_usage",
    "error_type",
    "status_code",
    "exit_reason",
    "discrepancies",
}

# Strict allowlist of token_usage sub-keys
ALLOWED_TOKEN_USAGE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
}


def create_safe_receipt_payload(
    task_id: str,
    status: str,
    tool_name: Optional[str] = None,
    probe_type: str = "pipeline",
    model_id: Optional[str] = None,
    provenance: Optional[str] = None,
    argument_keys: Optional[Union[List[str], Dict[str, Any]]] = None,
    finish_reason: Optional[str] = None,
    proposed_postconditions: Optional[List[Any]] = None,
    per_predicate_results: Optional[List[Any]] = None,
    verified: bool = False,
    recovery_count: int = 0,
    token_usage: Optional[Dict[str, int]] = None,
    error_type: Optional[str] = None,
    status_code: Optional[int] = None,
    exit_reason: Optional[str] = None,
    discrepancies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Construct an allowlisted dictionary representation of a receipt.
    Guarantees that raw exception messages, authorization headers,
    credentials, argument values, and model reasoning cannot leak.
    """
    safe_usage: Dict[str, int] = {}
    if token_usage and isinstance(token_usage, dict):
        for k in ALLOWED_TOKEN_USAGE_KEYS:
            if k in token_usage and isinstance(token_usage[k], int):
                safe_usage[k] = token_usage[k]

    # Argument keys ONLY — never argument values!
    safe_arg_keys: List[str] = []
    if argument_keys is not None:
        if isinstance(argument_keys, dict):
            safe_arg_keys = sorted(str(k)[:64] for k in argument_keys.keys())
        elif isinstance(argument_keys, (list, tuple, set)):
            safe_arg_keys = sorted(str(k)[:64] for k in argument_keys)

    # Postconditions: predicate types and targets ONLY with strict validation and length bounding
    safe_postconditions: List[Dict[str, Optional[str]]] = []
    if proposed_postconditions:
        for p in proposed_postconditions:
            raw_pred = getattr(p, "predicate", None) or (p.get("predicate") if isinstance(p, dict) else str(p))
            raw_target = getattr(p, "target", None) or (p.get("target") if isinstance(p, dict) else None)

            clean_pred = None
            if raw_pred is not None:
                p_str = str(raw_pred).strip()
                if p_str and all(c.isalnum() or c == "_" for c in p_str):
                    clean_pred = p_str[:64]
                else:
                    clean_pred = "INVALID_PREDICATE_TYPE"

            clean_target = None
            if raw_target is not None:
                if isinstance(raw_target, (str, int, float, bool)):
                    clean_target = str(raw_target).replace("\n", " ").replace("\r", "")[:200].strip()
                else:
                    clean_target = "INVALID_TARGET_TYPE"

            safe_postconditions.append({
                "predicate": clean_pred,
                "target": clean_target,
            })

    # Per-predicate pass/fail status with strict validation and length bounding
    safe_predicate_results: List[Dict[str, Any]] = []
    if per_predicate_results:
        for r in per_predicate_results:
            raw_pred = getattr(r, "assertion_type", None) or (r.get("predicate") if isinstance(r, dict) else None)
            raw_target = getattr(r, "key_path", None) or (r.get("target") if isinstance(r, dict) else None)
            passed = getattr(r, "passed", None)
            if passed is None and isinstance(r, dict):
                passed = r.get("passed")

            clean_pred = None
            if raw_pred is not None:
                p_str = str(raw_pred).strip()
                if p_str and all(c.isalnum() or c == "_" for c in p_str):
                    clean_pred = p_str[:64]
                else:
                    clean_pred = "INVALID_PREDICATE_TYPE"

            clean_target = None
            if raw_target is not None:
                if isinstance(raw_target, (str, int, float, bool)):
                    clean_target = str(raw_target).replace("\n", " ").replace("\r", "")[:200].strip()
                else:
                    clean_target = "INVALID_TARGET_TYPE"

            safe_predicate_results.append({
                "predicate": clean_pred,
                "target": clean_target,
                "passed": bool(passed) if passed is not None else False,
            })

    # Discrepancies: bounded to 200 chars, no raw file or model text
    safe_discrepancies: List[str] = []
    if discrepancies and isinstance(discrepancies, list):
        for d in discrepancies:
            if isinstance(d, str):
                cleaned = d.replace("\n", " ").replace("\r", "")[:200].strip()
                safe_discrepancies.append(cleaned)

    clean_error_type = (
        str(error_type).replace("\n", " ").replace("\r", "")[:100].strip()
        if error_type else None
    )
    clean_exit_reason = (
        str(exit_reason).replace("\n", " ").replace("\r", "")[:200].strip()
        if exit_reason else None
    )
    clean_tool_name = (
        str(tool_name)[:64].strip()
        if tool_name else None
    )

    payload = {
        "receipt_id": str(uuid.uuid4()),
        "timestamp": get_utc_now(),
        "task_id": str(task_id)[:64],
        "probe_type": str(probe_type)[:64],
        "model_id": str(model_id)[:100] if model_id else None,
        "provenance": str(provenance)[:32] if provenance else None,
        "tool_name": clean_tool_name,
        "argument_keys": safe_arg_keys,
        "finish_reason": str(finish_reason)[:32] if finish_reason else None,
        "proposed_postconditions": safe_postconditions,
        "per_predicate_results": safe_predicate_results,
        "status": str(status)[:64],
        "verified": bool(verified),
        "recovery_count": int(recovery_count),
        "token_usage": safe_usage,
        "error_type": clean_error_type,
        "status_code": int(status_code) if isinstance(status_code, int) else None,
        "exit_reason": clean_exit_reason,
        "discrepancies": safe_discrepancies,
    }

    # Strict allowlist filtering
    return {k: v for k, v in payload.items() if k in ALLOWED_RECEIPT_KEYS}


def save_allowlisted_receipt(
    receipt_dir: str,
    task_id: str,
    status: str,
    tool_name: Optional[str] = None,
    probe_type: str = "pipeline",
    model_id: Optional[str] = None,
    provenance: Optional[str] = None,
    argument_keys: Optional[Union[List[str], Dict[str, Any]]] = None,
    finish_reason: Optional[str] = None,
    proposed_postconditions: Optional[List[Any]] = None,
    per_predicate_results: Optional[List[Any]] = None,
    verified: bool = False,
    recovery_count: int = 0,
    token_usage: Optional[Dict[str, int]] = None,
    error_type: Optional[str] = None,
    status_code: Optional[int] = None,
    exit_reason: Optional[str] = None,
    discrepancies: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Atomically writes an allowlisted receipt JSON file to receipt_dir.
    Returns the absolute path to the saved receipt, or None if write failed.
    A failure to save a receipt does NOT alter the execution status or claim false success.
    """
    payload = create_safe_receipt_payload(
        task_id=task_id,
        status=status,
        tool_name=tool_name,
        probe_type=probe_type,
        model_id=model_id,
        provenance=provenance,
        argument_keys=argument_keys,
        finish_reason=finish_reason,
        proposed_postconditions=proposed_postconditions,
        per_predicate_results=per_predicate_results,
        verified=verified,
        recovery_count=recovery_count,
        token_usage=token_usage,
        error_type=error_type,
        status_code=status_code,
        exit_reason=exit_reason,
        discrepancies=discrepancies,
    )

    try:
        os.makedirs(receipt_dir, exist_ok=True)
        filename = f"receipt_{payload['receipt_id']}.json"
        dest_path = os.path.join(receipt_dir, filename)
        tmp_path = f"{dest_path}.tmp.{uuid.uuid4().hex}"

        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, dest_path)
        return dest_path
    except Exception as exc:
        logger.warning(
            "save_allowlisted_receipt: failed to write receipt: %s",
            type(exc).__name__
        )
        return None
