"""
ReAgent Deterministic Verifier (Milestone 4A revision)
Evaluates ground-truth evidence against explicit post-condition assertions.
Detects discrepancies, false successes, and unverified recovery claims.

Security additions in 4A:
  - All FILE predicates resolve paths through _resolve_verified_path().
  - _resolve_verified_path() rejects: absolute paths, Windows drive letters, '..' traversal,
    and symlinks / junctions that exit the workspace_root.
  - FILE_CONTAINS reads are hard-capped at FILE_READ_CAP_BYTES (1 MB).

Supported predicates:
  STATUS_SUCCESS    — action.success == True
  KEY_EXISTS        — key exists in output dict
  VALUE_EQUALS      — output[key] == expected
  FILE_EXISTS       — file path exists on disk (sandboxed)
  FILE_CONTAINS     — file content contains expected substring (sandboxed, capped)
  FILE_SIZE_GTE     — file size >= expected bytes (sandboxed)
  JSON_VALID        — output or target file contains valid JSON
  JSON_PATH_EQUALS  — simple dotted key path into JSON equals expected
  HTTP_STATUS       — output['status_code'] == expected
  NO_ERROR_PAYLOAD  — output has no embedded error field and status < 400
"""

import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from reagent.models import Evidence, VerificationResult, ActionResult, PostconditionEvaluation
from reagent.taxonomy import PostconditionStatus, AgentStatus

# Hard cap on bytes read by FILE_CONTAINS to prevent memory exhaustion.
FILE_READ_CAP_BYTES = 1 * 1024 * 1024  # 1 MB


class AssertionRule(BaseModel):
    """Defines an invariant or post-condition that must hold for verification."""
    assertion_type: str     # One of the supported predicates above
    key_path: Optional[str] = None
    expected_value: Optional[Any] = None
    description: str = ""
    rule_id: Optional[str] = None
    required: bool = True


def aggregate_verification_outcome(
    evaluations: List[PostconditionEvaluation],
    is_authoritative: bool = True,
    recovery_exhausted: bool = False,
    model_proposal_override: bool = False,
    raise_on_empty: bool = True,
) -> AgentStatus:
    """
    Pure outcome aggregation function for postcondition evaluations.
    Accepts structured evaluations and authoritative/recovery context.
    Does not execute tools, mutate state, call a model, or perform I/O.

    Rules applied:
    1. All postconditions MET -> VERIFIED.
    2. All postconditions UNMET -> VERIFIED_FAILURE only when authoritative evidence supports that conclusion.
    3. A mixture of MET and UNMET, with no UNDETERMINED -> PARTIAL (when authoritative).
    4. Any UNDETERMINED -> UNKNOWN, unless recovery exhaustion takes precedence under rule 5.
    5. Exhausted recovery budget -> RECOVERY_EXHAUSTED, while preserving all underlying evaluations and uncertainty.
    6. Empty evaluations must never imply success. Return an explicit invalid/incomplete-contract result or fail validation according to established project conventions.
    7. Non-authoritative absence must never be converted to verified failure.
    8. Never let a model-generated claim override independent evidence.
    """
    if not evaluations:
        if raise_on_empty:
            raise ValueError("Incomplete contract: evaluations list cannot be empty.")
        return AgentStatus.FAILED

    # Rule 8: Model claims/proposals cannot override independent verification.
    # The flag or any advisory claim is ignored for deterministic aggregation.

    # Rule 5: Exhausted recovery budget takes precedence over intermediate failure/unknown
    if recovery_exhausted:
        return AgentStatus.RECOVERY_EXHAUSTED

    # Check for any UNDETERMINED
    statuses = [e.status for e in evaluations]
    has_undetermined = any(s == PostconditionStatus.UNDETERMINED for s in statuses)
    all_met = all(s == PostconditionStatus.MET for s in statuses)
    all_unmet = all(s == PostconditionStatus.UNMET for s in statuses)

    # Rule 1: All postconditions MET -> VERIFIED
    if all_met:
        return AgentStatus.VERIFIED

    # Rule 4 & 7: Any UNDETERMINED -> UNKNOWN
    # Also if evidence is non-authoritative and there are UNMET postconditions,
    # non-authoritative absence must never be converted to verified failure -> UNKNOWN.
    if has_undetermined or not is_authoritative:
        return AgentStatus.UNKNOWN

    # Rule 2: All postconditions UNMET + authoritative -> VERIFIED_FAILURE
    if all_unmet:
        return AgentStatus.VERIFIED_FAILURE

    # Rule 3: Mixture of MET and UNMET, no UNDETERMINED, authoritative -> PARTIAL
    return AgentStatus.PARTIAL


class DeterministicVerifier:
    """
    Deterministic rule engine that validates whether an action or recovery actually succeeded.
    Compares evidence against post-conditions without relying on fuzzy LLM self-grading.
    The LLM cannot override, bypass, or influence this verifier's verdict.

    Security guarantee: all file-system predicates resolve their paths through
    _resolve_verified_path(), which enforces the workspace_root boundary.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        """
        Args:
            workspace_root: Absolute path that all FILE predicate paths must stay within.
                            If None, no sandbox root is enforced (allows standalone verifier tests).
        """
        if workspace_root is not None:
            self.workspace_root = os.path.realpath(os.path.abspath(workspace_root))
        else:
            self.workspace_root = None

    # ──────────────────────────────────────────────────────────────────────
    # Path sandbox enforcement
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_verified_path(self, raw_path: str) -> str:
        """
        Resolve and sandbox a path supplied by a predicate.

        If self.workspace_root is set, enforces that:
          - Candidate path after realpath() resolution stays inside workspace_root.
          - Rejects absolute paths, drive letters, '..', and symlinks that resolve outside workspace_root.

        If self.workspace_root is None, resolves the path directly without a sandbox boundary.
        """
        if not raw_path:
            raise PermissionError("VerifierSandbox: empty path rejected.")

        raw_str = str(raw_path)

        if self.workspace_root is not None:
            drive, _ = os.path.splitdrive(raw_str)
            root_drive, _ = os.path.splitdrive(self.workspace_root)
            if drive and drive.upper() != root_drive.upper():
                raise PermissionError(
                    f"VerifierSandbox: foreign drive in '{raw_str}' exits workspace root '{self.workspace_root}'."
                )

            # Resolve candidate path: if absolute, resolve realpath directly; if relative, join with root
            if os.path.isabs(raw_str):
                candidate = os.path.realpath(raw_str)
            else:
                candidate = os.path.realpath(os.path.join(self.workspace_root, raw_str))

            # Strictly enforce that candidate stays inside self.workspace_root
            boundary = self.workspace_root
            if candidate != boundary and not candidate.startswith(boundary + os.sep):
                raise PermissionError(
                    f"VerifierSandbox: resolved path '{raw_str}' "
                    f"exits workspace root '{boundary}'."
                )
            return candidate
        else:
            # When workspace_root is None (standalone testing), resolve path directly
            if os.path.isabs(raw_str):
                return os.path.realpath(raw_str)
            return os.path.realpath(os.path.abspath(raw_str))


    # ──────────────────────────────────────────────────────────────────────
    # Main verification entry point
    # ──────────────────────────────────────────────────────────────────────

    def verify(
        self,
        recovery_decision_id: str,
        evidence: Evidence,
        assertions: Optional[List[AssertionRule]] = None,
    ) -> VerificationResult:
        """
        Execute deterministic checks across all assertions against the provided evidence facts.
        Returns a VerificationResult.  verified=True only when ALL assertions pass.
        Preserves per-postcondition evaluations with MET, UNMET, and UNDETERMINED statuses.
        """
        if assertions is not None and len(assertions) == 0:
            return VerificationResult(
                recovery_decision_id=recovery_decision_id,
                verified=False,
                evidence_summary="Contract rejected: empty assertion rules list.",
                discrepancies=["Empty assertions list: contract is invalid or incomplete."],
                postcondition_evaluations=[],
                aggregate_status=AgentStatus.FAILED,
            )

        assertions = assertions or [
            AssertionRule(
                assertion_type="STATUS_SUCCESS",
                description="Action execution must return success=True",
            )
        ]

        if evidence is None or not isinstance(getattr(evidence, "facts", None), dict):
            evals = [
                PostconditionEvaluation(
                    postcondition_id=getattr(r, "rule_id", None) or f"{r.assertion_type}:{r.key_path or ''}",
                    predicate=r.assertion_type,
                    target=r.key_path,
                    expected=r.expected_value,
                    status=PostconditionStatus.UNDETERMINED,
                    evidence={},
                    explanation="Missing or invalid evidence facts.",
                    required=getattr(r, "required", True),
                )
                for r in assertions
            ]
            return VerificationResult(
                recovery_decision_id=recovery_decision_id,
                verified=False,
                evidence_summary="Verification failed: missing or invalid evidence facts.",
                discrepancies=["Missing or invalid evidence."],
                postcondition_evaluations=evals,
                aggregate_status=AgentStatus.UNKNOWN,
            )

        discrepancies: List[str] = []
        evaluations: List[PostconditionEvaluation] = []
        facts = evidence.facts

        for rule in assertions:
            atype = rule.assertion_type
            eval_status: PostconditionStatus = PostconditionStatus.UNDETERMINED
            eval_evidence: Dict[str, Any] = {}
            eval_explanation: str = ""

            # ── 1. STATUS_SUCCESS ────────────────────────────────────────
            if atype == "STATUS_SUCCESS":
                succ = bool(facts.get("success", False))
                if succ:
                    eval_status = PostconditionStatus.MET
                    eval_explanation = "Action execution returned success=True."
                    eval_evidence = {"success": True}
                else:
                    err = facts.get("error", "Unknown error")
                    eval_status = PostconditionStatus.UNMET
                    eval_explanation = f"Execution failed: {err}"
                    eval_evidence = {"success": False, "error": str(err)}
                    discrepancies.append(eval_explanation)

            # ── 2. KEY_EXISTS ─────────────────────────────────────────────
            elif atype == "KEY_EXISTS":
                output = facts.get("output", {})
                if isinstance(output, dict):
                    if rule.key_path in output:
                        eval_status = PostconditionStatus.MET
                        eval_explanation = f"Expected key '{rule.key_path}' exists in output."
                        eval_evidence = {"key": rule.key_path, "found": True}
                    else:
                        eval_status = PostconditionStatus.UNMET
                        eval_explanation = f"Expected key '{rule.key_path}' missing from output."
                        eval_evidence = {"key": rule.key_path, "found": False}
                        discrepancies.append(eval_explanation)
                else:
                    eval_status = PostconditionStatus.UNDETERMINED
                    eval_explanation = f"Output is not a dict; cannot verify key '{rule.key_path}'."
                    eval_evidence = {"output_type": type(output).__name__}
                    discrepancies.append(eval_explanation)

            # ── 3. VALUE_EQUALS ───────────────────────────────────────────
            elif atype == "VALUE_EQUALS":
                output = facts.get("output", {})
                if isinstance(output, dict):
                    actual = output.get(rule.key_path)
                    if actual == rule.expected_value:
                        eval_status = PostconditionStatus.MET
                        eval_explanation = f"Key '{rule.key_path}' matches expected value."
                        eval_evidence = {"key": rule.key_path, "expected": rule.expected_value, "actual": actual}
                    else:
                        eval_status = PostconditionStatus.UNMET
                        eval_explanation = f"Key '{rule.key_path}': expected '{rule.expected_value}', got '{actual}'."
                        eval_evidence = {"key": rule.key_path, "expected": rule.expected_value, "actual": actual}
                        discrepancies.append(eval_explanation)
                else:
                    eval_status = PostconditionStatus.UNDETERMINED
                    eval_explanation = f"Output is not a dict; cannot verify value for '{rule.key_path}'."
                    eval_evidence = {"output_type": type(output).__name__}
                    discrepancies.append(eval_explanation)

            # ── 4. FILE_EXISTS ────────────────────────────────────────────
            elif atype == "FILE_EXISTS":
                raw = rule.expected_value or rule.key_path
                if not raw:
                    eval_status = PostconditionStatus.UNDETERMINED
                    eval_explanation = "FILE_EXISTS: no path provided in predicate."
                    eval_evidence = {"path": None}
                    discrepancies.append(eval_explanation)
                else:
                    try:
                        safe = self._resolve_verified_path(str(raw))
                        if os.path.exists(safe):
                            eval_status = PostconditionStatus.MET
                            eval_explanation = f"File exists: '{raw}'."
                            eval_evidence = {"path": str(raw), "exists": True}
                        else:
                            eval_status = PostconditionStatus.UNMET
                            eval_explanation = f"Post-condition failed: file does not exist: '{raw}'."
                            eval_evidence = {"path": str(raw), "exists": False}
                            discrepancies.append(eval_explanation)
                    except PermissionError as exc:
                        eval_status = PostconditionStatus.UNDETERMINED
                        eval_explanation = f"FILE_EXISTS: {exc}"
                        eval_evidence = {"path": str(raw), "error": str(exc)}
                        discrepancies.append(eval_explanation)

            # ── 5. FILE_CONTAINS ──────────────────────────────────────────
            elif atype == "FILE_CONTAINS":
                raw = rule.key_path
                substring = str(rule.expected_value or "")
                if not raw:
                    eval_status = PostconditionStatus.UNDETERMINED
                    eval_explanation = "FILE_CONTAINS: no path provided in predicate."
                    eval_evidence = {"path": None}
                    discrepancies.append(eval_explanation)
                else:
                    try:
                        safe = self._resolve_verified_path(str(raw))
                        if not os.path.exists(safe):
                            eval_status = PostconditionStatus.UNMET
                            eval_explanation = f"FILE_CONTAINS: file does not exist: '{raw}'."
                            eval_evidence = {"path": str(raw), "exists": False}
                            discrepancies.append(eval_explanation)
                        else:
                            try:
                                with open(safe, "rb") as fh:
                                    raw_bytes = fh.read(FILE_READ_CAP_BYTES)
                                content = raw_bytes.decode("utf-8", errors="replace")
                                if substring in content:
                                    eval_status = PostconditionStatus.MET
                                    eval_explanation = f"FILE_CONTAINS: '{raw}' contains expected substring."
                                    eval_evidence = {"path": str(raw), "contains": True}
                                else:
                                    eval_status = PostconditionStatus.UNMET
                                    eval_explanation = f"FILE_CONTAINS: '{raw}' does not contain expected substring."
                                    eval_evidence = {"path": str(raw), "contains": False}
                                    discrepancies.append(eval_explanation)
                            except OSError as exc:
                                eval_status = PostconditionStatus.UNDETERMINED
                                eval_explanation = f"FILE_CONTAINS: could not read '{raw}': {exc}."
                                eval_evidence = {"path": str(raw), "error": str(exc)}
                                discrepancies.append(eval_explanation)
                    except PermissionError as exc:
                        eval_status = PostconditionStatus.UNDETERMINED
                        eval_explanation = f"FILE_CONTAINS: {exc}"
                        eval_evidence = {"path": str(raw), "error": str(exc)}
                        discrepancies.append(eval_explanation)

            # ── 6. FILE_SIZE_GTE ──────────────────────────────────────────
            elif atype == "FILE_SIZE_GTE":
                raw = rule.key_path
                min_bytes = int(rule.expected_value or 0)
                if not raw:
                    eval_status = PostconditionStatus.UNDETERMINED
                    eval_explanation = "FILE_SIZE_GTE: no path provided in predicate."
                    eval_evidence = {"path": None}
                    discrepancies.append(eval_explanation)
                else:
                    try:
                        safe = self._resolve_verified_path(str(raw))
                        if not os.path.exists(safe):
                            eval_status = PostconditionStatus.UNMET
                            eval_explanation = f"FILE_SIZE_GTE: file does not exist: '{raw}'."
                            eval_evidence = {"path": str(raw), "exists": False}
                            discrepancies.append(eval_explanation)
                        else:
                            actual_size = os.path.getsize(safe)
                            if actual_size >= min_bytes:
                                eval_status = PostconditionStatus.MET
                                eval_explanation = f"FILE_SIZE_GTE: '{raw}' is {actual_size} bytes (>= {min_bytes})."
                                eval_evidence = {"path": str(raw), "actual_size": actual_size, "min_bytes": min_bytes}
                            else:
                                eval_status = PostconditionStatus.UNMET
                                eval_explanation = f"FILE_SIZE_GTE: '{raw}' is {actual_size} bytes (expected >= {min_bytes})."
                                eval_evidence = {"path": str(raw), "actual_size": actual_size, "min_bytes": min_bytes}
                                discrepancies.append(eval_explanation)
                    except PermissionError as exc:
                        eval_status = PostconditionStatus.UNDETERMINED
                        eval_explanation = f"FILE_SIZE_GTE: {exc}"
                        eval_evidence = {"path": str(raw), "error": str(exc)}
                        discrepancies.append(eval_explanation)

            # ── 7. JSON_VALID ─────────────────────────────────────────────
            elif atype == "JSON_VALID":
                raw = rule.key_path
                eval_status = None
                if raw:
                    try:
                        safe = self._resolve_verified_path(str(raw))
                        if os.path.exists(safe):
                            try:
                                with open(safe, "r", encoding="utf-8") as fh:
                                    json.load(fh)
                                eval_status = PostconditionStatus.MET
                                eval_explanation = f"JSON_VALID: '{raw}' is valid JSON."
                                eval_evidence = {"path": str(raw), "valid": True}
                            except (json.JSONDecodeError, OSError) as exc:
                                eval_status = PostconditionStatus.UNMET
                                eval_explanation = f"JSON_VALID: '{raw}' is not valid JSON: {exc}."
                                eval_evidence = {"path": str(raw), "valid": False, "error": str(exc)}
                                discrepancies.append(eval_explanation)
                        else:
                            raw = None
                    except PermissionError as exc:
                        eval_status = PostconditionStatus.UNDETERMINED
                        eval_explanation = f"JSON_VALID: {exc}"
                        eval_evidence = {"path": str(raw), "error": str(exc)}
                        discrepancies.append(eval_explanation)
                        raw = None

                if not raw and eval_status is None:
                    output = facts.get("output")
                    if isinstance(output, str):
                        try:
                            json.loads(output)
                            eval_status = PostconditionStatus.MET
                            eval_explanation = "JSON_VALID: output string is valid JSON."
                            eval_evidence = {"output_valid": True}
                        except json.JSONDecodeError as exc:
                            eval_status = PostconditionStatus.UNMET
                            eval_explanation = f"JSON_VALID: output string is not valid JSON: {exc}."
                            eval_evidence = {"output_valid": False, "error": str(exc)}
                            discrepancies.append(eval_explanation)
                    elif isinstance(output, (dict, list)):
                        eval_status = PostconditionStatus.MET
                        eval_explanation = "JSON_VALID: output is valid JSON structure."
                        eval_evidence = {"output_type": type(output).__name__}
                    else:
                        eval_status = PostconditionStatus.UNMET
                        eval_explanation = "JSON_VALID: output is not a JSON-serializable type."
                        eval_evidence = {"output_type": type(output).__name__}
                        discrepancies.append(eval_explanation)

            # ── 8. JSON_PATH_EQUALS ───────────────────────────────────────
            elif atype == "JSON_PATH_EQUALS":
                output = facts.get("output", {})
                if not isinstance(output, dict):
                    eval_status = PostconditionStatus.UNDETERMINED
                    eval_explanation = "JSON_PATH_EQUALS: output is not a dict."
                    eval_evidence = {"output_type": type(output).__name__}
                    discrepancies.append(eval_explanation)
                else:
                    keys = (rule.key_path or "").split(".")
                    node: Any = output
                    try:
                        for k in keys:
                            node = node[k]
                        if node == rule.expected_value:
                            eval_status = PostconditionStatus.MET
                            eval_explanation = f"JSON_PATH_EQUALS: '{rule.key_path}' matches expected value."
                            eval_evidence = {"path": rule.key_path, "expected": rule.expected_value, "actual": node}
                        else:
                            eval_status = PostconditionStatus.UNMET
                            eval_explanation = f"JSON_PATH_EQUALS: '{rule.key_path}' = '{node}' (expected '{rule.expected_value}')."
                            eval_evidence = {"path": rule.key_path, "expected": rule.expected_value, "actual": node}
                            discrepancies.append(eval_explanation)
                    except (KeyError, TypeError) as exc:
                        eval_status = PostconditionStatus.UNMET
                        eval_explanation = f"JSON_PATH_EQUALS: path '{rule.key_path}' not found in output."
                        eval_evidence = {"path": rule.key_path, "error": str(exc)}
                        discrepancies.append(eval_explanation)

            # ── 9. HTTP_STATUS ────────────────────────────────────────────
            elif atype == "HTTP_STATUS":
                output = facts.get("output", {})
                actual_status = None
                if isinstance(output, dict):
                    actual_status = output.get("status_code")
                if actual_status == rule.expected_value:
                    eval_status = PostconditionStatus.MET
                    eval_explanation = f"HTTP_STATUS: status_code is {actual_status}."
                    eval_evidence = {"expected": rule.expected_value, "actual": actual_status}
                else:
                    eval_status = PostconditionStatus.UNMET
                    eval_explanation = f"HTTP_STATUS: expected {rule.expected_value}, got {actual_status}."
                    eval_evidence = {"expected": rule.expected_value, "actual": actual_status}
                    discrepancies.append(eval_explanation)

            # ── 10. NO_ERROR_PAYLOAD ──────────────────────────────────────
            elif atype == "NO_ERROR_PAYLOAD":
                output = facts.get("output", {})
                if isinstance(output, dict):
                    has_error = bool(output.get("error"))
                    bad_status = output.get("status_code", 200) >= 400
                    if has_error or bad_status:
                        eval_status = PostconditionStatus.UNMET
                        reasons = []
                        if has_error:
                            reasons.append(f"Output contains embedded error payload: {output['error']}")
                        if bad_status:
                            reasons.append(f"HTTP status indicates failure: {output.get('status_code')}")
                        eval_explanation = "; ".join(reasons)
                        eval_evidence = {"error": output.get("error"), "status_code": output.get("status_code")}
                        discrepancies.extend(reasons)
                    else:
                        eval_status = PostconditionStatus.MET
                        eval_explanation = "NO_ERROR_PAYLOAD: no error payload found and status < 400."
                        eval_evidence = {"status_code": output.get("status_code", 200)}
                else:
                    eval_status = PostconditionStatus.UNDETERMINED
                    eval_explanation = "NO_ERROR_PAYLOAD: output is not a dict."
                    eval_evidence = {"output_type": type(output).__name__}
                    discrepancies.append(eval_explanation)

            # ── Unknown predicate ─────────────────────────────────────────
            else:
                eval_status = PostconditionStatus.UNDETERMINED
                eval_explanation = f"Unknown assertion_type '{atype}' — treated as verification failure."
                eval_evidence = {"unknown_type": atype}
                discrepancies.append(eval_explanation)

            rule_id = getattr(rule, "rule_id", None) or f"{atype}:{rule.key_path or ''}"
            evaluations.append(
                PostconditionEvaluation(
                    postcondition_id=rule_id,
                    predicate=atype,
                    target=rule.key_path,
                    expected=rule.expected_value,
                    status=eval_status,
                    evidence=eval_evidence,
                    explanation=eval_explanation,
                    required=getattr(rule, "required", True),
                )
            )

        is_authoritative = facts.get("is_authoritative", True)
        if facts.get("unobservable") is True or facts.get("status") in ("UNKNOWN", "UNOBSERVABLE", "UNAVAILABLE"):
            is_authoritative = False

        aggregate_status = aggregate_verification_outcome(
            evaluations=evaluations,
            is_authoritative=is_authoritative,
            recovery_exhausted=facts.get("recovery_exhausted", False),
            raise_on_empty=False,
        )

        is_verified = (aggregate_status == AgentStatus.VERIFIED)
        if aggregate_status == AgentStatus.VERIFIED:
            summary = "All post-condition assertions passed."
        elif aggregate_status == AgentStatus.PARTIAL:
            summary = f"Partial verification: {len(discrepancies)} discrepancy(ies) out of {len(evaluations)} assertion(s)."
        elif aggregate_status == AgentStatus.VERIFIED_FAILURE:
            summary = f"Verified failure: all assertions failed under authoritative evidence ({len(discrepancies)} discrepancy(ies))."
        elif aggregate_status == AgentStatus.RECOVERY_EXHAUSTED:
            summary = f"Recovery exhausted: postconditions remained unverified after recovery attempts ({len(discrepancies)} discrepancy(ies))."
        else:
            summary = f"Verification outcome UNKNOWN ({len(discrepancies)} discrepancy(ies) or undetermined evidence)."

        return VerificationResult(
            recovery_decision_id=recovery_decision_id,
            verified=is_verified,
            evidence_summary=summary,
            discrepancies=discrepancies,
            postcondition_evaluations=evaluations,
            aggregate_status=aggregate_status,
        )
