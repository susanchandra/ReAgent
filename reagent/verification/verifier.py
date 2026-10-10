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

from reagent.models import Evidence, VerificationResult, ActionResult

# Hard cap on bytes read by FILE_CONTAINS to prevent memory exhaustion.
FILE_READ_CAP_BYTES = 1 * 1024 * 1024  # 1 MB


class AssertionRule(BaseModel):
    """Defines an invariant or post-condition that must hold for verification."""
    assertion_type: str     # One of the supported predicates above
    key_path: Optional[str] = None
    expected_value: Optional[Any] = None
    description: str = ""


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
        """
        assertions = assertions or [
            AssertionRule(
                assertion_type="STATUS_SUCCESS",
                description="Action execution must return success=True",
            )
        ]

        if evidence is None or not isinstance(getattr(evidence, "facts", None), dict):
            return VerificationResult(
                recovery_decision_id=recovery_decision_id,
                verified=False,
                evidence_summary="Verification failed: missing or invalid evidence facts.",
                discrepancies=["Missing or invalid evidence."],
            )

        discrepancies: List[str] = []
        facts = evidence.facts

        for rule in assertions:
            atype = rule.assertion_type

            # ── 1. STATUS_SUCCESS ────────────────────────────────────────
            if atype == "STATUS_SUCCESS":
                if not facts.get("success", False):
                    err = facts.get("error", "Unknown error")
                    discrepancies.append(f"Execution failed: {err}")

            # ── 2. KEY_EXISTS ─────────────────────────────────────────────
            elif atype == "KEY_EXISTS":
                output = facts.get("output", {})
                if isinstance(output, dict):
                    if rule.key_path not in output:
                        discrepancies.append(
                            f"Expected key '{rule.key_path}' missing from output."
                        )
                else:
                    discrepancies.append(
                        f"Output is not a dict; cannot verify key '{rule.key_path}'."
                    )

            # ── 3. VALUE_EQUALS ───────────────────────────────────────────
            elif atype == "VALUE_EQUALS":
                output = facts.get("output", {})
                if isinstance(output, dict):
                    actual = output.get(rule.key_path)
                    if actual != rule.expected_value:
                        discrepancies.append(
                            f"Key '{rule.key_path}': expected '{rule.expected_value}', got '{actual}'."
                        )
                else:
                    discrepancies.append(
                        f"Output is not a dict; cannot verify value for '{rule.key_path}'."
                    )

            # ── 4. FILE_EXISTS ────────────────────────────────────────────
            elif atype == "FILE_EXISTS":
                raw = rule.expected_value or rule.key_path
                if not raw:
                    discrepancies.append("FILE_EXISTS: no path provided in predicate.")
                else:
                    try:
                        safe = self._resolve_verified_path(str(raw))
                        if not os.path.exists(safe):
                            discrepancies.append(
                                f"Post-condition failed: file does not exist: '{raw}'."
                            )
                    except PermissionError as exc:
                        discrepancies.append(f"FILE_EXISTS: {exc}")

            # ── 5. FILE_CONTAINS ──────────────────────────────────────────
            elif atype == "FILE_CONTAINS":
                raw = rule.key_path
                substring = str(rule.expected_value or "")
                if not raw:
                    discrepancies.append("FILE_CONTAINS: no path provided in predicate.")
                else:
                    try:
                        safe = self._resolve_verified_path(str(raw))
                        if not os.path.exists(safe):
                            discrepancies.append(
                                f"FILE_CONTAINS: file does not exist: '{raw}'."
                            )
                        else:
                            try:
                                with open(safe, "rb") as fh:
                                    raw_bytes = fh.read(FILE_READ_CAP_BYTES)
                                content = raw_bytes.decode("utf-8", errors="replace")
                                if substring not in content:
                                    discrepancies.append(
                                        f"FILE_CONTAINS: '{raw}' does not contain expected substring."
                                    )
                            except OSError as exc:
                                discrepancies.append(
                                    f"FILE_CONTAINS: could not read '{raw}': {exc}."
                                )
                    except PermissionError as exc:
                        discrepancies.append(f"FILE_CONTAINS: {exc}")

            # ── 6. FILE_SIZE_GTE ──────────────────────────────────────────
            elif atype == "FILE_SIZE_GTE":
                raw = rule.key_path
                min_bytes = int(rule.expected_value or 0)
                if not raw:
                    discrepancies.append("FILE_SIZE_GTE: no path provided in predicate.")
                else:
                    try:
                        safe = self._resolve_verified_path(str(raw))
                        if not os.path.exists(safe):
                            discrepancies.append(
                                f"FILE_SIZE_GTE: file does not exist: '{raw}'."
                            )
                        else:
                            actual_size = os.path.getsize(safe)
                            if actual_size < min_bytes:
                                discrepancies.append(
                                    f"FILE_SIZE_GTE: '{raw}' is {actual_size} bytes "
                                    f"(expected >= {min_bytes})."
                                )
                    except PermissionError as exc:
                        discrepancies.append(f"FILE_SIZE_GTE: {exc}")

            # ── 7. JSON_VALID ─────────────────────────────────────────────
            elif atype == "JSON_VALID":
                # target can be a relative file path (key_path) or inline output
                raw = rule.key_path
                if raw:
                    try:
                        safe = self._resolve_verified_path(str(raw))
                        if os.path.exists(safe):
                            try:
                                with open(safe, "r", encoding="utf-8") as fh:
                                    json.load(fh)
                            except (json.JSONDecodeError, OSError) as exc:
                                discrepancies.append(
                                    f"JSON_VALID: '{raw}' is not valid JSON: {exc}."
                                )
                        # file doesn't exist — fall through to inline check
                        else:
                            raw = None
                    except PermissionError as exc:
                        discrepancies.append(f"JSON_VALID: {exc}")
                        raw = None  # skip inline check

                if not raw:
                    output = facts.get("output")
                    if isinstance(output, str):
                        try:
                            json.loads(output)
                        except json.JSONDecodeError as exc:
                            discrepancies.append(
                                f"JSON_VALID: output string is not valid JSON: {exc}."
                            )
                    elif not isinstance(output, (dict, list)):
                        discrepancies.append(
                            "JSON_VALID: output is not a JSON-serializable type."
                        )

            # ── 8. JSON_PATH_EQUALS ───────────────────────────────────────
            elif atype == "JSON_PATH_EQUALS":
                # key_path is a dot-separated path: "a.b.c"
                output = facts.get("output", {})
                if not isinstance(output, dict):
                    discrepancies.append(
                        "JSON_PATH_EQUALS: output is not a dict."
                    )
                else:
                    keys = (rule.key_path or "").split(".")
                    node: Any = output
                    try:
                        for k in keys:
                            node = node[k]
                        if node != rule.expected_value:
                            discrepancies.append(
                                f"JSON_PATH_EQUALS: '{rule.key_path}' = '{node}' "
                                f"(expected '{rule.expected_value}')."
                            )
                    except (KeyError, TypeError):
                        discrepancies.append(
                            f"JSON_PATH_EQUALS: path '{rule.key_path}' not found in output."
                        )

            # ── 9. HTTP_STATUS ────────────────────────────────────────────
            elif atype == "HTTP_STATUS":
                output = facts.get("output", {})
                actual_status = None
                if isinstance(output, dict):
                    actual_status = output.get("status_code")
                if actual_status != rule.expected_value:
                    discrepancies.append(
                        f"HTTP_STATUS: expected {rule.expected_value}, got {actual_status}."
                    )

            # ── 10. NO_ERROR_PAYLOAD ──────────────────────────────────────
            elif atype == "NO_ERROR_PAYLOAD":
                output = facts.get("output", {})
                if isinstance(output, dict):
                    if output.get("error"):
                        discrepancies.append(
                            f"Output contains embedded error payload: {output['error']}"
                        )
                    if output.get("status_code", 200) >= 400:
                        discrepancies.append(
                            f"HTTP status indicates failure: {output.get('status_code')}"
                        )

            # ── Unknown predicate ─────────────────────────────────────────
            else:
                discrepancies.append(
                    f"Unknown assertion_type '{atype}' — treated as verification failure."
                )

        is_verified = len(discrepancies) == 0
        summary = (
            "All post-condition assertions passed."
            if is_verified
            else f"Verification failed with {len(discrepancies)} discrepancy(ies)."
        )

        return VerificationResult(
            recovery_decision_id=recovery_decision_id,
            verified=is_verified,
            evidence_summary=summary,
            discrepancies=discrepancies,
        )
