"""
faults.py — Deterministic fault modes for the ambiguity benchmark.

Crucial constraint:
For matched FAIL_BEFORE_COMMIT and LOST_RESPONSE scenarios, the agent-visible
exception must be bit-for-bit / canonically serialized IDENTICAL:
same exception class, message, code, and exposed metadata.
"""

from typing import Dict, Any, Optional
import json
import hashlib
from reagent.experiments.ambiguity.manifest import (
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
    FAULT_DELAYED_VISIBILITY,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_UNOBSERVABLE,
)


class ToolExecutionTimeoutError(Exception):
    """
    Standard agent-visible exception thrown when a remote operation times out.
    Must maintain canonical fields across matched fault scenarios.
    """
    def __init__(self, message: str, operation: str, request_id: str, code: str = "ERR_TIMEOUT"):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.request_id = request_id
        self.code = code

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Canonical representation for identifiability verification."""
        return {
            "error_type": "ToolExecutionTimeoutError",
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "request_id": self.request_id,
        }

    def canonical_hash(self) -> str:
        data = json.dumps(self.to_canonical_dict(), sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


class StatusUnavailableError(Exception):
    """Thrown when status/probe endpoint is unavailable."""
    def __init__(self, message: str, operation: str, code: str = "ERR_STATUS_UNAVAILABLE"):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.code = code


class UnobservableEvidenceError(Exception):
    """Thrown when an external system has no verifiable observation surface."""
    def __init__(self, message: str, operation: str, code: str = "ERR_UNOBSERVABLE"):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.code = code


def make_standard_timeout_error(operation: str, request_id: str) -> ToolExecutionTimeoutError:
    """
    Factory creating byte-identical exceptions for FAIL_BEFORE_COMMIT
    and LOST_RESPONSE scenarios.
    """
    return ToolExecutionTimeoutError(
        message=f"Operation '{operation}' timed out after 5000ms waiting for upstream confirmation",
        operation=operation,
        request_id=request_id,
        code="ERR_TIMEOUT",
    )
