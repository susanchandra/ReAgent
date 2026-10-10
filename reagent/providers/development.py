"""
ReAgent Development LLM Provider
Deterministic, local intelligence provider for testing and offline development.
"""

import time
from typing import Dict, Any, Optional, List
import json
import re

from reagent.providers.base import LLMProvider, LLMResponse
from reagent.models import (
    FailureEvent,
    FailureDiagnosis,
    RecoveryDecision,
    StructuredDecision,
    Postcondition,
)
from reagent.taxonomy import FailureType, RecoveryStrategyType



class DevelopmentLLMProvider(LLMProvider):
    """
    Local deterministic provider.
    Enables immediate development, debugging, and offline unit testing without Nebius API access.
    """

    def __init__(self, model_name: str = "dev-deterministic-engine"):
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        content = f"Deterministic response to: {prompt[:80]}"
        parsed = {}
        if response_schema:
            for key, val_type in response_schema.get("properties", {}).items():
                parsed[key] = "mock_val"

        latency = int((time.perf_counter() - t0) * 1000)
        return LLMResponse(
            content=content,
            parsed_json=parsed if parsed else None,
            model_name=self.model_name,
            latency_ms=latency,
            token_usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": 20, "total_tokens": len(prompt) // 4 + 20},
            raw_response={"status": "mock_success"},
        )

    def diagnose_failure(
        self,
        failure: FailureEvent,
        execution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> FailureDiagnosis:
        """Deterministic root-cause classification based on error signatures."""
        err = failure.error_message.lower()

        # Classification heuristics
        if any(k in err for k in ["timeout", "timed out", "deadline exceeded"]):
            f_type = FailureType.TIMEOUT
            root_cause = "Operation exceeded allocated execution deadline"
            is_agent = False
            is_transient = True
        elif any(k in err for k in ["401", "403", "unauthorized", "forbidden", "permission denied", "bearer token"]):
            f_type = FailureType.AUTHORIZATION
            root_cause = "Missing or invalid credentials/permissions for resource access"
            is_agent = False
            is_transient = False
        elif any(k in err for k in ["jsondecodeerror", "json.decoder", "expecting property name", "schema", "invalid json", "unparsable", "format error"]):
            f_type = FailureType.MODEL_OUTPUT
            root_cause = "Agent output violated expected JSON syntax or schema specification"
            is_agent = True
            is_transient = True
        elif any(k in err for k in [
            "parametervalidationerror",
            "missing required parameter",
            "missing parameter",
            "invalid parameter",
            "missing argument",
            "validation error",
            "type error",
        ]):
            f_type = FailureType.INPUT_CONTEXT
            root_cause = "Invalid or incomplete parameter payload passed to tool"
            is_agent = True
            is_transient = False
        elif any(k in err for k in ["postconditionverificationfailed", "verification failed"]):
            f_type = FailureType.TOOL_FAILURE
            root_cause = "Action executed but post-condition assertions failed to verify"
            is_agent = False
            is_transient = False
        elif any(k in err for k in ["500", "502", "503", "504", "connection reset", "econnreset", "bad gateway", "temporarily unavailable"]):
            f_type = FailureType.TRANSIENT_EXTERNAL
            root_cause = "External dependency experienced temporary server failure or network drop"
            is_agent = False
            is_transient = True
        elif any(k in err for k in [
            "command not found",
            "no such file",
            "tool not found",
            "not in the approved allowlist",
            "securityviolation",
            "unregistered",
        ]):
            f_type = FailureType.ENVIRONMENT_FAILURE
            root_cause = "Tool executable or filesystem resource not present in allowlisted environment"
            is_agent = False
            is_transient = False
        elif failure.failure_type != FailureType.UNKNOWN:
            f_type = failure.failure_type
            root_cause = f"Pre-classified failure: {failure.error_message}"
            is_agent = f_type in (FailureType.AGENT_REASONING, FailureType.MODEL_OUTPUT, FailureType.INPUT_CONTEXT)
            is_transient = f_type in (FailureType.TIMEOUT, FailureType.TRANSIENT_EXTERNAL)
        else:
            f_type = FailureType.TOOL_FAILURE
            root_cause = f"Tool execution failed: {failure.error_message}"
            is_agent = False
            is_transient = False

        confidence = 0.95

        return FailureDiagnosis(
            failure_id=failure.failure_id,
            failure_type=f_type,
            root_cause=root_cause,
            confidence_score=confidence,
            is_agent_fault=is_agent,
            is_transient=is_transient,
            provenance="deterministic_rule",
        )

    def select_recovery(
        self,
        diagnosis: FailureDiagnosis,
        available_tools: List[str],
        policy_rules: Optional[List[str]] = None,
    ) -> RecoveryDecision:
        """Deterministic recovery strategy selector."""
        strategy = RecoveryStrategyType.ABORT
        rationale = "Default fallback"
        modified_input = None
        fallback_tool = None
        backoff = 0.0

        if diagnosis.failure_type == FailureType.TIMEOUT:
            strategy = RecoveryStrategyType.WAIT_AND_RETRY
            backoff = 2.0
            rationale = "Transient timeout detected; executing exponential backoff retry."
        elif diagnosis.failure_type == FailureType.TRANSIENT_EXTERNAL:
            strategy = RecoveryStrategyType.WAIT_AND_RETRY
            backoff = 1.0
            rationale = "Transient remote server error; retrying after brief backoff."
        elif diagnosis.failure_type in (FailureType.INPUT_CONTEXT, FailureType.MODEL_OUTPUT):
            strategy = RecoveryStrategyType.RETRY_MODIFIED_INPUT
            modified_input = {"retry_flag": True, "reformatted": True}
            rationale = "Payload schema discrepancy; reformatted parameters for retry."
        elif diagnosis.failure_type == FailureType.ENVIRONMENT_FAILURE:
            if available_tools:
                strategy = RecoveryStrategyType.FALLBACK_TOOL
                fallback_tool = available_tools[0]
                rationale = f"Primary tool unavailable; falling back to alternative: {fallback_tool}"
            else:
                strategy = RecoveryStrategyType.ESCALATE
                rationale = "Environment dependency failed and no fallback tools registered."
        elif diagnosis.failure_type == FailureType.TOOL_FAILURE:
            if "post-condition" in diagnosis.root_cause.lower() or "assertion" in diagnosis.root_cause.lower():
                strategy = RecoveryStrategyType.RETRY_MODIFIED_INPUT
                rationale = "Post-condition check failed; modifying execution parameters to satisfy assertion."
            elif available_tools:
                strategy = RecoveryStrategyType.FALLBACK_TOOL
                fallback_tool = available_tools[0]
                rationale = f"Tool failure detected; switching to fallback tool: {fallback_tool}"
            else:
                strategy = RecoveryStrategyType.ESCALATE
                rationale = "Tool failure without alternative tools available."
        elif diagnosis.failure_type == FailureType.AUTHORIZATION:
            strategy = RecoveryStrategyType.ESCALATE
            rationale = "Security/Permission violation requires human authorization."
        else:
            strategy = RecoveryStrategyType.ESCALATE
            rationale = f"Unrecoverable {diagnosis.failure_type.value} state requires user intervention."

        return RecoveryDecision(
            diagnosis_id=diagnosis.diagnosis_id,
            strategy=strategy,
            rationale=rationale,
            modified_input=modified_input,
            fallback_tool_name=fallback_tool,
            backoff_seconds=backoff,
            guard_approved=True,
            guard_reasons=[],
            provenance="deterministic_rule",
        )

    def make_structured_decision(
        self,
        task_objective: str,
        available_tools: List[str],
        context_summary: Optional[str] = None,
    ) -> Optional[StructuredDecision]:
        """
        Deterministic structured decision for offline testing.
        Chooses the first available tool and proposes STATUS_SUCCESS post-condition.
        Provenance is NOT set here — the pipeline assigns 'fixture' from code.
        """
        tool = available_tools[0] if available_tools else "json_validator"
        return StructuredDecision(
            tool=tool,
            arguments={},
            proposed_postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    description="Tool must succeed",
                )
            ],
            diagnosis_category=None,
            diagnosis_confidence=0.0,
            diagnosis_reason="Deterministic default decision",
            recovery_strategy=None,
            # provenance intentionally omitted — pipeline sets it
        )
