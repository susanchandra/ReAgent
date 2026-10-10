"""
ReAgent Nebius Provider  (Milestone 4B revision)
Uses the openai Python SDK with Nebius Token Factory base_url override.
Produces and validates STRUCTURED JSON decisions from Nemotron.
API key is read exclusively from NEBIUS_API_KEY env var — never logged or committed.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

from openai import OpenAI, APIError

from reagent.providers.base import LLMProvider, LLMResponse
from reagent.models import (
    FailureEvent, FailureDiagnosis, RecoveryDecision,
    StructuredDecision, Postcondition,
)
from reagent.taxonomy import FailureType, RecoveryStrategyType

logger = logging.getLogger(__name__)

# Default verified model (confirmed live on Nebius Token Factory).
# Override via NEBIUS_NEMOTRON_MODEL env var.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

DEFAULT_MAX_TOKENS = 1536
HARD_MAX_OUTPUT_TOKENS = 4096
HARD_MIN_OUTPUT_TOKENS = 16


def get_configured_max_tokens(override: Optional[int] = None) -> int:
    """
    Read REAGENT_MAX_TOKENS with safe default (1536) and strict code-enforced bounds [16, 4096].
    Rejects invalid/out-of-range values.
    """
    val = override
    if val is None:
        raw = os.environ.get("REAGENT_MAX_TOKENS")
        if raw is not None and raw.strip():
            try:
                val = int(raw.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid REAGENT_MAX_TOKENS value '{raw}': must be an integer.") from exc

    if val is None:
        val = DEFAULT_MAX_TOKENS

    if not isinstance(val, int) or val < HARD_MIN_OUTPUT_TOKENS or val > HARD_MAX_OUTPUT_TOKENS:
        raise ValueError(
            f"REAGENT_MAX_TOKENS {val} out of permitted range "
            f"[{HARD_MIN_OUTPUT_TOKENS}, {HARD_MAX_OUTPUT_TOKENS}]."
        )
    return val


# ---------------------------------------------------------------------------
# Structured output schema (as a JSON string we inject into the system prompt)
# ---------------------------------------------------------------------------
STRUCTURED_DECISION_SCHEMA = """{
  "tool": "<tool_name_string>",
  "arguments": {},
  "proposed_postconditions": [
    {"predicate": "STATUS_SUCCESS", "description": "Action must succeed"}
  ],
  "diagnosis_category": "<FailureType or null>",
  "diagnosis_confidence": 0.0,
  "diagnosis_reason": "<string or null>",
  "recovery_strategy": "<RecoveryStrategyType or null>"
}"""

DIAGNOSIS_SCHEMA = """{
  "failure_type": "<FailureType value>",
  "root_cause": "<concise string>",
  "confidence_score": 0.95,
  "is_agent_fault": false,
  "is_transient": true
}"""

RECOVERY_SCHEMA = """{
  "strategy": "<RecoveryStrategyType value>",
  "rationale": "<concise string>",
  "modified_input": {},
  "fallback_tool_name": null,
  "backoff_seconds": 0.0
}"""

ALLOWED_STRUCTURED_DECISION_KEYS = {
    "tool",
    "arguments",
    "proposed_postconditions",
    "diagnosis_category",
    "diagnosis_confidence",
    "diagnosis_reason",
    "recovery_strategy",
}


def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts JSON extraction from raw model content.
    First tries a direct parse; then a fence-stripped parse.
    Returns None if both fail — callers MUST handle None as a MODEL_OUTPUT failure.
    """
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting first {...} block (handles markdown code fences)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


class NebiusNemotronProvider(LLMProvider):
    """
    Nebius Token Factory provider using the OpenAI-compatible SDK.

    Security rules (enforced here):
    - API key read ONLY from env var NEBIUS_API_KEY.
    - Key is NEVER logged, printed, stored in responses, or included in prompts.
    - Model reasoning content (chain-of-thought) is stripped before returning
      structured decisions — it is never exposed to callers, stored, or parsed.
    - An injected client can be supplied for offline unit testing without network calls.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.model = model or os.environ.get("NEBIUS_NEMOTRON_MODEL", DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL)

        self.last_usage: Optional[Dict[str, int]] = None
        self.last_finish_reason: Optional[str] = None
        self.last_validation_error: Optional[str] = None

        if client is not None:
            # Client injected directly (for offline testing)
            self._client = client
            self._is_configured = True
        else:
            # Key comes exclusively from environment — never accept as constructor arg.
            api_key = os.environ.get("NEBIUS_API_KEY", "")
            self._is_configured = bool(api_key and api_key.strip())
            if self._is_configured:
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=self._base_url,
                )
            else:
                self._client = None

    def is_configured(self) -> bool:
        return self._is_configured

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Return non-sensitive prompt/completion/total token counts from last call."""
        return dict(self.last_usage) if self.last_usage else None

    def estimate_request_tokens(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> int:
        from reagent.policy import estimate_prompt_tokens
        return estimate_prompt_tokens(prompt=prompt, system_prompt=system_prompt, messages=messages)

    def estimate_decision_request_tokens(
        self,
        task_objective: str,
        available_tools: List[str],
        context_summary: Optional[str] = None,
    ) -> int:
        """
        Estimate tokens for the ACTUAL request messages sent in make_structured_decision:
        Includes system prompt instructions, schema definition, available tools, context, and user objective.
        """
        system = (
            "You are ReAgent's Decision Engine. "
            "You produce a SINGLE structured JSON object describing the next action. "
            "Output ONLY valid JSON matching exactly the provided schema. "
            "Do NOT include explanation, markdown, or chain-of-thought in your output. "
            f"Available tools: {available_tools}. "
            f"Failure taxonomy: {[t.value for t in FailureType]}. "
            f"Recovery strategies: {[s.value for s in RecoveryStrategyType]}."
        )
        prompt = (
            f"Task: {task_objective}\n"
            + (f"Context: {context_summary}\n" if context_summary else "")
            + f"Schema to fill:\n{STRUCTURED_DECISION_SCHEMA}"
        )
        return self.estimate_request_tokens(prompt=prompt, system_prompt=system)

    def estimate_diagnosis_request_tokens(
        self,
        failure: FailureEvent,
        execution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Estimate tokens for the actual request messages sent in diagnose_failure."""
        system = (
            "You are ReAgent's Diagnostic Engine. "
            "Analyze the failure event and categorize it into the exact FailureType taxonomy. "
            "Output ONLY valid JSON matching the schema. "
            "Do NOT include explanation outside the JSON."
        )
        safe_history = [
            e.model_dump() if hasattr(e, "model_dump") else (e.dict() if hasattr(e, "dict") else str(e))
            for e in (execution_history or [])[-5:]
        ]
        prompt = (
            f"Failure:\n"
            f"  Tool: {failure.tool_name}\n"
            f"  Error: {failure.error_message}\n"
            f"  Context keys: {list(failure.raw_context.keys())}\n"
            f"Recent history ({len(safe_history)} events): {json.dumps(safe_history, default=str)}\n\n"
            f"Taxonomy options: {[t.value for t in FailureType]}\n"
            f"Schema:\n{DIAGNOSIS_SCHEMA}"
        )
        return self.estimate_request_tokens(prompt=prompt, system_prompt=system)

    def estimate_recovery_request_tokens(
        self,
        diagnosis: FailureDiagnosis,
        available_tools: List[str],
        policy_rules: Optional[List[str]] = None,
    ) -> int:
        """Estimate tokens for the actual request messages sent in select_recovery."""
        system = (
            "You are ReAgent's Recovery Decision Engine. "
            "Select the safest, most appropriate RecoveryStrategy for the diagnosed failure. "
            "Output ONLY valid JSON matching the schema. "
            "Prefer conservative strategies. "
            "Do NOT invent new tools. Only select tools from available_tools."
        )
        prompt = (
            f"Diagnosis:\n"
            f"  Type: {diagnosis.failure_type.value}\n"
            f"  Root Cause: {diagnosis.root_cause}\n"
            f"  Is Transient: {diagnosis.is_transient}\n"
            f"  Is Agent Fault: {diagnosis.is_agent_fault}\n"
            f"Available tools: {available_tools}\n"
            f"Policy rules: {policy_rules or []}\n"
            f"Strategies: {[s.value for s in RecoveryStrategyType]}\n"
            f"Schema:\n{RECOVERY_SCHEMA}"
        )
        return self.estimate_request_tokens(prompt=prompt, system_prompt=system)

    # ------------------------------------------------------------------ #
    # Core generate()                                                      #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        self.last_usage = None
        self.last_finish_reason = None
        self.last_validation_error = None
        if not self._is_configured:
            raise RuntimeError(
                "NEBIUS_API_KEY is not set. "
                "Use DevelopmentLLMProvider for offline testing."
            )

        validated_max_tokens = get_configured_max_tokens(max_tokens)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": validated_max_tokens,
        }
        # Request JSON mode when a schema is supplied
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        except APIError as exc:
            raise RuntimeError(f"Nebius API error: {exc.message}") from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)

        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", None)
        self.last_finish_reason = finish_reason

        # STRICT SEPARATION: content must come from message.content ONLY.
        # NEVER parse reasoning_content / reasoning as the completion content!
        content: str = getattr(message, "content", None) or ""

        # Attempt structured parse only if schema was requested and not truncated
        parsed: Optional[Dict[str, Any]] = None
        if finish_reason == "length":
            logger.warning("Nebius model response truncated (finish_reason='length'). Parse rejected.")
        elif response_schema is not None and content.strip():
            parsed = _safe_parse_json(content)
            if parsed is None:
                logger.warning(
                    "Nebius model returned non-parseable JSON for structured request. "
                    "Content length=%d", len(content)
                )

        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0)),
                "completion_tokens": int(getattr(response.usage, "completion_tokens", 0)),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0)),
            }
        self.last_usage = dict(usage) if usage else None

        return LLMResponse(
            content=content,
            parsed_json=parsed,
            model_name=self.model,
            latency_ms=latency_ms,
            token_usage=usage,
            finish_reason=finish_reason,
            raw_response=None,          # Never store raw response (may contain key echoes)
        )

    # ------------------------------------------------------------------ #
    # Structured decision: ask Nemotron for a typed ReAgent decision       #
    # ------------------------------------------------------------------ #

    def make_structured_decision(
        self,
        task_objective: str,
        available_tools: List[str],
        context_summary: Optional[str] = None,
        max_tokens: int = 512,
    ) -> Optional[StructuredDecision]:
        """
        Ask Nemotron to produce a typed, schema-validated structured decision.
        Returns None if the model output is invalid (treated as MODEL_OUTPUT failure).
        Reasoning content is stripped — only structured JSON is returned.
        """
        system = (
            "You are ReAgent's Decision Engine. "
            "You produce a SINGLE structured JSON object describing the next action. "
            "Output ONLY valid JSON matching exactly the provided schema. "
            "Do NOT include explanation, markdown, or chain-of-thought in your output. "
            f"Available tools: {available_tools}. "
            f"Failure taxonomy: {[t.value for t in FailureType]}. "
            f"Recovery strategies: {[s.value for s in RecoveryStrategyType]}."
        )
        prompt = (
            f"Task: {task_objective}\n"
            + (f"Context: {context_summary}\n" if context_summary else "")
            + f"Schema to fill:\n{STRUCTURED_DECISION_SCHEMA}"
        )

        res = self.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=0.1,
            max_tokens=max_tokens,
            response_schema={},
        )

        if res.finish_reason == "length":
            logger.warning("make_structured_decision: truncated by finish_reason='length'.")
            self.last_validation_error = "truncated"
            return None

        if res.parsed_json is None:
            logger.warning("make_structured_decision: model returned unparseable output.")
            self.last_validation_error = "unparseable_json"
            return None

        decision = self._validate_structured_decision(res.parsed_json)
        if decision is None:
            self.last_validation_error = "malformed_output"
        return decision

    def _validate_structured_decision(
        self, raw: Dict[str, Any]
    ) -> Optional[StructuredDecision]:
        """
        Schema-validate and construct StructuredDecision.
        Strictly rejects unknown fields, injected provenance, and malformed structures.
        """
        if not isinstance(raw, dict):
            logger.warning("StructuredDecision raw payload is not a dict.")
            return None

        # Disallow model attempting to set provenance
        if "provenance" in raw:
            logger.warning("StructuredDecision rejected: payload attempted to set 'provenance'.")
            return None

        # Reject unknown top-level JSON fields
        unknown_fields = set(raw.keys()) - ALLOWED_STRUCTURED_DECISION_KEYS
        if unknown_fields:
            logger.warning("StructuredDecision rejected: unknown fields present: %s", unknown_fields)
            return None

        if "tool" not in raw or not isinstance(raw.get("tool"), str):
            logger.warning("StructuredDecision missing/invalid 'tool' field.")
            return None
        if not raw["tool"].strip():
            logger.warning("StructuredDecision 'tool' field is empty.")
            return None

        # Parse proposed postconditions (advisory — pipeline may override)
        pcs: List[Postcondition] = []
        raw_pcs = raw.get("proposed_postconditions")
        if raw_pcs is not None:
            if not isinstance(raw_pcs, list):
                logger.warning("StructuredDecision 'proposed_postconditions' is not a list.")
                return None
            for pc_raw in raw_pcs:
                if isinstance(pc_raw, dict) and "predicate" in pc_raw:
                    try:
                        pcs.append(Postcondition(**pc_raw))
                    except Exception:
                        pass  # skip malformed individual postcondition

        args = raw.get("arguments", {})
        if not isinstance(args, dict):
            logger.warning("StructuredDecision 'arguments' is not a dict.")
            return None

        return StructuredDecision(
            tool=raw["tool"].strip(),
            arguments=args,
            proposed_postconditions=pcs,
            diagnosis_category=raw.get("diagnosis_category"),
            diagnosis_confidence=float(raw.get("diagnosis_confidence", 0.0)),
            diagnosis_reason=raw.get("diagnosis_reason"),
            recovery_strategy=raw.get("recovery_strategy"),
        )

    # ------------------------------------------------------------------ #
    # diagnose_failure()                                                   #
    # ------------------------------------------------------------------ #

    def diagnose_failure(
        self,
        failure: FailureEvent,
        execution_history: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 256,
    ) -> FailureDiagnosis:
        system = (
            "You are ReAgent's Diagnostic Engine. "
            "Analyze the failure event and categorize it into the exact FailureType taxonomy. "
            "Output ONLY valid JSON matching the schema. "
            "Do NOT include explanation outside the JSON."
        )
        safe_history = [
            e.model_dump() if hasattr(e, "model_dump") else (e.dict() if hasattr(e, "dict") else str(e))
            for e in (execution_history or [])[-5:]
        ]
        prompt = (
            f"Failure:\n"
            f"  Tool: {failure.tool_name}\n"
            f"  Error: {failure.error_message}\n"
            f"  Context keys: {list(failure.raw_context.keys())}\n"
            f"Recent history ({len(safe_history)} events): {json.dumps(safe_history, default=str)}\n\n"
            f"Taxonomy options: {[t.value for t in FailureType]}\n"
            f"Schema:\n{DIAGNOSIS_SCHEMA}"
        )

        res = self.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=0.1,
            max_tokens=max_tokens,
            response_schema={},
        )
        parsed = res.parsed_json or {}

        if not parsed or res.finish_reason == "length":
            # MODEL_OUTPUT failure: cannot parse diagnosis
            return FailureDiagnosis(
                failure_id=failure.failure_id,
                failure_type=FailureType.MODEL_OUTPUT,
                root_cause="Nemotron returned unparseable diagnosis output.",
                confidence_score=0.0,
                is_agent_fault=True,
                is_transient=False,
                provenance="fallback",
            )

        f_type_str = parsed.get("failure_type", FailureType.UNKNOWN.value)
        try:
            f_type = FailureType(f_type_str)
        except ValueError:
            f_type = FailureType.UNKNOWN

        return FailureDiagnosis(
            failure_id=failure.failure_id,
            failure_type=f_type,
            root_cause=str(parsed.get("root_cause", "Nemotron diagnosis"))[:500],
            confidence_score=min(1.0, max(0.0, float(parsed.get("confidence_score", 0.9)))),
            is_agent_fault=bool(parsed.get("is_agent_fault", False)),
            is_transient=bool(parsed.get("is_transient", False)),
            provenance="model",
        )

    # ------------------------------------------------------------------ #
    # select_recovery()                                                    #
    # ------------------------------------------------------------------ #

    def select_recovery(
        self,
        diagnosis: FailureDiagnosis,
        available_tools: List[str],
        policy_rules: Optional[List[str]] = None,
        max_tokens: int = 256,
    ) -> RecoveryDecision:
        system = (
            "You are ReAgent's Recovery Decision Engine. "
            "Select the safest, most appropriate RecoveryStrategy for the diagnosed failure. "
            "Output ONLY valid JSON matching the schema. "
            "Prefer conservative strategies. "
            "Do NOT invent new tools. Only select tools from available_tools."
        )
        prompt = (
            f"Diagnosis:\n"
            f"  Type: {diagnosis.failure_type.value}\n"
            f"  Root Cause: {diagnosis.root_cause}\n"
            f"  Is Transient: {diagnosis.is_transient}\n"
            f"  Is Agent Fault: {diagnosis.is_agent_fault}\n"
            f"Available tools: {available_tools}\n"
            f"Policy rules: {policy_rules or []}\n"
            f"Strategies: {[s.value for s in RecoveryStrategyType]}\n"
            f"Schema:\n{RECOVERY_SCHEMA}"
        )

        res = self.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=0.1,
            max_tokens=max_tokens,
            response_schema={},
        )
        parsed = res.parsed_json or {}

        if not parsed or res.finish_reason == "length":
            # Fallback: escalate when recovery selection fails
            return RecoveryDecision(
                diagnosis_id=diagnosis.diagnosis_id,
                strategy=RecoveryStrategyType.ESCALATE,
                rationale="Recovery selection failed: Nemotron returned unparseable output.",
                guard_approved=True,
                guard_reasons=[],
                provenance="fallback",
            )

        strat_str = parsed.get("strategy", RecoveryStrategyType.ESCALATE.value)
        try:
            strat = RecoveryStrategyType(strat_str)
        except ValueError:
            strat = RecoveryStrategyType.ESCALATE

        # Validate fallback tool is in allowlist
        fallback = parsed.get("fallback_tool_name")
        if fallback and fallback not in available_tools:
            fallback = None

        return RecoveryDecision(
            diagnosis_id=diagnosis.diagnosis_id,
            strategy=strat,
            rationale=str(parsed.get("rationale", "Nemotron recovery decision"))[:500],
            modified_input=parsed.get("modified_input") if isinstance(parsed.get("modified_input"), dict) else None,
            fallback_tool_name=fallback,
            backoff_seconds=max(0.0, float(parsed.get("backoff_seconds", 0.0))),
            guard_approved=True,
            guard_reasons=[],
            provenance="model",
        )
