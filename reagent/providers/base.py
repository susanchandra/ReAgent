"""
ReAgent Base LLM Provider
Defines the abstract interface for intelligence providers (Nemotron, Development, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from reagent.models import FailureEvent, FailureDiagnosis, RecoveryDecision, StructuredDecision


class LLMResponse(BaseModel):
    content: str
    parsed_json: Optional[Dict[str, Any]] = None
    model_name: str
    latency_ms: int = 0
    token_usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None



class LLMProvider(ABC):
    """Abstract interface for all model backends in ReAgent."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Synchronous text / structured completion."""
        pass

    @abstractmethod
    def diagnose_failure(
        self,
        failure: FailureEvent,
        execution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> FailureDiagnosis:
        """Diagnose root cause and fault attribution of an agent execution failure."""
        pass

    @abstractmethod
    def select_recovery(
        self,
        diagnosis: FailureDiagnosis,
        available_tools: List[str],
        policy_rules: Optional[List[str]] = None,
    ) -> RecoveryDecision:
        """Select the optimal recovery strategy based on diagnosis and environmental constraints."""
        pass

    def make_structured_decision(
        self,
        task_objective: str,
        available_tools: List[str],
        context_summary: Optional[str] = None,
    ) -> Optional[StructuredDecision]:
        """
        Produce a typed StructuredDecision for an incoming task objective.

        Returns StructuredDecision on success, None if the decision cannot be produced
        (treated as MODEL_OUTPUT failure by the pipeline).

        IMPORTANT: Provenance is assigned by the pipeline from code, NOT by reading
        any provenance field from the model's JSON output.
        """
        return None

    def estimate_request_tokens(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> int:
        """
        Estimate tokens for a complete provider request payload.
        Default implementation uses conservative character heuristics and framing.
        """
        from reagent.policy import estimate_prompt_tokens
        return estimate_prompt_tokens(prompt=prompt, system_prompt=system_prompt, messages=messages)
