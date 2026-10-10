"""
ReAgent Context Selector
Constructs minimal, allowlisted, token-efficient context for the LLM.
Prevents context pollution by strictly selecting relevant state, bounded history,
active policies, and approved tool schemas.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json

from reagent.models import AgentState, TaskState, ExecutionEvent, FailureEvent, FailureDiagnosis


class ContextWindow(BaseModel):
    """Structured, bounded context payload ready for LLM consumption."""
    agent_id: str
    task_objective: str
    current_status: str
    step_index: int
    failure_streak: int
    bounded_history: List[Dict[str, Any]] = Field(default_factory=list)
    failure_details: Optional[Dict[str, Any]] = None
    allowlisted_tools: List[Dict[str, Any]] = Field(default_factory=list)
    active_policy_rules: List[str] = Field(default_factory=list)

    def to_system_context(self) -> str:
        """Render minimal structured context as a compact string."""
        return (
            f"AGENT_ID: {self.agent_id} | STATUS: {self.current_status} | STEP: {self.step_index}\n"
            f"OBJECTIVE: {self.task_objective}\n"
            f"FAILURE_STREAK: {self.failure_streak}\n"
            f"POLICIES: {', '.join(self.active_policy_rules) if self.active_policy_rules else 'STANDARD'}\n"
            f"AVAILABLE_TOOLS: {[t.get('name') for t in self.allowlisted_tools]}\n"
        )


class ContextSelector:
    """
    Selects and sanitizes only the necessary context items.
    Enforces a strict sliding window on execution events and truncates oversized payloads.
    """

    def __init__(
        self,
        max_history_items: int = 4,
        max_payload_chars: int = 400,
    ):
        self.max_history_items = max_history_items
        self.max_payload_chars = max_payload_chars

    def _sanitize_payload(self, payload: Any) -> Any:
        """Truncates long strings or large arrays in payloads."""
        if isinstance(payload, str):
            if len(payload) > self.max_payload_chars:
                return payload[:self.max_payload_chars] + "...[TRUNCATED]"
            return payload
        elif isinstance(payload, dict):
            return {k: self._sanitize_payload(v) for k, v in payload.items()}
        elif isinstance(payload, list):
            return [self._sanitize_payload(item) for item in payload[:5]]
        return payload

    def select_context(
        self,
        agent_state: AgentState,
        task: TaskState,
        execution_history: Optional[List[ExecutionEvent]] = None,
        failure_event: Optional[FailureEvent] = None,
        failure_diagnosis: Optional[FailureDiagnosis] = None,
        allowlisted_tools: Optional[List[Dict[str, Any]]] = None,
        policy_rules: Optional[List[str]] = None,
    ) -> ContextWindow:
        """Constructs the bounded ContextWindow."""
        # 1. Bounded history (last N events, sanitized)
        history_slice: List[Dict[str, Any]] = []
        if execution_history:
            recent = execution_history[-self.max_history_items:]
            for evt in recent:
                history_slice.append({
                    "step": evt.step_index,
                    "tool": evt.tool_name,
                    "input": self._sanitize_payload(evt.input_payload),
                    "exit_code": evt.exit_code,
                    "duration_ms": evt.duration_ms,
                })

        # 2. Failure context (if present)
        failure_dict = None
        if failure_event:
            failure_dict = {
                "step": failure_event.step_index,
                "tool": failure_event.tool_name,
                "error": self._sanitize_payload(failure_event.error_message),
                "type": failure_event.failure_type.value,
            }
            if failure_diagnosis:
                failure_dict["diagnosis"] = {
                    "root_cause": failure_diagnosis.root_cause,
                    "confidence": failure_diagnosis.confidence_score,
                    "is_agent_fault": failure_diagnosis.is_agent_fault,
                    "is_transient": failure_diagnosis.is_transient,
                }

        # 3. Allowlisted tools minimal schemas
        tools_summary = []
        if allowlisted_tools:
            for t in allowlisted_tools:
                tools_summary.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                })

        return ContextWindow(
            agent_id=agent_state.agent_id,
            task_objective=task.objective,
            current_status=agent_state.status.value,
            step_index=agent_state.step_index,
            failure_streak=agent_state.failure_streak,
            bounded_history=history_slice,
            failure_details=failure_dict,
            allowlisted_tools=tools_summary,
            active_policy_rules=policy_rules or ["MAX_RETRIES=3", "ALLOWLISTED_TOOLS_ONLY"],
        )
