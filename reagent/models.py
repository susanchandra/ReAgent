"""
ReAgent Domain Models
Standard schemas for state, failures, diagnostics, recovery decisions, and verifications.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
import uuid

from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus, PostconditionStatus


def get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentState(BaseModel):
    agent_id: str
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: Optional[str] = None
    step_index: int = 0
    failure_streak: int = 0
    context_variables: Dict[str, Any] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=get_utc_now)


class TaskState(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str
    status: str = "PENDING"
    steps_total: int = 0
    steps_completed: int = 0
    created_at: str = Field(default_factory=get_utc_now)


class ExecutionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    step_index: int
    tool_name: str
    input_payload: Dict[str, Any]
    raw_output: Optional[str] = None
    exit_code: int = 0
    duration_ms: int = 0
    timestamp: str = Field(default_factory=get_utc_now)


class FailureEvent(BaseModel):
    failure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    step_index: int
    tool_name: Optional[str] = None
    failure_type: FailureType = FailureType.UNKNOWN
    error_message: str
    stack_trace: Optional[str] = None
    raw_context: Dict[str, Any] = Field(default_factory=dict)
    detected_at: str = Field(default_factory=get_utc_now)


class FailureDiagnosis(BaseModel):
    diagnosis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    failure_id: str
    failure_type: FailureType
    root_cause: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    is_agent_fault: bool
    is_transient: bool
    provenance: str = "deterministic_rule"  # "model" | "deterministic_rule" | "fallback" | "fixture"
    diagnosed_at: str = Field(default_factory=get_utc_now)


class RecoveryDecision(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    diagnosis_id: str
    strategy: RecoveryStrategyType
    rationale: str
    modified_input: Optional[Dict[str, Any]] = None
    fallback_tool_name: Optional[str] = None
    backoff_seconds: float = 0.0
    guard_approved: bool = True
    guard_reasons: List[str] = Field(default_factory=list)
    provenance: str = "deterministic_rule"  # "model" | "deterministic_rule" | "fallback" | "fixture"
    created_at: str = Field(default_factory=get_utc_now)


class PolicyDecision(BaseModel):
    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    is_allowed: bool
    violations: List[str] = Field(default_factory=list)
    checked_rules: List[str] = Field(default_factory=list)
    enforced_at: str = Field(default_factory=get_utc_now)


class ActionResult(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    facts: Dict[str, Any]
    observed_at: str = Field(default_factory=get_utc_now)


class PostconditionEvaluation(BaseModel):
    """Structured evaluation record for a single postcondition."""
    model_config = ConfigDict(frozen=True)
    postcondition_id: Optional[str] = None
    predicate: str
    target: Optional[str] = None
    expected: Optional[Any] = None
    status: PostconditionStatus
    evidence: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    required: bool = True


class VerificationResult(BaseModel):
    verification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recovery_decision_id: str
    verified: bool
    evidence_summary: str
    discrepancies: List[str] = Field(default_factory=list)
    verified_at: str = Field(default_factory=get_utc_now)
    postcondition_evaluations: List[PostconditionEvaluation] = Field(default_factory=list)
    aggregate_status: Optional[AgentStatus] = None


class ReliabilityState(BaseModel):
    agent_id: str
    total_actions: int = 0
    total_failures: int = 0
    successful_recoveries: int = 0
    escalations: int = 0
    mttr_ms_avg: float = 0.0
    reliability_score: float = 100.0  # Percentage
    updated_at: str = Field(default_factory=get_utc_now)


# ─────────────────────────────────────────────────
# Milestone 3 additions
# ─────────────────────────────────────────────────

class Postcondition(BaseModel):
    """A single deterministically verifiable post-condition predicate."""
    model_config = ConfigDict(frozen=True)
    predicate: str          # FILE_EXISTS | FILE_CONTAINS | FILE_SIZE_GTE | JSON_VALID |
                            # JSON_PATH_EQUALS | KEY_EXISTS | VALUE_EQUALS |
                            # HTTP_STATUS | NO_ERROR_PAYLOAD | STATUS_SUCCESS
    target: Optional[str] = None        # file path, json path, HTTP url, etc.
    expected: Optional[Any] = None      # expected value or threshold
    description: str = ""
    required: bool = True               # Whether this postcondition is required vs optional
    postcondition_id: Optional[str] = None




class SuccessContract(BaseModel):
    """
    Immutable specification of how task success will be independently verified.
    Frozen BEFORE tool execution.  Must not be mutated after freezing.
    """
    model_config = ConfigDict(frozen=True)
    contract_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    postconditions: List[Postcondition]
    provenance: str = "fixture"         # "fixture" | "hardcoded_default" | "model"
    frozen_at: str = Field(default_factory=get_utc_now)


class VerifiedReceipt(BaseModel):
    """
    Immutable audit record produced after every verified (or escalated) action.
    VERIFIED  → all postconditions in frozen_contract passed.
    ESCALATED → recovery was attempted but ultimately failed.
    """
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    action_id: str
    tool: str
    frozen_contract: SuccessContract
    evidence: Evidence
    verification_result: VerificationResult
    timestamp: str = Field(default_factory=get_utc_now)
    recovery_count: int = 0
    final_status: AgentStatus       # VERIFIED | ESCALATED


class ObservabilityEvent(BaseModel):
    """
    Single immutable event in the ordered observability timeline.
    Appended in strict chronological order; never mutated.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    phase: str              # TASK_RECEIVED | CONTEXT_SELECTED | MODEL_DECISION |
                            # DECISION_VALIDATED | CONTRACT_FROZEN | POLICY_ALLOWED |
                            # TOOL_EXECUTED | EVIDENCE_COLLECTED | VERIFICATION_FAILED |
                            # FAILURE_ATTRIBUTED | RECOVERY_SELECTED | RECOVERY_ALLOWED |
                            # RECOVERY_EXECUTED | REVERIFICATION | VERIFIED_RECEIPT
    detail: Dict[str, Any] = Field(default_factory=dict)
    occurred_at: str = Field(default_factory=get_utc_now)


class StructuredDecision(BaseModel):
    """
    Typed schema for the Nemotron model's structured JSON output.
    The pipeline validates every model response against this schema.
    Invalid/malformed responses are REJECTED — never treated as success.
    """
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    # Advisory fields (model proposes; pipeline may accept or override)
    proposed_postconditions: List[Postcondition] = Field(default_factory=list)
    diagnosis_category: Optional[str] = None    # FailureType value string
    diagnosis_confidence: float = 0.0
    diagnosis_reason: Optional[str] = None
    recovery_strategy: Optional[str] = None     # RecoveryStrategyType value
    provenance: str = "model"


# ─────────────────────────────────────────────────
# Phase 3 additions: Generic Composite Workflows
# ─────────────────────────────────────────────────

class WorkflowStep(BaseModel):
    """
    Specification of a single step within a CompositeTaskPlan.
    Contains stable step_id, tool/capability, parameters, postconditions, and execution constraints.
    """
    model_config = ConfigDict(frozen=True)
    step_id: str
    tool_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    postconditions: List[Postcondition] = Field(default_factory=list)
    description: str = ""
    required: bool = True
    depends_on: List[str] = Field(default_factory=list)
    supports_idempotency: Optional[bool] = None
    max_recovery_attempts: int = 1


class CompositeTaskPlan(BaseModel):
    """
    Immutable plan defining a multi-step composite workflow.
    Validates step ordering, unique step IDs, and dependency validity.
    """
    model_config = ConfigDict(frozen=True)
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str
    steps: List[WorkflowStep]
    created_at: str = Field(default_factory=get_utc_now)

    def validate_plan(self) -> None:
        """
        Validate workflow plan structure:
        - At least one step
        - Unique step_ids
        - Dependencies refer to preceding steps (no circular or forward references)
        """
        if not self.steps:
            raise ValueError(f"CompositeTaskPlan '{self.task_id}' requires at least one step.")

        seen_ids = set()
        for i, step in enumerate(self.steps):
            if not step.step_id:
                raise ValueError(f"Step at index {i} in plan '{self.task_id}' is missing a step_id.")
            if step.step_id in seen_ids:
                raise ValueError(f"Duplicate step_id '{step.step_id}' found in plan '{self.task_id}'.")
            for dep in step.depends_on:
                if dep not in seen_ids:
                    raise ValueError(
                        f"Step '{step.step_id}' depends on '{dep}', which is not a preceding step in plan '{self.task_id}'."
                    )
            seen_ids.add(step.step_id)


class StepAttempt(BaseModel):
    """
    Audit record of a single execution/recovery attempt for a workflow step.
    Preserves evidence and diagnostic context across retries.
    """
    model_config = ConfigDict(frozen=True)
    attempt_number: int
    action_result: Optional[ActionResult] = None
    evidence: Optional[Evidence] = None
    verification_result: Optional[VerificationResult] = None
    status: AgentStatus
    error: Optional[str] = None
    timestamp: str = Field(default_factory=get_utc_now)


class StepReceipt(BaseModel):
    """
    Final or current outcome receipt for an individual workflow step.
    Preserves full attempt history, authoritative evidence, and structured verification.
    """
    step_id: str
    tool_name: str
    status: AgentStatus
    attempts: List[StepAttempt] = Field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    action_result: Optional[ActionResult] = None
    evidence: Optional[Evidence] = None
    recovery_count: int = 0
    executed: bool = True
    timestamp: str = Field(default_factory=get_utc_now)


class WorkflowReceipt(BaseModel):
    """
    Immutable audit record representing the overall execution of a CompositeTaskPlan.
    Distinguishes completed, incomplete, failed, and unresolved steps.
    """
    task_id: str
    plan: CompositeTaskPlan
    step_receipts: Dict[str, StepReceipt] = Field(default_factory=dict)
    status: AgentStatus
    completed_steps: List[str] = Field(default_factory=list)
    incomplete_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    unresolved_steps: List[str] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list)
    total_recovery_attempts: int = 0
    created_at: str = Field(default_factory=get_utc_now)
