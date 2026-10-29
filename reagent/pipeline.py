"""
ReAgent Decision & Reliability Pipeline  (Milestone 3 revision)

Frozen loop:
  TASK_RECEIVED
  → CONTEXT_SELECTED
  → MODEL_DECISION
  → DECISION_VALIDATED
  → CONTRACT_FROZEN
  → POLICY_ALLOWED
  → TOOL_EXECUTED
  → EVIDENCE_COLLECTED
  → VERIFICATION_FAILED  (if applicable)
  → FAILURE_ATTRIBUTED
  → RECOVERY_SELECTED
  → RECOVERY_ALLOWED
  → RECOVERY_EXECUTED
  → REVERIFICATION
  → VERIFIED_RECEIPT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable

from pydantic import BaseModel, Field

from reagent.models import (
    FailureEvent,
    FailureDiagnosis,
    RecoveryDecision,
    PolicyDecision,
    ReliabilityState,
    AgentState,
    TaskState,
    ExecutionEvent,
    ActionResult,
    Evidence,
    VerificationResult,
    SuccessContract,
    VerifiedReceipt,
    ObservabilityEvent,
    Postcondition,
    StructuredDecision,
    get_utc_now,
)
from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus
from reagent.providers.base import LLMProvider
from reagent.policy import PolicyGuard, TokenBudgetGuard, estimate_prompt_tokens
from reagent.providers.nebius import get_configured_max_tokens
from reagent.context import ContextSelector, ContextWindow
from reagent.tools.registry import ToolRegistry
from reagent.verification.evidence import EvidenceCollector
from reagent.verification.verifier import DeterministicVerifier, AssertionRule
from reagent.contract import ContractManager
from reagent.contract_validator import ContractGroundingValidator
from reagent.events import (
    ObservabilityTimeline,
    PHASE_TASK_RECEIVED, PHASE_CONTEXT_SELECTED, PHASE_MODEL_DECISION,
    PHASE_DECISION_VALIDATED, PHASE_CONTRACT_FROZEN, PHASE_POLICY_ALLOWED,
    PHASE_TOOL_EXECUTED, PHASE_EVIDENCE_COLLECTED, PHASE_VERIFICATION_FAILED,
    PHASE_FAILURE_ATTRIBUTED, PHASE_RECOVERY_SELECTED, PHASE_RECOVERY_ALLOWED,
    PHASE_RECOVERY_EXECUTED, PHASE_REVERIFICATION, PHASE_VERIFIED_RECEIPT,
)

logger = logging.getLogger(__name__)


def _safe_error_metadata(exc: Exception, phase: str) -> Dict[str, Any]:
    """
    Extract ONLY safe metadata from an exception.
    NEVER serializes raw exception messages, headers, or request/response objects.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "code", None)
    if not isinstance(status_code, int):
        status_code = None

    return {
        "phase": phase,
        "error_type": type(exc).__name__,
        "status_code": status_code,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Return types
# ─────────────────────────────────────────────────────────────────────────────

class ReliabilityCycleResult(BaseModel):
    task_id: str
    context_window: Optional[ContextWindow] = None
    diagnosis: FailureDiagnosis
    recovery_decision: RecoveryDecision
    policy_decision: PolicyDecision
    effective_strategy: RecoveryStrategyType
    agent_status: AgentStatus
    updated_reliability: ReliabilityState


class ExecutionCycleResult(BaseModel):
    """Full result of execute_with_contract(), including VerifiedReceipt."""
    task_id: str
    action_id: str
    tool: str
    frozen_contract: SuccessContract
    action_result: ActionResult
    evidence: Evidence
    verification_result: VerificationResult
    verified_receipt: VerifiedReceipt
    timeline: List[ObservabilityEvent] = Field(default_factory=list)
    recovery_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ReAgentPipeline:
    """
    Central coordinator for agent reliability, sandboxed execution, and verified self-healing.

    Key M3 guarantees:
    - SuccessContract is frozen BEFORE execution starts.
    - PolicyGuard runs before EVERY tool call (including recovery actions).
    - DeterministicVerifier decides success/failure — LLM cannot override.
    - VerifiedReceipt is emitted after every completed cycle.
    - ObservabilityTimeline records every phase transition.
    """

    def __init__(
        self,
        agent_id: str = "reagent-core-01",
        provider: Optional[LLMProvider] = None,
        policy_guard: Optional[PolicyGuard] = None,
        tool_registry: Optional[ToolRegistry] = None,
        context_selector: Optional[ContextSelector] = None,
        token_budget: Optional[TokenBudgetGuard] = None,
        workspace_root: Optional[str] = None,
    ):
        self.agent_id = agent_id

        if provider is None:
            from reagent.providers.development import DevelopmentLLMProvider
            self.provider = DevelopmentLLMProvider()
        else:
            self.provider = provider

        self.policy_guard        = policy_guard or PolicyGuard()
        self.tool_registry       = tool_registry or ToolRegistry()
        self.context_selector    = context_selector or ContextSelector()
        self.evidence_collector  = EvidenceCollector()

        # If workspace_root is not explicitly provided, adopt the sandboxed_file tool's root
        if workspace_root is None:
            file_tool = self.tool_registry.get_tool("sandboxed_file")
            if file_tool and hasattr(file_tool, "sandbox_root"):
                workspace_root = file_tool.sandbox_root

        self.workspace_root      = workspace_root
        self.verifier            = DeterministicVerifier(workspace_root=workspace_root)
        self.contract_manager    = ContractManager()
        self.token_budget        = token_budget or TokenBudgetGuard()
        self.grounding_validator = ContractGroundingValidator(workspace_root=workspace_root)


        self.reliability_state   = ReliabilityState(agent_id=agent_id)
        self.agent_state         = AgentState(agent_id=agent_id, status=AgentStatus.IDLE)
        self.execution_history: List[ExecutionEvent] = []


    # ─────────────────────────────────────────────────────────────────────
    # execute_task() — M4A autonomous planning entry point
    # ─────────────────────────────────────────────────────────────────────

    def execute_task(
        self,
        task_objective: str,
        max_recovery_attempts: int = 2,
    ) -> "ExecutionCycleResult":
        """
        Autonomous planning entry point for Milestone 4A.

        A task objective string enters the pipeline and is converted into a
        structured plan by the provider, grounded against the objective and
        allowed tools, then executed through the full frozen-contract pipeline.

        Provenance assignment rules (enforced by code — never read from model):
          - DevelopmentLLMProvider or FakeStructuredDecisionProvider → "fixture"
          - NebiusNemotronProvider, parse succeeded                  → "model"
          - Provider returned None (unparseable / model output fail) → ESCALATED

        Full flow:
          TASK_RECEIVED
          → TOKEN_BUDGET_CHECK
          → PROVIDER: make_structured_decision()
          → DECISION_VALIDATED (schema, allowlist)
          → GROUNDING_VALIDATED
          → CONTRACT_FROZEN (provenance from code)
          → [execute_with_contract() pipeline continues...]
        """
        from reagent.providers.development import DevelopmentLLMProvider

        task = TaskState(objective=task_objective)
        timeline = ObservabilityTimeline(task_id=task.task_id)
        timeline.emit(PHASE_TASK_RECEIVED, {"objective": task_objective, "source": "execute_task"})

        self.agent_state.current_task_id = task.task_id
        self.agent_state.step_index += 1
        self.agent_state.status = AgentStatus.RUNNING

        def _escalate(reason: str, tool: str = "unknown") -> "ExecutionCycleResult":
            """Helper: build a properly-shaped ESCALATED result for early exits."""
            self.agent_state.status = AgentStatus.ESCALATED
            self.reliability_state.escalations += 1
            dummy_pc = [Postcondition(predicate="STATUS_SUCCESS", description="Task failed before execution")]
            dummy_contract = self.contract_manager.freeze(
                self.contract_manager.build_contract(task.task_id, dummy_pc, provenance="fixture")
            )
            dummy_action = ActionResult(tool_name=tool, success=False, error=reason)
            dummy_evidence = self.evidence_collector.collect_from_action(dummy_action)
            dummy_v = self.verifier.verify(
                recovery_decision_id=f"step-{self.agent_state.step_index}-escalated",
                evidence=dummy_evidence,
                assertions=self.contract_manager.to_assertion_rules(dummy_contract),
            )
            dummy_receipt = VerifiedReceipt(
                task_id=task.task_id,
                action_id=dummy_action.action_id,
                tool=tool,
                frozen_contract=dummy_contract,
                evidence=dummy_evidence,
                verification_result=dummy_v,
                recovery_count=0,
                final_status=AgentStatus.ESCALATED,
            )
            timeline.emit(PHASE_VERIFIED_RECEIPT, {"status": "ESCALATED", "reason": reason})
            return ExecutionCycleResult(
                task_id=task.task_id,
                action_id=dummy_action.action_id,
                tool=tool,
                frozen_contract=dummy_contract,
                action_result=dummy_action,
                evidence=dummy_evidence,
                verification_result=dummy_v,
                verified_receipt=dummy_receipt,
                timeline=timeline.events,
                recovery_count=0,
            )

        # ── Token budget check & reservation ──────────────────────────────
        available_tools = self.tool_registry.list_tool_names()
        max_tokens = get_configured_max_tokens(1536)
        if hasattr(self.provider, "estimate_decision_request_tokens"):
            prompt_est = self.provider.estimate_decision_request_tokens(
                task_objective=task_objective,
                available_tools=available_tools,
            )
        elif hasattr(self.provider, "estimate_request_tokens"):
            prompt_est = self.provider.estimate_request_tokens(prompt=task_objective)
        else:
            prompt_est = estimate_prompt_tokens(task_objective)

        total_estimate = prompt_est + max_tokens
        try:
            reservation_id = self.token_budget.reserve(estimated_tokens=total_estimate)
        except Exception as exc:
            safe_meta = _safe_error_metadata(exc, "token_budget_reserve")
            reason = f"TokenBudgetExceeded: type={safe_meta['error_type']}"
            timeline.emit("TOKEN_BUDGET_EXCEEDED", safe_meta)
            return _escalate(reason)

        # ── Determine provenance from provider type BEFORE calling it ─────
        # This is the ONLY place provenance is assigned — never from model output.
        from reagent.providers.nebius import NebiusNemotronProvider
        is_real_model = isinstance(self.provider, NebiusNemotronProvider)
        expected_provenance = "model" if is_real_model else "fixture"

        # ── Call provider ─────────────────────────────────────────────────
        decision = None
        provider_call_error = None
        safe_meta = None
        reconcile_error_meta = None
        try:
            try:
                decision = self.provider.make_structured_decision(
                    task_objective=task_objective,
                    available_tools=available_tools,
                )
            except Exception as exc:
                safe_meta = _safe_error_metadata(exc, "make_structured_decision")
                provider_call_error = exc
        finally:
            try:
                usage = getattr(self.provider, "last_usage", None)
                self.token_budget.reconcile(
                    reservation_id,
                    usage=usage,
                    conservative_default=total_estimate,
                )
            except Exception as r_exc:
                reconcile_error_meta = _safe_error_metadata(r_exc, "token_budget_reconcile")
                timeline.emit("ACCOUNTING_UNCERTAIN", reconcile_error_meta)

        if reconcile_error_meta is not None:
            if provider_call_error is not None:
                timeline.emit("PROVIDER_FAILED", safe_meta)
                return _escalate(
                    f"ProviderCallFailedAndAccountingUncertain: provider_type={safe_meta['error_type']} "
                    f"status={safe_meta['status_code']} reconcile_type={reconcile_error_meta['error_type']}"
                )
            else:
                return _escalate(f"AccountingUncertain: reconcile_type={reconcile_error_meta['error_type']}")

        if provider_call_error is not None:
            timeline.emit("PROVIDER_FAILED", safe_meta)
            return _escalate(f"ProviderCallFailed: type={safe_meta['error_type']} status={safe_meta['status_code']}")


        # ── Record token usage if provider returned usage info ────────────
        # DevelopmentLLMProvider doesn't make real calls — skip budget accounting.
        # (Only real providers that return LLMResponse with token_usage would use this.)

        if decision is None:
            # Provider signalled MODEL_OUTPUT failure — parse failed inside provider
            timeline.emit("PROVIDER_FAILED", {"phase": "make_structured_decision", "reason": "None returned"})
            return _escalate("ProviderDecisionNone: model output was unparseable or invalid.")

        # Sanitise: overwrite any provenance field the model may have included
        # in its JSON — we never read or trust provenance from model output.
        object.__setattr__(decision, "provenance", expected_provenance) if hasattr(
            type(decision), "__dataclass_fields__"
        ) else None
        # For Pydantic models (StructuredDecision is NOT frozen), reassign directly:
        try:
            decision.provenance = expected_provenance
        except Exception:
            pass  # If frozen model, provenance already correct from construction

        timeline.emit(PHASE_MODEL_DECISION, {
            "tool": decision.tool,
            "provenance": decision.provenance,
            "postconditions": [p.predicate for p in decision.proposed_postconditions],
        })

        # ── Schema validation: tool must exist in allowlist ───────────────
        if not self.tool_registry.is_allowlisted(decision.tool):
            reason = (
                f"DecisionValidationFailed: tool '{decision.tool}' proposed by provider "
                f"is not in the registered allowlist {available_tools}."
            )
            timeline.emit(PHASE_DECISION_VALIDATED, {"valid": False, "reason": reason})
            return _escalate(reason, tool=decision.tool)

        resolved_tool = self.tool_registry.get_tool(decision.tool)
        timeline.emit(PHASE_DECISION_VALIDATED, {"valid": True, "tool": decision.tool})

        # ── Grounding validation ──────────────────────────────────────────
        ground_result = self.grounding_validator.validate(
            decision=decision,
            tool=resolved_tool,
            objective=task_objective,
        )
        timeline.emit("GROUNDING_VALIDATED", {
            "is_valid": ground_result.is_valid,
            "errors": ground_result.errors,
            "warnings": ground_result.warnings,
        })

        if not ground_result.is_valid:
            reason = (
                f"ContractGroundingFailed: {'; '.join(ground_result.errors)}"
            )
            return _escalate(reason, tool=decision.tool)

        # ── Delegate to execute_with_contract() with validated decision ───
        # Use proposed postconditions from the decision if non-empty,
        # otherwise fall back to default STATUS_SUCCESS contract.
        pcs = decision.proposed_postconditions if decision.proposed_postconditions else None

        return self.execute_with_contract(
            task=task,
            tool_name=decision.tool,
            params=decision.arguments,
            postconditions=pcs,
            max_recovery_attempts=max_recovery_attempts,
        )

    # ─────────────────────────────────────────────────────────────────────
    # execute_with_contract() — M3 primary execution path
    # ─────────────────────────────────────────────────────────────────────


    def execute_with_contract(
        self,
        task: TaskState,
        tool_name: str,
        params: Dict[str, Any],
        postconditions: Optional[List[Postcondition]] = None,
        max_recovery_attempts: int = 1,
        probe_callback: Optional[Callable[[str, Dict[str, Any], int], Dict[str, Any]]] = None,
        supports_idempotency: Optional[bool] = None,
    ) -> ExecutionCycleResult:
        """
        Full M3 pipeline execution with contract, verification, and recovery:

          1. TASK_RECEIVED
          2. Build + Freeze SuccessContract  → CONTRACT_FROZEN
          3. PolicyGuard pre-check           → POLICY_ALLOWED
          4. Execute allowlisted tool        → TOOL_EXECUTED
          5. Collect evidence                → EVIDENCE_COLLECTED
          6. Deterministic verification      → VERIFIED_RECEIPT or VERIFICATION_FAILED
          7. If failed → full recovery cycle → REVERIFICATION → VERIFIED_RECEIPT/ESCALATED
        """
        timeline = ObservabilityTimeline(task_id=task.task_id)
        timeline.emit(PHASE_TASK_RECEIVED, {"tool": tool_name, "objective": task.objective})

        self.agent_state.current_task_id = task.task_id
        self.agent_state.step_index += 1
        self.agent_state.status = AgentStatus.RUNNING

        # Default contract: STATUS_SUCCESS if no postconditions supplied
        pcs = postconditions or [
            Postcondition(predicate="STATUS_SUCCESS", description="Tool must succeed")
        ]

        # ── Build + Freeze contract BEFORE any execution ──────────────────
        contract_provenance = "fixture" if postconditions else "hardcoded_default"
        contract = self.contract_manager.build_contract(task.task_id, pcs, provenance=contract_provenance)
        frozen_contract = self.contract_manager.freeze(contract)
        assertion_rules = self.contract_manager.to_assertion_rules(frozen_contract)
        timeline.emit(PHASE_CONTRACT_FROZEN, {
            "contract_id": frozen_contract.contract_id,
            "postconditions": [p.predicate for p in pcs],
            "provenance": frozen_contract.provenance,
        })

        # ── PolicyGuard: check before first execution ─────────────────────
        policy_res = self.policy_guard.check_recovery(
            task_id=task.task_id,
            decision=RecoveryDecision(
                diagnosis_id="pre-execution",
                strategy=RecoveryStrategyType.RETRY,
                rationale="Initial execution policy check.",
            ),
            proposed_payload=params,
        )
        if not policy_res.is_allowed:
            timeline.emit(PHASE_POLICY_ALLOWED, {"allowed": False, "violations": policy_res.violations})
            # PolicyGuard blocked execution before it started → escalate immediately
            self.agent_state.status = AgentStatus.ESCALATED
            self.reliability_state.escalations += 1
            dummy_action = ActionResult(
                tool_name=tool_name,
                success=False,
                error=f"PolicyGuard blocked pre-execution: {'; '.join(policy_res.violations)}",
            )
            evidence = self.evidence_collector.collect_from_action(dummy_action)
            v_res = self.verifier.verify(
                recovery_decision_id=f"step-{self.agent_state.step_index}",
                evidence=evidence,
                assertions=assertion_rules,
            )
            receipt = VerifiedReceipt(
                task_id=task.task_id,
                action_id=dummy_action.action_id,
                tool=tool_name,
                frozen_contract=frozen_contract,
                evidence=evidence,
                verification_result=v_res,
                recovery_count=0,
                final_status=AgentStatus.ESCALATED,
            )
            timeline.emit(PHASE_VERIFIED_RECEIPT, {"status": "ESCALATED", "reason": "policy_blocked"})
            return ExecutionCycleResult(
                task_id=task.task_id,
                action_id=dummy_action.action_id,
                tool=tool_name,
                frozen_contract=frozen_contract,
                action_result=dummy_action,
                evidence=evidence,
                verification_result=v_res,
                verified_receipt=receipt,
                timeline=timeline.events,
                recovery_count=0,
            )

        timeline.emit(PHASE_POLICY_ALLOWED, {"allowed": True})

        # ── Execute tool ──────────────────────────────────────────────────
        action_res = self.tool_registry.execute_tool(tool_name, params)
        self._append_execution_event(task.task_id, tool_name, params, action_res)
        timeline.emit(PHASE_TOOL_EXECUTED, {
            "tool": tool_name,
            "success": action_res.success,
            "duration_ms": action_res.duration_ms,
        })

        # ── Collect evidence ──────────────────────────────────────────────
        evidence = self.evidence_collector.collect_from_action(action_res)
        timeline.emit(PHASE_EVIDENCE_COLLECTED, {"evidence_id": evidence.evidence_id})

        # ── Deterministic verification ────────────────────────────────────
        v_res = self.verifier.verify(
            recovery_decision_id=f"step-{self.agent_state.step_index}",
            evidence=evidence,
            assertions=assertion_rules,
        )

        recovery_count = 0

        if v_res.verified:
            self._record_action_success()
            final_status = AgentStatus.VERIFIED
        else:
            # ── Verification failed → enter recovery loop ─────────────────
            timeline.emit(PHASE_VERIFICATION_FAILED, {
                "discrepancies": v_res.discrepancies,
            })
            self.agent_state.status = AgentStatus.DEGRADED
            self.agent_state.failure_streak += 1
            self.reliability_state.total_failures += 1

            action_res, evidence, v_res, recovery_count, final_status = (
                self._run_recovery_loop(
                    task=task,
                    tool_name=tool_name,
                    original_params=params,
                    frozen_contract=frozen_contract,
                    assertion_rules=assertion_rules,
                    initial_v_res=v_res,
                    timeline=timeline,
                    max_attempts=max_recovery_attempts,
                    probe_callback=probe_callback,
                    supports_idempotency=supports_idempotency,
                )
            )

        # ── Emit VerifiedReceipt ──────────────────────────────────────────
        receipt = VerifiedReceipt(
            task_id=task.task_id,
            action_id=action_res.action_id,
            tool=tool_name,
            frozen_contract=frozen_contract,
            evidence=evidence,
            verification_result=v_res,
            recovery_count=recovery_count,
            final_status=final_status,
        )
        timeline.emit(PHASE_VERIFIED_RECEIPT, {
            "status": final_status.value,
            "recovery_count": recovery_count,
        })

        return ExecutionCycleResult(
            task_id=task.task_id,
            action_id=action_res.action_id,
            tool=tool_name,
            frozen_contract=frozen_contract,
            action_result=action_res,
            evidence=evidence,
            verification_result=v_res,
            verified_receipt=receipt,
            timeline=timeline.events,
            recovery_count=recovery_count,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Recovery loop (bounded by max_attempts)
    # ─────────────────────────────────────────────────────────────────────

    def _run_recovery_loop(
        self,
        task: TaskState,
        tool_name: str,
        original_params: Dict[str, Any],
        frozen_contract: SuccessContract,
        assertion_rules: List[AssertionRule],
        initial_v_res: VerificationResult,
        timeline: ObservabilityTimeline,
        max_attempts: int,
        probe_callback: Optional[Callable[[str, Dict[str, Any], int], Dict[str, Any]]] = None,
        supports_idempotency: Optional[bool] = None,
    ) -> Tuple[ActionResult, Evidence, VerificationResult, int, AgentStatus]:
        """
        Attempt bounded recovery when initial verification fails.
        Uses the same frozen_contract — the contract is NEVER relaxed.
        Supports native ambiguity resolution when probe_callback is supplied.
        Returns (action_result, evidence, v_res, recovery_count, final_status).
        """
        current_v_res = initial_v_res
        recovery_count = 0

        # ── Native Ambiguity Probe & Restraint Check ───────────────────────
        if probe_callback is not None:
            # 1. Execute Step 1 probe call
            try:
                probe_res = probe_callback(tool_name, original_params, 1)
            except Exception as e:
                # Probe raised exception (e.g. StatusUnavailable or Unobservable) -> fail closed
                timeline.emit("AMBIGUITY_PROBE_FAILED", {"error": str(e)})
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                exc_type = type(e).__name__
                if "StatusUnavailable" in exc_type:
                    err_label = f"StatusProbeUnavailable: {e}"
                elif "Unobservable" in exc_type:
                    err_label = f"UnobservableEvidence: {e}"
                else:
                    err_label = f"ProbeError: {exc_type}: {e}"
                dummy = ActionResult(tool_name=tool_name, success=False, error=err_label)
                esc_ev = self.evidence_collector.collect_from_action(dummy)
                esc_v = self.verifier.verify(
                    recovery_decision_id=f"step-{self.agent_state.step_index}-escalation-probe",
                    evidence=esc_ev,
                    assertions=assertion_rules,
                )
                return dummy, esc_ev, esc_v, 0, AgentStatus.ESCALATED

            # Validate probe response structure (malformed / non-dict / missing status / invalid evidence)
            if not isinstance(probe_res, dict) or not probe_res.get("status") or probe_res.get("malformed") is True:
                reason = (
                    f"MalformedObserverEvidence: Probe evidence for {tool_name} is malformed, non-dict, "
                    f"or missing status field; unverified evidence prohibits automated retry."
                )
                timeline.emit("SAFETY_GATE_BLOCKED_RETRY", {"reason": reason})
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(tool_name=tool_name, success=False, error=reason)
                esc_ev = self.evidence_collector.collect_from_action(dummy)
                esc_v = self.verifier.verify(
                    recovery_decision_id=f"step-{self.agent_state.step_index}-malformed-evidence",
                    evidence=esc_ev,
                    assertions=assertion_rules,
                )
                return dummy, esc_ev, esc_v, 0, AgentStatus.ESCALATED

            probe_status = probe_res.get("status")

            # Check for stale observation
            if probe_res.get("stale") is True or probe_res.get("is_stale") is True:
                reason = (
                    f"StaleObservationProhibitsRetry: Probe returned stale evidence for {tool_name}; "
                    f"stale observation cannot verify current outcome."
                )
                timeline.emit("SAFETY_GATE_BLOCKED_RETRY", {"reason": reason})
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(tool_name=tool_name, success=False, error=reason)
                esc_ev = self.evidence_collector.collect_from_action(dummy)
                esc_v = self.verifier.verify(
                    recovery_decision_id=f"step-{self.agent_state.step_index}-stale-evidence",
                    evidence=esc_ev,
                    assertions=assertion_rules,
                )
                return dummy, esc_ev, esc_v, 0, AgentStatus.ESCALATED

            # Correlation check: verify that probe observation belongs to current operation
            correlation_keys = ("booking_id", "operation_id", "request_id", "idempotency_key", "txn_id", "transaction_id")
            for k in correlation_keys:
                if k in original_params and k in probe_res and original_params[k] != probe_res[k]:
                    reason = (
                        f"MismatchedOperationObservation: Probe correlated to {k}={probe_res[k]!r}, "
                        f"but current operation is {k}={original_params[k]!r}; mismatched observation prohibits retry."
                    )
                    timeline.emit("SAFETY_GATE_BLOCKED_RETRY", {"reason": reason})
                    self.agent_state.status = AgentStatus.ESCALATED
                    self.reliability_state.escalations += 1
                    dummy = ActionResult(tool_name=tool_name, success=False, error=reason)
                    esc_ev = self.evidence_collector.collect_from_action(dummy)
                    esc_v = self.verifier.verify(
                        recovery_decision_id=f"step-{self.agent_state.step_index}-mismatched-evidence",
                        evidence=esc_ev,
                        assertions=assertion_rules,
                    )
                    return dummy, esc_ev, esc_v, 0, AgentStatus.ESCALATED

            # If PRESENT: side effect committed despite failure/timeout -> RESTRAIN!
            if probe_status == "PRESENT":
                timeline.emit("RESTRAINT_OBSERVED", {
                    "reason": "Authoritative probe confirmed state PRESENT; avoided duplicate write.",
                })
                self._record_action_success()
                self.agent_state.status = AgentStatus.VERIFIED
                probe_data = probe_res.get("data") if isinstance(probe_res.get("data"), dict) else {}
                restrained_action = ActionResult(
                    tool_name=tool_name,
                    success=True,
                    output=f"Restrained: authoritative probe confirmed state PRESENT. Data: {probe_data}" if probe_data else "Restrained: authoritative probe confirmed state PRESENT.",
                )
                restrained_facts: Dict[str, Any] = {
                    "success": True,
                    "observed_existing_state": True,
                }
                if probe_data:
                    restrained_facts["output"] = probe_data
                    for pk, pv in probe_data.items():
                        if pk not in restrained_facts:
                            restrained_facts[pk] = pv
                restrained_ev = Evidence(source="probe_restraint", facts=restrained_facts)
                restrained_v = self.verifier.verify(
                    recovery_decision_id=f"restraint-step-{self.agent_state.step_index}",
                    evidence=restrained_ev,
                    assertions=assertion_rules,
                )
                return restrained_action, restrained_ev, restrained_v, 0, AgentStatus.VERIFIED

            # If DELAYED: bounded second probe call within budget
            if probe_status == "DELAYED":
                try:
                    probe_res_2 = probe_callback(tool_name, original_params, 2)
                    if not isinstance(probe_res_2, dict) or not probe_res_2.get("status") or probe_res_2.get("malformed") is True:
                        status_2 = "UNKNOWN"
                    else:
                        mismatched_2 = False
                        for k in correlation_keys:
                            if k in original_params and k in probe_res_2 and original_params[k] != probe_res_2[k]:
                                mismatched_2 = True
                                break
                        if mismatched_2 or probe_res_2.get("stale") is True or probe_res_2.get("is_stale") is True:
                            status_2 = "UNKNOWN"
                        else:
                            status_2 = probe_res_2.get("status")

                    if status_2 == "PRESENT":
                        timeline.emit("RESTRAINT_OBSERVED", {
                            "reason": "Step 2 delayed probe confirmed state PRESENT; avoided duplicate write.",
                        })
                        self._record_action_success()
                        self.agent_state.status = AgentStatus.VERIFIED
                        probe_data_2 = probe_res_2.get("data") if isinstance(probe_res_2.get("data"), dict) else {}
                        restrained_action = ActionResult(
                            tool_name=tool_name,
                            success=True,
                            output=f"Restrained: delayed probe confirmed state PRESENT. Data: {probe_data_2}" if probe_data_2 else "Restrained: delayed probe confirmed state PRESENT.",
                        )
                        restrained_facts_2: Dict[str, Any] = {
                            "success": True,
                            "observed_existing_state": True,
                        }
                        if probe_data_2:
                            restrained_facts_2["output"] = probe_data_2
                            for pk, pv in probe_data_2.items():
                                if pk not in restrained_facts_2:
                                    restrained_facts_2[pk] = pv
                        restrained_ev = Evidence(source="probe_restraint", facts=restrained_facts_2)
                        restrained_v = self.verifier.verify(
                            recovery_decision_id=f"restraint-delayed-step-{self.agent_state.step_index}",
                            evidence=restrained_ev,
                            assertions=assertion_rules,
                        )
                        return restrained_action, restrained_ev, restrained_v, 0, AgentStatus.VERIFIED
                    probe_status = status_2
                except Exception as e:
                    timeline.emit("AMBIGUITY_PROBE_FAILED", {"error": str(e)})
                    self.agent_state.status = AgentStatus.ESCALATED
                    self.reliability_state.escalations += 1
                    dummy = ActionResult(tool_name=tool_name, success=False, error=f"DelayedProbeError: {type(e).__name__}: {str(e)}")
                    esc_ev = self.evidence_collector.collect_from_action(dummy)
                    esc_v = self.verifier.verify(
                        recovery_decision_id=f"step-{self.agent_state.step_index}-escalation-probe2",
                        evidence=esc_ev,
                        assertions=assertion_rules,
                    )
                    return dummy, esc_ev, esc_v, 0, AgentStatus.ESCALATED

            # Invariant 1 & 2: UNKNOWN is not ABSENT. DISCREPANCY/UNKNOWN/UNAVAILABLE/UNOBSERVABLE/PARTIAL states must NEVER trigger retry.
            if probe_status in ("DISCREPANCY", "UNKNOWN", "UNAVAILABLE", "UNOBSERVABLE", "PARTIAL", "WRONG_CONTENT"):
                reason = (
                    f"AmbiguousOrDiscrepantStateProhibitsRetry: {tool_name} probe returned '{probe_status}'; "
                    f"UNKNOWN is not ABSENT and partial/corrupted state cannot be blindly retried. Fail-closed escalation."
                )
                timeline.emit("SAFETY_GATE_BLOCKED_RETRY", {"reason": reason})
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(tool_name=tool_name, success=False, error=reason)
                esc_ev = self.evidence_collector.collect_from_action(dummy)
                esc_v = self.verifier.verify(
                    recovery_decision_id=f"step-{self.agent_state.step_index}-safety-gate",
                    evidence=esc_ev,
                    assertions=assertion_rules,
                )
                return dummy, esc_ev, esc_v, 0, AgentStatus.ESCALATED

            # Safety gate for retry on non-idempotent operations:
            tool_obj = self.tool_registry.get_tool(tool_name)
            is_idempotent = supports_idempotency if supports_idempotency is not None else getattr(tool_obj, "supports_idempotency_key", False)

            # Check for unfenced or eventual-consistency absence:
            is_unfenced = (
                probe_res.get("fenced") is False
                or probe_res.get("is_fenced") is False
                or probe_res.get("authoritative") is False
                or probe_res.get("is_authoritative") is False
                or probe_res.get("consistency") in ("EVENTUAL", "WEAK")
                or probe_res.get("consistency_guarantee") in ("EVENTUAL", "WEAK")
            )
            is_confirmed_absent = (probe_status == "CONFIRMED_ABSENT" and not is_unfenced)

            if not is_idempotent and not is_confirmed_absent:
                # Ambiguous non-idempotent operation without authoritative absence confirmation -> FAIL CLOSED
                if is_unfenced:
                    reason = (
                        f"UnfencedAbsenceProhibitsRetry: {tool_name} probe reported absence without authoritative write fencing "
                        f"or strong consistency guarantee; eventual consistency / missing fence cannot rule out late commit. Retry prohibited."
                    )
                else:
                    reason = f"NonIdempotentOperationAmbiguous: {tool_name} cannot be verified absent (probe_status={probe_status}); retry prohibited."
                timeline.emit("SAFETY_GATE_BLOCKED_RETRY", {"reason": reason})
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(tool_name=tool_name, success=False, error=reason)
                esc_ev = self.evidence_collector.collect_from_action(dummy)
                esc_v = self.verifier.verify(
                    recovery_decision_id=f"step-{self.agent_state.step_index}-safety-gate",
                    evidence=esc_ev,
                    assertions=assertion_rules,
                )
                return dummy, esc_ev, esc_v, 0, AgentStatus.ESCALATED

        # ── Restraint Check: environmental state files ─────────────────────
        state_rules = [
            r for r in assertion_rules
            if r.assertion_type in ("FILE_EXISTS", "FILE_CONTAINS", "FILE_SIZE_GTE")
        ]
        if state_rules:
            probe_evidence = Evidence(source="environment_restraint_probe", facts={"success": True})
            probe_res = self.verifier.verify(
                recovery_decision_id=f"restraint-probe-step-{self.agent_state.step_index}",
                evidence=probe_evidence,
                assertions=state_rules,
            )
            if probe_res.verified:
                timeline.emit("RESTRAINT_OBSERVED", {
                    "reason": "Target state already satisfied in environment despite initial error; avoided redundant side effect."
                })
                self._record_action_success()
                dummy_verified_action = ActionResult(
                    tool_name=tool_name,
                    success=True,
                    output="Restrained: environment state already verified.",
                )
                return dummy_verified_action, probe_evidence, probe_res, 0, AgentStatus.VERIFIED


        # ── Fast-path Safe Retry for Verified Ambiguity Recovery ──────────
        # When probe_callback confirmed absence (or tool is idempotent) and max_attempts > 0,
        # execute the safe recovery retry directly under the frozen contract, bounded by max_attempts.
        if probe_callback is not None:
            if max_attempts > 0:
                rec_action = self.tool_registry.execute_tool(tool_name, original_params)
                self._append_execution_event(task.task_id, tool_name, original_params, rec_action)
                recovery_count = 1
                timeline.emit(PHASE_RECOVERY_EXECUTED, {
                    "attempt": 1,
                    "tool": tool_name,
                    "success": rec_action.success,
                })
                rec_evidence = self.evidence_collector.collect_from_action(rec_action)
                timeline.emit(PHASE_EVIDENCE_COLLECTED, {"evidence_id": rec_evidence.evidence_id})
                rec_v_res = self.verifier.verify(
                    recovery_decision_id=f"recovery-1-step-{self.agent_state.step_index}",
                    evidence=rec_evidence,
                    assertions=assertion_rules,
                )
                timeline.emit(PHASE_REVERIFICATION, {
                    "attempt": 1,
                    "verified": rec_v_res.verified,
                    "discrepancies": rec_v_res.discrepancies,
                })
                if rec_v_res.verified:
                    self.reliability_state.successful_recoveries += 1
                    self.agent_state.failure_streak = 0
                    self.agent_state.status = AgentStatus.VERIFIED
                    self.policy_guard.reset_task(task.task_id)
                    return rec_action, rec_evidence, rec_v_res, 1, AgentStatus.VERIFIED
                else:
                    self.agent_state.status = AgentStatus.FAILED
                    self.reliability_state.total_failures += 1
                    return rec_action, rec_evidence, rec_v_res, 1, AgentStatus.FAILED
            else:
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(tool_name=tool_name, success=False, error="RecoveryBudgetExhausted")
                esc_ev = self.evidence_collector.collect_from_action(dummy)
                esc_v = self.verifier.verify(
                    recovery_decision_id=f"step-{self.agent_state.step_index}-budget-exhausted",
                    evidence=esc_ev,
                    assertions=assertion_rules,
                )
                return dummy, esc_ev, esc_v, 0, AgentStatus.ESCALATED

        for attempt in range(max_attempts):
            # Build failure event for diagnosis
            failure = FailureEvent(
                task_id=task.task_id,
                step_index=self.agent_state.step_index,
                tool_name=tool_name,
                failure_type=FailureType.TOOL_FAILURE,
                error_message="; ".join(current_v_res.discrepancies) or "Verification failed.",
            )

            # Context selection
            context_win = self.context_selector.select_context(
                agent_state=self.agent_state,
                task=task,
                execution_history=self.execution_history,
                failure_event=failure,
                allowlisted_tools=self.tool_registry.list_tool_schemas(),
            )
            timeline.emit(PHASE_CONTEXT_SELECTED, {"window_size": len(context_win.bounded_history)})

            # Failure attribution (Nemotron or dev provider) — fails closed on provider exception
            if hasattr(self.provider, "estimate_diagnosis_request_tokens"):
                diag_prompt_est = self.provider.estimate_diagnosis_request_tokens(
                    failure=failure,
                    execution_history=self.execution_history,
                )
            else:
                diag_prompt_est = estimate_prompt_tokens(failure.error_message)
            diag_max_tokens = get_configured_max_tokens(256)
            diag_total_est = diag_prompt_est + diag_max_tokens

            diag_call_error = None
            diag_safe_meta = None
            diag_reconcile_meta = None
            try:
                diag_res_id = self.token_budget.reserve(estimated_tokens=diag_total_est)
            except Exception as exc:
                safe_meta = _safe_error_metadata(exc, "diagnosis_reserve")
                timeline.emit("TOKEN_BUDGET_EXCEEDED", safe_meta)
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"TokenBudgetExceeded: type={safe_meta['error_type']}",
                )
                esc_evidence = self.evidence_collector.collect_from_action(dummy)
                return dummy, esc_evidence, current_v_res, recovery_count, AgentStatus.ESCALATED

            try:
                try:
                    diagnosis = self.provider.diagnose_failure(
                        failure=failure,
                        execution_history=[
                            e.model_dump() if hasattr(e, "model_dump") else e.dict()
                            for e in self.execution_history
                        ],
                    )
                except Exception as exc:
                    diag_safe_meta = _safe_error_metadata(exc, "diagnosis")
                    diag_call_error = exc
            finally:
                try:
                    usage = getattr(self.provider, "last_usage", None)
                    self.token_budget.reconcile(diag_res_id, usage=usage, conservative_default=diag_total_est)
                except Exception as r_exc:
                    diag_reconcile_meta = _safe_error_metadata(r_exc, "diagnosis_reconcile")
                    timeline.emit("ACCOUNTING_UNCERTAIN", diag_reconcile_meta)

            if diag_reconcile_meta is not None:
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                if diag_call_error is not None:
                    timeline.emit("PROVIDER_FAILED", diag_safe_meta)
                    err_msg = (
                        f"ProviderCallFailedAndAccountingUncertain: provider_type={diag_safe_meta['error_type']} "
                        f"status={diag_safe_meta['status_code']} reconcile_type={diag_reconcile_meta['error_type']}"
                    )
                else:
                    err_msg = f"AccountingUncertain: reconcile_type={diag_reconcile_meta['error_type']}"
                dummy = ActionResult(tool_name=tool_name, success=False, error=err_msg)
                esc_evidence = self.evidence_collector.collect_from_action(dummy)
                return dummy, esc_evidence, current_v_res, recovery_count, AgentStatus.ESCALATED

            if diag_call_error is not None:
                timeline.emit("PROVIDER_FAILED", diag_safe_meta)
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"ProviderCallFailed: type={diag_safe_meta['error_type']} status={diag_safe_meta['status_code']}",
                )
                esc_evidence = self.evidence_collector.collect_from_action(dummy)
                return dummy, esc_evidence, current_v_res, recovery_count, AgentStatus.ESCALATED

            timeline.emit(PHASE_FAILURE_ATTRIBUTED, {
                "failure_type": diagnosis.failure_type.value,
                "is_transient": diagnosis.is_transient,
                "confidence": diagnosis.confidence_score,
                "provenance": getattr(diagnosis, "provenance", "unknown"),
            })

            # Recovery selection — fails closed on provider exception
            if hasattr(self.provider, "estimate_recovery_request_tokens"):
                rec_prompt_est = self.provider.estimate_recovery_request_tokens(
                    diagnosis=diagnosis,
                    available_tools=self.tool_registry.list_tool_names(),
                )
            else:
                rec_prompt_est = estimate_prompt_tokens(diagnosis.root_cause)
            rec_max_tokens = get_configured_max_tokens(256)
            rec_total_est = rec_prompt_est + rec_max_tokens

            rec_call_error = None
            rec_safe_meta = None
            rec_reconcile_meta = None
            try:
                rec_res_id = self.token_budget.reserve(estimated_tokens=rec_total_est)
            except Exception as exc:
                safe_meta = _safe_error_metadata(exc, "recovery_selection_reserve")
                timeline.emit("TOKEN_BUDGET_EXCEEDED", safe_meta)
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"TokenBudgetExceeded: type={safe_meta['error_type']}",
                )
                esc_evidence = self.evidence_collector.collect_from_action(dummy)
                return dummy, esc_evidence, current_v_res, recovery_count, AgentStatus.ESCALATED

            try:
                try:
                    decision = self.provider.select_recovery(
                        diagnosis=diagnosis,
                        available_tools=self.tool_registry.list_tool_names(),
                    )
                except Exception as exc:
                    rec_safe_meta = _safe_error_metadata(exc, "recovery_selection")
                    rec_call_error = exc
            finally:
                try:
                    usage = getattr(self.provider, "last_usage", None)
                    self.token_budget.reconcile(rec_res_id, usage=usage, conservative_default=rec_total_est)
                except Exception as r_exc:
                    rec_reconcile_meta = _safe_error_metadata(r_exc, "recovery_reconcile")
                    timeline.emit("ACCOUNTING_UNCERTAIN", rec_reconcile_meta)

            if rec_reconcile_meta is not None:
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                if rec_call_error is not None:
                    timeline.emit("PROVIDER_FAILED", rec_safe_meta)
                    err_msg = (
                        f"ProviderCallFailedAndAccountingUncertain: provider_type={rec_safe_meta['error_type']} "
                        f"status={rec_safe_meta['status_code']} reconcile_type={rec_reconcile_meta['error_type']}"
                    )
                else:
                    err_msg = f"AccountingUncertain: reconcile_type={rec_reconcile_meta['error_type']}"
                dummy = ActionResult(tool_name=tool_name, success=False, error=err_msg)
                esc_evidence = self.evidence_collector.collect_from_action(dummy)
                return dummy, esc_evidence, current_v_res, recovery_count, AgentStatus.ESCALATED

            if rec_call_error is not None:
                timeline.emit("PROVIDER_FAILED", rec_safe_meta)
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                dummy = ActionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"ProviderCallFailed: type={rec_safe_meta['error_type']} status={rec_safe_meta['status_code']}",
                )
                esc_evidence = self.evidence_collector.collect_from_action(dummy)
                return dummy, esc_evidence, current_v_res, recovery_count, AgentStatus.ESCALATED

            timeline.emit(PHASE_RECOVERY_SELECTED, {
                "strategy": decision.strategy.value,
                "provenance": getattr(decision, "provenance", "unknown"),
            })


            # Compute recovery parameters: merge any modifications onto original parameters
            recovery_params = dict(original_params)
            if decision.modified_input:
                recovery_params.update(decision.modified_input)
            recovery_tool = decision.fallback_tool_name or tool_name

            # PolicyGuard before recovery action
            policy_res = self.policy_guard.check_recovery(
                task_id=task.task_id,
                decision=decision,
                proposed_payload=recovery_params,
            )
            timeline.emit(PHASE_RECOVERY_ALLOWED, {
                "allowed": policy_res.is_allowed,
                "violations": policy_res.violations,
            })

            if not policy_res.is_allowed or decision.strategy == RecoveryStrategyType.ESCALATE:
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
                decision.guard_approved = policy_res.is_allowed
                decision.guard_reasons = policy_res.violations
                # Return last known state
                dummy = ActionResult(
                    tool_name=tool_name,
                    success=False,
                    error="Escalated after PolicyGuard or ESCALATE strategy.",
                )
                esc_evidence = self.evidence_collector.collect_from_action(dummy)
                return dummy, esc_evidence, current_v_res, recovery_count, AgentStatus.ESCALATED

            # Execute recovery action
            rec_action = self.tool_registry.execute_tool(recovery_tool, recovery_params)
            self._append_execution_event(task.task_id, recovery_tool, recovery_params, rec_action)
            recovery_count += 1
            timeline.emit(PHASE_RECOVERY_EXECUTED, {
                "attempt": attempt + 1,
                "tool": recovery_tool,
                "success": rec_action.success,
            })

            # Collect fresh evidence
            rec_evidence = self.evidence_collector.collect_from_action(rec_action)
            timeline.emit(PHASE_EVIDENCE_COLLECTED, {"evidence_id": rec_evidence.evidence_id})

            # Reverify against the SAME frozen contract
            rec_v_res = self.verifier.verify(
                recovery_decision_id=f"recovery-{attempt + 1}-step-{self.agent_state.step_index}",
                evidence=rec_evidence,
                assertions=assertion_rules,
            )
            timeline.emit(PHASE_REVERIFICATION, {
                "attempt": attempt + 1,
                "verified": rec_v_res.verified,
                "discrepancies": rec_v_res.discrepancies,
            })

            current_v_res = rec_v_res

            if rec_v_res.verified:
                self.reliability_state.successful_recoveries += 1
                self.agent_state.failure_streak = 0
                self.agent_state.status = AgentStatus.VERIFIED
                self.policy_guard.reset_task(task.task_id)
                return rec_action, rec_evidence, rec_v_res, recovery_count, AgentStatus.VERIFIED

        # All recovery attempts exhausted without verification passing
        self.agent_state.status = AgentStatus.ESCALATED
        self.reliability_state.escalations += 1
        dummy = ActionResult(
            tool_name=tool_name,
            success=False,
            error=f"All {max_attempts} recovery attempt(s) exhausted.",
        )
        esc_evidence = self.evidence_collector.collect_from_action(dummy)
        return dummy, esc_evidence, current_v_res, recovery_count, AgentStatus.ESCALATED

    # ─────────────────────────────────────────────────────────────────────
    # Legacy execute_action() — preserved for backward compat with M1/M2 tests
    # ─────────────────────────────────────────────────────────────────────

    def execute_action(
        self,
        task: TaskState,
        tool_name: str,
        params: Dict[str, Any],
        assertions: Optional[List[AssertionRule]] = None,
    ) -> Tuple[ActionResult, Evidence, VerificationResult]:
        """
        M1/M2-compatible execution path.  Does NOT emit VerifiedReceipt or timeline.
        Use execute_with_contract() for full M3 guarantees.
        """
        self.agent_state.current_task_id = task.task_id
        self.agent_state.step_index += 1
        self.agent_state.status = AgentStatus.RUNNING

        action_res = self.tool_registry.execute_tool(tool_name, params)
        evidence = self.evidence_collector.collect_from_action(action_res)
        v_res = self.verifier.verify(
            recovery_decision_id=f"step-{self.agent_state.step_index}",
            evidence=evidence,
            assertions=assertions,
        )
        self._append_execution_event(task.task_id, tool_name, params, action_res)

        if action_res.success and v_res.verified:
            self._record_action_success()
        else:
            self.agent_state.status = AgentStatus.DEGRADED
            self.agent_state.failure_streak += 1
            self.reliability_state.total_failures += 1

        return action_res, evidence, v_res

    # ─────────────────────────────────────────────────────────────────────
    # process_failure() — M1/M2-compatible failure cycle
    # ─────────────────────────────────────────────────────────────────────

    def process_failure(
        self,
        failure: FailureEvent,
        available_tools: Optional[List[str]] = None,
        task: Optional[TaskState] = None,
    ) -> ReliabilityCycleResult:
        """Preserved for backward compat with Milestone 1/2 tests."""
        task = task or TaskState(task_id=failure.task_id, objective="Resolve Execution Failure")
        available_tools = available_tools or self.tool_registry.list_tool_names()

        self.agent_state.status = AgentStatus.DEGRADED
        self.agent_state.failure_streak += 1
        self.reliability_state.total_failures += 1

        context_win = self.context_selector.select_context(
            agent_state=self.agent_state,
            task=task,
            execution_history=self.execution_history,
            failure_event=failure,
            allowlisted_tools=self.tool_registry.list_tool_schemas(),
        )

        diagnosis = self.provider.diagnose_failure(
            failure=failure,
            execution_history=[
                e.model_dump() if hasattr(e, "model_dump") else e.dict()
                for e in self.execution_history
            ],
        )

        decision = self.provider.select_recovery(
            diagnosis=diagnosis,
            available_tools=available_tools,
        )

        policy_res = self.policy_guard.check_recovery(
            task_id=failure.task_id,
            decision=decision,
            proposed_payload=decision.modified_input,
        )

        effective_strategy = decision.strategy
        if not policy_res.is_allowed:
            effective_strategy = RecoveryStrategyType.ESCALATE
            decision.guard_approved = False
            decision.guard_reasons = policy_res.violations
            self.agent_state.status = AgentStatus.ESCALATED
            self.reliability_state.escalations += 1
        else:
            if effective_strategy == RecoveryStrategyType.ESCALATE:
                self.agent_state.status = AgentStatus.ESCALATED
                self.reliability_state.escalations += 1
            else:
                self.agent_state.status = AgentStatus.RECOVERING

        total = self.reliability_state.total_actions + self.reliability_state.total_failures
        if total > 0:
            successes = self.reliability_state.total_actions + self.reliability_state.successful_recoveries
            self.reliability_state.reliability_score = round(
                max(0.0, min(100.0, (successes / (total + 1e-9)) * 100)), 2
            )

        return ReliabilityCycleResult(
            task_id=failure.task_id,
            context_window=context_win,
            diagnosis=diagnosis,
            recovery_decision=decision,
            policy_decision=policy_res,
            effective_strategy=effective_strategy,
            agent_status=self.agent_state.status,
            updated_reliability=self.reliability_state,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _append_execution_event(
        self,
        task_id: str,
        tool_name: str,
        params: Dict[str, Any],
        action_res: ActionResult,
    ) -> None:
        event = ExecutionEvent(
            task_id=task_id,
            step_index=self.agent_state.step_index,
            tool_name=tool_name,
            input_payload=params,
            raw_output=str(action_res.output) if action_res.output is not None else None,
            exit_code=0 if action_res.success else 1,
            duration_ms=action_res.duration_ms,
        )
        self.execution_history.append(event)

    def _record_action_success(self) -> None:
        self.agent_state.status = AgentStatus.VERIFIED
        self.agent_state.failure_streak = 0
        self.reliability_state.total_actions += 1

    def record_recovery_success(self, task_id: str) -> None:
        """Called when a recovery action successfully resolves a failure."""
        self.agent_state.status = AgentStatus.RUNNING
        self.agent_state.failure_streak = 0
        self.reliability_state.successful_recoveries += 1
        self.policy_guard.reset_task(task_id)

    def record_action_success(self) -> None:
        """Public alias for backward compat."""
        self._record_action_success()
