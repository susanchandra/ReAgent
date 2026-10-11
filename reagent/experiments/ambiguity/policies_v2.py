"""
policies_v2.py — Fair baseline and ReAgent policy implementations for Ambiguity V2.

All policies operate strictly through the AgentView facade.
Budgets:
- Observation budget: at most 2 probe calls
- Action budget: 1 initial attempt + 1 bounded retry
"""

from typing import Dict, Any, Optional
from reagent.experiments.ambiguity.agent_view import AgentView
from reagent.experiments.ambiguity.manifest_v2 import (
    POLICY_ALWAYS_ESCALATE,
    POLICY_NEVER_RETRY,
    POLICY_NAIVE_RETRY,
    POLICY_IDEMPOTENCY_KEY,
    POLICY_SHARED_OBSERVATION,
    POLICY_REAGENT_CONTRACT,
    IDEMPOTENCY_KEY_SUPPORT,
    FAULT_UNOBSERVABLE,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_DELAYED_VISIBILITY,
)
from reagent.experiments.ambiguity.faults import ToolExecutionTimeoutError, StatusUnavailableError
from reagent.experiments.ambiguity.pipeline_adapter import run_pipeline_with_contract


class PolicyRunResult:
    def __init__(
        self,
        policy_name: str,
        operation: str,
        agent_status: str,              # "COMPLETED", "ESCALATED", "FAILED"
        claimed_complete_verified: bool,
        claimed_complete_unverified: bool,
        escalated: bool,
        failed_terminal: bool,
        retried: bool,
        observed_existing_state: bool,
        escalation_reason: Optional[str] = None,
    ):
        self.policy_name = policy_name
        self.operation = operation
        self.agent_status = agent_status
        self.claimed_complete_verified = claimed_complete_verified
        self.claimed_complete_unverified = claimed_complete_unverified
        self.escalated = escalated
        self.failed_terminal = failed_terminal
        self.retried = retried
        self.observed_existing_state = observed_existing_state
        self.escalation_reason = escalation_reason


def execute_policy_v2(
    policy_name: str,
    agent_view: AgentView,
    op_name: str,
    params: Dict[str, Any],
    fault_mode: str,
    task_id: str,
) -> PolicyRunResult:
    """Executes a policy arm via AgentView."""

    # 1. ALWAYS_ESCALATE
    if policy_name == POLICY_ALWAYS_ESCALATE:
        try:
            agent_view.execute(op_name, params)
            return PolicyRunResult(policy_name, op_name, "COMPLETED", True, False, False, False, False, False)
        except ToolExecutionTimeoutError:
            return PolicyRunResult(policy_name, op_name, "ESCALATED", False, False, True, False, False, False, "PolicyEscalatedImmediately")

    # 2. NEVER_RETRY
    elif policy_name == POLICY_NEVER_RETRY:
        try:
            agent_view.execute(op_name, params)
            return PolicyRunResult(policy_name, op_name, "COMPLETED", True, False, False, False, False, False)
        except ToolExecutionTimeoutError:
            return PolicyRunResult(policy_name, op_name, "FAILED", False, False, False, True, False, False, "ToolTimeoutNeverRetry")

    # 3. NAIVE_RETRY
    elif policy_name == POLICY_NAIVE_RETRY:
        try:
            agent_view.execute(op_name, params)
            return PolicyRunResult(policy_name, op_name, "COMPLETED", False, True, False, False, False, False)
        except ToolExecutionTimeoutError:
            # Blind retry without probe
            try:
                agent_view.execute(op_name, params)
                return PolicyRunResult(policy_name, op_name, "COMPLETED", False, True, False, False, True, False)
            except Exception as e:
                return PolicyRunResult(policy_name, op_name, "FAILED", False, False, False, True, True, False, str(e))

    # 4. IDEMPOTENCY_KEY
    elif policy_name == POLICY_IDEMPOTENCY_KEY:
        idem_key = f"idem-{params.get('request_id', 'req-001')}"
        try:
            agent_view.execute(op_name, params, idempotency_key=idem_key)
            return PolicyRunResult(policy_name, op_name, "COMPLETED", False, True, False, False, False, False)
        except ToolExecutionTimeoutError:
            try:
                agent_view.execute(op_name, params, idempotency_key=idem_key)
                return PolicyRunResult(policy_name, op_name, "COMPLETED", False, True, False, False, True, False)
            except Exception as e:
                return PolicyRunResult(policy_name, op_name, "FAILED", False, False, False, True, True, False, str(e))

    # 5. SHARED_OBSERVATION_POLICY (Symmetrically fair to ReAgent)
    elif policy_name == POLICY_SHARED_OBSERVATION:
        try:
            agent_view.execute(op_name, params)
            return PolicyRunResult(policy_name, op_name, "COMPLETED", True, False, False, False, False, False)
        except ToolExecutionTimeoutError:
            # Check unobservable constraint
            if fault_mode == FAULT_UNOBSERVABLE:
                return PolicyRunResult(policy_name, op_name, "ESCALATED", False, False, True, False, False, False, "EvidenceUnobservable")

            # Probe step 1
            try:
                state_observed = agent_view.probe(op_name, params, probe_step=1)
            except StatusUnavailableError:
                return PolicyRunResult(policy_name, op_name, "ESCALATED", False, False, True, False, False, False, "StatusUnavailable")

            if state_observed:
                return PolicyRunResult(policy_name, op_name, "COMPLETED", True, False, False, False, False, True)

            # Probe step 2 for delayed visibility
            if fault_mode == FAULT_DELAYED_VISIBILITY:
                try:
                    if agent_view.probe(op_name, params, probe_step=2):
                        return PolicyRunResult(policy_name, op_name, "COMPLETED", True, False, False, False, False, True)
                except Exception:
                    pass

            # Effect confirmed absent -> safe retry
            try:
                agent_view.execute(op_name, params)
                return PolicyRunResult(policy_name, op_name, "COMPLETED", True, False, False, False, True, False)
            except Exception as e:
                return PolicyRunResult(policy_name, op_name, "FAILED", False, False, False, True, True, False, str(e))

    # 6. REAGENT_CONTRACT_POLICY (Production pipeline integration)
    elif policy_name == POLICY_REAGENT_CONTRACT:
        res = run_pipeline_with_contract(agent_view, op_name, params, fault_mode, task_id)
        return PolicyRunResult(
            policy_name=policy_name,
            operation=op_name,
            agent_status=res["agent_status"],
            claimed_complete_verified=res["claimed_complete_verified"],
            claimed_complete_unverified=False,
            escalated=res["escalated"],
            failed_terminal=(res["agent_status"] == "FAILED"),
            retried=res["retried"],
            observed_existing_state=res["observed_existing_state"],
            escalation_reason=res.get("escalation_reason"),
        )

    raise ValueError(f"Unknown policy: {policy_name}")
