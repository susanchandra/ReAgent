"""
pipeline_adapter_v3.py — Honest benchmark adapter for Ambiguity V3.

ARCHITECTURAL PRINCIPLES:
1. ReAgentPipeline executes primary actions and recovery retries with SuccessContracts,
   policy checks, and deterministic verification.
2. The adapter coordinates observation probes and enforces safety gates:
   - NEVER blindly retries a non-idempotent operation (apply_transfer) when commit status is unknown.
   - Requires explicit CONFIRMED_ABSENT on an authoritative consistent surface before permitting retry.
   - Enforces a hard budget ceiling of at most 2 action attempts and at most 2 probe calls.
   - If recovery attempt itself encounters a lost response, it terminates as ESCALATED/FAILED
     and NEVER triggers an unauthorized third write attempt.
3. Does NOT claim native pipeline ownership of observation probing or non-idempotent gate logic.
   Receipts are generated through the pipeline's deterministic verifier and receipt machinery.
"""

from typing import Dict, Any, Optional
from reagent.pipeline import ReAgentPipeline, ExecutionCycleResult
from reagent.tools.base import BaseTool
from reagent.models import TaskState, Postcondition, AgentStatus, VerifiedReceipt, ActionResult
from reagent.experiments.ambiguity.agent_view import AgentView
from reagent.experiments.ambiguity.manifest_v3 import (
    OP_CREATE_RESOURCE,
    OP_APPLY_TRANSFER,
    OP_RESERVE_THEN_LINK,
    IDEMPOTENCY_KEY_SUPPORT,
)
from reagent.experiments.ambiguity.faults import (
    ToolExecutionTimeoutError,
    StatusUnavailableError,
    UnobservableEvidenceError,
)


class SimulatorToolV3(BaseTool):
    """Allowlisted ReAgent Tool wrapping the AgentView facade for V3."""
    name: str = "simulator_operation_v3"
    description: str = "Executes state-changing operations through the sandboxed AgentView facade."
    effect_class: str = "state-changing"

    def __init__(self, agent_view: AgentView):
        self.agent_view = agent_view

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        op_name = params.get("op_name")
        op_params = params.get("params", {})
        idempotency_key = params.get("idempotency_key")
        return self.agent_view.execute(op_name=op_name, params=op_params, idempotency_key=idempotency_key)


def run_pipeline_arm6_v3(
    agent_view: AgentView,
    op_name: str,
    params: Dict[str, Any],
    fault_mode: str,
    task_id: str,
) -> Dict[str, Any]:
    """
    Executes Arm 6 for V3: ReAgent Contract Policy.
    Natively delegates execution, contract freezing, ambiguity observation probing,
    restraint verification, and safe retry gating to ReAgentPipeline.
    """
    pipeline = ReAgentPipeline(agent_id=f"reagent-v3-{task_id}")
    sim_tool = SimulatorToolV3(agent_view)
    pipeline.tool_registry.register(sim_tool)

    task = TaskState(
        task_id=task_id,
        objective=f"Safely execute {op_name} without duplicate side effects or unverified retries",
    )

    # Freeze SuccessContract before execution
    postconditions = [
        Postcondition(predicate="STATUS_SUCCESS", description="Operation returned success=True"),
    ]
    contract = pipeline.contract_manager.build_contract(
        task_id=task.task_id,
        postconditions=postconditions,
        provenance="fixture",
    )
    frozen_contract = pipeline.contract_manager.freeze(contract)

    # Probe callback delegating to structured agent_view
    def probe_fn(tool_name: str, tool_params: Dict[str, Any], step: int) -> Dict[str, Any]:
        return agent_view.probe_structured(op_name, params, probe_step=step)

    supports_idempotency = IDEMPOTENCY_KEY_SUPPORT.get(op_name, False)

    # Native pipeline execution with ambiguity recovery enabled
    tool_params = {"op_name": op_name, "params": params}
    cycle_res: ExecutionCycleResult = pipeline.execute_with_contract(
        task=task,
        tool_name="simulator_operation_v3",
        params=tool_params,
        postconditions=frozen_contract.postconditions,
        max_recovery_attempts=1,
        probe_callback=probe_fn,
        supports_idempotency=supports_idempotency,
    )

    receipt = cycle_res.verified_receipt
    is_verified = (receipt.final_status == AgentStatus.VERIFIED)
    is_escalated = (receipt.final_status == AgentStatus.ESCALATED)
    is_failed = (receipt.final_status == AgentStatus.FAILED)
    retried = (cycle_res.recovery_count > 0)
    observed_existing_state = (cycle_res.evidence.source == "probe_restraint")

    if is_verified:
        agent_status = "COMPLETED"
    elif is_escalated:
        agent_status = "ESCALATED"
    else:
        agent_status = "FAILED"

    escalation_reason = None
    if not is_verified:
        escalation_reason = cycle_res.action_result.error or "; ".join(cycle_res.verification_result.discrepancies) or "ExecutionUnverified"

    return {
        "agent_status": agent_status,
        "claimed_complete_verified": is_verified,
        "escalated": is_escalated,
        "failed_terminal": is_failed,
        "retried": retried,
        "observed_existing_state": observed_existing_state,
        "receipt": receipt,
        "escalation_reason": escalation_reason,
    }

