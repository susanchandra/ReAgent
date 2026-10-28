"""
reagent/workflow.py — Generic Composite Workflow Engine (Phase 3)

Orchestrates multi-step, ordered workflows where every step has its own frozen contract
and structured tri-state verification result.

Guarantees & Invariants:
1. Ordered Execution: Steps execute in declared order, respecting explicit dependencies.
2. Frozen Contracts: Every step builds and freezes its contract before execution.
3. Completion Preservation: Completed (VERIFIED) steps are NEVER rerun during recovery.
4. Selective Resumption: Recovery/resumption targets only eligible unfinished steps.
5. Safety Gates: UNKNOWN on non-idempotent operations without authoritative fenced probes
   blocks automated retry (Phase 1 invariants enforced).
6. Evidence & History Preservation: Step receipts preserve full attempt history and evidence.
7. Genericity: No capability-specific branching (zero "if hotel", "if wallet", etc.).
8. Explicit Boundaries: Same-run in-memory selective resumption is guaranteed; cross-process
   durable crash recovery requires an external persistent checkpoint store.
"""

from typing import Dict, List, Any, Optional, Callable
import uuid

from reagent.taxonomy import AgentStatus, PostconditionStatus
from reagent.models import (
    CompositeTaskPlan,
    WorkflowStep,
    StepAttempt,
    StepReceipt,
    WorkflowReceipt,
    TaskState,
    Postcondition,
    Evidence,
    ActionResult,
    VerificationResult,
    get_utc_now,
)
from reagent.pipeline import ReAgentPipeline, ExecutionCycleResult
from reagent.contract import ContractManager


class WorkflowRunner:
    """
    Generic orchestration engine for CompositeTaskPlan workflows.
    Delegates single-action execution, contract freezing, policy checking,
    and deterministic verification to ReAgentPipeline.
    """

    def __init__(
        self,
        pipeline: Optional[ReAgentPipeline] = None,
        max_workflow_recovery_budget: int = 5,
    ) -> None:
        """
        Args:
            pipeline: ReAgentPipeline instance for executing individual steps.
                      If None, instantiates a default pipeline.
            max_workflow_recovery_budget: Maximum total recovery attempts allowed across all steps.
        """
        self.pipeline = pipeline or ReAgentPipeline()
        self.max_workflow_recovery_budget = max_workflow_recovery_budget

    def execute_workflow(
        self,
        plan: CompositeTaskPlan,
        existing_receipt: Optional[WorkflowReceipt] = None,
        probe_callbacks: Optional[Dict[str, Callable[[str, Dict[str, Any], int], Dict[str, Any]]]] = None,
    ) -> WorkflowReceipt:
        """
        Execute or resume a CompositeTaskPlan.

        If existing_receipt is provided, selectively resumes:
        - Already VERIFIED steps are preserved and skipped (never rerun).
        - Eligible unfinished or failed steps are evaluated under Phase 1/2 safety rules.
        - Previous attempt history is preserved rather than overwritten.
        """
        # Validate plan integrity (non-empty, unique IDs, valid dependencies)
        plan.validate_plan()

        probe_callbacks = probe_callbacks or {}
        step_receipts: Dict[str, StepReceipt] = {}
        completed_steps: List[str] = []
        incomplete_steps: List[str] = []
        failed_steps: List[str] = []
        unresolved_steps: List[str] = []
        execution_order: List[str] = []
        total_recovery_attempts = 0

        # ── 1. Ingest existing receipt for selective resumption ───────────
        if existing_receipt is not None:
            total_recovery_attempts = existing_receipt.total_recovery_attempts
            for s in plan.steps:
                if s.step_id in existing_receipt.step_receipts:
                    prior_sr = existing_receipt.step_receipts[s.step_id]
                    if prior_sr.status == AgentStatus.VERIFIED:
                        # Completed step: PRESERVE, DO NOT RERUN
                        step_receipts[s.step_id] = prior_sr
                        completed_steps.append(s.step_id)
                        execution_order.append(s.step_id)

        # ── 2. Ordered Step Execution ─────────────────────────────────────
        for step in plan.steps:
            # If already verified from prior receipt, skip execution
            if step.step_id in completed_steps:
                continue

            # Dependency check: prior steps in depends_on must be VERIFIED
            deps_met = True
            for dep in step.depends_on:
                if dep not in completed_steps:
                    deps_met = False
                    break

            if not deps_met:
                # Prior step failed or not completed; cannot proceed with this step
                incomplete_steps.append(step.step_id)
                if step.required:
                    # Halt workflow on required dependency failure
                    break
                continue

            # Check remaining workflow recovery budget
            if total_recovery_attempts >= self.max_workflow_recovery_budget:
                unresolved_steps.append(step.step_id)
                step_receipts[step.step_id] = StepReceipt(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=AgentStatus.RECOVERY_EXHAUSTED,
                    attempts=[],
                    action_result=ActionResult(
                        tool_name=step.tool_name,
                        success=False,
                        error=f"Total workflow recovery budget exhausted ({self.max_workflow_recovery_budget}).",
                    ),
                    recovery_count=0,
                    executed=False,
                )
                execution_order.append(step.step_id)
                break

            # Resolve probe callback for this step (reusing Phase 1 safety gates)
            probe_cb = (
                probe_callbacks.get(step.step_id)
                or probe_callbacks.get(step.tool_name)
            )

            # Check existing attempt history for this step
            prior_attempts: List[StepAttempt] = []
            if existing_receipt and step.step_id in existing_receipt.step_receipts:
                prior_receipt = existing_receipt.step_receipts[step.step_id]
                prior_attempts = list(prior_receipt.attempts)

                # Invariant 3: UNKNOWN on non-idempotent action blocks blind retry
                is_idempotent = step.supports_idempotency
                if is_idempotent is None:
                    tool_obj = self.pipeline.tool_registry.get_tool(step.tool_name)
                    is_idempotent = getattr(tool_obj, "supports_idempotency_key", False) if tool_obj else False

                if prior_receipt.status in (AgentStatus.UNKNOWN, AgentStatus.ESCALATED) and not is_idempotent:
                    if probe_cb is None:
                        # Non-idempotent action in ambiguous state without probe: DO NOT RETRY
                        unresolved_steps.append(step.step_id)
                        step_receipts[step.step_id] = prior_receipt
                        execution_order.append(step.step_id)
                        if step.required:
                            break
                        continue

            # Execute step through pipeline with frozen contract
            step_task = TaskState(
                task_id=f"{plan.task_id}::{step.step_id}",
                objective=step.description or f"Execute step {step.step_id} ({step.tool_name})",
            )
            pcs = step.postconditions or [
                Postcondition(predicate="STATUS_SUCCESS", description=f"Step {step.step_id} must succeed")
            ]

            cycle_res: ExecutionCycleResult = self.pipeline.execute_with_contract(
                task=step_task,
                tool_name=step.tool_name,
                params=step.params,
                postconditions=pcs,
                max_recovery_attempts=step.max_recovery_attempts,
                probe_callback=probe_cb,
                supports_idempotency=step.supports_idempotency,
            )

            total_recovery_attempts += cycle_res.recovery_count

            # Check whether safety gate or probe blocked retry
            safety_gate_blocked = any(
                getattr(e, "phase", None) in ("SAFETY_GATE_BLOCKED_RETRY", "AMBIGUITY_PROBE_FAILED")
                for e in cycle_res.timeline
            )

            # Determine step status from verification result and pipeline receipt
            if cycle_res.verified_receipt.final_status == AgentStatus.VERIFIED or cycle_res.verification_result.aggregate_status == AgentStatus.VERIFIED:
                step_status = AgentStatus.VERIFIED
            elif safety_gate_blocked:
                # Restraint/safety gate blocked automated retry on ambiguity or unfenced absence
                step_status = AgentStatus.ESCALATED
            elif (cycle_res.recovery_count >= step.max_recovery_attempts and step.max_recovery_attempts > 0) or (total_recovery_attempts >= self.max_workflow_recovery_budget):
                step_status = AgentStatus.RECOVERY_EXHAUSTED
            elif cycle_res.verification_result.aggregate_status == AgentStatus.PARTIAL:
                step_status = AgentStatus.PARTIAL
            elif cycle_res.verification_result.aggregate_status == AgentStatus.VERIFIED_FAILURE:
                step_status = AgentStatus.VERIFIED_FAILURE
            elif cycle_res.verified_receipt.final_status == AgentStatus.ESCALATED:
                step_status = AgentStatus.ESCALATED
            else:
                step_status = cycle_res.verification_result.aggregate_status or cycle_res.verified_receipt.final_status



            # Record this attempt while preserving all prior attempts
            new_attempt = StepAttempt(
                attempt_number=len(prior_attempts) + 1,
                action_result=cycle_res.action_result,
                evidence=cycle_res.evidence,
                verification_result=cycle_res.verification_result,
                status=step_status,
                error=cycle_res.action_result.error if not cycle_res.action_result.success else None,
            )
            all_attempts = prior_attempts + [new_attempt]

            step_receipt = StepReceipt(
                step_id=step.step_id,
                tool_name=step.tool_name,
                status=step_status,
                attempts=all_attempts,
                verification_result=cycle_res.verification_result,
                action_result=cycle_res.action_result,
                evidence=cycle_res.evidence,
                recovery_count=cycle_res.recovery_count + len(prior_attempts),
                executed=True,
            )
            step_receipts[step.step_id] = step_receipt
            execution_order.append(step.step_id)

            if step_status == AgentStatus.VERIFIED:
                completed_steps.append(step.step_id)
            elif step_status in (AgentStatus.VERIFIED_FAILURE, AgentStatus.FAILED):
                failed_steps.append(step.step_id)
                if step.required:
                    break
            else:
                unresolved_steps.append(step.step_id)
                if step.required:
                    break

        # ── 3. Populate remaining incomplete steps ────────────────────────
        for s in plan.steps:
            if s.step_id not in step_receipts and s.step_id not in incomplete_steps:
                incomplete_steps.append(s.step_id)

        # ── 4. Aggregate Workflow Outcome ─────────────────────────────────
        overall_status = self._aggregate_workflow_status(
            plan=plan,
            completed_steps=completed_steps,
            incomplete_steps=incomplete_steps,
            failed_steps=failed_steps,
            unresolved_steps=unresolved_steps,
            step_receipts=step_receipts,
        )

        return WorkflowReceipt(
            task_id=plan.task_id,
            plan=plan,
            step_receipts=step_receipts,
            status=overall_status,
            completed_steps=completed_steps,
            incomplete_steps=incomplete_steps,
            failed_steps=failed_steps,
            unresolved_steps=unresolved_steps,
            execution_order=execution_order,
            total_recovery_attempts=total_recovery_attempts,
        )

    def _aggregate_workflow_status(
        self,
        plan: CompositeTaskPlan,
        completed_steps: List[str],
        incomplete_steps: List[str],
        failed_steps: List[str],
        unresolved_steps: List[str],
        step_receipts: Dict[str, StepReceipt],
    ) -> AgentStatus:
        """
        Pure aggregation of overall workflow status from step outcomes.
        """
        total_steps = len(plan.steps)

        # 1. All steps verified
        if len(completed_steps) == total_steps:
            return AgentStatus.VERIFIED

        # 2. Check for recovery exhaustion (Rule 5)
        has_exhausted = any(
            step_receipts[sid].status == AgentStatus.RECOVERY_EXHAUSTED
            for sid in (unresolved_steps + failed_steps) if sid in step_receipts
        )
        if has_exhausted:
            return AgentStatus.RECOVERY_EXHAUSTED

        # 3. Check for any unresolved / unknown steps (Rule 4)
        has_unknown = any(
            step_receipts[sid].status in (AgentStatus.UNKNOWN, AgentStatus.ESCALATED)
            for sid in unresolved_steps if sid in step_receipts
        )
        if has_unknown:
            return AgentStatus.UNKNOWN

        # 4. Mixture of completed and failed/incomplete steps (no undetermined) (Rule 3)
        if completed_steps and (failed_steps or incomplete_steps):
            return AgentStatus.PARTIAL

        # 5. All attempted steps are verified failures (Rule 2)
        if failed_steps and not completed_steps:
            return AgentStatus.VERIFIED_FAILURE

        return AgentStatus.FAILED


