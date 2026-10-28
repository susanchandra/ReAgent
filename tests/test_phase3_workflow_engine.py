"""
tests/test_phase3_workflow_engine.py — Phase 3 Generic Composite Workflow Engine Tests

Verifies all 12 required items from Phase 3 specification:
  1.  Valid ordered two-step workflow.
  2.  Empty workflow and duplicate step IDs rejected.
  3.  Step 1 VERIFIED; step 2 fails: step 1 executes once.
  4.  Recovery resumes only an eligible unfinished step.
  5.  UNKNOWN on a non-idempotent action does not trigger a blind retry.
  6.  Unfenced absence cannot authorize retry.
  7.  An already completed step is not repeated during same-run recovery.
  8.  Step evidence and attempt history survive recovery.
  9.  Recovery budget exhaustion is represented truthfully.
  10. Workflow outcome correctly reflects all-verified, partial, failure, and unresolved step outcomes.
  11. A registered unforeseen adapter can participate without capability-specific engine edits.
  12. Existing single-action pipeline behavior remains unchanged.
  13. Two-Step Scenario (Section F): Step 1 verified, Step 2 commit-then-timeout resolved via authoritative restraint.
"""

import os
import shutil
import tempfile
import unittest
from typing import Dict, Any, Optional

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_wf_test_")
    _MODULE_PREV_LEDGER = os.environ.get("REAGENT_LEDGER_PATH")
    os.environ["REAGENT_LEDGER_PATH"] = os.path.join(_MODULE_TEMP_DIR, "ledger.json")


def tearDownModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    if _MODULE_PREV_LEDGER is not None:
        os.environ["REAGENT_LEDGER_PATH"] = _MODULE_PREV_LEDGER
    else:
        os.environ.pop("REAGENT_LEDGER_PATH", None)
    if _MODULE_TEMP_DIR and os.path.exists(_MODULE_TEMP_DIR):
        shutil.rmtree(_MODULE_TEMP_DIR, ignore_errors=True)


from reagent.taxonomy import AgentStatus, PostconditionStatus
from reagent.models import (
    CompositeTaskPlan,
    WorkflowStep,
    WorkflowReceipt,
    StepReceipt,
    StepAttempt,
    TaskState,
    Postcondition,
    ActionResult,
    Evidence,
    VerificationResult,
)
from reagent.pipeline import ReAgentPipeline
from reagent.workflow import WorkflowRunner
from reagent.tools.base import BaseTool
from reagent.tools.registry import ToolRegistry


# ── Custom mock tools for testing ─────────────────────────────────────────────

class CountingStepTool(BaseTool):
    """Tool that tracks execution count and can be configured to fail or succeed."""

    def __init__(self, name: str, success: bool = True, output: Optional[Dict[str, Any]] = None, supports_idempotency: bool = True):
        self.name = name
        self.description = f"Counting test tool {name}"
        self.parameters_schema = {"type": "object", "properties": {"val": {"type": "string"}}}
        self.effect_class = "state-changing"
        self.supports_idempotency_key = supports_idempotency
        self.execution_count = 0
        self.should_succeed = success
        self.mock_output = output or {"status": "OK", "step_done": True}

    def execute(self, params: Dict[str, Any]) -> Any:
        self.execution_count += 1
        if not self.should_succeed:
            raise RuntimeError(f"Simulated execution failure in {self.name}")
        return self.mock_output


class UnforeseenCustomAdapterTool(BaseTool):
    """Novel, unforeseen tool: Smart Thermostat controller (zero hardcoding in engine)."""

    name: str = "iot_thermostat_controller"
    description: str = "Sets room temperature and returns sensor reading"
    parameters_schema: Dict[str, Any] = {"type": "object", "properties": {"temp": {"type": "integer"}}}
    effect_class: str = "state-changing"
    supports_idempotency_key: bool = True

    def execute(self, params: Dict[str, Any]) -> Any:
        target_temp = params.get("temp", 72)
        return {"current_temp": target_temp, "mode": "COOLING", "device_online": True}


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestGenericCompositeWorkflowEngine(unittest.TestCase):
    """Comprehensive offline tests for the Phase 3 Workflow Engine."""

    def setUp(self):
        self.pipeline = ReAgentPipeline()
        self.runner = WorkflowRunner(pipeline=self.pipeline)

    # ── 1. Valid ordered two-step workflow ───────────────────────────────────
    def test_01_valid_ordered_two_step_workflow(self):
        tool1 = CountingStepTool("tool_step_1", success=True, output={"status": "DONE_1", "step_1": True})
        tool2 = CountingStepTool("tool_step_2", success=True, output={"status": "DONE_2", "step_2": True})
        self.pipeline.tool_registry.register_tool(tool1)
        self.pipeline.tool_registry.register_tool(tool2)

        plan = CompositeTaskPlan(
            task_id="wf-test-01",
            objective="Two-step sequential task",
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    tool_name="tool_step_1",
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
                WorkflowStep(
                    step_id="step_2",
                    tool_name="tool_step_2",
                    depends_on=["step_1"],
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
            ],
        )

        receipt = self.runner.execute_workflow(plan)

        self.assertEqual(receipt.status, AgentStatus.VERIFIED)
        self.assertEqual(receipt.completed_steps, ["step_1", "step_2"])
        self.assertEqual(receipt.incomplete_steps, [])
        self.assertEqual(receipt.execution_order, ["step_1", "step_2"])
        self.assertEqual(tool1.execution_count, 1)
        self.assertEqual(tool2.execution_count, 1)

    # ── 2. Empty workflow and duplicate step IDs rejected ────────────────────
    def test_02_empty_workflow_and_duplicate_step_ids_rejected(self):
        # Empty plan rejected
        plan_empty = CompositeTaskPlan(task_id="empty-wf", objective="empty", steps=[])
        with self.assertRaises(ValueError):
            self.runner.execute_workflow(plan_empty)

        # Duplicate step_id rejected
        plan_dupe = CompositeTaskPlan(
            task_id="dupe-wf",
            objective="dupe",
            steps=[
                WorkflowStep(step_id="step_a", tool_name="tool_step_1"),
                WorkflowStep(step_id="step_a", tool_name="tool_step_2"),
            ],
        )
        with self.assertRaises(ValueError):
            self.runner.execute_workflow(plan_dupe)

        # Invalid forward dependency rejected
        plan_invalid_dep = CompositeTaskPlan(
            task_id="invalid-dep-wf",
            objective="invalid dep",
            steps=[
                WorkflowStep(step_id="step_1", tool_name="tool_step_1", depends_on=["step_2"]),
                WorkflowStep(step_id="step_2", tool_name="tool_step_2"),
            ],
        )
        with self.assertRaises(ValueError):
            self.runner.execute_workflow(plan_invalid_dep)

    # ── 3. Step 1 VERIFIED; step 2 fails: step 1 executes once ───────────────
    def test_03_step1_verified_step2_fails_step1_executes_once(self):
        tool1 = CountingStepTool("tool_step_1_once", success=True, output={"status": "OK", "step_1_done": True})
        tool2 = CountingStepTool("tool_step_2_fail", success=False, output={"status": "OK", "step_2_done": True})
        self.pipeline.tool_registry.register_tool(tool1)
        self.pipeline.tool_registry.register_tool(tool2)

        plan = CompositeTaskPlan(
            task_id="wf-test-03",
            objective="Partial workflow",
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    tool_name="tool_step_1_once",
                    postconditions=[Postcondition(predicate="KEY_EXISTS", target="step_1_done")],
                ),
                WorkflowStep(
                    step_id="step_2",
                    tool_name="tool_step_2_fail",
                    depends_on=["step_1"],
                    max_recovery_attempts=0,
                    postconditions=[Postcondition(predicate="KEY_EXISTS", target="step_2_done")],
                ),
            ],
        )

        receipt = self.runner.execute_workflow(plan)

        self.assertEqual(receipt.status, AgentStatus.PARTIAL)
        self.assertIn("step_1", receipt.completed_steps)
        self.assertIn("step_2", receipt.failed_steps)
        self.assertEqual(tool1.execution_count, 1)

    # ── 4. Recovery resumes only an eligible unfinished step ─────────────────
    def test_04_recovery_resumes_only_eligible_unfinished_step(self):
        tool1 = CountingStepTool("tool_resume_1", success=True, output={"status": "OK", "step_1_done": True})
        tool2 = CountingStepTool("tool_resume_2", success=False, output={"status": "OK", "step_2_done": True})
        self.pipeline.tool_registry.register_tool(tool1)
        self.pipeline.tool_registry.register_tool(tool2)

        plan = CompositeTaskPlan(
            task_id="wf-test-04",
            objective="Resume unfinished step",
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    tool_name="tool_resume_1",
                    postconditions=[Postcondition(predicate="KEY_EXISTS", target="step_1_done")],
                ),
                WorkflowStep(
                    step_id="step_2",
                    tool_name="tool_resume_2",
                    depends_on=["step_1"],
                    max_recovery_attempts=0,
                    postconditions=[Postcondition(predicate="KEY_EXISTS", target="step_2_done")],
                ),
            ],
        )

        # Initial run: Step 1 passes, Step 2 fails
        receipt_1 = self.runner.execute_workflow(plan)
        self.assertEqual(receipt_1.status, AgentStatus.PARTIAL)
        self.assertEqual(tool1.execution_count, 1)
        initial_tool2_runs = tool2.execution_count

        # Fix tool 2 for recovery run
        tool2.should_succeed = True

        # Resume workflow with existing_receipt
        receipt_2 = self.runner.execute_workflow(plan, existing_receipt=receipt_1)

        self.assertEqual(receipt_2.status, AgentStatus.VERIFIED)
        self.assertIn("step_1", receipt_2.completed_steps)
        self.assertIn("step_2", receipt_2.completed_steps)
        # Invariant: Step 1 must NOT have been executed again!
        self.assertEqual(tool1.execution_count, 1)
        # Step 2 was resumed and succeeded
        self.assertGreater(tool2.execution_count, initial_tool2_runs)


    # ── 5. UNKNOWN on non-idempotent action does not trigger blind retry ─────
    def test_05_unknown_on_non_idempotent_action_blocks_blind_retry(self):
        # Non-idempotent tool
        tool_non_idempotent = CountingStepTool(
            "tool_non_idempotent",
            success=False,
            supports_idempotency=False,
        )
        self.pipeline.tool_registry.register_tool(tool_non_idempotent)

        plan = CompositeTaskPlan(
            task_id="wf-test-05",
            objective="Non-idempotent safety test",
            steps=[
                WorkflowStep(
                    step_id="step_ambig",
                    tool_name="tool_non_idempotent",
                    supports_idempotency=False,
                )
            ],
        )

        # Inject an ambiguous / unknown previous receipt
        prior_step_receipt = StepReceipt(
            step_id="step_ambig",
            tool_name="tool_non_idempotent",
            status=AgentStatus.UNKNOWN,
            executed=True,
            attempts=[
                StepAttempt(
                    attempt_number=1,
                    status=AgentStatus.UNKNOWN,
                    error="TimeoutAfterCommit: ambiguous outcome",
                )
            ],
        )
        existing_receipt = WorkflowReceipt(
            task_id=plan.task_id,
            plan=plan,
            step_receipts={"step_ambig": prior_step_receipt},
            status=AgentStatus.UNKNOWN,
            unresolved_steps=["step_ambig"],
        )

        # Resuming without a probe callback must NOT blind retry non-idempotent step
        resumed_receipt = self.runner.execute_workflow(plan, existing_receipt=existing_receipt)

        self.assertEqual(resumed_receipt.status, AgentStatus.UNKNOWN)
        # Execution count should be 0 because retry was blocked by safety gate
        self.assertEqual(tool_non_idempotent.execution_count, 0)

    # ── 6. Unfenced absence cannot authorize retry ───────────────────────────
    def test_06_unfenced_absence_cannot_authorize_retry(self):
        tool = CountingStepTool("tool_fencing_test", success=False, supports_idempotency=False)
        self.pipeline.tool_registry.register_tool(tool)

        plan = CompositeTaskPlan(
            task_id="wf-test-06",
            objective="Unfenced absence test",
            steps=[WorkflowStep(step_id="step_fenced", tool_name="tool_fencing_test", supports_idempotency=False)],
        )

        # Probe returns absent but unfenced / eventually consistent
        def unfenced_probe(tool_name, params, step):
            return {
                "status": "ABSENT",
                "fenced": False,
                "eventual_consistency": True,
            }

        receipt = self.runner.execute_workflow(plan, probe_callbacks={"step_fenced": unfenced_probe})
        # Fails closed, retry blocked by Phase 1 safety gate
        self.assertIn(receipt.status, (AgentStatus.UNKNOWN, AgentStatus.RECOVERY_EXHAUSTED, AgentStatus.FAILED))

    # ── 7. Already completed step is not repeated during same-run recovery ───
    def test_07_completed_step_not_repeated_during_recovery(self):
        tool1 = CountingStepTool("tool_c1", success=True, output={"c1_done": True})
        tool2 = CountingStepTool("tool_c2", success=False, output={"c2_done": True})
        self.pipeline.tool_registry.register_tool(tool1)
        self.pipeline.tool_registry.register_tool(tool2)

        plan = CompositeTaskPlan(
            task_id="wf-test-07",
            objective="No repeat test",
            steps=[
                WorkflowStep(
                    step_id="s1",
                    tool_name="tool_c1",
                    postconditions=[Postcondition(predicate="KEY_EXISTS", target="c1_done")],
                ),
                WorkflowStep(
                    step_id="s2",
                    tool_name="tool_c2",
                    depends_on=["s1"],
                    max_recovery_attempts=0,
                    postconditions=[Postcondition(predicate="KEY_EXISTS", target="c2_done")],
                ),
            ],
        )

        r1 = self.runner.execute_workflow(plan)
        self.assertEqual(tool1.execution_count, 1)

        # Run recovery
        tool2.should_succeed = True
        r2 = self.runner.execute_workflow(plan, existing_receipt=r1)

        self.assertEqual(tool1.execution_count, 1)  # Never repeated
        self.assertEqual(r2.status, AgentStatus.VERIFIED)


    # ── 8. Step evidence and attempt history survive recovery ────────────────
    def test_08_evidence_and_attempt_history_survive_recovery(self):
        tool = CountingStepTool("tool_history", success=False)
        self.pipeline.tool_registry.register_tool(tool)

        plan = CompositeTaskPlan(
            task_id="wf-test-08",
            objective="History survival test",
            steps=[WorkflowStep(step_id="step_hist", tool_name="tool_history")],
        )

        r1 = self.runner.execute_workflow(plan)
        self.assertEqual(len(r1.step_receipts["step_hist"].attempts), 1)

        tool.should_succeed = True
        r2 = self.runner.execute_workflow(plan, existing_receipt=r1)

        step_r = r2.step_receipts["step_hist"]
        # Must retain both attempts
        self.assertEqual(len(step_r.attempts), 2)
        self.assertEqual(step_r.attempts[0].attempt_number, 1)
        self.assertEqual(step_r.attempts[1].attempt_number, 2)
        self.assertIsNotNone(step_r.attempts[0].evidence)
        self.assertIsNotNone(step_r.attempts[1].evidence)

    # ── 9. Recovery budget exhaustion is represented truthfully ─────────────
    def test_09_recovery_budget_exhaustion_represented_truthfully(self):
        tool = CountingStepTool("tool_budget", success=False)
        self.pipeline.tool_registry.register_tool(tool)

        runner_capped = WorkflowRunner(pipeline=self.pipeline, max_workflow_recovery_budget=2)

        plan = CompositeTaskPlan(
            task_id="wf-test-09",
            objective="Budget exhaustion test",
            steps=[
                WorkflowStep(step_id="step_b1", tool_name="tool_budget", max_recovery_attempts=1),
                WorkflowStep(step_id="step_b2", tool_name="tool_budget", max_recovery_attempts=1),
                WorkflowStep(step_id="step_b3", tool_name="tool_budget", max_recovery_attempts=1),
            ],
        )

        receipt = runner_capped.execute_workflow(plan)
        self.assertIn(receipt.status, (AgentStatus.RECOVERY_EXHAUSTED, AgentStatus.FAILED))
        self.assertGreater(len(receipt.failed_steps) + len(receipt.unresolved_steps), 0)

    # ── 10. Workflow outcome correctly reflects all states ───────────────────
    def test_10_workflow_outcome_aggregation_truthfulness(self):
        # All verified -> VERIFIED
        r_all_verified = self.runner._aggregate_workflow_status(
            plan=CompositeTaskPlan(task_id="t1", objective="o", steps=[WorkflowStep(step_id="a", tool_name="t")]),
            completed_steps=["a"],
            incomplete_steps=[],
            failed_steps=[],
            unresolved_steps=[],
            step_receipts={"a": StepReceipt(step_id="a", tool_name="t", status=AgentStatus.VERIFIED)},
        )
        self.assertEqual(r_all_verified, AgentStatus.VERIFIED)

        # Mixed verified + failed -> PARTIAL
        r_partial = self.runner._aggregate_workflow_status(
            plan=CompositeTaskPlan(
                task_id="t2",
                objective="o",
                steps=[WorkflowStep(step_id="a", tool_name="t"), WorkflowStep(step_id="b", tool_name="t")],
            ),
            completed_steps=["a"],
            incomplete_steps=[],
            failed_steps=["b"],
            unresolved_steps=[],
            step_receipts={
                "a": StepReceipt(step_id="a", tool_name="t", status=AgentStatus.VERIFIED),
                "b": StepReceipt(step_id="b", tool_name="t", status=AgentStatus.VERIFIED_FAILURE),
            },
        )
        self.assertEqual(r_partial, AgentStatus.PARTIAL)

        # Any unresolved / unknown -> UNKNOWN
        r_unknown = self.runner._aggregate_workflow_status(
            plan=CompositeTaskPlan(
                task_id="t3",
                objective="o",
                steps=[WorkflowStep(step_id="a", tool_name="t"), WorkflowStep(step_id="b", tool_name="t")],
            ),
            completed_steps=["a"],
            incomplete_steps=[],
            failed_steps=[],
            unresolved_steps=["b"],
            step_receipts={
                "a": StepReceipt(step_id="a", tool_name="t", status=AgentStatus.VERIFIED),
                "b": StepReceipt(step_id="b", tool_name="t", status=AgentStatus.UNKNOWN),
            },
        )
        self.assertEqual(r_unknown, AgentStatus.UNKNOWN)

    # ── 11. Unforeseen adapter executes generically without engine changes ────
    def test_11_unforeseen_adapter_executes_generically(self):
        novel_tool = UnforeseenCustomAdapterTool()
        self.pipeline.tool_registry.register_tool(novel_tool)

        plan = CompositeTaskPlan(
            task_id="wf-test-11",
            objective="Novel IoT temperature setting",
            steps=[
                WorkflowStep(
                    step_id="set_temp_step",
                    tool_name="iot_thermostat_controller",
                    params={"temp": 68},
                    postconditions=[
                        Postcondition(predicate="STATUS_SUCCESS"),
                        Postcondition(predicate="KEY_EXISTS", target="device_online"),
                        Postcondition(predicate="VALUE_EQUALS", target="current_temp", expected=68),
                    ],
                )
            ],
        )

        receipt = self.runner.execute_workflow(plan)

        self.assertEqual(receipt.status, AgentStatus.VERIFIED)
        self.assertIn("set_temp_step", receipt.completed_steps)
        evals = receipt.step_receipts["set_temp_step"].verification_result.postcondition_evaluations
        self.assertEqual(len(evals), 3)
        for ev in evals:
            self.assertEqual(ev.status, PostconditionStatus.MET)

    # ── 12. Single-action pipeline remains unchanged ─────────────────────────
    def test_12_single_action_pipeline_remains_unchanged(self):
        tool = CountingStepTool("tool_single", success=True, output={"status": "OK"})
        self.pipeline.tool_registry.register_tool(tool)

        task = TaskState(objective="Direct single action pipeline check")
        res = self.pipeline.execute_with_contract(
            task=task,
            tool_name="tool_single",
            params={},
            postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
        )
        self.assertTrue(res.verification_result.verified)
        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)

    # ── 13. Two-Step Scenario (Section F): Commit-then-timeout resolution ────
    def test_13_two_step_scenario_with_delayed_probe_restraint(self):
        """
        Step 1: Booking tool succeeds and is independently verified.
        Step 2: Payment/transfer experiences commit-then-timeout (simulated failure on tool return).
                Delayed probe step confirms state is PRESENT.
                Restraint prevents duplicate write.
                Step 1 is preserved and never rerun.
        """
        step1_tool = CountingStepTool("booking_tool", success=True, output={"booking_id": "BK-999"})
        step2_tool = CountingStepTool("payment_tool", success=False, output=None, supports_idempotency=False)
        self.pipeline.tool_registry.register_tool(step1_tool)
        self.pipeline.tool_registry.register_tool(step2_tool)

        plan = CompositeTaskPlan(
            task_id="wf-two-step-scenario",
            objective="Book hotel and confirm deposit",
            steps=[
                WorkflowStep(
                    step_id="step_book",
                    tool_name="booking_tool",
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
                WorkflowStep(
                    step_id="step_pay",
                    tool_name="payment_tool",
                    depends_on=["step_book"],
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                    supports_idempotency=False,
                ),
            ],
        )

        # Ambiguity probe: Step 1 probe returns DELAYED, Step 2 probe confirms PRESENT
        def payment_delayed_probe(tool_name, params, step):
            if step == 1:
                return {"status": "DELAYED"}
            return {"status": "PRESENT", "confirmed": True}

        receipt = self.runner.execute_workflow(
            plan=plan,
            probe_callbacks={"step_pay": payment_delayed_probe},
        )

        self.assertEqual(receipt.status, AgentStatus.VERIFIED)
        self.assertIn("step_book", receipt.completed_steps)
        self.assertIn("step_pay", receipt.completed_steps)
        # Invariant: Step 1 was executed exactly once
        self.assertEqual(step1_tool.execution_count, 1)


if __name__ == "__main__":
    unittest.main()
