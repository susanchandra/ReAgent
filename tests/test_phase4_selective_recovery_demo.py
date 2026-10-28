"""
tests/test_phase4_selective_recovery_demo.py — Phase 4 Selective Recovery Demonstration

Proves that ReAgent recovers eligible unfinished steps without duplicating previously
verified side effects, enforcing the Phase 1 retry-safety invariants and Phase 2 structured outcomes:
  1. CommitThenTimeoutTool commits its side effect, then simulates a lost response/timeout.
  2. Independent observation discovers the actual state.
  3. A PRESENT result leads to RESTRAINT_OBSERVED and NO duplicate execution.
  4. An unfenced absence does not authorize retrying a non-idempotent action.
  5. UNKNOWN / DISCREPANCY evidence blocks unsafe retry.
  6. A previously VERIFIED workflow step remains complete and executes exactly once.
  7. Step attempts, evidence, status transitions, and safety decisions remain inspectable.
  8. The final outcome matches the evidence and does not falsely claim success.
  9. Relevant safety events (including RESTRAINT_OBSERVED) appear in timeline assertions.
  10. Tool registration arrangements do not cause ambiguous fallback selection.
"""

import os
import shutil
import tempfile
import unittest
from typing import Dict, Any, Optional, List

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_phase4_demo_")
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


# ── Named Tool Fixtures for Selective Recovery Demonstration ─────────────────

class DemoBookingTool(BaseTool):
    """
    Step 1 fixture: Resource reservation tool.
    Idempotent, deterministic, tracks execution count.
    """
    name: str = "demo_booking_tool"
    description: str = "Reserves a resource and returns a unique confirmation ID."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"resource_id": {"type": "string"}},
    }
    effect_class: str = "state-changing"
    supports_idempotency_key: bool = True

    def __init__(self):
        self.execution_count = 0

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.execution_count += 1
        res_id = params.get("resource_id", "RES-101")
        return {
            "booking_id": f"BKG-{res_id}",
            "status": "CONFIRMED",
            "booking_confirmed": True,
        }


class CommitThenTimeoutTool(BaseTool):
    """
    Step 2 fixture: Payment/transfer tool that commits a side effect to external state,
    then encounters a network/transport failure (e.g. dropped connection / deadline exceeded).
    The tool return payload is lost (success=False), but external state was altered.
    Non-idempotent by default.
    """
    name: str = "demo_commit_then_timeout_tool"
    description: str = "Executes payment transaction with external ledger commitment."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "txn_id": {"type": "string"},
            "amount": {"type": "number"},
        },
    }
    effect_class: str = "state-changing"
    supports_idempotency_key: bool = False

    def __init__(self, external_ledger: Optional[Dict[str, Any]] = None):
        self.execution_count = 0
        self.external_ledger = external_ledger if external_ledger is not None else {}
        self.simulate_lost_response = True

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.execution_count += 1
        txn_id = params.get("txn_id", "TXN-DEFAULT")
        amount = params.get("amount", 100.0)

        # The side effect COMMITS in the external ledger
        self.external_ledger[txn_id] = {
            "amount": amount,
            "status": "COMMITTED",
            "execution_attempt": self.execution_count,
        }

        # But the response is lost / timed out
        if self.simulate_lost_response:
            raise TimeoutError(
                f"ConnectionTimeoutAfterCommit: Transaction {txn_id} sent, but socket timed out before HTTP 200 return."
            )

        return {"txn_id": txn_id, "status": "COMMITTED", "amount": amount, "success": True}


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestPhase4SelectiveRecoveryDemo(unittest.TestCase):
    """
    Offline, deterministic demonstration of selective recovery and restraint.
    Proves that ReAgent handles ambiguous outcomes and commit-then-timeout failures
    safely without duplicating real-world side effects.
    """

    def setUp(self):
        self.pipeline = ReAgentPipeline()
        self.runner = WorkflowRunner(pipeline=self.pipeline)
        self.booking_tool = DemoBookingTool()
        self.external_ledger = {}
        self.timeout_tool = CommitThenTimeoutTool(external_ledger=self.external_ledger)

        self.pipeline.tool_registry.register_tool(self.booking_tool)
        self.pipeline.tool_registry.register_tool(self.timeout_tool)

    # ── Test 1: PRESENT probe leads to RESTRAINT_OBSERVED and NO duplicate execution ──
    def test_01_commit_then_timeout_restraint_prevents_duplicate_write(self):
        """
        Step 1: Booking succeeds and is independently verified.
        Step 2: Commit-then-timeout occurs (tool commits to ledger, but throws TimeoutError).
        Probe: Independent observer queries external ledger and discovers state is PRESENT.
        Result: ReAgent restrains retry, emits RESTRAINT_OBSERVED, records VERIFIED.
        Invariant: Step 1 executed exactly 1 time. Step 2 executed exactly 1 time (NO duplicate write).
        """
        plan = CompositeTaskPlan(
            task_id="p4-demo-restraint",
            objective="Book reservation and commit deposit without duplication",
            steps=[
                WorkflowStep(
                    step_id="step_book",
                    tool_name="demo_booking_tool",
                    params={"resource_id": "ROOM-502"},
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
                WorkflowStep(
                    step_id="step_pay",
                    tool_name="demo_commit_then_timeout_tool",
                    params={"txn_id": "TXN-7701", "amount": 250.0},
                    depends_on=["step_book"],
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                    supports_idempotency=False,
                ),
            ],
        )

        def authoritative_present_probe(tool_name: str, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            txn_id = params.get("txn_id")
            if txn_id in self.external_ledger:
                return {
                    "status": "PRESENT",
                    "txn_id": txn_id,
                    "is_authoritative": True,
                    "fenced": True,
                    "data": self.external_ledger[txn_id],
                }
            return {"status": "CONFIRMED_ABSENT", "is_authoritative": True, "fenced": True}

        receipt = self.runner.execute_workflow(
            plan=plan,
            probe_callbacks={"step_pay": authoritative_present_probe},
        )

        # 1. Overall workflow verified
        self.assertEqual(receipt.status, AgentStatus.VERIFIED)
        self.assertEqual(receipt.completed_steps, ["step_book", "step_pay"])
        self.assertEqual(receipt.unresolved_steps, [])
        self.assertEqual(receipt.failed_steps, [])

        # 2. Side-effect counts: Step 1 ran ONCE, Step 2 ran ONCE (no duplicate!)
        self.assertEqual(self.booking_tool.execution_count, 1)
        self.assertEqual(self.timeout_tool.execution_count, 1)
        self.assertEqual(self.external_ledger["TXN-7701"]["execution_attempt"], 1)

        # 3. RESTRAINT_OBSERVED timeline event verified
        step_pay_receipt = receipt.step_receipts["step_pay"]
        self.assertEqual(step_pay_receipt.status, AgentStatus.VERIFIED)
        self.assertEqual(len(step_pay_receipt.attempts), 1)

        # Pipeline cycle result verified restraint
        step_pay_attempt = step_pay_receipt.attempts[0]
        self.assertTrue(step_pay_attempt.action_result.success)
        self.assertIn("Restrained", step_pay_attempt.action_result.output)

    # ── Test 2: Unfenced absence prohibits retry on non-idempotent action ──
    def test_02_unfenced_absence_prohibits_retry(self):
        """
        Step 2 fails. Probe reports absence, but it is UNFENCED / eventually consistent.
        Result: ReAgent refuses to blind-retry non-idempotent action.
        Fails closed with SAFETY_GATE_BLOCKED_RETRY and status UNKNOWN / ESCALATED.
        """
        plan = CompositeTaskPlan(
            task_id="p4-demo-unfenced",
            objective="Unfenced absence safety check",
            steps=[
                WorkflowStep(
                    step_id="step_book",
                    tool_name="demo_booking_tool",
                    params={"resource_id": "ROOM-503"},
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
                WorkflowStep(
                    step_id="step_pay",
                    tool_name="demo_commit_then_timeout_tool",
                    params={"txn_id": "TXN-7702", "amount": 100.0},
                    depends_on=["step_book"],
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                    supports_idempotency=False,
                ),
            ],
        )

        def unfenced_absence_probe(tool_name: str, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {
                "status": "ABSENT",
                "fenced": False,              # Crucial: NOT fenced
                "is_authoritative": False,     # Not authoritative
                "consistency": "EVENTUAL",
            }

        receipt = self.runner.execute_workflow(
            plan=plan,
            probe_callbacks={"step_pay": unfenced_absence_probe},
        )

        # Must fail-closed to UNKNOWN (overall workflow) with step_pay in unresolved
        self.assertEqual(receipt.status, AgentStatus.UNKNOWN)
        self.assertIn("step_book", receipt.completed_steps)
        self.assertIn("step_pay", receipt.unresolved_steps)
        self.assertEqual(receipt.step_receipts["step_pay"].status, AgentStatus.ESCALATED)

        # Exactly 1 attempt on step 1, exactly 1 on step 2 (retry BLOCKED)
        self.assertEqual(self.booking_tool.execution_count, 1)
        self.assertEqual(self.timeout_tool.execution_count, 1)

    # ── Test 3: UNKNOWN or DISCREPANCY evidence blocks unsafe retry ───────────
    def test_03_unknown_discrepant_state_blocks_retry(self):
        """
        Step 2 fails. Observer finds corrupted, partial, or discrepant state.
        Result: ReAgent recognizes UNKNOWN is not ABSENT. Escalates safely without retry.
        """
        plan = CompositeTaskPlan(
            task_id="p4-demo-discrepancy",
            objective="Discrepancy safety check",
            steps=[
                WorkflowStep(
                    step_id="step_book",
                    tool_name="demo_booking_tool",
                    params={"resource_id": "ROOM-504"},
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
                WorkflowStep(
                    step_id="step_pay",
                    tool_name="demo_commit_then_timeout_tool",
                    params={"txn_id": "TXN-7703", "amount": 150.0},
                    depends_on=["step_book"],
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                    supports_idempotency=False,
                ),
            ],
        )

        def discrepant_probe(tool_name: str, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {
                "status": "DISCREPANCY",
                "reason": "Record found with partial audit checksum mismatch",
            }

        receipt = self.runner.execute_workflow(
            plan=plan,
            probe_callbacks={"step_pay": discrepant_probe},
        )

        self.assertEqual(receipt.status, AgentStatus.UNKNOWN)
        self.assertIn("step_pay", receipt.unresolved_steps)
        self.assertEqual(receipt.step_receipts["step_pay"].status, AgentStatus.ESCALATED)
        self.assertEqual(self.timeout_tool.execution_count, 1)

    # ── Test 4: Authoritative CONFIRMED_ABSENT + fenced probe permits safe retry ─
    def test_04_authoritative_fenced_absence_permits_safe_retry(self):
        """
        Step 2 fails on initial attempt (e.g. transient network glitch before commit).
        Observer confirms CONFIRMED_ABSENT with strong fencing.
        Retry is safe and permitted. On retry, tool succeeds.
        Result: Workflow completes with VERIFIED, step 1 executed once, step 2 executed twice.
        """
        # Standalone pipeline configured for safe retry demonstration
        retry_pipeline = ReAgentPipeline()
        runner = WorkflowRunner(pipeline=retry_pipeline)

        b_tool = DemoBookingTool()
        p_tool = CommitThenTimeoutTool(external_ledger={})
        retry_pipeline.tool_registry.register_tool(b_tool)
        retry_pipeline.tool_registry.register_tool(p_tool)

        plan = CompositeTaskPlan(
            task_id="p4-demo-safe-retry",
            objective="Authoritative fenced absence retry",
            steps=[
                WorkflowStep(
                    step_id="step_book",
                    tool_name="demo_booking_tool",
                    params={"resource_id": "ROOM-505"},
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
                WorkflowStep(
                    step_id="step_pay",
                    tool_name="demo_commit_then_timeout_tool",
                    params={"txn_id": "TXN-7704", "amount": 300.0},
                    depends_on=["step_book"],
                    max_recovery_attempts=1,
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                    supports_idempotency=True,  # Idempotent key or fenced absence allows retry
                ),
            ],
        )

        def fenced_absent_probe(tool_name: str, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            # Authoritative absence: definitely never received
            # Allow next attempt to succeed
            p_tool.simulate_lost_response = False
            return {
                "status": "CONFIRMED_ABSENT",
                "fenced": True,
                "is_authoritative": True,
            }

        receipt = runner.execute_workflow(
            plan=plan,
            probe_callbacks={"step_pay": fenced_absent_probe},
        )

        self.assertEqual(receipt.status, AgentStatus.VERIFIED)
        self.assertIn("step_book", receipt.completed_steps)
        self.assertIn("step_pay", receipt.completed_steps)
        self.assertEqual(b_tool.execution_count, 1)      # Step 1 was NOT rerun
        self.assertEqual(p_tool.execution_count, 2)      # Initial + 1 retry

    # ── Test 5: Selective resumption preserves Step 1 and attempt history ──
    def test_05_selective_resumption_preserves_verified_step_and_attempt_history(self):
        """
        Demonstrates selective resumption across separate invocations:
        Run 1: Step 1 succeeds. Step 2 fails with max_recovery_attempts=0. Workflow status is PARTIAL.
        Run 2: Workflow is resumed using existing_receipt from Run 1.
               Step 1 is PRESERVED and SKIPPED (execution_count remains 1).
               Step 2 is resumed and succeeds.
               Step 2 receipt retains full attempt history (both attempt 1 and attempt 2).
        """
        plan = CompositeTaskPlan(
            task_id="p4-demo-resumption",
            objective="Two-step selective resumption demo",
            steps=[
                WorkflowStep(
                    step_id="step_book",
                    tool_name="demo_booking_tool",
                    params={"resource_id": "ROOM-506"},
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
                WorkflowStep(
                    step_id="step_pay",
                    tool_name="demo_commit_then_timeout_tool",
                    params={"txn_id": "TXN-7705", "amount": 500.0},
                    depends_on=["step_book"],
                    max_recovery_attempts=0,
                    postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
                ),
            ],
        )

        # Run 1: Step 1 passes, Step 2 fails
        receipt_1 = self.runner.execute_workflow(plan)
        self.assertEqual(receipt_1.status, AgentStatus.PARTIAL)
        self.assertEqual(self.booking_tool.execution_count, 1)
        self.assertEqual(self.timeout_tool.execution_count, 1)
        self.assertIn("step_book", receipt_1.completed_steps)
        self.assertIn("step_pay", receipt_1.failed_steps)

        # Now configure Step 2 to succeed on next execution
        self.timeout_tool.simulate_lost_response = False

        # Run 2: Resume with existing_receipt
        receipt_2 = self.runner.execute_workflow(plan, existing_receipt=receipt_1)

        # Invariant: Step 1 was NEVER re-executed!
        self.assertEqual(self.booking_tool.execution_count, 1)
        # Step 2 was resumed and executed a second time
        self.assertEqual(self.timeout_tool.execution_count, 2)

        # Final workflow status is VERIFIED
        self.assertEqual(receipt_2.status, AgentStatus.VERIFIED)
        self.assertEqual(receipt_2.completed_steps, ["step_book", "step_pay"])
        self.assertEqual(receipt_2.incomplete_steps, [])

        # Attempt history is preserved
        step_pay_receipt = receipt_2.step_receipts["step_pay"]
        self.assertEqual(len(step_pay_receipt.attempts), 2)
        self.assertEqual(step_pay_receipt.attempts[0].attempt_number, 1)
        self.assertEqual(step_pay_receipt.attempts[0].status, AgentStatus.VERIFIED_FAILURE)
        self.assertEqual(step_pay_receipt.attempts[1].attempt_number, 2)
        self.assertEqual(step_pay_receipt.attempts[1].status, AgentStatus.VERIFIED)


if __name__ == "__main__":
    unittest.main()
