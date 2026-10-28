"""
tests/test_phase1_safety_foundation.py — Verification of Phase 1 Safety Foundation.

Covers:
1. Observer unavailable or malformed evidence -> no retry authorized.
2. Unfenced absence -> no non-idempotent retry authorized.
3. Stale or mismatched operation observation -> no retry authorized.
4. Supported genuinely-enforced safety mechanism -> only permitted retry behavior.
5. Ambiguous outcome -> escalates or remains uncertain (never guessed).
6. Existing safe retry behavior and tests remain intact.
7. Deterministic late-commit demonstration: unfenced late-commit produces duplicate under naive retry,
   while ReAgent's authoritative fencing gate safely blocks retry and escalates.
"""

from __future__ import annotations

import unittest
from typing import Dict, Any

from reagent.models import (
    TaskState,
    AgentStatus,
    Postcondition,
)
from reagent.pipeline import ReAgentPipeline
from reagent.tools.base import BaseTool
from reagent.tools.hotel_tool import StatusUnavailableError


class CountingMockTool(BaseTool):
    """Mock tool that records every execution attempt and injects controlled errors."""

    def __init__(self, name: str = "mock_effect_tool", fail_first: bool = True, supports_idempotency_key: bool = False):
        self.name = name
        self.description = "Test mock tool"
        self.effect_class = "state-changing"
        self.parameters_schema = {"type": "object"}
        self.supports_idempotency_key = supports_idempotency_key
        self.fail_first = fail_first
        self.attempts = 0
        self.executed_params: list = []

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.attempts += 1
        self.executed_params.append(dict(params))
        if self.fail_first and self.attempts == 1:
            raise TimeoutError("SimulatedTimeoutError: Tool execution timed out before acknowledgment")
        return {"status": "SUCCESS", "attempt": self.attempts, **params}


def _make_pipeline(tool: BaseTool) -> ReAgentPipeline:
    pipeline = ReAgentPipeline(agent_id="test-agent")
    pipeline.tool_registry.register(tool)
    return pipeline


class TestPhase1SafetyFoundation(unittest.TestCase):

    # ── Test 1: Observer Unavailable or Malformed Evidence ────────────────────

    def test_01a_observer_unavailable_exception_blocks_retry(self):
        """When probe raises StatusUnavailableError, retry is prohibited and status is ESCALATED."""
        tool = CountingMockTool(fail_first=True)
        pipeline = _make_pipeline(tool)
        task = TaskState(task_id="t_obs_unavail", objective="Test observer unavailable")
        params = {"booking_id": "BKG-001"}

        def probe_raise(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            raise StatusUnavailableError("Observer connection dropped (503)")

        res = pipeline.execute_with_contract(
            task=task,
            tool_name=tool.name,
            params=params,
            probe_callback=probe_raise,
            max_recovery_attempts=1,
            supports_idempotency=False,
        )
        self.assertEqual(res.action_result.success, False)
        self.assertEqual(pipeline.agent_state.status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry when observer is unavailable")

    def test_01b_empty_or_malformed_probe_evidence_blocks_retry(self):
        """When probe returns empty dict or malformed flag, retry is prohibited."""
        for malformed_payload in [{}, {"status": ""}, {"malformed": True}, {"data": "no_status"}]:
            with self.subTest(payload=malformed_payload):
                tool = CountingMockTool(fail_first=True)
                pipeline = _make_pipeline(tool)
                task = TaskState(task_id="t_malformed", objective="Test malformed evidence")
                params = {"booking_id": "BKG-002"}

                def probe_malformed(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
                    return malformed_payload

                res = pipeline.execute_with_contract(
                    task=task,
                    tool_name=tool.name,
                    params=params,
                    probe_callback=probe_malformed,
                    max_recovery_attempts=1,
                    supports_idempotency=False,
                )
                self.assertEqual(res.action_result.success, False)
                self.assertEqual(pipeline.agent_state.status, AgentStatus.ESCALATED)
                self.assertEqual(tool.attempts, 1, "Must NOT retry when probe evidence is malformed")

    # ── Test 2: Unfenced Absence Blocks Non-Idempotent Retry ──────────────────

    def test_02a_unfenced_absence_blocks_non_idempotent_retry(self):
        """When probe reports absence but declares fenced=False, non-idempotent retry is blocked."""
        tool = CountingMockTool(fail_first=True)
        pipeline = _make_pipeline(tool)
        task = TaskState(task_id="t_unfenced", objective="Test unfenced absence")
        params = {"booking_id": "BKG-003"}

        def probe_unfenced(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {"status": "CONFIRMED_ABSENT", "fenced": False, "booking_id": "BKG-003"}

        res = pipeline.execute_with_contract(
            task=task,
            tool_name=tool.name,
            params=params,
            probe_callback=probe_unfenced,
            max_recovery_attempts=1,
            supports_idempotency=False,
        )
        self.assertEqual(res.action_result.success, False)
        self.assertEqual(pipeline.agent_state.status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry non-idempotent operation when absence is unfenced")

    def test_02b_eventual_consistency_absence_blocks_non_idempotent_retry(self):
        """Eventual consistency absence cannot rule out in-flight commit; retry is blocked."""
        tool = CountingMockTool(fail_first=True)
        pipeline = _make_pipeline(tool)
        task = TaskState(task_id="t_eventual", objective="Test eventual consistency absence")
        params = {"booking_id": "BKG-004"}

        def probe_eventual(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {"status": "CONFIRMED_ABSENT", "consistency": "EVENTUAL", "booking_id": "BKG-004"}

        res = pipeline.execute_with_contract(
            task=task,
            tool_name=tool.name,
            params=params,
            probe_callback=probe_eventual,
            max_recovery_attempts=1,
            supports_idempotency=False,
        )
        self.assertEqual(res.action_result.success, False)
        self.assertEqual(pipeline.agent_state.status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry non-idempotent operation under eventual consistency")

    # ── Test 3: Stale or Mismatched Operation Observation ─────────────────────

    def test_03a_mismatched_operation_id_blocks_retry(self):
        """When probe returns evidence for a DIFFERENT operation ID, retry is blocked."""
        tool = CountingMockTool(fail_first=True)
        pipeline = _make_pipeline(tool)
        task = TaskState(task_id="t_mismatch", objective="Test mismatched operation ID")
        params = {"booking_id": "BKG-CORRECT-100"}

        def probe_mismatched(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {"status": "CONFIRMED_ABSENT", "booking_id": "BKG-DIFFERENT-999"}

        res = pipeline.execute_with_contract(
            task=task,
            tool_name=tool.name,
            params=params,
            probe_callback=probe_mismatched,
            max_recovery_attempts=1,
            supports_idempotency=False,
        )
        self.assertEqual(res.action_result.success, False)
        self.assertEqual(pipeline.agent_state.status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry when observation operation ID does not match")

    def test_03b_stale_observation_blocks_retry(self):
        """When probe returns stale=True, retry is blocked."""
        tool = CountingMockTool(fail_first=True)
        pipeline = _make_pipeline(tool)
        task = TaskState(task_id="t_stale", objective="Test stale observation")
        params = {"booking_id": "BKG-005"}

        def probe_stale(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {"status": "CONFIRMED_ABSENT", "booking_id": "BKG-005", "stale": True}

        res = pipeline.execute_with_contract(
            task=task,
            tool_name=tool.name,
            params=params,
            probe_callback=probe_stale,
            max_recovery_attempts=1,
            supports_idempotency=False,
        )
        self.assertEqual(res.action_result.success, False)
        self.assertEqual(pipeline.agent_state.status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry when observation is marked stale")

    # ── Test 4: Supported Safety Mechanism Permits Retry ──────────────────────

    def test_04a_idempotent_tool_permits_safe_retry(self):
        """Tool with supports_idempotency=True safely retries and passes verification."""
        tool = CountingMockTool(fail_first=True, supports_idempotency_key=True)
        pipeline = _make_pipeline(tool)
        task = TaskState(task_id="t_idempotent", objective="Test idempotent safe retry")
        params = {"booking_id": "BKG-IDEMP-001"}

        def probe_any(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {"status": "CONFIRMED_ABSENT", "booking_id": "BKG-IDEMP-001"}

        res = pipeline.execute_with_contract(
            task=task,
            tool_name=tool.name,
            params=params,
            probe_callback=probe_any,
            max_recovery_attempts=1,
            supports_idempotency=True,
        )
        self.assertEqual(res.action_result.success, True)
        self.assertEqual(pipeline.agent_state.status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 2, "Genuinely idempotent tool must be allowed 1 safe retry")

    def test_04b_authoritative_fenced_absence_permits_safe_retry(self):
        """Non-idempotent tool with authoritative, fenced absence is safely retried once."""
        tool = CountingMockTool(fail_first=True, supports_idempotency_key=False)
        pipeline = _make_pipeline(tool)
        task = TaskState(task_id="t_fenced_safe", objective="Test fenced safe retry")
        params = {"booking_id": "BKG-FENCED-001"}

        def probe_fenced(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {
                "status": "CONFIRMED_ABSENT",
                "fenced": True,
                "authoritative": True,
                "booking_id": "BKG-FENCED-001",
            }

        res = pipeline.execute_with_contract(
            task=task,
            tool_name=tool.name,
            params=params,
            probe_callback=probe_fenced,
            max_recovery_attempts=1,
            supports_idempotency=False,
        )
        self.assertEqual(res.action_result.success, True)
        self.assertEqual(pipeline.agent_state.status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 2, "Authoritatively fenced absence permits exactly 1 safe retry")

    # ── Test 5: Ambiguous Outcome Escalates Without Guessing ───────────────────

    def test_05_ambiguous_outcomes_escalate_without_guessing(self):
        """Statuses UNKNOWN, DISCREPANCY, PARTIAL, UNAVAILABLE must never trigger retry."""
        ambiguous_statuses = ["UNKNOWN", "DISCREPANCY", "PARTIAL", "WRONG_CONTENT", "UNAVAILABLE", "UNOBSERVABLE"]
        for status in ambiguous_statuses:
            with self.subTest(status=status):
                tool = CountingMockTool(fail_first=True)
                pipeline = _make_pipeline(tool)
                task = TaskState(task_id=f"t_ambig_{status}", objective="Test ambiguous escalation")
                params = {"booking_id": "BKG-AMBIG"}

                def probe_ambig(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
                    return {"status": status, "booking_id": "BKG-AMBIG"}

                res = pipeline.execute_with_contract(
                    task=task,
                    tool_name=tool.name,
                    params=params,
                    probe_callback=probe_ambig,
                    max_recovery_attempts=1,
                    supports_idempotency=False,
                )
                self.assertEqual(res.action_result.success, False)
                self.assertEqual(pipeline.agent_state.status, AgentStatus.ESCALATED)
                self.assertEqual(tool.attempts, 1, f"Must NEVER retry when probe returns {status}")

    # ── Test 6: Existing Safe Retry Behavior Remains Intact ───────────────────

    def test_06_standard_confirmed_absent_backward_compatibility(self):
        """Standard probe returning {"status": "CONFIRMED_ABSENT"} continues to work seamlessly."""
        tool = CountingMockTool(fail_first=True)
        pipeline = _make_pipeline(tool)
        task = TaskState(task_id="t_standard", objective="Test standard CONFIRMED_ABSENT")
        params = {"booking_id": "BKG-STD-001"}

        def probe_standard(name: str, p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return {"status": "CONFIRMED_ABSENT", "booking_id": "BKG-STD-001"}

        res = pipeline.execute_with_contract(
            task=task,
            tool_name=tool.name,
            params=params,
            probe_callback=probe_standard,
            max_recovery_attempts=1,
            supports_idempotency=False,
        )
        self.assertEqual(res.action_result.success, True)
        self.assertEqual(pipeline.agent_state.status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 2)

    # ── Test 7: Deterministic Late-Commit Demonstration ───────────────────────

    def test_07_deterministic_late_commit_demonstration(self):
        """
        Models a distributed late-commit race:
        1. Request 1 is sent. Transport drops connection before acknowledgment.
        2. At probe time (t=1), the database has NOT completed the write yet (probe sees absent).
        3. If system is unfenced:
           - Naive retry blindly sends Request 2.
           - Both Request 1 and Request 2 commit to storage -> 2 rows (DUPLICATE STATE CORRUPTION).
        4. Under ReAgent's authoritative safety gate:
           - The probe reports unfenced absence (or eventually consistent).
           - ReAgent REFUSES to retry, preventing duplicate execution and state corruption.
        """
        # Simulated unfenced storage ledger
        ledger: list = []

        class UnfencedServiceSimulator:
            def __init__(self):
                self.in_flight_operations = []

            def execute(self, params: Dict[str, Any]) -> None:
                op_id = params["operation_id"]
                # In-flight operation that will commit delayed
                self.in_flight_operations.append(op_id)
                # Network times out immediately
                raise TimeoutError("SocketTimeout: No acknowledgment received")

            def probe(self, op_id: str) -> Dict[str, Any]:
                # At probe time, not yet in ledger; but unfenced / no generation token
                is_in_ledger = op_id in ledger
                return {
                    "status": "PRESENT" if is_in_ledger else "CONFIRMED_ABSENT",
                    "operation_id": op_id,
                    "fenced": False,  # UNFENCED: does not guarantee absence against in-flight race
                    "consistency": "EVENTUAL",
                }

            def late_commit_all(self):
                # Simulated delayed commit of in-flight operations
                while self.in_flight_operations:
                    op = self.in_flight_operations.pop(0)
                    ledger.append(op)

        sim = UnfencedServiceSimulator()

        # Step A: Naive approach without fencing
        # Attempt 1: In-flight timeout
        try:
            sim.execute({"operation_id": "TXN-LATE-001"})
        except TimeoutError:
            pass

        # Naive probe sees absent (not in ledger yet)
        probe_obs = sim.probe("TXN-LATE-001")
        self.assertEqual(probe_obs["status"], "CONFIRMED_ABSENT")

        # Naive agent blindly retries because probe said absent
        ledger.append("TXN-LATE-001-RETRY")

        # Late commit of original in-flight request arrives!
        sim.late_commit_all()

        # Result under naive approach: DUPLICATE SIDE EFFECTS in ledger!
        self.assertEqual(len(ledger), 2, "Naive retry produced 2 entries in ledger (duplicate debit/booking)!")

        # Step B: ReAgent with Authoritative Fencing
        # Reset ledger
        ledger.clear()
        sim2 = UnfencedServiceSimulator()

        class UnfencedTool(BaseTool):
            name = "unfenced_transfer"
            description = "Unfenced tool"
            effect_class = "state-changing"
            parameters_schema = {"type": "object"}

            def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
                return sim2.execute(params)

        p = _make_pipeline(UnfencedTool())
        t = TaskState(task_id="t_late_commit", objective="Test late commit safety")
        target_params = {"operation_id": "TXN-REAGENT-SAFE"}

        def probe_cb(name: str, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
            return sim2.probe("TXN-REAGENT-SAFE")

        # ReAgent executes: initial attempt times out, probe returns unfenced absence
        result = p.execute_with_contract(
            task=t,
            tool_name="unfenced_transfer",
            params=target_params,
            probe_callback=probe_cb,
            max_recovery_attempts=1,
            supports_idempotency=False,
        )

        # ReAgent safely BLOCKED the unsafe retry and escalated!
        self.assertEqual(p.agent_state.status, AgentStatus.ESCALATED)
        self.assertFalse(result.action_result.success)

        # Even when the delayed in-flight operation commits, exactly ONE entry enters ledger
        sim2.late_commit_all()
        self.assertEqual(len(ledger), 1, "ReAgent prevented duplicate execution: exactly 1 committed entry!")


if __name__ == "__main__":
    unittest.main()
