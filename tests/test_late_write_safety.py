"""
tests/test_late_write_safety.py — Deterministic Tests for Late-Write Risk & Safe Retry Guarantees

Covers all 10 Task 3 requirements:
1. Original operation commits before observer read (Scenario B: restraint).
2. Original operation is delayed and commits after observer initially sees absence.
3. A retry preserves the same logical operation identifier.
4. Duplicate delivery of the same operation cannot create duplicate primary effects when adapter is idempotent.
5. A stale attempt is rejected when fencing is implemented.
6. An adapter without a safe retry guarantee escalates on timeout instead of blindly retrying.
7. Existing commit-then-timeout behavior remains correct.
8. Existing observer-failure and UNKNOWN behavior remains correct.
9. Existing partial-write behavior remains safe.
10. Existing tests and demo scenarios remain intact.
"""

import sqlite3
import tempfile
import unittest
from typing import Dict, Any, Optional

from reagent.models import TaskState, Postcondition, AgentStatus, ActionResult
from reagent.pipeline import ReAgentPipeline
from reagent.policy import TokenBudgetGuard
from reagent.providers.base import LLMProvider, LLMResponse
from reagent.tools.base import BaseTool
from reagent.services.hotel_booking_service import HotelBookingServiceSimulator, HotelBookingDB
from reagent.services.hotel_demo import run_hotel_booking_demo
from reagent.tools.hotel_tool import HotelBookingHttpTool, make_hotel_booking_probe_callback


class MockDeterministicProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(content="{}", parsed_json={}, model_name="mock-provider")

    def diagnose_failure(self, failure: Any, execution_history: Optional[list] = None) -> Any:
        from reagent.models import FailureDiagnosis
        from reagent.taxonomy import FailureType
        return FailureDiagnosis(
            failure_id=failure.failure_id,
            failure_type=FailureType.TRANSIENT_TOOL_ERROR,
            root_cause="Transient error",
            confidence_score=0.9,
            is_agent_fault=False,
            is_transient=True,
            provenance="fixture",
        )

    def select_recovery(self, diagnosis: Any, available_tools: list, policy_rules: Optional[list] = None) -> Any:
        from reagent.models import RecoveryDecision
        from reagent.taxonomy import RecoveryStrategyType
        return RecoveryDecision(
            diagnosis_id=diagnosis.diagnosis_id,
            strategy=RecoveryStrategyType.RETRY,
            rationale="Retry under frozen contract",
            provenance="fixture",
        )


class UnfencedNonIdempotentTool(BaseTool):
    """A tool that does NOT support idempotency keys and has NO fencing."""
    name = "unfenced_append_tool"
    description = "Non-idempotent tool with no fencing"
    effect_class = "state-changing"
    supports_idempotency_key = False
    supports_fencing = False

    def __init__(self, failure_on_first: bool = True):
        self.call_count = 0
        self.failure_on_first = failure_on_first
        self.history = []

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        self.history.append(dict(params))
        if self.call_count == 1 and self.failure_on_first:
            raise TimeoutError("HTTPConnectionTimeout: Upstream socket timed out after 1.0s")
        return {"status": "APPENDED", "call": self.call_count}


class IdempotentFencedTool(BaseTool):
    """A tool that enforces stable operation identity and fencing."""
    name = "idempotent_fenced_tool"
    description = "Idempotent tool with fencing support"
    effect_class = "state-changing"
    supports_idempotency_key = True
    supports_fencing = True

    def __init__(self, db: HotelBookingDB, simulate_timeout_on_first: bool = True):
        self.db = db
        self.call_count = 0
        self.simulate_timeout_on_first = simulate_timeout_on_first
        self.attempt_params = []

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        self.attempt_params.append(dict(params))
        if self.call_count == 1 and self.simulate_timeout_on_first:
            raise TimeoutError("HTTPConnectionTimeout: dropped packet before response")
        return self.db.create_reservation(
            booking_id=params["booking_id"],
            guest_name=params.get("guest_name", "Alice"),
            room_number=params.get("room_number", "101"),
            amount=params.get("amount", 200.0),
            operation_id=params.get("operation_id"),
            fencing_token=params.get("fencing_token"),
        )


class TestLateWriteSafety(unittest.TestCase):

    def setUp(self):
        self.provider = MockDeterministicProvider()
        self.budget = TokenBudgetGuard(max_calls_per_run=10, max_tokens_per_run=20000)

    # ── 1. Original operation commits before observer read (Restraint) ─────────
    def test_01_original_commits_before_observer_read_restrained(self):
        """Scenario B: Original write commits in DB before timeout; observer confirms PRESENT -> Restrained."""
        sim = HotelBookingServiceSimulator()
        sim.start()
        try:
            artifact = run_hotel_booking_demo(scenario_id="B", simulator=sim)
            # ReAgent restrains retry, preventing duplicate reservation
            self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED")
            self.assertFalse(artifact.reagent_arm.duplicate_occurred)
            self.assertEqual(artifact.reagent_arm.tool_attempts, 1)
            self.assertEqual(sim.db.count_reservations(), 1)
        finally:
            sim.stop()

    # ── 2 & 4. Delayed write arriving after retry rejected by idempotency ─────
    def test_02_and_04_delayed_write_after_retry_rejected_by_idempotency(self):
        """
        Delayed original request arrives after retry has committed.
        The effect boundary (HotelBookingDB) enforces UNIQUE(operation_id) & PRIMARY KEY(booking_id).
        The delayed attempt is rejected as duplicate (IntegrityError), preserving exactly 1 effect.
        """
        db = HotelBookingDB()
        try:
            op_id = "op_stable_shared_123"
            bkg_id = "BKG-DELAY-001"

            # 1. Retry succeeds first
            res_retry = db.create_reservation(
                booking_id=bkg_id,
                guest_name="Maya Lin",
                room_number="14B",
                amount=350.0,
                operation_id=op_id,
            )
            self.assertEqual(db.count_reservations(), 1)

            # 2. Original delayed write arrives LATER with the same operation_id
            with self.assertRaises(sqlite3.IntegrityError):
                db.create_reservation(
                    booking_id=bkg_id,
                    guest_name="Maya Lin",
                    room_number="14B",
                    amount=350.0,
                    operation_id=op_id,
                )

            # Still exactly 1 primary record in database
            self.assertEqual(db.count_reservations(), 1)
        finally:
            db.close()

    # ── 3. A retry preserves the same logical operation identifier ─────────────
    def test_03_retry_preserves_logical_operation_identifier(self):
        """Pipeline must pass the same operation_id to the retry attempt."""
        db = HotelBookingDB()
        try:
            tool = IdempotentFencedTool(db, simulate_timeout_on_first=True)
            pipeline = ReAgentPipeline(agent_id="test_agent", provider=self.provider, token_budget=self.budget)
            pipeline.tool_registry.register(tool)

            booking_id = "BKG-OP-ID-PRESERVE"
            initial_op_id = "op_logical_transaction_42"

            def probe_cb(t_name, p, attempt):
                # Probe checks DB: first attempt timed out before write
                res = db.get_reservation(booking_id)
                if res is None:
                    return {"status": "CONFIRMED_ABSENT", "booking_id": booking_id, "fenced": True}
                return {"status": "PRESENT", "data": res}

            pcs = [
                Postcondition(predicate="STATUS_SUCCESS", description="must succeed"),
                Postcondition(predicate="KEY_EXISTS", target="booking_id", expected=booking_id),
            ]

            result = pipeline.execute_with_contract(
                task=TaskState(task_id="task_op_id", objective="Test op id preservation"),
                tool_name=tool.name,
                params={"booking_id": booking_id, "operation_id": initial_op_id},
                postconditions=pcs,
                max_recovery_attempts=1,
                probe_callback=probe_cb,
                supports_idempotency=True,
                supports_fencing=True,
            )

            self.assertEqual(result.verified_receipt.final_status, AgentStatus.VERIFIED)
            self.assertEqual(tool.call_count, 2)
            # Verify attempt 1 and attempt 2 used the EXACT same operation_id
            self.assertEqual(tool.attempt_params[0]["operation_id"], initial_op_id)
            self.assertEqual(tool.attempt_params[1]["operation_id"], initial_op_id)
        finally:
            db.close()

    # ── 5. Stale attempt is rejected when fencing is implemented ───────────────
    def test_05_stale_attempt_rejected_when_fencing_implemented(self):
        """
        When fencing is enabled, a newer retry commits with fencing_token=2.
        A delayed attempt arriving with stale fencing_token=1 is rejected at effect boundary.
        """
        db = HotelBookingDB()
        try:
            bkg_id = "BKG-FENCE-001"
            # Retry committed with generation 2
            db.create_reservation(
                booking_id=bkg_id,
                guest_name="Bob",
                room_number="202",
                amount=150.0,
                operation_id="op_retry",
                fencing_token=2,
            )
            self.assertEqual(db.count_reservations(), 1)

            # Stale original attempt arriving later with token 1 must raise IntegrityError
            with self.assertRaises(sqlite3.IntegrityError) as ctx:
                db.create_reservation(
                    booking_id=bkg_id,
                    guest_name="Bob",
                    room_number="202",
                    amount=150.0,
                    operation_id="op_stale_delayed",
                    fencing_token=1,  # <--- Stale! 1 <= 2
                )
            self.assertIn("StaleFencingToken", str(ctx.exception))
            # Primary row count remains exactly 1
            self.assertEqual(db.count_reservations(), 1)
        finally:
            db.close()

    # ── 6. Adapter without safe retry guarantee escalates on timeout ───────────
    def test_06_adapter_without_safe_retry_guarantee_escalates_on_timeout(self):
        """
        If a non-idempotent tool without write fencing times out,
        observing absence cannot prove the write will never arrive.
        ReAgent MUST fail closed to ESCALATED rather than retrying blindly.
        """
        tool = UnfencedNonIdempotentTool(failure_on_first=True)
        pipeline = ReAgentPipeline(agent_id="test_agent", provider=self.provider, token_budget=self.budget)
        pipeline.tool_registry.register(tool)

        # Observer reports in_flight_possible on an adapter without idempotency or fencing
        def probe_cb(t_name, p, attempt):
            return {"status": "CONFIRMED_ABSENT", "data": None, "in_flight_possible": True}

        pcs = [Postcondition(predicate="STATUS_SUCCESS", description="must succeed")]

        result = pipeline.execute_with_contract(
            task=TaskState(task_id="task_late_risk", objective="Test late write risk"),
            tool_name=tool.name,
            params={"data": "unfenced_payload"},
            postconditions=pcs,
            max_recovery_attempts=1,
            probe_callback=probe_cb,
            supports_idempotency=False,
            supports_fencing=False,
        )

        # Invariant enforced: ESCALATED, exactly 1 tool execution, no blind retry
        self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.call_count, 1)
        self.assertIn("LateWriteRiskProhibitsRetry", result.action_result.error)

    # ── 7. Existing commit-then-timeout behavior remains correct ───────────────
    def test_07_existing_commit_then_timeout_behavior_correct(self):
        """Commit-then-timeout in multi-scenario hotel demo preserves single reservation."""
        sim = HotelBookingServiceSimulator()
        sim.start()
        try:
            artifact = run_hotel_booking_demo(scenario_id="B", simulator=sim)
            self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED")
            self.assertFalse(artifact.reagent_arm.duplicate_occurred)
            self.assertEqual(sim.db.count_reservations(), 1)
        finally:
            sim.stop()

    # ── 8. Existing observer-failure and UNKNOWN behavior remains correct ──────
    def test_08_existing_observer_failure_and_unknown_behavior_correct(self):
        """Scenario D: Observer returns 503 Service Unavailable -> ESCALATED without guessing."""
        sim = HotelBookingServiceSimulator()
        sim.start()
        try:
            artifact = run_hotel_booking_demo(scenario_id="D", simulator=sim)
            self.assertEqual(artifact.reagent_arm.final_status, "ESCALATED")
            self.assertFalse(artifact.reagent_arm.duplicate_occurred)
        finally:
            sim.stop()

    # ── 9. Existing partial-write behavior remains safe ────────────────────────
    def test_09_existing_partial_write_behavior_safe(self):
        """Scenario E: Split-brain partial commit detected as discrepancy -> ESCALATED."""
        sim = HotelBookingServiceSimulator()
        sim.start()
        try:
            artifact = run_hotel_booking_demo(scenario_id="E", simulator=sim)
            self.assertEqual(artifact.reagent_arm.final_status, "ESCALATED")
            self.assertFalse(artifact.reagent_arm.contract_postconditions_verified)
        finally:
            sim.stop()

    # ── 10. Existing tests and demo scenarios remain intact ────────────────────
    def test_10_all_flagship_scenarios_execute_cleanly(self):
        """All scenarios A through E execute with expected deterministic outcomes."""
        sim = HotelBookingServiceSimulator()
        sim.start()
        try:
            for s in ("A", "B", "C", "D", "E"):
                artifact = run_hotel_booking_demo(scenario_id=s, simulator=sim)
                self.assertIsNotNone(artifact.run_id)
                self.assertIsNotNone(artifact.reagent_arm)
                self.assertIsNotNone(artifact.naive_retry_arm)
                if s in ("A", "B", "C"):
                    self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED")
                else:
                    self.assertEqual(artifact.reagent_arm.final_status, "ESCALATED")
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
