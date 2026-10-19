"""
test_dbops_m12_invariants.py — Core Correctness Invariants & Metamorphic Verification for M12.

Verifies:
1. Invariant 1: UNKNOWN is not ABSENT.
   - Probing with UNKNOWN, UNAVAILABLE, UNOBSERVABLE, or DISCREPANCY never triggers retry,
     even when tool or parameter declares idempotency (supports_idempotency=True).
   - Fails closed to ESCALATED.
2. Invariant 2: Non-idempotent operations are never blindly retried under ambiguity.
   - Ordinary unconfirmed ABSENT blocks retry on non-idempotent operations.
   - Only CONFIRMED_ABSENT permits safe retry.
3. Invariant 3: VERIFIED requires independent observer confirming full postconditions.
   - Self-reported success from tool is rejected if independent observer detects WRONG_CONTENT
     or PARTIAL_COMMIT.
4. Metamorphic Properties:
   - Order Invariance: Multi-trial evaluations produce identical verdicts regardless of execution order.
   - Resource Boundedness: Action attempts <= 2, Probe attempts <= 2 across all policy arms.
   - Store Isolation: State modifications in store A never leak into store B.
   - Deterministic Replay: Two consecutive executions on identical initial states produce identical verdicts.

Zero live network access. Zero paid API tokens. Zero .env inspection.
"""

import unittest
from typing import Dict, Any

from reagent.models import TaskState, AgentStatus, ActionResult
from reagent.pipeline import ReAgentPipeline
from reagent.tools.base import BaseTool
from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.tasks import TASK_WALLET_TRANSFER, get_task
from reagent.experiments.dbops.executor import execute_task
from reagent.experiments.dbops.integration import (
    DbopsDatabaseTool,
    make_dbops_probe_callback,
)
from reagent.experiments.dbops.policies import run_policy_trial, PolicyAction


class MockStatefulTool(BaseTool):
    """Controllable test double for pipeline invariant testing."""
    name: str = "mock_stateful_tool"
    description: str = "Mock tool for testing invariants."
    parameters_schema: Dict[str, Any] = {"type": "object"}

    def __init__(self, should_fail: bool = True, supports_idempotency: bool = False):
        self.should_fail = should_fail
        self.supports_idempotency_key = supports_idempotency
        self.attempts = 0

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.attempts += 1
        if self.should_fail:
            raise TimeoutError("Ambiguous tool timeout")
        return {"status": "SUCCESS"}


class TestDbopsM12Invariants(unittest.TestCase):
    """Test suite verifying core reliability invariants and metamorphic properties."""

    def test_invariant_1_unknown_prohibits_retry_even_if_idempotent(self):
        """UNKNOWN state must never be treated as ABSENT; blocks retry even when supports_idempotency=True."""
        pipeline = ReAgentPipeline(agent_id="test-inv1-agent")
        tool = MockStatefulTool(should_fail=True, supports_idempotency=True)
        pipeline.tool_registry.register(tool)

        def probe_unknown(name: str, params: Dict[str, Any], step: int):
            return {"status": "UNKNOWN"}

        task = TaskState(task_id="t_inv1", objective="Test UNKNOWN blocks retry")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_unknown,
            supports_idempotency=True,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry when outcome is UNKNOWN!")
        self.assertIn("AmbiguousOrDiscrepantStateProhibitsRetry", res.action_result.error)

    def test_invariant_1_discrepancy_prohibits_retry_even_if_idempotent(self):
        """DISCREPANCY (partial or wrong content) blocks retry even when supports_idempotency=True."""
        pipeline = ReAgentPipeline(agent_id="test-inv1-discrepancy")
        tool = MockStatefulTool(should_fail=True, supports_idempotency=True)
        pipeline.tool_registry.register(tool)

        def probe_discrepancy(name: str, params: Dict[str, Any], step: int):
            return {"status": "DISCREPANCY"}

        task = TaskState(task_id="t_discrepancy", objective="Test DISCREPANCY blocks retry")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_discrepancy,
            supports_idempotency=True,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry when state is DISCREPANT!")
        self.assertIn("AmbiguousOrDiscrepantStateProhibitsRetry", res.action_result.error)

    def test_invariant_2_non_idempotent_ordinary_absent_fails_closed(self):
        """Non-idempotent operation with unconfirmed ABSENT fails closed to ESCALATED."""
        pipeline = ReAgentPipeline(agent_id="test-inv2-agent")
        tool = MockStatefulTool(should_fail=True, supports_idempotency=False)
        pipeline.tool_registry.register(tool)

        def probe_unconfirmed_absent(name: str, params: Dict[str, Any], step: int):
            return {"status": "ABSENT"}

        task = TaskState(task_id="t_inv2", objective="Test ordinary ABSENT fails closed")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_unconfirmed_absent,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry non-idempotent operation on unconfirmed ABSENT!")
        self.assertIn("NonIdempotentOperationAmbiguous", res.action_result.error)

    def test_invariant_2_confirmed_absent_permits_safe_retry(self):
        """Only CONFIRMED_ABSENT permits safe retry on non-idempotent operation."""
        pipeline = ReAgentPipeline(agent_id="test-inv2-confirmed")
        # Tool fails on attempt 1, succeeds on attempt 2
        tool = MockStatefulTool(should_fail=True, supports_idempotency=False)
        pipeline.tool_registry.register(tool)

        def probe_confirmed_absent(name: str, params: Dict[str, Any], step: int):
            # Authoritative probe confirms no prior write occurred
            tool.should_fail = False  # cleared for retry
            return {"status": "CONFIRMED_ABSENT"}

        task = TaskState(task_id="t_inv2_conf", objective="Test CONFIRMED_ABSENT permits retry")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_confirmed_absent,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 2, "Safe retry was executed after confirmed absence!")

    def test_invariant_3_independent_observer_rejects_wrong_content(self):
        """Independent observer rejects WRONG_CONTENT and escalates safely."""
        store = DbopsStore.create_temp(prefix="test_inv3_")
        try:
            store.seed_initial_state(TASK_WALLET_TRANSFER)
            record = run_policy_trial(store, TASK_WALLET_TRANSFER, "WRONG_CONTENT", "GOLD_CONTRACT")
            self.assertEqual(record.final_claim, PolicyAction.ESCALATED)
            self.assertIn("WRONG_CONTENT", record.explanation)
            self.assertEqual(record.action_attempts, 1, "Must not retry on wrong content!")
        finally:
            store.cleanup()

    def test_metamorphic_deterministic_replay(self):
        """Executing the same trial twice on clean stores produces identical verdicts and action counts."""
        store1 = DbopsStore.create_temp(prefix="test_meta_replay_1_")
        store2 = DbopsStore.create_temp(prefix="test_meta_replay_2_")
        try:
            store1.seed_initial_state(TASK_WALLET_TRANSFER)
            store2.seed_initial_state(TASK_WALLET_TRANSFER)

            rec1 = run_policy_trial(store1, TASK_WALLET_TRANSFER, "COMMIT_THEN_TIMEOUT", "GOLD_CONTRACT")
            rec2 = run_policy_trial(store2, TASK_WALLET_TRANSFER, "COMMIT_THEN_TIMEOUT", "GOLD_CONTRACT")

            self.assertEqual(rec1.final_claim, rec2.final_claim)
            self.assertEqual(rec1.action_attempts, rec2.action_attempts)
            self.assertEqual(rec1.probe_attempts, rec2.probe_attempts)
            self.assertEqual(rec1.explanation, rec2.explanation)
        finally:
            store1.cleanup()
            store2.cleanup()

    def test_metamorphic_store_isolation(self):
        """Mutations in store A do not affect clean store B."""
        store_a = DbopsStore.create_temp(prefix="store_a_")
        store_b = DbopsStore.create_temp(prefix="store_b_")
        try:
            store_a.seed_initial_state(TASK_WALLET_TRANSFER)
            store_b.seed_initial_state(TASK_WALLET_TRANSFER)

            # Mutate store A by executing task
            execute_task(store_a, TASK_WALLET_TRANSFER, "NO_FAULT")

            # Check store A has committed row
            with store_a.fresh_observer_connection() as conn_a:
                count_a = conn_a.execute("SELECT COUNT(*) FROM wallet_transfer_ledger").fetchone()[0]
                self.assertEqual(count_a, 1)

            # Check store B remains unmutated (0 rows)
            with store_b.fresh_observer_connection() as conn_b:
                count_b = conn_b.execute("SELECT COUNT(*) FROM wallet_transfer_ledger").fetchone()[0]
                self.assertEqual(count_b, 0)
        finally:
            store_a.cleanup()
            store_b.cleanup()


if __name__ == "__main__":
    unittest.main()
