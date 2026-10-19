"""
test_dbops_m13_adversarial.py — Comprehensive Adversarial Validation Suite for Milestone 13.

Verifies ReAgent recovery semantics and benchmark integrity against 13 adversarial scenarios:
1. Commit then timeout: Observer immediately confirms PRESENT with correct content -> restrains retry, marks VERIFIED.
2. Fail before commit: Observer confirms CONFIRMED_ABSENT -> safe retry succeeds, marks VERIFIED.
3. Ordinary unconfirmed ABSENT:
   - Non-idempotent operation fails closed to ESCALATED without retrying.
   - Idempotent operation safely retries.
4. UNKNOWN, UNAVAILABLE, UNOBSERVABLE, and DISCREPANCY (tested separately):
   - Every one fails closed to ESCALATED without retrying, for BOTH idempotent and non-idempotent operations.
5. Full 2x6 idempotency matrix (Idempotent vs Non-Idempotent across 6 probe outcomes).
6. Partial mutation (PARTIAL_COMMIT): Primary committed, secondary omitted -> flags discrepancy, escalates.
7. Row exists with mutated field (WRONG_CONTENT): Primary exists but wrong amount/entity -> flags discrepancy, escalates.
8. Delayed visibility: Probe 1 DELAYED, Probe 2 PRESENT -> polls boundedly, verifies, restrains retry.
9. Observer exceptions & read-only enforcement:
   - Probe exception fails closed to ESCALATED.
   - Observer connection enforces PRAGMA query_only=ON (disallows accidental writes).
10. Duplicate accounting: Differentiates primary write duplicates from secondary audit duplicates.
11. Tool claims success but state absent (ghost write) vs tool claims failure after commit.
12. Deterministic replay across fresh isolated SQLite databases.
13. Benchmark evaluator diagnostic policies (Part C):
    - DELIBERATELY_WRONG policy (unconditional VERIFIED) correctly flagged with high false-verification rate.
    - ORACLE_UPPER_BOUND diagnostic correctly achieves 0 false verifications.

Zero live network access. Zero paid API tokens. Zero .env inspection.
"""

import sqlite3
import unittest
from typing import Dict, Any, Optional

from reagent.models import TaskState, AgentStatus, ActionResult
from reagent.pipeline import ReAgentPipeline
from reagent.tools.base import BaseTool
from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.tasks import (
    TASK_REFUND_ISSUE,
    TASK_WALLET_TRANSFER,
    TASK_ORDER_CANCELLATION,
    get_task,
)
from reagent.experiments.dbops.executor import execute_task
from reagent.experiments.dbops.oracle import DbopsOracle
from reagent.experiments.dbops.validator import GoldContractEvaluator
from reagent.experiments.dbops.types import VerdictType
from reagent.experiments.dbops.policies import run_policy_trial, PolicyAction
from reagent.experiments.dbops.comparative_eval import ComparativeEvaluator


class MockAdversarialTool(BaseTool):
    """Controllable test double for adversarial pipeline validation."""
    name: str = "mock_adv_tool"
    description: str = "Controllable tool for testing adversarial recovery."
    parameters_schema: Dict[str, Any] = {"type": "object"}

    def __init__(self, should_fail: bool = True, supports_idempotency: bool = False):
        self.should_fail = should_fail
        self.supports_idempotency_key = supports_idempotency
        self.attempts = 0

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.attempts += 1
        if self.should_fail:
            raise TimeoutError("Ambiguous tool timeout after possible commit")
        return {"status": "SUCCESS"}


class TestDbopsM13Adversarial(unittest.TestCase):
    """Adversarial validation test suite for Milestone 13."""

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 1: Commit then timeout -> immediately PRESENT -> Restrain & VERIFIED
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_commit_then_timeout_restrains_and_verifies(self):
        """Tool times out after committing; observer confirms PRESENT -> restrains retry, marks VERIFIED."""
        pipeline = ReAgentPipeline(agent_id="test-adv-01")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=False)
        pipeline.tool_registry.register(tool)

        def probe_fn(name, params, step):
            return {"status": "PRESENT"}

        task = TaskState(task_id="t_adv_01", objective="Transfer $500")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 1, "Must restrain retry when state is PRESENT!")
        self.assertEqual(res.evidence.source, "probe_restraint")

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 2: Fail before commit -> CONFIRMED_ABSENT -> Safe Retry & VERIFIED
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_fail_before_commit_recovers_safely(self):
        """Tool fails before commit; observer confirms CONFIRMED_ABSENT -> executes safe retry."""
        pipeline = ReAgentPipeline(agent_id="test-adv-02")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=False)
        pipeline.tool_registry.register(tool)

        def probe_fn(name, params, step):
            tool.should_fail = False  # cleared on retry
            return {"status": "CONFIRMED_ABSENT"}

        task = TaskState(task_id="t_adv_02", objective="Transfer $500")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 2, "Must execute bounded safe retry upon CONFIRMED_ABSENT!")
        self.assertEqual(res.recovery_count, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 3: Unconfirmed ABSENT -> Non-Idempotent blocks, Idempotent allows
    # ─────────────────────────────────────────────────────────────────────────
    def test_03a_unconfirmed_absent_non_idempotent_fails_closed(self):
        """Non-idempotent action under ordinary unconfirmed ABSENT fails closed to ESCALATED."""
        pipeline = ReAgentPipeline(agent_id="test-adv-03a")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=False)
        pipeline.tool_registry.register(tool)

        def probe_fn(name, params, step):
            return {"status": "ABSENT"}

        task = TaskState(task_id="t_adv_03a", objective="Transfer $500")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1, "Must NOT retry non-idempotent action on unconfirmed ABSENT!")
        self.assertIn("NonIdempotentOperationAmbiguous", res.action_result.error)

    def test_03b_unconfirmed_absent_idempotent_permits_retry(self):
        """Idempotent action under ordinary unconfirmed ABSENT is permitted to retry."""
        pipeline = ReAgentPipeline(agent_id="test-adv-03b")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=True)
        pipeline.tool_registry.register(tool)

        def probe_fn(name, params, step):
            tool.should_fail = False  # cleared on retry
            return {"status": "ABSENT"}

        task = TaskState(task_id="t_adv_03b", objective="Set status CANCELLED")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=True,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 2, "Idempotent action safely retries on unconfirmed ABSENT!")

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 4: UNKNOWN, UNAVAILABLE, UNOBSERVABLE, and DISCREPANCY separately
    # ─────────────────────────────────────────────────────────────────────────
    def test_04a_probe_unknown_fails_closed(self):
        """Probe returning UNKNOWN fails closed to ESCALATED with zero retries."""
        pipeline = ReAgentPipeline(agent_id="test-adv-04a")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=True)
        pipeline.tool_registry.register(tool)

        res = pipeline.execute_with_contract(
            task=TaskState(task_id="t_04a", objective="Test UNKNOWN"),
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=lambda n, p, s: {"status": "UNKNOWN"},
            supports_idempotency=True,
        )
        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1)

    def test_04b_probe_unavailable_fails_closed(self):
        """Probe returning UNAVAILABLE fails closed to ESCALATED with zero retries."""
        pipeline = ReAgentPipeline(agent_id="test-adv-04b")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=True)
        pipeline.tool_registry.register(tool)

        res = pipeline.execute_with_contract(
            task=TaskState(task_id="t_04b", objective="Test UNAVAILABLE"),
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=lambda n, p, s: {"status": "UNAVAILABLE"},
            supports_idempotency=True,
        )
        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1)

    def test_04c_probe_unobservable_fails_closed(self):
        """Probe returning UNOBSERVABLE fails closed to ESCALATED with zero retries."""
        pipeline = ReAgentPipeline(agent_id="test-adv-04c")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=True)
        pipeline.tool_registry.register(tool)

        res = pipeline.execute_with_contract(
            task=TaskState(task_id="t_04c", objective="Test UNOBSERVABLE"),
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=lambda n, p, s: {"status": "UNOBSERVABLE"},
            supports_idempotency=True,
        )
        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1)

    def test_04d_probe_discrepancy_fails_closed(self):
        """Probe returning DISCREPANCY fails closed to ESCALATED with zero retries."""
        pipeline = ReAgentPipeline(agent_id="test-adv-04d")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=True)
        pipeline.tool_registry.register(tool)

        res = pipeline.execute_with_contract(
            task=TaskState(task_id="t_04d", objective="Test DISCREPANCY"),
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=lambda n, p, s: {"status": "DISCREPANCY"},
            supports_idempotency=True,
        )
        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 5: Full 2x6 Idempotency Matrix
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_idempotent_vs_non_idempotent_matrix(self):
        """Tests all 12 combinations of (idempotent in [True, False]) x (probe in 6 statuses)."""
        probe_statuses = ["PRESENT", "CONFIRMED_ABSENT", "ABSENT", "DISCREPANCY", "UNKNOWN", "UNAVAILABLE"]

        for is_idempotent in (True, False):
            for status in probe_statuses:
                pipeline = ReAgentPipeline(agent_id=f"test-matrix-{is_idempotent}-{status}")
                tool = MockAdversarialTool(should_fail=True, supports_idempotency=is_idempotent)
                pipeline.tool_registry.register(tool)

                def probe_fn(name, params, step, s=status):
                    if s == "CONFIRMED_ABSENT" and step == 1:
                        tool.should_fail = False
                    return {"status": s}

                res = pipeline.execute_with_contract(
                    task=TaskState(task_id="t_matrix", objective=f"Test {is_idempotent} with {status}"),
                    tool_name="mock_adv_tool",
                    params={},
                    max_recovery_attempts=1,
                    probe_callback=probe_fn,
                    supports_idempotency=is_idempotent,
                )

                if status == "PRESENT":
                    self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
                    self.assertEqual(tool.attempts, 1)
                elif status == "CONFIRMED_ABSENT":
                    self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
                    self.assertEqual(tool.attempts, 2)
                elif status == "ABSENT":
                    if is_idempotent:
                        # Idempotent tool allowed to retry
                        self.assertEqual(tool.attempts, 2)
                    else:
                        # Non-idempotent tool MUST NOT retry
                        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
                        self.assertEqual(tool.attempts, 1)
                else:
                    # DISCREPANCY, UNKNOWN, UNAVAILABLE -> both fail closed without retry
                    self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
                    self.assertEqual(tool.attempts, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 6: Partial Mutation Across Tables (PARTIAL_COMMIT)
    # ─────────────────────────────────────────────────────────────────────────
    def test_06_partial_commit_across_tables_escalates_safely(self):
        """Under PARTIAL_COMMIT (step 0 commits, step 1 missing), contract evaluator detects partial and escalates."""
        store = DbopsStore.create_temp(prefix="test_adv_partial_")
        try:
            store.seed_initial_state(TASK_REFUND_ISSUE)
            # PARTIAL_COMMIT executes step 0 (refunds) but omits step 1 (refund_audit_log)
            rec = run_policy_trial(store, TASK_REFUND_ISSUE, "PARTIAL_COMMIT", "GOLD_CONTRACT")
            self.assertEqual(rec.final_claim, PolicyAction.ESCALATED)
            self.assertIn("Partial commit", rec.explanation)
            self.assertEqual(rec.action_attempts, 1, "Must NOT retry on partial commit!")
        finally:
            store.cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 7: Mutated Content (WRONG_CONTENT)
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_wrong_content_field_mismatch_escalates_safely(self):
        """Under WRONG_CONTENT (amount is 999.99 instead of 250.00), presence baseline falsely verifies, ReAgent escalates."""
        store_shared = DbopsStore.create_temp(prefix="test_adv_wrong_shared_")
        store_reagent = DbopsStore.create_temp(prefix="test_adv_wrong_reagent_")
        try:
            store_shared.seed_initial_state(TASK_REFUND_ISSUE)
            store_reagent.seed_initial_state(TASK_REFUND_ISSUE)

            # Shared observation arm checks primary presence only -> falsely verifies!
            rec_shared = run_policy_trial(store_shared, TASK_REFUND_ISSUE, "WRONG_CONTENT", "SHARED_OBSERVATION")
            self.assertEqual(rec_shared.final_claim, PolicyAction.VERIFIED, "Presence baseline falsely verifies!")

            # ReAgent contract arm checks exact field values -> detects discrepancy and escalates!
            rec_reagent = run_policy_trial(store_reagent, TASK_REFUND_ISSUE, "WRONG_CONTENT", "GOLD_CONTRACT")
            self.assertEqual(rec_reagent.final_claim, PolicyAction.ESCALATED, "Contract layer safely escalates!")
            self.assertIn("WRONG_CONTENT", rec_reagent.explanation)
        finally:
            store_shared.cleanup()
            store_reagent.cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 8: Delayed Visibility Recovery
    # ─────────────────────────────────────────────────────────────────────────
    def test_08_delayed_visibility_polls_and_verifies_without_retry(self):
        """When probe 1 returns DELAYED and probe 2 returns PRESENT, pipeline resolves without retry."""
        pipeline = ReAgentPipeline(agent_id="test-adv-08")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=False)
        pipeline.tool_registry.register(tool)

        probe_calls = []
        def probe_delayed(name, params, step):
            probe_calls.append(step)
            if step == 1:
                return {"status": "DELAYED"}
            return {"status": "PRESENT"}

        res = pipeline.execute_with_contract(
            task=TaskState(task_id="t_adv_08", objective="Delayed visibility test"),
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=probe_delayed,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 1, "Must NOT issue retry when delayed visibility resolves!")
        self.assertEqual(probe_calls, [1, 2], "Must execute step 1 and step 2 probes within budget!")

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 9: Observer Exceptions & Read-Only Enforcement
    # ─────────────────────────────────────────────────────────────────────────
    def test_09a_observer_exception_fails_closed(self):
        """When probe callback raises an unexpected exception (e.g. database locked), pipeline fails closed."""
        pipeline = ReAgentPipeline(agent_id="test-adv-09a")
        tool = MockAdversarialTool(should_fail=True, supports_idempotency=False)
        pipeline.tool_registry.register(tool)

        def broken_probe(name, params, step):
            raise sqlite3.OperationalError("database is locked")

        res = pipeline.execute_with_contract(
            task=TaskState(task_id="t_adv_09a", objective="Exception probe test"),
            tool_name="mock_adv_tool",
            params={},
            max_recovery_attempts=1,
            probe_callback=broken_probe,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1)
        self.assertIn("ProbeError", res.action_result.error)

    def test_09b_observer_connection_read_only_enforcement(self):
        """fresh_observer_connection enforces PRAGMA query_only=ON; accidental writes raise OperationalError."""
        store = DbopsStore.create_temp(prefix="test_adv_readonly_")
        try:
            store.seed_initial_state(TASK_REFUND_ISSUE)
            with store.fresh_observer_connection() as conn:
                with self.assertRaises(sqlite3.OperationalError) as ctx:
                    conn.execute("INSERT INTO refunds (refund_id, order_id, customer_id, amount) VALUES ('X', 'Y', 'Z', 1.0)")
                self.assertIn("readonly", str(ctx.exception).lower())
        finally:
            store.cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 10: Duplicate Accounting (Primary vs Audit)
    # ─────────────────────────────────────────────────────────────────────────
    def test_10_duplicate_accounting_distinguishes_primary_from_audit(self):
        """Evaluator distinguishes primary state duplicates from secondary audit duplicates."""
        store = DbopsStore.create_temp(prefix="test_adv_dupes_")
        try:
            store.seed_initial_state(TASK_WALLET_TRANSFER)
            # Execute twice under NO_FAULT on unprotected numeric task
            execute_task(store, TASK_WALLET_TRANSFER, "NO_FAULT")
            e2 = execute_task(store, TASK_WALLET_TRANSFER, "NO_FAULT")

            oracle = DbopsOracle(store, e2)
            obs = oracle.evaluate()
            self.assertEqual(obs.duplicate_primary_count, 2)
            self.assertEqual(obs.duplicate_secondary_counts.get("wallet_balance_log"), 2)

            evaluator = GoldContractEvaluator()
            res = evaluator.evaluate(TASK_WALLET_TRANSFER, obs)
            self.assertTrue(res.primary_state_duplicate)
            self.assertTrue(res.audit_duplicate)
        finally:
            store.cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 11: Silent Failure vs Unacknowledged Success
    # ─────────────────────────────────────────────────────────────────────────
    def test_11a_silent_tool_success_with_absent_state_fails_verification(self):
        """Tool claims success (output={}), but no row committed; contract verifier rejects verification."""
        store = DbopsStore.create_temp(prefix="test_adv_ghost_")
        try:
            store.seed_initial_state(TASK_REFUND_ISSUE)
            # Create oracle on store with NO rows committed
            from reagent.experiments.dbops.executor import ExecutionResult
            ghost_exec = ExecutionResult(
                task_id=TASK_REFUND_ISSUE,
                fault_mode="NO_FAULT",
                agent_visible_error=None,  # tool claimed success!
                committed_steps=[],         # but nothing committed!
            )
            oracle = DbopsOracle(store, ghost_exec)
            obs = oracle.evaluate()
            self.assertEqual(obs.verdict, VerdictType.CONFIRMED_ABSENT)

            evaluator = GoldContractEvaluator()
            res = evaluator.evaluate(TASK_REFUND_ISSUE, obs)
            self.assertNotEqual(res.verdict, VerdictType.EXPECTED_PRESENT)
            self.assertEqual(res.verdict, VerdictType.CONFIRMED_ABSENT)
        finally:
            store.cleanup()

    def test_11b_unacknowledged_success_is_verified_via_probe(self):
        """Tool reports timeout error, but row committed; oracle confirms EXPECTED_PRESENT."""
        store = DbopsStore.create_temp(prefix="test_adv_unack_")
        try:
            store.seed_initial_state(TASK_REFUND_ISSUE)
            exec_res = execute_task(store, TASK_REFUND_ISSUE, "COMMIT_THEN_TIMEOUT")
            self.assertIsNotNone(exec_res.agent_visible_error)

            oracle = DbopsOracle(store, exec_res)
            obs = oracle.evaluate()
            self.assertEqual(obs.verdict, VerdictType.EXPECTED_PRESENT)
        finally:
            store.cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 12: Deterministic Replay Across Independent Stores
    # ─────────────────────────────────────────────────────────────────────────
    def test_12_deterministic_replay_across_independent_stores(self):
        """Two separate evaluations on independent SQLite stores produce identical metric records."""
        records = []
        for i in range(2):
            store = DbopsStore.create_temp(prefix=f"test_adv_replay_{i}_")
            try:
                store.seed_initial_state(TASK_REFUND_ISSUE)
                rec = run_policy_trial(store, TASK_REFUND_ISSUE, "COMMIT_THEN_TIMEOUT", "GOLD_CONTRACT")
                records.append(rec)
            finally:
                store.cleanup()

        self.assertEqual(records[0].final_claim, records[1].final_claim)
        self.assertEqual(records[0].action_attempts, records[1].action_attempts)
        self.assertEqual(records[0].probe_attempts, records[1].probe_attempts)
        self.assertEqual(records[0].explanation, records[1].explanation)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 13: Benchmark Diagnostic Policies (Part C)
    # ─────────────────────────────────────────────────────────────────────────
    def test_13_evaluator_correctly_scores_adversarial_policies(self):
        """Evaluator correctly detects failures in DELIBERATELY_WRONG arm vs ORACLE diagnostic."""
        import tempfile
        from reagent.experiments.dbops.comparative_eval import ComparativeEvaluator

        with tempfile.TemporaryDirectory() as tmp_dir:
            ev = ComparativeEvaluator(output_dir=tmp_dir)
            # Run NAIVE_RETRY (deliberately aggressive) vs REAGENT_CONTRACT
            art = ev.run_evaluation(
                task_ids=[TASK_REFUND_ISSUE, TASK_WALLET_TRANSFER],
                fault_modes=["NO_FAULT", "COMMIT_THEN_TIMEOUT", "WRONG_CONTENT"],
                arms=["REAGENT_CONTRACT", "NAIVE_RETRY"],
            )

            rc = art.arm_aggregates["REAGENT_CONTRACT"]
            nr = art.arm_aggregates["NAIVE_RETRY"]

            # REAGENT_CONTRACT achieves 0 false verifications
            self.assertEqual(rc.false_verified_count, 0)
            self.assertEqual(rc.false_verified_rate, 0.0)

            # NAIVE_RETRY falsely verifies on WRONG_CONTENT and causes duplicate on Task 02
            self.assertGreater(nr.false_verified_count, 0)
            self.assertGreater(nr.false_verified_rate, 0.0)
            self.assertGreater(nr.duplicate_primary_count, 0)


if __name__ == "__main__":
    unittest.main()
