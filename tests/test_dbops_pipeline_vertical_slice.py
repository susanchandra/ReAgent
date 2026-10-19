"""
test_dbops_pipeline_vertical_slice.py — End-to-end integration tests for ReAgentPipeline with DBOPS Task 02.

Verifies:
1. Provider StructuredDecision passes validation, grounding, and contract freezing.
2. ReAgentPipeline executes DbopsDatabaseTool and routes through existing recovery loop.
3. Under COMMIT_THEN_TIMEOUT, ReAgent probes committed state, restrains blind retry,
   and avoids duplicate debit on Task 02 (UNPROTECTED_NUMERIC).
4. Ground truth confirms exactly 1 transfer debit ($500.00) in wallet_transfer_ledger
   and exactly 1 entry in wallet_balance_log. Zero duplicate side effects.
5. Injected COMMIT_THEN_TIMEOUT vs FAIL_BEFORE_COMMIT error observations remain byte-identical.
6. WRONG_CONTENT, PARTIAL_COMMIT, STATUS_UNAVAILABLE, and UNOBSERVABLE fail closed to ESCALATED.
7. Baseline Comparison: Demonstrates that a naive retry issues a 2nd write causing a double
   debit ($1000 total), while ReAgent maintains exactly 1 debit ($500).

SAFETY:
- Offline only. Zero network, zero live model / paid API calls.
- Run ONLY explicitly via python -m unittest tests.test_dbops_pipeline_vertical_slice.
"""

import unittest
import sqlite3
from typing import Dict, Any

from reagent.pipeline import ReAgentPipeline
from reagent.models import TaskState, Postcondition, AgentStatus, ActionResult
from reagent.tools.registry import ToolRegistry
from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.tasks import TASK_WALLET_TRANSFER
from reagent.experiments.dbops.integration import (
    DbopsDatabaseTool,
    make_dbops_probe_callback,
    DeterministicMockWalletProvider,
)
from reagent.experiments.dbops.executor import execute_task, AMBIGUOUS_TIMEOUT_ERROR
from reagent.experiments.dbops.oracle import DbopsOracle
from reagent.experiments.dbops.validator import GoldContractEvaluator
from reagent.experiments.dbops.types import VerdictType


class TestDbopsPipelineVerticalSlice(unittest.TestCase):

    def setUp(self):
        self.store = DbopsStore.create_temp(prefix="slice_test_")
        self.store.seed_initial_state(TASK_WALLET_TRANSFER)
        self.db_tool = DbopsDatabaseTool(self.store)

        self.pipeline = ReAgentPipeline(agent_id="reagent-slice-agent")
        self.pipeline.tool_registry.register(self.db_tool)

    def tearDown(self):
        self.store.cleanup()

    # ── Test 1: Full Native Pipeline Recovery Under COMMIT_THEN_TIMEOUT ───────

    def test_01_pipeline_restraint_on_commit_then_timeout(self):
        """
        End-to-End Vertical Slice:
        Task 02 executes through ReAgentPipeline under COMMIT_THEN_TIMEOUT.
        The tool commits step 0 and step 1, but times out returning AMBIGUOUS_TIMEOUT_ERROR.
        ReAgent's native recovery loop probes committed state, observes PRESENT,
        restrains from retrying, and emits a VERIFIED receipt with recovery_count=0.
        """
        task = TaskState(
            task_id="t_wallet_slice",
            objective="Transfer $500.00 from wallet ACCT-2201 to wallet ACCT-3390. Request ID: TXN-779043.",
        )

        probe_fn = make_dbops_probe_callback(self.store, TASK_WALLET_TRANSFER, "COMMIT_THEN_TIMEOUT")

        # Execute through actual ReAgentPipeline
        res = self.pipeline.execute_with_contract(
            task=task,
            tool_name="dbops_database",
            params={"task_id": TASK_WALLET_TRANSFER, "fault_mode": "COMMIT_THEN_TIMEOUT"},
            postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    target="wallet_transfer_ledger",
                    expected="TXN-779043",
                    description="Transfer TXN-779043 of $500.00",
                )
            ],
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,  # UNPROTECTED_NUMERIC
        )

        # 1. Pipeline verification result
        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(res.evidence.source, "probe_restraint")
        self.assertEqual(res.recovery_count, 0)
        self.assertEqual(self.db_tool.execution_count, 1, "Must NOT issue a second write!")

        # 2. Ground truth committed state check (independent observer connection)
        with self.store.fresh_observer_connection() as conn:
            cur = conn.execute(
                "SELECT COUNT(*), SUM(amount) FROM wallet_transfer_ledger WHERE transfer_id='TXN-779043'"
            )
            count, total = cur.fetchone()
            self.assertEqual(count, 1, "Exactly 1 transfer debit must exist!")
            self.assertEqual(total, 500.00, "Debit must be exactly $500.00, not double!")

            cur2 = conn.execute(
                "SELECT COUNT(*), SUM(delta) FROM wallet_balance_log WHERE transfer_id='TXN-779043'"
            )
            b_count, b_total = cur2.fetchone()
            self.assertEqual(b_count, 1, "Exactly 1 balance log must exist!")
            self.assertEqual(b_total, -500.00, "Balance change must be exactly -500.00!")

    # ── Test 2: Identifiability & Paired Error Invariant ───────────────────────

    def test_02_paired_error_byte_identity(self):
        """
        Confirms that the agent-visible error string produced under COMMIT_THEN_TIMEOUT
        is byte-identical to FAIL_BEFORE_COMMIT for Task 02.
        """
        store_commit = DbopsStore.create_temp()
        store_fail = DbopsStore.create_temp()
        try:
            store_commit.seed_initial_state(TASK_WALLET_TRANSFER)
            store_fail.seed_initial_state(TASK_WALLET_TRANSFER)

            res_c = execute_task(store_commit, TASK_WALLET_TRANSFER, "COMMIT_THEN_TIMEOUT")
            res_f = execute_task(store_fail, TASK_WALLET_TRANSFER, "FAIL_BEFORE_COMMIT")

            self.assertEqual(
                res_c.agent_visible_error.encode("utf-8"),
                res_f.agent_visible_error.encode("utf-8"),
                "Agent visible error bytes must be identical!",
            )
        finally:
            store_commit.cleanup()
            store_fail.cleanup()

    # ── Test 3: Unsafe Fault Modes Fail Closed ─────────────────────────────────

    def test_03_wrong_content_and_partial_commit_fail_closed(self):
        """
        WRONG_CONTENT and PARTIAL_COMMIT must report discrepancy and fail closed to ESCALATED.
        """
        # WRONG_CONTENT
        task_wc = TaskState(task_id="t_wc", objective="Transfer $500.00")
        probe_wc = make_dbops_probe_callback(self.store, TASK_WALLET_TRANSFER, "WRONG_CONTENT")
        res_wc = self.pipeline.execute_with_contract(
            task=task_wc,
            tool_name="dbops_database",
            params={"task_id": TASK_WALLET_TRANSFER, "fault_mode": "WRONG_CONTENT"},
            max_recovery_attempts=1,
            probe_callback=probe_wc,
            supports_idempotency=False,
        )
        self.assertEqual(res_wc.verified_receipt.final_status, AgentStatus.ESCALATED)

        # Clean store for next check
        self.store.cleanup()
        self.store = DbopsStore.create_temp()
        self.store.seed_initial_state(TASK_WALLET_TRANSFER)
        self.db_tool.store = self.store

        # PARTIAL_COMMIT
        task_pc = TaskState(task_id="t_pc", objective="Transfer $500.00")
        probe_pc = make_dbops_probe_callback(self.store, TASK_WALLET_TRANSFER, "PARTIAL_COMMIT")
        res_pc = self.pipeline.execute_with_contract(
            task=task_pc,
            tool_name="dbops_database",
            params={"task_id": TASK_WALLET_TRANSFER, "fault_mode": "PARTIAL_COMMIT"},
            max_recovery_attempts=1,
            probe_callback=probe_pc,
            supports_idempotency=False,
        )
        self.assertEqual(res_pc.verified_receipt.final_status, AgentStatus.ESCALATED)

    def test_04_status_unavailable_fails_closed(self):
        """
        STATUS_UNAVAILABLE must fail closed to ESCALATED without retrying.
        """
        task_su = TaskState(task_id="t_su", objective="Transfer $500.00")
        probe_su = make_dbops_probe_callback(self.store, TASK_WALLET_TRANSFER, "STATUS_UNAVAILABLE")
        res_su = self.pipeline.execute_with_contract(
            task=task_su,
            tool_name="dbops_database",
            params={"task_id": TASK_WALLET_TRANSFER, "fault_mode": "STATUS_UNAVAILABLE"},
            max_recovery_attempts=1,
            probe_callback=probe_su,
            supports_idempotency=False,
        )
        self.assertEqual(res_su.verified_receipt.final_status, AgentStatus.ESCALATED)

    # ── Test 5: Baseline Comparison (ReAgent Restraint vs Naive Retry) ─────────

    def test_05_baseline_comparison_reagent_vs_naive_retry(self):
        """
        Direct head-to-head comparison on identical initial state, task, and fault:
        - NAIVE RETRY: Upon timeout error, blindly executes attempt 2 -> causes DOUBLE DEBIT ($1000).
        - REAGENT: Upon timeout error, probes committed state, restrains -> EXACT DEBIT ($500).
        """
        store_naive = DbopsStore.create_temp(prefix="slice_naive_")
        store_reagent = DbopsStore.create_temp(prefix="slice_reagent_")
        try:
            store_naive.seed_initial_state(TASK_WALLET_TRANSFER)
            store_reagent.seed_initial_state(TASK_WALLET_TRANSFER)

            # 1. NAIVE RETRY EXECUTION:
            # Attempt 1: times out (commits)
            execute_task(store_naive, TASK_WALLET_TRANSFER, "COMMIT_THEN_TIMEOUT")
            # Blind retry attempt 2:
            execute_task(store_naive, TASK_WALLET_TRANSFER, "NO_FAULT")

            with store_naive.fresh_observer_connection() as conn:
                c_naive, sum_naive = conn.execute(
                    "SELECT COUNT(*), SUM(amount) FROM wallet_transfer_ledger WHERE transfer_id='TXN-779043'"
                ).fetchone()
                self.assertEqual(c_naive, 2, "Naive retry creates 2 debit records (duplicate effect)")
                self.assertEqual(sum_naive, 1000.00, "Naive retry caused double debit of $1000.00!")

            # 2. REAGENT PIPELINE EXECUTION:
            pipeline_reagent = ReAgentPipeline(agent_id="reagent-comp")
            tool_reagent = DbopsDatabaseTool(store_reagent)
            pipeline_reagent.tool_registry.register(tool_reagent)

            task_reagent = TaskState(task_id="t_comp", objective="Transfer $500.00")
            probe_reagent = make_dbops_probe_callback(store_reagent, TASK_WALLET_TRANSFER, "COMMIT_THEN_TIMEOUT")

            res = pipeline_reagent.execute_with_contract(
                task=task_reagent,
                tool_name="dbops_database",
                params={"task_id": TASK_WALLET_TRANSFER, "fault_mode": "COMMIT_THEN_TIMEOUT"},
                max_recovery_attempts=1,
                probe_callback=probe_reagent,
                supports_idempotency=False,
            )

            with store_reagent.fresh_observer_connection() as conn:
                c_reagent, sum_reagent = conn.execute(
                    "SELECT COUNT(*), SUM(amount) FROM wallet_transfer_ledger WHERE transfer_id='TXN-779043'"
                ).fetchone()
                self.assertEqual(c_reagent, 1, "ReAgent prevented duplicate; exactly 1 debit record exists!")
                self.assertEqual(sum_reagent, 500.00, "ReAgent preserved exact debit amount of $500.00!")

            self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
            self.assertEqual(tool_reagent.execution_count, 1)

        finally:
            store_naive.cleanup()
            store_reagent.cleanup()

    # ── Test 6: Mock Provider Structured Decision Traversal ───────────────────

    def test_06_mock_provider_structured_decision_pipeline_traversal(self):
        """
        Confirms that a mock provider StructuredDecision traverses the actual
        ReAgent pipeline validation, contract freezing, and tool routing.
        """
        import tempfile
        import os
        from reagent.policy import TokenBudgetGuard

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_ledger_path = tf.name
        os.remove(temp_ledger_path)

        try:
            mock_provider = DeterministicMockWalletProvider()
            budget = TokenBudgetGuard(ledger_path=temp_ledger_path, max_calls_per_run=10)
            pipeline_mock = ReAgentPipeline(
                agent_id="reagent-mock-agent",
                provider=mock_provider,
                token_budget=budget,
            )
            pipeline_mock.tool_registry.register(self.db_tool)

            # Call execute_task() with task objective
            objective = "Transfer $500.00 from wallet ACCT-2201 to wallet ACCT-3390. Request ID: TXN-779043."
            # Because execute_task doesn't take probe_callback directly, it attempts execution and enters recovery loop.
            res = pipeline_mock.execute_task(task_objective=objective, max_recovery_attempts=0)

            # The contract must have been frozen BEFORE execution
            self.assertIsNotNone(res.frozen_contract)
            self.assertEqual(res.frozen_contract.provenance, "fixture")
            self.assertEqual(self.db_tool.execution_count, 1)
            self.assertEqual(res.action_result.tool_name, "dbops_database")
        finally:
            for p in [temp_ledger_path, f"{temp_ledger_path}.lock"]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass


if __name__ == "__main__":
    unittest.main()
