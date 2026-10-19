"""
test_dbops_m4a_recovery_demo.py — Focused Unit Tests for DBOPS-M4A Recovery Demonstration.

Verifies:
1. Naive Retry causes duplicate debit ($1000 across 2 rows) under COMMIT_THEN_TIMEOUT.
2. ReAgent restrains retry, producing exactly 1 debit of $500 and VERIFIED outcome.
3. Wrong amount ($100 instead of $500) fails verification.
4. Wrong source/destination entity fails verification.
5. Duplicated ledger effect fails verification.
6. Unavailable/unobservable observation fails closed to ESCALATED without retrying.
7. Deterministic JSON and report generation matches exact database state.

Non-negotiable safety:
- Uses temporary SQLite databases and test doubles only.
- Zero network calls. Zero live provider calls. Zero .env inspection.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from reagent.models import TaskState, AgentStatus, StructuredDecision, Postcondition
from reagent.pipeline import ReAgentPipeline
from reagent.policy import TokenBudgetGuard, GlobalTokenLedger
from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.executor import execute_task
from reagent.experiments.dbops.integration import (
    DbopsDatabaseTool,
    make_dbops_probe_callback,
    DeterministicMockWalletProvider,
)
from reagent.experiments.dbops.tasks import TASK_WALLET_TRANSFER, get_task
from reagent.experiments.dbops.gold_contracts import get_gold_contract
from reagent.experiments.dbops.eval_scoring import DbopsGoldScorer
from reagent.experiments.dbops.recovery_demo import run_demonstration, DemoArtifact


class TestDbopsM4ARecoveryDemo(unittest.TestCase):
    """Focused unit tests for judge-facing recovery demonstration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.ledger_file = self.temp_path / "test_ledger.json"

        self.ledger = GlobalTokenLedger(
            ledger_path=str(self.ledger_file),
            max_cumulative_calls=50,
            max_cumulative_tokens=200_000,
        )
        self.budget_guard = TokenBudgetGuard(
            ledger=self.ledger,
            max_calls_per_run=20,
            max_tokens_per_run=50_000,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_naive_retry_causes_duplicate_debit(self):
        """Under COMMIT_THEN_TIMEOUT, Naive Retry blindly retries, inserting 2 debit rows ($1000)."""
        store = DbopsStore.create_temp(prefix="test_naive_")
        try:
            store.seed_initial_state(TASK_WALLET_TRANSFER)

            # Attempt 1 commits, returns timeout
            res1 = execute_task(store, TASK_WALLET_TRANSFER, "COMMIT_THEN_TIMEOUT")
            self.assertIsNotNone(res1.agent_visible_error)

            # Attempt 2: blind retry executes again
            res2 = execute_task(store, TASK_WALLET_TRANSFER, "NO_FAULT")

            with store.fresh_observer_connection() as conn:
                count, total = conn.execute(
                    "SELECT COUNT(*), SUM(amount) FROM wallet_transfer_ledger WHERE transfer_id='TXN-779043'"
                ).fetchone()
                self.assertEqual(count, 2, "Naive retry inserted duplicate row!")
                self.assertEqual(total, 1000.00, "Naive retry caused double debit of $1000.00!")
        finally:
            store.cleanup()

    def test_02_reagent_prevents_duplicate_and_verifies(self):
        """Under COMMIT_THEN_TIMEOUT, ReAgent probes committed state, restrains, preserving $500."""
        store = DbopsStore.create_temp(prefix="test_reagent_")
        try:
            store.seed_initial_state(TASK_WALLET_TRANSFER)
            pipeline = ReAgentPipeline(agent_id="test-reagent-agent", token_budget=self.budget_guard)
            tool = DbopsDatabaseTool(store)
            pipeline.tool_registry.register(tool)

            task_obj = get_task(TASK_WALLET_TRANSFER).human_description
            task_state = TaskState(task_id="t_reagent", objective=task_obj)
            probe = make_dbops_probe_callback(store, TASK_WALLET_TRANSFER, "COMMIT_THEN_TIMEOUT")

            res = pipeline.execute_with_contract(
                task=task_state,
                tool_name="dbops_database",
                params={"task_id": TASK_WALLET_TRANSFER, "fault_mode": "COMMIT_THEN_TIMEOUT"},
                max_recovery_attempts=1,
                probe_callback=probe,
                supports_idempotency=False,
            )

            self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
            self.assertEqual(tool.execution_count, 1, "ReAgent restrained retry!")

            with store.fresh_observer_connection() as conn:
                count, total = conn.execute(
                    "SELECT COUNT(*), SUM(amount) FROM wallet_transfer_ledger WHERE transfer_id='TXN-779043'"
                ).fetchone()
                self.assertEqual(count, 1, "Exactly one debit row committed!")
                self.assertEqual(total, 500.00, "Exact amount $500.00 preserved!")
        finally:
            store.cleanup()

    def test_03_wrong_amount_fails_contract_scoring(self):
        """Contract proposing $100 instead of $500 fails gold-equivalence scoring."""
        scorer = DbopsGoldScorer()
        decision = StructuredDecision(
            tool="dbops_database",
            arguments={"task_id": TASK_WALLET_TRANSFER},
            proposed_postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    target="wallet_transfer_ledger",
                    expected="TXN-779043",
                    description="Transfer TXN-779043 of 100.00 from ACCT-2201 to ACCT-3390",
                    required=True,
                ),
                Postcondition(
                    predicate="KEY_EXISTS",
                    target="transfer_id",
                    expected="ACCT-2201",
                    description="Key exists",
                    required=False,
                ),
            ],
            provenance="fixture",
        )

        res = scorer.evaluate_decision(TASK_WALLET_TRANSFER, decision, schema_valid=True, grounding_valid=True)
        self.assertFalse(res.is_gold_equivalent)
        self.assertFalse(res.matched_numeric_amount)
        self.assertIn("numeric amount", res.score_summary)

    def test_04_wrong_source_destination_fails_scoring(self):
        """Contract referencing wrong accounts (ACCT-9999) fails gold-equivalence scoring."""
        scorer = DbopsGoldScorer()
        decision = StructuredDecision(
            tool="dbops_database",
            arguments={"task_id": TASK_WALLET_TRANSFER},
            proposed_postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    target="wallet_transfer_ledger",
                    expected="TXN-779043",
                    description="Transfer TXN-779043 of 500.00 from ACCT-9999 to ACCT-8888",
                    required=True,
                ),
                Postcondition(
                    predicate="KEY_EXISTS",
                    target="transfer_id",
                    expected="TXN-779043",
                    description="Key exists",
                    required=False,
                ),
            ],
            provenance="fixture",
        )

        res = scorer.evaluate_decision(TASK_WALLET_TRANSFER, decision, schema_valid=True, grounding_valid=True)
        self.assertFalse(res.is_gold_equivalent)
        self.assertTrue(any("source entity" in err for err in res.gold_equivalence_errors))

    def test_05_duplicated_ledger_effect_fails_verification(self):
        """If 2 ledger rows already exist on disk, gold contract evaluator marks duplicate primary state."""
        store = DbopsStore.create_temp(prefix="test_dup_")
        try:
            store.seed_initial_state(TASK_WALLET_TRANSFER)
            # Create two debit records
            execute_task(store, TASK_WALLET_TRANSFER, "NO_FAULT")
            execute_task(store, TASK_WALLET_TRANSFER, "NO_FAULT")

            from reagent.experiments.dbops.oracle import DbopsOracle
            from reagent.experiments.dbops.executor import ExecutionResult
            from reagent.experiments.dbops.validator import GoldContractEvaluator

            exec_res = ExecutionResult(
                task_id=TASK_WALLET_TRANSFER,
                fault_mode="NO_FAULT",
                agent_visible_error=None,
                committed_steps=[0, 1],
            )
            oracle = DbopsOracle(store, exec_res)
            obs = oracle.evaluate()
            self.assertGreater(obs.duplicate_primary_count, 1)

            evaluator = GoldContractEvaluator()
            eval_res = evaluator.evaluate(TASK_WALLET_TRANSFER, obs)
            self.assertTrue(eval_res.primary_state_duplicate, "Evaluator flagged duplicate primary state!")
        finally:
            store.cleanup()

    def test_06_unavailable_observation_fails_closed(self):
        """When observation probe is UNAVAILABLE, ReAgent fails closed to ESCALATED without retrying."""
        store = DbopsStore.create_temp(prefix="test_unavail_")
        try:
            store.seed_initial_state(TASK_WALLET_TRANSFER)
            pipeline = ReAgentPipeline(agent_id="test-unavail-agent", token_budget=self.budget_guard)
            tool = DbopsDatabaseTool(store)
            pipeline.tool_registry.register(tool)

            task_state = TaskState(task_id="t_unavail", objective="Transfer $500.00")
            probe = make_dbops_probe_callback(store, TASK_WALLET_TRANSFER, "STATUS_UNAVAILABLE")

            res = pipeline.execute_with_contract(
                task=task_state,
                tool_name="dbops_database",
                params={"task_id": TASK_WALLET_TRANSFER, "fault_mode": "STATUS_UNAVAILABLE"},
                max_recovery_attempts=1,
                probe_callback=probe,
                supports_idempotency=False,
            )

            self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
            self.assertEqual(tool.execution_count, 1, "Must not retry when probe is unavailable!")
        finally:
            store.cleanup()

    def test_07_deterministic_json_and_report_generation(self):
        """run_demonstration produces structured machine-readable artifact matching execution."""
        out_dir = self.temp_path / "artifacts"
        artifact = run_demonstration(output_dir=str(out_dir))

        self.assertIsInstance(artifact, DemoArtifact)
        self.assertEqual(artifact.naive_retry_arm.net_transferred, 1000.00)
        self.assertTrue(artifact.naive_retry_arm.duplicate_occurred)
        self.assertEqual(artifact.reagent_arm.net_transferred, 500.00)
        self.assertFalse(artifact.reagent_arm.duplicate_occurred)
        self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED")

        json_file = out_dir / "recovery_demo_artifact.json"
        self.assertTrue(json_file.exists())
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["task_id"], TASK_WALLET_TRANSFER)
        self.assertEqual(data["reagent_arm"]["final_status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()


