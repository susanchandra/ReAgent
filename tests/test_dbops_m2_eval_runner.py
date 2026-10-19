"""
test_dbops_m2_eval_runner.py — Guarded Live-Model Evaluation Runner Tests (DBOPS-M2).

Verifies all DBOPS-M2 requirements under zero live provider calls and zero network requests:
1. Dry-run preflight mode executes with zero provider/network calls by default.
2. Explicit live-run opt-in required (fails closed if unconfigured).
3. Configuration freeze & fingerprint determinism; aborts on configuration drift.
4. Hard ceiling of 36 sample requests (12 tasks * 3 samples).
5. Zero automatic retries: error/timeout responses are durably recorded as failures, never retried.
6. Fail-closed budget and token ledger error handling.
7. Durable JSONL sample ledger and interruption/resume idempotency.
8. Gold-equivalence scoring: separates syntax validity, grounding, and gold equivalence.
9. Aggregation logic: task passes with >= 2/3 gold-equivalent; benchmark passes with >= 9/12 tasks.

Non-negotiable safety:
- Zero network requests. Zero live provider calls. Zero .env inspection.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from reagent.experiments.dbops.manifest_dbops_v1 import (
    ALL_TASK_IDS,
    LIVE_MODEL_TOTAL_SAMPLES,
    LIVE_MODEL_SAMPLES_PER_TASK,
    LIVE_MODEL_MIN_PASSING_TASKS,
)
from reagent.experiments.dbops.eval_runner import (
    DbopsEvaluationRunner,
    BenchmarkEvaluationSummary,
    FrozenEvaluationConfig,
)
from reagent.experiments.dbops.eval_scoring import DbopsGoldScorer
from reagent.experiments.dbops.fixture_replay import generate_valid_fixture
from reagent.policy import TokenBudgetGuard, GlobalTokenLedger, CorruptLedgerError
from reagent.providers.nebius import NebiusNemotronProvider


class TestDbopsM2EvalRunner(unittest.TestCase):
    """Guarded evaluation runner unit test suite for DBOPS-M2."""

    def setUp(self):
        # Create isolated temporary directory for results and ledger
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.ledger_file = self.temp_path / "test_ledger.json"
        self.results_dir = self.temp_path / "results"

        self.ledger = GlobalTokenLedger(
            ledger_path=str(self.ledger_file),
            max_cumulative_calls=50,
            max_cumulative_tokens=200_000,
        )
        self.budget_guard = TokenBudgetGuard(
            ledger=self.ledger,
            max_calls_per_run=45,
            max_tokens_per_run=150_000,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_default_mode_is_dry_run_with_zero_calls(self):
        """Runner defaults to dry-run preflight and makes zero provider or network calls."""
        with patch("urllib.request.urlopen") as mock_url, \
             patch("http.client.HTTPConnection.request") as mock_http:

            runner = DbopsEvaluationRunner(
                ledger_path=str(self.ledger_file),
                results_dir=str(self.results_dir),
                live_opt_in=False,
                budget_guard=self.budget_guard,
            )

            summary = runner.run_evaluation()

            self.assertIsInstance(summary, BenchmarkEvaluationSummary)
            self.assertFalse(summary.is_live)
            self.assertEqual(summary.total_samples_attempted, 0)
            self.assertFalse(summary.overall_gate_passed)
            self.assertEqual(len(summary.task_summaries), 12)

            # Zero network calls
            self.assertEqual(mock_url.call_count, 0)
            self.assertEqual(mock_http.call_count, 0)

    def test_02_explicit_opt_in_fails_closed_when_unconfigured(self):
        """Live mode with live_opt_in=True fails closed if provider is unconfigured."""
        with patch.dict(os.environ, {"NEBIUS_API_KEY": ""}):
            unconfigured_provider = NebiusNemotronProvider(client=None)
            self.assertFalse(unconfigured_provider.is_configured())

            runner = DbopsEvaluationRunner(
                ledger_path=str(self.ledger_file),
                results_dir=str(self.results_dir),
                live_opt_in=True,
                provider=unconfigured_provider,
                budget_guard=self.budget_guard,
            )

            with self.assertRaises(RuntimeError) as cm:
                runner.run_evaluation()
            self.assertIn("LiveExecutionRefused", str(cm.exception))

    def test_03_config_freeze_and_drift_detection(self):
        """Fingerprint is frozen deterministically, and mid-run drift aborts immediately."""
        runner = DbopsEvaluationRunner(
            ledger_path=str(self.ledger_file),
            results_dir=str(self.results_dir),
            live_opt_in=False,
            budget_guard=self.budget_guard,
        )

        fp1 = runner.frozen_config.config_fingerprint
        self.assertTrue(len(fp1) == 64)

        # Mutate frozen config hashes to simulate drift
        mutated_hashes = dict(runner.frozen_config.prompt_hashes)
        mutated_hashes["task_01_refund_issue"] = "tampered_hash_value"
        runner.frozen_config = FrozenEvaluationConfig(
            manifest_version=runner.frozen_config.manifest_version,
            target_model=runner.frozen_config.target_model,
            temperature=runner.frozen_config.temperature,
            max_tokens=runner.frozen_config.max_tokens,
            samples_per_task=runner.frozen_config.samples_per_task,
            total_samples=runner.frozen_config.total_samples,
            min_passing_tasks=runner.frozen_config.min_passing_tasks,
            prompt_hashes=mutated_hashes,
            config_fingerprint="tampered_fp",
        )

        def dummy_gen(tid, sidx):
            return generate_valid_fixture(tid)

        with self.assertRaises(RuntimeError) as cm:
            runner.run_evaluation(mock_response_generator=dummy_gen)
        self.assertIn("ConfigDriftDetected", str(cm.exception))

    def test_04_hard_ceiling_and_mock_execution(self):
        """Evaluates exactly 36 samples (12 tasks * 3) using mock generator without live calls."""
        runner = DbopsEvaluationRunner(
            ledger_path=str(self.ledger_file),
            results_dir=str(self.results_dir),
            live_opt_in=False,
            budget_guard=self.budget_guard,
        )

        call_log = []

        def mock_generator(task_id: str, sample_idx: int) -> str:
            call_log.append((task_id, sample_idx))
            return generate_valid_fixture(task_id)

        summary = runner.run_evaluation(mock_response_generator=mock_generator)

        # Verify call counts
        self.assertEqual(len(call_log), 36)
        self.assertEqual(summary.total_samples_attempted, 36)
        self.assertEqual(summary.schema_valid_samples, 36)
        self.assertEqual(summary.grounded_samples, 36)
        self.assertEqual(summary.gold_equivalent_samples, 36)
        self.assertEqual(summary.passing_tasks_count, 12)
        self.assertTrue(summary.overall_gate_passed)

    def test_05_no_retries_on_error_or_timeout(self):
        """Errors and timeouts are counted as sample failures without automatic retry."""
        runner = DbopsEvaluationRunner(
            ledger_path=str(self.ledger_file),
            results_dir=str(self.results_dir),
            live_opt_in=False,
            budget_guard=self.budget_guard,
        )

        call_counts = {}

        def faulty_generator(task_id: str, sample_idx: int) -> str:
            call_counts[task_id] = call_counts.get(task_id, 0) + 1
            if task_id == "task_01_refund_issue" and sample_idx == 0:
                raise TimeoutError("Simulated timeout")
            if task_id == "task_01_refund_issue" and sample_idx == 1:
                return "MALFORMED_NON_JSON"
            return generate_valid_fixture(task_id)

        summary = runner.run_evaluation(mock_response_generator=faulty_generator)

        # Total calls for task 01 must be exactly 3 (no 4th retry call)
        self.assertEqual(call_counts["task_01_refund_issue"], 3)
        t1 = summary.task_summaries["task_01_refund_issue"]
        self.assertEqual(t1.total_samples, 3)
        self.assertEqual(t1.gold_equivalent_count, 1)  # Only sample 2 passed
        self.assertFalse(t1.task_passed)  # Failed 2/3 majority requirement

    def test_06_budget_ledger_fail_closed(self):
        """Corrupt or invalid ledger file causes immediate fail-closed error."""
        # Corrupt the ledger file
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            f.write("corrupted_json_content{")

        with self.assertRaises(CorruptLedgerError):
            runner = DbopsEvaluationRunner(
                ledger_path=str(self.ledger_file),
                results_dir=str(self.results_dir),
                live_opt_in=False,
            )

    def test_07_resume_idempotency_prevents_duplicate_samples(self):
        """Completed samples recorded in durable JSONL ledger are never repeated on resume."""
        runner1 = DbopsEvaluationRunner(
            ledger_path=str(self.ledger_file),
            results_dir=str(self.results_dir),
            live_opt_in=False,
            budget_guard=self.budget_guard,
        )

        attempts_phase1 = []

        def gen_phase1(task_id: str, sample_idx: int) -> str:
            # Complete task_01 only, then simulate abort
            if task_id != "task_01_refund_issue":
                raise KeyboardInterrupt("Simulated user interruption")
            attempts_phase1.append(f"{task_id}_s{sample_idx}")
            return generate_valid_fixture(task_id)

        try:
            runner1.run_evaluation(mock_response_generator=gen_phase1)
        except KeyboardInterrupt:
            pass

        self.assertEqual(len(attempts_phase1), 3)

        # Resume with runner2
        runner2 = DbopsEvaluationRunner(
            ledger_path=str(self.ledger_file),
            results_dir=str(self.results_dir),
            live_opt_in=False,
            budget_guard=self.budget_guard,
        )

        attempts_phase2 = []

        def gen_phase2(task_id: str, sample_idx: int) -> str:
            attempts_phase2.append(f"{task_id}_s{sample_idx}")
            return generate_valid_fixture(task_id)

        summary2 = runner2.run_evaluation(mock_response_generator=gen_phase2)

        # Task 01 was already completed; phase 2 must NOT repeat task 01 samples
        self.assertNotIn("task_01_refund_issue_s0", attempts_phase2)
        self.assertNotIn("task_01_refund_issue_s1", attempts_phase2)
        self.assertNotIn("task_01_refund_issue_s2", attempts_phase2)
        # Phase 2 executed remaining 11 tasks * 3 = 33 samples
        self.assertEqual(len(attempts_phase2), 33)

    def test_08_gold_equivalence_scoring_separation(self):
        """Scorer distinguishes schema-valid, grounded, and true gold-equivalent contracts."""
        scorer = DbopsGoldScorer()
        task_id = "task_02_wallet_transfer"
        from reagent.models import StructuredDecision, Postcondition

        valid_decision = StructuredDecision(
            tool="dbops_database",
            arguments={"task_id": task_id},
            proposed_postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    target="wallet_transfer_ledger",
                    expected="TXN-779043",
                    description="Transfer TXN-779043 of 500.00 from ACCT-2201 to ACCT-3390",
                    required=True,
                ),
                Postcondition(
                    predicate="KEY_EXISTS",
                    target="transfer_id",
                    expected="ACCT-2201",
                    description="Transfer ID key exists",
                    required=False,
                ),
            ],
            provenance="fixture",
        )

        res_valid = scorer.evaluate_decision(task_id, valid_decision, schema_valid=True, grounding_valid=True)
        self.assertTrue(res_valid.is_schema_valid)
        self.assertTrue(res_valid.is_grounded)
        self.assertTrue(res_valid.is_gold_equivalent)
        self.assertTrue(res_valid.matched_request_id)
        self.assertTrue(res_valid.matched_numeric_amount)

        # Case 2: Schema-valid & grounded, but wrong amount ($100 instead of $500)
        wrong_amt_decision = StructuredDecision(
            tool="dbops_database",
            arguments={"task_id": task_id},
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

        res_amt = scorer.evaluate_decision(task_id, wrong_amt_decision, schema_valid=True, grounding_valid=True)
        self.assertTrue(res_amt.is_schema_valid)
        self.assertTrue(res_amt.is_grounded)
        self.assertFalse(res_amt.is_gold_equivalent)
        self.assertFalse(res_amt.matched_numeric_amount)
        self.assertIn("numeric amount", res_amt.score_summary)

        # Case 3: Schema-valid & grounded, but wrong task table
        wrong_tbl_decision = StructuredDecision(
            tool="dbops_database",
            arguments={"task_id": task_id},
            proposed_postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    target="refunds",
                    expected="TXN-779043",
                    description="Transfer TXN-779043 of 500.00",
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

        res_tbl = scorer.evaluate_decision(task_id, wrong_tbl_decision, schema_valid=True, grounding_valid=True)
        self.assertFalse(res_tbl.is_gold_equivalent)
        self.assertFalse(res_tbl.matched_target_tables)


if __name__ == "__main__":
    unittest.main()
