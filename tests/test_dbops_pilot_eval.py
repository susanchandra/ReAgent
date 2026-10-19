"""
test_dbops_pilot_eval.py — Focused Unit Tests for DBOPS Nemotron Pilot Evaluation Preparation.

Verifies:
1. Pilot config frozen integrity, prompt isolation, and task representation.
2. Default path makes zero network calls and fails closed without opt-in.
3. Append-only ledger integrity, duplicate sample rejection, and corruption detection.
4. Telemetry correctly distinguishes missing usage from zero.
5. Deterministic offline mock execution end-to-end via CandidateContractValidationGate.
6. Fail-closed behavior on model refusals, malformed outputs, and semantic discrepancies.
7. Interrupted runs resume cleanly without duplicate requests or retries.
8. Config drift detection aborts mid-run fail-closed.

OFFLINE ONLY: Zero live provider calls. Zero network requests. Zero .env inspection.
"""

import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

from reagent.experiments.dbops.pilot_eval import (
    PILOT_MANIFEST_VERSION,
    PILOT_TASK_IDS,
    PILOT_SAMPLES_PER_TASK,
    PILOT_TOTAL_SAMPLES,
    PILOT_TARGET_MODEL,
    PILOT_TEMPERATURE,
    PILOT_MAX_TOKENS,
    PILOT_SCHEMA_VERSION,
    PilotEvaluationRunner,
    PilotLedger,
    PilotSampleLedgerRecord,
    FrozenPilotConfig,
)
from reagent.experiments.dbops.candidate_gate import CandidateContractValidationGate
from reagent.experiments.dbops.tasks import get_task
from reagent.experiments.dbops.types import TaskProtectionClass
from reagent.policy import GlobalTokenLedger, TokenBudgetGuard, CorruptLedgerError


class TestDbopsPilotEvaluation(unittest.TestCase):
    """
    Focused unit tests for the controlled offline pilot preparation.
    """

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reagent_pilot_test_"))
        self.ledger_dir = self.temp_dir / "ledger_dir"
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.token_ledger_path = self.temp_dir / "token_ledger.json"

        gt_ledger = GlobalTokenLedger(
            ledger_path=str(self.token_ledger_path),
            max_cumulative_calls=20,
            max_cumulative_tokens=50_000,
        )
        self.budget_guard = TokenBudgetGuard(
            ledger=gt_ledger,
            max_calls_per_run=10,
            max_tokens_per_run=10_000,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_pilot_config_frozen_and_isolated(self):
        """Pilot config is frozen with 3 representative tasks, correct settings, and zero prompt leakage."""
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)
        cfg = runner.frozen_config

        # Check tasks and counts
        self.assertEqual(cfg.pilot_manifest_version, PILOT_MANIFEST_VERSION)
        self.assertEqual(cfg.task_ids, PILOT_TASK_IDS)
        self.assertEqual(len(cfg.task_ids), 3)
        self.assertEqual(cfg.samples_per_task, 2)
        self.assertEqual(cfg.total_samples, 6)
        self.assertEqual(cfg.temperature, 0.0)
        self.assertEqual(cfg.max_tokens, 512)
        self.assertEqual(cfg.schema_version, "structured_decision_v1")

        # Verify representation across protection classes
        task_01 = get_task("task_01_refund_issue")
        task_02 = get_task("task_02_wallet_transfer")
        task_05 = get_task("task_05_order_cancellation")

        # 1. Single Key Protected
        self.assertEqual(task_01.protection_class, TaskProtectionClass.KEY_PROTECTED)
        self.assertEqual(len(task_01.table_unique_constraints), 1)

        # 2. Non-idempotent Numeric Mutation
        self.assertEqual(task_02.protection_class, TaskProtectionClass.UNPROTECTED_NUMERIC)
        self.assertEqual(task_02.prompt_numeric_amount, 500.00)
        self.assertTrue(task_02.primary_state_duplicate_on_retry)

        # 3. Composite Business Key Protected
        self.assertEqual(task_05.protection_class, TaskProtectionClass.KEY_PROTECTED)
        self.assertEqual(task_05.table_unique_constraints[0].columns, ("order_id", "event_type"))

        # Verify prompt isolation (zero leaked fault modes or gold queries)
        isolation_results = runner.verify_pilot_prompt_isolation()
        for task_id, res in isolation_results.items():
            self.assertTrue(res["is_clean"], f"Task {task_id} prompt leaked metadata: {res}")
            self.assertEqual(len(res["leaked_faults"]), 0)
            self.assertEqual(len(res["leaked_gold_markers"]), 0)

    def test_02_default_path_zero_network_calls(self):
        """Default runner mode is strictly offline preflight; live calls fail closed without opt-in."""
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)
        self.assertFalse(runner.live_opt_in)
        self.assertIsNone(runner.provider)

        # Calling run_pilot() with defaults executes preflight dry-run only
        summary = runner.run_pilot()
        self.assertFalse(summary.is_live)
        self.assertEqual(summary.total_samples_attempted, 0)
        self.assertIn("PREFLIGHT_ONLY", summary.task_summaries)
        preflight_info = summary.task_summaries["PREFLIGHT_ONLY"]
        self.assertEqual(preflight_info["network_calls_made"], 0)
        self.assertEqual(preflight_info["paid_tokens_consumed"], 0)

        # Attempting live opt-in without configured provider raises RuntimeError fail-closed
        with patch.dict(os.environ, {"NEBIUS_API_KEY": ""}):
            live_runner = PilotEvaluationRunner(
                pilot_dir=self.temp_dir,
                live_opt_in=True,
                budget_guard=self.budget_guard,
            )
            with self.assertRaises(RuntimeError) as ctx:
                live_runner.run_pilot()
            self.assertIn("LiveExecutionRefused", str(ctx.exception))

    def test_03_append_only_ledger_integrity_and_duplicate_rejection(self):
        """Ledger supports durable appending, rejects duplicate sample IDs, and detects corrupt lines."""
        ledger_file = self.temp_dir / "test_ledger.jsonl"
        raw_dir = self.temp_dir / "raw_responses"
        ledger = PilotLedger(ledger_file, raw_dir)

        # Save a sample raw response
        art_path = ledger.save_raw_response_artifact("sample_01", '{"tool": "dbops_database"}')
        self.assertTrue(os.path.exists(art_path))

        rec1 = PilotSampleLedgerRecord(
            sample_id="pilot_task_01_refund_issue_s0",
            request_id="req_001",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier=PILOT_TARGET_MODEL,
            config_fingerprint="cfg_fp_123",
            prompt_fingerprint="pmt_fp_123",
            schema_fingerprint="sch_fp_123",
            timestamp_utc=1000.0,
            timestamp_iso="2026-10-14T00:00:00Z",
            is_live_call=False,
            status="COMPLETED",
            latency_ms=120,
            raw_response_artifact_path=art_path,
            generation_parse_success=True,
            schema_valid=True,
            schema_error=None,
            grounding_valid=True,
            grounding_errors=[],
            semantic_gold_agreement=True,
            missing_required_postconditions=[],
            extra_unsupported_predicates=[],
            unsupported_tables_or_columns=[],
            uniqueness_respected=True,
            invalid_contract_rejected=False,
            fallback_used=False,
            rejection_reason=None,
            token_telemetry=None,
            cost_telemetry_usd=None,
        )
        ledger.append_record(rec1)

        completed = ledger.load_completed_sample_ids()
        self.assertEqual(completed, {"pilot_task_01_refund_issue_s0"})

        # Attempt duplicate append -> fails closed with ValueError
        with self.assertRaises(ValueError) as ctx:
            ledger.append_record(rec1)
        self.assertIn("DuplicateRecordAttempt", str(ctx.exception))

        # Test corrupt JSON detection
        with open(ledger_file, "a", encoding="utf-8") as f:
            f.write("{corrupt_json_line\n")

        with self.assertRaises(CorruptLedgerError):
            ledger.load_completed_sample_ids()

    def test_04_telemetry_distinguishes_missing_from_zero(self):
        """Missing token/cost telemetry is recorded as None, never masked as zero."""
        ledger_file = self.temp_dir / "telemetry_ledger.jsonl"
        ledger = PilotLedger(ledger_file, self.temp_dir / "raw")

        rec = PilotSampleLedgerRecord(
            sample_id="sample_missing_telemetry",
            request_id="req_tel_01",
            task_id="task_02_wallet_transfer",
            sample_index=0,
            model_identifier=PILOT_TARGET_MODEL,
            config_fingerprint="cfg_fp",
            prompt_fingerprint="pmt_fp",
            schema_fingerprint="sch_fp",
            timestamp_utc=2000.0,
            timestamp_iso="2026-10-14T00:00:00Z",
            is_live_call=False,
            status="COMPLETED",
            latency_ms=150,
            raw_response_artifact_path=None,
            generation_parse_success=True,
            schema_valid=True,
            schema_error=None,
            grounding_valid=True,
            grounding_errors=[],
            semantic_gold_agreement=True,
            missing_required_postconditions=[],
            extra_unsupported_predicates=[],
            unsupported_tables_or_columns=[],
            uniqueness_respected=True,
            invalid_contract_rejected=False,
            fallback_used=False,
            rejection_reason=None,
            token_telemetry=None,       # Missing telemetry
            cost_telemetry_usd=None,    # Missing telemetry
        )
        ledger.append_record(rec)

        # Inspect raw written json
        with open(ledger_file, "r", encoding="utf-8") as f:
            loaded = json.loads(f.readline())
        self.assertIsNone(loaded["token_telemetry"])
        self.assertIsNone(loaded["cost_telemetry_usd"])
        self.assertNotEqual(loaded["token_telemetry"], 0)
        self.assertNotEqual(loaded["cost_telemetry_usd"], 0.0)

    def test_05_offline_mock_execution_end_to_end(self):
        """Offline mock pilot executes all 6 samples, scores them via validation gate, and verifies gate outcome."""
        gate = CandidateContractValidationGate()
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)

        def mock_generator(task_id: str, sample_idx: int) -> str:
            # Deterministic gold-equivalent fixture from candidate gate
            return gate.generate_candidate_fixture(task_id, "VALID_GOLD_EQUIVALENT")

        summary = runner.run_pilot(mock_response_generator=mock_generator)

        self.assertEqual(summary.total_samples_attempted, 6)
        self.assertEqual(summary.schema_valid_samples, 6)
        self.assertEqual(summary.grounded_samples, 6)
        self.assertEqual(summary.gold_equivalent_samples, 6)
        self.assertEqual(summary.passing_tasks_count, 3)
        self.assertTrue(summary.overall_gate_passed)

        # Verify fallback was NEVER used
        for rec in summary.sample_records:
            self.assertFalse(rec.fallback_used)
            self.assertTrue(rec.generation_parse_success)
            self.assertTrue(rec.schema_valid)
            self.assertTrue(rec.grounding_valid)
            self.assertTrue(rec.semantic_gold_agreement)

        # Verify raw artifacts were persisted
        for rec in summary.sample_records:
            self.assertIsNotNone(rec.raw_response_artifact_path)
            self.assertTrue(os.path.exists(rec.raw_response_artifact_path))

    def test_06_fail_closed_on_refusals_malformed_and_discrepancies(self):
        """Refusals, malformed outputs, and semantic errors fail closed and fail the pilot gate."""
        gate = CandidateContractValidationGate()
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)

        def faulty_mock_generator(task_id: str, sample_idx: int) -> str:
            if task_id == "task_01_refund_issue":
                # Sample 0 is refusal; sample 1 is malformed JSON
                return gate.generate_candidate_fixture(
                    task_id, "MODEL_REFUSAL" if sample_idx == 0 else "MALFORMED_JSON"
                )
            elif task_id == "task_02_wallet_transfer":
                # Wrong amount on task 2 (which has numeric transfer of $500.00)
                return gate.generate_candidate_fixture(task_id, "WRONG_EXPECTED_AMOUNT")
            else:
                # Missing required trace postcondition on task 5
                return gate.generate_candidate_fixture(task_id, "MISSING_REQUIRED_POSTCONDITION")

        summary = runner.run_pilot(mock_response_generator=faulty_mock_generator)

        self.assertEqual(summary.total_samples_attempted, 6)
        self.assertEqual(summary.gold_equivalent_samples, 0)
        self.assertEqual(summary.passing_tasks_count, 0)
        self.assertFalse(summary.overall_gate_passed)

        # Verify rejection reasons recorded in ledger
        statuses = [rec.status for rec in summary.sample_records]
        self.assertIn("REFUSAL", statuses)
        self.assertIn("PARSE_ERROR", statuses)
        self.assertIn("SEMANTIC_ERROR", statuses)

        # Confirm zero retries occurred
        self.assertEqual(len(summary.sample_records), 6)

    def test_07_interrupted_run_resumes_without_duplicate_requests(self):
        """Interrupted pilot run resumes cleanly and does not re-request completed samples."""
        gate = CandidateContractValidationGate()
        runner1 = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)

        # Simulate interruption after sample 0 of task_01
        rec0 = PilotSampleLedgerRecord(
            sample_id="pilot_task_01_refund_issue_s0",
            request_id="req_interrupted_0",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier=PILOT_TARGET_MODEL,
            config_fingerprint=runner1.frozen_config.config_fingerprint,
            prompt_fingerprint=runner1.frozen_config.prompt_hashes["task_01_refund_issue"],
            schema_fingerprint=runner1.frozen_config.schema_fingerprint,
            timestamp_utc=100.0,
            timestamp_iso="2026-10-14T00:00:00Z",
            is_live_call=False,
            status="COMPLETED",
            latency_ms=100,
            raw_response_artifact_path=None,
            generation_parse_success=True,
            schema_valid=True,
            schema_error=None,
            grounding_valid=True,
            grounding_errors=[],
            semantic_gold_agreement=True,
            missing_required_postconditions=[],
            extra_unsupported_predicates=[],
            unsupported_tables_or_columns=[],
            uniqueness_respected=True,
            invalid_contract_rejected=False,
            fallback_used=False,
            rejection_reason=None,
            token_telemetry=None,
            cost_telemetry_usd=None,
        )
        runner1.ledger.append_record(rec0)

        calls_made = []

        def tracking_generator(task_id: str, sample_idx: int) -> str:
            calls_made.append((task_id, sample_idx))
            return gate.generate_candidate_fixture(task_id, "VALID_GOLD_EQUIVALENT")

        # Resume execution
        runner2 = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)
        summary2 = runner2.run_pilot(mock_response_generator=tracking_generator)

        # Exactly 5 new samples should have been run (sample 0 was skipped!)
        self.assertEqual(len(calls_made), 5)
        self.assertNotIn(("task_01_refund_issue", 0), calls_made)
        self.assertIn(("task_01_refund_issue", 1), calls_made)

        # Total completed records in ledger is now exactly 6
        all_completed = runner2.ledger.load_completed_sample_ids()
        self.assertEqual(len(all_completed), 6)

    def test_08_config_drift_detection_fails_closed(self):
        """Runner aborts immediately if a prompt hash or config drifts mid-run."""
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)

        # Mutate frozen config hash to simulate drift
        mutated_hashes = dict(runner.frozen_config.prompt_hashes)
        mutated_hashes["task_01_refund_issue"] = "drifted_fake_hash_12345"

        drifted_config = FrozenPilotConfig(
            pilot_manifest_version=runner.frozen_config.pilot_manifest_version,
            task_ids=runner.frozen_config.task_ids,
            target_model=runner.frozen_config.target_model,
            temperature=runner.frozen_config.temperature,
            max_tokens=runner.frozen_config.max_tokens,
            samples_per_task=runner.frozen_config.samples_per_task,
            total_samples=runner.frozen_config.total_samples,
            schema_version=runner.frozen_config.schema_version,
            prompt_hashes=mutated_hashes,
            schema_fingerprint=runner.frozen_config.schema_fingerprint,
            config_fingerprint=runner.frozen_config.config_fingerprint,
            acceptance_criteria=runner.frozen_config.acceptance_criteria,
            stopping_conditions=runner.frozen_config.stopping_conditions,
        )
        runner.frozen_config = drifted_config

        gate = CandidateContractValidationGate()
        with self.assertRaises(RuntimeError) as ctx:
            runner.run_pilot(mock_response_generator=lambda t, s: gate.generate_candidate_fixture(t, "VALID_GOLD_EQUIVALENT"))
        self.assertIn("ConfigDriftDetected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
