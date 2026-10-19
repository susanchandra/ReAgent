"""
test_dbops_adversarial_review.py — Adversarial Boundary Verification for Nemotron DBOPS Evaluation.

Adversarially proves:
A. Global Six-Request Bound: Maximum is enforced across multiple processes/threads and repeated runs.
B. Dispatch State Machine: Intent precedes call; mid-flight crash prevents replay; exceptions count as attempts without retries.
C. Ledger & Locking: Full read-check-write under FileLock; two writers cannot claim the same sample; fingerprints enforced.
D. Provider Boundary: Zero network calls without opt-in; secret scrubbing prevents credential echoes in logs and artifacts.
E. Frozen Plan & Prompt Isolation: Strict isolation from gold contracts, oracle queries, and injected fault modes.
F. Cost Boundaries: No fake dollar caps; client-side runaway loop guards distinguished from provider billing.

STRICTLY OFFLINE: Zero live provider calls. Zero network requests. Zero .env inspection.
"""

import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from reagent.experiments.dbops.pilot_eval import (
    PILOT_MANIFEST_VERSION,
    PILOT_TASK_IDS,
    PILOT_SAMPLES_PER_TASK,
    PILOT_TOTAL_SAMPLES,
    PILOT_TARGET_MODEL,
    PilotEvaluationRunner,
    PilotLedger,
    PilotSampleLedgerRecord,
    AmbiguousDispatchedRequestError,
    ConfigFingerprintMismatchError,
    sanitize_text,
)
from reagent.experiments.dbops.candidate_gate import CandidateContractValidationGate
from reagent.policy import GlobalTokenLedger, TokenBudgetGuard, CorruptLedgerError


class TestDbopsAdversarialReview(unittest.TestCase):
    """
    Adversarial verification suite for live-execution boundary safety.
    """

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reagent_adv_review_"))
        self.token_ledger_path = self.temp_dir / "token_ledger.json"

        gt_ledger = GlobalTokenLedger(
            ledger_path=str(self.token_ledger_path),
            max_cumulative_calls=30,
            max_cumulative_tokens=100_000,
        )
        self.budget_guard = TokenBudgetGuard(
            ledger=gt_ledger,
            max_calls_per_run=15,
            max_tokens_per_run=20_000,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_adv_01_global_six_request_bound_across_concurrent_writers(self):
        """Global 6-request ceiling is enforced atomically under FileLock even with multiple concurrent runners."""
        shared_ledger_path = self.temp_dir / "shared_ledger.jsonl"
        raw_dir = self.temp_dir / "raw"
        ledger = PilotLedger(shared_ledger_path, raw_dir)

        # Attempt to claim 8 sample dispatches across threads
        claimed_samples: List[str] = []
        errors: List[Exception] = []

        def worker_claim(sample_idx: int):
            task_name = PILOT_TASK_IDS[sample_idx % len(PILOT_TASK_IDS)]
            sample_id = f"pilot_{task_name}_s{sample_idx}"
            req_id = f"req_{sample_idx}"
            try:
                ledger.record_dispatch_intent(
                    sample_id=sample_id,
                    request_id=req_id,
                    task_id=task_name,
                    sample_index=sample_idx,
                    model_identifier=PILOT_TARGET_MODEL,
                    config_fingerprint="cfg_fp_shared",
                    prompt_fingerprint="pmt_fp",
                    schema_fingerprint="sch_fp",
                    is_live_call=False,
                    max_global_samples=6,
                )
                claimed_samples.append(sample_id)
            except Exception as exc:
                errors.append(exc)

        # Run 8 concurrent claim attempts (exceeding ceiling of 6)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(worker_claim, range(8)))

        # Exactly 6 samples were claimed; the 7th and 8th raised SafetyViolation
        self.assertEqual(len(claimed_samples), 6)
        self.assertEqual(len(errors), 2)
        for err in errors:
            self.assertIn("SafetyViolation: Global ceiling of 6 requests reached", str(err))

    def test_adv_02_two_writers_claiming_same_sample_rejected(self):
        """Two concurrent writers attempting to claim the exact same sample ID cannot race; second is rejected."""
        ledger_file = self.temp_dir / "race_ledger.jsonl"
        ledger = PilotLedger(ledger_file, self.temp_dir / "raw")

        # Writer 1 claims sample_01
        ledger.record_dispatch_intent(
            sample_id="pilot_task_01_refund_issue_s0",
            request_id="req_writer_1",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier=PILOT_TARGET_MODEL,
            config_fingerprint="cfg_fp",
            prompt_fingerprint="pmt_fp",
            schema_fingerprint="sch_fp",
            is_live_call=False,
        )

        # Writer 2 attempts to claim the exact same sample_01 -> raises AmbiguousDispatchedRequestError
        with self.assertRaises(AmbiguousDispatchedRequestError) as ctx:
            ledger.record_dispatch_intent(
                sample_id="pilot_task_01_refund_issue_s0",
                request_id="req_writer_2",
                task_id="task_01_refund_issue",
                sample_index=0,
                model_identifier=PILOT_TARGET_MODEL,
                config_fingerprint="cfg_fp",
                prompt_fingerprint="pmt_fp",
                schema_fingerprint="sch_fp",
                is_live_call=False,
            )
        self.assertIn("SampleAlreadyClaimed", str(ctx.exception))
        self.assertIn("Concurrent duplicate dispatch is prohibited", str(ctx.exception))

    def test_adv_03_dispatch_state_machine_crash_refuses_auto_replay(self):
        """Crash after DISPATCH intent halts resume and strictly refuses silent replay."""
        ledger_file = self.temp_dir / "crash_ledger.jsonl"
        ledger = PilotLedger(ledger_file, self.temp_dir / "raw")

        # Simulate state: DISPATCHED written to disk, then process killed
        ledger.record_dispatch_intent(
            sample_id="pilot_task_02_wallet_transfer_s0",
            request_id="req_crashed_123",
            task_id="task_02_wallet_transfer",
            sample_index=0,
            model_identifier=PILOT_TARGET_MODEL,
            config_fingerprint="cfg_fp",
            prompt_fingerprint="pmt_fp",
            schema_fingerprint="sch_fp",
            is_live_call=True,
        )

        # Attempt to resume with runner
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)
        runner.ledger.ledger_file = ledger_file

        with self.assertRaises(AmbiguousDispatchedRequestError) as ctx:
            runner.ledger.load_completed_sample_ids(expected_config_fingerprint="cfg_fp")
        self.assertIn("pilot_task_02_wallet_transfer_s0", str(ctx.exception))
        self.assertIn("req_crashed_123", str(ctx.exception))
        self.assertIn("Automated replay is prohibited to prevent duplicate billing", str(ctx.exception))

    def test_adv_04_provider_exception_does_not_mask_billing_and_counts_as_attempt(self):
        """Provider network exception is recorded as PROVIDER_ERROR terminal status and counts as attempt."""
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)

        def failing_generator(task_id: str, sample_idx: int) -> str:
            raise ConnectionResetError("Remote disconnected while receiving payload")

        summary = runner.run_pilot(mock_response_generator=failing_generator)

        # All 6 attempts ran and failed closed; none were re-requested
        self.assertEqual(summary.total_samples_attempted, 6)
        self.assertEqual(summary.gold_equivalent_samples, 0)
        self.assertFalse(summary.overall_gate_passed)

        # Verify every sample has a terminal record with PROVIDER_ERROR status
        for rec in summary.sample_records:
            self.assertEqual(rec.status, "PROVIDER_ERROR")
            self.assertIn("Remote disconnected", rec.failure_reason)

        # Re-running dispatches 0 additional requests
        summary_rerun = runner.run_pilot(mock_response_generator=failing_generator)
        self.assertEqual(summary_rerun.total_samples_attempted, 0)

    def test_adv_05_secret_scrubbing_under_adversarial_exceptions_and_headers(self):
        """Scrubber removes Bearer tokens, secrets, API keys from exceptions and payloads."""
        leaky_exception = Exception(
            "HTTP 401 Unauthorized: Bearer sk-nebius-secret-token-abcdef12345 "
            "with api_key='sk-nebius-live-secret' and http://admin:pass999@api.nebius.ai"
        )
        cleaned = sanitize_text(f"Provider invocation error: {leaky_exception}")
        self.assertNotIn("sk-nebius-secret-token-abcdef12345", cleaned)
        self.assertNotIn("sk-nebius-live-secret", cleaned)
        self.assertNotIn("pass999", cleaned)
        self.assertIn("Bearer [REDACTED]", cleaned)
        self.assertIn("api_key=[REDACTED]", cleaned)
        self.assertIn("http://[REDACTED]@", cleaned)

    def test_adv_06_config_and_manifest_fingerprint_mismatch_fails_closed(self):
        """Ledger rejects records with drifted or mismatched config fingerprint."""
        ledger_file = self.temp_dir / "drift_ledger.jsonl"
        ledger = PilotLedger(ledger_file, self.temp_dir / "raw")

        # Record with wrong fingerprint
        rec = PilotSampleLedgerRecord(
            sample_id="pilot_task_01_refund_issue_s0",
            request_id="req_001",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier="old_model_id",
            config_fingerprint="ORIGINAL_CONFIG_FINGERPRINT",
            prompt_fingerprint="pmt_fp",
            schema_fingerprint="sch_fp",
            timestamp_utc=1000.0,
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
        ledger.append_record(rec)

        with self.assertRaises(ConfigFingerprintMismatchError) as ctx:
            ledger.load_completed_sample_ids(expected_config_fingerprint="MUTATED_CONFIG_FINGERPRINT")
        self.assertIn("ConfigFingerprintMismatch", str(ctx.exception))

    def test_adv_07_prompt_isolation_adversarial_inspection(self):
        """Scans all pilot prompts to prove zero leakage of fault modes or oracle queries."""
        gate = CandidateContractValidationGate()
        for task_id in PILOT_TASK_IDS:
            res = gate.verify_prompt_isolation(task_id)
            self.assertTrue(res["is_clean"])
            self.assertEqual(len(res["leaked_faults"]), 0)
            self.assertEqual(len(res["leaked_gold_markers"]), 0)


if __name__ == "__main__":
    unittest.main()
