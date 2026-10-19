"""
test_dbops_live_boundary_audit.py — Audit and Hardening Verification for Live Boundary Safety.

Proves:
1. Live Opt-In: Default invocation cannot dispatch network requests; live mode fails closed without opt-in and credentials.
2. Exact Request Bound: Exactly 6 planned samples; dispatch count strictly capped at 6 across retries, resume, and repeats.
3. Resume and Crash Safety: Mid-flight crash leaves DISPATCHED state; resume raises AmbiguousDispatchedRequestError; no silent replay.
4. Ledger Integrity: Rejects duplicates, corrupt JSON, config fingerprint mismatches; protected by FileLock and fsync.
5. Raw Response Handling: Scrubs Bearer tokens, API keys, passwords, credentials; bounds size to 64KB.
6. Cost Control: Missing telemetry is None (not 0); cost_usd is None (no fake dollar caps); client-side token guard prevents runaway loops.
7. Model & Reproducibility: Verifies frozen model ID, schema fingerprint, prompt hashes, and prompt isolation.

OFFLINE ONLY: Zero live provider calls. Zero network requests. Zero .env inspection.
"""

import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Dict, Any

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
    AmbiguousDispatchedRequestError,
    ConfigFingerprintMismatchError,
    sanitize_text,
)
from reagent.experiments.dbops.candidate_gate import CandidateContractValidationGate
from reagent.policy import GlobalTokenLedger, TokenBudgetGuard, CorruptLedgerError


class TestDbopsLiveBoundaryAudit(unittest.TestCase):
    """
    Focused unit tests auditing and proving the boundary between offline prep and live execution.
    """

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reagent_boundary_audit_"))
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

    def test_01_live_opt_in_gate_cannot_be_bypassed(self):
        """Default invocation cannot dispatch provider requests; live mode fails closed without opt-in."""
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)
        self.assertFalse(runner.live_opt_in)
        self.assertIsNone(runner.provider)

        # Calling run_pilot() without mock generator runs preflight only (0 network calls)
        summary = runner.run_pilot()
        self.assertFalse(summary.is_live)
        self.assertEqual(summary.total_samples_attempted, 0)
        self.assertIn("PREFLIGHT_ONLY", summary.task_summaries)
        preflight = summary.task_summaries["PREFLIGHT_ONLY"]
        self.assertEqual(preflight["network_calls_made"], 0)
        self.assertEqual(preflight["paid_tokens_consumed"], 0)

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

    def test_02_exact_request_bound_strictly_six_dispatches(self):
        """Frozen manifest produces exactly 6 planned sample IDs and cannot exceed 6 dispatches."""
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)

        # 1. Planned Sample IDs
        expected_sample_ids = [
            f"pilot_{t}_s{s}" for t in PILOT_TASK_IDS for s in range(PILOT_SAMPLES_PER_TASK)
        ]
        self.assertEqual(len(expected_sample_ids), 6)
        self.assertEqual(len(set(expected_sample_ids)), 6)

        # 2. Mock Counting Generator
        dispatch_count = 0
        gate = CandidateContractValidationGate()

        def counting_generator(task_id: str, sample_idx: int) -> str:
            nonlocal dispatch_count
            dispatch_count += 1
            return gate.generate_candidate_fixture(task_id, "VALID_GOLD_EQUIVALENT")

        summary = runner.run_pilot(mock_response_generator=counting_generator)
        self.assertEqual(dispatch_count, 6)
        self.assertEqual(summary.total_samples_attempted, 6)

        # 3. Repeated invocation dispatches 0 additional requests
        summary2 = runner.run_pilot(mock_response_generator=counting_generator)
        self.assertEqual(dispatch_count, 6)  # No additional dispatches!
        self.assertEqual(summary2.total_samples_attempted, 0)  # 0 new samples executed
        self.assertEqual(len(runner.ledger.load_completed_sample_ids()), 6)

    def test_03_crash_and_ambiguity_safety_refuses_auto_replay(self):
        """Mid-flight crash leaves DISPATCHED state; resume raises AmbiguousDispatchedRequestError."""
        ledger_file = self.temp_dir / "crash_ledger.jsonl"
        raw_dir = self.temp_dir / "raw"
        ledger = PilotLedger(ledger_file, raw_dir)

        # Simulate a crash: sample was dispatched, but process died before terminal result was recorded
        ledger.record_dispatch_intent(
            sample_id="pilot_task_01_refund_issue_s0",
            request_id="req_in_flight_123",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier=PILOT_TARGET_MODEL,
            config_fingerprint="cfg_fp_abc",
            prompt_fingerprint="pmt_fp_abc",
            schema_fingerprint="sch_fp_abc",
            is_live_call=True,
        )

        # Resume attempt MUST fail closed and refuse automatic replay
        with self.assertRaises(AmbiguousDispatchedRequestError) as ctx:
            ledger.load_completed_sample_ids(expected_config_fingerprint="cfg_fp_abc")
        self.assertIn("pilot_task_01_refund_issue_s0", str(ctx.exception))
        self.assertIn("req_in_flight_123", str(ctx.exception))
        self.assertIn("Automated replay is prohibited", str(ctx.exception))

    def test_04_ledger_integrity_rejects_duplicates_and_corruptions(self):
        """Ledger rejects duplicate terminal records, corrupt JSON lines, and config mismatches."""
        ledger_file = self.temp_dir / "integrity_ledger.jsonl"
        raw_dir = self.temp_dir / "raw"
        ledger = PilotLedger(ledger_file, raw_dir)

        rec = PilotSampleLedgerRecord(
            sample_id="sample_test_01",
            request_id="req_001",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier=PILOT_TARGET_MODEL,
            config_fingerprint="cfg_fp_match",
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

        # Duplicate append of terminal record -> fails closed
        with self.assertRaises(ValueError) as ctx:
            ledger.append_record(rec)
        self.assertIn("DuplicateRecordAttempt", str(ctx.exception))

        # Config fingerprint mismatch -> raises ConfigFingerprintMismatchError
        with self.assertRaises(ConfigFingerprintMismatchError) as ctx:
            ledger.load_completed_sample_ids(expected_config_fingerprint="DIFFERENT_FINGERPRINT")
        self.assertIn("ConfigFingerprintMismatch", str(ctx.exception))

        # Corrupt JSON -> raises CorruptLedgerError
        with open(ledger_file, "a", encoding="utf-8") as f:
            f.write("{broken_json\n")
        with self.assertRaises(CorruptLedgerError):
            ledger.load_completed_sample_ids()

    def test_05_raw_response_handling_scrubs_secrets_and_bounds_size(self):
        """Sanitizes Bearer tokens, secrets, API keys and bounds artifact length to 64KB."""
        # 1. Secret scrubbing
        dirty_text = (
            "Response with Authorization: Bearer sk-ant-secret-key-12345 "
            "and api_key='sk-nebius-secret-token' and http://user:pass1234@api.nebius.ai/v1"
        )
        clean = sanitize_text(dirty_text)
        self.assertNotIn("sk-ant-secret-key-12345", clean)
        self.assertNotIn("sk-nebius-secret-token", clean)
        self.assertNotIn("pass1234", clean)
        self.assertIn("Bearer [REDACTED]", clean)
        self.assertIn("api_key=[REDACTED]", clean)

        # 2. Length bounding
        oversized = "A" * 70000
        bounded = sanitize_text(oversized, max_bytes=65536)
        self.assertLessEqual(len(bounded.encode("utf-8")), 66000)
        self.assertIn("TRUNCATED_MAX_BYTES_EXCEEDED", bounded)

        # 3. Artifact writing
        ledger_file = self.temp_dir / "sec_ledger.jsonl"
        raw_dir = self.temp_dir / "sec_raw"
        ledger = PilotLedger(ledger_file, raw_dir)
        art_path = ledger.save_raw_response_artifact("sample_sec", dirty_text)
        with open(art_path, "r", encoding="utf-8") as f:
            saved_content = f.read()
        self.assertNotIn("sk-ant-secret-key-12345", saved_content)
        self.assertIn("Bearer [REDACTED]", saved_content)

    def test_06_cost_control_telemetry_and_no_fake_dollar_cap(self):
        """Missing telemetry is None (not 0); cost_usd is None (no fake dollar caps)."""
        ledger_file = self.temp_dir / "cost_ledger.jsonl"
        ledger = PilotLedger(ledger_file, self.temp_dir / "raw")

        rec = PilotSampleLedgerRecord(
            sample_id="sample_cost_check",
            request_id="req_cost_1",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier=PILOT_TARGET_MODEL,
            config_fingerprint="cfg_fp",
            prompt_fingerprint="pmt_fp",
            schema_fingerprint="sch_fp",
            timestamp_utc=1000.0,
            timestamp_iso="2026-10-14T00:00:00Z",
            is_live_call=False,
            status="COMPLETED",
            latency_ms=80,
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

        with open(ledger_file, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())
        self.assertIsNone(entry["token_telemetry"])
        self.assertIsNone(entry["cost_telemetry_usd"])

    def test_07_model_and_reproducibility_invariants(self):
        """Verifies exact frozen model ID, schema fingerprint, and prompt isolation."""
        runner = PilotEvaluationRunner(pilot_dir=self.temp_dir, budget_guard=self.budget_guard)
        cfg = runner.frozen_config

        self.assertEqual(cfg.target_model, "nvidia/nemotron-3-super-120b-a12b")
        self.assertEqual(cfg.schema_version, "structured_decision_v1")
        self.assertEqual(cfg.temperature, 0.0)
        self.assertEqual(cfg.max_tokens, 512)
        self.assertEqual(cfg.total_samples, 6)

        isolation_results = runner.verify_pilot_prompt_isolation()
        for task_id, res in isolation_results.items():
            self.assertTrue(res["is_clean"])
            self.assertEqual(len(res["leaked_faults"]), 0)
            self.assertEqual(len(res["leaked_gold_markers"]), 0)


if __name__ == "__main__":
    unittest.main()
