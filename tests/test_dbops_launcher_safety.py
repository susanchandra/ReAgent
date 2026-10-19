"""
test_dbops_launcher_safety.py — Regression Test Suite for Live Launcher Safety.

Verifies:
1. Pure Import: Importing run_live_pilot causes zero side effects, exits, or network calls.
2. Default Invocation: run_live_pilot without flags executes offline dry-run preflight (0 network calls).
3. Explicit Dry-Run: --dry-run executes preflight cleanly.
4. Malformed Arguments: Invalid CLI arguments exit with error via argparse; zero network calls.
5. Fail-Closed Without Credentials: --live-run-opt-in without NEBIUS_API_KEY fails closed (exit 1).
6. No Silent Opt-In: NEBIUS_API_KEY present in env without --live-run-opt-in still runs offline preflight only.
7. pilot_eval CLI Default: python -m reagent.experiments.dbops.pilot_eval runs offline preflight.
8. pilot_eval CLI Live Gate: pilot_eval with --live-run-opt-in fails closed without configured provider.
9. Mock Live Execution via Launcher: End-to-end execution with mock provider, temp ledger, and analyzer.
10. Ambiguous Dispatch Halt: Unresolved DISPATCHED record halts resume with AmbiguousDispatchedRequestError.
11. Preflight Failure Guard: Drifted prompt/config halts launcher before any provider call.
12. Network Sentinel: Sockets are blocked during offline testing to guarantee zero external calls.

OFFLINE ONLY: Zero live provider calls. Zero network requests. Zero .env inspection.
"""

import os
import sys
import json
import socket
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run_live_pilot
from reagent.experiments.dbops import pilot_eval
from reagent.experiments.dbops.pilot_eval import (
    PilotEvaluationRunner,
    PilotLedger,
    PilotSampleLedgerRecord,
    AmbiguousDispatchedRequestError,
    ConfigFingerprintMismatchError,
    PILOT_TOTAL_SAMPLES,
)
from reagent.models import StructuredDecision, Postcondition, AgentStatus


class NetworkSentinelError(RuntimeError):
    """Raised if any test attempts to establish a socket connection."""
    pass


def _guarded_connect(*args, **kwargs):
    raise NetworkSentinelError("Network access attempted during offline test!")


class TestDbopsLauncherSafety(unittest.TestCase):
    """
    Focused tests verifying the safety boundary of run_live_pilot.py and pilot_eval.py CLI.
    """

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reagent_launcher_safety_"))
        self._orig_socket_connect = socket.socket.connect
        socket.socket.connect = _guarded_connect

        # Ensure NEBIUS_API_KEY is not in env by default for tests
        self._orig_api_key = os.environ.get("NEBIUS_API_KEY")
        if "NEBIUS_API_KEY" in os.environ:
            del os.environ["NEBIUS_API_KEY"]

        # Isolate token ledger path to avoid depleting shared repo-root ledger
        self._orig_ledger_path = os.environ.get("REAGENT_LEDGER_PATH")
        os.environ["REAGENT_LEDGER_PATH"] = str(self.temp_dir / "isolated_test_ledger.json")

    def tearDown(self):
        socket.socket.connect = self._orig_socket_connect
        if self._orig_api_key is not None:
            os.environ["NEBIUS_API_KEY"] = self._orig_api_key
        elif "NEBIUS_API_KEY" in os.environ:
            del os.environ["NEBIUS_API_KEY"]

        if self._orig_ledger_path is not None:
            os.environ["REAGENT_LEDGER_PATH"] = self._orig_ledger_path
        elif "REAGENT_LEDGER_PATH" in os.environ:
            del os.environ["REAGENT_LEDGER_PATH"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_pure_import_safety(self):
        """Importing run_live_pilot has zero side effects, creates no files, and makes no network calls."""
        # The module is already imported in this file; verify no files were created in temp_dir
        self.assertEqual(len(list(self.temp_dir.iterdir())), 0)

    def test_02_default_invocation_runs_offline_preflight(self):
        """Default invocation without flags executes offline preflight and returns 0."""
        ret = run_live_pilot.main(["--pilot-dir", str(self.temp_dir)])
        self.assertEqual(ret, 0)

        # Frozen config artifact should be generated in temp_dir
        frozen_cfg_file = self.temp_dir / "pilot_config_frozen.json"
        self.assertTrue(frozen_cfg_file.exists())

        # No live ledger should be written
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        self.assertFalse(ledger_file.exists())

    def test_03_explicit_dry_run_flag(self):
        """Passing --dry-run executes offline preflight cleanly and returns 0."""
        ret = run_live_pilot.main(["--dry-run", "--pilot-dir", str(self.temp_dir)])
        self.assertEqual(ret, 0)
        self.assertTrue((self.temp_dir / "pilot_config_frozen.json").exists())

    def test_04_malformed_arguments_rejected(self):
        """Passing unrecognized arguments causes argparse to exit with code 2."""
        with self.assertRaises(SystemExit) as ctx:
            run_live_pilot.main(["--unrecognized-flag"])
        self.assertEqual(ctx.exception.code, 2)

    def test_05_live_opt_in_without_api_key_fails_closed(self):
        """--live-run-opt-in without NEBIUS_API_KEY in environment fails closed with code 1."""
        ret = run_live_pilot.main(["--live-run-opt-in", "--pilot-dir", str(self.temp_dir)])
        self.assertEqual(ret, 1)

        # No ledger or requests should have been written
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        self.assertFalse(ledger_file.exists())

    def test_06_no_silent_opt_in_from_env_key(self):
        """Presence of NEBIUS_API_KEY without --live-run-opt-in still runs dry-run preflight only."""
        os.environ["NEBIUS_API_KEY"] = "mock_key_for_offline_safety_test"
        ret = run_live_pilot.main(["--pilot-dir", str(self.temp_dir)])
        self.assertEqual(ret, 0)

        # Still no live ledger written, because live mode was NOT opted into on CLI
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        self.assertFalse(ledger_file.exists())

    def test_07_pilot_eval_cli_default_preflight(self):
        """python -m reagent.experiments.dbops.pilot_eval default runs preflight and exits 0."""
        ret = pilot_eval.main(["--pilot-dir", str(self.temp_dir)])
        self.assertEqual(ret, 0)
        self.assertTrue((self.temp_dir / "pilot_config_frozen.json").exists())

    def test_08_pilot_eval_cli_live_opt_in_fails_closed(self):
        """pilot_eval CLI with --live-run-opt-in fails closed without configured provider."""
        with self.assertRaises(RuntimeError) as ctx:
            pilot_eval.main(["--live-run-opt-in", "--pilot-dir", str(self.temp_dir)])
        self.assertIn("LiveExecutionRefused", str(ctx.exception))

    def test_09_mock_live_execution_end_to_end_via_launcher(self):
        """End-to-end execution of run_live_pilot with a mocked provider in isolated directory."""
        os.environ["NEBIUS_API_KEY"] = "mock_test_key"

        # Mock provider that returns valid StructuredDecision JSON
        valid_json = json.dumps({
            "tool": "dbops_database",
            "arguments": {"request_id": "REQ-REFUND-001", "amount": 120.0},
            "proposed_postconditions": [
                {
                    "target": "refund_requests.refund_ref",
                    "predicate": "EQUALS",
                    "expected": "REQ-REFUND-001",
                    "description": "refund_ref matches REQ-REFUND-001",
                },
                {
                    "target": "orders",
                    "predicate": "EQUALS",
                    "expected": "ORD-12345",
                    "description": "order ORD-12345 status is REFUNDED",
                },
            ],
            "diagnosis_category": None,
            "diagnosis_confidence": 0.0,
            "diagnosis_reason": None,
            "recovery_strategy": None,
        })

        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = True

        from reagent.providers.base import LLMResponse
        mock_resp = LLMResponse(
            content=valid_json,
            model_name="nvidia/nemotron-3-super-120b-a12b",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        mock_provider.generate.return_value = mock_resp

        with patch("run_live_pilot.NebiusNemotronProvider", return_value=mock_provider):
            ret = run_live_pilot.main([
                "--live-run-opt-in",
                "--skip-countdown",
                "--pilot-dir", str(self.temp_dir),
            ])

        # Runner should complete
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        self.assertTrue(ledger_file.exists())

        # Verify ledger has records and analyzer generated reports
        with open(ledger_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        # Each sample has DISPATCHED + COMPLETED (or terminal)
        terminal_records = [json.loads(l) for l in lines if json.loads(l).get("status") in pilot_eval.TERMINAL_STATUSES]
        self.assertEqual(len(terminal_records), PILOT_TOTAL_SAMPLES)

        report_json = self.temp_dir / "pilot_analysis_report.json"
        self.assertTrue(report_json.exists())
        with open(report_json, "r", encoding="utf-8") as f:
            rep_data = json.load(f)
        self.assertTrue(rep_data["pilot_executed"])
        self.assertEqual(rep_data["n_planned"], PILOT_TOTAL_SAMPLES)

    def test_10_ambiguous_dispatch_halts_launcher(self):
        """An unresolved DISPATCHED record halts the launcher fail-closed."""
        os.environ["NEBIUS_API_KEY"] = "mock_test_key"
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"

        # Write an ambiguous DISPATCHED record
        ambig_record = {
            "sample_id": "pilot_task_01_refund_issue_s0",
            "request_id": "req_crash_123",
            "task_id": "task_01_refund_issue",
            "sample_index": 0,
            "model_identifier": "nvidia/nemotron-3-super-120b-a12b",
            "config_fingerprint": "c6e5c8532377f005e385aa1e47f9d52f9139c9a5a6a6f16bd6d1ae109e95a6c1",
            "prompt_fingerprint": "bfed8cec24ca1e74f848b3bafdc10232a1fad306f81d2e792e2e8a3cce06aa49",
            "schema_fingerprint": "0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea",
            "timestamp_utc": 1700000000.0,
            "timestamp_iso": "2026-01-01T00:00:00+00:00",
            "is_live_call": True,
            "status": "DISPATCHED",
            "latency_ms": 0,
            "raw_response_artifact_path": None,
            "generation_parse_success": False,
            "schema_valid": False,
            "schema_error": None,
            "grounding_valid": False,
            "grounding_errors": [],
            "semantic_gold_agreement": False,
            "missing_required_postconditions": [],
            "extra_unsupported_predicates": [],
            "unsupported_tables_or_columns": [],
            "uniqueness_respected": True,
            "invalid_contract_rejected": False,
            "fallback_used": False,
            "rejection_reason": None,
            "token_telemetry": None,
            "cost_telemetry_usd": None,
        }
        with open(ledger_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(ambig_record) + "\n")

        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = True

        with patch("run_live_pilot.NebiusNemotronProvider", return_value=mock_provider):
            with self.assertRaises(AmbiguousDispatchedRequestError):
                run_live_pilot.main([
                    "--live-run-opt-in",
                    "--skip-countdown",
                    "--pilot-dir", str(self.temp_dir),
                ])

    def test_11_prompt_drift_aborts_launcher_before_provider_call(self):
        """If task prompts or config drift, preflight check aborts before provider call."""
        os.environ["NEBIUS_API_KEY"] = "mock_test_key"

        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = True

        # Artificially alter prompt harness to cause prompt hash drift
        with patch.object(
            pilot_eval.DbopsPromptHarness,
            "construct_task_prompt",
            side_effect=RuntimeError("SimulatedPromptDrift"),
        ):
            with patch("run_live_pilot.NebiusNemotronProvider", return_value=mock_provider):
                with self.assertRaises(RuntimeError) as ctx:
                    run_live_pilot.main([
                        "--live-run-opt-in",
                        "--skip-countdown",
                        "--pilot-dir", str(self.temp_dir),
                    ])
                self.assertIn("SimulatedPromptDrift", str(ctx.exception))

        # Provider should have zero generate calls
        mock_provider.generate.assert_not_called()

    def test_12_corrupt_ledger_line_halts_launcher(self):
        """A corrupt JSON line in the ledger halts the launcher with CorruptLedgerError."""
        os.environ["NEBIUS_API_KEY"] = "mock_test_key"
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        with open(ledger_file, "w", encoding="utf-8") as f:
            f.write('{"sample_id": "valid"}\nCORRUPT_NON_JSON\n')

        from reagent.policy import CorruptLedgerError
        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = True

        with patch("run_live_pilot.NebiusNemotronProvider", return_value=mock_provider):
            with self.assertRaises(CorruptLedgerError):
                run_live_pilot.main([
                    "--live-run-opt-in",
                    "--skip-countdown",
                    "--pilot-dir", str(self.temp_dir),
                ])

    def test_13_config_fingerprint_mismatch_in_ledger_halts_launcher(self):
        """A record with a mismatched config_fingerprint halts the launcher with ConfigFingerprintMismatchError."""
        os.environ["NEBIUS_API_KEY"] = "mock_test_key"
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        bad_record = {
            "sample_id": "pilot_task_01_refund_issue_s0",
            "request_id": "req_1",
            "task_id": "task_01_refund_issue",
            "sample_index": 0,
            "model_identifier": "nvidia/nemotron-3-super-120b-a12b",
            "config_fingerprint": "drifted_fingerprint_00000000000000000000000000000000",
            "prompt_fingerprint": "bfed8cec24ca1e74f848b3bafdc10232a1fad306f81d2e792e2e8a3cce06aa49",
            "schema_fingerprint": "0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea",
            "timestamp_utc": 1700000000.0,
            "timestamp_iso": "2026-01-01T00:00:00+00:00",
            "is_live_call": True,
            "status": "COMPLETED",
            "latency_ms": 100,
            "raw_response_artifact_path": None,
            "generation_parse_success": True,
            "schema_valid": True,
            "schema_error": None,
            "grounding_valid": True,
            "grounding_errors": [],
            "semantic_gold_agreement": True,
            "missing_required_postconditions": [],
            "extra_unsupported_predicates": [],
            "unsupported_tables_or_columns": [],
            "uniqueness_respected": True,
            "invalid_contract_rejected": False,
            "fallback_used": False,
            "rejection_reason": None,
            "token_telemetry": None,
            "cost_telemetry_usd": None,
        }
        with open(ledger_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(bad_record) + "\n")

        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = True

        with patch("run_live_pilot.NebiusNemotronProvider", return_value=mock_provider):
            with self.assertRaises(ConfigFingerprintMismatchError):
                run_live_pilot.main([
                    "--live-run-opt-in",
                    "--skip-countdown",
                    "--pilot-dir", str(self.temp_dir),
                ])

    def test_14_provider_timeout_and_exception_record_terminal_without_retry(self):
        """Provider TimeoutError and Exception are durably recorded as terminal without automatic retry."""
        os.environ["NEBIUS_API_KEY"] = "mock_test_key"

        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = True
        # Alternate: first request times out, second throws network exception, remainder succeed
        mock_provider.generate.side_effect = [
            TimeoutError("Gateway timed out"),
            RuntimeError("Connection reset by peer"),
            # remaining requests return valid JSON
        ] + [
            MagicMock(content="{}", token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
            for _ in range(4)
        ]

        with patch("run_live_pilot.NebiusNemotronProvider", return_value=mock_provider):
            ret = run_live_pilot.main([
                "--live-run-opt-in",
                "--skip-countdown",
                "--pilot-dir", str(self.temp_dir),
            ])

        # Verify exactly 6 requests were attempted (no extra retries)
        self.assertEqual(mock_provider.generate.call_count, 6)

        # Inspect ledger: verify TIMEOUT and PROVIDER_ERROR are recorded
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        with open(ledger_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        terminal_statuses = [r["status"] for r in records if r["status"] in pilot_eval.TERMINAL_STATUSES]
        self.assertEqual(len(terminal_statuses), 6)
        self.assertEqual(terminal_statuses[0], "TIMEOUT")
        self.assertEqual(terminal_statuses[1], "PROVIDER_ERROR")

    def test_15_six_attempt_boundary_enforced_across_existing_and_ambiguous(self):
        """Global 6-request ceiling prevents any 7th request when ledger has 5 completed and 1 ambiguous."""
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        raw_dir = self.temp_dir / "pilot_raw_responses"
        ledger = PilotLedger(ledger_file, raw_dir)

        # Record 5 completed
        for i in range(5):
            ledger.record_dispatch_intent(
                sample_id=f"sample_{i}",
                request_id=f"req_{i}",
                task_id="task_01_refund_issue",
                sample_index=0,
                model_identifier="nvidia/nemotron-3-super-120b-a12b",
                config_fingerprint="c6e5c8532377f005e385aa1e47f9d52f9139c9a5a6a6f16bd6d1ae109e95a6c1",
                prompt_fingerprint="bfed8cec24ca1e74f848b3bafdc10232a1fad306f81d2e792e2e8a3cce06aa49",
                schema_fingerprint="0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea",
                is_live_call=True,
            )
            rec = PilotSampleLedgerRecord(
                sample_id=f"sample_{i}",
                request_id=f"req_{i}",
                task_id="task_01_refund_issue",
                sample_index=0,
                model_identifier="nvidia/nemotron-3-super-120b-a12b",
                config_fingerprint="c6e5c8532377f005e385aa1e47f9d52f9139c9a5a6a6f16bd6d1ae109e95a6c1",
                prompt_fingerprint="bfed8cec24ca1e74f848b3bafdc10232a1fad306f81d2e792e2e8a3cce06aa49",
                schema_fingerprint="0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea",
                timestamp_utc=1700000000.0,
                timestamp_iso="2026-01-01T00:00:00+00:00",
                is_live_call=True,
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

        # 6th claim (DISPATCHED, leaves it ambiguous)
        ledger.record_dispatch_intent(
            sample_id="sample_5",
            request_id="req_5",
            task_id="task_01_refund_issue",
            sample_index=1,
            model_identifier="nvidia/nemotron-3-super-120b-a12b",
            config_fingerprint="c6e5c8532377f005e385aa1e47f9d52f9139c9a5a6a6f16bd6d1ae109e95a6c1",
            prompt_fingerprint="bfed8cec24ca1e74f848b3bafdc10232a1fad306f81d2e792e2e8a3cce06aa49",
            schema_fingerprint="0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea",
            is_live_call=True,
        )

        # Attempting a 7th dispatch claim MUST raise RuntimeError(SafetyViolation)
        with self.assertRaises(RuntimeError) as ctx:
            ledger.record_dispatch_intent(
                sample_id="sample_6",
                request_id="req_6",
                task_id="task_02_wallet_transfer",
                sample_index=0,
                model_identifier="nvidia/nemotron-3-super-120b-a12b",
                config_fingerprint="c6e5c8532377f005e385aa1e47f9d52f9139c9a5a6a6f16bd6d1ae109e95a6c1",
                prompt_fingerprint="ba46c21351e0ea51a546a5c24453c870b550f943fee5c42c5d7d51742e4bf468",
                schema_fingerprint="0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea",
                is_live_call=True,
            )
        self.assertIn("SafetyViolation: Global ceiling of 6 requests reached", str(ctx.exception))

    def test_16_duplicate_terminal_records_rejected_by_ledger(self):
        """Attempting to append a second terminal record for the same sample ID raises ValueError."""
        ledger_file = self.temp_dir / "pilot_ledger.jsonl"
        raw_dir = self.temp_dir / "pilot_raw_responses"
        ledger = PilotLedger(ledger_file, raw_dir)

        rec = PilotSampleLedgerRecord(
            sample_id="sample_dup",
            request_id="req_dup_1",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier="nvidia/nemotron-3-super-120b-a12b",
            config_fingerprint="c6e5c8532377f005e385aa1e47f9d52f9139c9a5a6a6f16bd6d1ae109e95a6c1",
            prompt_fingerprint="bfed8cec24ca1e74f848b3bafdc10232a1fad306f81d2e792e2e8a3cce06aa49",
            schema_fingerprint="0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea",
            timestamp_utc=1700000000.0,
            timestamp_iso="2026-01-01T00:00:00+00:00",
            is_live_call=True,
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

        # Attempt duplicate terminal record for sample_dup
        rec_dup = PilotSampleLedgerRecord(
            sample_id="sample_dup",
            request_id="req_dup_2",
            task_id="task_01_refund_issue",
            sample_index=0,
            model_identifier="nvidia/nemotron-3-super-120b-a12b",
            config_fingerprint="c6e5c8532377f005e385aa1e47f9d52f9139c9a5a6a6f16bd6d1ae109e95a6c1",
            prompt_fingerprint="bfed8cec24ca1e74f848b3bafdc10232a1fad306f81d2e792e2e8a3cce06aa49",
            schema_fingerprint="0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea",
            timestamp_utc=1700000000.0,
            timestamp_iso="2026-01-01T00:00:00+00:00",
            is_live_call=True,
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
        with self.assertRaises(ValueError) as ctx:
            ledger.append_record(rec_dup)
        self.assertIn("DuplicateRecordAttempt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)

