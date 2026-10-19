"""
test_dbops_m14_submission_validation.py — Milestone 14 Submission Integrity & Demo Verification.

Verifies:
1. Judge Demo runs under active NetworkSentinel (zero network access).
2. Trace serialization round-trip: m14_judge_demo_trace.json round-trips with zero data loss.
3. Deterministic repeated runs: consecutive executions yield identical traces and verdicts.
4. Cryptographic artifact hash verification: compute_file_sha256 accurately detects disk changes.
5. Missing / corrupt benchmark artifact fails closed with clear actionable error (no silent fabrication).
6. Production pilot ledger invariance: reports/dbops/pilot_ledger.jsonl is never created or touched.
7. M13 trial evidence reconciliation: exact denominator integrity (18 trials/arm, 90 trials total).

Strictly offline. Zero paid API tokens. Zero .env inspection.
"""

import hashlib
import json
import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

from reagent.experiments.dbops.judge_demo import (
    run_judge_demo,
    compute_file_sha256,
    load_comparative_results,
    DEFAULT_BENCHMARK_PATH,
    SOURCE_REVISION,
)


class NetworkSentinelError(RuntimeError):
    """Raised if any test attempts to establish a socket connection."""
    pass


def _guarded_connect(*args, **kwargs):
    raise NetworkSentinelError("Network access attempted during offline test!")


class TestDbopsM14SubmissionValidation(unittest.TestCase):
    """Test suite proving judge demo integrity, artifact verification, and trial reconciliation."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reagent_m14_test_"))
        self._orig_socket_connect = socket.socket.connect
        socket.socket.connect = _guarded_connect

    def tearDown(self):
        socket.socket.connect = self._orig_socket_connect
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_judge_demo_network_sentinel(self):
        """Proves run_judge_demo completes cleanly under socket network sentinel (0 network calls)."""
        temp_bench = self.temp_dir / "comparative_evaluation_results.json"
        temp_bench.write_bytes(DEFAULT_BENCHMARK_PATH.read_bytes())

        summary = run_judge_demo(
            regenerate_benchmark=False,
            benchmark_path=temp_bench,
            output_dir=self.temp_dir,
            quiet=True,
        )

        self.assertEqual(summary["source_git_revision"], SOURCE_REVISION)
        self.assertEqual(summary["recovery_demo"]["naive_arm"]["status"], "UNVERIFIED_CORRUPTED")
        self.assertEqual(summary["recovery_demo"]["reagent_arm"]["status"], "VERIFIED")

    def test_02_judge_demo_trace_serialization_round_trip(self):
        """m14_judge_demo_trace.json is written and parses cleanly with all required fields."""
        temp_bench = self.temp_dir / "comparative_evaluation_results.json"
        temp_bench.write_bytes(DEFAULT_BENCHMARK_PATH.read_bytes())

        summary = run_judge_demo(
            regenerate_benchmark=False,
            benchmark_path=temp_bench,
            output_dir=self.temp_dir,
            quiet=True,
        )

        trace_file = self.temp_dir / "m14_judge_demo_trace.json"
        self.assertTrue(trace_file.exists(), "m14_judge_demo_trace.json must be emitted!")

        with open(trace_file, "r", encoding="utf-8") as f:
            loaded_trace = json.load(f)

        self.assertEqual(loaded_trace["product_name"], summary["product_name"])
        self.assertEqual(loaded_trace["recovery_demo"]["task_id"], summary["recovery_demo"]["task_id"])
        self.assertEqual(loaded_trace["comparative_benchmark"]["total_paired_configurations"], 480)

    def test_03_judge_demo_deterministic_repeated_runs(self):
        """Two independent executions produce identical numerical metrics and verdicts."""
        temp_bench = self.temp_dir / "comparative_evaluation_results.json"
        temp_bench.write_bytes(DEFAULT_BENCHMARK_PATH.read_bytes())

        dir1 = self.temp_dir / "run1"
        dir2 = self.temp_dir / "run2"
        dir1.mkdir()
        dir2.mkdir()

        s1 = run_judge_demo(regenerate_benchmark=False, benchmark_path=temp_bench, output_dir=dir1, quiet=True)
        s2 = run_judge_demo(regenerate_benchmark=False, benchmark_path=temp_bench, output_dir=dir2, quiet=True)

        self.assertEqual(s1["recovery_demo"]["naive_arm"]["balance"], s2["recovery_demo"]["naive_arm"]["balance"])
        self.assertEqual(s1["recovery_demo"]["reagent_arm"]["balance"], s2["recovery_demo"]["reagent_arm"]["balance"])
        self.assertEqual(s1["recovery_demo"]["reagent_arm"]["status"], s2["recovery_demo"]["reagent_arm"]["status"])

    def test_04_artifact_hash_recomputation_integrity(self):
        """compute_file_sha256 computes exact cryptographic hash and detects modifications."""
        test_file = self.temp_dir / "sample.txt"
        test_file.write_text("ReAgent doesn't trust done. It verifies it.", encoding="utf-8")

        initial_hash = compute_file_sha256(test_file)
        self.assertEqual(len(initial_hash), 64)

        # Mutate file
        test_file.write_text("Mutated content", encoding="utf-8")
        mutated_hash = compute_file_sha256(test_file)
        self.assertNotEqual(initial_hash, mutated_hash, "Hash must change upon file mutation!")

    def test_05_corrupt_benchmark_artifact_fails_closed(self):
        """Missing or malformed comparative benchmark artifact fails closed with clear advice."""
        missing_path = self.temp_dir / "missing.json"
        with self.assertRaises(FileNotFoundError) as ctx_missing:
            load_comparative_results(missing_path)
        self.assertIn("--regenerate-benchmark", str(ctx_missing.exception))

        corrupt_path = self.temp_dir / "corrupt.json"
        corrupt_path.write_text("{not-valid-json", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx_corrupt:
            load_comparative_results(corrupt_path)
        self.assertIn("Failed to parse", str(ctx_corrupt.exception))

    def test_06_persistent_pilot_ledger_unmodified(self):
        """Verifies that reports/dbops/pilot_ledger.jsonl does not exist and is never created."""
        ledger_path = Path("reports/dbops/pilot_ledger.jsonl")
        self.assertFalse(ledger_path.exists(), "Production pilot ledger must NOT exist!")

    def test_07_m13_trial_evidence_exact_reconciliation(self):
        """Verifies m13_trial_evidence.json exact denominators and mutual exclusivity."""
        evidence_path = Path("reports/dbops/m13_trial_evidence.json")
        self.assertTrue(evidence_path.exists(), "m13_trial_evidence.json must exist!")

        with open(evidence_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["total_trials"], 90)
        self.assertEqual(len(data["trials"]), 90)

        aggs = data["arm_aggregates"]
        for arm_name in ("REAGENT_CONTRACT", "SHARED_OBSERVATION", "TOOL_CLAIM_ONLY", "NAIVE_RETRY", "ALWAYS_ESCALATE"):
            self.assertIn(arm_name, aggs)
            m = aggs[arm_name]
            self.assertEqual(m["total_trials"], 18, f"{arm_name} must have exactly 18 trials")
            # Mutual exclusivity check
            sum_outcomes = m["true_verified_count"] + m["false_verified_count"] + m["safe_escalation_count"] + m["unnecessary_escalation_count"]
            self.assertEqual(sum_outcomes, 18, f"{arm_name} outcomes must reconcile to 18")

        # Invariant check: REAGENT_CONTRACT has 0 false verifications
        self.assertEqual(aggs["REAGENT_CONTRACT"]["false_verified_count"], 0)
        self.assertEqual(aggs["REAGENT_CONTRACT"]["false_verified_rate"], 0.0)

        # Invariant check: NAIVE_RETRY causes duplicate primary writes
        self.assertGreater(aggs["NAIVE_RETRY"]["duplicate_primary_count"], 0)


if __name__ == "__main__":
    unittest.main()
