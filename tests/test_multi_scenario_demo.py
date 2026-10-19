"""
test_multi_scenario_demo.py — Test suite for 4-scenario deterministic recovery engine.
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from reagent.experiments.dbops.multi_scenario_demo import (
    run_multi_scenario_demonstration,
    SCENARIO_CONFIGS,
)


class TestMultiScenarioDemo(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_multi_scenario_")
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_all_scenario_keys_registered(self):
        """SCENARIO_CONFIGS contains all 4 expected scenario keys."""
        expected = {"ambiguous_timeout", "safe_retry", "unsafe_retry_prevention", "conflicting_evidence"}
        self.assertEqual(set(SCENARIO_CONFIGS.keys()), expected)

    def test_02_ambiguous_timeout_scenario(self):
        """Under ambiguous_timeout, ReAgent restrains retry, preserving $500, while Naive double-debits."""
        artifact = run_multi_scenario_demonstration(
            scenario="ambiguous_timeout",
            output_dir=str(self.temp_path),
        )
        self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED")
        self.assertEqual(artifact.reagent_arm.net_transferred, 500.00)
        self.assertEqual(artifact.reagent_arm.tool_attempts, 1)
        self.assertFalse(artifact.reagent_arm.duplicate_occurred)

        self.assertEqual(artifact.naive_retry_arm.final_status, "UNVERIFIED_CORRUPTED")
        self.assertEqual(artifact.naive_retry_arm.net_transferred, 1000.00)
        self.assertTrue(artifact.naive_retry_arm.duplicate_occurred)

    def test_03_safe_retry_scenario(self):
        """Under safe_retry (FAIL_BEFORE_COMMIT), ReAgent confirms absence, safely retries, and verifies."""
        artifact = run_multi_scenario_demonstration(
            scenario="safe_retry",
            output_dir=str(self.temp_path),
        )
        self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED")
        self.assertEqual(artifact.reagent_arm.net_transferred, 500.00)
        self.assertEqual(artifact.reagent_arm.tool_attempts, 2)
        self.assertFalse(artifact.reagent_arm.duplicate_occurred)
        self.assertEqual(artifact.naive_retry_arm.final_status, "UNVERIFIED_RETRY")

    def test_04_unsafe_retry_prevention_scenario(self):
        """Under unsafe_retry_prevention, ReAgent blocks duplicate write on non-idempotent ledger."""
        artifact = run_multi_scenario_demonstration(
            scenario="unsafe_retry_prevention",
            output_dir=str(self.temp_path),
        )
        self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED")
        self.assertEqual(artifact.reagent_arm.net_transferred, 500.00)
        self.assertEqual(artifact.reagent_arm.tool_attempts, 1)
        self.assertFalse(artifact.reagent_arm.duplicate_occurred)
        self.assertTrue(artifact.naive_retry_arm.duplicate_occurred)
        self.assertEqual(artifact.naive_retry_arm.net_transferred, 1000.00)

    def test_05_conflicting_evidence_scenario(self):
        """Under conflicting_evidence (PARTIAL_COMMIT), ReAgent safely escalates to human without duplicate retry."""
        artifact = run_multi_scenario_demonstration(
            scenario="conflicting_evidence",
            output_dir=str(self.temp_path),
        )
        self.assertEqual(artifact.reagent_arm.final_status, "ESCALATED")
        self.assertEqual(artifact.reagent_arm.tool_attempts, 1, "Must not retry on discrepant state!")
        self.assertFalse(artifact.reagent_arm.duplicate_occurred)
        self.assertTrue(artifact.naive_retry_arm.duplicate_occurred)


if __name__ == "__main__":
    unittest.main()
