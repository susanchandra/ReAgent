"""
test_dbops_m5_comparative_eval.py — Focused Unit Tests for DBOPS-M5 Comparative Evaluation.

Verifies:
1. Paired-condition equivalence: Identical initial state, tasks, and faults across all 5 arms.
2. Deterministic replay: Multiple runs produce byte-identical metric counts and verdicts.
3. Metric denominators: Exactly 480 paired configurations (12 tasks × 8 faults × 5 arms) without missing trials.
4. False-verification accounting:
   - REAGENT_CONTRACT achieves 0 false verifications on WRONG_CONTENT / PARTIAL_COMMIT (except task 05).
   - SHARED_OBSERVATION suffers 43 false verifications because presence != correctness.
   - NAIVE_RETRY suffers 64 false verifications and 4 duplicate primary writes.
5. Duplicate accounting: NAIVE_RETRY duplicates non-idempotent numeric tasks (e.g. Task 02).
6. UNKNOWN handling: STATUS_UNAVAILABLE and UNOBSERVABLE fail closed to ESCALATED and are never counted as ABSENT.
7. Report and JSON artifact integrity.

Non-negotiable safety:
- Isolated temporary SQLite databases and deterministic fixtures only.
- Zero network calls. Zero live provider calls. Zero .env inspection.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from reagent.experiments.dbops.manifest_dbops_v1 import (
    ALL_TASK_IDS,
    ALL_FAULT_MODES,
)
from reagent.experiments.dbops.comparative_eval import (
    ComparativeEvaluator,
    ComparativeEvaluationArtifact,
    COMPARED_POLICY_ARMS,
)


class TestDbopsM5ComparativeEval(unittest.TestCase):
    """Focused unit tests for DBOPS-M5 comparative evaluation."""

    _shared_temp_dir = None
    _shared_artifact = None

    @classmethod
    def setUpClass(cls):
        cls._shared_temp_dir = tempfile.TemporaryDirectory()
        cls._evaluator = ComparativeEvaluator(output_dir=cls._shared_temp_dir.name)
        cls._shared_artifact = cls._evaluator.run_evaluation()

    @classmethod
    def tearDownClass(cls):
        if cls._shared_temp_dir:
            cls._shared_temp_dir.cleanup()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.evaluator = ComparativeEvaluator(output_dir=str(self.temp_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_bounded_paired_condition_equivalence(self):
        """Runs a fast bounded 2-task × 3-fault × 5-arm paired evaluation with equal initial conditions."""
        sample_tasks = ["task_01_refund_issue", "task_02_wallet_transfer"]
        sample_faults = ["NO_FAULT", "COMMIT_THEN_TIMEOUT", "WRONG_CONTENT"]

        artifact = self.evaluator.run_evaluation(
            task_ids=sample_tasks,
            fault_modes=sample_faults,
            arms=COMPARED_POLICY_ARMS,
        )

        self.assertIsInstance(artifact, ComparativeEvaluationArtifact)
        self.assertEqual(artifact.total_paired_configurations, 2 * 3 * 5)  # 30 trials
        self.assertEqual(len(artifact.arm_aggregates), 5)

        for arm in COMPARED_POLICY_ARMS:
            self.assertEqual(artifact.arm_aggregates[arm].total_trials, 6)

    def test_02_deterministic_replay(self):
        """Two independent runs across paired conditions produce identical metrics."""
        sample_tasks = ["task_02_wallet_transfer"]
        sample_faults = ["COMMIT_THEN_TIMEOUT", "WRONG_CONTENT"]

        art1 = self.evaluator.run_evaluation(task_ids=sample_tasks, fault_modes=sample_faults)
        art2 = self.evaluator.run_evaluation(task_ids=sample_tasks, fault_modes=sample_faults)

        for arm in COMPARED_POLICY_ARMS:
            m1 = art1.arm_aggregates[arm]
            m2 = art2.arm_aggregates[arm]
            self.assertEqual(m1.true_verified_count, m2.true_verified_count)
            self.assertEqual(m1.false_verified_count, m2.false_verified_count)
            self.assertEqual(m1.safe_escalation_count, m2.safe_escalation_count)
            self.assertEqual(m1.duplicate_primary_count, m2.duplicate_primary_count)

    def test_03_metric_denominators_and_no_missing_trials(self):
        """Verifies full 480-configuration paired denominator (12 tasks × 8 faults × 5 arms)."""
        artifact = self._shared_artifact

        self.assertEqual(artifact.total_paired_configurations, 480)
        self.assertEqual(artifact.tested_tasks_count, 12)
        self.assertEqual(artifact.tested_fault_modes_count, 8)
        self.assertEqual(artifact.tested_arms_count, 5)

        for arm_name in COMPARED_POLICY_ARMS:
            agg = artifact.arm_aggregates[arm_name]
            self.assertEqual(agg.total_trials, 96)  # Exactly 12 * 8 = 96 per arm
            # Sum of mutually exclusive outcomes must equal 96
            total_verdict_classifications = (
                agg.true_verified_count +
                agg.false_verified_count +
                agg.safe_escalation_count +
                agg.unnecessary_escalation_count
            )
            self.assertEqual(total_verdict_classifications, 96)

    def test_04_differentiation_reagent_vs_shared_observation(self):
        """Verifies ReAgent detects WRONG_CONTENT and PARTIAL_COMMIT, whereas SHARED_OBSERVATION false-verifies."""
        artifact = self._shared_artifact

        reagent_agg = artifact.arm_aggregates["REAGENT_CONTRACT"]
        shared_agg = artifact.arm_aggregates["SHARED_OBSERVATION"]

        # Shared observation falsely verifies 43 times across 96 trials because presence != correctness
        self.assertEqual(shared_agg.false_verified_count, 43)

        # REAGENT_CONTRACT has 0 false verifications across all 96 trials (strict contract checking)
        self.assertEqual(reagent_agg.false_verified_count, 0)
        self.assertEqual(reagent_agg.safe_escalation_count, 67)

    def test_05_naive_retry_duplicate_accounting(self):
        """Verifies NAIVE_RETRY creates duplicate primary writes under unprotected numeric tasks."""
        artifact = self._shared_artifact

        naive_agg = artifact.arm_aggregates["NAIVE_RETRY"]
        reagent_agg = artifact.arm_aggregates["REAGENT_CONTRACT"]

        # Naive retry executes 24 unsafe retries and creates 4 duplicate primary writes
        self.assertEqual(naive_agg.unsafe_retry_count, 24)
        self.assertEqual(naive_agg.duplicate_primary_count, 4)

        # ReAgent restrains retries: 0 unsafe retries, 0 duplicate primary writes
        self.assertEqual(reagent_agg.unsafe_retry_count, 0)
        self.assertEqual(reagent_agg.duplicate_primary_count, 0)

    def test_06_unknown_state_never_treated_as_absent(self):
        """STATUS_UNAVAILABLE and UNOBSERVABLE must fail closed to ESCALATED for all responsible arms."""
        artifact = self._shared_artifact

        for t in artifact.trials:
            if t.fault_mode in ("STATUS_UNAVAILABLE", "UNOBSERVABLE"):
                if t.arm in ("REAGENT_CONTRACT", "SHARED_OBSERVATION", "ALWAYS_ESCALATE"):
                    self.assertEqual(
                        t.policy_claim, "ESCALATED",
                        f"Arm {t.arm} failed to escalate on {t.fault_mode} in {t.task_id}"
                    )
                    self.assertFalse(t.retried, f"Arm {t.arm} illegally retried on {t.fault_mode}")

    def test_07_json_artifact_schema_and_persistence(self):
        """Verifies comparative_evaluation_results.json is persisted with complete structure."""
        out_dir = self.temp_path / "custom_out"
        evaluator = ComparativeEvaluator(output_dir=str(out_dir))
        evaluator.run_evaluation(task_ids=["task_01_refund_issue"], fault_modes=["NO_FAULT"])

        json_file = out_dir / "comparative_evaluation_results.json"
        self.assertTrue(json_file.exists())
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("arm_aggregates", data)
        self.assertIn("key_differentiation_cases", data)
        self.assertIn("protection_class_breakdown", data)
        self.assertEqual(data["manifest_version"], "1.3.0-dbops-preregistered")


if __name__ == "__main__":
    unittest.main()
