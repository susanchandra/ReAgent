"""
test_m5_benchmark_accounting.py — Automated Accounting Invariants for DBOPS M5 Comparative Benchmark.

Proves:
1. Exact trial count = 480 across 5 arms (exactly 96 trials per arm).
2. Exactly 12 tasks x 8 fault modes = 96 paired configurations per arm.
3. Every raw trial record is categorized into exactly one mutually exclusive outcome:
   true_verified + false_verified + safe_escalation + unnecessary_escalation == 96.
4. Explains the previously perceived 'missing' 7 trials for TOOL_CLAIM_ONLY (unnecessary_escalation = 7)
   and 'missing' 22 trials for ALWAYS_ESCALATE (unnecessary_escalation = 22).
5. Asserts REAGENT_CONTRACT achieves 0.0% false verification rate and 0 duplicate writes.
"""

import json
import unittest
from pathlib import Path


class TestM5BenchmarkAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench_path = Path("reports/dbops/comparative_evaluation_results.json")
        if not cls.bench_path.exists():
            raise FileNotFoundError(f"Missing comparative benchmark at {cls.bench_path}")

        with open(cls.bench_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_01_total_trial_scale_and_provenance(self):
        """Validates total paired configurations and synthetic deterministic provenance."""
        self.assertEqual(self.data["total_paired_configurations"], 480)
        self.assertEqual(len(self.data["trials"]), 480)
        self.assertEqual(self.data["tested_tasks_count"], 12)
        self.assertEqual(self.data["tested_fault_modes_count"], 8)
        self.assertEqual(self.data["tested_arms_count"], 5)
        self.assertEqual(self.data["provenance"], "SYNTHETIC_DETERMINISTIC_BENCHMARK")

    def test_02_arm_aggregates_exact_denominator_reconciliation(self):
        """
        Validates that every arm aggregate satisfies:
        true_verified + false_verified + safe_escalation + unnecessary_escalation == 96.
        """
        aggs = self.data["arm_aggregates"]
        expected_arms = [
            "REAGENT_CONTRACT",
            "SHARED_OBSERVATION",
            "TOOL_CLAIM_ONLY",
            "NAIVE_RETRY",
            "ALWAYS_ESCALATE",
        ]

        for arm in expected_arms:
            self.assertIn(arm, aggs, f"Missing arm '{arm}' in arm_aggregates")
            m = aggs[arm]
            total = m["total_trials"]
            self.assertEqual(total, 96, f"Arm '{arm}' total trials must be exactly 96")

            tv = m["true_verified_count"]
            fv = m["false_verified_count"]
            se = m["safe_escalation_count"]
            ue = m["unnecessary_escalation_count"]

            # Mutual exclusivity equation
            sum_outcomes = tv + fv + se + ue
            self.assertEqual(
                sum_outcomes,
                total,
                f"Arm '{arm}' outcomes ({tv} + {fv} + {se} + {ue} = {sum_outcomes}) must reconcile to {total}",
            )

        # Explicit verification of previously missing categories
        self.assertEqual(aggs["TOOL_CLAIM_ONLY"]["unnecessary_escalation_count"], 7)
        self.assertEqual(
            aggs["TOOL_CLAIM_ONLY"]["true_verified_count"]
            + aggs["TOOL_CLAIM_ONLY"]["false_verified_count"]
            + aggs["TOOL_CLAIM_ONLY"]["safe_escalation_count"],
            89,
            "15 + 45 + 29 = 89 without unnecessary escalations",
        )
        self.assertEqual(
            aggs["TOOL_CLAIM_ONLY"]["true_verified_count"]
            + aggs["TOOL_CLAIM_ONLY"]["false_verified_count"]
            + aggs["TOOL_CLAIM_ONLY"]["safe_escalation_count"]
            + aggs["TOOL_CLAIM_ONLY"]["unnecessary_escalation_count"],
            96,
            "89 + 7 = 96 with unnecessary escalations fully reconciled",
        )

        self.assertEqual(aggs["ALWAYS_ESCALATE"]["unnecessary_escalation_count"], 22)
        self.assertEqual(
            aggs["ALWAYS_ESCALATE"]["safe_escalation_count"],
            74,
            "Safe escalations = 74 without unnecessary escalations",
        )
        self.assertEqual(
            aggs["ALWAYS_ESCALATE"]["safe_escalation_count"]
            + aggs["ALWAYS_ESCALATE"]["unnecessary_escalation_count"],
            96,
            "74 + 22 = 96 with unnecessary escalations fully reconciled",
        )

    def test_03_raw_trial_records_mutual_exclusivity(self):
        """
        Validates every raw trial record is categorized into exactly one mutually exclusive outcome
        with zero unflagged trials and zero multi-flagged trials.
        """
        trials = self.data["trials"]
        self.assertEqual(len(trials), 480)

        trial_ids_seen = set()

        for t in trials:
            # Check unique trial identification
            trial_key = (t["arm"], t["task_id"], t["fault_mode"])
            self.assertNotIn(trial_key, trial_ids_seen, f"Duplicate trial detected: {trial_key}")
            trial_ids_seen.add(trial_key)

            tv = bool(t["true_verified"])
            fv = bool(t["false_verified"])
            se = bool(t["safe_escalation"])
            ue = bool(t["unnecessary_escalation"])

            active_flags = [k for k, v in [("true_verified", tv), ("false_verified", fv), ("safe_escalation", se), ("unnecessary_escalation", ue)] if v]
            self.assertEqual(
                len(active_flags),
                1,
                f"Trial {trial_key} must have exactly one active outcome flag; got {active_flags}",
            )

    def test_04_core_reliability_invariants(self):
        """
        Validates core ReAgent safety advantages over baselines:
        1. REAGENT_CONTRACT false verification rate is 0.0% (0 false verifications).
        2. REAGENT_CONTRACT duplicate primary writes is 0.
        3. NAIVE_RETRY causes permanent duplicate primary writes on non-idempotent operations.
        4. SHARED_OBSERVATION suffers >40% false verification rate.
        """
        aggs = self.data["arm_aggregates"]

        # 1. ReAgent achieves 0 false verifications
        self.assertEqual(aggs["REAGENT_CONTRACT"]["false_verified_count"], 0)
        self.assertEqual(aggs["REAGENT_CONTRACT"]["false_verified_rate"], 0.0)
        self.assertEqual(aggs["REAGENT_CONTRACT"]["duplicate_primary_count"], 0)
        self.assertEqual(aggs["REAGENT_CONTRACT"]["unsafe_retry_count"], 0)

        # 2. Shared observation false verification rate
        so_fv_rate = aggs["SHARED_OBSERVATION"]["false_verified_rate"]
        self.assertGreater(so_fv_rate, 0.40, "Shared observation must demonstrate >40% false verification rate")

        # 3. Naive retry duplicates
        self.assertGreater(aggs["NAIVE_RETRY"]["duplicate_primary_count"], 0, "Naive retry must produce duplicate writes")
        self.assertGreater(aggs["NAIVE_RETRY"]["unsafe_retry_count"], 0, "Naive retry must produce unsafe retries")


if __name__ == "__main__":
    unittest.main()
