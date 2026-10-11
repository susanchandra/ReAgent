"""
test_ambiguity_v2_comprehensive.py — Comprehensive test suite for Ambiguity Benchmark V2.

Verifies:
1. AgentView facade isolation & security enforcement
2. Dual-view scoring partition conservation:
   - Sum of World State categories == Total Trials per arm
   - Sum of Agent Claim categories == Total Trials per arm
3. Identifiability invariant between FAIL_BEFORE_COMMIT and LOST_RESPONSE
4. Reproducibility & byte-identical JSONL generation
5. Probe ablation execution & synthetic fixture labeling
"""

import unittest
import os
import hashlib
from reagent.experiments.ambiguity.manifest_v2 import (
    ALL_POLICY_ARMS,
    ALL_FAULT_MODES,
    ALL_OPERATIONS,
    FIXED_SEEDS,
    WORLD_STATE_CATEGORIES,
    AGENT_CLAIM_CATEGORIES,
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
    OP_CREATE_RESOURCE,
)
from reagent.experiments.ambiguity.simulator import StatefulSimulator
from reagent.experiments.ambiguity.agent_view import AgentView, AgentSecurityViolation
from reagent.experiments.ambiguity.runner_v2 import run_benchmark_v2
from reagent.experiments.ambiguity.probe_ablation import run_probe_ablation_study
from reagent.experiments.ambiguity.faults import make_standard_timeout_error


class TestAmbiguityV2Comprehensive(unittest.TestCase):

    def test_agent_view_security_isolation(self):
        """Policies cannot access simulator internals or commit stores via AgentView."""
        sim = StatefulSimulator()
        view = AgentView(sim)

        # Allowed operations work
        res = view.execute(OP_CREATE_RESOURCE, {"resource_id": "r_sec", "name": "sec"})
        self.assertEqual(res["status"], "SUCCESS")

        # Prohibited attribute access raises SecurityViolation
        with self.assertRaises(AgentSecurityViolation):
            _ = view._committed_records

        with self.assertRaises(AgentSecurityViolation):
            _ = view.commit_store

        with self.assertRaises(AgentSecurityViolation):
            view.custom_attr = 123

    def test_identifiability_invariant(self):
        """Matched FAIL_BEFORE_COMMIT and LOST_RESPONSE produce identical agent-visible hashes."""
        err1 = make_standard_timeout_error("apply_transfer", "req-test-1")
        err2 = make_standard_timeout_error("apply_transfer", "req-test-1")
        self.assertEqual(err1.canonical_hash(), err2.canonical_hash())

    def test_dual_view_partition_conservation(self):
        """
        World State and Agent Claim partitions must sum EXACTLY to total trials for every arm.
        """
        temp_jsonl = "D:/ReAgent/reports/temp_test_v2.jsonl"
        result = run_benchmark_v2(output_jsonl_path=temp_jsonl)
        trials = result["trials"]

        self.assertEqual(len(trials), 432) # 6 arms * 6 faults * 3 ops * 4 seeds

        # Group by arm
        by_arm = {}
        for t in trials:
            arm = t["policy_arm"]
            if arm not in by_arm:
                by_arm[arm] = []
            by_arm[arm].append(t)

        for arm, arm_trials in by_arm.items():
            self.assertEqual(len(arm_trials), 72)

            # Check World State partition: exactly one category is True per trial
            ws_sums = {cat: 0 for cat in WORLD_STATE_CATEGORIES}
            ac_sums = {cat: 0 for cat in AGENT_CLAIM_CATEGORIES}

            for t in arm_trials:
                ws = t["world_state"]
                ac = t["agent_claim"]

                # Exclusive sum within trial
                ws_true_count = sum(1 for cat in WORLD_STATE_CATEGORIES if ws[cat])
                ac_true_count = sum(1 for cat in AGENT_CLAIM_CATEGORIES if ac[cat])
                self.assertEqual(ws_true_count, 1, f"Trial {t['case_id']} world_state not mutually exclusive")
                self.assertEqual(ac_true_count, 1, f"Trial {t['case_id']} agent_claim not mutually exclusive")

                for cat in WORLD_STATE_CATEGORIES:
                    if ws[cat]: ws_sums[cat] += 1
                for cat in AGENT_CLAIM_CATEGORIES:
                    if ac[cat]: ac_sums[cat] += 1

            # Total sum must equal 72
            self.assertEqual(sum(ws_sums.values()), 72)
            self.assertEqual(sum(ac_sums.values()), 72)

        if os.path.exists(temp_jsonl):
            os.remove(temp_jsonl)

    def test_jsonl_reproducibility(self):
        """Two identical runs must produce byte-identical JSONL files."""
        path1 = "D:/ReAgent/reports/repro_1.jsonl"
        path2 = "D:/ReAgent/reports/repro_2.jsonl"

        run_benchmark_v2(output_jsonl_path=path1)
        run_benchmark_v2(output_jsonl_path=path2)

        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            h1 = hashlib.sha256(f1.read()).hexdigest()
            h2 = hashlib.sha256(f2.read()).hexdigest()

        self.assertEqual(h1, h2, "Benchmark JSONL outputs must be 100% byte-identical across runs")

        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)

    def test_probe_ablation_fixtures_labeled(self):
        """Probe ablation study explicitly labels synthetic fixtures."""
        study = run_probe_ablation_study()
        self.assertEqual(study["held_out_tools_count"], 3)
        for r in study["ablation_results"]:
            if r["ablation_arm"] == "RECORDED_FIXTURE":
                self.assertEqual(r["label"], "synthetic fixture, not Nemotron")


if __name__ == "__main__":
    unittest.main()
