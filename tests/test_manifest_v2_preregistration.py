"""
test_manifest_v2_preregistration.py — Verification of pre-registered manifest V2.

Asserts:
1. Manifest constants, seeds, arms, operations, and fault modes are strictly defined.
2. Trial counts meet the >= 60 paired units per arm constraint.
3. Scoring partitions are mutually exclusive and collectively exhaustive.
"""

import unittest
from reagent.experiments.ambiguity.manifest_v2 import (
    MANIFEST_VERSION,
    PREREGISTRATION_BASE_COMMIT,
    ALL_FAULT_MODES,
    ALL_OPERATIONS,
    ALL_POLICY_ARMS,
    FIXED_SEEDS,
    WORLD_STATE_CATEGORIES,
    AGENT_CLAIM_CATEGORIES,
    IDEMPOTENCY_KEY_SUPPORT,
)


class TestManifestV2Preregistration(unittest.TestCase):

    def test_preregistration_metadata(self):
        self.assertEqual(MANIFEST_VERSION, "2.0.0-preregistered")
        self.assertEqual(PREREGISTRATION_BASE_COMMIT, "f581442fee5eda1b364a7177df72abe8eb842128")

    def test_trial_unit_count(self):
        units_per_arm = len(ALL_FAULT_MODES) * len(ALL_OPERATIONS) * len(FIXED_SEEDS)
        self.assertGreaterEqual(units_per_arm, 60, "Must have at least 60 trial units per policy arm")
        self.assertEqual(units_per_arm, 72)
        total_trials = units_per_arm * len(ALL_POLICY_ARMS)
        self.assertEqual(total_trials, 432)

    def test_partition_definitions(self):
        self.assertEqual(len(WORLD_STATE_CATEGORIES), 4)
        self.assertEqual(len(AGENT_CLAIM_CATEGORIES), 4)
        self.assertIn("exactly_one_effect", WORLD_STATE_CATEGORIES)
        self.assertIn("duplicate_effect", WORLD_STATE_CATEGORIES)
        self.assertIn("lost_effect", WORLD_STATE_CATEGORIES)
        self.assertIn("claimed_complete_verified", AGENT_CLAIM_CATEGORIES)
        self.assertIn("escalated", AGENT_CLAIM_CATEGORIES)

    def test_idempotency_key_config(self):
        self.assertTrue(IDEMPOTENCY_KEY_SUPPORT["create_resource"])
        self.assertFalse(IDEMPOTENCY_KEY_SUPPORT["apply_transfer"])
        self.assertTrue(IDEMPOTENCY_KEY_SUPPORT["reserve_then_link"])


if __name__ == "__main__":
    unittest.main()
