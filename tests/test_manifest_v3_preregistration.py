"""
test_manifest_v3_preregistration.py — Verification of pre-registered Manifest V3.

Validates:
1. Manifest V3 metadata and rollback anchor.
2. 18 distinct scenario conditions and 72 executions per arm (432 total).
3. Authority of absence observations specification.
4. Mutually exclusive scoring partitions.
5. Shared action and observation budgets.
"""

import unittest
from reagent.experiments.ambiguity.manifest_v3 import (
    MANIFEST_VERSION,
    PREREGISTRATION_BASE_COMMIT,
    CHECKPOINT_ROLLBACK_ANCHOR,
    ALL_FAULT_MODES,
    ALL_OPERATIONS,
    ALL_POLICY_ARMS,
    FIXED_SEEDS,
    DISTINCT_CONDITIONS_COUNT,
    EXECUTIONS_PER_ARM,
    TOTAL_BENCHMARK_EXECUTIONS,
    MAX_ACTION_ATTEMPTS,
    MAX_PROBE_OBSERVATIONS,
    WORLD_STATE_CATEGORIES,
    AGENT_CLAIM_CATEGORIES,
    IDEMPOTENCY_KEY_SUPPORT,
    PROBE_AUTHORITY_MAPPING,
)


class TestManifestV3Preregistration(unittest.TestCase):

    def test_metadata_and_anchors(self):
        self.assertEqual(MANIFEST_VERSION, "3.0.0-preregistered")
        self.assertEqual(PREREGISTRATION_BASE_COMMIT, "3270ca8038c65bda4d6ef5ae2bbe103eb858d0a6")
        self.assertEqual(CHECKPOINT_ROLLBACK_ANCHOR, "f581442fee5eda1b364a7177df72abe8eb842128")

    def test_condition_and_replicate_counts(self):
        self.assertEqual(len(ALL_FAULT_MODES), 6)
        self.assertEqual(len(ALL_OPERATIONS), 3)
        self.assertEqual(len(FIXED_SEEDS), 4)
        self.assertEqual(DISTINCT_CONDITIONS_COUNT, 18, "Primary unit of analysis must be 18 conditions")
        self.assertEqual(EXECUTIONS_PER_ARM, 72, "Must be 72 executions per policy arm")
        self.assertEqual(TOTAL_BENCHMARK_EXECUTIONS, 432, "Total trials must equal 432")

    def test_shared_budgets(self):
        self.assertEqual(MAX_ACTION_ATTEMPTS, 2)
        self.assertEqual(MAX_PROBE_OBSERVATIONS, 2)

    def test_probe_authority_mapping(self):
        self.assertIn("create_resource", PROBE_AUTHORITY_MAPPING)
        self.assertIn("apply_transfer", PROBE_AUTHORITY_MAPPING)
        self.assertIn("reserve_then_link", PROBE_AUTHORITY_MAPPING)

    def test_idempotency_key_support_configuration(self):
        self.assertTrue(IDEMPOTENCY_KEY_SUPPORT["create_resource"])
        self.assertFalse(IDEMPOTENCY_KEY_SUPPORT["apply_transfer"], "Transfer must be configured as non-idempotent")
        self.assertTrue(IDEMPOTENCY_KEY_SUPPORT["reserve_then_link"])

    def test_partition_categories_completeness(self):
        self.assertEqual(len(WORLD_STATE_CATEGORIES), 4)
        self.assertEqual(len(AGENT_CLAIM_CATEGORIES), 4)
        self.assertIn("exactly_one_effect", WORLD_STATE_CATEGORIES)
        self.assertIn("duplicate_effect", WORLD_STATE_CATEGORIES)
        self.assertIn("lost_effect", WORLD_STATE_CATEGORIES)
        self.assertIn("claimed_complete_verified", AGENT_CLAIM_CATEGORIES)
        self.assertIn("escalated", AGENT_CLAIM_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
