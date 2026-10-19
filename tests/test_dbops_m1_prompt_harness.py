"""
test_dbops_m1_prompt_harness.py — Offline Prompt & Schema Harness Verification (DBOPS-M1).

Verifies:
1. Deterministic prompt construction for all 12 preregistered DBOPS tasks.
2. Stable SHA-256 prompt hashing and schema version recording.
3. Prompt token bounds (< 1500 tokens) using conservative character heuristic.
4. Schema validation and error diagnosis on valid, malformed, invalid, and ungrounded fixtures.
5. Rejection of ungrounded and wrong-task fixtures via ContractGroundingValidator.
6. Rejection of STATUS_SUCCESS-only contracts on state-changing database tool.
7. Rejection of filesystem predicates on non-filesystem database tool.
8. Proof of zero network calls or provider invocations (including retries).

Non-negotiable safety:
- Zero network requests. Zero live provider calls. Zero .env inspection.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from reagent.experiments.dbops.manifest_dbops_v1 import (
    ALL_TASK_IDS,
    MANIFEST_VERSION,
)
from reagent.experiments.dbops.tasks import get_task
from reagent.experiments.dbops.prompt_harness import (
    DbopsPromptHarness,
    TaskPromptBundle,
)
from reagent.experiments.dbops.fixture_replay import (
    OfflineFixtureReplayEngine,
    generate_valid_fixture,
    generate_negative_fixtures,
)
from reagent.providers.nebius import NebiusNemotronProvider


class TestDbopsM1PromptHarness(unittest.TestCase):
    """Verifies DBOPS-M1 prompt construction and token estimation bounds."""

    def setUp(self):
        self.harness = DbopsPromptHarness()

    def test_01_all_12_prompts_constructed_deterministically(self):
        """Constructs prompts for all 12 tasks with stable hashes across repeated runs."""
        bundles_run_1 = self.harness.construct_all_prompts()
        bundles_run_2 = self.harness.construct_all_prompts()

        self.assertEqual(len(bundles_run_1), 12)
        self.assertEqual(set(bundles_run_1.keys()), set(ALL_TASK_IDS))

        for task_id in ALL_TASK_IDS:
            b1 = bundles_run_1[task_id]
            b2 = bundles_run_2[task_id]

            self.assertEqual(b1.task_id, task_id)
            self.assertEqual(b1.manifest_version, MANIFEST_VERSION)
            self.assertEqual(b1.prompt_hash, b2.prompt_hash, f"Hash unstable for {task_id}")
            self.assertEqual(b1.full_prompt_text, b2.full_prompt_text)

            # Ensure prompt contains task description
            task = get_task(task_id)
            self.assertIn(task.human_description, b1.user_prompt)

            # Ensure schema definition is included
            self.assertIn('"tool": "<tool_name_string>"', b1.user_prompt)
            self.assertIn('"proposed_postconditions"', b1.user_prompt)

    def test_02_prompt_token_estimation_and_ceiling(self):
        """Validates token bounds (< 1500 tokens) and records proxy estimator type."""
        bundles = self.harness.construct_all_prompts()

        for task_id, bundle in bundles.items():
            self.assertTrue(
                bundle.token_limit_safe,
                f"Task {task_id} exceeded token limit: {bundle.estimated_tokens} >= 1500"
            )
            self.assertLess(bundle.estimated_tokens, 1500)
            self.assertGreater(bundle.estimated_tokens, 100, f"Token estimate suspiciously low for {task_id}")
            self.assertEqual(bundle.token_estimator_type, "conservative_char_ratio_proxy")

    def test_03_zero_live_provider_invocations(self):
        """Proves that unconfigured provider and harness make zero network calls."""
        with patch("urllib.request.urlopen") as mock_url, \
             patch("http.client.HTTPConnection.request") as mock_http, \
             patch.dict(os.environ, {"NEBIUS_API_KEY": ""}):
            # Constructing all prompts
            bundles = self.harness.construct_all_prompts()
            self.assertEqual(len(bundles), 12)

            # Instantiating unconfigured provider
            provider = NebiusNemotronProvider(client=None)
            self.assertFalse(provider.is_configured())

            # Attempting to call generate() on unconfigured provider must fail fast offline
            with self.assertRaises(RuntimeError) as cm:
                provider.generate(prompt="Test prompt")
            self.assertIn("nebius_api_key is not set", str(cm.exception).lower())

            # Zero network calls occurred
            self.assertEqual(mock_url.call_count, 0)
            self.assertEqual(mock_http.call_count, 0)


class TestDbopsM1FixtureReplay(unittest.TestCase):
    """Verifies schema validation and grounding rejection across fixture classes."""

    def setUp(self):
        self.replay_engine = OfflineFixtureReplayEngine()

    def test_04_valid_fixtures_pass_all_12_tasks(self):
        """Authoring valid synthetic fixtures passes parsing, schema, and grounding for all 12 tasks."""
        for task_id in ALL_TASK_IDS:
            raw_fixture = generate_valid_fixture(task_id)
            outcome = self.replay_engine.replay_fixture(
                fixture_id=f"valid_{task_id}",
                task_id=task_id,
                category="VALID",
                raw_response_text=raw_fixture,
            )
            self.assertTrue(outcome.parsed_json_ok, f"JSON parse failed for {task_id}")
            self.assertTrue(outcome.schema_valid, f"Schema validation failed for {task_id}: {outcome.schema_error}")
            self.assertTrue(
                outcome.grounding_valid,
                f"Grounding failed for {task_id}: {outcome.grounding_errors}"
            )
            self.assertTrue(outcome.overall_passed, f"Overall passed failed for {task_id}")
            self.assertIsNotNone(outcome.structured_decision)
            self.assertEqual(outcome.structured_decision.tool, "dbops_database")

    def test_05_malformed_json_rejected(self):
        """Malformed raw string is caught at parsing stage and classified cleanly."""
        neg = generate_negative_fixtures("task_01_refund_issue")
        outcome = self.replay_engine.replay_fixture(
            fixture_id="neg_malformed",
            task_id="task_01_refund_issue",
            category="MALFORMED_JSON",
            raw_response_text=neg["MALFORMED_JSON"],
        )
        self.assertFalse(outcome.parsed_json_ok)
        self.assertFalse(outcome.schema_valid)
        self.assertFalse(outcome.overall_passed)
        self.assertIn("MALFORMED_JSON", outcome.failure_reason)

    def test_06_schema_invalids_rejected_with_specific_diagnostics(self):
        """Schema violations (provenance injection, unknown keys, missing tool, empty tool) rejected."""
        task_id = "task_02_wallet_transfer"
        neg = generate_negative_fixtures(task_id)

        # 1. Injected provenance
        res = self.replay_engine.replay_fixture("neg_prov", task_id, "PROVENANCE", neg["SCHEMA_INVALID_PROVENANCE_INJECTED"])
        self.assertTrue(res.parsed_json_ok)
        self.assertFalse(res.schema_valid)
        self.assertIn("provenance", res.schema_error.lower())

        # 2. Unknown keys
        res = self.replay_engine.replay_fixture("neg_keys", task_id, "UNKNOWN_KEYS", neg["SCHEMA_INVALID_UNKNOWN_KEYS"])
        self.assertTrue(res.parsed_json_ok)
        self.assertFalse(res.schema_valid)
        self.assertIn("unknown keys", res.schema_error.lower())

        # 3. Missing tool
        res = self.replay_engine.replay_fixture("neg_missing_tool", task_id, "MISSING_TOOL", neg["SCHEMA_INVALID_MISSING_TOOL"])
        self.assertTrue(res.parsed_json_ok)
        self.assertFalse(res.schema_valid)
        self.assertIn("missing", res.schema_error.lower())

        # 4. Empty tool
        res = self.replay_engine.replay_fixture("neg_empty_tool", task_id, "EMPTY_TOOL", neg["SCHEMA_INVALID_EMPTY_TOOL"])
        self.assertTrue(res.parsed_json_ok)
        self.assertFalse(res.schema_valid)
        self.assertIn("empty", res.schema_error.lower())

    def test_07_grounding_rejection_rules(self):
        """Tests that ContractGroundingValidator rejects status-only, ungrounded, wrong-task, and bad predicates."""
        task_id = "task_03_seat_booking"
        neg = generate_negative_fixtures(task_id)

        # 1. STATUS_SUCCESS only on state-changing tool
        res = self.replay_engine.replay_fixture("neg_status_only", task_id, "STATUS_ONLY", neg["STATUS_ONLY_WRITE"])
        self.assertTrue(res.schema_valid)
        self.assertFalse(res.grounding_valid)
        self.assertTrue(any("state-changing but has no independent" in err for err in res.grounding_errors))

        # 2. Ungrounded objective
        res = self.replay_engine.replay_fixture("neg_ungrounded", task_id, "UNGROUNDED", neg["UNGROUNDED_OBJECTIVE"])
        self.assertTrue(res.schema_valid)
        self.assertFalse(res.grounding_valid)
        self.assertTrue(any("UngroundedContract" in err for err in res.grounding_errors))

        # 3. Wrong task target
        res = self.replay_engine.replay_fixture("neg_wrong_task", task_id, "WRONG_TASK", neg["WRONG_TASK_TARGET"])
        self.assertTrue(res.schema_valid)
        self.assertFalse(res.grounding_valid)
        self.assertTrue(any("UngroundedContract" in err for err in res.grounding_errors))

        # 4. Incompatible filesystem predicate on database tool
        res = self.replay_engine.replay_fixture("neg_file_pred", task_id, "FILE_PRED", neg["UNSUPPORTED_PREDICATE_FILESYSTEM"])
        self.assertTrue(res.schema_valid)
        self.assertFalse(res.grounding_valid)
        self.assertTrue(any("IncompatibleToolPredicate" in err for err in res.grounding_errors))


if __name__ == "__main__":
    unittest.main()
