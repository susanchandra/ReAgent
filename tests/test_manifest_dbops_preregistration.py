"""
test_manifest_dbops_preregistration.py — Verification of pre-registered Manifest DBOPS V1 (Phase A.1 Final).

Validates:
1. Manifest version, baseline commit at authoring (format and semantics), and heritage anchors.
2. 12 unique held-out tasks spanning diverse state-changing database operations.
3. 8 fault modes (including NO_FAULT, WRONG_CONTENT, PARTIAL_COMMIT).
4. 7 offline experimental arms and 1 reserved live arm (LLM_AGENT_WITH_DB_READ_TOOL).
5. Exact matrix arithmetic: 12 tasks * 8 fault modes * 7 arms = 672 planned configurations (0 inapplicable).
6. Deduplication fairness: exactly 6 KEY_PROTECTED tasks and 6 unprotected tasks.
7. 8 distinct adversarial contract rejection classes with specific reasons.
8. Delayed visibility parameters (k=1, mode="INJECTED_PROBE_DELAY").
9. Live-model protocol specification (generic held-out model id, temperature, 3 samples/task, 36 samples, >= 9/12 passing tasks).
10. Shared action budget (2) and probe budget (2).
11. COMMIT_THEN_TIMEOUT / FAIL_BEFORE_COMMIT identifiability requirements (executable tests deferred to B2).
"""

import inspect
import re
import unittest

import reagent.experiments.dbops.manifest_dbops_v1 as manifest_module
from reagent.experiments.dbops.manifest_dbops_v1 import (
    MANIFEST_VERSION,
    BASELINE_COMMIT_AT_AUTHORING,
    PREREGISTRATION_BASE_COMMIT,
    CHECKPOINT_ROLLBACK_ANCHOR,
    ALL_TASK_IDS,
    ALL_FAULT_MODES,
    ALL_OFFLINE_ARMS,
    ARM_RESERVED_LIVE_MODEL,
    SCHEMA_DEDUPLICATION_SUPPORT,
    ADVERSARIAL_REJECTION_CLASSES,
    TOTAL_TASKS,
    TOTAL_FAULTS,
    TOTAL_OFFLINE_ARMS,
    INAPPLICABLE_CONFIGURATIONS_COUNT,
    TOTAL_PLANNED_CONFIGURATIONS,
    MAX_ACTION_ATTEMPTS,
    MAX_PROBE_OBSERVATIONS,
    DELAYED_VISIBILITY_K_READS,
    DELAYED_VISIBILITY_MODEL,
    LIVE_MODEL_TARGET_IDENTIFIER,
    LIVE_MODEL_SAMPLES_PER_TASK,
    LIVE_MODEL_TOTAL_SAMPLES,
    LIVE_MODEL_MIN_PASSING_TASKS,
    LIVE_MODEL_TEMPERATURE,
    LIVE_MODEL_MAX_TOKENS,
    FAULT_PAIR_AGENT_ERROR_BYTES_MUST_BE_IDENTICAL,
    FAULT_PAIR_IDENTIFIABILITY_EXECUTABLE_TESTS_PHASE,
    INTEGRITY_ERROR_ON_RETRY_NOT_PROOF_OF_PRIOR_COMMIT,
    FAULT_NO_FAULT,
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_COMMIT_THEN_TIMEOUT,
    FAULT_DELAYED_VISIBILITY,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_UNOBSERVABLE,
    FAULT_WRONG_CONTENT,
    FAULT_PARTIAL_COMMIT,
    ARM_GOLD_CONTRACT,
    ARM_DERIVED_MODEL_FIXTURE,
    ARM_HAND_AUTHORED_BASELINE,
    ARM_SHARED_OBSERVATION,
    ARM_TOOL_CLAIM_ONLY,
    ARM_NAIVE_RETRY,
    ARM_ALWAYS_ESCALATE,
)


class TestManifestDbOpsPreregistration(unittest.TestCase):

    def test_metadata_and_rollback_anchors(self):
        self.assertEqual(MANIFEST_VERSION, "1.3.0-dbops-preregistered")
        self.assertEqual(PREREGISTRATION_BASE_COMMIT, BASELINE_COMMIT_AT_AUTHORING)
        self.assertRegex(
            BASELINE_COMMIT_AT_AUTHORING,
            r"^[0-9a-f]{40}$",
            "Baseline commit at authoring must be a full 40-character Git object name",
        )
        self.assertEqual(
            CHECKPOINT_ROLLBACK_ANCHOR,
            "5a0542d13b4845ffeb08e6040846ea259463c631",
        )
        self.assertNotEqual(
            BASELINE_COMMIT_AT_AUTHORING,
            CHECKPOINT_ROLLBACK_ANCHOR,
            "Authoring-time HEAD baseline must not be confused with heritage rollback anchor",
        )

    def test_baseline_commit_semantics_documented(self):
        """Baseline hash records Git HEAD at manifest authoring time, not the future manifest commit."""
        source = inspect.getsource(manifest_module)
        self.assertIn("BASELINE_COMMIT_AT_AUTHORING", source)
        self.assertIn("not the future manifest commit hash", source)
        self.assertIn("never hand-entered", source.lower())

    def test_task_dimensions_and_uniqueness(self):
        self.assertEqual(TOTAL_TASKS, 12, "Must specify exactly 12 held-out database tasks")
        self.assertEqual(len(ALL_TASK_IDS), 12)
        self.assertEqual(len(set(ALL_TASK_IDS)), 12, "All task IDs must be unique")

    def test_fault_modes_completeness(self):
        self.assertEqual(TOTAL_FAULTS, 8, "Must declare exactly 8 fault modes")
        self.assertEqual(len(ALL_FAULT_MODES), 8)
        expected_faults = {
            FAULT_NO_FAULT,
            FAULT_FAIL_BEFORE_COMMIT,
            FAULT_COMMIT_THEN_TIMEOUT,
            FAULT_DELAYED_VISIBILITY,
            FAULT_STATUS_UNAVAILABLE,
            FAULT_UNOBSERVABLE,
            FAULT_WRONG_CONTENT,
            FAULT_PARTIAL_COMMIT,
        }
        self.assertEqual(set(ALL_FAULT_MODES), expected_faults)

    def test_experimental_arms_and_reserved_live_arm(self):
        self.assertEqual(TOTAL_OFFLINE_ARMS, 7, "Must declare exactly 7 offline arms")
        self.assertEqual(len(ALL_OFFLINE_ARMS), 7)
        expected_offline = {
            ARM_GOLD_CONTRACT,
            ARM_DERIVED_MODEL_FIXTURE,
            ARM_HAND_AUTHORED_BASELINE,
            ARM_SHARED_OBSERVATION,
            ARM_TOOL_CLAIM_ONLY,
            ARM_NAIVE_RETRY,
            ARM_ALWAYS_ESCALATE,
        }
        self.assertEqual(set(ALL_OFFLINE_ARMS), expected_offline)
        self.assertEqual(ARM_RESERVED_LIVE_MODEL, "LLM_AGENT_WITH_DB_READ_TOOL")
        self.assertNotIn(ARM_RESERVED_LIVE_MODEL, ALL_OFFLINE_ARMS, "Live arm must not be in offline list")

    def test_matrix_configuration_arithmetic(self):
        self.assertEqual(INAPPLICABLE_CONFIGURATIONS_COUNT, 0, "No inapplicable configurations excluded")
        self.assertEqual(TOTAL_PLANNED_CONFIGURATIONS, 672, "12 tasks * 8 faults * 7 arms = 672")
        self.assertEqual(
            (TOTAL_TASKS * TOTAL_FAULTS * TOTAL_OFFLINE_ARMS) - INAPPLICABLE_CONFIGURATIONS_COUNT,
            TOTAL_PLANNED_CONFIGURATIONS,
        )

    def test_schema_deduplication_fairness(self):
        supported = [k for k, v in SCHEMA_DEDUPLICATION_SUPPORT.items() if v]
        unsupported = [k for k, v in SCHEMA_DEDUPLICATION_SUPPORT.items() if not v]
        self.assertEqual(len(supported), 6, "Must have exactly 6 KEY_PROTECTED (schema-unique) tasks")
        self.assertEqual(len(unsupported), 6, "Must have exactly 6 unprotected tasks")
        self.assertEqual(set(SCHEMA_DEDUPLICATION_SUPPORT.keys()), set(ALL_TASK_IDS))

    def test_adversarial_rejection_classes_completeness(self):
        self.assertEqual(len(ADVERSARIAL_REJECTION_CLASSES), 8, "Must declare exactly 8 rejection classes")
        expected_classes = {
            "CLASS_UNGROUNDED_LITERAL",
            "CLASS_STATUS_ONLY_ON_WRITE",
            "CLASS_TABLE_OUTSIDE_ALLOWLIST",
            "CLASS_RAW_SQL_INJECTION",
            "CLASS_VACUOUS_PRECONDITION",
            "CLASS_WRONG_AMOUNT",
            "CLASS_MISSING_REQUEST_ID",
            "CLASS_UNDECLARED_EXTRA_FIELD",
        }
        self.assertEqual(set(ADVERSARIAL_REJECTION_CLASSES.keys()), expected_classes)
        for cls_name, reason in ADVERSARIAL_REJECTION_CLASSES.items():
            self.assertTrue(len(reason) > 10, f"Class {cls_name} must have a non-trivial reason")

    def test_delayed_visibility_protocol_settings(self):
        self.assertEqual(DELAYED_VISIBILITY_K_READS, 1, "k must be fixed to 1")
        self.assertEqual(DELAYED_VISIBILITY_MODEL, "INJECTED_PROBE_DELAY")

    def test_live_model_protocol_parameters(self):
        self.assertEqual(LIVE_MODEL_TARGET_IDENTIFIER, "held_out_instruction_model_v1")
        self.assertNotIn("/", LIVE_MODEL_TARGET_IDENTIFIER, "Offline manifest must not pin vendor model paths")
        self.assertEqual(LIVE_MODEL_SAMPLES_PER_TASK, 3)
        self.assertEqual(LIVE_MODEL_TOTAL_SAMPLES, 36)
        self.assertEqual(LIVE_MODEL_MIN_PASSING_TASKS, 9)
        self.assertEqual(LIVE_MODEL_TEMPERATURE, 0.0)
        self.assertEqual(LIVE_MODEL_MAX_TOKENS, 1024)

    def test_shared_budgets(self):
        self.assertEqual(MAX_ACTION_ATTEMPTS, 2)
        self.assertEqual(MAX_PROBE_OBSERVATIONS, 2)

    def test_commit_ambiguity_identifiability_frozen_not_executable(self):
        """Manifest requires byte-identical agent errors; pair tests belong to B2."""
        self.assertTrue(FAULT_PAIR_AGENT_ERROR_BYTES_MUST_BE_IDENTICAL)
        self.assertEqual(FAULT_PAIR_IDENTIFIABILITY_EXECUTABLE_TESTS_PHASE, "B2")
        self.assertTrue(INTEGRITY_ERROR_ON_RETRY_NOT_PROOF_OF_PRIOR_COMMIT)
        doc = manifest_module.__doc__ or ""
        self.assertIn("byte-identical", doc.lower())
        self.assertIn("FAIL_BEFORE_COMMIT", doc)
        self.assertIn("COMMIT_THEN_TIMEOUT", doc)
        self.assertRegex(
            doc,
            re.compile(r"integrity error alone", re.IGNORECASE),
            "Manifest must state integrity errors alone do not prove prior commit",
        )
        self.assertIn("deferred to Phase B2", doc)


if __name__ == "__main__":
    unittest.main()
