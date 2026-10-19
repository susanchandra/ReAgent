"""
test_dbops_candidate_validation_gate.py — Focused Unit Tests for Model-Derived Contract Validation Gate.

Verifies:
1. Strict three-way separation: Gold contract, candidate contract, and runtime observation.
2. Prompt isolation: Zero leakage of gold queries, injected fault modes, or oracle data into prompts.
3. Production parsing & schema compliance: Valid candidates preserve canonical fields; unknown keys
   and provenance forgery are rejected.
4. Grounding & DBOPS schema enforcement: Rejects unsupported tables, undeclared columns, empty
   predicates, and raw SQL injection.
5. Task uniqueness rules: Enforces composite uniqueness semantics on Task 05 (order_event_log)
   and Task 08 (invoice_payments), preventing confusion between request IDs and unique business keys.
6. Semantic scoring: Distinguishes schema validity from semantic agreement; mismatches are never
   silently repaired or replaced with the gold contract.
7. Fail-closed pipeline guarantee: Invalid candidates result in ESCALATED, never VERIFIED.
8. Zero network calls, zero live provider calls, and zero .env inspection.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from reagent.models import AgentStatus, StructuredDecision, Postcondition, TaskState
from reagent.pipeline import ReAgentPipeline
from reagent.policy import TokenBudgetGuard, GlobalTokenLedger
from reagent.providers.base import LLMResponse
from reagent.providers.nebius import NebiusNemotronProvider
from reagent.experiments.dbops.manifest_dbops_v1 import ALL_TASK_IDS
from reagent.experiments.dbops.tasks import get_task
from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.integration import DbopsDatabaseTool, make_dbops_probe_callback
from reagent.experiments.dbops.candidate_gate import (
    CandidateContractValidationGate,
    CandidateEvaluationRecord,
    PROVENANCE_LABEL,
)


class TestCandidateContractValidationGate(unittest.TestCase):
    """Focused unit tests for the model-derived contract validation gate."""

    def setUp(self):
        self.gate = CandidateContractValidationGate()

    def test_01_three_way_separation_and_prompt_isolation(self):
        """Pre-execution prompts for all 12 tasks contain zero gold queries, faults, or oracle leaks."""
        for task_id in ALL_TASK_IDS:
            iso = self.gate.verify_prompt_isolation(task_id)
            self.assertTrue(iso["is_clean"], f"Prompt leakage detected for {task_id}: {iso}")
            self.assertEqual(len(iso["leaked_faults"]), 0, f"Leaked fault modes in {task_id}")
            self.assertEqual(len(iso["leaked_gold_markers"]), 0, f"Leaked gold markers in {task_id}")
            self.assertTrue(iso["human_description_present"])

    def test_02_valid_candidate_contracts_pass_and_preserve_canonical_schema(self):
        """Valid candidate fixtures pass parsing, schema, grounding, and gold-equivalence."""
        for task_id in ["task_01_refund_issue", "task_02_wallet_transfer", "task_05_order_cancellation", "task_08_invoice_settlement"]:
            fixture = self.gate.generate_candidate_fixture(task_id, "VALID_GOLD_EQUIVALENT")
            record = self.gate.validate_candidate_raw(task_id, fixture)

            self.assertIsInstance(record, CandidateEvaluationRecord)
            self.assertTrue(record.generation_parse_success)
            self.assertTrue(record.schema_valid)
            self.assertIsNone(record.schema_error)
            self.assertTrue(record.grounding_valid)
            self.assertTrue(record.semantic_gold_agreement)
            self.assertFalse(record.fallback_used, "Must never use gold contract fallback!")
            self.assertEqual(record.final_verdict, "GOLD_EQUIVALENT")
            self.assertIn("SYNTHETIC", record.provenance)

            # Preserves canonical structured decision fields
            d = record.decision
            self.assertIsNotNone(d)
            self.assertEqual(d.tool, "dbops_database")
            self.assertIsInstance(d.arguments, dict)
            self.assertGreaterEqual(len(d.proposed_postconditions), 2)
            for pc in d.proposed_postconditions:
                self.assertIsInstance(pc, Postcondition)
                self.assertTrue(pc.predicate)

    def test_03_malformed_and_refusal_responses_fail_closed(self):
        """Malformed JSON and refusal text are caught early and fail closed."""
        task_id = "task_01_refund_issue"

        # 1. Malformed JSON
        malformed = self.gate.generate_candidate_fixture(task_id, "MALFORMED_JSON")
        rec_mal = self.gate.validate_candidate_raw(task_id, malformed)
        self.assertFalse(rec_mal.generation_parse_success)
        self.assertFalse(rec_mal.schema_valid)
        self.assertTrue(rec_mal.invalid_contract_rejected)
        self.assertEqual(rec_mal.final_verdict, "REJECTED_MALFORMED_JSON")

        # 2. Model refusal
        refusal = self.gate.generate_candidate_fixture(task_id, "MODEL_REFUSAL")
        rec_ref = self.gate.validate_candidate_raw(task_id, refusal)
        self.assertFalse(rec_ref.generation_parse_success)
        self.assertTrue(rec_ref.invalid_contract_rejected)
        self.assertEqual(rec_ref.final_verdict, "REJECTED_REFUSAL")

    def test_04_schema_violations_fail_closed(self):
        """Unknown fields, provenance injection, and empty predicates are rejected."""
        task_id = "task_02_wallet_transfer"

        # 1. Unknown fields
        unknown = self.gate.generate_candidate_fixture(task_id, "UNKNOWN_FIELDS")
        rec_unk = self.gate.validate_candidate_raw(task_id, unknown)
        self.assertTrue(rec_unk.generation_parse_success)
        self.assertFalse(rec_unk.schema_valid)
        self.assertIn("unknown keys", rec_unk.schema_error.lower())
        self.assertTrue(rec_unk.invalid_contract_rejected)

        # 2. Provenance injection
        prov = self.gate.generate_candidate_fixture(task_id, "PROVENANCE_INJECTION")
        rec_prov = self.gate.validate_candidate_raw(task_id, prov)
        self.assertFalse(rec_prov.schema_valid)
        self.assertIn("provenance", rec_prov.schema_error.lower())
        self.assertTrue(rec_prov.invalid_contract_rejected)

        # 3. Empty predicates
        empty = self.gate.generate_candidate_fixture(task_id, "EMPTY_PREDICATES")
        rec_empty = self.gate.validate_candidate_raw(task_id, empty)
        self.assertTrue(rec_empty.schema_valid)
        self.assertFalse(rec_empty.grounding_valid)
        self.assertTrue(any("at least 1 postcondition" in err for err in rec_empty.grounding_errors))

    def test_05_grounding_and_unsupported_schema_rejected(self):
        """Unsupported tables, undeclared columns, and raw SQL injection are rejected."""
        task_id = "task_02_wallet_transfer"

        # 1. Unsupported table outside allowlist
        unsupp_tbl = self.gate.generate_candidate_fixture(task_id, "UNSUPPORTED_TABLE")
        rec_tbl = self.gate.validate_candidate_raw(task_id, unsupp_tbl)
        self.assertFalse(rec_tbl.semantic_gold_agreement)
        self.assertTrue(rec_tbl.invalid_contract_rejected)
        self.assertTrue(any("unauthorized_external_table" in err for err in rec_tbl.unsupported_tables_or_columns + rec_tbl.grounding_errors))

        # 2. Undeclared column in schema
        undecl_col = self.gate.generate_candidate_fixture(task_id, "UNDECLARED_EXTRA_COLUMN")
        rec_col = self.gate.validate_candidate_raw(task_id, undecl_col)
        self.assertFalse(rec_col.semantic_gold_agreement)
        self.assertTrue(rec_col.invalid_contract_rejected)
        self.assertTrue(any("phantom_undeclared_column" in err for err in rec_col.unsupported_tables_or_columns + rec_col.grounding_errors))

        # 3. Raw SQL injection marker
        raw_sql = self.gate.generate_candidate_fixture(task_id, "RAW_SQL_INJECTION")
        rec_sql = self.gate.validate_candidate_raw(task_id, raw_sql)
        self.assertFalse(rec_sql.semantic_gold_agreement)
        self.assertTrue(rec_sql.invalid_contract_rejected)
        self.assertGreater(len(rec_sql.extra_unsupported_predicates), 0)

    def test_06_wrong_values_and_missing_postconditions_scored_as_mismatches(self):
        """Wrong amounts and missing required postconditions are scored as discrepancies, not repaired."""
        task_id = "task_02_wallet_transfer"

        # 1. Wrong expected numeric amount
        wrong_amt = self.gate.generate_candidate_fixture(task_id, "WRONG_EXPECTED_AMOUNT")
        rec_amt = self.gate.validate_candidate_raw(task_id, wrong_amt)
        self.assertTrue(rec_amt.schema_valid)
        self.assertTrue(rec_amt.grounding_valid)
        self.assertFalse(rec_amt.semantic_gold_agreement)
        self.assertIn("numeric amount", rec_amt.score_summary)

        # 2. Missing required request ID binding
        missing = self.gate.generate_candidate_fixture(task_id, "MISSING_REQUIRED_POSTCONDITION")
        rec_mis = self.gate.validate_candidate_raw(task_id, missing)
        self.assertFalse(rec_mis.semantic_gold_agreement)
        self.assertIn("mandatory request ID", rec_mis.score_summary)

    def test_07_task_05_and_task_08_uniqueness_grounding_rules(self):
        """Enforces composite business uniqueness rules for Task 05 and Task 08."""
        # Task 05: order_event_log has UNIQUE(order_id, event_type); orders does not have cancellation_id unique
        t5_bad = self.gate.generate_candidate_fixture("task_05_order_cancellation", "TASK_05_BAD_UNIQUENESS")
        rec_t5 = self.gate.validate_candidate_raw("task_05_order_cancellation", t5_bad)
        self.assertFalse(rec_t5.semantic_gold_agreement)
        self.assertFalse(rec_t5.uniqueness_respected)
        self.assertIn("orders table does not enforce uniqueness", rec_t5.score_summary)

        # Task 08: invoice_payments has UNIQUE(invoice_id, settlement_ref); invoices does not have settlement_ref unique
        t8_bad = self.gate.generate_candidate_fixture("task_08_invoice_settlement", "TASK_08_BAD_UNIQUENESS")
        rec_t8 = self.gate.validate_candidate_raw("task_08_invoice_settlement", t8_bad)
        self.assertFalse(rec_t8.semantic_gold_agreement)
        self.assertFalse(rec_t8.uniqueness_respected)
        self.assertIn("invoices table does not enforce uniqueness", rec_t8.score_summary)

    def test_08_pipeline_fail_closed_guarantee(self):
        """Proves that passing an invalid candidate to ReAgentPipeline escalates and never verifies."""
        from reagent.providers.base import LLMProvider

        class MockInvalidCandidateProvider(LLMProvider):
            def generate(self, *a, **k):
                return LLMResponse(content='{}', model_name='mock-bad-provider')
            def diagnose_failure(self, *a, **k):
                pass
            def select_recovery(self, *a, **k):
                pass
            def make_structured_decision(self, *a, **k):
                # Returns candidate with empty proposed postconditions
                return StructuredDecision(
                    tool="dbops_database",
                    arguments={"task_id": "task_02_wallet_transfer"},
                    proposed_postconditions=[],
                    provenance="fixture",
                )

        store = DbopsStore.create_temp()
        try:
            store.seed_initial_state("task_02_wallet_transfer")
            tool = DbopsDatabaseTool(store)
            pipeline = ReAgentPipeline(
                agent_id="test-gate-agent",
                provider=MockInvalidCandidateProvider(),
            )
            pipeline.tool_registry.register(tool)

            # Executing task objective with invalid candidate fails grounding and escalates
            result = pipeline.execute_task(
                task_objective="Transfer $500.00 from wallet ACCT-2201 to wallet ACCT-3390. Transfer request ID: TXN-779043."
            )

            self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
            self.assertNotEqual(result.verified_receipt.final_status, AgentStatus.VERIFIED)
            self.assertEqual(pipeline.agent_state.status, AgentStatus.ESCALATED)
        finally:
            store.cleanup()

    def test_09_zero_network_calls_and_model_name_compatibility(self):
        """Proves zero network calls occur and LLMResponse.model_name remains warning-free."""
        with patch("urllib.request.urlopen") as mock_url, \
             patch("http.client.HTTPConnection.request") as mock_http, \
             patch.dict(os.environ, {"NEBIUS_API_KEY": ""}):

            # Instantiating LLMResponse with model_name
            resp = LLMResponse(content="{}", model_name="nemotron-offline-fixture")
            self.assertEqual(resp.model_name, "nemotron-offline-fixture")

            # Unconfigured provider fails fast offline
            provider = NebiusNemotronProvider(client=None)
            self.assertFalse(provider.is_configured())

            self.assertEqual(mock_url.call_count, 0)
            self.assertEqual(mock_http.call_count, 0)


if __name__ == "__main__":
    unittest.main()
