"""
ReAgent Phase 2 Tests: Structured Tri-State Outcomes & Deterministic Verification

Validates:
- Explicit PostconditionStatus enum (MET, UNMET, UNDETERMINED)
- PostconditionEvaluation record preservation
- Pure aggregate_verification_outcome rules (1-8):
  * all MET -> VERIFIED
  * all UNMET + authoritative -> VERIFIED_FAILURE
  * all UNMET + non-authoritative -> UNKNOWN
  * mixed MET/UNMET -> PARTIAL
  * any UNDETERMINED -> UNKNOWN
  * recovery exhausted -> RECOVERY_EXHAUSTED (with known and unknown underlying evaluations)
  * empty evaluations -> rejected / invalid (never success)
  * tool success != task success
  * model proposal cannot override independent evidence
- Backward compatibility with VerificationResult, VerifiedReceipt, and AgentStatus consumers
"""

import unittest
import pytest
from typing import List

from reagent.taxonomy import PostconditionStatus, AgentStatus
from reagent.models import (
    Postcondition,
    PostconditionEvaluation,
    VerificationResult,
    VerifiedReceipt,
    SuccessContract,
    Evidence,
    ActionResult,
)
from reagent.verification.verifier import (
    DeterministicVerifier,
    AssertionRule,
    aggregate_verification_outcome,
)
from reagent.contract import ContractManager


class TestPostconditionStatusEnum(unittest.TestCase):
    """Ensure PostconditionStatus and AgentStatus extensions are complete and valid."""

    def test_postcondition_status_enum_values(self):
        self.assertEqual(PostconditionStatus.MET.value, "MET")
        self.assertEqual(PostconditionStatus.UNMET.value, "UNMET")
        self.assertEqual(PostconditionStatus.UNDETERMINED.value, "UNDETERMINED")

    def test_agent_status_extensions(self):
        self.assertEqual(AgentStatus.PARTIAL.value, "PARTIAL")
        self.assertEqual(AgentStatus.VERIFIED_FAILURE.value, "VERIFIED_FAILURE")
        self.assertEqual(AgentStatus.UNKNOWN.value, "UNKNOWN")
        self.assertEqual(AgentStatus.RECOVERY_EXHAUSTED.value, "RECOVERY_EXHAUSTED")


class TestPureOutcomeAggregation(unittest.TestCase):
    """Direct tests for the pure aggregate_verification_outcome function."""

    def test_all_met_yields_verified(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="STATUS_SUCCESS",
                status=PostconditionStatus.MET,
                explanation="success=True",
            ),
            PostconditionEvaluation(
                postcondition_id="pc2",
                predicate="KEY_EXISTS",
                target="booking_id",
                status=PostconditionStatus.MET,
                explanation="booking_id present",
            ),
        ]
        status = aggregate_verification_outcome(evals, is_authoritative=True)
        self.assertEqual(status, AgentStatus.VERIFIED)

    def test_all_unmet_authoritative_yields_verified_failure(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="KEY_EXISTS",
                target="booking_id",
                status=PostconditionStatus.UNMET,
                explanation="key missing",
            ),
            PostconditionEvaluation(
                postcondition_id="pc2",
                predicate="VALUE_EQUALS",
                target="status",
                expected="CONFIRMED",
                status=PostconditionStatus.UNMET,
                explanation="status is not CONFIRMED",
            ),
        ]
        status = aggregate_verification_outcome(evals, is_authoritative=True)
        self.assertEqual(status, AgentStatus.VERIFIED_FAILURE)

    def test_all_unmet_non_authoritative_yields_unknown(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="KEY_EXISTS",
                target="booking_id",
                status=PostconditionStatus.UNMET,
                explanation="key missing in unconfirmed state",
            )
        ]
        status = aggregate_verification_outcome(evals, is_authoritative=False)
        self.assertEqual(status, AgentStatus.UNKNOWN)

    def test_mixed_met_and_unmet_authoritative_yields_partial(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="STATUS_SUCCESS",
                status=PostconditionStatus.MET,
                explanation="tool succeeded",
            ),
            PostconditionEvaluation(
                postcondition_id="pc2",
                predicate="KEY_EXISTS",
                target="confirmation_code",
                status=PostconditionStatus.UNMET,
                explanation="confirmation_code missing",
            ),
        ]
        status = aggregate_verification_outcome(evals, is_authoritative=True)
        self.assertEqual(status, AgentStatus.PARTIAL)

    def test_mixed_met_and_unmet_non_authoritative_yields_unknown(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="STATUS_SUCCESS",
                status=PostconditionStatus.MET,
                explanation="tool succeeded",
            ),
            PostconditionEvaluation(
                postcondition_id="pc2",
                predicate="KEY_EXISTS",
                target="confirmation_code",
                status=PostconditionStatus.UNMET,
                explanation="confirmation_code missing",
            ),
        ]
        status = aggregate_verification_outcome(evals, is_authoritative=False)
        self.assertEqual(status, AgentStatus.UNKNOWN)

    def test_any_undetermined_yields_unknown(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="STATUS_SUCCESS",
                status=PostconditionStatus.MET,
                explanation="succeeded",
            ),
            PostconditionEvaluation(
                postcondition_id="pc2",
                predicate="FILE_EXISTS",
                target="audit.log",
                status=PostconditionStatus.UNDETERMINED,
                explanation="permission denied inspecting audit.log",
            ),
        ]
        status = aggregate_verification_outcome(evals, is_authoritative=True)
        self.assertEqual(status, AgentStatus.UNKNOWN)

    def test_recovery_exhausted_with_known_evaluations(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="KEY_EXISTS",
                target="receipt_id",
                status=PostconditionStatus.UNMET,
                explanation="missing",
            )
        ]
        status = aggregate_verification_outcome(
            evals,
            is_authoritative=True,
            recovery_exhausted=True,
        )
        self.assertEqual(status, AgentStatus.RECOVERY_EXHAUSTED)

    def test_recovery_exhausted_with_unknown_evaluations(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="FILE_EXISTS",
                target="db.sqlite",
                status=PostconditionStatus.UNDETERMINED,
                explanation="file lock timeout",
            )
        ]
        status = aggregate_verification_outcome(
            evals,
            is_authoritative=False,
            recovery_exhausted=True,
        )
        self.assertEqual(status, AgentStatus.RECOVERY_EXHAUSTED)

    def test_empty_evaluations_fails_validation(self):
        with self.assertRaises(ValueError):
            aggregate_verification_outcome([], raise_on_empty=True)

        explicit_failed = aggregate_verification_outcome([], raise_on_empty=False)
        self.assertEqual(explicit_failed, AgentStatus.FAILED)

    def test_model_proposal_cannot_override_independent_evidence(self):
        evals = [
            PostconditionEvaluation(
                postcondition_id="pc1",
                predicate="VALUE_EQUALS",
                target="balance",
                expected=500,
                status=PostconditionStatus.UNMET,
                explanation="balance is 0, not 500",
            )
        ]
        # Model claims it passed and requests override
        status = aggregate_verification_outcome(
            evals,
            is_authoritative=True,
            model_proposal_override=True,
        )
        self.assertEqual(status, AgentStatus.VERIFIED_FAILURE)
        self.assertNotEqual(status, AgentStatus.VERIFIED)


class TestDeterministicVerifierIntegration(unittest.TestCase):
    """Tests for DeterministicVerifier integrating per-postcondition evaluations."""

    def setUp(self):
        self.verifier = DeterministicVerifier()

    def test_tool_success_does_not_imply_task_success(self):
        """Action returned success=True, but an invariant postcondition failed."""
        evidence = Evidence(
            source="test",
            facts={
                "success": True,  # Tool claims transport/execution success
                "output": {"status": "PENDING"},
                "is_authoritative": True,
            },
        )

        assertions = [
            AssertionRule(assertion_type="STATUS_SUCCESS"),
            AssertionRule(assertion_type="KEY_EXISTS", key_path="booking_id"),
            AssertionRule(assertion_type="VALUE_EQUALS", key_path="status", expected_value="CONFIRMED"),
        ]
        res = self.verifier.verify(
            recovery_decision_id="dec-1",
            evidence=evidence,
            assertions=assertions,
        )
        self.assertFalse(res.verified)
        self.assertEqual(res.aggregate_status, AgentStatus.PARTIAL)
        self.assertEqual(len(res.postcondition_evaluations), 3)

        # First postcondition passed
        self.assertEqual(res.postcondition_evaluations[0].status, PostconditionStatus.MET)
        # Second postcondition failed
        self.assertEqual(res.postcondition_evaluations[1].status, PostconditionStatus.UNMET)
        # Third postcondition failed
        self.assertEqual(res.postcondition_evaluations[2].status, PostconditionStatus.UNMET)

    def test_all_assertions_pass_produces_verified_result(self):
        evidence = Evidence(
            source="test",
            facts={
                "success": True,
                "output": {"status": "CONFIRMED", "booking_id": "BK-123"},
                "is_authoritative": True,
            },
        )
        assertions = [
            AssertionRule(assertion_type="STATUS_SUCCESS"),
            AssertionRule(assertion_type="KEY_EXISTS", key_path="booking_id"),
            AssertionRule(assertion_type="VALUE_EQUALS", key_path="status", expected_value="CONFIRMED"),
        ]
        res = self.verifier.verify(
            recovery_decision_id="dec-2",
            evidence=evidence,
            assertions=assertions,
        )
        self.assertTrue(res.verified)
        self.assertEqual(res.aggregate_status, AgentStatus.VERIFIED)
        self.assertEqual(len(res.discrepancies), 0)
        for eval_rec in res.postcondition_evaluations:
            self.assertEqual(eval_rec.status, PostconditionStatus.MET)

    def test_unobservable_evidence_marks_undetermined_and_unknown(self):
        evidence = Evidence(
            source="test",
            facts={
                "success": True,
                "unobservable": True,
                "status": "UNOBSERVABLE",
            },
        )
        assertions = [
            AssertionRule(assertion_type="STATUS_SUCCESS"),
            AssertionRule(assertion_type="KEY_EXISTS", key_path="ledger_id"),
        ]
        res = self.verifier.verify(
            recovery_decision_id="dec-3",
            evidence=evidence,
            assertions=assertions,
        )
        self.assertFalse(res.verified)
        self.assertEqual(res.aggregate_status, AgentStatus.UNKNOWN)

    def test_empty_assertions_list_rejected(self):
        evidence = Evidence(source="test", facts={"success": True})
        res = self.verifier.verify(
            recovery_decision_id="dec-empty",
            evidence=evidence,
            assertions=[],
        )
        self.assertFalse(res.verified)
        self.assertEqual(res.aggregate_status, AgentStatus.FAILED)
        self.assertEqual(len(res.postcondition_evaluations), 0)


class TestBackwardCompatibilityAndReceipts(unittest.TestCase):
    """Ensure existing consumers of VerificationResult and VerifiedReceipt function without errors."""

    def test_verification_result_legacy_attributes(self):
        res = VerificationResult(
            recovery_decision_id="legacy-1",
            verified=True,
            evidence_summary="All passed",
            discrepancies=[],
        )
        self.assertTrue(res.verified)
        self.assertEqual(res.evidence_summary, "All passed")
        self.assertEqual(res.discrepancies, [])
        self.assertEqual(res.postcondition_evaluations, [])
        self.assertIsNone(res.aggregate_status)

    def test_verified_receipt_with_structured_statuses(self):
        contract = SuccessContract(
            task_id="task-100",
            postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
        )
        evidence = Evidence(source="audit", facts={"success": True})
        v_res = VerificationResult(
            recovery_decision_id="rec-100",
            verified=False,
            evidence_summary="Partial success",
            discrepancies=["one failure"],
            postcondition_evaluations=[
                PostconditionEvaluation(
                    postcondition_id="pc1",
                    predicate="STATUS_SUCCESS",
                    status=PostconditionStatus.MET,
                ),
            ],
            aggregate_status=AgentStatus.PARTIAL,
        )

        receipt = VerifiedReceipt(
            task_id="task-100",
            action_id="act-100",
            tool="hotel_booking",
            frozen_contract=contract,
            evidence=evidence,
            verification_result=v_res,
            recovery_count=1,
            final_status=AgentStatus.PARTIAL,
        )
        self.assertEqual(receipt.final_status, AgentStatus.PARTIAL)
        self.assertEqual(receipt.verification_result.aggregate_status, AgentStatus.PARTIAL)
        receipt_dict = receipt.model_dump()
        self.assertEqual(receipt_dict["final_status"], "PARTIAL")

    def test_contract_manager_preserves_postcondition_ids(self):
        mgr = ContractManager()
        pcs = [
            Postcondition(
                predicate="KEY_EXISTS",
                target="receipt_num",
                description="Receipt must exist",
                postcondition_id="custom-pc-id-42",
            )
        ]
        contract = mgr.build_contract("task-xyz", pcs)
        frozen = mgr.freeze(contract)
        rules = mgr.to_assertion_rules(frozen)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].rule_id, "custom-pc-id-42")


if __name__ == "__main__":
    unittest.main()
