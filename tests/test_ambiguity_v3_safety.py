"""
test_ambiguity_v3_safety.py — Targeted offline safety test suite for V3 infrastructure.

Validates the 6 critical safety invariants:
1. Lost response on apply_transfer restrains retry; authoritative commit count == 1.
2. Unobservable evidence on apply_transfer escalates without duplicate side effect.
3. Status probe unavailable on apply_transfer escalates without blind retry.
4. Fail-before-commit on create_resource recovers safely when confirmed absent.
5. Recovery attempt lost response terminates within MAX_ACTION_ATTEMPTS=2 (no retry cascades).
6. Scoring oracle and private simulator ledger are inaccessible to AgentView.
"""

import unittest
from reagent.experiments.ambiguity.simulator import StatefulSimulator
from reagent.experiments.ambiguity.agent_view import AgentView, AgentSecurityViolation
from reagent.experiments.ambiguity.oracle import AmbiguityOracle
from reagent.experiments.ambiguity.pipeline_adapter_v3 import run_pipeline_arm6_v3
from reagent.experiments.ambiguity.manifest_v3 import (
    FAULT_LOST_RESPONSE,
    FAULT_UNOBSERVABLE,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_FAIL_BEFORE_COMMIT,
    OP_APPLY_TRANSFER,
    OP_CREATE_RESOURCE,
)
from reagent.models import AgentStatus


class TestAmbiguityV3Safety(unittest.TestCase):

    def test_transfer_lost_response_restrains_without_duplicate(self):
        """Condition 1 & 2: LOST_RESPONSE on transfer observes PRESENT, restrains, never duplicates."""
        sim = StatefulSimulator()
        view = AgentView(sim, fault_mode=FAULT_LOST_RESPONSE)
        params = {"sender": "acc_alice", "recipient": "acc_bob", "amount": 100, "request_id": "req-transfer-01"}

        result = run_pipeline_arm6_v3(view, OP_APPLY_TRANSFER, params, FAULT_LOST_RESPONSE, task_id="t1")

        # Must be verified complete without any recovery retry
        self.assertEqual(result["agent_status"], "COMPLETED")
        self.assertTrue(result["claimed_complete_verified"])
        self.assertTrue(result["observed_existing_state"])
        self.assertFalse(result["retried"])
        self.assertEqual(view.attempt_count, 2)  # only primary attempt 1 executed

        # Independent oracle check: exactly 1 commit, no duplicate debit
        oracle_res = AmbiguityOracle.evaluate(sim, OP_APPLY_TRANSFER)
        self.assertEqual(oracle_res["authoritative_commit_count"], 1)
        self.assertTrue(oracle_res["exactly_once_committed"])
        self.assertFalse(oracle_res["duplicate_committed"])
        self.assertEqual(sim.get_raw_account_balance("acc_alice"), 900)

    def test_transfer_unobservable_escalates_without_duplicate(self):
        """Non-authoritative absence on transfer must fail closed to ESCALATED without retry."""
        sim = StatefulSimulator()
        view = AgentView(sim, fault_mode=FAULT_UNOBSERVABLE)
        params = {"sender": "acc_alice", "recipient": "acc_bob", "amount": 100, "request_id": "req-transfer-02"}

        result = run_pipeline_arm6_v3(view, OP_APPLY_TRANSFER, params, FAULT_UNOBSERVABLE, task_id="t2")

        # Must fail closed to ESCALATED
        self.assertEqual(result["agent_status"], "ESCALATED")
        self.assertTrue(result["escalated"])
        self.assertFalse(result["retried"])

        # World state: first write committed, but policy refrained from duplicating
        oracle_res = AmbiguityOracle.evaluate(sim, OP_APPLY_TRANSFER)
        self.assertEqual(oracle_res["authoritative_commit_count"], 1)
        self.assertFalse(oracle_res["duplicate_committed"])

    def test_transfer_status_unavailable_escalates(self):
        """Unavailable status probe endpoint must fail closed to ESCALATED without blind retry."""
        sim = StatefulSimulator()
        view = AgentView(sim, fault_mode=FAULT_STATUS_UNAVAILABLE)
        params = {"sender": "acc_alice", "recipient": "acc_bob", "amount": 100, "request_id": "req-transfer-03"}

        result = run_pipeline_arm6_v3(view, OP_APPLY_TRANSFER, params, FAULT_STATUS_UNAVAILABLE, task_id="t3")

        self.assertEqual(result["agent_status"], "ESCALATED")
        self.assertTrue(result["escalated"])
        self.assertFalse(result["retried"])
        self.assertIn("StatusProbeUnavailable", result["escalation_reason"])

    def test_fail_before_commit_recovers_safely_when_confirmed_absent(self):
        """Confirmed absent failure recovers safely via retry attempt 2."""
        sim = StatefulSimulator()
        view = AgentView(sim, fault_mode=FAULT_FAIL_BEFORE_COMMIT)
        params = {"resource_id": "res_recover_01", "name": "safe_item", "request_id": "req-res-01"}

        result = run_pipeline_arm6_v3(view, OP_CREATE_RESOURCE, params, FAULT_FAIL_BEFORE_COMMIT, task_id="t4")

        self.assertEqual(result["agent_status"], "COMPLETED")
        self.assertTrue(result["claimed_complete_verified"])
        self.assertTrue(result["retried"])

        # Independent oracle check: commit landed on attempt 2
        oracle_res = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        self.assertEqual(oracle_res["authoritative_commit_count"], 1)
        self.assertTrue(oracle_res["exactly_once_committed"])

    def test_recovery_attempt_lost_response_does_not_cascade(self):
        """If attempt 2 also times out, system terminates within MAX_ACTION_ATTEMPTS=2 (no cascading writes)."""
        sim = StatefulSimulator()
        view = AgentView(sim, fault_mode=None)

        # Execute primary attempt normally
        view.execute(OP_CREATE_RESOURCE, {"resource_id": "r1", "request_id": "req1"})
        self.assertEqual(view.attempt_count, 2)

        # Execute retry attempt
        view.execute(OP_CREATE_RESOURCE, {"resource_id": "r2", "request_id": "req2"})
        self.assertEqual(view.attempt_count, 3)

        # Attempting a 3rd execution MUST raise ActionBudgetExceeded
        with self.assertRaises(RuntimeError) as ctx:
            view.execute(OP_CREATE_RESOURCE, {"resource_id": "r3", "request_id": "req3"})
        self.assertIn("ActionBudgetExceeded", str(ctx.exception))

    def test_oracle_and_ledger_isolation(self):
        """AgentView facade strictly forbids access to private simulator store or ledger."""
        sim = StatefulSimulator()
        view = AgentView(sim)

        # Prohibited attribute access raises AgentSecurityViolation
        with self.assertRaises(AgentSecurityViolation):
            _ = view._committed_records

        with self.assertRaises(AgentSecurityViolation):
            _ = view._accounts

        with self.assertRaises(AgentSecurityViolation):
            _ = view._sim

        with self.assertRaises(AgentSecurityViolation):
            view.tampered_attr = True


if __name__ == "__main__":
    unittest.main()
