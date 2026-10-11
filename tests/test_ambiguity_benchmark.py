"""
test_ambiguity_benchmark.py — Comprehensive offline test suite for ambiguity experiment.
Covers:
1. Identifiability invariant between FAIL_BEFORE_COMMIT and LOST_RESPONSE
2. Authoritative Oracle isolation
3. Policy arm execution
4. Pre-registered expectations confirmation
"""

import unittest
from reagent.experiments.ambiguity.manifest import (
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
    FAULT_UNOBSERVABLE,
    FAULT_DELAYED_VISIBILITY,
    FAULT_STATUS_UNAVAILABLE,
    OP_CREATE_RESOURCE,
    OP_APPLY_TRANSFER,
    POLICY_NAIVE_RETRY,
    POLICY_REAGENT_CONTRACT,
    POLICY_ALWAYS_ESCALATE,
    POLICY_NEVER_RETRY,
    POLICY_SHARED_OBSERVATION,
)
from reagent.experiments.ambiguity.faults import (
    ToolExecutionTimeoutError,
    make_standard_timeout_error,
)
from reagent.experiments.ambiguity.simulator import StatefulSimulator
from reagent.experiments.ambiguity.oracle import AmbiguityOracle
from reagent.experiments.ambiguity.policies import run_policy
from reagent.experiments.ambiguity.runner import verify_identifiability, run_benchmark


class TestAmbiguityBenchmark(unittest.TestCase):

    def test_identifiability_invariant(self):
        """Matched FAIL_BEFORE_COMMIT and LOST_RESPONSE must produce identical agent errors."""
        self.assertTrue(verify_identifiability())

        err_fail = make_standard_timeout_error("create_resource", "req-1")
        err_lost = make_standard_timeout_error("create_resource", "req-1")
        self.assertEqual(err_fail.canonical_hash(), err_lost.canonical_hash())
        self.assertEqual(err_fail.to_canonical_dict(), err_lost.to_canonical_dict())

    def test_oracle_isolation(self):
        """Oracle observes authoritative state without importing pipeline or verifier."""
        sim = StatefulSimulator()
        sim.execute_operation(OP_CREATE_RESOURCE, {"resource_id": "r1"}, fault_mode=None)
        eval_res = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        self.assertEqual(eval_res["authoritative_commit_count"], 1)
        self.assertTrue(eval_res["exactly_once_committed"])
        self.assertFalse(eval_res["duplicate_committed"])

    def test_naive_retry_duplicates_on_lost_response(self):
        """Naive retry causes duplicate commit under LOST_RESPONSE."""
        sim = StatefulSimulator()
        params = {"resource_id": "res_dup", "request_id": "r1"}
        res = run_policy(POLICY_NAIVE_RETRY, sim, OP_CREATE_RESOURCE, params, fault_mode=FAULT_LOST_RESPONSE)
        self.assertTrue(res.retried)

        oracle_eval = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        self.assertEqual(oracle_eval["authoritative_commit_count"], 2)
        self.assertTrue(oracle_eval["duplicate_committed"])

    def test_reagent_contract_restraint_on_lost_response(self):
        """ReAgent restraint check prevents duplicate commit on LOST_RESPONSE."""
        sim = StatefulSimulator()
        params = {"resource_id": "res_restraint", "request_id": "r1"}
        res = run_policy(POLICY_REAGENT_CONTRACT, sim, OP_CREATE_RESOURCE, params, fault_mode=FAULT_LOST_RESPONSE)
        self.assertFalse(res.retried)
        self.assertTrue(res.observed_existing_state)

        oracle_eval = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        self.assertEqual(oracle_eval["authoritative_commit_count"], 1)
        self.assertTrue(oracle_eval["exactly_once_committed"])
        self.assertFalse(oracle_eval["duplicate_committed"])

    def test_reagent_contract_retries_on_fail_before_commit(self):
        """ReAgent retries safely when probe confirms effect did not commit."""
        sim = StatefulSimulator()
        params = {"resource_id": "res_retry", "request_id": "r1"}
        res = run_policy(POLICY_REAGENT_CONTRACT, sim, OP_CREATE_RESOURCE, params, fault_mode=FAULT_FAIL_BEFORE_COMMIT)
        self.assertTrue(res.retried)

        oracle_eval = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        self.assertEqual(oracle_eval["authoritative_commit_count"], 1)
        self.assertTrue(oracle_eval["exactly_once_committed"])

    def test_reagent_contract_escalates_on_unobservable(self):
        """ReAgent contract policy fails-closed (escalates) on unobservable operations."""
        sim = StatefulSimulator()
        params = {"resource_id": "res_unobs", "request_id": "r1"}
        res = run_policy(POLICY_REAGENT_CONTRACT, sim, OP_CREATE_RESOURCE, params, fault_mode=FAULT_UNOBSERVABLE)
        self.assertTrue(res.escalated)
        self.assertEqual(res.agent_status, "ESCALATED")

    def test_benchmark_full_run(self):
        """Runs the entire benchmark matrix and verifies counts."""
        res = run_benchmark()
        self.assertTrue(res["identifiability_passed"])
        self.assertEqual(res["total_trials"], 90)  # 6 policies * 5 fault modes * 3 operations


if __name__ == "__main__":
    unittest.main()
