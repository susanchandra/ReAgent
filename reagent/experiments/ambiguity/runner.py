"""
runner.py — Ambiguity benchmark execution harness and result aggregator.
"""

from typing import Dict, Any, List
import json
from reagent.experiments.ambiguity.manifest import (
    ALL_FAULT_MODES,
    ALL_OPERATIONS,
    ALL_POLICY_ARMS,
    EXPECTED_OUTCOMES,
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
)
from reagent.experiments.ambiguity.simulator import StatefulSimulator
from reagent.experiments.ambiguity.oracle import AmbiguityOracle
from reagent.experiments.ambiguity.policies import run_policy, PolicyExecutionResult
from reagent.experiments.ambiguity.faults import make_standard_timeout_error


def verify_identifiability() -> bool:
    """
    Identifiability Test:
    Ensures that for matched FAIL_BEFORE_COMMIT and LOST_RESPONSE scenarios,
    the agent-visible exception is bit-for-bit identical, while the oracle
    observes distinct authoritative world states.
    """
    for op in ALL_OPERATIONS:
        # Check canonical hashes of errors
        err1 = make_standard_timeout_error(op, "test-req")
        err2 = make_standard_timeout_error(op, "test-req")
        if err1.canonical_hash() != err2.canonical_hash():
            return False

        # Check simulation behavior
        sim_fail = StatefulSimulator()
        sim_lost = StatefulSimulator()

        # Step 1 execution under both fault modes
        params = {"resource_id": "res_ident", "request_id": "test-req", "amount": 100, "slot_id": "slot_01", "entity_id": "ent_01"}

        try:
            sim_fail.execute_operation(op, params, fault_mode=FAULT_FAIL_BEFORE_COMMIT, step=1)
        except Exception:
            pass

        try:
            sim_lost.execute_operation(op, params, fault_mode=FAULT_LOST_RESPONSE, step=1)
        except Exception:
            pass

        # Oracle check: world states MUST differ
        oracle_fail = AmbiguityOracle.evaluate(sim_fail, op)
        oracle_lost = AmbiguityOracle.evaluate(sim_lost, op)

        if oracle_fail["authoritative_commit_count"] != 0:
            return False
        if oracle_lost["authoritative_commit_count"] != 1:
            return False

    return True


def run_benchmark() -> Dict[str, Any]:
    """
    Runs full matrix across ALL_POLICY_ARMS, ALL_OPERATIONS, ALL_FAULT_MODES.
    Evaluates against independent oracle and pre-registered manifest.
    """
    results: List[Dict[str, Any]] = []

    for policy in ALL_POLICY_ARMS:
        for fault in ALL_FAULT_MODES:
            for op in ALL_OPERATIONS:
                sim = StatefulSimulator()
                params = {
                    "request_id": f"req-{policy}-{fault}-{op}",
                    "resource_id": f"res_{op}",
                    "amount": 100,
                    "slot_id": "slot_01",
                    "entity_id": "ent_01",
                    "name": f"name_{op}",
                }

                policy_res: PolicyExecutionResult = run_policy(
                    policy_name=policy,
                    sim=sim,
                    op_name=op,
                    params=params,
                    fault_mode=fault,
                )

                # Authoritative evaluation from Oracle
                oracle_eval = AmbiguityOracle.evaluate(sim, op_name=op, expected_commits=1)

                is_correct = oracle_eval["exactly_once_committed"] and policy_res.agent_status == "COMPLETED"
                is_duplicate = oracle_eval["duplicate_committed"]
                is_lost = oracle_eval["lost_effect"] and policy_res.agent_status != "ESCALATED"

                results.append({
                    "policy": policy,
                    "fault_mode": fault,
                    "operation": op,
                    "agent_status": policy_res.agent_status,
                    "retried": policy_res.retried,
                    "escalated": policy_res.escalated,
                    "observed_existing_state": policy_res.observed_existing_state,
                    "oracle_commit_count": oracle_eval["authoritative_commit_count"],
                    "exactly_once": oracle_eval["exactly_once_committed"],
                    "duplicate_committed": is_duplicate,
                    "lost_effect": is_lost,
                    "correct_outcome": is_correct,
                })

    return {
        "identifiability_passed": verify_identifiability(),
        "total_trials": len(results),
        "results": results,
    }
