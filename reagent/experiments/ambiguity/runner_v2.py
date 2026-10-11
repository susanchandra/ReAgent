"""
runner_v2.py — Execution harness and dual-view scoring for Ambiguity Benchmark V2.

Guarantees:
1. Strict Dual-View Partition Scoring:
   - World State: [exactly_one_effect, duplicate_effect, lost_effect, no_effect_expected] sums to total trials
   - Agent Claim: [claimed_complete_verified, claimed_complete_unverified, escalated, failed_terminal] sums to total trials
2. Deterministic JSONL generation with normalized fields.
3. Independent Oracle ground-truth evaluation.
"""

from typing import Dict, Any, List
import json
import hashlib
from reagent.experiments.ambiguity.manifest_v2 import (
    ALL_FAULT_MODES,
    ALL_OPERATIONS,
    ALL_POLICY_ARMS,
    FIXED_SEEDS,
    MANIFEST_VERSION,
    FAULT_TO_OBSERVABILITY_CLASS,
    WORLD_STATE_CATEGORIES,
    AGENT_CLAIM_CATEGORIES,
    FAULT_NO_FAULT,
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
    FAULT_DELAYED_VISIBILITY,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_UNOBSERVABLE,
)
from reagent.experiments.ambiguity.simulator import StatefulSimulator
from reagent.experiments.ambiguity.agent_view import AgentView
from reagent.experiments.ambiguity.oracle import AmbiguityOracle
from reagent.experiments.ambiguity.policies_v2 import execute_policy_v2, PolicyRunResult


def run_benchmark_v2(output_jsonl_path: str = "D:/ReAgent/reports/ambiguity_trials_v2.jsonl") -> Dict[str, Any]:
    """
    Executes the full Ambiguity V2 benchmark matrix:
    6 policy arms * 6 fault modes * 3 operations * 4 seeds = 432 trials.
    Writes deterministic JSONL logs and returns partition metrics.
    """
    trials: List[Dict[str, Any]] = []

    for arm in ALL_POLICY_ARMS:
        for fault in ALL_FAULT_MODES:
            for op in ALL_OPERATIONS:
                for seed in FIXED_SEEDS:
                    case_id = f"case_{arm}_{fault}_{op}_s{seed}"
                    task_id = f"task_{arm}_{fault}_{op}_{seed}"

                    # Fresh stateful simulator per trial
                    sim = StatefulSimulator()
                    agent_view = AgentView(simulator=sim, fault_mode=fault)

                    params = {
                        "request_id": f"req_{seed}_{op}",
                        "resource_id": f"res_{seed}_{op}",
                        "amount": 100,
                        "slot_id": "slot_01",
                        "entity_id": f"ent_{seed}",
                        "name": f"item_{seed}",
                    }

                    # Execute policy strictly through AgentView
                    policy_res: PolicyRunResult = execute_policy_v2(
                        policy_name=arm,
                        agent_view=agent_view,
                        op_name=op,
                        params=params,
                        fault_mode=fault,
                        task_id=task_id,
                    )

                    # Authoritative independent oracle evaluation
                    oracle_eval = AmbiguityOracle.evaluate(sim, op_name=op, expected_commits=1)
                    commit_count = oracle_eval["authoritative_commit_count"]

                    # ── View A: World State Partition ─────────────────────────
                    # Mutually exclusive: exactly one of these 4 must be True
                    ws_exactly_one = (commit_count == 1)
                    ws_duplicate = (commit_count > 1)
                    ws_lost = (commit_count == 0)
                    ws_no_effect_expected = False # All our ops intended 1 commit

                    # ── View B: Agent Claim Partition ─────────────────────────
                    # Mutually exclusive: exactly one of these 4 must be True
                    claim_verified = policy_res.claimed_complete_verified
                    claim_unverified = policy_res.claimed_complete_unverified
                    claim_escalated = policy_res.escalated
                    claim_failed = policy_res.failed_terminal

                    trial_record = {
                        "manifest_version": MANIFEST_VERSION,
                        "case_id": case_id,
                        "seed": seed,
                        "policy_arm": arm,
                        "fault_mode": fault,
                        "observability_class": FAULT_TO_OBSERVABILITY_CLASS[fault],
                        "operation": op,
                        "agent_status": policy_res.agent_status,
                        # World State Partition
                        "world_state": {
                            "commit_count": commit_count,
                            "exactly_one_effect": ws_exactly_one,
                            "duplicate_effect": ws_duplicate,
                            "lost_effect": ws_lost,
                            "no_effect_expected": ws_no_effect_expected,
                        },
                        # Agent Claim Partition
                        "agent_claim": {
                            "claimed_complete_verified": claim_verified,
                            "claimed_complete_unverified": claim_unverified,
                            "escalated": claim_escalated,
                            "failed_terminal": claim_failed,
                        },
                        "behavior": {
                            "retried": policy_res.retried,
                            "observed_existing_state": policy_res.observed_existing_state,
                            "escalation_reason": policy_res.escalation_reason,
                        },
                    }
                    trials.append(trial_record)

    # Write deterministic JSONL (normalized without random timestamps)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t, sort_keys=True) + "\n")

    return {
        "manifest_version": MANIFEST_VERSION,
        "total_trials": len(trials),
        "jsonl_path": output_jsonl_path,
        "trials": trials,
    }
