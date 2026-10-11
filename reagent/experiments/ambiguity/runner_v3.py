"""
runner_v3.py — Execution harness and dual-view scoring for Ambiguity Benchmark V3.

DESIGN PRINCIPLES
-----------------
1. Strictly follows committed manifest 5a0542d. No scoring rules are altered
   post-execution. Expected outcomes are recorded in manifest_v3.py comments.
2. Strict Dual-View Partition Conservation:
   - World State: [exactly_one_effect, duplicate_effect, lost_effect, no_effect_expected]
     must sum to total trials per arm (72).
   - Agent Claim: [claimed_complete_verified, claimed_complete_unverified, escalated,
     failed_terminal] must sum to total trials per arm (72).
   Both conservation invariants are asserted before JSONL is written.
3. Oracle is called AFTER policy execution and its result is NEVER passed to the
   policy or verifier as a runtime input. Oracle isolation is test-enforced.
4. JSONL fields are normalized (no timestamps in compared fields); sort_keys=True
   for deterministic output ordering.
5. Policy arm 6 (REAGENT_CONTRACT_POLICY) execution may be slower due to pipeline
   overhead. The runner does not skip or short-circuit any arm.
"""

import json
import os
from typing import Dict, Any, List

from reagent.experiments.ambiguity.manifest_v3 import (
    ALL_FAULT_MODES,
    ALL_OPERATIONS,
    ALL_POLICY_ARMS,
    FIXED_SEEDS,
    MANIFEST_VERSION,
    FAULT_TO_OBSERVABILITY_CLASS,
    WORLD_STATE_CATEGORIES,
    AGENT_CLAIM_CATEGORIES,
    EXECUTIONS_PER_ARM,
    TOTAL_BENCHMARK_EXECUTIONS,
)
from reagent.experiments.ambiguity.simulator import StatefulSimulator
from reagent.experiments.ambiguity.agent_view import AgentView
from reagent.experiments.ambiguity.oracle import AmbiguityOracle
from reagent.experiments.ambiguity.policies_v3 import execute_policy_v3, PolicyRunResult


# Default output paths (reports/ is git-ignored; these are ephemeral outputs)
DEFAULT_JSONL_PATH = "D:/ReAgent/reports/ambiguity_trials_v3.jsonl"


def _build_params(seed: int, op: str) -> Dict[str, Any]:
    """Deterministic parameter set from seed and operation name."""
    return {
        "request_id": f"req_{seed}_{op}",
        "resource_id": f"res_{seed}_{op}",
        "amount": 100,
        "slot_id": "slot_01",
        "entity_id": f"ent_{seed}",
        "name": f"item_{seed}",
        "sender": "acc_alice",
    }


def _assert_partition_conservation(
    trials: List[Dict[str, Any]],
    arm: str,
) -> None:
    """
    Asserts that both View A (world state) and View B (agent claim) partitions
    sum correctly for the given arm subset. Raises AssertionError on violation.
    """
    arm_trials = [t for t in trials if t["policy_arm"] == arm]
    expected_count = EXECUTIONS_PER_ARM

    assert len(arm_trials) == expected_count, (
        f"Arm {arm}: expected {expected_count} trials, got {len(arm_trials)}"
    )

    # View A: World State partition
    ws_sum = sum(
        int(t["world_state"]["exactly_one_effect"]) +
        int(t["world_state"]["duplicate_effect"]) +
        int(t["world_state"]["lost_effect"]) +
        int(t["world_state"]["no_effect_expected"])
        for t in arm_trials
    )
    assert ws_sum == expected_count, (
        f"Arm {arm}: World State partition sum {ws_sum} != {expected_count}. "
        "Partition is not conserved."
    )

    # View B: Agent Claim partition
    ac_sum = sum(
        int(t["agent_claim"]["claimed_complete_verified"]) +
        int(t["agent_claim"]["claimed_complete_unverified"]) +
        int(t["agent_claim"]["escalated"]) +
        int(t["agent_claim"]["failed_terminal"])
        for t in arm_trials
    )
    assert ac_sum == expected_count, (
        f"Arm {arm}: Agent Claim partition sum {ac_sum} != {expected_count}. "
        "Partition is not conserved."
    )


def run_benchmark_v3(
    output_jsonl_path: str = DEFAULT_JSONL_PATH,
) -> Dict[str, Any]:
    """
    Executes the full Ambiguity V3 benchmark matrix:
    6 policy arms × 6 fault modes × 3 operations × 4 seeds = 432 trials.

    Returns a summary dict with:
    - manifest_version, total_trials, jsonl_path
    - per_arm_metrics: {arm: {world_state: {...}, agent_claim: {...}, behavior: {...}}}
    - trials: flat list of all trial records
    """
    trials: List[Dict[str, Any]] = []

    for arm in ALL_POLICY_ARMS:
        for fault in ALL_FAULT_MODES:
            for op in ALL_OPERATIONS:
                for seed in FIXED_SEEDS:
                    case_id = f"v3_{arm}_{fault}_{op}_s{seed}"
                    task_id = f"task_v3_{arm}_{fault}_{op}_{seed}"

                    # Fresh stateful simulator and view per trial (no shared state)
                    sim = StatefulSimulator()
                    agent_view = AgentView(simulator=sim, fault_mode=fault)
                    params = _build_params(seed, op)

                    # Execute policy through AgentView exclusively
                    policy_res: PolicyRunResult = execute_policy_v3(
                        policy_name=arm,
                        agent_view=agent_view,
                        op_name=op,
                        params=params,
                        fault_mode=fault,
                        task_id=task_id,
                    )

                    # Independent oracle evaluation — called AFTER policy execution.
                    # Oracle result is NEVER passed to the policy or verifier.
                    oracle_eval = AmbiguityOracle.evaluate(sim, op_name=op, expected_commits=1)
                    commit_count = oracle_eval["authoritative_commit_count"]

                    # ── View A: World State Partition (mutually exclusive) ──────
                    ws_exactly_one = (commit_count == 1)
                    ws_duplicate   = (commit_count > 1)
                    ws_lost        = (commit_count == 0)
                    ws_no_effect   = False   # All V3 conditions intend exactly 1 commit

                    # Verify exactly one world-state category is true
                    ws_count = int(ws_exactly_one) + int(ws_duplicate) + int(ws_lost) + int(ws_no_effect)
                    assert ws_count == 1, (
                        f"Trial {case_id}: World State partition sum={ws_count} "
                        f"(commits={commit_count}). Must be exactly 1."
                    )

                    # ── View B: Agent Claim Partition (mutually exclusive) ──────
                    claim_verified   = policy_res.claimed_complete_verified
                    claim_unverified = policy_res.claimed_complete_unverified
                    claim_escalated  = policy_res.escalated
                    claim_failed     = policy_res.failed_terminal

                    # Verify exactly one agent-claim category is true
                    ac_count = (
                        int(claim_verified) + int(claim_unverified) +
                        int(claim_escalated) + int(claim_failed)
                    )
                    assert ac_count == 1, (
                        f"Trial {case_id}: Agent Claim partition sum={ac_count}. "
                        f"Policy={arm}, Fault={fault}, Op={op}. Must be exactly 1."
                    )

                    trial_record = {
                        "manifest_version": MANIFEST_VERSION,
                        "case_id": case_id,
                        "seed": seed,
                        "policy_arm": arm,
                        "fault_mode": fault,
                        "observability_class": FAULT_TO_OBSERVABILITY_CLASS[fault],
                        "operation": op,
                        "agent_status": policy_res.agent_status,
                        # World State Partition (View A — authoritative oracle)
                        "world_state": {
                            "commit_count": commit_count,
                            "exactly_one_effect": ws_exactly_one,
                            "duplicate_effect": ws_duplicate,
                            "lost_effect": ws_lost,
                            "no_effect_expected": ws_no_effect,
                        },
                        # Agent Claim Partition (View B — policy declaration)
                        "agent_claim": {
                            "claimed_complete_verified": claim_verified,
                            "claimed_complete_unverified": claim_unverified,
                            "escalated": claim_escalated,
                            "failed_terminal": claim_failed,
                        },
                        # Behavior metadata
                        "behavior": {
                            "retried": policy_res.retried,
                            "observed_existing_state": policy_res.observed_existing_state,
                            "escalation_reason": policy_res.escalation_reason,
                        },
                    }
                    trials.append(trial_record)

    # Assert partition conservation per arm before writing output
    for arm in ALL_POLICY_ARMS:
        _assert_partition_conservation(trials, arm)

    # Assert total trial count matches manifest
    assert len(trials) == TOTAL_BENCHMARK_EXECUTIONS, (
        f"Total trial count {len(trials)} != manifest expected {TOTAL_BENCHMARK_EXECUTIONS}."
    )

    # Write deterministic JSONL (normalized, sort_keys for reproducibility)
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t, sort_keys=True) + "\n")

    # Compute per-arm summary metrics
    per_arm_metrics: Dict[str, Any] = {}
    for arm in ALL_POLICY_ARMS:
        arm_trials = [t for t in trials if t["policy_arm"] == arm]
        n = len(arm_trials)

        ws = {cat: sum(int(t["world_state"][cat]) for t in arm_trials) for cat in WORLD_STATE_CATEGORIES}
        ac = {cat: sum(int(t["agent_claim"][cat]) for t in arm_trials) for cat in AGENT_CLAIM_CATEGORIES}

        retried_count = sum(int(t["behavior"]["retried"]) for t in arm_trials)
        restraint_count = sum(int(t["behavior"]["observed_existing_state"]) for t in arm_trials)

        per_arm_metrics[arm] = {
            "trial_count": n,
            "world_state": ws,
            "agent_claim": ac,
            "behavior": {
                "retried": retried_count,
                "observed_existing_state": restraint_count,
            },
        }

    return {
        "manifest_version": MANIFEST_VERSION,
        "total_trials": len(trials),
        "jsonl_path": output_jsonl_path,
        "per_arm_metrics": per_arm_metrics,
        "trials": trials,
    }
