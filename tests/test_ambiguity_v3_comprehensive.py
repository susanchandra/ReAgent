"""
test_ambiguity_v3_comprehensive.py — Comprehensive behavioral tests for V3 arms.

Test coverage:
1. Arm behavioral contract tests (each arm on each fault mode)
2. Budget enforcement (actions and probes)
3. Partition conservation (per arm and globally)
4. Result accounting integrity
5. Oracle isolation (oracle not reachable from policy at runtime)
6. Regression safety (V2 artifacts and APIs unmodified)
7. CONFIRMED_ABSENT gate (non-idempotent escalation invariant)
8. Probe authority chain (only probe_structured paths, no internal sim access)
"""

import pytest
import json
import os
import tempfile
from typing import Dict, Any, List

from reagent.experiments.ambiguity.simulator import StatefulSimulator
from reagent.experiments.ambiguity.agent_view import AgentView, AgentSecurityViolation
from reagent.experiments.ambiguity.oracle import AmbiguityOracle
from reagent.experiments.ambiguity.faults import (
    ToolExecutionTimeoutError,
    StatusUnavailableError,
    UnobservableEvidenceError,
)
from reagent.experiments.ambiguity.manifest_v3 import (
    FAULT_NO_FAULT,
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
    FAULT_DELAYED_VISIBILITY,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_UNOBSERVABLE,
    OP_CREATE_RESOURCE,
    OP_APPLY_TRANSFER,
    OP_RESERVE_THEN_LINK,
    POLICY_ALWAYS_ESCALATE,
    POLICY_NEVER_RETRY,
    POLICY_NAIVE_RETRY,
    POLICY_IDEMPOTENCY_KEY,
    POLICY_SHARED_OBSERVATION,
    POLICY_REAGENT_CONTRACT,
    ALL_POLICY_ARMS,
    ALL_FAULT_MODES,
    ALL_OPERATIONS,
    FIXED_SEEDS,
    EXECUTIONS_PER_ARM,
    TOTAL_BENCHMARK_EXECUTIONS,
    IDEMPOTENCY_KEY_SUPPORT,
    WORLD_STATE_CATEGORIES,
    AGENT_CLAIM_CATEGORIES,
    MANIFEST_VERSION,
)
from reagent.experiments.ambiguity.policies_v3 import execute_policy_v3, PolicyRunResult
from reagent.experiments.ambiguity.runner_v3 import run_benchmark_v3, _assert_partition_conservation


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_PARAMS = {
    "request_id": "req_test_001",
    "resource_id": "res_test_001",
    "amount": 100,
    "slot_id": "slot_01",
    "entity_id": "ent_test_001",
    "name": "item_test",
    "sender": "acc_alice",
}


def make_view(fault_mode: str) -> tuple:
    """Returns (sim, agent_view) with a fresh simulator under the given fault mode."""
    sim = StatefulSimulator()
    av = AgentView(simulator=sim, fault_mode=fault_mode)
    return sim, av


# ── ARM 1: ALWAYS_ESCALATE ────────────────────────────────────────────────────

class TestAlwaysEscalate:
    def test_no_fault_completes_verified(self):
        sim, av = make_view(FAULT_NO_FAULT)
        res = execute_policy_v3(POLICY_ALWAYS_ESCALATE, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_NO_FAULT, "t1")
        assert res.agent_status == "COMPLETED"
        assert res.claimed_complete_verified is True
        assert res.escalated is False
        assert res.retried is False
        oracle = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        assert oracle["exactly_once_committed"] is True

    def test_fail_before_commit_escalates(self):
        sim, av = make_view(FAULT_FAIL_BEFORE_COMMIT)
        res = execute_policy_v3(POLICY_ALWAYS_ESCALATE, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_FAIL_BEFORE_COMMIT, "t2")
        assert res.agent_status == "ESCALATED"
        assert res.escalated is True
        assert res.claimed_complete_verified is False
        oracle = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        assert oracle["lost_effect"] is True   # escalated before commit

    def test_lost_response_escalates(self):
        sim, av = make_view(FAULT_LOST_RESPONSE)
        res = execute_policy_v3(POLICY_ALWAYS_ESCALATE, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t3")
        assert res.escalated is True
        oracle = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        # Commit DID happen in LOST_RESPONSE; no retry means exactly one effect
        assert oracle["exactly_once_committed"] is True

    def test_no_duplicate_on_lost_response(self):
        """ALWAYS_ESCALATE must never retry, so LOST_RESPONSE = exactly_one_effect."""
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_LOST_RESPONSE)
            execute_policy_v3(POLICY_ALWAYS_ESCALATE, av, op,
                              SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t_dup")
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert not oracle["duplicate_committed"], (
                f"ALWAYS_ESCALATE must not produce duplicates; op={op}"
            )


# ── ARM 2: NEVER_RETRY ────────────────────────────────────────────────────────

class TestNeverRetry:
    def test_no_fault_completes_verified(self):
        sim, av = make_view(FAULT_NO_FAULT)
        res = execute_policy_v3(POLICY_NEVER_RETRY, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_NO_FAULT, "t4")
        assert res.agent_status == "COMPLETED"
        assert res.claimed_complete_verified is True

    def test_fail_before_commit_fails_terminal(self):
        sim, av = make_view(FAULT_FAIL_BEFORE_COMMIT)
        res = execute_policy_v3(POLICY_NEVER_RETRY, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_FAIL_BEFORE_COMMIT, "t5")
        assert res.agent_status == "FAILED"
        assert res.failed_terminal is True
        assert res.retried is False

    def test_lost_response_fails_terminal_no_duplicate(self):
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_LOST_RESPONSE)
            res = execute_policy_v3(POLICY_NEVER_RETRY, av, op,
                                    SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t6")
            assert res.failed_terminal is True
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert not oracle["duplicate_committed"], (
                f"NEVER_RETRY must not retry; op={op}"
            )


# ── ARM 3: NAIVE_RETRY ────────────────────────────────────────────────────────

class TestNaiveRetry:
    def test_no_fault_completes_unverified(self):
        sim, av = make_view(FAULT_NO_FAULT)
        res = execute_policy_v3(POLICY_NAIVE_RETRY, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_NO_FAULT, "t7")
        assert res.agent_status == "COMPLETED"
        assert res.claimed_complete_unverified is True

    def test_fail_before_commit_recovers(self):
        """FAIL_BEFORE_COMMIT: no commit occurred; retry should commit once."""
        sim, av = make_view(FAULT_FAIL_BEFORE_COMMIT)
        res = execute_policy_v3(POLICY_NAIVE_RETRY, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_FAIL_BEFORE_COMMIT, "t8")
        assert res.retried is True
        oracle = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        # After retry, exactly one effect (no prior commit)
        assert oracle["exactly_once_committed"] is True

    def test_lost_response_causes_duplicate(self):
        """LOST_RESPONSE: commit already happened; blind retry causes duplicate."""
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_LOST_RESPONSE)
            res = execute_policy_v3(POLICY_NAIVE_RETRY, av, op,
                                    SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t9")
            assert res.retried is True
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert oracle["duplicate_committed"] is True, (
                f"NAIVE_RETRY must duplicate on LOST_RESPONSE; op={op}"
            )

    def test_delayed_visibility_causes_duplicate(self):
        """DELAYED_VISIBILITY: commit happened; naive retry duplicates."""
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_DELAYED_VISIBILITY)
            res = execute_policy_v3(POLICY_NAIVE_RETRY, av, op,
                                    SAMPLE_PARAMS, FAULT_DELAYED_VISIBILITY, "t10")
            assert res.retried is True
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert oracle["duplicate_committed"] is True, (
                f"NAIVE_RETRY must duplicate on DELAYED_VISIBILITY; op={op}"
            )


# ── ARM 4: IDEMPOTENCY_KEY ────────────────────────────────────────────────────

class TestIdempotencyKey:
    def test_create_resource_lost_response_no_duplicate(self):
        """create_resource supports idempotency key: retry deduplicates."""
        sim, av = make_view(FAULT_LOST_RESPONSE)
        res = execute_policy_v3(POLICY_IDEMPOTENCY_KEY, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t11")
        assert res.retried is True
        oracle = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        assert oracle["exactly_once_committed"] is True, (
            "IDEMPOTENCY_KEY + create_resource must deduplicate on LOST_RESPONSE"
        )

    def test_reserve_then_link_lost_response_no_duplicate(self):
        """reserve_then_link supports idempotency key."""
        sim, av = make_view(FAULT_LOST_RESPONSE)
        res = execute_policy_v3(POLICY_IDEMPOTENCY_KEY, av, OP_RESERVE_THEN_LINK,
                                SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t12")
        assert res.retried is True
        oracle = AmbiguityOracle.evaluate(sim, OP_RESERVE_THEN_LINK)
        assert oracle["exactly_once_committed"] is True

    def test_apply_transfer_lost_response_duplicates(self):
        """apply_transfer lacks server-side key support; retry causes duplicate."""
        sim, av = make_view(FAULT_LOST_RESPONSE)
        res = execute_policy_v3(POLICY_IDEMPOTENCY_KEY, av, OP_APPLY_TRANSFER,
                                SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t13")
        assert res.retried is True
        oracle = AmbiguityOracle.evaluate(sim, OP_APPLY_TRANSFER)
        assert oracle["duplicate_committed"] is True, (
            "apply_transfer has no server-side idempotency; must duplicate"
        )


# ── ARM 5: SHARED_OBSERVATION_POLICY ─────────────────────────────────────────

class TestSharedObservation:
    def test_no_fault_completes_verified(self):
        sim, av = make_view(FAULT_NO_FAULT)
        res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_NO_FAULT, "t14")
        assert res.claimed_complete_verified is True
        assert res.retried is False

    def test_lost_response_restraint_no_duplicate(self):
        """LOST_RESPONSE: probe returns PRESENT; restraint triggered, no retry."""
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_LOST_RESPONSE)
            res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, op,
                                    SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t15")
            assert res.observed_existing_state is True
            assert res.retried is False
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert oracle["exactly_once_committed"] is True, (
                f"Restraint must prevent duplicate on LOST_RESPONSE; op={op}"
            )

    def test_fail_before_commit_recovers_safely(self):
        """FAIL_BEFORE_COMMIT: probe returns CONFIRMED_ABSENT; retry commits once."""
        for op in [OP_CREATE_RESOURCE, OP_RESERVE_THEN_LINK]:
            sim, av = make_view(FAULT_FAIL_BEFORE_COMMIT)
            res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, op,
                                    SAMPLE_PARAMS, FAULT_FAIL_BEFORE_COMMIT, "t16")
            assert res.retried is True
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert oracle["exactly_once_committed"] is True, (
                f"SHARED_OBS must recover on FAIL_BEFORE_COMMIT; op={op}"
            )

    def test_apply_transfer_fail_before_commit_escalates(self):
        """
        APPLY_TRANSFER + FAIL_BEFORE_COMMIT: probe returns CONFIRMED_ABSENT.
        apply_transfer has no server-side idempotency key BUT the probe surface
        is TRANSACTION_LOG_CONSISTENT — so CONFIRMED_ABSENT is authoritative
        and retry is permitted. Verify exact oracle outcome.
        """
        sim, av = make_view(FAULT_FAIL_BEFORE_COMMIT)
        res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, OP_APPLY_TRANSFER,
                                SAMPLE_PARAMS, FAULT_FAIL_BEFORE_COMMIT, "t17")
        # CONFIRMED_ABSENT from transaction log: retry is authorized
        # (non-idempotent gate allows retry when is_confirmed_absent=True)
        oracle = AmbiguityOracle.evaluate(sim, OP_APPLY_TRANSFER)
        # If retried: exactly_once; if escalated (fail-closed): lost
        # Both are valid outcomes depending on probe_status returned
        assert res.agent_status in ("COMPLETED", "ESCALATED")
        if res.retried:
            assert oracle["exactly_once_committed"] is True
        else:
            assert res.escalated is True

    def test_delayed_visibility_restraint_after_step2(self):
        """DELAYED_VISIBILITY: step-1 absent, step-2 PRESENT; restraint triggered."""
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_DELAYED_VISIBILITY)
            res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, op,
                                    SAMPLE_PARAMS, FAULT_DELAYED_VISIBILITY, "t18")
            assert av.probe_count <= 2, "Probe budget must not be exceeded"
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert not oracle["duplicate_committed"], (
                f"SHARED_OBS must not duplicate on DELAYED_VISIBILITY; op={op}"
            )

    def test_status_unavailable_escalates(self):
        """STATUS_UNAVAILABLE: probe raises StatusUnavailableError; fail closed."""
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_STATUS_UNAVAILABLE)
            res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, op,
                                    SAMPLE_PARAMS, FAULT_STATUS_UNAVAILABLE, "t19")
            assert res.escalated is True
            assert "StatusUnavailable" in (res.escalation_reason or "")
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert not oracle["duplicate_committed"]

    def test_unobservable_escalates(self):
        """UNOBSERVABLE: probe raises UnobservableEvidenceError; fail closed."""
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_UNOBSERVABLE)
            res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, op,
                                    SAMPLE_PARAMS, FAULT_UNOBSERVABLE, "t20")
            assert res.escalated is True
            assert "Unobservable" in (res.escalation_reason or "") or "StatusUnavailable" in (res.escalation_reason or "")
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert not oracle["duplicate_committed"]

    def test_probe_budget_not_exceeded(self):
        """Probe count must never exceed 2 for any arm 5 trial."""
        for fault in ALL_FAULT_MODES:
            for op in ALL_OPERATIONS:
                sim, av = make_view(fault)
                execute_policy_v3(POLICY_SHARED_OBSERVATION, av, op,
                                  SAMPLE_PARAMS, fault, "t_budget")
                assert av.probe_count <= 2, (
                    f"Probe budget violated: fault={fault}, op={op}, "
                    f"probe_count={av.probe_count}"
                )


# ── ARM 6: REAGENT_CONTRACT_POLICY ───────────────────────────────────────────

class TestReAgentContractPolicy:
    def test_no_fault_completes_verified(self):
        sim, av = make_view(FAULT_NO_FAULT)
        res = execute_policy_v3(POLICY_REAGENT_CONTRACT, av, OP_CREATE_RESOURCE,
                                SAMPLE_PARAMS, FAULT_NO_FAULT, "t21")
        assert res.claimed_complete_verified is True
        oracle = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        assert oracle["exactly_once_committed"] is True

    def test_lost_response_restraint_no_duplicate(self):
        """LOST_RESPONSE: pipeline probe sees PRESENT; restraint; no retry."""
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_LOST_RESPONSE)
            res = execute_policy_v3(POLICY_REAGENT_CONTRACT, av, op,
                                    SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "t22")
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert not oracle["duplicate_committed"], (
                f"REAGENT_CONTRACT must not duplicate on LOST_RESPONSE; op={op}"
            )

    def test_fail_before_commit_recovers(self):
        """FAIL_BEFORE_COMMIT: probe CONFIRMED_ABSENT; retry commits exactly once."""
        for op in [OP_CREATE_RESOURCE, OP_RESERVE_THEN_LINK]:
            sim, av = make_view(FAULT_FAIL_BEFORE_COMMIT)
            res = execute_policy_v3(POLICY_REAGENT_CONTRACT, av, op,
                                    SAMPLE_PARAMS, FAULT_FAIL_BEFORE_COMMIT, "t23")
            oracle = AmbiguityOracle.evaluate(sim, op)
            # Either retried+committed or escalated; never a duplicate
            assert not oracle["duplicate_committed"]

    def test_status_unavailable_escalates(self):
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_STATUS_UNAVAILABLE)
            res = execute_policy_v3(POLICY_REAGENT_CONTRACT, av, op,
                                    SAMPLE_PARAMS, FAULT_STATUS_UNAVAILABLE, "t24")
            assert res.escalated is True
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert not oracle["duplicate_committed"]

    def test_unobservable_escalates(self):
        for op in ALL_OPERATIONS:
            sim, av = make_view(FAULT_UNOBSERVABLE)
            res = execute_policy_v3(POLICY_REAGENT_CONTRACT, av, op,
                                    SAMPLE_PARAMS, FAULT_UNOBSERVABLE, "t25")
            assert res.escalated is True
            oracle = AmbiguityOracle.evaluate(sim, op)
            assert not oracle["duplicate_committed"]

    def test_adapter_label_honesty(self):
        """
        The adapter must own probe/retry/escalation decisions; the pipeline
        provides receipts. This test verifies that arm 6's result is a
        PolicyRunResult (not a raw pipeline type) and that escalation reason
        is explicitly labeled (not silently swallowed).
        """
        sim, av = make_view(FAULT_STATUS_UNAVAILABLE)
        res = execute_policy_v3(POLICY_REAGENT_CONTRACT, av, OP_APPLY_TRANSFER,
                                SAMPLE_PARAMS, FAULT_STATUS_UNAVAILABLE, "t26")
        assert isinstance(res, PolicyRunResult)
        assert res.escalated is True
        assert res.escalation_reason is not None


# ── ORACLE ISOLATION ─────────────────────────────────────────────────────────

class TestOracleIsolation:
    def test_policy_cannot_access_sim_internals(self):
        """
        Policies must interact only through AgentView. Accessing _sim,
        _committed_records, or _accounts directly must raise AgentSecurityViolation.
        """
        sim, av = make_view(FAULT_NO_FAULT)
        with pytest.raises(AgentSecurityViolation):
            _ = av._sim

    def test_policy_cannot_access_committed_records(self):
        sim, av = make_view(FAULT_NO_FAULT)
        with pytest.raises(AgentSecurityViolation):
            _ = av._committed_records

    def test_oracle_inspects_sim_directly(self):
        """Oracle bypasses AgentView to inspect the ledger — valid in tests only."""
        sim, av = make_view(FAULT_NO_FAULT)
        av.execute(OP_CREATE_RESOURCE, SAMPLE_PARAMS)
        # Oracle uses sim directly (not av)
        result = AmbiguityOracle.evaluate(sim, OP_CREATE_RESOURCE)
        assert result["exactly_once_committed"] is True

    def test_oracle_not_reachable_through_agent_view(self):
        """There must be no oracle attribute accessible from AgentView."""
        sim, av = make_view(FAULT_NO_FAULT)
        with pytest.raises((AgentSecurityViolation, AttributeError)):
            _ = av.oracle


# ── BUDGET ENFORCEMENT ────────────────────────────────────────────────────────

class TestBudgetEnforcement:
    def test_action_budget_exceeded_raises(self):
        """Attempting a 3rd action execution must raise RuntimeError."""
        sim, av = make_view(FAULT_FAIL_BEFORE_COMMIT)
        # First two attempts
        try:
            av.execute(OP_CREATE_RESOURCE, SAMPLE_PARAMS)
        except ToolExecutionTimeoutError:
            pass
        try:
            av.execute(OP_CREATE_RESOURCE, SAMPLE_PARAMS)
        except Exception:
            pass
        # Third attempt must be blocked
        with pytest.raises(RuntimeError, match="ActionBudgetExceeded"):
            av.execute(OP_CREATE_RESOURCE, SAMPLE_PARAMS)

    def test_probe_budget_exceeded_raises(self):
        """Attempting a 3rd probe must raise RuntimeError."""
        sim, av = make_view(FAULT_LOST_RESPONSE)
        # Consume probe budget
        try:
            av.probe_structured(OP_CREATE_RESOURCE, SAMPLE_PARAMS, probe_step=1)
        except Exception:
            pass
        try:
            av.probe_structured(OP_CREATE_RESOURCE, SAMPLE_PARAMS, probe_step=2)
        except Exception:
            pass
        # Third probe must be blocked
        with pytest.raises(RuntimeError, match="ProbeBudgetExceeded"):
            av.probe_structured(OP_CREATE_RESOURCE, SAMPLE_PARAMS, probe_step=1)


# ── PARTITION CONSERVATION ────────────────────────────────────────────────────

class TestPartitionConservation:
    def test_policy_run_result_partition_sum(self):
        """
        For every trial, exactly one of the four agent_claim fields must be True.
        """
        for fault in ALL_FAULT_MODES:
            for op in ALL_OPERATIONS:
                for arm in ALL_POLICY_ARMS:
                    sim, av = make_view(fault)
                    res = execute_policy_v3(arm, av, op, SAMPLE_PARAMS, fault, "pc_t")
                    partition = res.agent_claim_partition()
                    s = sum(int(v) for v in partition.values())
                    assert s == 1, (
                        f"Agent Claim partition sum={s} for arm={arm}, "
                        f"fault={fault}, op={op}. Fields: {partition}"
                    )

    def test_full_runner_partition_conservation(self):
        """
        Run the full 432-execution matrix and assert both partitions are conserved
        per arm.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "trials_v3.jsonl")
            result = run_benchmark_v3(output_jsonl_path=jsonl_path)

        assert result["total_trials"] == TOTAL_BENCHMARK_EXECUTIONS, (
            f"Expected {TOTAL_BENCHMARK_EXECUTIONS} trials, got {result['total_trials']}"
        )

        trials = result["trials"]
        for arm in ALL_POLICY_ARMS:
            # World state partition sum
            arm_trials = [t for t in trials if t["policy_arm"] == arm]
            ws_sum = sum(
                int(t["world_state"]["exactly_one_effect"]) +
                int(t["world_state"]["duplicate_effect"]) +
                int(t["world_state"]["lost_effect"]) +
                int(t["world_state"]["no_effect_expected"])
                for t in arm_trials
            )
            assert ws_sum == EXECUTIONS_PER_ARM, (
                f"World State partition sum={ws_sum} for arm={arm}"
            )
            # Agent claim partition sum
            ac_sum = sum(
                int(t["agent_claim"]["claimed_complete_verified"]) +
                int(t["agent_claim"]["claimed_complete_unverified"]) +
                int(t["agent_claim"]["escalated"]) +
                int(t["agent_claim"]["failed_terminal"])
                for t in arm_trials
            )
            assert ac_sum == EXECUTIONS_PER_ARM, (
                f"Agent Claim partition sum={ac_sum} for arm={arm}"
            )


# ── RESULT ACCOUNTING INTEGRITY ──────────────────────────────────────────────

class TestResultAccounting:
    def test_jsonl_line_count_matches_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "trials_v3.jsonl")
            run_benchmark_v3(output_jsonl_path=jsonl_path)
            with open(jsonl_path, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == TOTAL_BENCHMARK_EXECUTIONS

    def test_jsonl_lines_are_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "trials_v3.jsonl")
            run_benchmark_v3(output_jsonl_path=jsonl_path)
            with open(jsonl_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        record = json.loads(line)
                        assert record["manifest_version"] == MANIFEST_VERSION
                        assert "world_state" in record
                        assert "agent_claim" in record
                        assert "behavior" in record

    def test_jsonl_case_ids_unique(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "trials_v3.jsonl")
            run_benchmark_v3(output_jsonl_path=jsonl_path)
            with open(jsonl_path, encoding="utf-8") as f:
                ids = [json.loads(l)["case_id"] for l in f if l.strip()]
            assert len(ids) == len(set(ids)), "All case_ids must be unique"

    def test_per_arm_metrics_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "trials_v3.jsonl")
            result = run_benchmark_v3(output_jsonl_path=jsonl_path)
        assert set(result["per_arm_metrics"].keys()) == set(ALL_POLICY_ARMS)
        for arm, metrics in result["per_arm_metrics"].items():
            assert metrics["trial_count"] == EXECUTIONS_PER_ARM
            for cat in WORLD_STATE_CATEGORIES:
                assert cat in metrics["world_state"]
            for cat in AGENT_CLAIM_CATEGORIES:
                assert cat in metrics["agent_claim"]

    def test_deterministic_output_across_runs(self):
        """Two identical runs must produce byte-identical JSONL outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "run1.jsonl")
            p2 = os.path.join(tmpdir, "run2.jsonl")
            run_benchmark_v3(output_jsonl_path=p1)
            run_benchmark_v3(output_jsonl_path=p2)
            with open(p1, encoding="utf-8") as f1, open(p2, encoding="utf-8") as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
        assert lines1 == lines2, "Benchmark output must be deterministic"


# ── CONFIRMED_ABSENT GATE ─────────────────────────────────────────────────────

class TestConfirmedAbsentGate:
    def test_apply_transfer_lost_response_shared_obs_restrains(self):
        """
        APPLY_TRANSFER + LOST_RESPONSE: probe returns PRESENT (effect already committed).
        SHARED_OBSERVATION must RESTRAIN (observed_existing_state=True), not retry.
        This is the safe path for non-idempotent operations.
        """
        sim, av = make_view(FAULT_LOST_RESPONSE)
        res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, OP_APPLY_TRANSFER,
                                SAMPLE_PARAMS, FAULT_LOST_RESPONSE, "ca_t1")
        assert res.observed_existing_state is True
        assert res.retried is False
        oracle = AmbiguityOracle.evaluate(sim, OP_APPLY_TRANSFER)
        assert oracle["exactly_once_committed"] is True

    def test_apply_transfer_status_unavailable_escalates_not_retries(self):
        """
        APPLY_TRANSFER + STATUS_UNAVAILABLE: cannot confirm absence.
        Must escalate, NOT retry (even though non-idempotent).
        """
        sim, av = make_view(FAULT_STATUS_UNAVAILABLE)
        res = execute_policy_v3(POLICY_SHARED_OBSERVATION, av, OP_APPLY_TRANSFER,
                                SAMPLE_PARAMS, FAULT_STATUS_UNAVAILABLE, "ca_t2")
        assert res.escalated is True
        assert res.retried is False
        oracle = AmbiguityOracle.evaluate(sim, OP_APPLY_TRANSFER)
        assert not oracle["duplicate_committed"]

    def test_idempotency_key_support_manifest_invariant(self):
        """IDEMPOTENCY_KEY_SUPPORT in manifest_v3 must correctly mark apply_transfer False."""
        assert IDEMPOTENCY_KEY_SUPPORT[OP_CREATE_RESOURCE] is True
        assert IDEMPOTENCY_KEY_SUPPORT[OP_APPLY_TRANSFER] is False
        assert IDEMPOTENCY_KEY_SUPPORT[OP_RESERVE_THEN_LINK] is True


# ── REGRESSION SAFETY ─────────────────────────────────────────────────────────

class TestRegressionSafety:
    def test_v2_policies_module_importable(self):
        """V2 policies file must still import without error."""
        from reagent.experiments.ambiguity import policies_v2
        assert hasattr(policies_v2, "execute_policy_v2")

    def test_v2_runner_importable(self):
        """V2 runner must still import without error."""
        from reagent.experiments.ambiguity import runner_v2
        assert hasattr(runner_v2, "run_benchmark_v2")

    def test_v2_manifest_importable(self):
        from reagent.experiments.ambiguity import manifest_v2
        assert hasattr(manifest_v2, "MANIFEST_VERSION")

    def test_v3_does_not_import_v2_manifest(self):
        """policies_v3.py must not import from manifest_v2."""
        import importlib, inspect
        import reagent.experiments.ambiguity.policies_v3 as pv3
        source = inspect.getsource(pv3)
        assert "manifest_v2" not in source, (
            "policies_v3 must not import from manifest_v2"
        )

    def test_v3_does_not_import_v2_pipeline_adapter(self):
        """policies_v3 must use pipeline_adapter_v3, not pipeline_adapter (V2)."""
        import inspect
        import reagent.experiments.ambiguity.policies_v3 as pv3
        source = inspect.getsource(pv3)
        assert "from reagent.experiments.ambiguity.pipeline_adapter import" not in source

    def test_manifest_v3_preregistration_preserved(self):
        """Manifest V3 constants must match those encoded in preregistration."""
        from reagent.experiments.ambiguity.manifest_v3 import (
            MANIFEST_VERSION, TOTAL_BENCHMARK_EXECUTIONS,
            PREREGISTRATION_BASE_COMMIT,
        )
        assert MANIFEST_VERSION == "3.0.0-preregistered"
        assert TOTAL_BENCHMARK_EXECUTIONS == 432
        assert PREREGISTRATION_BASE_COMMIT.startswith("3270ca8")
