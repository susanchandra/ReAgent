"""
policies.py — The 7 offline experimental policy arms for DBOPS V1.

Every arm operates within identical bounds:
- MAX_ACTION_ATTEMPTS = 2 (1 initial execution + at most 1 retry)
- MAX_PROBE_OBSERVATIONS = 2 (at most 2 probe reads)

Policy Decision types:
- VERIFIED: Policy claims the operation succeeded.
- ESCALATED: Policy halted or safely escalated without claiming verified success.
- RETRIED: Policy issued a retry write.

The 7 Offline Arms (Manifest Section C):
1. GOLD_CONTRACT:
   Uses the validated, grounded GoldContract. Probes committed state.
   Detects WRONG_CONTENT and PARTIAL_COMMIT (fails verification -> ESCALATE).
   Detects CONFIRMED_ABSENT on pure commit failure -> safe retry within budget.
   Resolves DELAYED_VISIBILITY via second probe.
   Escalates on STATUS_UNAVAILABLE and UNOBSERVABLE.
2. DERIVED_MODEL_FIXTURE:
   Synthetic fixture representing a derived contract passed through ContractValidator.
   Because it is valid and grounded against gold postconditions, behaves identically
   to GOLD_CONTRACT in contract evaluation.
3. HAND_AUTHORED_BASELINE:
   Conventional baseline checking primary table presence and success status.
   Lacks secondary table inspection and fine-grained field value cross-examination.
   False-verifies on WRONG_CONTENT (primary row exists) and PARTIAL_COMMIT (primary row exists).
4. SHARED_OBSERVATION:
   Independent probe enabled with identical budgets. Inspects primary presence.
   Does not cross-examine fine-grained field arguments across secondary tables.
   Distinguishes pure presence from absence, but cannot detect WRONG_CONTENT or PARTIAL_COMMIT.
5. TOOL_CLAIM_ONLY:
   Blind baseline relying solely on tool execution result (error == None).
   Claims VERIFIED whenever error is None (even under WRONG_CONTENT, or undetected partials if tool claimed success).
   Escalates or gives up on error.
6. NAIVE_RETRY:
   Blind retry upon any error without probing or restraint.
   Retries immediately on error. Causes duplicate writes on COMMIT_THEN_TIMEOUT.
7. ALWAYS_ESCALATE:
   Safety anchor. Escalates on any error without retrying or probing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.executor import execute_task, ExecutionResult
from reagent.experiments.dbops.oracle import DbopsOracle, ProbeResult
from reagent.experiments.dbops.validator import ContractValidator, GoldContractEvaluator
from reagent.experiments.dbops.gold_contracts import get_gold_contract
from reagent.experiments.dbops.types import VerdictType


class PolicyAction(str, Enum):
    VERIFIED = "VERIFIED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"


@dataclass
class PolicyExecutionRecord:
    task_id: str
    fault_mode: str
    arm: str
    final_claim: PolicyAction
    action_attempts: int
    probe_attempts: int
    retried: bool
    explanation: str
    execution_results: list[ExecutionResult] = field(default_factory=list)


def run_policy_trial(
    store: DbopsStore,
    task_id: str,
    fault_mode: str,
    arm: str,
) -> PolicyExecutionRecord:
    """
    Execute a single trial of (task_id, fault_mode, arm) against store.
    Tracks action and probe attempts strictly within budgets.
    """
    action_attempts = 1
    probe_attempts = 0
    retried = False
    exec_history: list[ExecutionResult] = []

    # Initial attempt
    exec_res_1 = execute_task(store, task_id, fault_mode)
    exec_history.append(exec_res_1)

    oracle = DbopsOracle(store, exec_res_1)

    # ─────────────────────────────────────────────────────────────────────────
    # ARM 7: ALWAYS_ESCALATE
    # Unconditional safety anchor per manifest C.7: every trial terminates
    # with ESCALATED and no VERIFIED claims, regardless of tool outcome.
    # ─────────────────────────────────────────────────────────────────────────
    if arm == "ALWAYS_ESCALATE":
        return PolicyExecutionRecord(
            task_id=task_id, fault_mode=fault_mode, arm=arm,
            final_claim=PolicyAction.ESCALATED,
            action_attempts=action_attempts, probe_attempts=probe_attempts,
            retried=retried, explanation="Unconditional safety anchor: always escalates.",
            execution_results=exec_history,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ARM 5: TOOL_CLAIM_ONLY
    # Blind baseline relying solely on tool execution ActionResult.success.
    # ─────────────────────────────────────────────────────────────────────────
    if arm == "TOOL_CLAIM_ONLY":
        if exec_res_1.agent_visible_error is None:
            # Blindly trusts tool return
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.VERIFIED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="Tool claim accepted without verification.",
                execution_results=exec_history,
            )
        else:
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.ESCALATED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="Tool returned error; no recovery attempted.",
                execution_results=exec_history,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # ARM 6: NAIVE_RETRY
    # Blind retry upon any ambiguous outcome without probing or restraint.
    # ─────────────────────────────────────────────────────────────────────────
    if arm == "NAIVE_RETRY":
        if exec_res_1.agent_visible_error is None:
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.VERIFIED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="Initial attempt succeeded.",
                execution_results=exec_history,
            )
        else:
            # Blindly retry without probing
            retried = True
            action_attempts += 1
            # Second attempt runs under NO_FAULT (transient error cleared)
            exec_res_2 = execute_task(store, task_id, "NO_FAULT")
            exec_history.append(exec_res_2)
            if exec_res_2.agent_visible_error is None:
                claim = PolicyAction.VERIFIED
                expl = "Retried blindly; second attempt succeeded."
            else:
                claim = PolicyAction.ESCALATED
                expl = "Retried blindly; second attempt failed."
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=claim, action_attempts=action_attempts,
                probe_attempts=probe_attempts, retried=retried, explanation=expl,
                execution_results=exec_history,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # ARM 3: HAND_AUTHORED_BASELINE
    # Conventional baseline: checks tool status + single primary row read.
    # Does NOT perform multi-probe delayed-visibility recovery or multi-table checks.
    # ─────────────────────────────────────────────────────────────────────────
    if arm == "HAND_AUTHORED_BASELINE":
        # Always probes primary state (single read) to verify tool status
        probe_attempts += 1
        p1 = oracle.probe(call_number=1)

        if p1.status == "PRESENT":
            # Primary row exists: claims VERIFIED (misses WRONG_CONTENT / PARTIAL_COMMIT)
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.VERIFIED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="Hand-authored baseline: primary row confirmed present.",
                execution_results=exec_history,
            )
        elif p1.status == "DELAYED":
            # Does NOT perform second probe; escalates on delay
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.ESCALATED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="Hand-authored baseline: single read was delayed, cannot recover.",
                execution_results=exec_history,
            )
        elif p1.status == "ABSENT":
            # If absent, attempts simple retry once if tool reported error
            retried = True
            action_attempts += 1
            exec_res_2 = execute_task(store, task_id, "NO_FAULT")
            exec_history.append(exec_res_2)
            claim = PolicyAction.VERIFIED if exec_res_2.agent_visible_error is None else PolicyAction.ESCALATED
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=claim, action_attempts=action_attempts,
                probe_attempts=probe_attempts, retried=retried,
                explanation="Hand-authored baseline: retried after absent read.",
                execution_results=exec_history,
            )
        else:
            # UNAVAILABLE or UNOBSERVABLE
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.ESCALATED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation=f"Hand-authored baseline: read returned {p1.status}.",
                execution_results=exec_history,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # ARM 4: SHARED_OBSERVATION
    # Probe-enabled baseline with full probe budget (up to 2 probes) and
    # delayed-visibility recovery. Lacks field-level multi-table contract checking.
    # ─────────────────────────────────────────────────────────────────────────
    if arm == "SHARED_OBSERVATION":
        probe_attempts += 1
        p1 = oracle.probe(call_number=1)

        if p1.status == "DELAYED":
            # Recovers using 2nd probe budget
            probe_attempts += 1
            p2 = oracle.probe(call_number=2)
            if p2.status == "PRESENT":
                return PolicyExecutionRecord(
                    task_id=task_id, fault_mode=fault_mode, arm=arm,
                    final_claim=PolicyAction.VERIFIED,
                    action_attempts=action_attempts, probe_attempts=probe_attempts,
                    retried=retried, explanation="Shared observation: delayed visibility resolved on second probe.",
                    execution_results=exec_history,
                )
            else:
                return PolicyExecutionRecord(
                    task_id=task_id, fault_mode=fault_mode, arm=arm,
                    final_claim=PolicyAction.ESCALATED,
                    action_attempts=action_attempts, probe_attempts=probe_attempts,
                    retried=retried, explanation="Shared observation: delayed probe unresolved after second probe.",
                    execution_results=exec_history,
                )

        if p1.status == "PRESENT":
            # Primary row present: claims VERIFIED (misses WRONG_CONTENT and PARTIAL_COMMIT)
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.VERIFIED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="Shared observation: primary row confirmed present via probe.",
                execution_results=exec_history,
            )

        if p1.status == "ABSENT":
            # Safe retry when confirmed absent
            retried = True
            action_attempts += 1
            exec_res_2 = execute_task(store, task_id, "NO_FAULT")
            exec_history.append(exec_res_2)
            # Re-probe after retry
            probe_attempts += 1
            oracle_2 = DbopsOracle(store, exec_res_2)
            p_retry = oracle_2.probe(call_number=1)
            claim = PolicyAction.VERIFIED if p_retry.status == "PRESENT" else PolicyAction.ESCALATED
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=claim, action_attempts=action_attempts,
                probe_attempts=probe_attempts, retried=retried,
                explanation="Shared observation: row absent; retried and verified via probe.",
                execution_results=exec_history,
            )

        # UNAVAILABLE or UNOBSERVABLE -> Escalate
        return PolicyExecutionRecord(
            task_id=task_id, fault_mode=fault_mode, arm=arm,
            final_claim=PolicyAction.ESCALATED,
            action_attempts=action_attempts, probe_attempts=probe_attempts,
            retried=retried, explanation=f"Shared observation: probe returned {p1.status}; cannot verify.",
            execution_results=exec_history,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ARMS 1 & 2: GOLD_CONTRACT & DERIVED_MODEL_FIXTURE
    # Full field-level contract evaluation across primary and secondary tables.
    # ─────────────────────────────────────────────────────────────────────────
    if arm in ("GOLD_CONTRACT", "DERIVED_MODEL_FIXTURE"):
        gold = get_gold_contract(task_id)

        # If arm is DERIVED_MODEL_FIXTURE, validate contract grounding first
        if arm == "DERIVED_MODEL_FIXTURE":
            validator = ContractValidator()
            val_res = validator.validate(gold, task_id)
            if not val_res.valid:
                return PolicyExecutionRecord(
                    task_id=task_id, fault_mode=fault_mode, arm=arm,
                    final_claim=PolicyAction.ESCALATED,
                    action_attempts=action_attempts, probe_attempts=probe_attempts,
                    retried=retried, explanation=f"Contract validation rejected: {val_res.rejection_class}",
                    execution_results=exec_history,
                )

        evaluator = GoldContractEvaluator()

        # Handle DELAYED_VISIBILITY protocol
        if fault_mode == "DELAYED_VISIBILITY":
            probe_attempts += 1
            p1 = oracle.probe(call_number=1)
            if p1.status == "DELAYED":
                probe_attempts += 1
                p2 = oracle.probe(call_number=2)
                # Second probe connects to committed state
                obs = oracle.evaluate()
                eval_res = evaluator.evaluate(task_id, obs)
                claim = PolicyAction.VERIFIED if eval_res.verdict == VerdictType.EXPECTED_PRESENT else PolicyAction.ESCALATED
                return PolicyExecutionRecord(
                    task_id=task_id, fault_mode=fault_mode, arm=arm,
                    final_claim=claim, action_attempts=action_attempts,
                    probe_attempts=probe_attempts, retried=retried,
                    explanation="Delayed visibility resolved on second probe; contract verified.",
                    execution_results=exec_history,
                )

        # For ambiguous error or non-clean states: evaluate committed world state
        probe_attempts += 1
        obs = oracle.evaluate()
        eval_res = evaluator.evaluate(task_id, obs)

        if eval_res.verdict == VerdictType.EXPECTED_PRESENT:
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.VERIFIED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="All contract postconditions verified in committed state.",
                execution_results=exec_history,
            )

        if eval_res.verdict == VerdictType.CONFIRMED_ABSENT:
            # Operation did not commit: Safe Retry!
            retried = True
            action_attempts += 1
            exec_res_2 = execute_task(store, task_id, "NO_FAULT")
            exec_history.append(exec_res_2)
            probe_attempts += 1
            oracle_2 = DbopsOracle(store, exec_res_2)
            obs_2 = oracle_2.evaluate()
            eval_res_2 = evaluator.evaluate(task_id, obs_2)
            claim = PolicyAction.VERIFIED if eval_res_2.verdict == VerdictType.EXPECTED_PRESENT else PolicyAction.ESCALATED
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=claim, action_attempts=action_attempts,
                probe_attempts=probe_attempts, retried=retried,
                explanation="Confirmed absent; safely retried and verified postconditions.",
                execution_results=exec_history,
            )

        if eval_res.verdict == VerdictType.PARTIAL:
            # PARTIAL_COMMIT detected! Discrepancy -> Escalate safely, do not retry!
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.ESCALATED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="Partial commit detected across workflow steps; safe escalation.",
                execution_results=exec_history,
            )

        if eval_res.verdict == VerdictType.WRONG_CONTENT:
            # WRONG_CONTENT detected! Discrepancy -> Escalate safely, do not retry!
            return PolicyExecutionRecord(
                task_id=task_id, fault_mode=fault_mode, arm=arm,
                final_claim=PolicyAction.ESCALATED,
                action_attempts=action_attempts, probe_attempts=probe_attempts,
                retried=retried, explanation="Field-level discrepancy detected (WRONG_CONTENT); safe escalation.",
                execution_results=exec_history,
            )

        # UNKNOWN or any other state -> Escalate safely
        return PolicyExecutionRecord(
            task_id=task_id, fault_mode=fault_mode, arm=arm,
            final_claim=PolicyAction.ESCALATED,
            action_attempts=action_attempts, probe_attempts=probe_attempts,
            retried=retried, explanation="Outcome UNKNOWN or observation surface unavailable; safe escalation.",
            execution_results=exec_history,
        )

    raise ValueError(f"Unknown experimental arm: {arm!r}")
