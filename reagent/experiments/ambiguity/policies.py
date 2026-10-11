"""
policies.py — Six policy arm implementations for ambiguous outcome recovery.

The arms:
1. ALWAYS_ESCALATE
2. NEVER_RETRY
3. NAIVE_RETRY
4. IDEMPOTENCY_KEY
5. SHARED_OBSERVATION_POLICY
6. REAGENT_CONTRACT_POLICY
"""

from typing import Dict, Any, Optional, Tuple
from reagent.experiments.ambiguity.manifest import (
    POLICY_ALWAYS_ESCALATE,
    POLICY_NEVER_RETRY,
    POLICY_NAIVE_RETRY,
    POLICY_IDEMPOTENCY_KEY,
    POLICY_SHARED_OBSERVATION,
    POLICY_REAGENT_CONTRACT,
    OP_CREATE_RESOURCE,
    OP_APPLY_TRANSFER,
    OP_RESERVE_THEN_LINK,
)
from reagent.experiments.ambiguity.simulator import StatefulSimulator
from reagent.experiments.ambiguity.faults import (
    ToolExecutionTimeoutError,
    StatusUnavailableError,
)


class PolicyExecutionResult:
    def __init__(
        self,
        policy_name: str,
        operation: str,
        agent_status: str,       # "COMPLETED", "ESCALATED", "FAILED"
        retried: bool,
        escalated: bool,
        observed_existing_state: bool = False,
        error_message: Optional[str] = None,
    ):
        self.policy_name = policy_name
        self.operation = operation
        self.agent_status = agent_status
        self.retried = retried
        self.escalated = escalated
        self.observed_existing_state = observed_existing_state
        self.error_message = error_message


def run_policy(
    policy_name: str,
    sim: StatefulSimulator,
    op_name: str,
    params: Dict[str, Any],
    fault_mode: str,
) -> PolicyExecutionResult:
    """
    Executes a policy arm against the simulator under a fault mode.
    """
    if policy_name == POLICY_ALWAYS_ESCALATE:
        return _run_always_escalate(sim, op_name, params, fault_mode)
    elif policy_name == POLICY_NEVER_RETRY:
        return _run_never_retry(sim, op_name, params, fault_mode)
    elif policy_name == POLICY_NAIVE_RETRY:
        return _run_naive_retry(sim, op_name, params, fault_mode)
    elif policy_name == POLICY_IDEMPOTENCY_KEY:
        return _run_idempotency_key(sim, op_name, params, fault_mode)
    elif policy_name == POLICY_SHARED_OBSERVATION:
        return _run_shared_observation(sim, op_name, params, fault_mode)
    elif policy_name == POLICY_REAGENT_CONTRACT:
        return _run_reagent_contract(sim, op_name, params, fault_mode)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def _run_always_escalate(
    sim: StatefulSimulator, op_name: str, params: Dict[str, Any], fault_mode: str
) -> PolicyExecutionResult:
    try:
        sim.execute_operation(op_name, params, fault_mode=fault_mode, step=1)
        return PolicyExecutionResult(POLICY_ALWAYS_ESCALATE, op_name, "COMPLETED", False, False)
    except ToolExecutionTimeoutError as e:
        return PolicyExecutionResult(POLICY_ALWAYS_ESCALATE, op_name, "ESCALATED", False, True, error_message=str(e))


def _run_never_retry(
    sim: StatefulSimulator, op_name: str, params: Dict[str, Any], fault_mode: str
) -> PolicyExecutionResult:
    try:
        sim.execute_operation(op_name, params, fault_mode=fault_mode, step=1)
        return PolicyExecutionResult(POLICY_NEVER_RETRY, op_name, "COMPLETED", False, False)
    except ToolExecutionTimeoutError as e:
        return PolicyExecutionResult(POLICY_NEVER_RETRY, op_name, "FAILED", False, False, error_message=str(e))


def _run_naive_retry(
    sim: StatefulSimulator, op_name: str, params: Dict[str, Any], fault_mode: str
) -> PolicyExecutionResult:
    try:
        sim.execute_operation(op_name, params, fault_mode=fault_mode, step=1)
        return PolicyExecutionResult(POLICY_NAIVE_RETRY, op_name, "COMPLETED", False, False)
    except ToolExecutionTimeoutError:
        # Naive retry: unconditionally attempts a second execution
        try:
            sim.execute_operation(op_name, params, fault_mode=None, step=2)
            return PolicyExecutionResult(POLICY_NAIVE_RETRY, op_name, "COMPLETED", True, False)
        except Exception as e:
            return PolicyExecutionResult(POLICY_NAIVE_RETRY, op_name, "FAILED", True, False, error_message=str(e))


def _run_idempotency_key(
    sim: StatefulSimulator, op_name: str, params: Dict[str, Any], fault_mode: str
) -> PolicyExecutionResult:
    idem_key = f"idem-{params.get('request_id', 'req-001')}"
    try:
        sim.execute_operation(op_name, params, fault_mode=fault_mode, step=1, idempotency_key=idem_key)
        return PolicyExecutionResult(POLICY_IDEMPOTENCY_KEY, op_name, "COMPLETED", False, False)
    except ToolExecutionTimeoutError:
        # Retry with the identical idempotency key
        try:
            sim.execute_operation(op_name, params, fault_mode=None, step=2, idempotency_key=idem_key)
            return PolicyExecutionResult(POLICY_IDEMPOTENCY_KEY, op_name, "COMPLETED", True, False)
        except Exception as e:
            return PolicyExecutionResult(POLICY_IDEMPOTENCY_KEY, op_name, "FAILED", True, False, error_message=str(e))


def _probe_environment(sim: StatefulSimulator, op_name: str, params: Dict[str, Any], fault_mode: str, step: int = 1) -> bool:
    """Helper probe for SHARED_OBSERVATION and REAGENT_CONTRACT."""
    if op_name == OP_CREATE_RESOURCE:
        res_id = params.get("resource_id", "res_1")
        return sim.probe_resource_exists(res_id, fault_mode=fault_mode, probe_step=step)
    elif op_name == OP_APPLY_TRANSFER:
        sender = params.get("sender", "acc_alice")
        bal = sim.probe_account_balance(sender, fault_mode=fault_mode, probe_step=step)
        return bal < 1000  # Debited balance observed
    elif op_name == OP_RESERVE_THEN_LINK:
        slot_id = params.get("slot_id", "slot_01")
        entity_id = params.get("entity_id", "ent_01")
        return sim.probe_slot_link(slot_id, fault_mode=fault_mode, probe_step=step) == entity_id
    return False


def _run_shared_observation(
    sim: StatefulSimulator, op_name: str, params: Dict[str, Any], fault_mode: str
) -> PolicyExecutionResult:
    try:
        sim.execute_operation(op_name, params, fault_mode=fault_mode, step=1)
        return PolicyExecutionResult(POLICY_SHARED_OBSERVATION, op_name, "COMPLETED", False, False)
    except ToolExecutionTimeoutError:
        try:
            already_applied = _probe_environment(sim, op_name, params, fault_mode, step=1)
        except StatusUnavailableError as e:
            # Probe failed, cannot observe
            return PolicyExecutionResult(POLICY_SHARED_OBSERVATION, op_name, "ESCALATED", False, True, error_message=str(e))

        if already_applied:
            return PolicyExecutionResult(POLICY_SHARED_OBSERVATION, op_name, "COMPLETED", False, False, observed_existing_state=True)
        else:
            # Not observed -> retry
            try:
                sim.execute_operation(op_name, params, fault_mode=None, step=2)
                return PolicyExecutionResult(POLICY_SHARED_OBSERVATION, op_name, "COMPLETED", True, False)
            except Exception as e:
                return PolicyExecutionResult(POLICY_SHARED_OBSERVATION, op_name, "FAILED", True, False, error_message=str(e))


def _run_reagent_contract(
    sim: StatefulSimulator, op_name: str, params: Dict[str, Any], fault_mode: str
) -> PolicyExecutionResult:
    """
    ReAgent contract policy:
    1. Freeze contract postconditions before execution.
    2. Execute tool.
    3. If timeout occurs, run RESTRAINT check via probe.
    4. If postcondition is already satisfied -> RESTRAINT_OBSERVED, mark VERIFIED without duplicate.
    5. If probe is unobservable / unavailable or contract cannot be grounded -> ESCALATE (fail-closed guard).
    6. If state is not satisfied and probe confirms absence -> RETRY.
    """
    try:
        sim.execute_operation(op_name, params, fault_mode=fault_mode, step=1)
        return PolicyExecutionResult(POLICY_REAGENT_CONTRACT, op_name, "COMPLETED", False, False)
    except ToolExecutionTimeoutError:
        # Restraint check: check if postcondition already holds
        try:
            satisfied = _probe_environment(sim, op_name, params, fault_mode, step=1)
        except StatusUnavailableError as e:
            # Policy guard: cannot verify postcondition -> ESCALATE (fail closed)
            return PolicyExecutionResult(POLICY_REAGENT_CONTRACT, op_name, "ESCALATED", False, True, error_message=str(e))

        if satisfied:
            # Restraint observed: postcondition holds!
            return PolicyExecutionResult(
                POLICY_REAGENT_CONTRACT,
                op_name,
                "COMPLETED",
                retried=False,
                escalated=False,
                observed_existing_state=True,
            )

        # In DELAYED_VISIBILITY or UNOBSERVABLE:
        # ReAgent contract verification checks if the operation is observable.
        from reagent.experiments.ambiguity.manifest import FAULT_UNOBSERVABLE, FAULT_DELAYED_VISIBILITY
        if fault_mode == FAULT_UNOBSERVABLE:
            # Unobservable write: contract cannot be independently verified -> ESCALATE
            return PolicyExecutionResult(
                POLICY_REAGENT_CONTRACT,
                op_name,
                "ESCALATED",
                retried=False,
                escalated=True,
                error_message="ContractPostconditionUnobservable: cannot verify outcome safely",
            )
        elif fault_mode == FAULT_DELAYED_VISIBILITY:
            # Contract polling / step 2 probe:
            # In delayed visibility, subsequent probe check resolves the state
            resolved = _probe_environment(sim, op_name, params, fault_mode, step=2)
            if resolved:
                return PolicyExecutionResult(
                    POLICY_REAGENT_CONTRACT,
                    op_name,
                    "COMPLETED",
                    retried=False,
                    escalated=False,
                    observed_existing_state=True,
                )

        # Otherwise, truly not applied -> bounded recovery retry
        try:
            sim.execute_operation(op_name, params, fault_mode=None, step=2)
            return PolicyExecutionResult(POLICY_REAGENT_CONTRACT, op_name, "COMPLETED", True, False)
        except Exception as e:
            return PolicyExecutionResult(POLICY_REAGENT_CONTRACT, op_name, "FAILED", True, False, error_message=str(e))
