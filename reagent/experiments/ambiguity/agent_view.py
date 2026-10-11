"""
agent_view.py — Restricted facade separating policy execution from simulator internals.

Policies are ONLY permitted to interact with AgentView, which exposes:
- execute(op_name, params, idempotency_key=None)
- probe(op_name, params, step=1)

Access to private simulator attributes, commit records, balances, or oracles
is strictly blocked and tested.
"""

from typing import Dict, Any, Optional
from reagent.experiments.ambiguity.simulator import StatefulSimulator


class AgentSecurityViolation(Exception):
    """Raised if an agent or policy attempts to bypass AgentView facade."""
    pass


class AgentView:
    def __init__(self, simulator: StatefulSimulator, fault_mode: Optional[str] = None):
        # Simulator reference kept private
        object.__setattr__(self, "_sim", simulator)
        object.__setattr__(self, "_fault_mode", fault_mode)
        object.__setattr__(self, "_attempt", 1)
        object.__setattr__(self, "_probe_count", 0)

    @property
    def attempt_count(self) -> int:
        return object.__getattribute__(self, "_attempt")

    @property
    def probe_count(self) -> int:
        return object.__getattribute__(self, "_probe_count")

    def execute(self, op_name: str, params: Dict[str, Any], idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Allowed action execution. Strict enforcement: maximum 2 action attempts."""
        sim = object.__getattribute__(self, "_sim")
        fault = object.__getattribute__(self, "_fault_mode")
        attempt = object.__getattribute__(self, "_attempt")

        if attempt > 2:
            raise RuntimeError(f"ActionBudgetExceeded: attempted {attempt} executions (max allowed is 2).")

        try:
            res = sim.execute_operation(
                op_name=op_name,
                params=params,
                fault_mode=fault,
                step=attempt,
                idempotency_key=idempotency_key,
            )
            return res
        finally:
            object.__setattr__(self, "_attempt", attempt + 1)

    def probe(self, op_name: str, params: Dict[str, Any], probe_step: int = 1) -> bool:
        """
        Legacy environment observation probe (returns boolean).
        Maintains backward compatibility with V2.
        """
        p_count = object.__getattribute__(self, "_probe_count")
        object.__setattr__(self, "_probe_count", p_count + 1)
        sim = object.__getattribute__(self, "_sim")
        fault = object.__getattribute__(self, "_fault_mode")
        from reagent.experiments.ambiguity.manifest_v2 import (
            OP_CREATE_RESOURCE,
            OP_APPLY_TRANSFER,
            OP_RESERVE_THEN_LINK,
        )
        if op_name == OP_CREATE_RESOURCE:
            res_id = params.get("resource_id", "res_1")
            return sim.probe_resource_exists(res_id, fault_mode=fault, probe_step=probe_step)
        elif op_name == OP_APPLY_TRANSFER:
            sender = params.get("sender", "acc_alice")
            bal = sim.probe_account_balance(sender, fault_mode=fault, probe_step=probe_step)
            return bal < 1000
        elif op_name == OP_RESERVE_THEN_LINK:
            slot_id = params.get("slot_id", "slot_01")
            ent_id = params.get("entity_id", "ent_01")
            return sim.probe_slot_link(slot_id, fault_mode=fault, probe_step=probe_step) == ent_id
        return False

    def probe_structured(self, op_name: str, params: Dict[str, Any], probe_step: int = 1) -> Dict[str, Any]:
        """
        Structured observation probe for V3 returning explicit status and authority metadata.
        Strict enforcement: maximum 2 probe calls allowed.
        """
        p_count = object.__getattribute__(self, "_probe_count")
        if p_count >= 2:
            raise RuntimeError(f"ProbeBudgetExceeded: attempted {p_count + 1} probe calls (max allowed is 2).")
        object.__setattr__(self, "_probe_count", p_count + 1)

        sim = object.__getattribute__(self, "_sim")
        fault = object.__getattribute__(self, "_fault_mode")
        return sim.probe_structured(op_name, params, fault_mode=fault, probe_step=probe_step)

    def __getattribute__(self, name: str):
        if name in ("_sim", "_committed_records", "_accounts", "_slots", "_idempotency_store") or (name.startswith("_") and name not in ("_attempt", "_probe_count", "_fault_mode", "__class__", "__dict__")):
            raise AgentSecurityViolation(f"Unauthorized access to simulator internal: {name}")
        return super().__getattribute__(name)

    def __getattr__(self, name: str):
        if name.startswith("_") or "record" in name or "commit" in name or "account" in name:
            raise AgentSecurityViolation(f"Unauthorized access to simulator internal: {name}")
        raise AttributeError(f"AgentView has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        raise AgentSecurityViolation("AgentView is read-only and immutable for policies.")
