"""
simulator.py — Stateful service simulator with authoritative store.

Simulates three stateful side-effecting operations:
1. create_resource   (creates resource_id with initial metadata)
2. apply_transfer    (debits sender, credits recipient)
3. reserve_then_link (reserves slot, links entity_id to slot)

Authoritative Store:
The simulator maintains an internal commit ledger tracking every committed side effect
independently of whether a response was dropped.
"""

from typing import Dict, Any, List, Optional, Tuple
import copy
from reagent.experiments.ambiguity.manifest import (
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
    FAULT_DELAYED_VISIBILITY,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_UNOBSERVABLE,
    OP_CREATE_RESOURCE,
    OP_APPLY_TRANSFER,
    OP_RESERVE_THEN_LINK,
)
from reagent.experiments.ambiguity.faults import (
    ToolExecutionTimeoutError,
    StatusUnavailableError,
    UnobservableEvidenceError,
    make_standard_timeout_error,
)


class StatefulSimulator:
    def __init__(self):
        # Authoritative commit ledger: operation_name -> list of committed records
        self._committed_records: Dict[str, List[Dict[str, Any]]] = {
            OP_CREATE_RESOURCE: [],
            OP_APPLY_TRANSFER: [],
            OP_RESERVE_THEN_LINK: [],
        }
        # Balances for transfer tracking: account_id -> balance
        self._accounts: Dict[str, int] = {
            "acc_alice": 1000,
            "acc_bob": 500,
        }
        # Resource store: resource_id -> data
        self._resources: Dict[str, Dict[str, Any]] = {}
        # Reservation slots: slot_id -> linked_entity
        self._slots: Dict[str, Optional[str]] = {
            "slot_01": None,
            "slot_02": None,
        }
        # Processed idempotency keys (where supported): key -> response
        self._idempotency_store: Dict[str, Any] = {}

        # Call counters for tracking execution history
        self.call_history: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Simulator Core Operations
    # ─────────────────────────────────────────────────────────────────────────

    def execute_operation(
        self,
        op_name: str,
        params: Dict[str, Any],
        fault_mode: Optional[str] = None,
        step: int = 1,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes an operation subject to a designated fault mode.
        If idempotency_key is present and recognized (supported on create_resource),
        duplicate executions will return cached responses without committing duplicate state.
        """
        self.call_history.append({
            "step": step,
            "op_name": op_name,
            "params": copy.deepcopy(params),
            "fault_mode": fault_mode,
            "idempotency_key": idempotency_key,
        })

        request_id = params.get("request_id", "req-default-001")

        # Check idempotency support (only enabled for create_resource and reserve_then_link)
        if idempotency_key and op_name in (OP_CREATE_RESOURCE, OP_RESERVE_THEN_LINK):
            if idempotency_key in self._idempotency_store:
                return self._idempotency_store[idempotency_key]

        # Fault: FAIL_BEFORE_COMMIT
        # Fails BEFORE writing state to authoritative store
        if fault_mode == FAULT_FAIL_BEFORE_COMMIT and step == 1:
            raise make_standard_timeout_error(op_name, request_id)

        # Commit State based on operation
        result = self._commit(op_name, params)

        if idempotency_key and op_name in (OP_CREATE_RESOURCE, OP_RESERVE_THEN_LINK):
            self._idempotency_store[idempotency_key] = result

        # Fault: LOST_RESPONSE
        # State HAS committed, but transport drops the response and raises the exact same timeout
        if fault_mode == FAULT_LOST_RESPONSE and step == 1:
            raise make_standard_timeout_error(op_name, request_id)

        # Fault: DELAYED_VISIBILITY
        # State committed, but transport signals a timeout initially
        if fault_mode == FAULT_DELAYED_VISIBILITY and step == 1:
            raise make_standard_timeout_error(op_name, request_id)

        # Fault: STATUS_UNAVAILABLE
        # State committed, but downstream probe/status will fail
        if fault_mode == FAULT_STATUS_UNAVAILABLE and step == 1:
            raise make_standard_timeout_error(op_name, request_id)

        # Fault: UNOBSERVABLE
        # State committed, but cannot be observed via probes
        if fault_mode == FAULT_UNOBSERVABLE and step == 1:
            raise make_standard_timeout_error(op_name, request_id)

        return result

    def _commit(self, op_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal atomic commit to authoritative ledger."""
        if op_name == OP_CREATE_RESOURCE:
            res_id = params.get("resource_id", f"res_{len(self._resources) + 1}")
            record = {"resource_id": res_id, "name": params.get("name", "unnamed")}
            self._committed_records[op_name].append(record)
            self._resources[res_id] = record
            return {"status": "SUCCESS", "resource_id": res_id}

        elif op_name == OP_APPLY_TRANSFER:
            amount = params.get("amount", 100)
            sender = params.get("sender", "acc_alice")
            recipient = params.get("recipient", "acc_bob")
            self._accounts[sender] = self._accounts.get(sender, 0) - amount
            self._accounts[recipient] = self._accounts.get(recipient, 0) + amount
            record = {"sender": sender, "recipient": recipient, "amount": amount}
            self._committed_records[op_name].append(record)
            return {"status": "SUCCESS", "transferred": amount, "sender_balance": self._accounts[sender]}

        elif op_name == OP_RESERVE_THEN_LINK:
            slot_id = params.get("slot_id", "slot_01")
            entity_id = params.get("entity_id", "ent_01")
            self._slots[slot_id] = entity_id
            record = {"slot_id": slot_id, "entity_id": entity_id}
            self._committed_records[op_name].append(record)
            return {"status": "SUCCESS", "slot_id": slot_id, "entity_id": entity_id}

        raise ValueError(f"Unknown operation: {op_name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public Environment Observation Probes (Used by agents / policies)
    # ─────────────────────────────────────────────────────────────────────────

    def probe_resource_exists(self, resource_id: str, fault_mode: Optional[str] = None, probe_step: int = 1) -> bool:
        """Probe for create_resource."""
        if fault_mode == FAULT_STATUS_UNAVAILABLE:
            raise StatusUnavailableError("Status probe endpoint down", OP_CREATE_RESOURCE)
        if fault_mode == FAULT_UNOBSERVABLE:
            # External or third-party write with no read-back probe surface
            return False
        if fault_mode == FAULT_DELAYED_VISIBILITY and probe_step == 1:
            # Not yet visible on first probe attempt
            return False
        return resource_id in self._resources

    def probe_account_balance(self, account_id: str, fault_mode: Optional[str] = None, probe_step: int = 1) -> int:
        """Probe for apply_transfer balance."""
        if fault_mode == FAULT_STATUS_UNAVAILABLE:
            raise StatusUnavailableError("Status probe endpoint down", OP_APPLY_TRANSFER)
        if fault_mode == FAULT_UNOBSERVABLE:
            # External third-party ledger
            return 1000  # returns baseline pre-transfer balance
        if fault_mode == FAULT_DELAYED_VISIBILITY and probe_step == 1:
            return 1000  # un-updated read
        return self._accounts.get(account_id, 0)

    def probe_slot_link(self, slot_id: str, fault_mode: Optional[str] = None, probe_step: int = 1) -> Optional[str]:
        """Probe for reserve_then_link."""
        if fault_mode == FAULT_STATUS_UNAVAILABLE:
            raise StatusUnavailableError("Status probe endpoint down", OP_RESERVE_THEN_LINK)
        if fault_mode == FAULT_UNOBSERVABLE:
            return None
        if fault_mode == FAULT_DELAYED_VISIBILITY and probe_step == 1:
            return None
        return self._slots.get(slot_id)

    def probe_structured(
        self,
        op_name: str,
        params: Dict[str, Any],
        fault_mode: Optional[str] = None,
        probe_step: int = 1,
    ) -> Dict[str, Any]:
        """
        Public environment observation returning structured status and consistency metadata:
        - PRESENT: Effect is verified to have committed upstream.
        - CONFIRMED_ABSENT: Effect is authoritatively confirmed absent on consistent surface.
        - DELAYED: Visibility is lagging (requires wait/re-probe).
        - UNAVAILABLE: Status endpoint unreachable (StatusUnavailableError).
        - UNOBSERVABLE: Third-party write without probe surface (UnobservableEvidenceError).
        """
        if fault_mode == FAULT_STATUS_UNAVAILABLE:
            raise StatusUnavailableError("Status probe endpoint down", op_name)

        if fault_mode == FAULT_UNOBSERVABLE:
            raise UnobservableEvidenceError("Operation target has no read-back observation surface", op_name)

        if op_name == OP_CREATE_RESOURCE:
            res_id = params.get("resource_id", "res_1")
            if fault_mode == FAULT_DELAYED_VISIBILITY and probe_step == 1:
                return {"status": "DELAYED", "authority": "AUTHORITATIVE_CONSISTENT", "exists": False}
            exists = res_id in self._resources
            return {
                "status": "PRESENT" if exists else "CONFIRMED_ABSENT",
                "authority": "AUTHORITATIVE_CONSISTENT",
                "exists": exists,
                "resource_id": res_id,
            }

        elif op_name == OP_APPLY_TRANSFER:
            # Transfer observation on core banking ledger
            sender = params.get("sender", "acc_alice")
            amount = params.get("amount", 100)
            if fault_mode == FAULT_DELAYED_VISIBILITY and probe_step == 1:
                return {"status": "DELAYED", "authority": "TRANSACTION_LOG_CONSISTENT", "debited": False}
            current_bal = self._accounts.get(sender, 1000)
            # Authoritative presence is established if balance reflects debit
            is_debited = (current_bal <= 1000 - amount)
            return {
                "status": "PRESENT" if is_debited else "CONFIRMED_ABSENT",
                "authority": "TRANSACTION_LOG_CONSISTENT",
                "debited": is_debited,
                "current_balance": current_bal,
            }

        elif op_name == OP_RESERVE_THEN_LINK:
            slot_id = params.get("slot_id", "slot_01")
            ent_id = params.get("entity_id", "ent_01")
            if fault_mode == FAULT_DELAYED_VISIBILITY and probe_step == 1:
                return {"status": "DELAYED", "authority": "AUTHORITATIVE_CONSISTENT", "linked": False}
            linked = (self._slots.get(slot_id) == ent_id)
            return {
                "status": "PRESENT" if linked else "CONFIRMED_ABSENT",
                "authority": "AUTHORITATIVE_CONSISTENT",
                "linked": linked,
                "slot_id": slot_id,
                "entity_id": ent_id,
            }

        return {"status": "UNKNOWN", "authority": "NONE"}

    # ─────────────────────────────────────────────────────────────────────────
    # Authoritative Oracle Store Inspection (STRICTLY FOR INDEPENDENT ORACLE)
    # Never leaks to the agent or policy.
    # ─────────────────────────────────────────────────────────────────────────

    def get_commit_count(self, op_name: str) -> int:
        return len(self._committed_records.get(op_name, []))

    def get_committed_records(self, op_name: str) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._committed_records.get(op_name, []))

    def get_raw_account_balance(self, account_id: str) -> int:
        return self._accounts.get(account_id, 0)
