"""
executor.py — Deterministic task executor for DBOPS V1.

Executes each of the 12 DBOPS tasks against a real SQLite store, modeling
explicit step-by-step transaction boundaries. Each step is a separate
BEGIN/COMMIT so PARTIAL_COMMIT is reachable: step 0 may commit while
step 1 does not.

GROUND TRUTH MODEL:
  After execution, an ExecutionResult records:
    - which steps actually committed (ground_truth_committed_steps)
    - the injected fault mode (hidden from policies)
    - what the "agent-visible" error observation is (byte-identical for
      COMMIT_THEN_TIMEOUT and FAIL_BEFORE_COMMIT paired trials)
    - whether an IntegrityError was raised (not proof of prior success)

No imports from reagent.pipeline, reagent.experiments.ambiguity, runner, or policy.
No network calls, model calls, or .env access.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from reagent.experiments.dbops.store import DbopsStore


# ── Agent-visible error observation ───────────────────────────────────────────
# This is the serialized error that the policy arm can see.
# COMMIT_THEN_TIMEOUT and FAIL_BEFORE_COMMIT must produce byte-identical strings
# so a policy cannot distinguish them by inspection.
AMBIGUOUS_TIMEOUT_ERROR = (
    '{"status": "ERROR", "error_class": "OPERATION_TIMEOUT", '
    '"message": "The operation did not return a response within the allowed window. '
    'Outcome is unknown.", "retry_safe": null}'
)

# Status-unavailable error (UNOBSERVABLE: no probe surface at all)
UNOBSERVABLE_ERROR = (
    '{"status": "ERROR", "error_class": "NO_OBSERVATION_SURFACE", '
    '"message": "No queryable observation surface exists for this operation.", '
    '"retry_safe": null}'
)

# STATUS_UNAVAILABLE: probe exists but is currently unreachable
STATUS_UNAVAILABLE_ERROR = (
    '{"status": "ERROR", "error_class": "STATUS_PROBE_UNAVAILABLE", '
    '"message": "The status probe is temporarily unavailable. '
    'Cannot establish committed state.", "retry_safe": null}'
)


@dataclass
class StepResult:
    step_index: int
    table: str
    committed: bool
    integrity_error: bool = False
    rows_affected: int = 0
    error_detail: Optional[str] = None


@dataclass
class ExecutionResult:
    """Ground truth record — hidden from policy arms."""
    task_id: str
    fault_mode: str
    agent_visible_error: Optional[str]          # Serialized JSON; None = success response
    committed_steps: list[int]                   # step_index values that committed
    integrity_error_raised: bool = False         # True if sqlite3.IntegrityError was raised
    wrong_content_injected: bool = False         # True if WRONG_CONTENT fault was active
    step_results: list[StepResult] = field(default_factory=list)

    @property
    def primary_committed(self) -> bool:
        return 0 in self.committed_steps

    @property
    def all_committed(self) -> bool:
        return len(self.committed_steps) == self._expected_steps

    _expected_steps: int = field(default=2, init=False, repr=False)


# ── Per-task SQL execution ─────────────────────────────────────────────────────

def _execute_task_01(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Refund Issue: INSERT into refunds, then INSERT into refund_audit_log."""
    amount = 999.99 if wrong_content else 250.00
    results = []
    integrity_err = False

    # Step 0: refunds
    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO refunds (refund_id, order_id, customer_id, amount, status) VALUES (?,?,?,?,?)",
            ("REF-550291", "ORD-8821", "CUST-1042", amount, "ISSUED"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "refunds", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "refunds", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    # Step 1: refund_audit_log
    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO refund_audit_log (refund_id, event_type, amount) VALUES (?,?,?)",
            ("REF-550291", "REFUND_ISSUED", amount),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "refund_audit_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "refund_audit_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_02(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Wallet Transfer: INSERT into wallet_transfer_ledger, then wallet_balance_log."""
    amount = 999.99 if wrong_content else 500.00
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO wallet_transfer_ledger (transfer_id, source_acct, dest_acct, amount, direction) VALUES (?,?,?,?,?)",
            ("TXN-779043", "ACCT-2201", "ACCT-3390", amount, "DEBIT"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "wallet_transfer_ledger", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "wallet_transfer_ledger", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO wallet_balance_log (transfer_id, account_id, delta) VALUES (?,?,?)",
            ("TXN-779043", "ACCT-2201", -amount),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "wallet_balance_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "wallet_balance_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_03(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Seat Booking: INSERT into seat_bookings (UNIQUE flight_id,seat_num), then booking_event_log."""
    passenger = "PAX-WRONG" if wrong_content else "PAX-9912"
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO seat_bookings (booking_ref, flight_id, seat_num, passenger_id, status) VALUES (?,?,?,?,?)",
            ("BKG-003341", "FL-2047", "14B", passenger, "CONFIRMED"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "seat_bookings", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "seat_bookings", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO booking_event_log (booking_ref, event_type, flight_id, seat_num) VALUES (?,?,?,?)",
            ("BKG-003341", "BOOKING_CONFIRMED", "FL-2047", "14B"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "booking_event_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "booking_event_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_04(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Customer Merge: UPDATE customers, then INSERT into customer_merge_log."""
    target = "CUST-WRONG" if wrong_content else "CUST-1042"
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "UPDATE customers SET merged_into=?, merge_id=?, status='MERGED', updated_at=datetime('now') WHERE customer_id=?",
            (target, "MRG-002291", "CUST-8810"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "customers", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "customers", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO customer_merge_log (merge_id, source_id, target_id) VALUES (?,?,?)",
            ("MRG-002291", "CUST-8810", target),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "customer_merge_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "customer_merge_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_05(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Order Cancellation: UPDATE orders, then INSERT into order_event_log (UNIQUE order_id,event_type)."""
    customer = "CUST-WRONG" if wrong_content else "CUST-7731"
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "UPDATE orders SET status='CANCELLED', updated_at=datetime('now') WHERE order_id=?",
            ("ORD-5540",),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "orders", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "orders", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO order_event_log (order_id, cancellation_id, event_type) VALUES (?,?,?)",
            ("ORD-5540", "CXL-019882", "CANCELLATION"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "order_event_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "order_event_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_06(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Subscription Upgrade: UPDATE subscriptions tier, then INSERT into subscription_change_log."""
    new_tier = "ENTERPRISE" if wrong_content else "PREMIUM"
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "UPDATE subscriptions SET tier=?, updated_at=datetime('now') WHERE subscription_id=?",
            (new_tier, "SUB-3301"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "subscriptions", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "subscriptions", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO subscription_change_log (upgrade_id, subscription_id, old_tier, new_tier) VALUES (?,?,?,?)",
            ("UPG-008812", "SUB-3301", "BASIC", new_tier),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "subscription_change_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "subscription_change_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_07(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Inventory Restock: UPDATE inventory stock_count, then INSERT into inventory_restock_log."""
    quantity = 999 if wrong_content else 200
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "UPDATE inventory SET stock_count=stock_count+?, updated_at=datetime('now') WHERE sku_id=? AND warehouse_id=?",
            (quantity, "SKU-40821", "WH-05"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "inventory", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "inventory", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO inventory_restock_log (restock_id, sku_id, warehouse_id, quantity) VALUES (?,?,?,?)",
            ("RST-110044", "SKU-40821", "WH-05", quantity),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "inventory_restock_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "inventory_restock_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_08(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Invoice Settlement: UPDATE invoices status, then INSERT into invoice_payments (UNIQUE invoice_id,settlement_ref)."""
    amount = 999.99 if wrong_content else 1200.00
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "UPDATE invoices SET status='SETTLED', updated_at=datetime('now') WHERE invoice_id=?",
            ("INV-7720",),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "invoices", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "invoices", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO invoice_payments (invoice_id, settlement_ref, payer_id, amount) VALUES (?,?,?,?)",
            ("INV-7720", "STL-002981", "PAY-3312", amount),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "invoice_payments", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "invoice_payments", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_09(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Discount Apply: INSERT into discount_redemptions (UNIQUE voucher_id,basket_id), then basket_discount_log."""
    discount_pct = 99.0 if wrong_content else 15.0
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO discount_redemptions (application_id, voucher_id, basket_id, discount_pct) VALUES (?,?,?,?)",
            ("DIS-007712", "VCH-9901", "BSK-6612", discount_pct),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "discount_redemptions", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "discount_redemptions", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO basket_discount_log (application_id, basket_id, voucher_id, discount_pct) VALUES (?,?,?,?)",
            ("DIS-007712", "BSK-6612", "VCH-9901", discount_pct),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "basket_discount_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "basket_discount_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_10(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Account Deactivation: UPDATE accounts status, then INSERT into account_status_history."""
    new_status = "SUSPENDED" if wrong_content else "INACTIVE"
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "UPDATE accounts SET status=?, updated_at=datetime('now') WHERE account_id=?",
            (new_status, "ACC-4490"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "accounts", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "accounts", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO account_status_history (deactivation_id, account_id, old_status, new_status) VALUES (?,?,?,?)",
            ("DAC-003312", "ACC-4490", "ACTIVE", new_status),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "account_status_history", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "account_status_history", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_11(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Escrow Release: INSERT into escrow_disbursement_journal, then UPDATE escrow_accounts balance."""
    amount = 999.99 if wrong_content else 3500.00
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO escrow_disbursement_journal (release_id, escrow_id, beneficiary_id, amount) VALUES (?,?,?,?)",
            ("REL-009921", "ESC-0091", "BEN-2240", amount),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "escrow_disbursement_journal", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "escrow_disbursement_journal", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "UPDATE escrow_accounts SET balance=balance-?, updated_at=datetime('now') WHERE escrow_id=?",
            (amount, "ESC-0091"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "escrow_accounts", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "escrow_accounts", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _execute_task_12(conn: sqlite3.Connection, wrong_content: bool) -> tuple[list[StepResult], bool]:
    """Schema Migration: INSERT into schema_migrations (UNIQUE version), then migration_audit_log."""
    version = 999 if wrong_content else 42
    results = []
    integrity_err = False

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO schema_migrations (migration_id, version, database_target) VALUES (?,?,?)",
            ("MIG-000042", version, "DB-PROD"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(0, "schema_migrations", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(0, "schema_migrations", False, integrity_error=True, error_detail=str(e)))
        return results, integrity_err

    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO migration_audit_log (migration_id, version, status) VALUES (?,?,?)",
            ("MIG-000042", version, "SUCCESS"),
        )
        conn.execute("COMMIT")
        results.append(StepResult(1, "migration_audit_log", True, rows_affected=cur.rowcount))
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        integrity_err = True
        results.append(StepResult(1, "migration_audit_log", False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


# ── Task dispatch map ──────────────────────────────────────────────────────────

_TASK_EXECUTORS = {
    "task_01_refund_issue":         _execute_task_01,
    "task_02_wallet_transfer":      _execute_task_02,
    "task_03_seat_booking":         _execute_task_03,
    "task_04_customer_merge":       _execute_task_04,
    "task_05_order_cancellation":   _execute_task_05,
    "task_06_subscription_upgrade": _execute_task_06,
    "task_07_inventory_restock":    _execute_task_07,
    "task_08_invoice_settlement":   _execute_task_08,
    "task_09_discount_apply":       _execute_task_09,
    "task_10_account_deactivation": _execute_task_10,
    "task_11_escrow_release":       _execute_task_11,
    "task_12_schema_migration":     _execute_task_12,
}


# ── Main execution entry point ─────────────────────────────────────────────────

def execute_task(
    store: DbopsStore,
    task_id: str,
    fault_mode: str,
) -> ExecutionResult:
    """
    Execute task_id against store under fault_mode.

    fault_mode is the hidden ground truth label, never exposed to policies.
    The ExecutionResult records what actually happened.

    Returns: ExecutionResult (hidden from policy arms).
    """
    if task_id not in _TASK_EXECUTORS:
        raise KeyError(f"Unknown task_id: {task_id}")

    executor = _TASK_EXECUTORS[task_id]
    wrong_content = (fault_mode == "WRONG_CONTENT")

    with store.executing_connection() as conn:
        if fault_mode == "FAIL_BEFORE_COMMIT":
            # Execute all SQL but rollback everything before any commit.
            # Agent sees AMBIGUOUS_TIMEOUT_ERROR (byte-identical to COMMIT_THEN_TIMEOUT).
            step_results, integrity_err = _execute_with_full_rollback(conn, executor)
            return ExecutionResult(
                task_id=task_id,
                fault_mode=fault_mode,
                agent_visible_error=AMBIGUOUS_TIMEOUT_ERROR,
                committed_steps=[],
                integrity_error_raised=integrity_err,
                wrong_content_injected=False,
                step_results=step_results,
            )

        elif fault_mode == "COMMIT_THEN_TIMEOUT":
            # Execute all steps normally (real commits), but return timeout error.
            # Agent sees AMBIGUOUS_TIMEOUT_ERROR (byte-identical to FAIL_BEFORE_COMMIT).
            step_results, integrity_err = executor(conn, wrong_content=False)
            committed = [sr.step_index for sr in step_results if sr.committed]
            return ExecutionResult(
                task_id=task_id,
                fault_mode=fault_mode,
                agent_visible_error=AMBIGUOUS_TIMEOUT_ERROR,
                committed_steps=committed,
                integrity_error_raised=integrity_err,
                wrong_content_injected=False,
                step_results=step_results,
            )

        elif fault_mode == "PARTIAL_COMMIT":
            # Execute step 0 (commits), do NOT execute step 1.
            step_results, integrity_err = _execute_only_step_zero(conn, executor, task_id)
            committed = [sr.step_index for sr in step_results if sr.committed]
            return ExecutionResult(
                task_id=task_id,
                fault_mode=fault_mode,
                agent_visible_error=AMBIGUOUS_TIMEOUT_ERROR,
                committed_steps=committed,
                integrity_error_raised=integrity_err,
                wrong_content_injected=False,
                step_results=step_results,
            )

        elif fault_mode == "NO_FAULT":
            step_results, integrity_err = executor(conn, wrong_content=False)
            committed = [sr.step_index for sr in step_results if sr.committed]
            return ExecutionResult(
                task_id=task_id,
                fault_mode=fault_mode,
                agent_visible_error=None,   # Success — tool returns normally
                committed_steps=committed,
                integrity_error_raised=integrity_err,
                wrong_content_injected=False,
                step_results=step_results,
            )

        elif fault_mode == "WRONG_CONTENT":
            step_results, integrity_err = executor(conn, wrong_content=True)
            committed = [sr.step_index for sr in step_results if sr.committed]
            return ExecutionResult(
                task_id=task_id,
                fault_mode=fault_mode,
                agent_visible_error=None,  # Tool reports "success" with wrong data
                committed_steps=committed,
                integrity_error_raised=integrity_err,
                wrong_content_injected=True,
                step_results=step_results,
            )

        elif fault_mode == "DELAYED_VISIBILITY":
            # Steps execute and commit normally, but first probe call returns DELAYED.
            step_results, integrity_err = executor(conn, wrong_content=False)
            committed = [sr.step_index for sr in step_results if sr.committed]
            return ExecutionResult(
                task_id=task_id,
                fault_mode=fault_mode,
                agent_visible_error=None,  # Tool returns success; probe is delayed
                committed_steps=committed,
                integrity_error_raised=integrity_err,
                wrong_content_injected=False,
                step_results=step_results,
            )

        elif fault_mode == "STATUS_UNAVAILABLE":
            # Steps execute and commit normally; probe surface is unavailable.
            step_results, integrity_err = executor(conn, wrong_content=False)
            committed = [sr.step_index for sr in step_results if sr.committed]
            return ExecutionResult(
                task_id=task_id,
                fault_mode=fault_mode,
                agent_visible_error=None,  # Tool succeeds; observation probe fails
                committed_steps=committed,
                integrity_error_raised=integrity_err,
                wrong_content_injected=False,
                step_results=step_results,
            )

        elif fault_mode == "UNOBSERVABLE":
            # Steps execute and commit normally; no observation surface at all.
            step_results, integrity_err = executor(conn, wrong_content=False)
            committed = [sr.step_index for sr in step_results if sr.committed]
            return ExecutionResult(
                task_id=task_id,
                fault_mode=fault_mode,
                agent_visible_error=None,  # Tool succeeds; no probe surface
                committed_steps=committed,
                integrity_error_raised=integrity_err,
                wrong_content_injected=False,
                step_results=step_results,
            )

        else:
            raise ValueError(f"Unknown fault_mode: {fault_mode!r}")


def _execute_with_full_rollback(
    conn: sqlite3.Connection,
    executor,
) -> tuple[list[StepResult], bool]:
    """
    Execute all SQL within a single transaction but always rollback.
    Step results record committed=False. Used for FAIL_BEFORE_COMMIT.
    """
    # Use a single wrapping savepoint to rollback everything
    conn.execute("BEGIN")
    step_results = []
    integrity_err = False
    try:
        # We still run the SQL so that integrity errors are exercised,
        # but we track that nothing will be kept.
        for step_fn in _build_step_fns_from_executor(executor):
            try:
                sr = step_fn(conn)
                step_results.append(StepResult(
                    sr.step_index, sr.table, False,  # committed=False regardless
                    rows_affected=sr.rows_affected,
                ))
            except sqlite3.IntegrityError as e:
                integrity_err = True
                step_results.append(StepResult(
                    0, "unknown", False, integrity_error=True, error_detail=str(e),
                ))
                break
    finally:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
    return step_results, integrity_err


def _build_step_fns_from_executor(executor):
    """
    Not used in real path — placeholder for rollback simulation.
    The real _execute_with_full_rollback uses a different approach below.
    """
    return []


def _execute_with_full_rollback(  # noqa: F811  (intentional override above)
    conn: sqlite3.Connection,
    executor,
) -> tuple[list[StepResult], bool]:
    """
    Run SQL in a single savepoint; always rollback at the end.
    This simulates FAIL_BEFORE_COMMIT: SQL ran but nothing committed.
    """
    conn.execute("SAVEPOINT fail_before_commit_sp")
    try:
        step_results, integrity_err = executor(conn, wrong_content=False)
    except Exception:
        integrity_err = True
        step_results = []
    finally:
        conn.execute("ROLLBACK TO SAVEPOINT fail_before_commit_sp")
        conn.execute("RELEASE SAVEPOINT fail_before_commit_sp")
    # Mark all as not committed
    for sr in step_results:
        object.__setattr__(sr, "committed", False)
    return step_results, integrity_err


def _execute_only_step_zero(
    conn: sqlite3.Connection,
    executor,
    task_id: str,
) -> tuple[list[StepResult], bool]:
    """
    Execute only step 0 (primary table effect) and commit it.
    Step 1 is never executed — models PARTIAL_COMMIT.
    """
    # We need to run only the step-0 SQL for each task.
    step0_fns = {
        "task_01_refund_issue": lambda c: (
            c.execute("BEGIN"),
            c.execute("INSERT INTO refunds (refund_id, order_id, customer_id, amount, status) VALUES (?,?,?,?,?)",
                      ("REF-550291", "ORD-8821", "CUST-1042", 250.00, "ISSUED")),
            c.execute("COMMIT"),
        ),
        "task_02_wallet_transfer": lambda c: (
            c.execute("BEGIN"),
            c.execute("INSERT INTO wallet_transfer_ledger (transfer_id, source_acct, dest_acct, amount, direction) VALUES (?,?,?,?,?)",
                      ("TXN-779043", "ACCT-2201", "ACCT-3390", 500.00, "DEBIT")),
            c.execute("COMMIT"),
        ),
        "task_03_seat_booking": lambda c: (
            c.execute("BEGIN"),
            c.execute("INSERT INTO seat_bookings (booking_ref, flight_id, seat_num, passenger_id, status) VALUES (?,?,?,?,?)",
                      ("BKG-003341", "FL-2047", "14B", "PAX-9912", "CONFIRMED")),
            c.execute("COMMIT"),
        ),
        "task_04_customer_merge": lambda c: (
            c.execute("BEGIN"),
            c.execute("UPDATE customers SET merged_into=?, merge_id=?, status='MERGED', updated_at=datetime('now') WHERE customer_id=?",
                      ("CUST-1042", "MRG-002291", "CUST-8810")),
            c.execute("COMMIT"),
        ),
        "task_05_order_cancellation": lambda c: (
            c.execute("BEGIN"),
            c.execute("UPDATE orders SET status='CANCELLED', updated_at=datetime('now') WHERE order_id=?",
                      ("ORD-5540",)),
            c.execute("COMMIT"),
        ),
        "task_06_subscription_upgrade": lambda c: (
            c.execute("BEGIN"),
            c.execute("UPDATE subscriptions SET tier='PREMIUM', updated_at=datetime('now') WHERE subscription_id=?",
                      ("SUB-3301",)),
            c.execute("COMMIT"),
        ),
        "task_07_inventory_restock": lambda c: (
            c.execute("BEGIN"),
            c.execute("UPDATE inventory SET stock_count=stock_count+200, updated_at=datetime('now') WHERE sku_id=? AND warehouse_id=?",
                      ("SKU-40821", "WH-05")),
            c.execute("COMMIT"),
        ),
        "task_08_invoice_settlement": lambda c: (
            c.execute("BEGIN"),
            c.execute("UPDATE invoices SET status='SETTLED', updated_at=datetime('now') WHERE invoice_id=?",
                      ("INV-7720",)),
            c.execute("COMMIT"),
        ),
        "task_09_discount_apply": lambda c: (
            c.execute("BEGIN"),
            c.execute("INSERT INTO discount_redemptions (application_id, voucher_id, basket_id, discount_pct) VALUES (?,?,?,?)",
                      ("DIS-007712", "VCH-9901", "BSK-6612", 15.0)),
            c.execute("COMMIT"),
        ),
        "task_10_account_deactivation": lambda c: (
            c.execute("BEGIN"),
            c.execute("UPDATE accounts SET status='INACTIVE', updated_at=datetime('now') WHERE account_id=?",
                      ("ACC-4490",)),
            c.execute("COMMIT"),
        ),
        "task_11_escrow_release": lambda c: (
            c.execute("BEGIN"),
            c.execute("INSERT INTO escrow_disbursement_journal (release_id, escrow_id, beneficiary_id, amount) VALUES (?,?,?,?)",
                      ("REL-009921", "ESC-0091", "BEN-2240", 3500.00)),
            c.execute("COMMIT"),
        ),
        "task_12_schema_migration": lambda c: (
            c.execute("BEGIN"),
            c.execute("INSERT INTO schema_migrations (migration_id, version, database_target) VALUES (?,?,?)",
                      ("MIG-000042", 42, "DB-PROD")),
            c.execute("COMMIT"),
        ),
    }

    step0_fn = step0_fns[task_id]
    results = []
    integrity_err = False
    try:
        step0_fn(conn)
        results.append(StepResult(0, _primary_table_for(task_id), True, rows_affected=1))
    except sqlite3.IntegrityError as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        integrity_err = True
        results.append(StepResult(0, _primary_table_for(task_id), False, integrity_error=True, error_detail=str(e)))

    return results, integrity_err


def _primary_table_for(task_id: str) -> str:
    _map = {
        "task_01_refund_issue": "refunds",
        "task_02_wallet_transfer": "wallet_transfer_ledger",
        "task_03_seat_booking": "seat_bookings",
        "task_04_customer_merge": "customers",
        "task_05_order_cancellation": "orders",
        "task_06_subscription_upgrade": "subscriptions",
        "task_07_inventory_restock": "inventory",
        "task_08_invoice_settlement": "invoices",
        "task_09_discount_apply": "discount_redemptions",
        "task_10_account_deactivation": "accounts",
        "task_11_escrow_release": "escrow_disbursement_journal",
        "task_12_schema_migration": "schema_migrations",
    }
    return _map.get(task_id, "unknown")
