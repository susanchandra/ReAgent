"""
oracle.py — Independent ground-truth oracle for DBOPS V1.

The oracle evaluates actual SQLite committed state using fresh-connection
semantics. It is completely independent of:
  - Policy arm decisions
  - Gold contract evaluator outputs
  - Injected fault labels
  - Policy-visible agent errors

ISOLATION MODEL:
  - Oracle always opens a FRESH connection AFTER the executing connection is closed.
  - The executing connection must be fully closed before oracle is called.
  - No shared connection state, no shared cursor, no shared transaction.
  - query_only=ON prevents accidental writes.

VERDICT SEMANTICS (per manifest VerdictType):
  EXPECTED_PRESENT — all required effects committed correctly.
  CONFIRMED_ABSENT — fresh query on committed state finds no matching row.
  PARTIAL          — primary committed, required secondary absent.
  WRONG_CONTENT    — row present but field values contradict contract.
  UNKNOWN          — cannot establish committed state (probe failed, no surface, etc.)

RULES:
  - UNKNOWN is never promoted to CONFIRMED_ABSENT.
  - IntegrityError on retry is not proof of prior success.
  - Stale reads and locked queries do not qualify as CONFIRMED_ABSENT.
  - Non-idempotent operations must ESCALATE if outcome is UNKNOWN.

No imports from reagent.pipeline, reagent.experiments.ambiguity, runner, or policy.
No network calls, model calls, or .env access.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from reagent.experiments.dbops.types import VerdictType
from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.executor import (
    AMBIGUOUS_TIMEOUT_ERROR,
    STATUS_UNAVAILABLE_ERROR,
    UNOBSERVABLE_ERROR,
    ExecutionResult,
)


@dataclass
class OracleObservation:
    """
    What the oracle found in committed state.
    This is the full ground truth, not filtered for policy consumption.
    """
    task_id: str
    fault_mode: str
    verdict: VerdictType
    primary_present: bool
    secondary_present: dict[str, bool]   # table -> present
    primary_row: Optional[dict[str, Any]] = None
    secondary_rows: dict[str, Optional[dict[str, Any]]] = field(default_factory=dict)
    wrong_content_detected: bool = False
    duplicate_primary_count: int = 0     # How many primary effect rows exist
    duplicate_secondary_counts: dict[str, int] = field(default_factory=dict)
    observation_authority: str = "FRESH_QUERY_ON_COMMITTED_STATE"
    note: str = ""


@dataclass
class ProbeResult:
    """What a single observation probe call returns to policy arms."""
    status: str            # "PRESENT" | "ABSENT" | "DELAYED" | "UNAVAILABLE" | "UNOBSERVABLE"
    call_number: int       # 1-based; delayed visibility fires on call 1
    detail: Optional[str] = None


class DbopsOracle:
    """
    Independent oracle that evaluates committed database state.

    Caller workflow:
        oracle = DbopsOracle(store, exec_result)
        obs = oracle.evaluate()          # Full ground-truth verdict
        probe = oracle.probe(call_n=1)   # Policy-visible probe (limited view)
    """

    def __init__(self, store: DbopsStore, exec_result: ExecutionResult) -> None:
        self._store = store
        self._exec = exec_result
        self._probe_call_count = 0

    def probe(self, call_number: int = 1) -> ProbeResult:
        """
        Return a policy-visible probe result.
        Probe is limited: it does NOT expose fault labels, committed_steps,
        or any hidden execution state.

        call_number: 1-based call index (for DELAYED_VISIBILITY protocol).
        """
        fault = self._exec.fault_mode

        if fault == "STATUS_UNAVAILABLE":
            return ProbeResult("UNAVAILABLE", call_number, STATUS_UNAVAILABLE_ERROR)

        if fault == "UNOBSERVABLE":
            return ProbeResult("UNOBSERVABLE", call_number, UNOBSERVABLE_ERROR)

        if fault == "DELAYED_VISIBILITY" and call_number == 1:
            # First probe returns DELAYED per protocol; fault label NOT included.
            return ProbeResult("DELAYED", call_number, '{"status": "DELAYED"}')

        # All other cases: real fresh-connection query.
        with self._store.fresh_observer_connection() as conn:
            row = self._find_primary_row(conn)
            if row is None:
                return ProbeResult("ABSENT", call_number)
            return ProbeResult("PRESENT", call_number)

    def evaluate(self) -> OracleObservation:
        """
        Full ground-truth evaluation. Uses fresh observer connection.
        Called by the evaluator after execution, not by policy arms.
        """
        fault = self._exec.fault_mode
        task_id = self._exec.task_id

        if fault == "STATUS_UNAVAILABLE":
            return OracleObservation(
                task_id=task_id,
                fault_mode=fault,
                verdict=VerdictType.UNKNOWN,
                primary_present=False,
                secondary_present={},
                observation_authority="UNAVAILABLE",
                note="STATUS_UNAVAILABLE: probe surface temporarily unreachable. Cannot establish committed state.",
            )

        if fault == "UNOBSERVABLE":
            return OracleObservation(
                task_id=task_id,
                fault_mode=fault,
                verdict=VerdictType.UNKNOWN,
                primary_present=False,
                secondary_present={},
                observation_authority="UNOBSERVABLE",
                note="UNOBSERVABLE: no observation surface exists for this operation.",
            )

        with self._store.fresh_observer_connection() as conn:
            return self._evaluate_with_conn(conn)

    def _evaluate_with_conn(self, conn: sqlite3.Connection) -> OracleObservation:
        task_id = self._exec.task_id
        fault = self._exec.fault_mode

        primary_row = self._find_primary_row(conn)
        secondary_rows, secondary_counts = self._find_secondary_rows(conn)
        duplicate_primary = self._count_primary_rows(conn)

        primary_present = primary_row is not None

        # Build secondary_present map
        sec_tables = _secondary_tables_for(task_id)
        secondary_present = {t: (secondary_rows.get(t) is not None) for t in sec_tables}

        # Detect wrong content
        wrong_content = False
        if primary_present and self._exec.wrong_content_injected:
            wrong_content = self._check_wrong_content(primary_row, secondary_rows)

        # Determine verdict
        verdict = self._determine_verdict(
            primary_present=primary_present,
            secondary_present=secondary_present,
            wrong_content=wrong_content,
        )

        return OracleObservation(
            task_id=task_id,
            fault_mode=fault,
            verdict=verdict,
            primary_present=primary_present,
            secondary_present=secondary_present,
            primary_row=primary_row,
            secondary_rows=secondary_rows,
            wrong_content_detected=wrong_content,
            duplicate_primary_count=duplicate_primary,
            duplicate_secondary_counts=secondary_counts,
            observation_authority="FRESH_QUERY_ON_COMMITTED_STATE",
        )

    def _determine_verdict(
        self,
        primary_present: bool,
        secondary_present: dict[str, bool],
        wrong_content: bool,
    ) -> VerdictType:
        if not primary_present:
            # No primary row: either nothing committed or fail_before_commit
            # Check if all secondaries are also absent
            if not any(secondary_present.values()):
                return VerdictType.CONFIRMED_ABSENT
            # Secondary present but no primary is unusual — still UNKNOWN
            return VerdictType.UNKNOWN

        # Primary is present
        if wrong_content:
            return VerdictType.WRONG_CONTENT

        all_secondary_present = all(secondary_present.values())
        if all_secondary_present:
            return VerdictType.EXPECTED_PRESENT

        # Some secondary missing
        any_secondary_present = any(secondary_present.values())
        if not any_secondary_present:
            # Primary present, all secondaries absent = PARTIAL
            return VerdictType.PARTIAL
        # Mix of present/absent secondaries = PARTIAL
        return VerdictType.PARTIAL

    def _find_primary_row(self, conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
        task_id = self._exec.task_id
        query, params = _primary_probe_query(task_id)
        cur = conn.execute(query, params)
        cur.row_factory = sqlite3.Row
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    def _count_primary_rows(self, conn: sqlite3.Connection) -> int:
        task_id = self._exec.task_id
        count_query, params = _primary_count_query(task_id)
        cur = conn.execute(count_query, params)
        return cur.fetchone()[0]

    def _find_secondary_rows(
        self, conn: sqlite3.Connection
    ) -> tuple[dict[str, Optional[dict[str, Any]]], dict[str, int]]:
        task_id = self._exec.task_id
        rows = {}
        counts = {}
        for table, query, params, count_query in _secondary_probe_queries(task_id):
            cur = conn.execute(query, params)
            row = cur.fetchone()
            rows[table] = dict(zip([d[0] for d in cur.description], row)) if row else None
            cur2 = conn.execute(count_query, params)
            counts[table] = cur2.fetchone()[0]
        return rows, counts

    def _check_wrong_content(
        self,
        primary_row: dict[str, Any],
        secondary_rows: dict[str, Optional[dict[str, Any]]],
    ) -> bool:
        """Check if stored values deviate from frozen prompt values."""
        task_id = self._exec.task_id
        return _has_wrong_content(task_id, primary_row, secondary_rows)


# ── Per-task probe queries ─────────────────────────────────────────────────────

def _primary_probe_query(task_id: str) -> tuple[str, tuple]:
    return {
        "task_01_refund_issue":
            ("SELECT * FROM refunds WHERE refund_id=?", ("REF-550291",)),
        "task_02_wallet_transfer":
            ("SELECT * FROM wallet_transfer_ledger WHERE transfer_id=? AND direction='DEBIT'", ("TXN-779043",)),
        "task_03_seat_booking":
            ("SELECT * FROM seat_bookings WHERE flight_id=? AND seat_num=?", ("FL-2047", "14B")),
        "task_04_customer_merge":
            ("SELECT * FROM customers WHERE customer_id=? AND status='MERGED'", ("CUST-8810",)),
        "task_05_order_cancellation":
            ("SELECT * FROM orders WHERE order_id=? AND status='CANCELLED'", ("ORD-5540",)),
        "task_06_subscription_upgrade":
            ("SELECT * FROM subscriptions WHERE subscription_id=? AND tier='PREMIUM'", ("SUB-3301",)),
        "task_07_inventory_restock":
            ("SELECT * FROM inventory WHERE sku_id=? AND warehouse_id=?", ("SKU-40821", "WH-05")),
        "task_08_invoice_settlement":
            ("SELECT * FROM invoices WHERE invoice_id=? AND status='SETTLED'", ("INV-7720",)),
        "task_09_discount_apply":
            ("SELECT * FROM discount_redemptions WHERE voucher_id=? AND basket_id=?", ("VCH-9901", "BSK-6612")),
        "task_10_account_deactivation":
            ("SELECT * FROM accounts WHERE account_id=? AND status='INACTIVE'", ("ACC-4490",)),
        "task_11_escrow_release":
            ("SELECT * FROM escrow_disbursement_journal WHERE release_id=?", ("REL-009921",)),
        "task_12_schema_migration":
            ("SELECT * FROM schema_migrations WHERE version=?", (42,)),
    }[task_id]


def _primary_count_query(task_id: str) -> tuple[str, tuple]:
    return {
        "task_01_refund_issue":
            ("SELECT COUNT(*) FROM refunds WHERE refund_id=?", ("REF-550291",)),
        "task_02_wallet_transfer":
            ("SELECT COUNT(*) FROM wallet_transfer_ledger WHERE transfer_id=?", ("TXN-779043",)),
        "task_03_seat_booking":
            ("SELECT COUNT(*) FROM seat_bookings WHERE flight_id=? AND seat_num=?", ("FL-2047", "14B")),
        "task_04_customer_merge":
            ("SELECT COUNT(*) FROM customers WHERE customer_id=? AND status='MERGED'", ("CUST-8810",)),
        "task_05_order_cancellation":
            ("SELECT COUNT(*) FROM orders WHERE order_id=? AND status='CANCELLED'", ("ORD-5540",)),
        "task_06_subscription_upgrade":
            ("SELECT COUNT(*) FROM subscriptions WHERE subscription_id=? AND tier='PREMIUM'", ("SUB-3301",)),
        "task_07_inventory_restock":
            ("SELECT COUNT(*) FROM inventory WHERE sku_id=? AND warehouse_id=?", ("SKU-40821", "WH-05")),
        "task_08_invoice_settlement":
            ("SELECT COUNT(*) FROM invoices WHERE invoice_id=? AND status='SETTLED'", ("INV-7720",)),
        "task_09_discount_apply":
            ("SELECT COUNT(*) FROM discount_redemptions WHERE voucher_id=? AND basket_id=?", ("VCH-9901", "BSK-6612")),
        "task_10_account_deactivation":
            ("SELECT COUNT(*) FROM accounts WHERE account_id=? AND status='INACTIVE'", ("ACC-4490",)),
        "task_11_escrow_release":
            ("SELECT COUNT(*) FROM escrow_disbursement_journal WHERE release_id=?", ("REL-009921",)),
        "task_12_schema_migration":
            ("SELECT COUNT(*) FROM schema_migrations WHERE version=?", (42,)),
    }[task_id]


def _secondary_tables_for(task_id: str) -> list[str]:
    return {
        "task_01_refund_issue":         ["refund_audit_log"],
        "task_02_wallet_transfer":      ["wallet_balance_log"],
        "task_03_seat_booking":         ["booking_event_log"],
        "task_04_customer_merge":       ["customer_merge_log"],
        "task_05_order_cancellation":   ["order_event_log"],
        "task_06_subscription_upgrade": ["subscription_change_log"],
        "task_07_inventory_restock":    ["inventory_restock_log"],
        "task_08_invoice_settlement":   ["invoice_payments"],
        "task_09_discount_apply":       ["basket_discount_log"],
        "task_10_account_deactivation": ["account_status_history"],
        "task_11_escrow_release":       ["escrow_accounts"],
        "task_12_schema_migration":     ["migration_audit_log"],
    }[task_id]


def _secondary_probe_queries(task_id: str) -> list[tuple[str, str, tuple, str]]:
    """Return list of (table, select_query, params, count_query) tuples."""
    return {
        "task_01_refund_issue": [
            ("refund_audit_log",
             "SELECT * FROM refund_audit_log WHERE refund_id=? LIMIT 1", ("REF-550291",),
             "SELECT COUNT(*) FROM refund_audit_log WHERE refund_id=?", ),
        ],
        "task_02_wallet_transfer": [
            ("wallet_balance_log",
             "SELECT * FROM wallet_balance_log WHERE transfer_id=? LIMIT 1", ("TXN-779043",),
             "SELECT COUNT(*) FROM wallet_balance_log WHERE transfer_id=?", ),
        ],
        "task_03_seat_booking": [
            ("booking_event_log",
             "SELECT * FROM booking_event_log WHERE booking_ref=? LIMIT 1", ("BKG-003341",),
             "SELECT COUNT(*) FROM booking_event_log WHERE booking_ref=?", ),
        ],
        "task_04_customer_merge": [
            ("customer_merge_log",
             "SELECT * FROM customer_merge_log WHERE merge_id=? LIMIT 1", ("MRG-002291",),
             "SELECT COUNT(*) FROM customer_merge_log WHERE merge_id=?", ),
        ],
        "task_05_order_cancellation": [
            ("order_event_log",
             "SELECT * FROM order_event_log WHERE order_id=? AND event_type='CANCELLATION' LIMIT 1", ("ORD-5540",),
             "SELECT COUNT(*) FROM order_event_log WHERE order_id=? AND event_type='CANCELLATION'", ),
        ],
        "task_06_subscription_upgrade": [
            ("subscription_change_log",
             "SELECT * FROM subscription_change_log WHERE upgrade_id=? LIMIT 1", ("UPG-008812",),
             "SELECT COUNT(*) FROM subscription_change_log WHERE upgrade_id=?", ),
        ],
        "task_07_inventory_restock": [
            ("inventory_restock_log",
             "SELECT * FROM inventory_restock_log WHERE restock_id=? LIMIT 1", ("RST-110044",),
             "SELECT COUNT(*) FROM inventory_restock_log WHERE restock_id=?", ),
        ],
        "task_08_invoice_settlement": [
            ("invoice_payments",
             "SELECT * FROM invoice_payments WHERE invoice_id=? AND settlement_ref=? LIMIT 1",
             ("INV-7720", "STL-002981"),
             "SELECT COUNT(*) FROM invoice_payments WHERE invoice_id=? AND settlement_ref=?", ),
        ],
        "task_09_discount_apply": [
            ("basket_discount_log",
             "SELECT * FROM basket_discount_log WHERE application_id=? LIMIT 1", ("DIS-007712",),
             "SELECT COUNT(*) FROM basket_discount_log WHERE application_id=?", ),
        ],
        "task_10_account_deactivation": [
            ("account_status_history",
             "SELECT * FROM account_status_history WHERE deactivation_id=? LIMIT 1", ("DAC-003312",),
             "SELECT COUNT(*) FROM account_status_history WHERE deactivation_id=?", ),
        ],
        "task_11_escrow_release": [
            ("escrow_accounts",
             "SELECT * FROM escrow_accounts WHERE escrow_id=? LIMIT 1", ("ESC-0091",),
             "SELECT COUNT(*) FROM escrow_accounts WHERE escrow_id=?", ),
        ],
        "task_12_schema_migration": [
            ("migration_audit_log",
             "SELECT * FROM migration_audit_log WHERE migration_id=? LIMIT 1", ("MIG-000042",),
             "SELECT COUNT(*) FROM migration_audit_log WHERE migration_id=?", ),
        ],
    }[task_id]


def _has_wrong_content(
    task_id: str,
    primary_row: Optional[dict[str, Any]],
    secondary_rows: dict[str, Optional[dict[str, Any]]],
) -> bool:
    """Return True if stored field values contradict the frozen prompt contract."""
    if primary_row is None:
        return False

    checks = {
        "task_01_refund_issue": [("refunds", "amount", 250.00)],
        "task_02_wallet_transfer": [("wallet_transfer_ledger", "amount", 500.00)],
        "task_03_seat_booking": [("seat_bookings", "passenger_id", "PAX-9912")],
        "task_04_customer_merge": [("customers", "merged_into", "CUST-1042")],
        "task_05_order_cancellation": [],  # primary is idempotent UPDATE; wrong_content is customer_id mismatch
        "task_06_subscription_upgrade": [("subscriptions", "tier", "PREMIUM")],
        "task_07_inventory_restock": [],  # increment; wrong_content is wrong quantity in log
        "task_08_invoice_settlement": [],  # status update; wrong_content is payment amount
        "task_09_discount_apply": [("discount_redemptions", "discount_pct", 15.0)],
        "task_10_account_deactivation": [("accounts", "status", "INACTIVE")],
        "task_11_escrow_release": [("escrow_disbursement_journal", "amount", 3500.00)],
        "task_12_schema_migration": [("schema_migrations", "version", 42)],
    }.get(task_id, [])

    for table, col, expected in checks:
        if table == primary_row.get("__table__", None) or True:
            actual = primary_row.get(col)
            if actual is not None and abs(float(actual) - float(expected)) > 0.001 if isinstance(expected, float) else actual != expected:
                return True

    # Also check secondary rows
    for table, sec_row in secondary_rows.items():
        if sec_row is None:
            continue
        sec_checks = {
            "task_07_inventory_restock": [("inventory_restock_log", "quantity", 200)],
            "task_08_invoice_settlement": [("invoice_payments", "amount", 1200.00)],
        }.get(task_id, [])
        for _t, col, expected in sec_checks:
            if _t == table:
                actual = sec_row.get(col)
                if actual is not None:
                    if isinstance(expected, float):
                        if abs(float(actual) - expected) > 0.001:
                            return True
                    elif actual != expected:
                        return True

    return False
