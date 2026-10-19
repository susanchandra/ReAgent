"""
schema.py — SQLite schema DDL for DBOPS V1 experiment tables.

CONSTRAINT NOTES (from manifest B1.1):
  - task_05: UNIQUE(order_id, event_type) is on order_event_log, NOT on orders.
  - task_08: UNIQUE(invoice_id, settlement_ref) is on invoice_payments, NOT on invoices.
  - tasks 02, 07, 11 (UNPROTECTED_NUMERIC): no UNIQUE constraints; duplicate rows are intentional.
  - tasks 04, 06, 10 (UNPROTECTED_AUDIT_ONLY): no UNIQUE constraints on audit tables.

No imports from reagent.pipeline, ambiguity, runner, or policy modules.
No live model calls, network access, or file I/O beyond SQLite.
"""

# Each tuple: (table_name, CREATE TABLE SQL)
# Tables must be created in this order (no FK dependency issues; SQLite is lenient).

SCHEMA_DDL: list[tuple[str, str]] = [

    # ── Task 01: Refund Issue ──────────────────────────────────────────────────
    ("refunds", """
        CREATE TABLE IF NOT EXISTS refunds (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            refund_id     TEXT NOT NULL,
            order_id      TEXT NOT NULL,
            customer_id   TEXT NOT NULL,
            amount        REAL NOT NULL,
            status        TEXT NOT NULL DEFAULT 'ISSUED',
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(refund_id)
        )
    """),
    ("refund_audit_log", """
        CREATE TABLE IF NOT EXISTS refund_audit_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            refund_id     TEXT NOT NULL,
            event_type    TEXT NOT NULL,
            amount        REAL NOT NULL,
            logged_at     TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),

    # ── Task 02: Wallet Transfer (UNPROTECTED_NUMERIC — no UNIQUE) ────────────
    ("wallet_transfer_ledger", """
        CREATE TABLE IF NOT EXISTS wallet_transfer_ledger (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            transfer_id   TEXT NOT NULL,
            source_acct   TEXT NOT NULL,
            dest_acct     TEXT NOT NULL,
            amount        REAL NOT NULL,
            direction     TEXT NOT NULL DEFAULT 'DEBIT',
            created_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),
    ("wallet_balance_log", """
        CREATE TABLE IF NOT EXISTS wallet_balance_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            transfer_id   TEXT NOT NULL,
            account_id    TEXT NOT NULL,
            delta         REAL NOT NULL,
            logged_at     TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),

    # ── Task 03: Seat Booking ─────────────────────────────────────────────────
    ("seat_bookings", """
        CREATE TABLE IF NOT EXISTS seat_bookings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            booking_ref   TEXT NOT NULL,
            flight_id     TEXT NOT NULL,
            seat_num      TEXT NOT NULL,
            passenger_id  TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'CONFIRMED',
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(flight_id, seat_num)
        )
    """),
    ("booking_event_log", """
        CREATE TABLE IF NOT EXISTS booking_event_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            booking_ref   TEXT NOT NULL,
            event_type    TEXT NOT NULL,
            flight_id     TEXT NOT NULL,
            seat_num      TEXT NOT NULL,
            logged_at     TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),

    # ── Task 04: Customer Merge (UNPROTECTED_AUDIT_ONLY — no UNIQUE on log) ───
    ("customers", """
        CREATE TABLE IF NOT EXISTS customers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id   TEXT NOT NULL,
            merge_id      TEXT,
            merged_into   TEXT,
            status        TEXT NOT NULL DEFAULT 'ACTIVE',
            updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(customer_id)
        )
    """),
    ("customer_merge_log", """
        CREATE TABLE IF NOT EXISTS customer_merge_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            merge_id      TEXT NOT NULL,
            source_id     TEXT NOT NULL,
            target_id     TEXT NOT NULL,
            logged_at     TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),

    # ── Task 05: Order Cancellation (UNIQUE on order_event_log, NOT on orders) ─
    ("orders", """
        CREATE TABLE IF NOT EXISTS orders (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id      TEXT NOT NULL,
            customer_id   TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'ACTIVE',
            updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(order_id)
        )
    """),
    ("order_event_log", """
        CREATE TABLE IF NOT EXISTS order_event_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id         TEXT NOT NULL,
            cancellation_id  TEXT NOT NULL,
            event_type       TEXT NOT NULL,
            logged_at        TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(order_id, event_type)
        )
    """),

    # ── Task 06: Subscription Upgrade (UNPROTECTED_AUDIT_ONLY) ───────────────
    ("subscriptions", """
        CREATE TABLE IF NOT EXISTS subscriptions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id TEXT NOT NULL,
            customer_id     TEXT NOT NULL,
            tier            TEXT NOT NULL DEFAULT 'BASIC',
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(subscription_id)
        )
    """),
    ("subscription_change_log", """
        CREATE TABLE IF NOT EXISTS subscription_change_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            upgrade_id      TEXT NOT NULL,
            subscription_id TEXT NOT NULL,
            old_tier        TEXT NOT NULL,
            new_tier        TEXT NOT NULL,
            logged_at       TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),

    # ── Task 07: Inventory Restock (UNPROTECTED_NUMERIC — no UNIQUE on either) ─
    ("inventory", """
        CREATE TABLE IF NOT EXISTS inventory (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id          TEXT NOT NULL,
            warehouse_id    TEXT NOT NULL,
            stock_count     INTEGER NOT NULL DEFAULT 0,
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(sku_id, warehouse_id)
        )
    """),
    ("inventory_restock_log", """
        CREATE TABLE IF NOT EXISTS inventory_restock_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            restock_id      TEXT NOT NULL,
            sku_id          TEXT NOT NULL,
            warehouse_id    TEXT NOT NULL,
            quantity        INTEGER NOT NULL,
            logged_at       TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),

    # ── Task 08: Invoice Settlement (UNIQUE on invoice_payments, NOT on invoices) ─
    ("invoices", """
        CREATE TABLE IF NOT EXISTS invoices (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id      TEXT NOT NULL,
            payer_id        TEXT NOT NULL,
            amount          REAL NOT NULL,
            status          TEXT NOT NULL DEFAULT 'PENDING',
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(invoice_id)
        )
    """),
    ("invoice_payments", """
        CREATE TABLE IF NOT EXISTS invoice_payments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id      TEXT NOT NULL,
            settlement_ref  TEXT NOT NULL,
            payer_id        TEXT NOT NULL,
            amount          REAL NOT NULL,
            logged_at       TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(invoice_id, settlement_ref)
        )
    """),

    # ── Task 09: Discount Apply ───────────────────────────────────────────────
    ("discount_redemptions", """
        CREATE TABLE IF NOT EXISTS discount_redemptions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id  TEXT NOT NULL,
            voucher_id      TEXT NOT NULL,
            basket_id       TEXT NOT NULL,
            discount_pct    REAL NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(voucher_id, basket_id)
        )
    """),
    ("basket_discount_log", """
        CREATE TABLE IF NOT EXISTS basket_discount_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id  TEXT NOT NULL,
            basket_id       TEXT NOT NULL,
            voucher_id      TEXT NOT NULL,
            discount_pct    REAL NOT NULL,
            logged_at       TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),

    # ── Task 10: Account Deactivation (UNPROTECTED_AUDIT_ONLY) ───────────────
    ("accounts", """
        CREATE TABLE IF NOT EXISTS accounts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id      TEXT NOT NULL,
            customer_id     TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'ACTIVE',
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(account_id)
        )
    """),
    ("account_status_history", """
        CREATE TABLE IF NOT EXISTS account_status_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            deactivation_id TEXT NOT NULL,
            account_id      TEXT NOT NULL,
            old_status      TEXT NOT NULL,
            new_status      TEXT NOT NULL,
            logged_at       TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),

    # ── Task 11: Escrow Release (UNPROTECTED_NUMERIC — no UNIQUE on journal) ──
    ("escrow_disbursement_journal", """
        CREATE TABLE IF NOT EXISTS escrow_disbursement_journal (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            release_id      TEXT NOT NULL,
            escrow_id       TEXT NOT NULL,
            beneficiary_id  TEXT NOT NULL,
            amount          REAL NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),
    ("escrow_accounts", """
        CREATE TABLE IF NOT EXISTS escrow_accounts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            escrow_id       TEXT NOT NULL,
            beneficiary_id  TEXT NOT NULL,
            balance         REAL NOT NULL DEFAULT 0.0,
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(escrow_id)
        )
    """),

    # ── Task 12: Schema Migration ─────────────────────────────────────────────
    ("schema_migrations", """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_id    TEXT NOT NULL,
            version         INTEGER NOT NULL,
            database_target TEXT NOT NULL,
            applied_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(version)
        )
    """),
    ("migration_audit_log", """
        CREATE TABLE IF NOT EXISTS migration_audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_id    TEXT NOT NULL,
            version         INTEGER NOT NULL,
            status          TEXT NOT NULL DEFAULT 'SUCCESS',
            logged_at       TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),
]


# Table names that each task is permitted to write to (mirrors tasks.py allowed_tables).
# Used by the validator to reject CLASS_TABLE_OUTSIDE_ALLOWLIST.
TASK_ALLOWED_TABLES: dict[str, frozenset[str]] = {
    "task_01_refund_issue":          frozenset({"refunds", "refund_audit_log"}),
    "task_02_wallet_transfer":       frozenset({"wallet_transfer_ledger", "wallet_balance_log"}),
    "task_03_seat_booking":          frozenset({"seat_bookings", "booking_event_log"}),
    "task_04_customer_merge":        frozenset({"customers", "customer_merge_log"}),
    "task_05_order_cancellation":    frozenset({"orders", "order_event_log"}),
    "task_06_subscription_upgrade":  frozenset({"subscriptions", "subscription_change_log"}),
    "task_07_inventory_restock":     frozenset({"inventory", "inventory_restock_log"}),
    "task_08_invoice_settlement":    frozenset({"invoices", "invoice_payments"}),
    "task_09_discount_apply":        frozenset({"discount_redemptions", "basket_discount_log"}),
    "task_10_account_deactivation":  frozenset({"accounts", "account_status_history"}),
    "task_11_escrow_release":        frozenset({"escrow_disbursement_journal", "escrow_accounts"}),
    "task_12_schema_migration":      frozenset({"schema_migrations", "migration_audit_log"}),
}

# All known table names in this benchmark.
ALL_TABLE_NAMES: frozenset[str] = frozenset(
    name for name, _ in SCHEMA_DDL
)
