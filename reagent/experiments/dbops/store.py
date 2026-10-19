"""
store.py — Isolated SQLite store for the DBOPS V1 experiment.

Responsibilities:
  - Create / reset per-trial temporary databases.
  - Apply schema DDL with correct UNIQUE constraints.
  - Provide connection factories (executing connection and fresh observer connection).
  - Manage initial state seeding (pre-existing rows for tasks that need them).
  - Safe cleanup of trial databases after use.

ISOLATION MODEL:
  Each trial runs in its own :memory: or named temp file database.
  The executing connection and the oracle connection are NEVER the same object.
  The oracle always opens a fresh connection AFTER the executing connection
  is closed/committed/aborted. This enforces the manifest's separate-connection rule.

No imports from reagent.pipeline, reagent.experiments.ambiguity, runner, or policy.
No network calls, model calls, or .env access.
"""

import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from reagent.experiments.dbops.schema import SCHEMA_DDL


class DbopsStore:
    """
    Manages an isolated SQLite database for a single DBOPS trial.

    Usage pattern:
        store = DbopsStore.create_temp()
        try:
            with store.executing_connection() as conn:
                # run task steps
                ...
            # conn is now closed / transaction committed or rolled back
            with store.fresh_observer_connection() as conn:
                # independent read-only probe
                ...
        finally:
            store.cleanup()
    """

    def __init__(self, db_path: str) -> None:
        """
        db_path: Absolute path to the SQLite file, or ':memory:' for ephemeral use.
        Note: :memory: databases cannot share state between connections; use file-based
        paths for the oracle isolation model.
        """
        if db_path == ":memory:":
            raise ValueError(
                "DbopsStore requires a file-based database path so that "
                "the fresh observer connection can read committed state "
                "from a separate connection. Use DbopsStore.create_temp()."
            )
        self._db_path = db_path
        self._executing_conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    @classmethod
    def create_temp(cls, prefix: str = "dbops_trial_") -> "DbopsStore":
        """Create a fresh temp-file-backed store and apply the schema."""
        fd, path = tempfile.mkstemp(suffix=".db", prefix=prefix)
        os.close(fd)
        store = cls(path)
        store._apply_schema()
        return store

    @classmethod
    def create_at(cls, path: str) -> "DbopsStore":
        """Create a store at a specific path (for deterministic replay tests)."""
        store = cls(path)
        store._apply_schema()
        return store

    def _apply_schema(self) -> None:
        """Apply all SCHEMA_DDL CREATE TABLE statements."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=OFF")  # no FK constraints in this schema
            for _table_name, ddl in SCHEMA_DDL:
                conn.execute(ddl)
            conn.commit()
        finally:
            conn.close()

    def seed_initial_state(self, task_id: str) -> None:
        """
        Seed pre-existing rows required by specific tasks.

        Tasks that UPDATE a primary row (05, 06, 08, 10, 12 uses insert for primary;
        07 updates stock count) need a pre-existing row to update.
        Tasks 02, 04, 11 also need initial state for meaningful operation.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            _seed_task_initial_state(conn, task_id)
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def executing_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Yield a connection for task execution.
        The connection is explicitly closed after the context exits.
        """
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def fresh_observer_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Yield a new independent read-only connection.
        Caller must ensure the executing connection is already closed before calling.
        This enforces the manifest's fresh-connection oracle semantics.
        """
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA query_only=ON")
        try:
            yield conn
        finally:
            conn.close()

    @property
    def db_path(self) -> str:
        return self._db_path

    def cleanup(self) -> None:
        """Remove the temp database file. Safe to call even if file is gone."""
        try:
            if os.path.exists(self._db_path):
                os.remove(self._db_path)
            # WAL side files
            for suffix in ["-wal", "-shm"]:
                side = self._db_path + suffix
                if os.path.exists(side):
                    os.remove(side)
        except OSError:
            pass  # best-effort cleanup


def _seed_task_initial_state(conn: sqlite3.Connection, task_id: str) -> None:
    """
    Insert pre-existing rows required by a task so that UPDATE operations
    have a target row, and numeric tasks start from a known baseline.
    """
    if task_id == "task_04_customer_merge":
        # Source and target customers must exist
        conn.execute(
            "INSERT OR IGNORE INTO customers (customer_id, status) VALUES (?, ?)",
            ("CUST-8810", "ACTIVE"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO customers (customer_id, status) VALUES (?, ?)",
            ("CUST-1042", "ACTIVE"),
        )

    elif task_id == "task_05_order_cancellation":
        # Order must exist with ACTIVE status
        conn.execute(
            "INSERT OR IGNORE INTO orders (order_id, customer_id, status) VALUES (?, ?, ?)",
            ("ORD-5540", "CUST-7731", "ACTIVE"),
        )

    elif task_id == "task_06_subscription_upgrade":
        # Subscription must exist with BASIC tier
        conn.execute(
            "INSERT OR IGNORE INTO subscriptions (subscription_id, customer_id, tier) VALUES (?, ?, ?)",
            ("SUB-3301", "CUST-5512", "BASIC"),
        )

    elif task_id == "task_07_inventory_restock":
        # Inventory row must exist with initial stock count
        conn.execute(
            "INSERT OR IGNORE INTO inventory (sku_id, warehouse_id, stock_count) VALUES (?, ?, ?)",
            ("SKU-40821", "WH-05", 100),
        )

    elif task_id == "task_08_invoice_settlement":
        # Invoice must exist with PENDING status
        conn.execute(
            "INSERT OR IGNORE INTO invoices (invoice_id, payer_id, amount, status) VALUES (?, ?, ?, ?)",
            ("INV-7720", "PAY-3312", 1200.00, "PENDING"),
        )

    elif task_id == "task_10_account_deactivation":
        # Account must exist with ACTIVE status
        conn.execute(
            "INSERT OR IGNORE INTO accounts (account_id, customer_id, status) VALUES (?, ?, ?)",
            ("ACC-4490", "CUST-8801", "ACTIVE"),
        )

    elif task_id == "task_11_escrow_release":
        # Escrow account must exist with a balance
        conn.execute(
            "INSERT OR IGNORE INTO escrow_accounts (escrow_id, beneficiary_id, balance) VALUES (?, ?, ?)",
            ("ESC-0091", "BEN-2240", 10000.00),
        )

    # Tasks 01, 02, 03, 09, 12 have no pre-existing rows needed (pure INSERTs).
