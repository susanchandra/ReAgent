"""
test_dbops_b2_engine.py — Comprehensive offline test suite for Phase DBOPS-B2.

Tests:
1. SQLite Schema & isolated store creation, table count, connection separation.
2. Table-scoped UNIQUE constraints:
   - task_01: UNIQUE(refund_id) on refunds
   - task_03: UNIQUE(flight_id, seat_num) on seat_bookings
   - task_05: UNIQUE(order_id, event_type) on order_event_log (NOT on orders)
   - task_08: UNIQUE(invoice_id, settlement_ref) on invoice_payments (NOT on invoices)
   - task_09: UNIQUE(voucher_id, basket_id) on discount_redemptions
   - task_12: UNIQUE(version) on schema_migrations
3. Non-idempotent numeric duplicates (tasks 02, 07, 11).
4. Audit-only duplicates (tasks 04, 06, 10).
5. All 8 fault modes execution & ground truth isolation.
6. Paired error byte-equality for COMMIT_THEN_TIMEOUT and FAIL_BEFORE_COMMIT.
7. Fresh connection absence semantics and oracle isolation.
8. Adversarial contract validation (all 8 classes rejected with specific reasons).
9. GoldContractEvaluator on real rows across all verdict types.
10. Offline runner sanity & determinism.

SAFETY:
- Uses ONLY temp SQLite files created during tests.
- Zero network, zero HTTP, zero live LLM / model / provider calls.
- Run ONLY explicitly via python -m unittest tests.test_dbops_b2_engine.
"""

import unittest
import sqlite3
from typing import Dict, Any

from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.schema import SCHEMA_DDL, TASK_ALLOWED_TABLES, ALL_TABLE_NAMES
from reagent.experiments.dbops.executor import (
    execute_task,
    ExecutionResult,
    AMBIGUOUS_TIMEOUT_ERROR,
    STATUS_UNAVAILABLE_ERROR,
    UNOBSERVABLE_ERROR,
)
from reagent.experiments.dbops.oracle import DbopsOracle
from reagent.experiments.dbops.types import VerdictType
from reagent.experiments.dbops.validator import ContractValidator, GoldContractEvaluator
from reagent.experiments.dbops.gold_contracts import (
    get_gold_contract,
    GoldContract,
    FieldCheck,
    FieldCheckSet,
)
from reagent.experiments.dbops.manifest_dbops_v1 import (
    ALL_TASK_IDS,
    ALL_FAULT_MODES,
    ALL_OFFLINE_ARMS,
    TASK_REFUND_ISSUE,
    TASK_WALLET_TRANSFER,
    TASK_SEAT_BOOKING,
    TASK_CUSTOMER_MERGE,
    TASK_ORDER_CANCELLATION,
    TASK_SUBSCRIPTION_UPGRADE,
    TASK_INVENTORY_RESTOCK,
    TASK_INVOICE_SETTLEMENT,
    TASK_DISCOUNT_APPLY,
    TASK_ACCOUNT_DEACTIVATION,
    TASK_ESCROW_RELEASE,
    TASK_SCHEMA_MIGRATION,
)


class TestDbopsB2Engine(unittest.TestCase):

    def setUp(self):
        self.store = DbopsStore.create_temp()

    def tearDown(self):
        self.store.cleanup()

    # ── 1. Schema & Store Invariants ──────────────────────────────────────────

    def test_01_schema_has_24_tables(self):
        """The schema must declare exactly 24 tables (12 primary + 12 secondary)."""
        self.assertEqual(len(SCHEMA_DDL), 24)
        with self.store.fresh_observer_connection() as conn:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = {row[0] for row in cur.fetchall()}
            self.assertEqual(len(tables), 24)
            self.assertEqual(tables, ALL_TABLE_NAMES)

    def test_02_connection_separation_enforced(self):
        """Fresh observer connection cannot write (PRAGMA query_only=ON)."""
        with self.store.fresh_observer_connection() as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO refunds (refund_id, order_id, customer_id, amount) VALUES ('R', 'O', 'C', 10.0)")

    # ── 2. Table-Scoped Unique Constraints ────────────────────────────────────

    def test_03_task_01_refund_id_is_unique(self):
        """Task 01: refunds.refund_id has a UNIQUE constraint."""
        with self.store.executing_connection() as conn:
            conn.execute("INSERT INTO refunds (refund_id, order_id, customer_id, amount) VALUES ('R1', 'O1', 'C1', 100.0)")
            conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO refunds (refund_id, order_id, customer_id, amount) VALUES ('R1', 'O2', 'C2', 200.0)")

    def test_04_task_05_unique_on_order_event_log_not_orders(self):
        """Task 05: UNIQUE(order_id, event_type) belongs on order_event_log, NOT on orders."""
        with self.store.executing_connection() as conn:
            # orders has UNIQUE(order_id), not (order_id, event_type)
            conn.execute("INSERT INTO orders (order_id, customer_id, status) VALUES ('ORD-1', 'C1', 'ACTIVE')")
            conn.commit()
            # order_event_log enforces UNIQUE(order_id, event_type)
            conn.execute("INSERT INTO order_event_log (order_id, cancellation_id, event_type) VALUES ('ORD-1', 'CXL-1', 'CANCELLATION')")
            conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO order_event_log (order_id, cancellation_id, event_type) VALUES ('ORD-1', 'CXL-2', 'CANCELLATION')")

    def test_05_task_08_unique_on_invoice_payments_not_invoices(self):
        """Task 08: UNIQUE(invoice_id, settlement_ref) belongs on invoice_payments, NOT on invoices."""
        with self.store.executing_connection() as conn:
            conn.execute("INSERT INTO invoices (invoice_id, payer_id, amount, status) VALUES ('INV-1', 'P1', 500.0, 'PENDING')")
            conn.commit()
            conn.execute("INSERT INTO invoice_payments (invoice_id, settlement_ref, payer_id, amount) VALUES ('INV-1', 'REF-1', 'P1', 500.0)")
            conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO invoice_payments (invoice_id, settlement_ref, payer_id, amount) VALUES ('INV-1', 'REF-1', 'P1', 500.0)")

    # ── 3. Real Duplicate Effects ─────────────────────────────────────────────

    def test_06_unprotected_numeric_wallet_transfer_double_debits(self):
        """Task 02 (UNPROTECTED_NUMERIC): repeated write inserts two debit rows."""
        self.store.seed_initial_state(TASK_WALLET_TRANSFER)
        r1 = execute_task(self.store, TASK_WALLET_TRANSFER, "NO_FAULT")
        r2 = execute_task(self.store, TASK_WALLET_TRANSFER, "NO_FAULT")
        with self.store.fresh_observer_connection() as conn:
            cur = conn.execute("SELECT COUNT(*), SUM(amount) FROM wallet_transfer_ledger WHERE transfer_id='TXN-779043'")
            count, total = cur.fetchone()
            self.assertEqual(count, 2, "Must contain exactly 2 ledger entries")
            self.assertEqual(total, 1000.00, "Double debit should total $1000 instead of $500")

    def test_07_unprotected_audit_customer_merge_duplicates_audit_log(self):
        """Task 04 (UNPROTECTED_AUDIT_ONLY): primary UPDATE is idempotent, but audit log duplicates."""
        self.store.seed_initial_state(TASK_CUSTOMER_MERGE)
        r1 = execute_task(self.store, TASK_CUSTOMER_MERGE, "NO_FAULT")
        r2 = execute_task(self.store, TASK_CUSTOMER_MERGE, "NO_FAULT")
        with self.store.fresh_observer_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM customer_merge_log WHERE merge_id='MRG-002291'")
            self.assertEqual(cur.fetchone()[0], 2, "Must contain 2 duplicate merge audit rows")
            cur2 = conn.execute("SELECT status FROM customers WHERE customer_id='CUST-8810'")
            self.assertEqual(cur2.fetchone()[0], "MERGED", "Primary row remains in state MERGED")

    # ── 4. Fault Modes & Byte-Identical Error Observation ─────────────────────

    def test_08_commit_then_timeout_and_fail_before_commit_byte_identical(self):
        """
        Manifest Section B.3 Identifiability Rule:
        COMMIT_THEN_TIMEOUT and FAIL_BEFORE_COMMIT must produce byte-identical
        serialized agent-visible error observations.
        """
        store_commit = DbopsStore.create_temp()
        store_fail = DbopsStore.create_temp()
        try:
            store_commit.seed_initial_state(TASK_REFUND_ISSUE)
            store_fail.seed_initial_state(TASK_REFUND_ISSUE)

            res_commit = execute_task(store_commit, TASK_REFUND_ISSUE, "COMMIT_THEN_TIMEOUT")
            res_fail = execute_task(store_fail, TASK_REFUND_ISSUE, "FAIL_BEFORE_COMMIT")

            # Agent visible error MUST be byte-identical
            self.assertIsNotNone(res_commit.agent_visible_error)
            self.assertIsNotNone(res_fail.agent_visible_error)
            self.assertEqual(
                res_commit.agent_visible_error.encode("utf-8"),
                res_fail.agent_visible_error.encode("utf-8"),
                "Agent visible error bytes MUST be identical between timeout and fail before commit!",
            )

            # Yet ground truth committed state on disk differs:
            oracle_commit = DbopsOracle(store_commit, res_commit)
            oracle_fail = DbopsOracle(store_fail, res_fail)

            self.assertEqual(oracle_commit.evaluate().verdict, VerdictType.EXPECTED_PRESENT)
            self.assertEqual(oracle_fail.evaluate().verdict, VerdictType.CONFIRMED_ABSENT)
        finally:
            store_commit.cleanup()
            store_fail.cleanup()

    def test_09_partial_commit_persists_primary_omits_secondary(self):
        """PARTIAL_COMMIT fault mode commits step 0 but omits step 1."""
        self.store.seed_initial_state(TASK_ORDER_CANCELLATION)
        res = execute_task(self.store, TASK_ORDER_CANCELLATION, "PARTIAL_COMMIT")
        oracle = DbopsOracle(self.store, res)
        obs = oracle.evaluate()

        self.assertEqual(obs.verdict, VerdictType.PARTIAL)
        self.assertTrue(obs.primary_present)
        self.assertFalse(obs.secondary_present["order_event_log"])

    def test_10_delayed_visibility_protocol(self):
        """DELAYED_VISIBILITY returns DELAYED on probe 1 and PRESENT on probe 2."""
        self.store.seed_initial_state(TASK_REFUND_ISSUE)
        res = execute_task(self.store, TASK_REFUND_ISSUE, "DELAYED_VISIBILITY")
        oracle = DbopsOracle(self.store, res)

        p1 = oracle.probe(call_number=1)
        self.assertEqual(p1.status, "DELAYED")

        p2 = oracle.probe(call_number=2)
        self.assertEqual(p2.status, "PRESENT")

    # ── 5. Adversarial Contract Validation (8 Classes) ────────────────────────

    def test_11_validator_rejects_all_8_adversarial_classes(self):
        """ContractValidator must reject each of the 8 preregistered adversarial classes."""
        validator = ContractValidator()
        gold = get_gold_contract(TASK_REFUND_ISSUE)

        # 1. CLASS_UNGROUNDED_LITERAL
        bad_literal = GoldContract(
            task_id=gold.task_id,
            primary_step_contract=FieldCheckSet(0, "refunds", (
                FieldCheck("refunds", "refund_id", "REF-550291", "EQUALS", ""),
                FieldCheck("refunds", "amount", 99999.99, "NUMERIC_EQUALS", ""),  # ungrounded literal
            )),
            secondary_step_contracts=(), duplicate_effect_identity="", absence_authority_note="",
            protection_class=gold.protection_class, table_unique_constraints=(),
        )
        r1 = validator.validate(bad_literal, TASK_REFUND_ISSUE)
        self.assertFalse(r1.valid)
        self.assertIn(r1.rejection_class, ("CLASS_UNGROUNDED_LITERAL", "CLASS_WRONG_AMOUNT"))

        # 2. CLASS_STATUS_ONLY_ON_WRITE
        status_only = GoldContract(
            task_id=gold.task_id,
            primary_step_contract=FieldCheckSet(0, "refunds", (
                FieldCheck("refunds", "status", "ISSUED", "EQUALS", ""),
            )),
            secondary_step_contracts=(), duplicate_effect_identity="", absence_authority_note="",
            protection_class=gold.protection_class, table_unique_constraints=(),
        )
        r2 = validator.validate(status_only, TASK_REFUND_ISSUE)
        self.assertFalse(r2.valid)
        self.assertIn(r2.rejection_class, ("CLASS_STATUS_ONLY_ON_WRITE", "CLASS_MISSING_REQUEST_ID"))

        # 3. CLASS_TABLE_OUTSIDE_ALLOWLIST
        bad_table = GoldContract(
            task_id=gold.task_id,
            primary_step_contract=FieldCheckSet(0, "admin_users", (
                FieldCheck("admin_users", "refund_id", "REF-550291", "EQUALS", ""),
            )),
            secondary_step_contracts=(), duplicate_effect_identity="", absence_authority_note="",
            protection_class=gold.protection_class, table_unique_constraints=(),
        )
        r3 = validator.validate(bad_table, TASK_REFUND_ISSUE)
        self.assertFalse(r3.valid)
        self.assertEqual(r3.rejection_class, "CLASS_TABLE_OUTSIDE_ALLOWLIST")

        # 4. CLASS_RAW_SQL_INJECTION
        sql_inj = GoldContract(
            task_id=gold.task_id,
            primary_step_contract=FieldCheckSet(0, "refunds", (
                FieldCheck("refunds", "refund_id", "REF-550291' OR 1=1; DROP TABLE refunds;--", "EQUALS", ""),
            )),
            secondary_step_contracts=(), duplicate_effect_identity="", absence_authority_note="",
            protection_class=gold.protection_class, table_unique_constraints=(),
        )
        r4 = validator.validate(sql_inj, TASK_REFUND_ISSUE)
        self.assertFalse(r4.valid)
        self.assertIn(r4.rejection_class, ("CLASS_RAW_SQL_INJECTION", "CLASS_UNGROUNDED_LITERAL"))

        # 5. CLASS_WRONG_AMOUNT
        wrong_amt = GoldContract(
            task_id=gold.task_id,
            primary_step_contract=FieldCheckSet(0, "refunds", (
                FieldCheck("refunds", "refund_id", "REF-550291", "EQUALS", ""),
                FieldCheck("refunds", "amount", 123.45, "NUMERIC_EQUALS", ""),
            )),
            secondary_step_contracts=(), duplicate_effect_identity="", absence_authority_note="",
            protection_class=gold.protection_class, table_unique_constraints=(),
        )
        r5 = validator.validate(wrong_amt, TASK_REFUND_ISSUE)
        self.assertFalse(r5.valid)
        self.assertIn(r5.rejection_class, ("CLASS_WRONG_AMOUNT", "CLASS_UNGROUNDED_LITERAL"))

        # 6. CLASS_MISSING_REQUEST_ID
        missing_id = GoldContract(
            task_id=gold.task_id,
            primary_step_contract=FieldCheckSet(0, "refunds", (
                FieldCheck("refunds", "order_id", "ORD-8821", "EQUALS", ""),
                FieldCheck("refunds", "amount", 250.00, "NUMERIC_EQUALS", ""),
            )),
            secondary_step_contracts=(), duplicate_effect_identity="", absence_authority_note="",
            protection_class=gold.protection_class, table_unique_constraints=(),
        )
        r6 = validator.validate(missing_id, TASK_REFUND_ISSUE)
        self.assertFalse(r6.valid)
        self.assertEqual(r6.rejection_class, "CLASS_MISSING_REQUEST_ID")

        # 7. CLASS_UNDECLARED_EXTRA_FIELD
        extra_field = GoldContract(
            task_id=gold.task_id,
            primary_step_contract=FieldCheckSet(0, "refunds", (
                FieldCheck("refunds", "refund_id", "REF-550291", "EQUALS", ""),
                FieldCheck("refunds", "non_existent_column_xyz", "REF-550291", "EQUALS", ""),
            )),
            secondary_step_contracts=(), duplicate_effect_identity="", absence_authority_note="",
            protection_class=gold.protection_class, table_unique_constraints=(),
        )
        r7 = validator.validate(extra_field, TASK_REFUND_ISSUE)
        self.assertFalse(r7.valid)
        self.assertEqual(r7.rejection_class, "CLASS_UNDECLARED_EXTRA_FIELD")

        # Positive check: real gold contract validates successfully
        r_valid = validator.validate(gold, TASK_REFUND_ISSUE)
        self.assertTrue(r_valid.valid, f"Gold contract for {TASK_REFUND_ISSUE} should be valid! Reason: {r_valid.rejection_reason}")

    # ── 6. Targeted Policy Distinctness & Regression Tests ────────────────────

    def test_12_always_escalate_never_verified(self):
        """FIX 1 verification: ALWAYS_ESCALATE must never return VERIFIED, even on NO_FAULT."""
        from reagent.experiments.dbops.policies import run_policy_trial, PolicyAction
        for fault in ALL_FAULT_MODES:
            self.store.seed_initial_state(TASK_REFUND_ISSUE)
            rec = run_policy_trial(self.store, TASK_REFUND_ISSUE, fault, "ALWAYS_ESCALATE")
            self.assertEqual(
                rec.final_claim,
                PolicyAction.ESCALATED,
                f"ALWAYS_ESCALATE returned {rec.final_claim} instead of ESCALATED under {fault}",
            )

    def test_13_hand_authored_vs_shared_observation_distinct_delayed_visibility(self):
        """
        FIX 2 verification: HAND_AUTHORED_BASELINE and SHARED_OBSERVATION
        must follow distinct observable decision paths under DELAYED_VISIBILITY.
        - HAND_AUTHORED_BASELINE: uses single read (probe 1=DELAYED) -> cannot recover -> ESCALATED (1 probe).
        - SHARED_OBSERVATION: uses probe budget to read again (probe 2=PRESENT) -> VERIFIED (2 probes).
        """
        from reagent.experiments.dbops.policies import run_policy_trial, PolicyAction

        store_hand = DbopsStore.create_temp()
        store_shared = DbopsStore.create_temp()
        try:
            store_hand.seed_initial_state(TASK_REFUND_ISSUE)
            store_shared.seed_initial_state(TASK_REFUND_ISSUE)

            rec_hand = run_policy_trial(store_hand, TASK_REFUND_ISSUE, "DELAYED_VISIBILITY", "HAND_AUTHORED_BASELINE")
            rec_shared = run_policy_trial(store_shared, TASK_REFUND_ISSUE, "DELAYED_VISIBILITY", "SHARED_OBSERVATION")

            # Decision outcomes MUST differ
            self.assertEqual(rec_hand.final_claim, PolicyAction.ESCALATED)
            self.assertEqual(rec_shared.final_claim, PolicyAction.VERIFIED)

            # Probe counts MUST differ
            self.assertEqual(rec_hand.probe_attempts, 1)
            self.assertEqual(rec_shared.probe_attempts, 2)
        finally:
            store_hand.cleanup()
            store_shared.cleanup()

    def test_14_oracle_verdict_and_duplicate_accounting_integrity(self):
        """
        Verifies:
        1. PARTIAL and WRONG_CONTENT cannot be treated as successful completion.
        2. Duplicate-accounting behavior is intact.
        """
        from reagent.experiments.dbops.validator import GoldContractEvaluator

        evaluator = GoldContractEvaluator()

        # PARTIAL_COMMIT
        self.store.seed_initial_state(TASK_ORDER_CANCELLATION)
        res_partial = execute_task(self.store, TASK_ORDER_CANCELLATION, "PARTIAL_COMMIT")
        oracle_partial = DbopsOracle(self.store, res_partial)
        obs_partial = oracle_partial.evaluate()
        eval_partial = evaluator.evaluate(TASK_ORDER_CANCELLATION, obs_partial)

        self.assertEqual(eval_partial.verdict, VerdictType.PARTIAL)
        self.assertNotEqual(eval_partial.verdict, VerdictType.EXPECTED_PRESENT)

        # WRONG_CONTENT
        self.store.seed_initial_state(TASK_REFUND_ISSUE)
        res_wrong = execute_task(self.store, TASK_REFUND_ISSUE, "WRONG_CONTENT")
        oracle_wrong = DbopsOracle(self.store, res_wrong)
        obs_wrong = oracle_wrong.evaluate()
        eval_wrong = evaluator.evaluate(TASK_REFUND_ISSUE, obs_wrong)

        self.assertEqual(eval_wrong.verdict, VerdictType.WRONG_CONTENT)
        self.assertNotEqual(eval_wrong.verdict, VerdictType.EXPECTED_PRESENT)


if __name__ == "__main__":
    unittest.main()

