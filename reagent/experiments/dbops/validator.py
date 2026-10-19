"""
validator.py — Contract validator for DBOPS V1.

Validates proposed contracts against:
  - Task prompt (grounding)
  - Declared schema (allowlist, columns)
  - Required field checks (request_id, amounts)
  - 8 adversarial rejection classes (as declared in the manifest)

Also implements a GoldContractEvaluator that evaluates GoldContract
against ACTUAL SQLite rows from the oracle, checking all field values.

ADVERSARIAL CLASSES REJECTED:
  CLASS_UNGROUNDED_LITERAL     — Value not in task prompt
  CLASS_STATUS_ONLY_ON_WRITE   — Contract only checks status field for a write op
  CLASS_TABLE_OUTSIDE_ALLOWLIST — Table not in task's allowed_tables
  CLASS_RAW_SQL_INJECTION       — Contract contains raw SQL predicate
  CLASS_VACUOUS_PRECONDITION    — Contract tests static precondition
  CLASS_WRONG_AMOUNT            — Numeric amount contradicts prompt
  CLASS_MISSING_REQUEST_ID      — Contract lacks required_request_id_field
  CLASS_UNDECLARED_EXTRA_FIELD  — Column not in declared table schema

No imports from reagent.pipeline, reagent.experiments.ambiguity, runner, or policy.
No network calls, model calls, or .env access.
No execution of raw SQL from proposed contracts.
"""

from dataclasses import dataclass
from typing import Any, Optional

from reagent.experiments.dbops.gold_contracts import (
    FieldCheck,
    FieldCheckSet,
    GoldContract,
    get_gold_contract,
)
from reagent.experiments.dbops.oracle import OracleObservation
from reagent.experiments.dbops.schema import TASK_ALLOWED_TABLES, ALL_TABLE_NAMES
from reagent.experiments.dbops.tasks import get_task
from reagent.experiments.dbops.types import VerdictType


# ── Known columns per table (schema-grounded) ─────────────────────────────────

TABLE_COLUMNS: dict[str, frozenset[str]] = {
    "refunds": frozenset({"id", "refund_id", "order_id", "customer_id", "amount", "status", "created_at"}),
    "refund_audit_log": frozenset({"id", "refund_id", "event_type", "amount", "logged_at"}),
    "wallet_transfer_ledger": frozenset({"id", "transfer_id", "source_acct", "dest_acct", "amount", "direction", "created_at"}),
    "wallet_balance_log": frozenset({"id", "transfer_id", "account_id", "delta", "logged_at"}),
    "seat_bookings": frozenset({"id", "booking_ref", "flight_id", "seat_num", "passenger_id", "status", "created_at"}),
    "booking_event_log": frozenset({"id", "booking_ref", "event_type", "flight_id", "seat_num", "logged_at"}),
    "customers": frozenset({"id", "customer_id", "merge_id", "merged_into", "status", "updated_at"}),
    "customer_merge_log": frozenset({"id", "merge_id", "source_id", "target_id", "logged_at"}),
    "orders": frozenset({"id", "order_id", "customer_id", "status", "updated_at"}),
    "order_event_log": frozenset({"id", "order_id", "cancellation_id", "event_type", "logged_at"}),
    "subscriptions": frozenset({"id", "subscription_id", "customer_id", "tier", "updated_at"}),
    "subscription_change_log": frozenset({"id", "upgrade_id", "subscription_id", "old_tier", "new_tier", "logged_at"}),
    "inventory": frozenset({"id", "sku_id", "warehouse_id", "stock_count", "updated_at"}),
    "inventory_restock_log": frozenset({"id", "restock_id", "sku_id", "warehouse_id", "quantity", "logged_at"}),
    "invoices": frozenset({"id", "invoice_id", "payer_id", "amount", "status", "updated_at"}),
    "invoice_payments": frozenset({"id", "invoice_id", "settlement_ref", "payer_id", "amount", "logged_at"}),
    "discount_redemptions": frozenset({"id", "application_id", "voucher_id", "basket_id", "discount_pct", "created_at"}),
    "basket_discount_log": frozenset({"id", "application_id", "basket_id", "voucher_id", "discount_pct", "logged_at"}),
    "accounts": frozenset({"id", "account_id", "customer_id", "status", "updated_at"}),
    "account_status_history": frozenset({"id", "deactivation_id", "account_id", "old_status", "new_status", "logged_at"}),
    "escrow_disbursement_journal": frozenset({"id", "release_id", "escrow_id", "beneficiary_id", "amount", "created_at"}),
    "escrow_accounts": frozenset({"id", "escrow_id", "beneficiary_id", "balance", "updated_at"}),
    "schema_migrations": frozenset({"id", "migration_id", "version", "database_target", "applied_at"}),
    "migration_audit_log": frozenset({"id", "migration_id", "version", "status", "logged_at"}),
}

# Status-only columns that are not sufficient for a write operation contract
STATUS_ONLY_COLUMNS = frozenset({"status", "result", "outcome", "response_code"})

# Raw SQL injection markers
RAW_SQL_MARKERS = ("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "DROP ", "CREATE ", "WHERE ", "FROM ")

# Values that must appear in prompts (never injected from outside)
GROUNDED_VALUES_BY_TASK: dict[str, frozenset] = {
    "task_01_refund_issue": frozenset({"REF-550291", "ORD-8821", "CUST-1042", "250.00", 250.00, 250}),
    "task_02_wallet_transfer": frozenset({"TXN-779043", "ACCT-2201", "ACCT-3390", "500.00", 500.00, 500}),
    "task_03_seat_booking": frozenset({"BKG-003341", "FL-2047", "14B", "PAX-9912"}),
    "task_04_customer_merge": frozenset({"MRG-002291", "CUST-8810", "CUST-1042"}),
    "task_05_order_cancellation": frozenset({"CXL-019882", "ORD-5540", "CUST-7731"}),
    "task_06_subscription_upgrade": frozenset({"UPG-008812", "SUB-3301", "CUST-5512", "BASIC", "PREMIUM"}),
    "task_07_inventory_restock": frozenset({"RST-110044", "SKU-40821", "WH-05", "200.0", 200.0, 200}),
    "task_08_invoice_settlement": frozenset({"STL-002981", "INV-7720", "PAY-3312", "1200.00", 1200.00, 1200}),
    "task_09_discount_apply": frozenset({"DIS-007712", "VCH-9901", "BSK-6612", "15.0", 15.0, 15}),
    "task_10_account_deactivation": frozenset({"DAC-003312", "ACC-4490", "CUST-8801", "INACTIVE"}),
    "task_11_escrow_release": frozenset({"REL-009921", "ESC-0091", "BEN-2240", "3500.00", 3500.00, 3500}),
    "task_12_schema_migration": frozenset({"MIG-000042", "DB-PROD", 42, "42"}),
}


# ── Validation result ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    rejection_class: Optional[str]      # None if valid
    rejection_reason: Optional[str]     # Human-readable detail
    checks_passed: int = 0
    checks_failed: int = 0


@dataclass
class EvaluationResult:
    """Result of running GoldContractEvaluator against actual db rows."""
    task_id: str
    verdict: VerdictType
    checks_passed: int
    checks_failed: int
    failed_checks: list[str]
    primary_state_duplicate: bool       # Extra primary rows found
    audit_duplicate: bool               # Extra secondary rows found
    notes: str = ""


# ── Contract Validator ─────────────────────────────────────────────────────────

class ContractValidator:
    """
    Validates a proposed GoldContract against grounding constraints.
    Never executes SQL from the proposed contract.
    """

    def validate(self, proposed: GoldContract, task_id: str) -> ValidationResult:
        """
        Run all 8 adversarial rejection class checks.
        Returns ValidationResult with first failing class, or valid=True.
        """
        task = get_task(task_id)
        allowed_tables = TASK_ALLOWED_TABLES.get(task_id, frozenset())
        grounded_values = GROUNDED_VALUES_BY_TASK.get(task_id, frozenset())

        all_checks = list(proposed.primary_step_contract.checks)
        for sc in proposed.secondary_step_contracts:
            all_checks.extend(sc.checks)

        # 1. CLASS_TABLE_OUTSIDE_ALLOWLIST
        for sc in proposed.all_step_contracts:
            if sc.table not in allowed_tables:
                return ValidationResult(
                    valid=False,
                    rejection_class="CLASS_TABLE_OUTSIDE_ALLOWLIST",
                    rejection_reason=f"Table '{sc.table}' is not in allowed_tables for {task_id}.",
                )
            for check in sc.checks:
                if check.table not in allowed_tables:
                    return ValidationResult(
                        valid=False,
                        rejection_class="CLASS_TABLE_OUTSIDE_ALLOWLIST",
                        rejection_reason=f"FieldCheck references table '{check.table}' not in allowed_tables.",
                    )

        # 2. CLASS_MISSING_REQUEST_ID
        all_checked_columns = {c.column for c in all_checks}
        if task.required_request_id_field not in all_checked_columns:
            return ValidationResult(
                valid=False,
                rejection_class="CLASS_MISSING_REQUEST_ID",
                rejection_reason=(
                    f"Contract does not bind required_request_id_field="
                    f"'{task.required_request_id_field}' for {task_id}."
                ),
            )

        # 3. CLASS_STATUS_ONLY_ON_WRITE
        non_status_columns = {c.column for c in all_checks} - STATUS_ONLY_COLUMNS
        if not non_status_columns:
            return ValidationResult(
                valid=False,
                rejection_class="CLASS_STATUS_ONLY_ON_WRITE",
                rejection_reason="Contract only asserts status/result columns for a state-changing write operation.",
            )

        # 4. CLASS_RAW_SQL_INJECTION
        for check in all_checks:
            val = str(check.expected_value).upper()
            for marker in RAW_SQL_MARKERS:
                if marker in val:
                    return ValidationResult(
                        valid=False,
                        rejection_class="CLASS_RAW_SQL_INJECTION",
                        rejection_reason=f"FieldCheck.expected_value contains raw SQL marker '{marker.strip()}'.",
                    )
            if str(check.comparison).upper() in ("RAW_SQL", "SQL", "EXECUTE"):
                return ValidationResult(
                    valid=False,
                    rejection_class="CLASS_RAW_SQL_INJECTION",
                    rejection_reason=f"FieldCheck.comparison='{check.comparison}' is a raw SQL execution.",
                )

        # 5. CLASS_UNGROUNDED_LITERAL
        for check in all_checks:
            val = check.expected_value
            if val is None:
                continue
            if isinstance(val, (int, float)):
                # Allow zero/one/null; reject unmapped numeric values
                if val not in (0, 1, None) and val not in grounded_values:
                    str_val = str(val)
                    if str_val not in grounded_values:
                        return ValidationResult(
                            valid=False,
                            rejection_class="CLASS_UNGROUNDED_LITERAL",
                            rejection_reason=(
                                f"Numeric value {val!r} in FieldCheck(column='{check.column}') "
                                f"is not present in the task prompt for {task_id}."
                            ),
                        )
            elif isinstance(val, str):
                if val not in grounded_values and val not in {"ISSUED", "CONFIRMED", "SETTLED", "MERGED",
                                                               "CANCELLED", "INACTIVE", "SUCCESS",
                                                               "DEBIT", "CREDIT", "ACTIVE"}:
                    # Allow common status strings, but reject arbitrary literals
                    if len(val) > 3:  # short strings like "OK" are allowed
                        return ValidationResult(
                            valid=False,
                            rejection_class="CLASS_UNGROUNDED_LITERAL",
                            rejection_reason=(
                                f"String value {val!r} in FieldCheck(column='{check.column}') "
                                f"is not present in the task prompt for {task_id}."
                            ),
                        )

        # 6. CLASS_WRONG_AMOUNT
        if task.prompt_numeric_amount is not None:
            for check in all_checks:
                if check.comparison in ("NUMERIC_EQUALS", "EQUALS") and isinstance(check.expected_value, (int, float)):
                    val = float(check.expected_value)
                    if abs(val - task.prompt_numeric_amount) > 0.001 and val not in (0, 1):
                        return ValidationResult(
                            valid=False,
                            rejection_class="CLASS_WRONG_AMOUNT",
                            rejection_reason=(
                                f"Contract asserts amount {val} but prompt specifies "
                                f"{task.prompt_numeric_amount} for {task_id}."
                            ),
                        )

        # 7. CLASS_VACUOUS_PRECONDITION
        # A contract that only checks pre-existing static values (before the operation)
        # is vacuous. We detect this by checking if ALL checks are against pk fields
        # that would be present even before the operation.
        # In our schema, pre-existing rows are for tasks 04, 05, 06, 07, 08, 10, 11.
        vacuous_cols = {"customer_id", "subscription_id", "order_id", "invoice_id",
                       "account_id", "escrow_id", "sku_id", "warehouse_id"}
        if non_status_columns and non_status_columns.issubset(vacuous_cols):
            return ValidationResult(
                valid=False,
                rejection_class="CLASS_VACUOUS_PRECONDITION",
                rejection_reason=(
                    "Contract only checks columns that are present before the operation "
                    "(pre-existing rows). This is a vacuous precondition test."
                ),
            )

        # 8. CLASS_UNDECLARED_EXTRA_FIELD
        for check in all_checks:
            table_cols = TABLE_COLUMNS.get(check.table)
            if table_cols is None:
                return ValidationResult(
                    valid=False,
                    rejection_class="CLASS_UNDECLARED_EXTRA_FIELD",
                    rejection_reason=f"Table '{check.table}' is not in the declared schema.",
                )
            if check.column not in table_cols:
                return ValidationResult(
                    valid=False,
                    rejection_class="CLASS_UNDECLARED_EXTRA_FIELD",
                    rejection_reason=(
                        f"Column '{check.column}' does not exist in table '{check.table}'. "
                        f"Declared columns: {sorted(table_cols)}."
                    ),
                )

        # All checks passed
        return ValidationResult(
            valid=True,
            rejection_class=None,
            rejection_reason=None,
            checks_passed=len(all_checks),
        )


# ── Gold Contract Evaluator ───────────────────────────────────────────────────

class GoldContractEvaluator:
    """
    Evaluates the gold contract for a task against actual committed database rows.

    Uses OracleObservation (which already has fresh-connection semantics).
    This is NOT the same as the policy arm's decision — it's the evaluator
    that runs AFTER every trial to measure truth.
    """

    def evaluate(
        self,
        task_id: str,
        observation: OracleObservation,
    ) -> EvaluationResult:
        """
        Evaluate the gold contract against the oracle observation.
        """
        contract = get_gold_contract(task_id)
        checks_passed = 0
        checks_failed = 0
        failed_checks = []

        # If oracle says UNKNOWN, we can't evaluate field-level checks
        if observation.verdict == VerdictType.UNKNOWN:
            return EvaluationResult(
                task_id=task_id,
                verdict=VerdictType.UNKNOWN,
                checks_passed=0,
                checks_failed=0,
                failed_checks=[],
                primary_state_duplicate=False,
                audit_duplicate=False,
                notes="Oracle returned UNKNOWN — cannot evaluate field-level checks.",
            )

        # Check primary step
        primary_row = observation.primary_row
        if primary_row is None and contract.primary_step_contract.is_required:
            # Primary absent: CONFIRMED_ABSENT or PARTIAL
            return EvaluationResult(
                task_id=task_id,
                verdict=observation.verdict,
                checks_passed=0,
                checks_failed=1,
                failed_checks=["primary_step_absent"],
                primary_state_duplicate=observation.duplicate_primary_count > 1,
                audit_duplicate=any(
                    v > 1 for v in observation.duplicate_secondary_counts.values()
                ),
                notes=f"Primary row absent. Oracle verdict: {observation.verdict}",
            )

        # Evaluate primary field checks
        if primary_row is not None:
            for check in contract.primary_step_contract.checks:
                result, note = self._evaluate_check(check, primary_row)
                if result:
                    checks_passed += 1
                else:
                    checks_failed += 1
                    failed_checks.append(f"{check.table}.{check.column}: {note}")

        # Evaluate secondary step checks
        for sc in contract.secondary_step_contracts:
            sec_row = observation.secondary_rows.get(sc.table)
            if sec_row is None and sc.is_required:
                checks_failed += 1
                failed_checks.append(f"{sc.table}: secondary row absent")
            elif sec_row is not None:
                for check in sc.checks:
                    result, note = self._evaluate_check(check, sec_row)
                    if result:
                        checks_passed += 1
                    else:
                        checks_failed += 1
                        failed_checks.append(f"{check.table}.{check.column}: {note}")

        # Determine evaluator verdict
        if checks_failed > 0:
            if observation.wrong_content_detected:
                verdict = VerdictType.WRONG_CONTENT
            elif observation.primary_present and not all(observation.secondary_present.values()):
                verdict = VerdictType.PARTIAL
            else:
                verdict = VerdictType.WRONG_CONTENT
        else:
            verdict = observation.verdict

        # Duplicate accounting
        primary_dup = observation.duplicate_primary_count > 1
        audit_dup = any(v > 1 for v in observation.duplicate_secondary_counts.values())

        return EvaluationResult(
            task_id=task_id,
            verdict=verdict,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            failed_checks=failed_checks,
            primary_state_duplicate=primary_dup,
            audit_duplicate=audit_dup,
            notes="",
        )

    def _evaluate_check(self, check: FieldCheck, row: dict) -> tuple[bool, str]:
        """Returns (passed, note)."""
        actual = row.get(check.column)
        expected = check.expected_value
        comparison = check.comparison

        if actual is None:
            return False, f"column '{check.column}' not found in row"

        if comparison == "EQUALS":
            ok = str(actual) == str(expected)
            return ok, f"expected={expected!r}, actual={actual!r}"
        elif comparison == "NUMERIC_EQUALS":
            try:
                ok = abs(float(actual) - float(expected)) < 0.001
                return ok, f"expected={expected}, actual={actual}"
            except (TypeError, ValueError):
                return False, f"non-numeric value for NUMERIC_EQUALS: {actual!r}"
        elif comparison == "CONTAINS":
            ok = str(expected) in str(actual)
            return ok, f"expected to contain {expected!r}, actual={actual!r}"
        elif comparison == "NOT_NULL":
            ok = actual is not None
            return ok, f"expected NOT_NULL, actual={actual!r}"
        elif comparison == "IN_SET":
            ok = actual in expected
            return ok, f"expected in {expected!r}, actual={actual!r}"
        elif comparison == "GREATER_THAN_ZERO":
            try:
                ok = float(actual) > 0
                return ok, f"expected >0, actual={actual}"
            except (TypeError, ValueError):
                return False, f"non-numeric: {actual!r}"
        else:
            return False, f"unknown comparison '{comparison}'"
