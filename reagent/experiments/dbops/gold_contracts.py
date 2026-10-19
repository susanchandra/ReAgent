"""
gold_contracts.py — Independent gold verification contracts for DBOPS V1.

AUTHORING RULES:
- Gold contracts are authored AFTER tasks.py and BEFORE synthetic fixtures.
- Contracts MUST NOT reference tool execution success claims (ActionResult.success).
- Contracts check independently observable world state only.
- This file MUST NOT import from reagent.pipeline, reagent.experiments.ambiguity,
  any policy, runner, or fixture module.
- The scoring oracle may read these contracts. Policy arms must not.

VERDICT SEMANTICS:
    EXPECTED_PRESENT  — all required postconditions satisfied.
    CONFIRMED_ABSENT  — fresh observer connection confirms no row. Requires
                        closed executing connection. Ordinary ABSENT (stale
                        read, lock failure) must NOT be escalated to this.
    UNKNOWN           — observer cannot establish committed state.
    PARTIAL           — primary step present, required secondary step absent.
    WRONG_CONTENT     — row present but field values contradict contract.

CONTRACT STRUCTURE:
Each GoldContract contains:
  - task_id: binds to the corresponding DbopsTask.
  - required_field_checks: List[FieldCheck] — all field values the evaluator
    must assert. The gold evaluator checks ALL of these.
  - primary_key_bindings: maps identifying column → expected value from prompt.
  - secondary_step_contracts: one FieldCheckSet per secondary table step.
  - duplicate_effect_identity: human description of what constitutes a
    duplicate effect for this task (for attribution reporting).
  - absence_authority_note: explains under what conditions CONFIRMED_ABSENT
    can be legitimately established for this task.
  - provenance: always "gold" — this is the human-reviewed reference contract.

These contracts are independent of any arm's policy. The evaluator runs them
against committed world state after every trial, regardless of arm decision.
"""

from dataclasses import dataclass, field as dc_field
from typing import Any, FrozenSet, List, Optional, Tuple

from reagent.experiments.dbops.manifest_dbops_v1 import (
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
from reagent.experiments.dbops.types import (
    IdempotencyClass,
    ObservationAuthority,
    TableUniqueConstraint,
    TaskProtectionClass,
    VerdictType,
)


@dataclass(frozen=True)
class FieldCheck:
    """
    A single expected field value assertion that the gold evaluator must verify.

    table:          Table to probe (must be in task's allowed_tables).
    column:         Column name to check.
    expected_value: The value as specified in the frozen prompt. Not derived
                    from tool output or execution result.
    comparison:     "EQUALS" | "NUMERIC_EQUALS" | "CONTAINS" | "NOT_NULL" |
                    "IN_SET" | "GREATER_THAN_ZERO"
    description:    Human-readable rationale for this check.
    is_required:    If True, failure on this check immediately marks verdict
                    WRONG_CONTENT or PARTIAL. If False, it is advisory.
    """
    table: str
    column: str
    expected_value: Any
    comparison: str
    description: str
    is_required: bool = True


@dataclass(frozen=True)
class FieldCheckSet:
    """
    All field checks for one workflow step (one table).

    step_index:  Matches WorkflowStep.step_index from tasks.py.
    table:       Table being checked.
    checks:      Tuple of FieldCheck for this step.
    is_required: If True, absence of any check-satisfying row is verdict PARTIAL.
    """
    step_index: int
    table: str
    checks: Tuple[FieldCheck, ...]
    is_required: bool = True


@dataclass(frozen=True)
class GoldContract:
    """
    Independent gold verification contract for a single DBOPS task.

    Attributes
    ----------
    task_id:
        Canonical task ID from the manifest.
    primary_step_contract:
        FieldCheckSet for step 0 (primary table write).
    secondary_step_contracts:
        One FieldCheckSet per secondary workflow step. May be empty only if
        the task has no secondary tables (none of the 12 tasks are in this case).
    duplicate_effect_identity:
        Human description of what constitutes a duplicate (second) effect.
        Used for attribution reporting: which duplicates were prevented by
        DB constraint vs ReAgent restraint.
    absence_authority_note:
        Conditions under which CONFIRMED_ABSENT can be established. Must
        specify that the executing connection is closed/aborted FIRST, and
        a fresh observer connection is used. For UNOBSERVABLE/STATUS_UNAVAILABLE
        fault modes, this field notes that CONFIRMED_ABSENT cannot be established.
    protection_class:
        TaskProtectionClass (KEY_PROTECTED, UNPROTECTED_NUMERIC, UNPROTECTED_AUDIT_ONLY).
    table_unique_constraints:
        Table-scoped unique constraints matching the task.
    provenance:
        Always "gold" for this file.
    notes:
        Optional additional context for evaluators.
    """
    task_id: str
    primary_step_contract: FieldCheckSet
    secondary_step_contracts: Tuple[FieldCheckSet, ...]
    duplicate_effect_identity: str
    absence_authority_note: str
    protection_class: TaskProtectionClass
    table_unique_constraints: Tuple[TableUniqueConstraint, ...]
    provenance: str = "gold"
    notes: str = ""

    @property
    def idempotency_class(self) -> IdempotencyClass:
        """Compatibility property mapping protection_class to IdempotencyClass."""
        if self.protection_class == TaskProtectionClass.KEY_PROTECTED:
            return IdempotencyClass.KEY_PROTECTED
        return IdempotencyClass.UNPROTECTED

    @property
    def all_step_contracts(self) -> Tuple[FieldCheckSet, ...]:
        """All step contracts in step_index order."""
        return (self.primary_step_contract,) + self.secondary_step_contracts

    @property
    def all_required_tables(self) -> FrozenSet[str]:
        """Set of tables the gold evaluator must inspect."""
        return frozenset(sc.table for sc in self.all_step_contracts)

    def verdict_for_partial(self) -> VerdictType:
        """Return the correct verdict when primary is present, secondary absent."""
        return VerdictType.PARTIAL

    def verdict_for_wrong_content(self) -> VerdictType:
        """Return the correct verdict when a row is present with wrong field values."""
        return VerdictType.WRONG_CONTENT


# ─── Gold Contract Definitions ────────────────────────────────────────────────


GOLD_01_REFUND_ISSUE = GoldContract(
    task_id=TASK_REFUND_ISSUE,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="refunds",
        is_required=True,
        checks=(
            FieldCheck(
                table="refunds", column="refund_id",
                expected_value="REF-550291",
                comparison="EQUALS",
                description="Refund reference ID matches prompt.",
            ),
            FieldCheck(
                table="refunds", column="order_id",
                expected_value="ORD-8821",
                comparison="EQUALS",
                description="Source order ID matches prompt.",
            ),
            FieldCheck(
                table="refunds", column="customer_id",
                expected_value="CUST-1042",
                comparison="EQUALS",
                description="Destination customer ID matches prompt.",
            ),
            FieldCheck(
                table="refunds", column="amount",
                expected_value=250.00,
                comparison="NUMERIC_EQUALS",
                description="Refund amount is exactly $250.00 as specified.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="refund_audit_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="refund_audit_log", column="refund_id",
                    expected_value="REF-550291",
                    comparison="EQUALS",
                    description="Audit log entry linked to correct refund_id.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: a second row in `refunds` with refund_id='REF-550291' "
        "is blocked by UNIQUE(refund_id) (request_id itself is unique). "
        "audit_duplicate: a second `refund_audit_log` row for the same refund_id is NOT "
        "schema-blocked. Attribution: DB constraint for primary; ReAgent restraint for audit."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: (1) executing connection closed/committed/aborted, "
        "(2) fresh observer connection opened, (3) SELECT on `refunds` WHERE refund_id='REF-550291' "
        "returns zero rows. Stale reads, locked queries, or STATUS_UNAVAILABLE do NOT qualify."
    ),
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="refunds", columns=("refund_id",)),
    ),
)


GOLD_02_WALLET_TRANSFER = GoldContract(
    task_id=TASK_WALLET_TRANSFER,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="wallet_transfer_ledger",
        is_required=True,
        checks=(
            FieldCheck(
                table="wallet_transfer_ledger", column="transfer_id",
                expected_value="TXN-779043",
                comparison="EQUALS",
                description="Transfer ID matches the prompt's request ID.",
            ),
            FieldCheck(
                table="wallet_transfer_ledger", column="source_account",
                expected_value="ACCT-2201",
                comparison="EQUALS",
                description="Source account matches prompt.",
            ),
            FieldCheck(
                table="wallet_transfer_ledger", column="destination_account",
                expected_value="ACCT-3390",
                comparison="EQUALS",
                description="Destination account matches prompt.",
            ),
            FieldCheck(
                table="wallet_transfer_ledger", column="amount",
                expected_value=500.00,
                comparison="NUMERIC_EQUALS",
                description="Transfer amount is exactly $500.00 as specified.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="wallet_balance_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="wallet_balance_log", column="transfer_id",
                    expected_value="TXN-779043",
                    comparison="EQUALS",
                    description="Balance log linked to the correct transfer_id.",
                ),
                FieldCheck(
                    table="wallet_balance_log", column="amount",
                    expected_value=500.00,
                    comparison="NUMERIC_EQUALS",
                    description="Balance log records same amount as ledger row.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: a SECOND row in `wallet_transfer_ledger` with "
        "transfer_id='TXN-779043' double-debits $500.00 from ACCT-2201. "
        "audit_duplicate: a second `wallet_balance_log` row for the same transfer_id. "
        "NO schema UNIQUE constraint exists. Attribution: ReAgent restraint only."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, "
        "fresh observer connection, SELECT on `wallet_transfer_ledger` "
        "WHERE transfer_id='TXN-779043' returns zero rows. "
        "For non-idempotent NON_UNIQUE task: CONFIRMED_ABSENT permits safe retry; "
        "UNKNOWN requires ESCALATE."
    ),
    protection_class=TaskProtectionClass.UNPROTECTED_NUMERIC,
    table_unique_constraints=(),
    notes=(
        "WRONG_CONTENT detection: if amount != 500.00 OR source_account != 'ACCT-2201', "
        "verdict is WRONG_CONTENT regardless of row presence."
    ),
)


GOLD_03_SEAT_BOOKING = GoldContract(
    task_id=TASK_SEAT_BOOKING,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="seat_bookings",
        is_required=True,
        checks=(
            FieldCheck(
                table="seat_bookings", column="booking_ref",
                expected_value="BKG-003341",
                comparison="EQUALS",
                description="Booking reference matches prompt.",
            ),
            FieldCheck(
                table="seat_bookings", column="flight_id",
                expected_value="FL-2047",
                comparison="EQUALS",
                description="Flight ID matches prompt.",
            ),
            FieldCheck(
                table="seat_bookings", column="seat_num",
                expected_value="14B",
                comparison="EQUALS",
                description="Seat number matches prompt exactly.",
            ),
            FieldCheck(
                table="seat_bookings", column="passenger_id",
                expected_value="PAX-9912",
                comparison="EQUALS",
                description="Passenger ID matches prompt.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="booking_event_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="booking_event_log", column="booking_ref",
                    expected_value="BKG-003341",
                    comparison="EQUALS",
                    description="Event log entry linked to booking_ref.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: a second `seat_bookings` row with "
        "flight_id='FL-2047' AND seat_num='14B' is blocked by UNIQUE(flight_id, seat_num) "
        "(business key, not booking_ref). "
        "audit_duplicate: a second `booking_event_log` row for booking_ref is NOT schema-blocked. "
        "Attribution: DB constraint for primary; ReAgent restraint for audit."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, "
        "fresh observer connection, SELECT on `seat_bookings` WHERE "
        "flight_id='FL-2047' AND seat_num='14B' returns zero rows."
    ),
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="seat_bookings", columns=("flight_id", "seat_num")),
    ),
)


GOLD_04_CUSTOMER_MERGE = GoldContract(
    task_id=TASK_CUSTOMER_MERGE,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="customers",
        is_required=True,
        checks=(
            FieldCheck(
                table="customers", column="customer_id",
                expected_value="CUST-1042",
                comparison="EQUALS",
                description="Canonical (surviving) customer ID matches destination.",
            ),
            FieldCheck(
                table="customers", column="merged_from_id",
                expected_value="CUST-8810",
                comparison="EQUALS",
                description="Source customer ID matches prompt.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="customer_merge_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="customer_merge_log", column="merge_id",
                    expected_value="MRG-002291",
                    comparison="EQUALS",
                    description="Merge log entry linked to the correct merge_id.",
                ),
                FieldCheck(
                    table="customer_merge_log", column="source_customer_id",
                    expected_value="CUST-8810",
                    comparison="EQUALS",
                    description="Merge log source customer matches prompt.",
                ),
                FieldCheck(
                    table="customer_merge_log", column="target_customer_id",
                    expected_value="CUST-1042",
                    comparison="EQUALS",
                    description="Merge log target customer matches prompt.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: customers UPDATE is idempotent by value (no extra primary row). "
        "audit_duplicate: a SECOND row in `customer_merge_log` with merge_id='MRG-002291'. "
        "NO schema UNIQUE constraint. Attribution: ReAgent restraint only."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT for the merge log requires: executing connection "
        "closed/aborted, fresh observer connection, SELECT on "
        "`customer_merge_log` WHERE merge_id='MRG-002291' returns zero rows. "
        "The primary customers UPDATE may be effectively idempotent in final "
        "state but the audit log duplication is the tracked duplicate effect. "
        "For non-idempotent NON_UNIQUE task: CONFIRMED_ABSENT permits safe retry; "
        "UNKNOWN requires ESCALATE."
    ),
    protection_class=TaskProtectionClass.UNPROTECTED_AUDIT_ONLY,
    table_unique_constraints=(),
)


GOLD_05_ORDER_CANCELLATION = GoldContract(
    task_id=TASK_ORDER_CANCELLATION,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="orders",
        is_required=True,
        checks=(
            FieldCheck(
                table="orders", column="order_id",
                expected_value="ORD-5540",
                comparison="EQUALS",
                description="Order ID matches prompt.",
            ),
            FieldCheck(
                table="orders", column="status",
                expected_value="CANCELLED",
                comparison="EQUALS",
                description="Order status must be CANCELLED, not any other value.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="order_event_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="order_event_log", column="order_id",
                    expected_value="ORD-5540",
                    comparison="EQUALS",
                    description="Event log entry linked to order_id.",
                ),
                FieldCheck(
                    table="order_event_log", column="event_type",
                    expected_value="CANCELLATION",
                    comparison="EQUALS",
                    description="Event type must be CANCELLATION.",
                ),
                FieldCheck(
                    table="order_event_log", column="cancellation_id",
                    expected_value="CXL-019882",
                    comparison="EQUALS",
                    description="Cancellation request ID matches prompt.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: orders.status UPDATE to CANCELLED is idempotent by value. "
        "audit_duplicate: a second CANCELLATION row in `order_event_log` with "
        "order_id='ORD-5540' AND event_type='CANCELLATION' is blocked by "
        "UNIQUE(order_id, event_type) on order_event_log, not on cancellation_id. "
        "Integrity error on retry is not proof the first attempt committed. "
        "Attribution: DB constraint on the event table."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, fresh "
        "observer connection, SELECT on `order_event_log` WHERE order_id='ORD-5540' "
        "AND event_type='CANCELLATION' returns zero rows."
    ),
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="order_event_log", columns=("order_id", "event_type")),
    ),
)


GOLD_06_SUBSCRIPTION_UPGRADE = GoldContract(
    task_id=TASK_SUBSCRIPTION_UPGRADE,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="subscriptions",
        is_required=True,
        checks=(
            FieldCheck(
                table="subscriptions", column="subscription_id",
                expected_value="SUB-3301",
                comparison="EQUALS",
                description="Subscription ID matches prompt.",
            ),
            FieldCheck(
                table="subscriptions", column="tier",
                expected_value="PREMIUM",
                comparison="EQUALS",
                description="Subscription tier must be PREMIUM (not BASIC or other).",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="subscription_change_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="subscription_change_log", column="upgrade_id",
                    expected_value="UPG-008812",
                    comparison="EQUALS",
                    description="Upgrade request ID matches prompt.",
                ),
                FieldCheck(
                    table="subscription_change_log", column="from_tier",
                    expected_value="BASIC",
                    comparison="EQUALS",
                    description="Source tier matches prompt (BASIC).",
                ),
                FieldCheck(
                    table="subscription_change_log", column="to_tier",
                    expected_value="PREMIUM",
                    comparison="EQUALS",
                    description="Destination tier matches prompt (PREMIUM).",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: subscriptions.tier UPDATE is idempotent by value. "
        "audit_duplicate: a SECOND row in `subscription_change_log` with "
        "upgrade_id='UPG-008812'. NO schema UNIQUE constraint. "
        "Attribution: ReAgent restraint only."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT for change log requires: executing connection "
        "closed/aborted, fresh observer, SELECT on `subscription_change_log` "
        "WHERE upgrade_id='UPG-008812' returns zero rows. "
        "For non-idempotent NON_UNIQUE task: CONFIRMED_ABSENT permits safe retry; "
        "UNKNOWN requires ESCALATE."
    ),
    protection_class=TaskProtectionClass.UNPROTECTED_AUDIT_ONLY,
    table_unique_constraints=(),
    notes=(
        "WRONG_CONTENT: if tier='BASIC' (unchanged) or tier='ENTERPRISE' "
        "(wrong destination), verdict is WRONG_CONTENT."
    ),
)


GOLD_07_INVENTORY_RESTOCK = GoldContract(
    task_id=TASK_INVENTORY_RESTOCK,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="inventory",
        is_required=True,
        checks=(
            FieldCheck(
                table="inventory", column="sku_id",
                expected_value="SKU-40821",
                comparison="EQUALS",
                description="SKU ID matches prompt.",
            ),
            FieldCheck(
                table="inventory", column="warehouse_id",
                expected_value="WH-05",
                comparison="EQUALS",
                description="Warehouse ID matches prompt.",
            ),
            FieldCheck(
                table="inventory", column="quantity_delta",
                expected_value=200,
                comparison="NUMERIC_EQUALS",
                description="Restocked quantity is exactly 200 units as specified.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="inventory_restock_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="inventory_restock_log", column="restock_id",
                    expected_value="RST-110044",
                    comparison="EQUALS",
                    description="Restock batch ID matches prompt.",
                ),
                FieldCheck(
                    table="inventory_restock_log", column="quantity",
                    expected_value=200,
                    comparison="NUMERIC_EQUALS",
                    description="Log records the same 200 units.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: a second stock increment (DOUBLE INCREMENT) of 200 units on `inventory` "
        "for SKU-40821/WH-05 (+400 total instead of +200). "
        "audit_duplicate: a SECOND `inventory_restock_log` row with restock_id='RST-110044'. "
        "NO schema UNIQUE constraint. Attribution: ReAgent restraint only."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, "
        "fresh observer, SELECT on `inventory_restock_log` WHERE "
        "restock_id='RST-110044' returns zero rows AND stock delta for "
        "SKU-40821/WH-05 has not increased. "
        "For non-idempotent NON_UNIQUE task: CONFIRMED_ABSENT permits safe retry; "
        "UNKNOWN requires ESCALATE."
    ),
    protection_class=TaskProtectionClass.UNPROTECTED_NUMERIC,
    table_unique_constraints=(),
    notes=(
        "WRONG_CONTENT: if quantity_delta != 200 (e.g. 100 or 400), verdict "
        "is WRONG_CONTENT. A double-applied increment is tracked as duplicate "
        "effect, not wrong content."
    ),
)


GOLD_08_INVOICE_SETTLEMENT = GoldContract(
    task_id=TASK_INVOICE_SETTLEMENT,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="invoices",
        is_required=True,
        checks=(
            FieldCheck(
                table="invoices", column="invoice_id",
                expected_value="INV-7720",
                comparison="EQUALS",
                description="Invoice ID matches prompt.",
            ),
            FieldCheck(
                table="invoices", column="status",
                expected_value="SETTLED",
                comparison="EQUALS",
                description="Invoice status must be SETTLED.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="invoice_payments",
            is_required=True,
            checks=(
                FieldCheck(
                    table="invoice_payments", column="invoice_id",
                    expected_value="INV-7720",
                    comparison="EQUALS",
                    description="Payment record references correct invoice_id.",
                ),
                FieldCheck(
                    table="invoice_payments", column="settlement_ref",
                    expected_value="STL-002981",
                    comparison="EQUALS",
                    description="Settlement reference matches prompt.",
                ),
                FieldCheck(
                    table="invoice_payments", column="amount",
                    expected_value=1200.00,
                    comparison="NUMERIC_EQUALS",
                    description="Payment amount is exactly $1200.00.",
                ),
                FieldCheck(
                    table="invoice_payments", column="payer_id",
                    expected_value="PAY-3312",
                    comparison="EQUALS",
                    description="Payer ID matches prompt.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: invoices.status UPDATE to SETTLED is idempotent by value. "
        "audit_duplicate: a second `invoice_payments` row with invoice_id='INV-7720' AND "
        "settlement_ref='STL-002981' is blocked by UNIQUE(invoice_id, settlement_ref) "
        "on invoice_payments, not on invoices and not as a lone request_id unique. "
        "Integrity error on retry is not proof the first attempt committed. "
        "Attribution: DB constraint on the payment table."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, "
        "fresh observer, SELECT on `invoice_payments` WHERE invoice_id='INV-7720' "
        "AND settlement_ref='STL-002981' returns zero rows."
    ),
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="invoice_payments", columns=("invoice_id", "settlement_ref")),
    ),
    notes=(
        "WRONG_CONTENT: if amount != 1200.00 or payer_id != 'PAY-3312', "
        "verdict is WRONG_CONTENT even if row is present."
    ),
)


GOLD_09_DISCOUNT_APPLY = GoldContract(
    task_id=TASK_DISCOUNT_APPLY,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="discount_redemptions",
        is_required=True,
        checks=(
            FieldCheck(
                table="discount_redemptions", column="voucher_id",
                expected_value="VCH-9901",
                comparison="EQUALS",
                description="Voucher ID matches prompt.",
            ),
            FieldCheck(
                table="discount_redemptions", column="basket_id",
                expected_value="BSK-6612",
                comparison="EQUALS",
                description="Basket ID matches prompt.",
            ),
            FieldCheck(
                table="discount_redemptions", column="discount_pct",
                expected_value=15.0,
                comparison="NUMERIC_EQUALS",
                description="Discount rate is exactly 15% as specified in prompt.",
            ),
            FieldCheck(
                table="discount_redemptions", column="application_id",
                expected_value="DIS-007712",
                comparison="EQUALS",
                description="Application request ID matches prompt.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="basket_discount_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="basket_discount_log", column="application_id",
                    expected_value="DIS-007712",
                    comparison="EQUALS",
                    description="Discount log linked to correct application_id.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: a second `discount_redemptions` row with "
        "voucher_id='VCH-9901' AND basket_id='BSK-6612' is blocked by "
        "UNIQUE(voucher_id, basket_id) (business key, not application_id). "
        "audit_duplicate: a second `basket_discount_log` row is NOT schema-blocked. "
        "Attribution: DB constraint for primary; ReAgent restraint for audit."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, "
        "fresh observer, SELECT on `discount_redemptions` WHERE "
        "voucher_id='VCH-9901' AND basket_id='BSK-6612' returns zero rows."
    ),
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="discount_redemptions", columns=("voucher_id", "basket_id")),
    ),
    notes=(
        "WRONG_CONTENT: if discount_pct != 15.0 (e.g. 10.0 or 20.0), "
        "verdict is WRONG_CONTENT."
    ),
)


GOLD_10_ACCOUNT_DEACTIVATION = GoldContract(
    task_id=TASK_ACCOUNT_DEACTIVATION,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="accounts",
        is_required=True,
        checks=(
            FieldCheck(
                table="accounts", column="account_id",
                expected_value="ACC-4490",
                comparison="EQUALS",
                description="Account ID matches prompt.",
            ),
            FieldCheck(
                table="accounts", column="status",
                expected_value="INACTIVE",
                comparison="EQUALS",
                description="Account status must be INACTIVE.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="account_status_history",
            is_required=True,
            checks=(
                FieldCheck(
                    table="account_status_history", column="deactivation_id",
                    expected_value="DAC-003312",
                    comparison="EQUALS",
                    description="Status history entry uses correct deactivation_id.",
                ),
                FieldCheck(
                    table="account_status_history", column="account_id",
                    expected_value="ACC-4490",
                    comparison="EQUALS",
                    description="History entry references the correct account.",
                ),
                FieldCheck(
                    table="account_status_history", column="new_status",
                    expected_value="INACTIVE",
                    comparison="EQUALS",
                    description="History entry records the correct new status.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: accounts.status UPDATE to INACTIVE is idempotent by value. "
        "audit_duplicate: a SECOND row in `account_status_history` with "
        "deactivation_id='DAC-003312'. NO schema UNIQUE constraint. "
        "Attribution: ReAgent restraint only."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, "
        "fresh observer, SELECT on `account_status_history` WHERE "
        "deactivation_id='DAC-003312' returns zero rows. "
        "Primary accounts UPDATE may be idempotent in final state but "
        "audit log duplication is the tracked effect. "
        "For non-idempotent NON_UNIQUE task: CONFIRMED_ABSENT permits safe retry; "
        "UNKNOWN requires ESCALATE."
    ),
    protection_class=TaskProtectionClass.UNPROTECTED_AUDIT_ONLY,
    table_unique_constraints=(),
    notes=(
        "WRONG_CONTENT: if accounts.status != 'INACTIVE' (e.g. still ACTIVE), "
        "verdict is WRONG_CONTENT."
    ),
)


GOLD_11_ESCROW_RELEASE = GoldContract(
    task_id=TASK_ESCROW_RELEASE,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="escrow_disbursement_journal",
        is_required=True,
        checks=(
            FieldCheck(
                table="escrow_disbursement_journal", column="release_id",
                expected_value="REL-009921",
                comparison="EQUALS",
                description="Release authorization ID matches prompt.",
            ),
            FieldCheck(
                table="escrow_disbursement_journal", column="escrow_id",
                expected_value="ESC-0091",
                comparison="EQUALS",
                description="Source escrow account matches prompt.",
            ),
            FieldCheck(
                table="escrow_disbursement_journal", column="beneficiary_id",
                expected_value="BEN-2240",
                comparison="EQUALS",
                description="Beneficiary matches prompt.",
            ),
            FieldCheck(
                table="escrow_disbursement_journal", column="amount",
                expected_value=3500.00,
                comparison="NUMERIC_EQUALS",
                description="Release amount is exactly $3500.00 as specified.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="escrow_accounts",
            is_required=True,
            checks=(
                FieldCheck(
                    table="escrow_accounts", column="escrow_id",
                    expected_value="ESC-0091",
                    comparison="EQUALS",
                    description="Escrow account balance row matches source escrow ID.",
                ),
                FieldCheck(
                    table="escrow_accounts", column="balance_delta",
                    expected_value=-3500.00,
                    comparison="NUMERIC_EQUALS",
                    description="Balance reduced by exactly $3500.00 (negative delta).",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: a SECOND `escrow_disbursement_journal` row with "
        "release_id='REL-009921' causes a second $3500.00 debit from ESC-0091 "
        "(double release = $7000.00). escrow_accounts balance UPDATE is also numeric. "
        "audit_duplicate: not modeled as a separate unconstrained history table. "
        "NO schema UNIQUE constraint. Attribution: ReAgent restraint only."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, "
        "fresh observer, SELECT on `escrow_disbursement_journal` WHERE "
        "release_id='REL-009921' returns zero rows. "
        "For non-idempotent NON_UNIQUE task: CONFIRMED_ABSENT permits safe retry; "
        "UNKNOWN requires ESCALATE."
    ),
    protection_class=TaskProtectionClass.UNPROTECTED_NUMERIC,
    table_unique_constraints=(),
    notes=(
        "WRONG_CONTENT: if amount != 3500.00 or beneficiary_id != 'BEN-2240', "
        "verdict is WRONG_CONTENT. This is one of the highest-value tasks: "
        "duplicate effect = $7000 disbursed vs $3500 intended."
    ),
)


GOLD_12_SCHEMA_MIGRATION = GoldContract(
    task_id=TASK_SCHEMA_MIGRATION,
    primary_step_contract=FieldCheckSet(
        step_index=0,
        table="schema_migrations",
        is_required=True,
        checks=(
            FieldCheck(
                table="schema_migrations", column="version",
                expected_value=42,
                comparison="NUMERIC_EQUALS",
                description="Migration version 42 matches prompt.",
            ),
            FieldCheck(
                table="schema_migrations", column="migration_id",
                expected_value="MIG-000042",
                comparison="EQUALS",
                description="Migration run ID matches prompt.",
            ),
        ),
    ),
    secondary_step_contracts=(
        FieldCheckSet(
            step_index=1,
            table="migration_audit_log",
            is_required=True,
            checks=(
                FieldCheck(
                    table="migration_audit_log", column="migration_id",
                    expected_value="MIG-000042",
                    comparison="EQUALS",
                    description="Audit log linked to the correct migration_id.",
                ),
                FieldCheck(
                    table="migration_audit_log", column="version",
                    expected_value=42,
                    comparison="NUMERIC_EQUALS",
                    description="Audit log version matches migration version.",
                ),
            ),
        ),
    ),
    duplicate_effect_identity=(
        "primary_state_duplicate: a second `schema_migrations` row with version=42 "
        "is blocked by UNIQUE(version) (not migration_id). "
        "audit_duplicate: a second `migration_audit_log` row is NOT schema-blocked. "
        "Attribution: DB constraint for primary; ReAgent restraint for audit."
    ),
    absence_authority_note=(
        "CONFIRMED_ABSENT requires: executing connection closed/aborted, "
        "fresh observer, SELECT on `schema_migrations` WHERE version=42 "
        "returns zero rows."
    ),
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="schema_migrations", columns=("version",)),
    ),
)


# ─── Registry ─────────────────────────────────────────────────────────────────

ALL_GOLD_CONTRACTS: List[GoldContract] = [
    GOLD_01_REFUND_ISSUE,
    GOLD_02_WALLET_TRANSFER,
    GOLD_03_SEAT_BOOKING,
    GOLD_04_CUSTOMER_MERGE,
    GOLD_05_ORDER_CANCELLATION,
    GOLD_06_SUBSCRIPTION_UPGRADE,
    GOLD_07_INVENTORY_RESTOCK,
    GOLD_08_INVOICE_SETTLEMENT,
    GOLD_09_DISCOUNT_APPLY,
    GOLD_10_ACCOUNT_DEACTIVATION,
    GOLD_11_ESCROW_RELEASE,
    GOLD_12_SCHEMA_MIGRATION,
]

GOLD_CONTRACT_REGISTRY: dict = {g.task_id: g for g in ALL_GOLD_CONTRACTS}


def get_gold_contract(task_id: str) -> GoldContract:
    """Retrieve the gold contract for a task. Raises KeyError if not found."""
    if task_id not in GOLD_CONTRACT_REGISTRY:
        raise KeyError(
            f"No gold contract for task_id '{task_id}'. "
            f"Available: {list(GOLD_CONTRACT_REGISTRY)}"
        )
    return GOLD_CONTRACT_REGISTRY[task_id]
