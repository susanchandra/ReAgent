"""
tasks.py — Frozen task definitions for the DBOPS V1 benchmark.

AUTHORING RULES (from manifest K.2 / K.3):
- Tasks are frozen here (Phase B.1) BEFORE synthetic fixtures are authored.
- This file MUST NOT import from reagent.pipeline, reagent.experiments.ambiguity,
  or any runner/policy/fixture/oracle module.
- Task definitions encode structural metadata only. They contain no SQL,
  no execution logic, no fault injection, and no live model calls.
- Multi-step tasks enumerate their steps and secondary effects so that gold
  contracts can check ALL steps without referencing this file's internals.

PARTIAL_COMMIT SEMANTICS (all tasks are MULTI_STEP_WORKFLOW):
- Step 0 may commit to the primary table while step 1 aborts before the
  secondary/audit write. The oracle marks primary PRESENT and required
  secondary ABSENT → PARTIAL.
- Gold contracts check ALL required steps; primary presence alone is not VERIFIED.

PROTECTION-CLASS CONSTRAINT:
- The six UNPROTECTED_* tasks MUST NOT acquire table_unique_constraints.
  Duplicate-effect risk on those tasks is an explicit experimental variable.
"""

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

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
    TaskWorkflowType,
)


@dataclass(frozen=True)
class WorkflowStep:
    """
    Describes one commit-boundary step within a task.

    step_index:   0-based ordering. Lower index commits first.
    table:        Table that this step writes to.
    effect_kind:  Human-readable label (e.g. "INSERT_ROW", "UPDATE_STATUS").
    is_required:  Whether the gold contract requires this step to have
                  committed for a VERIFIED outcome. All steps with
                  is_required=True must be present; absence of any one
                  produces a PARTIAL verdict.
    identifying_columns: Columns the observer uses to locate this step's
                  committed row. Used by the gold contract probe.
    """
    step_index: int
    table: str
    effect_kind: str
    is_required: bool
    identifying_columns: Tuple[str, ...]


@dataclass(frozen=True)
class DbopsTask:
    """
    Immutable description of a single DBOPS task.

    Attributes
    ----------
    task_id:
        Canonical ID matching the manifest constant (e.g. "task_01_refund_issue").
    human_description:
        Natural language prompt that the agent receives. Contains all
        values that must appear in the frozen gold contract. Values not
        present in this description are not valid for contract predicates
        (CLASS_UNGROUNDED_LITERAL rejection).
    primary_table:
        The main table this operation writes to.
    secondary_tables:
        Additional tables written as part of the same logical operation
        (audit log, ledger, history). All 12 tasks in this benchmark model
        reachable PARTIAL_COMMIT states across distinct tables.
    workflow_type:
        MULTI_STEP_WORKFLOW (all 12 tasks model sequential commit boundaries).
    protection_class:
        TaskProtectionClass (KEY_PROTECTED, UNPROTECTED_NUMERIC, UNPROTECTED_AUDIT_ONLY).
    table_unique_constraints:
        Tuple of TableUniqueConstraint objects declaring the exact table and column(s)
        for any database UNIQUE constraint protecting the logical effect.
    steps:
        Ordered tuple of WorkflowStep objects describing each
        commit-boundary. Always has at least one step (the primary write).
        Gold contracts iterate over steps to build their postcondition set.
    allowed_tables:
        All table names this task is permitted to write to.
        A contract targeting a table outside this set is CLASS_TABLE_OUTSIDE_ALLOWLIST.
    required_request_id_field:
        The column name that serves as the operation trace ID (for
        CLASS_MISSING_REQUEST_ID validation). Every contract MUST bind this.
    prompt_numeric_amount:
        The exact numeric amount from the human_description, if applicable.
        A contract asserting a different amount is CLASS_WRONG_AMOUNT.
        None if the task has no numeric amount.
    prompt_source_id:
        Source entity identifier from the prompt (account, customer, etc.).
        None if not applicable.
    prompt_destination_id:
        Destination entity identifier from the prompt.
        None if not applicable.
    can_have_partial_commit:
        True when the task's PARTIAL_COMMIT fault mode is meaningful, i.e.
        when a primary write can commit while a required secondary write
        does not. This is True for all 12 tasks (manifest: 0 inapplicable).
    observation_authority_default:
        Default observation authority under normal probe conditions.
        Overridden to UNAVAILABLE / UNOBSERVABLE for specific fault modes.
    primary_state_duplicate_on_retry:
        True if a blind retry can corrupt primary business/financial state
        (extra money movement, extra stock, extra primary INSERT).
    audit_duplicate_on_retry:
        True if a blind retry can insert an extra audit/history/event row
        that is not blocked by a table-scoped UNIQUE constraint.
    notes:
        Any task-specific notes for implementors or evaluators.
    """
    task_id: str
    human_description: str
    primary_table: str
    secondary_tables: Tuple[str, ...]
    workflow_type: TaskWorkflowType
    protection_class: TaskProtectionClass
    table_unique_constraints: Tuple[TableUniqueConstraint, ...]
    steps: Tuple[WorkflowStep, ...]
    allowed_tables: FrozenSet[str]
    required_request_id_field: str
    prompt_numeric_amount: Optional[float]
    prompt_source_id: Optional[str]
    prompt_destination_id: Optional[str]
    can_have_partial_commit: bool
    observation_authority_default: ObservationAuthority
    primary_state_duplicate_on_retry: bool
    audit_duplicate_on_retry: bool
    notes: str = ""

    @property
    def idempotency_class(self) -> IdempotencyClass:
        """Coarse wrapper: KEY_PROTECTED vs UNPROTECTED. Not request_id uniqueness."""
        if self.protection_class == TaskProtectionClass.KEY_PROTECTED:
            return IdempotencyClass.KEY_PROTECTED
        return IdempotencyClass.UNPROTECTED

    @property
    def has_unique_constraint(self) -> bool:
        return len(self.table_unique_constraints) > 0

    @property
    def request_id_is_schema_unique(self) -> bool:
        """
        True only when required_request_id_field is the sole column of a
        declared UNIQUE constraint. Composite business keys do not qualify.
        """
        rid = self.required_request_id_field
        return any(tc.columns == (rid,) for tc in self.table_unique_constraints)


# ─── Task Definitions ─────────────────────────────────────────────────────────
#
# All 12 tasks are defined below in manifest order.
# Each task is assigned to a module-level constant and registered in
# ALL_TASKS and TASK_REGISTRY at the bottom of this file.
# ──────────────────────────────────────────────────────────────────────────────


# Task 01 — Refund Issue
# Protection: KEY_PROTECTED: UNIQUE(refund_id) on primary table `refunds`.
TASK_01_REFUND_ISSUE = DbopsTask(
    task_id=TASK_REFUND_ISSUE,
    human_description=(
        "Issue refund of $250.00 for order ORD-8821 to customer CUST-1042. "
        "Refund reference: REF-550291. Write the refund record to the refunds "
        "table and a corresponding entry to the refund_audit_log table."
    ),
    primary_table="refunds",
    secondary_tables=("refund_audit_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="refunds", columns=("refund_id",)),
    ),
    steps=(
        WorkflowStep(
            step_index=0,
            table="refunds",
            effect_kind="INSERT_ROW",
            is_required=True,
            identifying_columns=("refund_id",),
        ),
        WorkflowStep(
            step_index=1,
            table="refund_audit_log",
            effect_kind="INSERT_AUDIT_ENTRY",
            is_required=True,
            identifying_columns=("refund_id",),
        ),
    ),
    allowed_tables=frozenset({"refunds", "refund_audit_log"}),
    required_request_id_field="refund_id",
    prompt_numeric_amount=250.00,
    prompt_source_id="ORD-8821",
    prompt_destination_id="CUST-1042",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=True,
    notes=(
        "KEY_PROTECTED on refunds.refund_id (this is the request id). "
        "refund_audit_log has no UNIQUE constraint; retry can still duplicate audit rows."
    ),
)


# Task 02 — Wallet Transfer
# Protection: UNPROTECTED_NUMERIC: append-only ledger; blind retry causes double debit.
TASK_02_WALLET_TRANSFER = DbopsTask(
    task_id=TASK_WALLET_TRANSFER,
    human_description=(
        "Transfer $500.00 from wallet ACCT-2201 to wallet ACCT-3390. "
        "Transfer request ID: TXN-779043. Record the debit in wallet_transfer_ledger "
        "and the balance change in wallet_balance_log."
    ),
    primary_table="wallet_transfer_ledger",
    secondary_tables=("wallet_balance_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.UNPROTECTED_NUMERIC,
    table_unique_constraints=(),
    steps=(
        WorkflowStep(
            step_index=0,
            table="wallet_transfer_ledger",
            effect_kind="INSERT_DEBIT_ROW",
            is_required=True,
            identifying_columns=("transfer_id",),
        ),
        WorkflowStep(
            step_index=1,
            table="wallet_balance_log",
            effect_kind="INSERT_BALANCE_CHANGE",
            is_required=True,
            identifying_columns=("transfer_id",),
        ),
    ),
    allowed_tables=frozenset({"wallet_transfer_ledger", "wallet_balance_log"}),
    required_request_id_field="transfer_id",
    prompt_numeric_amount=500.00,
    prompt_source_id="ACCT-2201",
    prompt_destination_id="ACCT-3390",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=True,
    audit_duplicate_on_retry=True,
    notes=(
        "UNPROTECTED_NUMERIC: two identical ledger INSERTs create two debit rows "
        "causing double debit. ReAgent restraint is the only prevention. "
        "transfer_id is a trace field, not a UNIQUE column."
    ),
)


# Task 03 — Seat Booking
# Protection: KEY_PROTECTED: UNIQUE(flight_id, seat_num) on `seat_bookings`.
TASK_03_SEAT_BOOKING = DbopsTask(
    task_id=TASK_SEAT_BOOKING,
    human_description=(
        "Book seat 14B on flight FL-2047 for passenger PAX-9912. "
        "Booking reference: BKG-003341. Insert the booking into seat_bookings "
        "and log the reservation event in booking_event_log."
    ),
    primary_table="seat_bookings",
    secondary_tables=("booking_event_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="seat_bookings", columns=("flight_id", "seat_num")),
    ),
    steps=(
        WorkflowStep(
            step_index=0,
            table="seat_bookings",
            effect_kind="INSERT_BOOKING",
            is_required=True,
            identifying_columns=("flight_id", "seat_num"),
        ),
        WorkflowStep(
            step_index=1,
            table="booking_event_log",
            effect_kind="INSERT_EVENT",
            is_required=True,
            identifying_columns=("booking_ref",),
        ),
    ),
    allowed_tables=frozenset({"seat_bookings", "booking_event_log"}),
    required_request_id_field="booking_ref",
    prompt_numeric_amount=None,
    prompt_source_id="PAX-9912",
    prompt_destination_id="FL-2047",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=True,
    notes=(
        "KEY_PROTECTED by business key seat_bookings.(flight_id, seat_num), "
        "not booking_ref. booking_event_log is unconstrained."
    ),
)


# Task 04 — Customer Merge
# Protection: UNPROTECTED_AUDIT_ONLY: primary update is idempotent by value, but merge audit log duplicates.
TASK_04_CUSTOMER_MERGE = DbopsTask(
    task_id=TASK_CUSTOMER_MERGE,
    human_description=(
        "Merge customer CUST-8810 into CUST-1042. Merge operation ID: MRG-002291. "
        "Update the canonical customer record in customers table and log the merge "
        "operation in customer_merge_log."
    ),
    primary_table="customers",
    secondary_tables=("customer_merge_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.UNPROTECTED_AUDIT_ONLY,
    table_unique_constraints=(),
    steps=(
        WorkflowStep(
            step_index=0,
            table="customers",
            effect_kind="UPDATE_MERGE_FLAG",
            is_required=True,
            identifying_columns=("customer_id",),
        ),
        WorkflowStep(
            step_index=1,
            table="customer_merge_log",
            effect_kind="INSERT_MERGE_AUDIT",
            is_required=True,
            identifying_columns=("merge_id",),
        ),
    ),
    allowed_tables=frozenset({"customers", "customer_merge_log"}),
    required_request_id_field="merge_id",
    prompt_numeric_amount=None,
    prompt_source_id="CUST-8810",
    prompt_destination_id="CUST-1042",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=True,
    notes=(
        "UNPROTECTED_AUDIT_ONLY: double write appends a duplicate history entry in "
        "customer_merge_log. Primary customers UPDATE is idempotent by value. "
        "merge_id is not UNIQUE."
    ),
)


# Task 05 — Order Cancellation
# Protection: KEY_PROTECTED: UNIQUE(order_id, event_type) on secondary table `order_event_log`.
# Note: Primary table `orders` does NOT have event_type; constraint belongs to `order_event_log`.
TASK_05_ORDER_CANCELLATION = DbopsTask(
    task_id=TASK_ORDER_CANCELLATION,
    human_description=(
        "Cancel order ORD-5540 for customer CUST-7731. Cancellation request ID: "
        "CXL-019882. Update the order status in orders table to CANCELLED and "
        "insert a CANCELLATION event into order_event_log."
    ),
    primary_table="orders",
    secondary_tables=("order_event_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="order_event_log", columns=("order_id", "event_type")),
    ),
    steps=(
        WorkflowStep(
            step_index=0,
            table="orders",
            effect_kind="UPDATE_STATUS_CANCELLED",
            is_required=True,
            identifying_columns=("order_id",),
        ),
        WorkflowStep(
            step_index=1,
            table="order_event_log",
            effect_kind="INSERT_CANCELLATION_EVENT",
            is_required=True,
            identifying_columns=("order_id", "event_type"),
        ),
    ),
    allowed_tables=frozenset({"orders", "order_event_log"}),
    required_request_id_field="cancellation_id",
    prompt_numeric_amount=None,
    prompt_source_id="ORD-5540",
    prompt_destination_id="CUST-7731",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=False,
    notes=(
        "KEY_PROTECTED on order_event_log.(order_id, event_type), not on orders "
        "and not on cancellation_id. Primary orders UPDATE is idempotent by value. "
        "A UNIQUE violation on the event log is not proof the first attempt committed."
    ),
)


# Task 06 — Subscription Upgrade
# Protection: UNPROTECTED_AUDIT_ONLY: tier update is idempotent by value, but change log duplicates.
TASK_06_SUBSCRIPTION_UPGRADE = DbopsTask(
    task_id=TASK_SUBSCRIPTION_UPGRADE,
    human_description=(
        "Upgrade subscription SUB-3301 from tier BASIC to PREMIUM for customer "
        "CUST-5512. Upgrade request ID: UPG-008812. Update the subscription record "
        "in subscriptions table and log the tier change in subscription_change_log."
    ),
    primary_table="subscriptions",
    secondary_tables=("subscription_change_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.UNPROTECTED_AUDIT_ONLY,
    table_unique_constraints=(),
    steps=(
        WorkflowStep(
            step_index=0,
            table="subscriptions",
            effect_kind="UPDATE_TIER_PREMIUM",
            is_required=True,
            identifying_columns=("subscription_id",),
        ),
        WorkflowStep(
            step_index=1,
            table="subscription_change_log",
            effect_kind="INSERT_TIER_CHANGE_EVENT",
            is_required=True,
            identifying_columns=("upgrade_id",),
        ),
    ),
    allowed_tables=frozenset({"subscriptions", "subscription_change_log"}),
    required_request_id_field="upgrade_id",
    prompt_numeric_amount=None,
    prompt_source_id="SUB-3301",
    prompt_destination_id="CUST-5512",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=True,
    notes=(
        "UNPROTECTED_AUDIT_ONLY: double write results in two identical tier-change events "
        "in subscription_change_log — redundant audit log entries. "
        "upgrade_id is not UNIQUE."
    ),
)


# Task 07 — Inventory Restock
# Protection: UNPROTECTED_NUMERIC: blind retry increments stock again (+400 total).
TASK_07_INVENTORY_RESTOCK = DbopsTask(
    task_id=TASK_INVENTORY_RESTOCK,
    human_description=(
        "Restock 200 units of SKU-40821 in warehouse WH-05. "
        "Restock batch ID: RST-110044. Increment the stock count in inventory table "
        "and log the batch event in inventory_restock_log."
    ),
    primary_table="inventory",
    secondary_tables=("inventory_restock_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.UNPROTECTED_NUMERIC,
    table_unique_constraints=(),
    steps=(
        WorkflowStep(
            step_index=0,
            table="inventory",
            effect_kind="INCREMENT_STOCK_COUNT",
            is_required=True,
            identifying_columns=("sku_id", "warehouse_id"),
        ),
        WorkflowStep(
            step_index=1,
            table="inventory_restock_log",
            effect_kind="INSERT_RESTOCK_BATCH",
            is_required=True,
            identifying_columns=("restock_id",),
        ),
    ),
    allowed_tables=frozenset({"inventory", "inventory_restock_log"}),
    required_request_id_field="restock_id",
    prompt_numeric_amount=200.0,
    prompt_source_id="WH-05",
    prompt_destination_id="SKU-40821",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=True,
    audit_duplicate_on_retry=True,
    notes=(
        "UNPROTECTED_NUMERIC: double write increments stock count twice (double stock "
        "increment) and creates two restock log rows. restock_id is not UNIQUE."
    ),
)


# Task 08 — Invoice Settlement
# Protection: KEY_PROTECTED: UNIQUE(invoice_id, settlement_ref) on secondary table `invoice_payments`.
# Note: Primary table `invoices` does NOT have settlement_ref; constraint belongs to `invoice_payments`.
TASK_08_INVOICE_SETTLEMENT = DbopsTask(
    task_id=TASK_INVOICE_SETTLEMENT,
    human_description=(
        "Settle invoice INV-7720 with payment of $1200.00 from payer PAY-3312. "
        "Settlement reference: STL-002981. Mark the invoice SETTLED in the invoices "
        "table and record the payment in invoice_payments table."
    ),
    primary_table="invoices",
    secondary_tables=("invoice_payments",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="invoice_payments", columns=("invoice_id", "settlement_ref")),
    ),
    steps=(
        WorkflowStep(
            step_index=0,
            table="invoices",
            effect_kind="UPDATE_STATUS_SETTLED",
            is_required=True,
            identifying_columns=("invoice_id",),
        ),
        WorkflowStep(
            step_index=1,
            table="invoice_payments",
            effect_kind="INSERT_PAYMENT_RECORD",
            is_required=True,
            identifying_columns=("invoice_id", "settlement_ref"),
        ),
    ),
    allowed_tables=frozenset({"invoices", "invoice_payments"}),
    required_request_id_field="settlement_ref",
    prompt_numeric_amount=1200.00,
    prompt_source_id="PAY-3312",
    prompt_destination_id="INV-7720",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=False,
    notes=(
        "KEY_PROTECTED on invoice_payments.(invoice_id, settlement_ref), not on invoices. "
        "settlement_ref is unique only as part of that composite, not as a lone request_id. "
        "Primary invoices UPDATE is idempotent by value."
    ),
)


# Task 09 — Discount Apply
# Protection: KEY_PROTECTED: UNIQUE(voucher_id, basket_id) on `discount_redemptions`.
TASK_09_DISCOUNT_APPLY = DbopsTask(
    task_id=TASK_DISCOUNT_APPLY,
    human_description=(
        "Apply voucher VCH-9901 for a 15% discount to basket BSK-6612. "
        "Application request ID: DIS-007712. Insert the voucher redemption into "
        "discount_redemptions table and log the basket update in basket_discount_log."
    ),
    primary_table="discount_redemptions",
    secondary_tables=("basket_discount_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="discount_redemptions", columns=("voucher_id", "basket_id")),
    ),
    steps=(
        WorkflowStep(
            step_index=0,
            table="discount_redemptions",
            effect_kind="INSERT_REDEMPTION",
            is_required=True,
            identifying_columns=("voucher_id", "basket_id"),
        ),
        WorkflowStep(
            step_index=1,
            table="basket_discount_log",
            effect_kind="INSERT_DISCOUNT_LOG",
            is_required=True,
            identifying_columns=("application_id",),
        ),
    ),
    allowed_tables=frozenset({"discount_redemptions", "basket_discount_log"}),
    required_request_id_field="application_id",
    prompt_numeric_amount=15.0,   # 15% discount rate from prompt
    prompt_source_id="VCH-9901",
    prompt_destination_id="BSK-6612",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=True,
    notes=(
        "KEY_PROTECTED by business key discount_redemptions.(voucher_id, basket_id), "
        "not application_id. basket_discount_log is unconstrained."
    ),
)


# Task 10 — Account Deactivation
# Protection: UNPROTECTED_AUDIT_ONLY: status update is idempotent by value, but status history duplicates.
TASK_10_ACCOUNT_DEACTIVATION = DbopsTask(
    task_id=TASK_ACCOUNT_DEACTIVATION,
    human_description=(
        "Deactivate account ACC-4490 for customer CUST-8801. "
        "Deactivation request ID: DAC-003312. Set the account status to INACTIVE "
        "in the accounts table and record the status change in account_status_history."
    ),
    primary_table="accounts",
    secondary_tables=("account_status_history",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.UNPROTECTED_AUDIT_ONLY,
    table_unique_constraints=(),
    steps=(
        WorkflowStep(
            step_index=0,
            table="accounts",
            effect_kind="UPDATE_STATUS_INACTIVE",
            is_required=True,
            identifying_columns=("account_id",),
        ),
        WorkflowStep(
            step_index=1,
            table="account_status_history",
            effect_kind="INSERT_STATUS_CHANGE",
            is_required=True,
            identifying_columns=("deactivation_id",),
        ),
    ),
    allowed_tables=frozenset({"accounts", "account_status_history"}),
    required_request_id_field="deactivation_id",
    prompt_numeric_amount=None,
    prompt_source_id="ACC-4490",
    prompt_destination_id="CUST-8801",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=True,
    notes=(
        "UNPROTECTED_AUDIT_ONLY: double write creates duplicate INACTIVE status entries "
        "in account_status_history — duplicate audit entries. "
        "deactivation_id is not UNIQUE."
    ),
)


# Task 11 — Escrow Release
# Protection: UNPROTECTED_NUMERIC: blind retry disbursements duplicate fund release ($7000 vs $3500).
TASK_11_ESCROW_RELEASE = DbopsTask(
    task_id=TASK_ESCROW_RELEASE,
    human_description=(
        "Release $3500.00 from escrow account ESC-0091 to beneficiary BEN-2240. "
        "Release authorization ID: REL-009921. Disburse the funds by inserting into "
        "escrow_disbursement_journal and updating the escrow balance in escrow_accounts."
    ),
    primary_table="escrow_disbursement_journal",
    secondary_tables=("escrow_accounts",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.UNPROTECTED_NUMERIC,
    table_unique_constraints=(),
    steps=(
        WorkflowStep(
            step_index=0,
            table="escrow_disbursement_journal",
            effect_kind="INSERT_DISBURSEMENT",
            is_required=True,
            identifying_columns=("release_id",),
        ),
        WorkflowStep(
            step_index=1,
            table="escrow_accounts",
            effect_kind="UPDATE_ESCROW_BALANCE",
            is_required=True,
            identifying_columns=("escrow_id",),
        ),
    ),
    allowed_tables=frozenset({"escrow_disbursement_journal", "escrow_accounts"}),
    required_request_id_field="release_id",
    prompt_numeric_amount=3500.00,
    prompt_source_id="ESC-0091",
    prompt_destination_id="BEN-2240",
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=True,
    audit_duplicate_on_retry=False,
    notes=(
        "UNPROTECTED_NUMERIC: double write inserts two disbursement journal entries, "
        "causing double release from the escrow account. release_id is not UNIQUE. "
        "escrow_accounts is a balance UPDATE (primary-state numeric), not an unconstrained audit log."
    ),
)


# Task 12 — Schema Migration
# Protection: KEY_PROTECTED: UNIQUE(version) on `schema_migrations`.
TASK_12_SCHEMA_MIGRATION = DbopsTask(
    task_id=TASK_SCHEMA_MIGRATION,
    human_description=(
        "Apply schema migration version 42 to database DB-PROD. "
        "Migration run ID: MIG-000042. Record the applied migration in "
        "schema_migrations table and log the execution result in migration_audit_log."
    ),
    primary_table="schema_migrations",
    secondary_tables=("migration_audit_log",),
    workflow_type=TaskWorkflowType.MULTI_STEP_WORKFLOW,
    protection_class=TaskProtectionClass.KEY_PROTECTED,
    table_unique_constraints=(
        TableUniqueConstraint(table="schema_migrations", columns=("version",)),
    ),
    steps=(
        WorkflowStep(
            step_index=0,
            table="schema_migrations",
            effect_kind="INSERT_MIGRATION_RECORD",
            is_required=True,
            identifying_columns=("version",),
        ),
        WorkflowStep(
            step_index=1,
            table="migration_audit_log",
            effect_kind="INSERT_AUDIT_ENTRY",
            is_required=True,
            identifying_columns=("migration_id",),
        ),
    ),
    allowed_tables=frozenset({"schema_migrations", "migration_audit_log"}),
    required_request_id_field="migration_id",
    prompt_numeric_amount=None,
    prompt_source_id="DB-PROD",
    prompt_destination_id=None,
    can_have_partial_commit=True,
    observation_authority_default=ObservationAuthority.FRESH_QUERY_ON_COMMITTED_STATE,
    primary_state_duplicate_on_retry=False,
    audit_duplicate_on_retry=True,
    notes=(
        "KEY_PROTECTED by schema_migrations.(version), not migration_id. "
        "migration_audit_log is unconstrained."
    ),
)


# ─── Registry ─────────────────────────────────────────────────────────────────

ALL_TASKS: List[DbopsTask] = [
    TASK_01_REFUND_ISSUE,
    TASK_02_WALLET_TRANSFER,
    TASK_03_SEAT_BOOKING,
    TASK_04_CUSTOMER_MERGE,
    TASK_05_ORDER_CANCELLATION,
    TASK_06_SUBSCRIPTION_UPGRADE,
    TASK_07_INVENTORY_RESTOCK,
    TASK_08_INVOICE_SETTLEMENT,
    TASK_09_DISCOUNT_APPLY,
    TASK_10_ACCOUNT_DEACTIVATION,
    TASK_11_ESCROW_RELEASE,
    TASK_12_SCHEMA_MIGRATION,
]

TASK_REGISTRY: dict = {t.task_id: t for t in ALL_TASKS}


def get_task(task_id: str) -> DbopsTask:
    """Retrieve a task by its manifest task_id. Raises KeyError if not found."""
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task_id '{task_id}'. Valid IDs: {list(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task_id]


def get_key_protected_tasks() -> List[DbopsTask]:
    """Return all tasks protected by a database uniqueness constraint (exactly 6)."""
    return [t for t in ALL_TASKS if t.protection_class == TaskProtectionClass.KEY_PROTECTED]


def get_unprotected_numeric_tasks() -> List[DbopsTask]:
    """Return all tasks with unprotected numeric balances/quantities (exactly 3)."""
    return [t for t in ALL_TASKS if t.protection_class == TaskProtectionClass.UNPROTECTED_NUMERIC]


def get_unprotected_audit_tasks() -> List[DbopsTask]:
    """Return all tasks with unprotected audit logs (exactly 3)."""
    return [t for t in ALL_TASKS if t.protection_class == TaskProtectionClass.UNPROTECTED_AUDIT_ONLY]


def get_unprotected_tasks() -> List[DbopsTask]:
    """Return UNPROTECTED_NUMERIC and UNPROTECTED_AUDIT_ONLY tasks (exactly 6)."""
    return [t for t in ALL_TASKS if t.protection_class != TaskProtectionClass.KEY_PROTECTED]

