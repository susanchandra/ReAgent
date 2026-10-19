"""
types.py — Shared enum types for the DBOPS experiment.

These enums are used by both tasks.py and gold_contracts.py.
They are deliberately separate from reagent.taxonomy to avoid coupling
the experiment layer to the core pipeline taxonomy.

No imports from reagent.pipeline, reagent.experiments.ambiguity, or any
runner/policy/fixture module are permitted here.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class TaskWorkflowType(str, Enum):
    """
    Commit structure for DBOPS V1 tasks.

    All 12 held-out tasks use MULTI_STEP_WORKFLOW: two or more application-level
    write steps (often separate tables) so PARTIAL_COMMIT is reachable when an
    early step commits and a required later step does not.

    A single SQLite transaction cannot partially commit internally; partial effects
    are modeled only across step/table boundaries, not inside one atomic transaction.
    """
    MULTI_STEP_WORKFLOW = "MULTI_STEP_WORKFLOW"


class TaskProtectionClass(str, Enum):
    """
    Three-way taxonomy of database-level protection against blind retry:

    UNPROTECTED_NUMERIC:
        Tasks where repeating the operation duplicates physical state,
        financial balances, or inventory counts (e.g. double debit,
        double stock increment, double escrow disbursement).
        No database unique constraint exists on the logical effect.

    UNPROTECTED_AUDIT_ONLY:
        Tasks where the primary row update is idempotent by value (e.g.
        updating status to INACTIVE or tier to PREMIUM), but repeating the
        operation corrupts audit/history logs with duplicate event rows.
        No database unique constraint prevents audit log pollution.

    KEY_PROTECTED:
        Tasks where a database UNIQUE constraint prevents duplicate logical
        effects. Note: the constraint is on the relevant business key
        (or composite key), not necessarily on request_id.
    """
    UNPROTECTED_NUMERIC = "UNPROTECTED_NUMERIC"
    UNPROTECTED_AUDIT_ONLY = "UNPROTECTED_AUDIT_ONLY"
    KEY_PROTECTED = "KEY_PROTECTED"


class IdempotencyClass(str, Enum):
    """
    Coarse two-way wrapper over TaskProtectionClass.

    KEY_PROTECTED means a table-scoped UNIQUE constraint exists on some
    business key. It does NOT mean request_id is unique.
    """
    KEY_PROTECTED = "KEY_PROTECTED"
    UNPROTECTED = "UNPROTECTED"


@dataclass(frozen=True)
class TableUniqueConstraint:
    """
    Explicit table-scoped database uniqueness constraint.

    table:   The exact table name that enforces this constraint.
    columns: The exact column tuple participating in the UNIQUE constraint.
    """
    table: str
    columns: Tuple[str, ...]


class ObservationAuthority(str, Enum):
    """
    Describes the evidence authority of an independent observation probe.

    FRESH_QUERY_ON_COMMITTED_STATE:
        The observer opens a fresh, separate SQLite connection after the
        executing connection is closed/committed/aborted. Reads from
        committed state on disk. Authority is sufficient to establish
        CONFIRMED_ABSENT when the row is not present.

    DELAYED_BY_PROTOCOL:
        The probe returns {\"status\": \"DELAYED\"} on the first call by
        deterministic injection protocol. A second probe executes a fresh
        query and returns the committed state. Authority is established
        only on the second probe.

    UNAVAILABLE:
        The observation surface is known to fail or not exist for this
        fault mode. Cannot establish CONFIRMED_ABSENT; must ESCALATE.

    UNOBSERVABLE:
        No queryable audit log or observation surface exists at all for
        this operation. Cannot establish any state; must ESCALATE.
    """
    FRESH_QUERY_ON_COMMITTED_STATE = "FRESH_QUERY_ON_COMMITTED_STATE"
    DELAYED_BY_PROTOCOL = "DELAYED_BY_PROTOCOL"
    UNAVAILABLE = "UNAVAILABLE"
    UNOBSERVABLE = "UNOBSERVABLE"


class VerdictType(str, Enum):
    """
    The range of possible gold contract evaluation outcomes.

    EXPECTED_PRESENT:
        All required postconditions for the operation are satisfied.
        The world state matches the frozen contract.

    CONFIRMED_ABSENT:
        The independent observer, using a fresh connection on committed
        state, confirms that no row matching the expected effect exists.
        Authority: requires FRESH_QUERY_ON_COMMITTED_STATE and an
        explicitly closed/aborted executing connection.
        MUST NOT be equated with ordinary ABSENT (stale read, locked query,
        or observation surface unavailability).

    UNKNOWN:
        The observer cannot establish committed state. Either the probe
        failed (STATUS_UNAVAILABLE), the observation surface does not
        exist (UNOBSERVABLE), or authority conditions for CONFIRMED_ABSENT
        are not met. Non-idempotent operations must ESCALATE from UNKNOWN.

    PARTIAL:
        The primary effect is EXPECTED_PRESENT but a required secondary
        effect (secondary table, audit row, ledger entry) is CONFIRMED_ABSENT.
        Applicable only to MULTI_STEP_WORKFLOW tasks under PARTIAL_COMMIT.
        Must produce DISCREPANCY and escalation, not VERIFIED.

    WRONG_CONTENT:
        A row is present but its field values contradict the frozen
        contract's required field checks (e.g. wrong amount, wrong
        destination). Must produce DISCREPANCY.
    """
    EXPECTED_PRESENT = "EXPECTED_PRESENT"
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"
    UNKNOWN = "UNKNOWN"
    PARTIAL = "PARTIAL"
    WRONG_CONTENT = "WRONG_CONTENT"
