"""
test_dbops_tasks_and_gold.py — DBOPS-B1 test suite.

Tests are organized into six groups:
  A. Task structural invariants
  B. Gold contract structural invariants
  C. Task–contract binding integrity
  D. Adversarial contract rejection (8 distinct classes)
  E. PARTIAL_COMMIT semantics
  F. Unprotected-task duplicate-effect demonstration (protection matrix)

SAFETY:
- No SQLite databases are created or opened in this test suite.
- No live model calls, no network, no subprocess calls.
- No imports from reagent.pipeline, reagent.experiments.ambiguity,
  runner, fixture, or oracle modules.
- No modifications to any manifest, task, or gold contract file.

Run:
  $env:PYTHONPATH = "D:\\ReAgent"
  D:\\ReAgent\\reagent_env\\Scripts\\python.exe -m unittest tests.test_dbops_tasks_and_gold -v
"""

import unittest
from typing import List, Set

from reagent.experiments.dbops.manifest_dbops_v1 import (
    ALL_TASK_IDS,
    ALL_FAULT_MODES,
    ALL_OFFLINE_ARMS,
    ADVERSARIAL_REJECTION_CLASSES,
    SCHEMA_DEDUPLICATION_SUPPORT,
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
    TaskProtectionClass,
    TaskWorkflowType,
    VerdictType,
)
from reagent.experiments.dbops.tasks import (
    ALL_TASKS,
    TASK_REGISTRY,
    DbopsTask,
    WorkflowStep,
    get_task,
    get_key_protected_tasks,
    get_unprotected_numeric_tasks,
    get_unprotected_audit_tasks,
    get_unprotected_tasks,
)
from reagent.experiments.dbops.gold_contracts import (
    ALL_GOLD_CONTRACTS,
    GOLD_CONTRACT_REGISTRY,
    GoldContract,
    FieldCheck,
    FieldCheckSet,
    get_gold_contract,
)


def _step_identifying_columns_for_table(task: DbopsTask, table: str) -> set:
    """Columns declared on workflow steps for the given table."""
    cols: set = set()
    for step in task.steps:
        if step.table == table:
            cols.update(step.identifying_columns)
    return cols


# ─── A. Task Structural Invariants ────────────────────────────────────────────

class TestTaskStructuralInvariants(unittest.TestCase):

    def test_A01_exactly_12_tasks_defined(self):
        """Registry must contain exactly 12 tasks matching the manifest."""
        self.assertEqual(len(ALL_TASKS), 12)
        self.assertEqual(len(TASK_REGISTRY), 12)

    def test_A02_task_ids_match_manifest(self):
        """All task_ids in TASK_REGISTRY must exactly match ALL_TASK_IDS from manifest."""
        registry_ids = set(TASK_REGISTRY.keys())
        manifest_ids = set(ALL_TASK_IDS)
        self.assertEqual(registry_ids, manifest_ids,
                         f"Extra: {registry_ids - manifest_ids}, Missing: {manifest_ids - registry_ids}")

    def test_A03_all_tasks_are_frozen(self):
        """DbopsTask objects are frozen dataclasses — mutation must raise FrozenInstanceError."""
        import dataclasses
        for task in ALL_TASKS:
            with self.assertRaises(dataclasses.FrozenInstanceError,
                                   msg=f"Task {task.task_id} must be immutable"):
                setattr(task, "primary_table", "MUTATED")

    def test_A04_exactly_6_key_protected_6_unprotected(self):
        """Manifest requires exactly 6 KEY_PROTECTED and 6 unprotected tasks."""
        key_protected = get_key_protected_tasks()
        unprotected = get_unprotected_tasks()
        self.assertEqual(len(key_protected), 6,
                         f"Expected 6 KEY_PROTECTED tasks, got {len(key_protected)}: "
                         f"{[t.task_id for t in key_protected]}")
        self.assertEqual(len(unprotected), 6,
                         f"Expected 6 unprotected tasks, got {len(unprotected)}: "
                         f"{[t.task_id for t in unprotected]}")

    def test_A04b_protection_class_counts(self):
        """Three-way taxonomy: 3 numeric, 3 audit-only, 6 key-protected."""
        numeric = get_unprotected_numeric_tasks()
        audit = get_unprotected_audit_tasks()
        key_protected = get_key_protected_tasks()
        self.assertEqual(len(numeric), 3)
        self.assertEqual(len(audit), 3)
        self.assertEqual(len(key_protected), 6)
        self.assertEqual(len(numeric) + len(audit) + len(key_protected), 12)

    def test_A05_key_protected_tasks_match_manifest_schema_dedup(self):
        """KEY_PROTECTED task_ids must match manifest SCHEMA_DEDUPLICATION_SUPPORT True set."""
        manifest_unique_ids = {tid for tid, supported in SCHEMA_DEDUPLICATION_SUPPORT.items()
                               if supported is True}
        task_unique_ids = {t.task_id for t in get_key_protected_tasks()}
        self.assertEqual(task_unique_ids, manifest_unique_ids,
                         f"Mismatch: tasks={task_unique_ids}, manifest={manifest_unique_ids}")

    def test_A06_unprotected_tasks_have_no_table_unique_constraints(self):
        """Unprotected tasks MUST NOT declare table_unique_constraints."""
        for task in get_unprotected_tasks():
            self.assertEqual(
                task.table_unique_constraints, (),
                f"Task {task.task_id} is unprotected but has table_unique_constraints="
                f"{task.table_unique_constraints}."
            )

    def test_A07_key_protected_tasks_declare_table_scoped_constraints(self):
        """KEY_PROTECTED tasks must declare at least one TableUniqueConstraint on an allowed table."""
        for task in get_key_protected_tasks():
            self.assertGreater(
                len(task.table_unique_constraints), 0,
                f"Task {task.task_id} is KEY_PROTECTED but has no table_unique_constraints."
            )
            for tc in task.table_unique_constraints:
                self.assertIn(
                    tc.table, task.allowed_tables,
                    f"Task {task.task_id}: constraint table '{tc.table}' not in allowed_tables.",
                )
                self.assertGreater(len(tc.columns), 0)
                step_cols = _step_identifying_columns_for_table(task, tc.table)
                for col in tc.columns:
                    self.assertIn(
                        col, step_cols,
                        f"Task {task.task_id}: constrained column '{col}' on '{tc.table}' "
                        f"is not declared on any workflow step for that table (cols={step_cols}).",
                    )

    def test_A07b_all_tasks_use_multi_step_workflow(self):
        """DBOPS V1 defines all 12 tasks as MULTI_STEP_WORKFLOW."""
        self.assertEqual(
            list(TaskWorkflowType),
            [TaskWorkflowType.MULTI_STEP_WORKFLOW],
            "SINGLE_STEP_ATOMIC is unused; do not revive it to satisfy tests.",
        )
        for task in ALL_TASKS:
            self.assertEqual(
                task.workflow_type, TaskWorkflowType.MULTI_STEP_WORKFLOW,
                f"Task {task.task_id} must be MULTI_STEP_WORKFLOW.",
            )

    def test_A07c_idempotency_class_does_not_alias_request_id_uniqueness(self):
        """KEY_PROTECTED is not UNIQUE_REQUEST_ID; request_id uniqueness is separate."""
        member_names = set(IdempotencyClass.__members__)
        self.assertNotIn("UNIQUE_REQUEST_ID", member_names)
        self.assertNotIn("NON_UNIQUE_REQUEST_ID", member_names)
        self.assertEqual(member_names, {"KEY_PROTECTED", "UNPROTECTED"})

    def test_A21_task_05_unique_on_order_event_log(self):
        """Order cancellation: UNIQUE(order_id, event_type) on secondary order_event_log."""
        task = get_task(TASK_ORDER_CANCELLATION)
        self.assertEqual(len(task.table_unique_constraints), 1)
        tc = task.table_unique_constraints[0]
        self.assertEqual(tc.table, "order_event_log")
        self.assertEqual(tc.columns, ("order_id", "event_type"))
        self.assertEqual(task.required_request_id_field, "cancellation_id")

    def test_A22_task_08_unique_on_invoice_payments(self):
        """Invoice settlement: UNIQUE(invoice_id, settlement_ref) on secondary invoice_payments."""
        task = get_task(TASK_INVOICE_SETTLEMENT)
        self.assertEqual(len(task.table_unique_constraints), 1)
        tc = task.table_unique_constraints[0]
        self.assertEqual(tc.table, "invoice_payments")
        self.assertEqual(tc.columns, ("invoice_id", "settlement_ref"))
        self.assertEqual(task.required_request_id_field, "settlement_ref")
        self.assertFalse(task.request_id_is_schema_unique)

    def test_A23_only_refund_issue_has_request_id_as_sole_unique_column(self):
        """Do not call a business-key UNIQUE a request_id unique constraint."""
        unique_request_id_tasks = [
            t for t in ALL_TASKS if t.request_id_is_schema_unique
        ]
        self.assertEqual(
            [t.task_id for t in unique_request_id_tasks],
            [TASK_REFUND_ISSUE],
        )
        refund = get_task(TASK_REFUND_ISSUE)
        self.assertEqual(refund.required_request_id_field, "refund_id")
        self.assertEqual(refund.table_unique_constraints[0].columns, ("refund_id",))

    def test_A24_protection_matrix_primary_vs_audit_duplicate_risk(self):
        """
        Frozen duplicate-risk flags from actual table-scoped metadata.
        KEY_PROTECTED does not imply zero audit_duplicate risk.
        """
        expected = {
            TASK_REFUND_ISSUE: (TaskProtectionClass.KEY_PROTECTED, False, True),
            TASK_WALLET_TRANSFER: (TaskProtectionClass.UNPROTECTED_NUMERIC, True, True),
            TASK_SEAT_BOOKING: (TaskProtectionClass.KEY_PROTECTED, False, True),
            TASK_CUSTOMER_MERGE: (TaskProtectionClass.UNPROTECTED_AUDIT_ONLY, False, True),
            TASK_ORDER_CANCELLATION: (TaskProtectionClass.KEY_PROTECTED, False, False),
            TASK_SUBSCRIPTION_UPGRADE: (TaskProtectionClass.UNPROTECTED_AUDIT_ONLY, False, True),
            TASK_INVENTORY_RESTOCK: (TaskProtectionClass.UNPROTECTED_NUMERIC, True, True),
            TASK_INVOICE_SETTLEMENT: (TaskProtectionClass.KEY_PROTECTED, False, False),
            TASK_DISCOUNT_APPLY: (TaskProtectionClass.KEY_PROTECTED, False, True),
            TASK_ACCOUNT_DEACTIVATION: (TaskProtectionClass.UNPROTECTED_AUDIT_ONLY, False, True),
            TASK_ESCROW_RELEASE: (TaskProtectionClass.UNPROTECTED_NUMERIC, True, False),
            TASK_SCHEMA_MIGRATION: (TaskProtectionClass.KEY_PROTECTED, False, True),
        }
        self.assertEqual(set(expected), set(TASK_REGISTRY))
        for tid, (cls, primary, audit) in expected.items():
            task = get_task(tid)
            self.assertEqual(task.protection_class, cls, tid)
            self.assertEqual(task.primary_state_duplicate_on_retry, primary, tid)
            self.assertEqual(task.audit_duplicate_on_retry, audit, tid)
            identity = get_gold_contract(tid).duplicate_effect_identity.lower()
            self.assertIn("primary_state_duplicate", identity, tid)
            self.assertIn("audit_duplicate", identity, tid)

    def test_A08_all_tasks_have_at_least_one_step(self):
        """Every task must declare at least one WorkflowStep."""
        for task in ALL_TASKS:
            self.assertGreater(len(task.steps), 0,
                               f"Task {task.task_id} has no workflow steps.")

    def test_A09_all_tasks_have_at_least_one_secondary_table(self):
        """All 12 tasks model PARTIAL_COMMIT across two tables; secondary_tables must be non-empty."""
        for task in ALL_TASKS:
            self.assertGreater(
                len(task.secondary_tables), 0,
                f"Task {task.task_id} has no secondary_tables. PARTIAL_COMMIT requires "
                f"at least one secondary write target."
            )

    def test_A10_all_tasks_can_have_partial_commit(self):
        """Manifest declares 0 inapplicable configurations; every task must support PARTIAL_COMMIT."""
        for task in ALL_TASKS:
            self.assertTrue(
                task.can_have_partial_commit,
                f"Task {task.task_id}.can_have_partial_commit is False. "
                f"Manifest requires 0 inapplicable configurations."
            )

    def test_A11_step_indices_are_contiguous_zero_based(self):
        """WorkflowStep indices must start at 0 and be contiguous."""
        for task in ALL_TASKS:
            indices = [step.step_index for step in task.steps]
            expected = list(range(len(task.steps)))
            self.assertEqual(sorted(indices), expected,
                             f"Task {task.task_id} step indices {indices} are not contiguous "
                             f"from 0. Expected {expected}.")

    def test_A12_step_zero_table_matches_primary_table(self):
        """The first step (step_index=0) must write to the primary_table."""
        for task in ALL_TASKS:
            step_0 = next(s for s in task.steps if s.step_index == 0)
            self.assertEqual(
                step_0.table, task.primary_table,
                f"Task {task.task_id}: step 0 table '{step_0.table}' != "
                f"primary_table '{task.primary_table}'."
            )

    def test_A13_all_step_tables_in_allowed_tables(self):
        """Every WorkflowStep table must be in allowed_tables."""
        for task in ALL_TASKS:
            for step in task.steps:
                self.assertIn(
                    step.table, task.allowed_tables,
                    f"Task {task.task_id} step {step.step_index} table '{step.table}' "
                    f"is not in allowed_tables={task.allowed_tables}."
                )

    def test_A14_required_request_id_field_is_non_empty(self):
        """Every task must declare a required_request_id_field for tracing."""
        for task in ALL_TASKS:
            self.assertTrue(
                task.required_request_id_field.strip(),
                f"Task {task.task_id} has empty required_request_id_field."
            )

    def test_A15_get_task_raises_for_unknown_id(self):
        """get_task() must raise KeyError for an unrecognized task_id."""
        with self.assertRaises(KeyError):
            get_task("task_99_does_not_exist")

    def test_A16_get_task_returns_correct_task(self):
        """get_task() must return the correct task object for every manifest ID."""
        for tid in ALL_TASK_IDS:
            task = get_task(tid)
            self.assertEqual(task.task_id, tid)

    def test_A17_specific_task_manifest_ids(self):
        """Spot-check: manifest task ID constants map to correct task objects."""
        self.assertEqual(get_task(TASK_REFUND_ISSUE).primary_table, "refunds")
        self.assertEqual(get_task(TASK_WALLET_TRANSFER).primary_table, "wallet_transfer_ledger")
        self.assertEqual(get_task(TASK_ESCROW_RELEASE).primary_table, "escrow_disbursement_journal")
        self.assertEqual(get_task(TASK_SCHEMA_MIGRATION).primary_table, "schema_migrations")

    def test_A18_human_description_contains_required_request_id_value(self):
        """Spot-check: prompt values used in gold contracts must appear in human_description."""
        spot_checks = [
            (TASK_REFUND_ISSUE, "REF-550291"),
            (TASK_WALLET_TRANSFER, "TXN-779043"),
            (TASK_SEAT_BOOKING, "BKG-003341"),
            (TASK_INVOICE_SETTLEMENT, "STL-002981"),
            (TASK_ESCROW_RELEASE, "REL-009921"),
        ]
        for tid, expected_ref in spot_checks:
            task = get_task(tid)
            self.assertIn(
                expected_ref, task.human_description,
                f"Task {tid} human_description does not contain request ID '{expected_ref}'. "
                f"Gold contracts grounding this value would be CLASS_UNGROUNDED_LITERAL."
            )

    def test_A19_numeric_amounts_match_human_description(self):
        """Spot-check: prompt_numeric_amount must appear literally in human_description."""
        amount_tasks = [
            (TASK_REFUND_ISSUE, 250.00, "250.00"),
            (TASK_WALLET_TRANSFER, 500.00, "500.00"),
            (TASK_INVOICE_SETTLEMENT, 1200.00, "1200.00"),
            (TASK_ESCROW_RELEASE, 3500.00, "3500.00"),
        ]
        for tid, expected_amount, amount_str in amount_tasks:
            task = get_task(tid)
            self.assertEqual(task.prompt_numeric_amount, expected_amount,
                             f"Task {tid} prompt_numeric_amount should be {expected_amount}.")
            self.assertIn(amount_str, task.human_description,
                          f"Task {tid} human_description does not contain amount '{amount_str}'.")

    def test_A20_no_task_imports_pipeline(self):
        """tasks.py must not have import statements for reagent.pipeline or ambiguity."""
        import ast, inspect
        import reagent.experiments.dbops.tasks as tasks_mod
        src = inspect.getsource(tasks_mod)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(
                    node.module == "reagent.pipeline"
                    or node.module.startswith("reagent.pipeline.")
                    or node.module.startswith("reagent.experiments.ambiguity"),
                    f"tasks.py ImportFrom prohibited module: '{node.module}'."
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name == "reagent.pipeline"
                        or alias.name.startswith("reagent.pipeline.")
                        or alias.name.startswith("reagent.experiments.ambiguity"),
                        f"tasks.py imports prohibited module: '{alias.name}'."
                    )


# ─── B. Gold Contract Structural Invariants ───────────────────────────────────

class TestGoldContractStructuralInvariants(unittest.TestCase):

    def test_B01_exactly_12_gold_contracts_defined(self):
        """Registry must contain exactly 12 gold contracts."""
        self.assertEqual(len(ALL_GOLD_CONTRACTS), 12)
        self.assertEqual(len(GOLD_CONTRACT_REGISTRY), 12)

    def test_B02_all_contracts_have_provenance_gold(self):
        """All gold contracts must have provenance='gold', not 'fixture' or 'model'."""
        for contract in ALL_GOLD_CONTRACTS:
            self.assertEqual(
                contract.provenance, "gold",
                f"Contract {contract.task_id} has provenance='{contract.provenance}'. "
                f"Must be 'gold'."
            )

    def test_B03_primary_step_contract_is_step_zero(self):
        """primary_step_contract must always have step_index=0."""
        for contract in ALL_GOLD_CONTRACTS:
            self.assertEqual(
                contract.primary_step_contract.step_index, 0,
                f"Contract {contract.task_id} primary_step_contract.step_index != 0."
            )

    def test_B04_all_contracts_have_secondary_steps(self):
        """All 12 gold contracts must have at least one secondary_step_contract."""
        for contract in ALL_GOLD_CONTRACTS:
            self.assertGreater(
                len(contract.secondary_step_contracts), 0,
                f"Contract {contract.task_id} has no secondary_step_contracts. "
                f"Cannot model PARTIAL_COMMIT without at least one secondary step."
            )

    def test_B05_all_field_checks_have_valid_comparison_operators(self):
        """FieldCheck comparison must be one of the declared valid comparison types."""
        valid_comparisons = {
            "EQUALS", "NUMERIC_EQUALS", "CONTAINS", "NOT_NULL",
            "IN_SET", "GREATER_THAN_ZERO",
        }
        for contract in ALL_GOLD_CONTRACTS:
            for step_contract in contract.all_step_contracts:
                for check in step_contract.checks:
                    self.assertIn(
                        check.comparison, valid_comparisons,
                        f"Contract {contract.task_id} table '{check.table}' column "
                        f"'{check.column}' has invalid comparison '{check.comparison}'. "
                        f"Valid: {valid_comparisons}."
                    )

    def test_B06_no_contract_uses_status_success_as_sole_check(self):
        """
        CLASS_STATUS_ONLY_ON_WRITE: a contract that checks only STATUS_SUCCESS
        for a state-changing database write must be rejected.
        This test confirms no gold contract commits this error.
        """
        for contract in ALL_GOLD_CONTRACTS:
            for step_contract in contract.all_step_contracts:
                if len(step_contract.checks) == 1:
                    check = step_contract.checks[0]
                    # A single STATUS_SUCCESS check with no field values is prohibited.
                    self.assertFalse(
                        check.column.upper() == "STATUS" and check.expected_value == "SUCCESS"
                        and check.table == "tool_output",
                        f"Contract {contract.task_id} has a bare STATUS_SUCCESS check "
                        f"(CLASS_STATUS_ONLY_ON_WRITE) as its only assertion."
                    )

    def test_B07_all_field_checks_reference_tables_not_tool_output(self):
        """
        Gold contracts check independently observable world state (DB tables),
        NOT tool execution output claims. Table names must not be 'tool_output',
        'action_result', or similar meta-columns from the pipeline.
        """
        prohibited_tables = {"tool_output", "action_result", "execution_result"}
        for contract in ALL_GOLD_CONTRACTS:
            for step_contract in contract.all_step_contracts:
                for check in step_contract.checks:
                    self.assertNotIn(
                        check.table.lower(), prohibited_tables,
                        f"Contract {contract.task_id} references a prohibited table "
                        f"'{check.table}' (tool execution claim, not DB world state)."
                    )

    def test_B08_all_required_step_contracts_are_marked_required(self):
        """Every secondary_step_contract that is declared must be is_required=True."""
        for contract in ALL_GOLD_CONTRACTS:
            for step_contract in contract.secondary_step_contracts:
                self.assertTrue(
                    step_contract.is_required,
                    f"Contract {contract.task_id} secondary step {step_contract.step_index} "
                    f"table '{step_contract.table}' is not marked is_required. "
                    f"All secondary steps modeling PARTIAL_COMMIT must be required."
                )

    def test_B09_all_required_tables_property_includes_primary_and_secondary(self):
        """all_required_tables must include both primary and secondary table names."""
        for contract in ALL_GOLD_CONTRACTS:
            primary_table = contract.primary_step_contract.table
            self.assertIn(primary_table, contract.all_required_tables,
                          f"Contract {contract.task_id}: primary table '{primary_table}' "
                          f"not in all_required_tables.")
            for sc in contract.secondary_step_contracts:
                self.assertIn(sc.table, contract.all_required_tables,
                              f"Contract {contract.task_id}: secondary table '{sc.table}' "
                              f"not in all_required_tables.")

    def test_B10_absence_authority_note_is_non_empty(self):
        """Every contract must document the conditions for CONFIRMED_ABSENT establishment."""
        for contract in ALL_GOLD_CONTRACTS:
            self.assertTrue(
                contract.absence_authority_note.strip(),
                f"Contract {contract.task_id} has empty absence_authority_note. "
                f"Must document CONFIRMED_ABSENT conditions."
            )

    def test_B11_absence_note_mentions_fresh_connection(self):
        """absence_authority_note must explicitly reference a fresh observer connection."""
        for contract in ALL_GOLD_CONTRACTS:
            note = contract.absence_authority_note.lower()
            self.assertIn(
                "fresh", note,
                f"Contract {contract.task_id} absence_authority_note does not mention "
                f"'fresh' observer connection. Ordinary absent reads do not qualify."
            )

    def test_B12_duplicate_effect_identity_is_non_empty(self):
        """Every contract must document what constitutes a duplicate effect."""
        for contract in ALL_GOLD_CONTRACTS:
            self.assertTrue(
                contract.duplicate_effect_identity.strip(),
                f"Contract {contract.task_id} has empty duplicate_effect_identity."
            )

    def test_B13_verdict_type_enum_has_required_values(self):
        """VerdictType enum must include all five required verdict values."""
        required = {"EXPECTED_PRESENT", "CONFIRMED_ABSENT", "UNKNOWN", "PARTIAL", "WRONG_CONTENT"}
        actual = {v.value for v in VerdictType}
        self.assertEqual(required, actual,
                         f"VerdictType missing values: {required - actual}. "
                         f"Extra: {actual - required}.")

    def test_B14_verdict_for_partial_returns_partial(self):
        """GoldContract.verdict_for_partial() must return VerdictType.PARTIAL."""
        for contract in ALL_GOLD_CONTRACTS:
            self.assertEqual(contract.verdict_for_partial(), VerdictType.PARTIAL,
                             f"Contract {contract.task_id}.verdict_for_partial() returned wrong value.")

    def test_B15_verdict_for_wrong_content_returns_wrong_content(self):
        """GoldContract.verdict_for_wrong_content() must return VerdictType.WRONG_CONTENT."""
        for contract in ALL_GOLD_CONTRACTS:
            self.assertEqual(contract.verdict_for_wrong_content(), VerdictType.WRONG_CONTENT,
                             f"Contract {contract.task_id}.verdict_for_wrong_content() wrong.")

    def test_B16_get_gold_contract_raises_for_unknown_id(self):
        """get_gold_contract() must raise KeyError for an unrecognized task_id."""
        with self.assertRaises(KeyError):
            get_gold_contract("task_99_does_not_exist")

    def test_B17_no_gold_contract_imports_pipeline(self):
        """gold_contracts.py must not have import statements for reagent.pipeline or ambiguity."""
        import ast, inspect
        import reagent.experiments.dbops.gold_contracts as gold_mod
        src = inspect.getsource(gold_mod)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(
                    node.module == "reagent.pipeline"
                    or node.module.startswith("reagent.pipeline.")
                    or node.module.startswith("reagent.experiments.ambiguity"),
                    f"gold_contracts.py ImportFrom prohibited module: '{node.module}'."
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name == "reagent.pipeline"
                        or alias.name.startswith("reagent.pipeline.")
                        or alias.name.startswith("reagent.experiments.ambiguity"),
                        f"gold_contracts.py imports prohibited module: '{alias.name}'."
                    )

    def test_B18_all_field_checks_in_contract_have_non_empty_description(self):
        """Every FieldCheck must have a non-empty description."""
        for contract in ALL_GOLD_CONTRACTS:
            for step_contract in contract.all_step_contracts:
                for check in step_contract.checks:
                    self.assertTrue(
                        check.description.strip(),
                        f"Contract {contract.task_id} table '{check.table}' "
                        f"column '{check.column}' has empty description."
                    )


# ─── C. Task–Contract Binding Integrity ───────────────────────────────────────

class TestTaskContractBinding(unittest.TestCase):

    def test_C01_every_task_has_exactly_one_gold_contract(self):
        """Every task must have exactly one gold contract in GOLD_CONTRACT_REGISTRY."""
        for task in ALL_TASKS:
            self.assertIn(
                task.task_id, GOLD_CONTRACT_REGISTRY,
                f"Task {task.task_id} has no gold contract."
            )
        self.assertEqual(
            set(TASK_REGISTRY.keys()), set(GOLD_CONTRACT_REGISTRY.keys()),
            "Task registry and gold contract registry have different task_ids."
        )

    def test_C02_contract_idempotency_class_matches_task(self):
        """Each gold contract's idempotency_class must match its task's idempotency_class."""
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            self.assertEqual(
                contract.idempotency_class, task.idempotency_class,
                f"Task {task.task_id} idempotency_class={task.idempotency_class} but "
                f"contract idempotency_class={contract.idempotency_class}."
            )

    def test_C03_contract_table_unique_constraints_match_task(self):
        """Each gold contract's table_unique_constraints must match its task."""
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            self.assertEqual(
                contract.table_unique_constraints, task.table_unique_constraints,
                f"Task {task.task_id} table_unique_constraints={task.table_unique_constraints} but "
                f"contract has {contract.table_unique_constraints}."
            )
            self.assertEqual(contract.protection_class, task.protection_class)

    def test_C04_contract_primary_table_matches_task(self):
        """Gold contract primary_step_contract.table must match task.primary_table."""
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            self.assertEqual(
                contract.primary_step_contract.table, task.primary_table,
                f"Task {task.task_id} primary_table='{task.primary_table}' but "
                f"contract primary step table='{contract.primary_step_contract.table}'."
            )

    def test_C05_contract_field_checks_only_target_allowed_tables(self):
        """All tables referenced in a gold contract must be in the task's allowed_tables."""
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            for step_contract in contract.all_step_contracts:
                self.assertIn(
                    step_contract.table, task.allowed_tables,
                    f"Task {task.task_id}: contract step {step_contract.step_index} "
                    f"checks table '{step_contract.table}' not in task.allowed_tables="
                    f"{task.allowed_tables}. Rejection: CLASS_TABLE_OUTSIDE_ALLOWLIST."
                )
                for check in step_contract.checks:
                    self.assertIn(
                        check.table, task.allowed_tables,
                        f"Task {task.task_id}: FieldCheck for column '{check.column}' "
                        f"references table '{check.table}' outside allowed_tables. "
                        f"Rejection: CLASS_TABLE_OUTSIDE_ALLOWLIST."
                    )

    def test_C06_contract_binds_required_request_id(self):
        """
        Every gold contract must include a FieldCheck for the task's
        required_request_id_field. Absence = CLASS_MISSING_REQUEST_ID.
        """
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            all_checked_columns = set()
            for step_contract in contract.all_step_contracts:
                for check in step_contract.checks:
                    all_checked_columns.add(check.column)
            self.assertIn(
                task.required_request_id_field, all_checked_columns,
                f"Gold contract for {task.task_id} does not bind required_request_id_field="
                f"'{task.required_request_id_field}'. Rejection: CLASS_MISSING_REQUEST_ID."
            )

    def test_C07_contract_numeric_amount_matches_task_prompt_amount(self):
        """
        For tasks with prompt_numeric_amount, the gold contract must assert
        that exact amount. Mismatch = CLASS_WRONG_AMOUNT.
        """
        for task in ALL_TASKS:
            if task.prompt_numeric_amount is None:
                continue
            contract = get_gold_contract(task.task_id)
            # Find any NUMERIC_EQUALS check asserting the amount
            amount_checks = [
                check
                for step_contract in contract.all_step_contracts
                for check in step_contract.checks
                if check.comparison == "NUMERIC_EQUALS"
                and check.expected_value == task.prompt_numeric_amount
            ]
            self.assertGreater(
                len(amount_checks), 0,
                f"Gold contract for {task.task_id} has no NUMERIC_EQUALS check for "
                f"prompt_numeric_amount={task.prompt_numeric_amount}. "
                f"A contract with a different amount = CLASS_WRONG_AMOUNT."
            )

    def test_C08_contract_secondary_table_count_matches_task_secondary_tables(self):
        """Number of secondary step contracts must match number of task secondary tables."""
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            self.assertEqual(
                len(contract.secondary_step_contracts), len(task.secondary_tables),
                f"Task {task.task_id} has {len(task.secondary_tables)} secondary tables "
                f"but contract has {len(contract.secondary_step_contracts)} secondary step contracts."
            )

    def test_C09_contract_secondary_tables_match_task_secondary_tables(self):
        """Secondary step contract tables must match task.secondary_tables."""
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            contract_secondary_tables = {sc.table for sc in contract.secondary_step_contracts}
            task_secondary_tables = set(task.secondary_tables)
            self.assertEqual(
                contract_secondary_tables, task_secondary_tables,
                f"Task {task.task_id}: contract secondary tables={contract_secondary_tables} "
                f"!= task secondary tables={task_secondary_tables}."
            )

    def test_C10_unique_constraint_columns_are_checked_on_declared_table(self):
        """Each UNIQUE column must be a FieldCheck on that same table, not a different one."""
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            for tc in task.table_unique_constraints:
                checked = {
                    check.column
                    for sc in contract.all_step_contracts
                    if sc.table == tc.table
                    for check in sc.checks
                    if check.table == tc.table
                }
                for col in tc.columns:
                    self.assertIn(
                        col, checked,
                        f"Task {task.task_id}: UNIQUE column '{col}' on '{tc.table}' "
                        f"is not asserted by gold FieldChecks on that table (have {checked}).",
                    )


# ─── D. Adversarial Contract Rejection ────────────────────────────────────────

class TestAdversarialContractRejection(unittest.TestCase):
    """
    These tests verify that a hypothetical adversarial validator would correctly
    identify and reject each of the 8 adversarial classes declared in the manifest.
    We do NOT call the real ContractGroundingValidator (future phase). Instead we
    verify that the rejection criteria are precisely defined and distinguishable.

    What we CAN test now:
    1. The 8 class names are declared in the manifest.
    2. Each class has a non-empty description.
    3. Gold contracts do NOT commit the errors that define each class.
    4. Each adversarial pattern is logically distinguishable from valid contracts.
    """

    def test_D00_manifest_declares_all_8_adversarial_classes(self):
        """Manifest must declare exactly 8 adversarial rejection classes."""
        self.assertEqual(
            len(ADVERSARIAL_REJECTION_CLASSES), 8,
            f"Expected 8 adversarial rejection classes, got "
            f"{len(ADVERSARIAL_REJECTION_CLASSES)}: {list(ADVERSARIAL_REJECTION_CLASSES)}"
        )

    def test_D01_class_ungrounded_literal_not_in_gold_contracts(self):
        """
        CLASS_UNGROUNDED_LITERAL: predicate specifies value not in prompt.
        Gold contracts must only assert values that appear in human_description.
        Spot-check the most risky cases: numeric amounts.
        """
        # For each task with a numeric amount, verify gold contract amount
        # equals the exact prompt amount (not an invented value like 999.99).
        amount_tasks = [
            (TASK_REFUND_ISSUE, 250.00),
            (TASK_WALLET_TRANSFER, 500.00),
            (TASK_INVOICE_SETTLEMENT, 1200.00),
            (TASK_ESCROW_RELEASE, 3500.00),
            (TASK_INVENTORY_RESTOCK, 200.0),
        ]
        for tid, expected_amount in amount_tasks:
            task = get_task(tid)
            contract = get_gold_contract(tid)
            amount_checks = [
                check
                for sc in contract.all_step_contracts
                for check in sc.checks
                if check.comparison == "NUMERIC_EQUALS"
            ]
            for check in amount_checks:
                # Every numeric check should use a value that actually appears in
                # the prompt. An invented value would be CLASS_UNGROUNDED_LITERAL.
                prompt = task.human_description
                val = abs(check.expected_value)
                val_str = str(int(val)) if val == int(val) else str(val)
                self.assertIn(
                    val_str,
                    prompt.replace(",", ""),
                    f"Task {tid}: FieldCheck amount {check.expected_value} does not "
                    f"appear in prompt. This would be CLASS_UNGROUNDED_LITERAL."
                )

    def test_D02_class_status_only_on_write_rejected_by_gold_contracts(self):
        """
        CLASS_STATUS_ONLY_ON_WRITE: contract specifies only STATUS_SUCCESS for
        a state-changing DB write. Gold contracts must always include DB table
        field checks, not just a STATUS_SUCCESS assertion.
        """
        for contract in ALL_GOLD_CONTRACTS:
            # Count total field checks across all steps
            total_checks = sum(
                len(sc.checks) for sc in contract.all_step_contracts
            )
            self.assertGreater(
                total_checks, 1,
                f"Contract {contract.task_id} has only {total_checks} field check(s). "
                f"A contract with a single STATUS_SUCCESS assertion for a DB write "
                f"is CLASS_STATUS_ONLY_ON_WRITE."
            )

    def test_D03_class_table_outside_allowlist_not_in_gold_contracts(self):
        """
        CLASS_TABLE_OUTSIDE_ALLOWLIST: contract targets a table outside the
        approved allowlist. Gold contracts must only target task.allowed_tables.
        (Covered by C05, included here for adversarial clarity.)
        """
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            for sc in contract.all_step_contracts:
                self.assertIn(
                    sc.table, task.allowed_tables,
                    f"CLASS_TABLE_OUTSIDE_ALLOWLIST: contract {contract.task_id} "
                    f"targets table '{sc.table}' outside allowed_tables."
                )

    def test_D04_class_raw_sql_injection_not_in_field_checks(self):
        """
        CLASS_RAW_SQL_INJECTION: contract predicate contains raw SQL.
        FieldCheck.expected_value must be a typed Python value, not a SQL string.
        """
        sql_patterns = ["SELECT", "INSERT", "UPDATE", "DELETE", "WHERE", "DROP", "--"]
        for contract in ALL_GOLD_CONTRACTS:
            for sc in contract.all_step_contracts:
                for check in sc.checks:
                    if isinstance(check.expected_value, str):
                        for pattern in sql_patterns:
                            self.assertNotIn(
                                pattern, check.expected_value.upper(),
                                f"CLASS_RAW_SQL_INJECTION: contract {contract.task_id} "
                                f"column '{check.column}' expected_value contains SQL keyword "
                                f"'{pattern}': '{check.expected_value}'."
                            )

    def test_D05_class_vacuous_precondition_not_in_gold_contracts(self):
        """
        CLASS_VACUOUS_PRECONDITION: contract checks a condition already satisfied
        BEFORE the operation executes (e.g. checking schema version that always exists).
        Gold contracts must check POST-operation state only.
        Structural check: contracts must not use comparison='STATIC_PRECONDITION'.
        """
        for contract in ALL_GOLD_CONTRACTS:
            for sc in contract.all_step_contracts:
                for check in sc.checks:
                    self.assertNotEqual(
                        check.comparison, "STATIC_PRECONDITION",
                        f"CLASS_VACUOUS_PRECONDITION: contract {contract.task_id} "
                        f"column '{check.column}' uses STATIC_PRECONDITION comparison."
                    )

    def test_D06_class_wrong_amount_not_in_gold_contracts(self):
        """
        CLASS_WRONG_AMOUNT: contract asserts amount contradicting the prompt.
        Gold contracts must assert exactly the prompt amount.
        """
        wrong_amounts = {
            TASK_REFUND_ISSUE: [100.0, 500.0, 0.0],
            TASK_WALLET_TRANSFER: [250.0, 1000.0, 0.0],
            TASK_INVOICE_SETTLEMENT: [600.0, 2400.0, 0.0],
            TASK_ESCROW_RELEASE: [1750.0, 7000.0, 0.0],
        }
        correct_amounts = {
            TASK_REFUND_ISSUE: 250.0,
            TASK_WALLET_TRANSFER: 500.0,
            TASK_INVOICE_SETTLEMENT: 1200.0,
            TASK_ESCROW_RELEASE: 3500.0,
        }
        for tid, wrong_list in wrong_amounts.items():
            contract = get_gold_contract(tid)
            for sc in contract.all_step_contracts:
                for check in sc.checks:
                    if check.comparison == "NUMERIC_EQUALS":
                        for wrong_val in wrong_list:
                            self.assertNotEqual(
                                check.expected_value, wrong_val,
                                f"CLASS_WRONG_AMOUNT: contract {tid} column "
                                f"'{check.column}' asserts wrong amount {wrong_val} "
                                f"(correct is {correct_amounts[tid]})."
                            )

    def test_D07_class_missing_request_id_not_in_gold_contracts(self):
        """
        CLASS_MISSING_REQUEST_ID: contract lacks mandatory request_id binding.
        Gold contracts must include a FieldCheck for required_request_id_field.
        (Covered by C06, included here for adversarial clarity.)
        """
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            all_columns = {
                check.column
                for sc in contract.all_step_contracts
                for check in sc.checks
            }
            self.assertIn(
                task.required_request_id_field, all_columns,
                f"CLASS_MISSING_REQUEST_ID: contract {task.task_id} lacks "
                f"FieldCheck for required_request_id_field="
                f"'{task.required_request_id_field}'."
            )

    def test_D08_class_undeclared_extra_field_not_in_gold_contracts(self):
        """
        CLASS_UNDECLARED_EXTRA_FIELD: contract specifies columns that do not
        exist in declared table schema.
        We cannot check schema here (no DB). Instead verify that each
        FieldCheck.column is non-empty and does not contain obvious inventions
        like '__phantom__' or 'nonexistent_col'.
        """
        suspicious_columns = {"__phantom__", "nonexistent_col", "fake_field", "undefined"}
        for contract in ALL_GOLD_CONTRACTS:
            for sc in contract.all_step_contracts:
                for check in sc.checks:
                    self.assertNotIn(
                        check.column.lower(), suspicious_columns,
                        f"CLASS_UNDECLARED_EXTRA_FIELD: contract {contract.task_id} "
                        f"references suspicious column '{check.column}'."
                    )
                    self.assertTrue(
                        check.column.strip(),
                        f"CLASS_UNDECLARED_EXTRA_FIELD: contract {contract.task_id} "
                        f"has an empty column name."
                    )

    def test_D09_all_8_adversarial_class_names_in_manifest_have_descriptions(self):
        """Each of the 8 adversarial classes must have a non-empty description."""
        for class_name, description in ADVERSARIAL_REJECTION_CLASSES.items():
            self.assertTrue(
                description.strip(),
                f"Adversarial class '{class_name}' has an empty description in manifest."
            )

    def test_D10_correct_amount_is_not_wrong(self):
        """Sanity: correct amounts from prompts do not appear in 'wrong' lists for other tasks."""
        # Escrow $3500 must not appear as a valid amount for wallet transfer ($500)
        wallet_contract = get_gold_contract(TASK_WALLET_TRANSFER)
        escrow_contract = get_gold_contract(TASK_ESCROW_RELEASE)
        wallet_amounts = {
            check.expected_value
            for sc in wallet_contract.all_step_contracts
            for check in sc.checks
            if check.comparison == "NUMERIC_EQUALS"
        }
        escrow_amounts = {
            check.expected_value
            for sc in escrow_contract.all_step_contracts
            for check in sc.checks
            if check.comparison == "NUMERIC_EQUALS"
        }
        self.assertIn(500.0, wallet_amounts,
                      "Wallet transfer gold contract must assert $500.00.")
        self.assertIn(3500.0, escrow_amounts,
                      "Escrow release gold contract must assert $3500.00.")
        self.assertNotIn(3500.0, wallet_amounts,
                         "Wallet transfer gold contract must NOT assert $3500.00 (cross-task pollution).")


# ─── E. PARTIAL_COMMIT Semantics ──────────────────────────────────────────────

class TestPartialCommitSemantics(unittest.TestCase):

    def test_E01_partial_commit_requires_multi_step_contract(self):
        """
        PARTIAL_COMMIT detection is only possible if the gold contract checks
        both primary and secondary steps. A contract with only one step cannot
        detect a partial commit.
        """
        for contract in ALL_GOLD_CONTRACTS:
            self.assertGreater(
                len(contract.all_step_contracts), 1,
                f"Contract {contract.task_id} has only 1 step contract. "
                f"Cannot model PARTIAL_COMMIT (primary present, secondary absent)."
            )

    def test_E02_partial_commit_verdict_is_not_verified(self):
        """
        Under PARTIAL_COMMIT, the expected verdict is PARTIAL, not VERIFIED.
        A single-table presence check that returns VERIFIED on partial commit
        is the specific failure mode for HAND_AUTHORED_BASELINE and
        SHARED_OBSERVATION (the falsification variable).
        """
        # This test verifies the verdict type hierarchy is correctly defined.
        self.assertNotEqual(VerdictType.PARTIAL, VerdictType.EXPECTED_PRESENT,
                            "PARTIAL must be distinct from EXPECTED_PRESENT.")
        self.assertNotEqual(VerdictType.PARTIAL, VerdictType.CONFIRMED_ABSENT,
                            "PARTIAL must be distinct from CONFIRMED_ABSENT.")

    def test_E03_partial_commit_requires_primary_present_secondary_absent(self):
        """
        PARTIAL_COMMIT is defined as: primary step row PRESENT, required secondary
        row ABSENT (via a fresh observer connection after executing connection is closed).
        Neither CONFIRMED_ABSENT nor UNKNOWN applies to the primary row.
        """
        for contract in ALL_GOLD_CONTRACTS:
            primary = contract.primary_step_contract
            self.assertTrue(
                primary.is_required,
                f"Contract {contract.task_id} primary step is not is_required. "
                f"PARTIAL_COMMIT semantics require a required primary step."
            )
            for secondary in contract.secondary_step_contracts:
                self.assertTrue(
                    secondary.is_required,
                    f"Contract {contract.task_id} secondary step '{secondary.table}' "
                    f"is not is_required. PARTIAL_COMMIT must catch absent secondary rows."
                )

    def test_E04_single_atomic_transaction_partial_commit_is_application_modeled(self):
        """
        For tasks, PARTIAL_COMMIT is modeled at the application layer (two separate
        writes), NOT as a single atomic SQLite transaction that partially committed
        (which is impossible). All 12 tasks model this via multi-step workflow
        or secondary write at application boundary.
        """
        # All tasks have secondary_tables (tested in A09). Verify none of them
        # describe a partial atomic transaction.
        for task in ALL_TASKS:
            # A true partial SQLite transaction is impossible. The steps model
            # application-level write boundaries.
            # We verify this is true: each task has >= 2 steps.
            self.assertGreaterEqual(
                len(task.steps), 2,
                f"Task {task.task_id} has only {len(task.steps)} step(s). "
                f"Modeling PARTIAL_COMMIT requires at least 2 distinct write steps."
            )

    def test_E05_gold_contract_full_postcondition_check_detects_partial(self):
        """
        The gold contract's all_step_contracts covers both primary and secondary
        tables. The evaluator running all step contracts detects PARTIAL when
        primary row is PRESENT but secondary row is ABSENT.
        The HAND_AUTHORED_BASELINE checking only primary table would miss this.
        """
        for task in ALL_TASKS:
            contract = get_gold_contract(task.task_id)
            primary_tables = {contract.primary_step_contract.table}
            all_tables = contract.all_required_tables
            secondary_only = all_tables - primary_tables

            self.assertGreater(
                len(secondary_only), 0,
                f"Contract {task.task_id} all_required_tables only includes primary table. "
                f"Cannot distinguish PARTIAL from EXPECTED_PRESENT."
            )


# ─── F. Unprotected Duplicate-Effect Demonstration ───────────────────────────

class TestNonUniqueDuplicateEffects(unittest.TestCase):

    def test_F01_unprotected_tasks_have_no_table_unique_constraints(self):
        """
        Unprotected tasks must have empty table_unique_constraints.
        (Covered by A06 — reproduced here for attribution clarity.)
        """
        for task in get_unprotected_tasks():
            self.assertEqual(
                task.table_unique_constraints, (),
                f"Unprotected task {task.task_id} has table_unique_constraints="
                f"{task.table_unique_constraints}. This would attribute duplicate "
                f"prevention to DB constraints instead of ReAgent."
            )

    def test_F02_non_unique_task_duplicate_identity_mentions_no_schema_protection(self):
        """
        Each unprotected task's gold contract duplicate_effect_identity must
        document that there is NO schema UNIQUE constraint and that ReAgent
        restraint is the only prevention.
        """
        for task in get_unprotected_tasks():
            contract = get_gold_contract(task.task_id)
            identity = contract.duplicate_effect_identity.upper()
            # Must NOT positively attribute to DB constraint.
            self.assertNotIn(
                "ATTRIBUTION: DB CONSTRAINT",
                identity,
                f"Unprotected task {task.task_id} wrongly attributes duplicate prevention to DB constraint."
            )
            self.assertIn(
                "NO SCHEMA UNIQUE",
                identity,
                f"Unprotected task {task.task_id} duplicate_effect_identity does not state 'NO schema UNIQUE'."
            )
            self.assertIn(
                "REAGENT RESTRAINT ONLY",
                identity,
                f"Unprotected task {task.task_id} duplicate_effect_identity does not attribute to ReAgent restraint only."
            )

    def test_F03_key_protected_duplicate_identity_mentions_db_constraint(self):
        """
        Each KEY_PROTECTED task's gold contract duplicate_effect_identity must
        attribute duplicate prevention to the DB constraint, not to ReAgent alone.
        """
        for task in get_key_protected_tasks():
            contract = get_gold_contract(task.task_id)
            identity = contract.duplicate_effect_identity.upper()
            self.assertIn(
                "CONSTRAINT",
                identity,
                f"UNIQUE task {task.task_id} duplicate_effect_identity does not "
                f"mention 'CONSTRAINT': '{contract.duplicate_effect_identity}'"
            )

    def test_F04_non_unique_wallet_transfer_double_write_is_double_debit(self):
        """
        Wallet transfer: double write = two debit rows = double debit.
        This is the mechanism that makes NON_UNIQUE dangerous without restraint.
        """
        contract = get_gold_contract(TASK_WALLET_TRANSFER)
        self.assertIn(
            "DOUBLE", contract.duplicate_effect_identity.upper(),
            "Wallet transfer duplicate identity must describe 'DOUBLE' debit effect."
        )
        self.assertIn(
            "500", contract.duplicate_effect_identity,
            "Wallet transfer duplicate identity should reference $500 amount."
        )

    def test_F05_non_unique_escrow_release_double_write_is_double_release(self):
        """
        Escrow release: double write = two disbursement rows = double release.
        Most financially impactful duplicate in the benchmark.
        """
        contract = get_gold_contract(TASK_ESCROW_RELEASE)
        self.assertIn(
            "DOUBLE", contract.duplicate_effect_identity.upper(),
            "Escrow release duplicate identity must describe 'DOUBLE' release."
        )
        self.assertIn(
            "3500", contract.duplicate_effect_identity,
            "Escrow release duplicate identity should reference $3500 amount."
        )

    def test_F06_non_unique_inventory_restock_double_write_is_double_increment(self):
        """Inventory restock: double write = double stock increment."""
        contract = get_gold_contract(TASK_INVENTORY_RESTOCK)
        self.assertIn(
            "DOUBLE", contract.duplicate_effect_identity.upper(),
            "Inventory restock duplicate identity must describe 'DOUBLE' increment."
        )

    def test_F07_exactly_6_tasks_are_unprotected_in_manifest_and_task_registry(self):
        """SCHEMA_DEDUPLICATION_SUPPORT False set must equal unprotected tasks."""
        manifest_unprotected = [tid for tid, unique in SCHEMA_DEDUPLICATION_SUPPORT.items()
                                if unique is False]
        self.assertEqual(len(manifest_unprotected), 6)
        task_unprotected = get_unprotected_tasks()
        self.assertEqual(len(task_unprotected), 6)
        self.assertEqual(
            set(manifest_unprotected), {t.task_id for t in task_unprotected},
            "Unprotected task_ids differ between manifest and task registry."
        )

    def test_F08_all_non_unique_absence_notes_distinguish_confirmed_from_stale(self):
        """
        For NON_UNIQUE non-idempotent operations, CONFIRMED_ABSENT must not be
        equated with ordinary stale reads. The absence note must make this distinction.
        """
        for task in get_unprotected_tasks():
            contract = get_gold_contract(task.task_id)
            note = contract.absence_authority_note.upper()
            self.assertIn(
                "FRESH", note,
                f"Unprotected task {task.task_id} absence note does not mention FRESH. "
                f"CONFIRMED_ABSENT requires a fresh connection."
            )
            escalation_mentioned = "ESCALATE" in note or "UNKNOWN" in note or "RESTRAINT" in note
            self.assertTrue(
                escalation_mentioned,
                f"Unprotected task {task.task_id} absence note does not mention "
                f"escalation for UNKNOWN states: '{contract.absence_authority_note}'"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
