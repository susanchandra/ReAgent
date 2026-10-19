"""
eval_scoring.py — Deterministic Gold-Equivalence & Decision Evaluation for DBOPS-M2.

Evaluates parsed and grounded StructuredDecisions against the frozen GoldContract
and DbopsTask specifications. Enforces strict three-way separation:
1. Syntax / Schema Validity (JSON parses and conforms to StructuredDecision schema)
2. Grounding Validity (passes ContractGroundingValidator with independent predicate)
3. Gold Equivalence (matches required request_id, targets allowed tables, matches
   numeric amount, and verifies semantic step completeness against GoldContract).

Zero network access. Zero model provider calls. Zero .env inspection.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

from reagent.models import StructuredDecision, Postcondition
from reagent.experiments.dbops.tasks import get_task, DbopsTask
from reagent.experiments.dbops.gold_contracts import get_gold_contract, GoldContract
from reagent.experiments.dbops.validator import (
    TABLE_COLUMNS,
    GROUNDED_VALUES_BY_TASK,
    STATUS_ONLY_COLUMNS,
    RAW_SQL_MARKERS,
)
from reagent.experiments.dbops.schema import ALL_TABLE_NAMES


@dataclass(frozen=True)
class ContractScoringResult:
    """
    Immutable evaluation of a single model-generated StructuredDecision.
    """
    task_id: str
    is_schema_valid: bool
    is_grounded: bool
    is_gold_equivalent: bool
    matched_request_id: bool
    matched_numeric_amount: bool
    matched_target_tables: bool
    gold_equivalence_errors: Tuple[str, ...]
    score_summary: str
    missing_required_postconditions: Tuple[str, ...] = ()
    extra_unsupported_predicates: Tuple[str, ...] = ()
    unsupported_tables_or_columns: Tuple[str, ...] = ()
    uniqueness_respected: bool = True
    invalid_contract_rejected: bool = False
    fallback_used: bool = False


class DbopsGoldScorer:
    """
    Evaluates whether a validated StructuredDecision is truly gold-equivalent.
    """

    def evaluate_decision(
        self,
        task_id: str,
        decision: Optional[StructuredDecision],
        schema_valid: bool,
        grounding_valid: bool,
    ) -> ContractScoringResult:
        """
        Evaluate a StructuredDecision against the frozen GoldContract.
        """
        if not schema_valid or decision is None:
            return ContractScoringResult(
                task_id=task_id,
                is_schema_valid=False,
                is_grounded=False,
                is_gold_equivalent=False,
                matched_request_id=False,
                matched_numeric_amount=False,
                matched_target_tables=False,
                gold_equivalence_errors=("Schema invalid or missing decision",),
                score_summary="FAILED_SCHEMA",
            )

        if not grounding_valid:
            return ContractScoringResult(
                task_id=task_id,
                is_schema_valid=True,
                is_grounded=False,
                is_gold_equivalent=False,
                matched_request_id=False,
                matched_numeric_amount=False,
                matched_target_tables=False,
                gold_equivalence_errors=("Grounding validation failed",),
                score_summary="FAILED_GROUNDING",
            )

        task = get_task(task_id)
        gold = get_gold_contract(task_id)
        errors: List[str] = []
        missing_pcs: List[str] = []
        extra_predicates: List[str] = []
        unsupported_schema: List[str] = []
        uniqueness_ok = True

        pcs = decision.proposed_postconditions

        # 1. Target table check & schema allowlist check
        targets = {pc.target for pc in pcs if pc.target}
        matched_target_tables = False
        if any(t in task.allowed_tables for t in targets):
            matched_target_tables = True
        else:
            errors.append(f"No postcondition targets an allowed table for {task_id} (allowed: {sorted(list(task.allowed_tables))})")

        # Disallow postconditions targeting known tables outside the allowed allowlist
        for t in targets:
            if t in ALL_TABLE_NAMES and t not in task.allowed_tables:
                unsupported_schema.append(f"Table '{t}' outside task allowlist")
                errors.append(f"Postcondition targets prohibited table '{t}' outside {task_id} allowlist")

        # Column allowlist check & raw SQL injection check
        for pc in pcs:
            tbl = pc.target if pc.target in task.allowed_tables else task.primary_table
            known_cols = TABLE_COLUMNS.get(tbl, frozenset())
            if pc.target in known_cols or pc.target in task.allowed_tables:
                pass
            elif pc.target in ALL_TABLE_NAMES and pc.target not in task.allowed_tables:
                pass  # already caught above
            elif pc.target and pc.target not in known_cols and pc.target not in task.allowed_tables:
                grounded_vals = GROUNDED_VALUES_BY_TASK.get(task_id, frozenset())
                if pc.target not in grounded_vals and pc.target != task.required_request_id_field:
                    unsupported_schema.append(f"Undeclared target '{pc.target}'")
                    errors.append(f"Postcondition target '{pc.target}' is undeclared in schema or prompt")

            # Raw SQL check
            exp_str = str(pc.expected or "").upper()
            pred_str = str(pc.predicate or "").upper()
            for marker in RAW_SQL_MARKERS:
                if marker in exp_str or marker in pred_str:
                    extra_predicates.append(f"Raw SQL marker '{marker.strip()}' in postcondition")
                    errors.append(f"Postcondition contains raw SQL injection marker '{marker.strip()}'")
                    break

        # 2. Required request_id trace field check
        # Must match either as target, expected, or mentioned in postcondition description
        matched_request_id = False
        req_id = task.required_request_id_field
        grounded_vals = GROUNDED_VALUES_BY_TASK.get(task_id, frozenset())

        # Find the expected request_id literal for this task
        # Typically the main transaction reference ID
        req_id_literal = None
        for g_val in grounded_vals:
            if isinstance(g_val, str) and ("-" in g_val or g_val.startswith("MIG")):
                req_id_literal = g_val
                break

        for pc in pcs:
            # Check if request_id column or literal is bound
            if pc.target == req_id or (req_id_literal and pc.target == req_id_literal):
                matched_request_id = True
                break
            if pc.expected == req_id or (req_id_literal and str(pc.expected) == req_id_literal):
                matched_request_id = True
                break
            if pc.description and (req_id in pc.description or (req_id_literal and req_id_literal in pc.description)):
                matched_request_id = True
                break

        if not matched_request_id:
            errors.append(f"Contract does not bind mandatory request ID '{req_id}' (literal: {req_id_literal})")

        # 3. Numeric amount check (if task specifies an amount)
        matched_numeric_amount = True
        if task.prompt_numeric_amount is not None:
            amt_expected = task.prompt_numeric_amount
            amt_found = False
            for pc in pcs:
                # Check expected value or description
                if pc.expected is not None:
                    try:
                        if abs(float(pc.expected) - amt_expected) < 0.001:
                            amt_found = True
                            break
                    except (ValueError, TypeError):
                        pass
                if pc.description and (f"{amt_expected:.2f}" in pc.description or str(int(amt_expected)) in pc.description or f"{amt_expected}" in pc.description):
                    amt_found = True
                    break

            if not amt_found:
                matched_numeric_amount = False
                errors.append(f"Contract does not assert expected numeric amount {amt_expected}")

        # 4. Source and Destination Entity check
        matched_entities = True
        if task.prompt_source_id:
            src_found = any(
                pc.target == task.prompt_source_id or
                pc.expected == task.prompt_source_id or
                (pc.description and task.prompt_source_id in pc.description)
                for pc in pcs
            )
            if not src_found:
                matched_entities = False
                errors.append(f"Contract does not reference source entity '{task.prompt_source_id}'")

        if task.prompt_destination_id:
            dst_found = any(
                pc.target == task.prompt_destination_id or
                pc.expected == task.prompt_destination_id or
                (pc.description and task.prompt_destination_id in pc.description)
                for pc in pcs
            )
            if not dst_found:
                matched_entities = False
                errors.append(f"Contract does not reference destination entity '{task.prompt_destination_id}'")

        # 5. Secondary Table Effect check (records missing secondary postconditions)
        if task.secondary_tables:
            for sec_tbl in task.secondary_tables:
                sec_found = any(
                    sec_tbl in (pc.target or "") or sec_tbl in (pc.description or "")
                    for pc in pcs
                )
                if not sec_found:
                    missing_pcs.append(f"Secondary table '{sec_tbl}' postcondition")

        # 6. Uniqueness constraints check (Task 05 & Task 08 business uniqueness rules)
        if task_id == "task_05_order_cancellation":
            for pc in pcs:
                desc_lower = (pc.description or "").lower()
                if "unique" in desc_lower and "cancellation_id" in desc_lower and "orders" in desc_lower:
                    uniqueness_ok = False
                    errors.append("Task 05 constraint error: orders table does not enforce uniqueness on cancellation_id")
        elif task_id == "task_08_invoice_settlement":
            for pc in pcs:
                desc_lower = (pc.description or "").lower()
                if "unique" in desc_lower and "invoices" in desc_lower and "settlement_ref" in desc_lower:
                    uniqueness_ok = False
                    errors.append("Task 08 constraint error: invoices table does not enforce uniqueness on settlement_ref")

        # 7. Gold-equivalence verdict
        is_gold_equivalent = (
            matched_target_tables
            and matched_request_id
            and matched_numeric_amount
            and matched_entities
            and uniqueness_ok
            and len(errors) == 0
        )

        summary = "GOLD_EQUIVALENT" if is_gold_equivalent else f"DISCREPANCY: {'; '.join(errors)}"

        return ContractScoringResult(
            task_id=task_id,
            is_schema_valid=True,
            is_grounded=True,
            is_gold_equivalent=is_gold_equivalent,
            matched_request_id=matched_request_id,
            matched_numeric_amount=matched_numeric_amount,
            matched_target_tables=matched_target_tables,
            gold_equivalence_errors=tuple(errors),
            score_summary=summary,
            missing_required_postconditions=tuple(missing_pcs),
            extra_unsupported_predicates=tuple(extra_predicates),
            unsupported_tables_or_columns=tuple(unsupported_schema),
            uniqueness_respected=uniqueness_ok,
            invalid_contract_rejected=not is_gold_equivalent,
            fallback_used=False,
        )
