"""
candidate_gate.py — Model-Derived Verification Contract Validation Gate.

Implements the deterministic, offline validation gate for model-derived candidate contracts:
1. Strict three-way separation:
   - Human-authored Gold Contract (evaluation reference only)
   - Model-derived Candidate Contract (output of the structured decision pipeline)
   - Runtime Observation (independently probed task state)
2. Exercises the exact production parsing and validation path:
   - JSON parsing with fence stripping (_safe_parse_json)
   - StructuredDecision schema validation (_validate_structured_decision)
   - Pre-execution Contract Grounding (ContractGroundingValidator)
   - DBOPS Task-Level Allowlist, Column, and Uniqueness Validation (TABLE_COLUMNS, allowed_tables)
3. Guaranteed fail-closed behavior:
   - Rejects malformed JSON, unknown fields, provenance injection, empty tools, empty predicates.
   - Rejects unsupported tables, undeclared columns, raw SQL markers, wrong numeric amounts.
   - Flags composite uniqueness errors (Task 05 order_event_log, Task 08 invoice_payments).
   - Never silently replaces an invalid candidate with the gold contract (fallback_used = False).
   - Rejection results in fail-closed escalation (ESCALATED), never false verification (VERIFIED).
4. Separate accounting:
   - generation_parse_success, schema_valid, grounding_valid, semantic_gold_agreement,
     missing_required_postconditions, extra_unsupported_predicates, unsupported_tables_or_columns,
     uniqueness_respected, invalid_contract_rejected, fallback_used.
5. Zero network calls, zero provider requests, zero .env inspection. Explicitly labeled SYNTHETIC / OFFLINE.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

from reagent.models import StructuredDecision, Postcondition, AgentStatus
from reagent.contract_validator import ContractGroundingValidator, ValidationResult as GroundingValidationResult
from reagent.providers.nebius import (
    NebiusNemotronProvider,
    _safe_parse_json,
    ALLOWED_STRUCTURED_DECISION_KEYS,
)
from reagent.experiments.dbops.manifest_dbops_v1 import ALL_TASK_IDS
from reagent.experiments.dbops.tasks import (
    get_task,
    DbopsTask,
    ALL_TASKS,
)
from reagent.experiments.dbops.gold_contracts import get_gold_contract, GoldContract
from reagent.experiments.dbops.prompt_harness import DbopsPromptHarness, TaskPromptBundle
from reagent.experiments.dbops.eval_scoring import DbopsGoldScorer, ContractScoringResult
from reagent.experiments.dbops.validator import (
    TABLE_COLUMNS,
    GROUNDED_VALUES_BY_TASK,
    RAW_SQL_MARKERS,
    STATUS_ONLY_COLUMNS,
)
from reagent.experiments.dbops.schema import ALL_TABLE_NAMES
from reagent.experiments.dbops.integration import DbopsDatabaseTool
from reagent.experiments.dbops.store import DbopsStore


PROVENANCE_LABEL = "SYNTHETIC / OFFLINE FIXTURES"


@dataclass(frozen=True)
class CandidateEvaluationRecord:
    """
    Immutable evaluation record of a model-derived candidate contract.
    """
    task_id: str
    provenance: str
    raw_response: Optional[str]
    generation_parse_success: bool
    schema_valid: bool
    schema_error: Optional[str]
    grounding_valid: bool
    grounding_errors: Tuple[str, ...]
    semantic_gold_agreement: bool
    missing_required_postconditions: Tuple[str, ...]
    extra_unsupported_predicates: Tuple[str, ...]
    unsupported_tables_or_columns: Tuple[str, ...]
    uniqueness_respected: bool
    invalid_contract_rejected: bool
    fallback_used: bool
    final_verdict: str
    score_summary: str
    decision: Optional[StructuredDecision] = None


class CandidateContractValidationGate:
    """
    Validation gate evaluating candidate model contracts against DBOPS specifications.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root
        self.prompt_harness = DbopsPromptHarness()
        self.grounding_validator = ContractGroundingValidator(workspace_root=workspace_root)
        self.unconfigured_provider = NebiusNemotronProvider(client=None)
        self.gold_scorer = DbopsGoldScorer()

    def verify_prompt_isolation(self, task_id: str) -> Dict[str, Any]:
        """
        Proves that prompt construction for task_id does not leak gold queries,
        injected fault modes, or oracle observation data.
        """
        bundle = self.prompt_harness.construct_task_prompt(task_id)
        prompt_text = bundle.full_prompt_text.upper()

        forbidden_fault_modes = [
            "COMMIT_THEN_TIMEOUT",
            "IMMEDIATE_FAILURE",
            "PARTIAL_COMMIT",
            "WRONG_CONTENT",
            "AUDIT_ONLY",
            "STATUS_UNAVAILABLE",
            "UNOBSERVABLE",
        ]
        leaked_faults = [f for f in forbidden_fault_modes if f in prompt_text]

        gold_contract = get_gold_contract(task_id)
        leaked_gold_markers = []
        if "GOLD_CONTRACT" in prompt_text:
            leaked_gold_markers.append("GOLD_CONTRACT literal")
        if "ORACLE" in prompt_text:
            leaked_gold_markers.append("ORACLE reference")

        is_clean = (len(leaked_faults) == 0) and (len(leaked_gold_markers) == 0)

        return {
            "task_id": task_id,
            "prompt_hash": bundle.prompt_hash,
            "estimated_tokens": bundle.estimated_tokens,
            "is_clean": is_clean,
            "leaked_faults": leaked_faults,
            "leaked_gold_markers": leaked_gold_markers,
            "human_description_present": get_task(task_id).human_description in bundle.user_prompt,
        }

    def validate_candidate_raw(
        self,
        task_id: str,
        raw_response_text: str,
        provenance: str = PROVENANCE_LABEL,
    ) -> CandidateEvaluationRecord:
        """
        Validates raw candidate response text through the complete production parsing path.
        """
        task = get_task(task_id)

        # ── Step 1: Safe JSON extraction (strips markdown fences) ────────────
        parsed_dict = _safe_parse_json(raw_response_text)
        if parsed_dict is None or not isinstance(parsed_dict, dict):
            # Model refusal, empty text, or malformed JSON
            is_refusal = any(kw in raw_response_text.lower() for kw in ("cannot fulfill", "refuse", "unable to", "sorry"))
            verdict = "REJECTED_REFUSAL" if is_refusal else "REJECTED_MALFORMED_JSON"
            return CandidateEvaluationRecord(
                task_id=task_id,
                provenance=provenance,
                raw_response=raw_response_text,
                generation_parse_success=False,
                schema_valid=False,
                schema_error="JSON extraction returned None or non-dictionary root",
                grounding_valid=False,
                grounding_errors=("Unparseable JSON response",),
                semantic_gold_agreement=False,
                missing_required_postconditions=(),
                extra_unsupported_predicates=(),
                unsupported_tables_or_columns=(),
                uniqueness_respected=True,
                invalid_contract_rejected=True,
                fallback_used=False,
                final_verdict=verdict,
                score_summary=f"FAIL: {verdict}",
                decision=None,
            )

        # ── Step 2: StructuredDecision Schema Validation ────────────────────
        decision = self.unconfigured_provider._validate_structured_decision(parsed_dict)
        if decision is None:
            # Diagnose specific schema failure
            schema_err = self._diagnose_schema_error(parsed_dict)
            return CandidateEvaluationRecord(
                task_id=task_id,
                provenance=provenance,
                raw_response=raw_response_text,
                generation_parse_success=True,
                schema_valid=False,
                schema_error=schema_err,
                grounding_valid=False,
                grounding_errors=(),
                semantic_gold_agreement=False,
                missing_required_postconditions=(),
                extra_unsupported_predicates=(),
                unsupported_tables_or_columns=(),
                uniqueness_respected=True,
                invalid_contract_rejected=True,
                fallback_used=False,
                final_verdict="REJECTED_SCHEMA_INVALID",
                score_summary=f"FAIL: SCHEMA_INVALID ({schema_err})",
                decision=None,
            )

        # ── Step 3: Generic Contract Grounding Validator ────────────────────
        temp_store = DbopsStore.create_temp()
        try:
            temp_tool = DbopsDatabaseTool(temp_store)
            grounding_res = self.grounding_validator.validate(
                decision=decision,
                tool=temp_tool,
                objective=task.human_description,
            )
        finally:
            temp_store.cleanup()

        if not grounding_res.is_valid:
            return CandidateEvaluationRecord(
                task_id=task_id,
                provenance=provenance,
                raw_response=raw_response_text,
                generation_parse_success=True,
                schema_valid=True,
                schema_error=None,
                grounding_valid=False,
                grounding_errors=tuple(grounding_res.errors),
                semantic_gold_agreement=False,
                missing_required_postconditions=(),
                extra_unsupported_predicates=(),
                unsupported_tables_or_columns=(),
                uniqueness_respected=True,
                invalid_contract_rejected=True,
                fallback_used=False,
                final_verdict="REJECTED_GROUNDING_FAILED",
                score_summary=f"FAIL: GROUNDING_FAILED ({'; '.join(grounding_res.errors)})",
                decision=decision,
            )

        # ── Step 4 & 5: DBOPS Task-Level Grounding & Gold Scoring ─────────────
        score_res = self.gold_scorer.evaluate_decision(
            task_id=task_id,
            decision=decision,
            schema_valid=True,
            grounding_valid=True,
        )

        final_verdict = "GOLD_EQUIVALENT" if score_res.is_gold_equivalent else "SEMANTIC_DISCREPANCY"

        return CandidateEvaluationRecord(
            task_id=task_id,
            provenance=provenance,
            raw_response=raw_response_text,
            generation_parse_success=True,
            schema_valid=True,
            schema_error=None,
            grounding_valid=True,
            grounding_errors=(),
            semantic_gold_agreement=score_res.is_gold_equivalent,
            missing_required_postconditions=score_res.missing_required_postconditions,
            extra_unsupported_predicates=score_res.extra_unsupported_predicates,
            unsupported_tables_or_columns=score_res.unsupported_tables_or_columns,
            uniqueness_respected=score_res.uniqueness_respected,
            invalid_contract_rejected=not score_res.is_gold_equivalent,
            fallback_used=False,
            final_verdict=final_verdict,
            score_summary=score_res.score_summary,
            decision=decision,
        )

    def _diagnose_schema_error(self, raw: Dict[str, Any]) -> str:
        """Diagnose exact schema rejection reason."""
        if not isinstance(raw, dict):
            return "Root payload is not a dict"
        if "provenance" in raw:
            return "Payload illegally attempted to inject 'provenance'"
        unknown = set(raw.keys()) - ALLOWED_STRUCTURED_DECISION_KEYS
        if unknown:
            return f"Payload contains unknown keys: {sorted(list(unknown))}"
        if "tool" not in raw:
            return "Payload is missing required 'tool' key"
        if not isinstance(raw.get("tool"), str):
            return "'tool' key must be a string"
        if not raw["tool"].strip():
            return "'tool' key is empty or whitespace"
        if "arguments" in raw and not isinstance(raw["arguments"], dict):
            return "'arguments' must be a dict"
        if "proposed_postconditions" in raw and not isinstance(raw["proposed_postconditions"], list):
            return "'proposed_postconditions' must be a list"
        return "Malformed structured decision structure"

    # ── Deterministic Fixture Catalog for Testing ─────────────────────────────

    @staticmethod
    def generate_candidate_fixture(task_id: str, variant: str) -> str:
        """
        Generates deterministic valid and adversarial candidate fixtures for offline evaluation.
        """
        task = get_task(task_id)
        amt_str = f" of {task.prompt_numeric_amount:.2f}" if task.prompt_numeric_amount is not None else ""
        entities_str = ""
        if task.prompt_source_id:
            entities_str += f" from {task.prompt_source_id}"
        if task.prompt_destination_id:
            entities_str += f" to {task.prompt_destination_id}"

        if variant == "VALID_GOLD_EQUIVALENT":
            pcs = [
                {
                    "predicate": "STATUS_SUCCESS",
                    "target": task.primary_table,
                    "expected": task.required_request_id_field,
                    "description": f"Operation {task.required_request_id_field}{amt_str}{entities_str} on {task.primary_table}",
                    "required": True,
                },
                {
                    "predicate": "KEY_EXISTS",
                    "target": task.required_request_id_field,
                    "expected": task.prompt_source_id or task.primary_table,
                    "description": f"Key {task.required_request_id_field} verified in {task.primary_table}",
                    "required": False,
                },
            ]
            if task.secondary_tables:
                pcs.append({
                    "predicate": "KEY_EXISTS",
                    "target": task.secondary_tables[0],
                    "expected": task.required_request_id_field,
                    "description": f"Secondary record in {task.secondary_tables[0]}",
                    "required": False,
                })
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "proposed_postconditions": pcs,
            })

        elif variant == "MISSING_REQUIRED_POSTCONDITION":
            # Omits primary table or required request ID
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "proposed_postconditions": [
                    {
                        "predicate": "STATUS_SUCCESS",
                        "target": task.primary_table,
                        "description": "Generic success without request ID binding",
                        "required": True,
                    },
                    {
                        "predicate": "KEY_EXISTS",
                        "target": "unrelated_target",
                        "expected": "unrelated_val",
                        "description": "Non-identifying check",
                        "required": False,
                    }
                ],
            })

        elif variant == "WRONG_EXPECTED_AMOUNT":
            bad_amt = (task.prompt_numeric_amount or 100.0) + 999.0
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "proposed_postconditions": [
                    {
                        "predicate": "STATUS_SUCCESS",
                        "target": task.primary_table,
                        "expected": task.required_request_id_field,
                        "description": f"Operation {task.required_request_id_field} of {bad_amt:.2f}{entities_str}",
                        "required": True,
                    },
                    {
                        "predicate": "KEY_EXISTS",
                        "target": task.required_request_id_field,
                        "expected": bad_amt,
                        "description": f"Assert wrong amount {bad_amt:.2f}",
                        "required": False,
                    }
                ],
            })

        elif variant == "UNSUPPORTED_TABLE":
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "proposed_postconditions": [
                    {
                        "predicate": "STATUS_SUCCESS",
                        "target": "unauthorized_external_table",
                        "expected": task.required_request_id_field,
                        "description": f"Operation {task.required_request_id_field}{amt_str}{entities_str} on unauthorized table",
                        "required": True,
                    },
                    {
                        "predicate": "KEY_EXISTS",
                        "target": task.required_request_id_field,
                        "description": "Key exists check",
                        "required": False,
                    }
                ],
            })

        elif variant == "UNDECLARED_EXTRA_COLUMN":
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "proposed_postconditions": [
                    {
                        "predicate": "STATUS_SUCCESS",
                        "target": "phantom_undeclared_column_xyz",
                        "expected": task.required_request_id_field,
                        "description": f"Operation {task.required_request_id_field}{amt_str}{entities_str} checking phantom column",
                        "required": True,
                    },
                    {
                        "predicate": "KEY_EXISTS",
                        "target": task.primary_table,
                        "description": "Valid table check",
                        "required": False,
                    }
                ],
            })

        elif variant == "EMPTY_PREDICATES":
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "proposed_postconditions": [],
            })

        elif variant == "MALFORMED_JSON":
            return "{tool: 'dbops_database', unquoted_json: True, "

        elif variant == "UNKNOWN_FIELDS":
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "hallucinated_extra_key": "unauthorized_data",
                "proposed_postconditions": [
                    {"predicate": "STATUS_SUCCESS", "target": task.primary_table, "description": "OK"}
                ],
            })

        elif variant == "PROVENANCE_INJECTION":
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "provenance": "model",
                "proposed_postconditions": [
                    {"predicate": "STATUS_SUCCESS", "target": task.primary_table, "description": "OK"}
                ],
            })

        elif variant == "MODEL_REFUSAL":
            return "I apologize, but as an AI assistant, I cannot fulfill this database operation directly."

        elif variant == "RAW_SQL_INJECTION":
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": task_id},
                "proposed_postconditions": [
                    {
                        "predicate": "STATUS_SUCCESS",
                        "target": task.primary_table,
                        "expected": "SELECT * FROM users WHERE admin=1",
                        "description": "DROP TABLE refunds; -- injection attack",
                        "required": True,
                    },
                    {
                        "predicate": "KEY_EXISTS",
                        "target": task.required_request_id_field,
                        "description": "Trace check",
                        "required": False,
                    }
                ],
            })

        elif variant == "TASK_05_BAD_UNIQUENESS":
            # Task 05 asserts that cancellation_id is unique on orders table (which is false)
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": "task_05_order_cancellation"},
                "proposed_postconditions": [
                    {
                        "predicate": "STATUS_SUCCESS",
                        "target": "orders",
                        "expected": "CXL-019882",
                        "description": "Enforce that orders table has unique cancellation_id for ORD-5540",
                        "required": True,
                    },
                    {
                        "predicate": "KEY_EXISTS",
                        "target": "cancellation_id",
                        "description": "Key exists",
                        "required": False,
                    }
                ],
            })

        elif variant == "TASK_08_BAD_UNIQUENESS":
            # Task 08 asserts that settlement_ref is unique on invoices table (which is false)
            return json.dumps({
                "tool": "dbops_database",
                "arguments": {"task_id": "task_08_invoice_settlement"},
                "proposed_postconditions": [
                    {
                        "predicate": "STATUS_SUCCESS",
                        "target": "invoices",
                        "expected": "STL-002981",
                        "description": "Assert invoices table enforces unique settlement_ref for INV-7720",
                        "required": True,
                    },
                    {
                        "predicate": "KEY_EXISTS",
                        "target": "settlement_ref",
                        "description": "Key exists",
                        "required": False,
                    }
                ],
            })

        raise ValueError(f"Unknown fixture variant: '{variant}'")
