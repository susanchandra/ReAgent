"""
fixture_replay.py — Offline Fixture Replay & Schema Validation Harness for DBOPS-M1.

Replays recorded/fixed model-response fixtures through the actual structured-decision
parsing (_validate_structured_decision) and ContractGroundingValidator path.

Tests both valid fixtures and diverse negative failure categories:
1. VALID: Schema-compliant and properly grounded postconditions.
2. MALFORMED_JSON: Non-JSON raw string response.
3. SCHEMA_INVALID_UNKNOWN_KEYS: Response containing forbidden/unexpected keys.
4. SCHEMA_INVALID_PROVENANCE_INJECTED: Model attempting to forge its own provenance.
5. SCHEMA_INVALID_MISSING_TOOL: Payload missing required 'tool' field.
6. SCHEMA_INVALID_EMPTY_TOOL: Payload with whitespace/empty 'tool' field.
7. UNGROUNDED_OBJECTIVE: Postconditions not grounded in the task objective.
8. WRONG_TASK_TARGET: Postcondition references entities belonging to a different task.
9. STATUS_ONLY_WRITE: State-changing write operation proposing only STATUS_SUCCESS.
10. UNSUPPORTED_PREDICATE_FILESYSTEM: Incompatible FILE_* predicate on database tool.

Zero network access. Zero model provider calls. Zero live API access.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from reagent.models import StructuredDecision, Postcondition
from reagent.contract_validator import ContractGroundingValidator, ValidationResult as GroundingValidationResult
from reagent.providers.nebius import (
    NebiusNemotronProvider,
    ALLOWED_STRUCTURED_DECISION_KEYS,
)
from reagent.tools.base import BaseTool
from reagent.experiments.dbops.integration import DbopsDatabaseTool
from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.tasks import get_task, ALL_TASKS


@dataclass
class ReplayOutcome:
    fixture_id: str
    task_id: str
    category: str
    parsed_json_ok: bool
    schema_valid: bool
    schema_error: Optional[str]
    structured_decision: Optional[StructuredDecision]
    grounding_valid: bool
    grounding_errors: List[str]
    overall_passed: bool
    failure_reason: Optional[str] = None


class OfflineFixtureReplayEngine:
    """
    Engine that evaluates response fixtures through ReAgent's real parser
    and contract grounding validator.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root
        self.grounding_validator = ContractGroundingValidator(workspace_root=workspace_root)
        self.unconfigured_provider = NebiusNemotronProvider(client=None)

    def replay_fixture(
        self,
        fixture_id: str,
        task_id: str,
        category: str,
        raw_response_text: str,
        tool: Optional[BaseTool] = None,
    ) -> ReplayOutcome:
        """
        Replay a single raw response text through JSON parse, schema validation,
        and ContractGroundingValidator.
        """
        task = get_task(task_id)
        store = None
        if tool is None:
            # Create isolated temporary tool instance
            store = DbopsStore.create_temp()
            tool = DbopsDatabaseTool(store)

        try:
            # Step 1: JSON Parsing
            try:
                parsed = json.loads(raw_response_text)
                parsed_ok = isinstance(parsed, dict)
                if not parsed_ok:
                    return ReplayOutcome(
                        fixture_id=fixture_id,
                        task_id=task_id,
                        category=category,
                        parsed_json_ok=False,
                        schema_valid=False,
                        schema_error="JSON root must be an object",
                        structured_decision=None,
                        grounding_valid=False,
                        grounding_errors=[],
                        overall_passed=False,
                        failure_reason="MALFORMED_JSON: Non-dictionary JSON root",
                    )
            except Exception as e:
                return ReplayOutcome(
                    fixture_id=fixture_id,
                    task_id=task_id,
                    category=category,
                    parsed_json_ok=False,
                    schema_valid=False,
                    schema_error=str(e),
                    structured_decision=None,
                    grounding_valid=False,
                    grounding_errors=[],
                    overall_passed=False,
                    failure_reason=f"MALFORMED_JSON: {e}",
                )

            # Step 2: Schema validation via NebiusNemotronProvider._validate_structured_decision
            decision = self.unconfigured_provider._validate_structured_decision(parsed)
            if decision is None:
                schema_err = self._diagnose_schema_error(parsed)
                return ReplayOutcome(
                    fixture_id=fixture_id,
                    task_id=task_id,
                    category=category,
                    parsed_json_ok=True,
                    schema_valid=False,
                    schema_error=schema_err,
                    structured_decision=None,
                    grounding_valid=False,
                    grounding_errors=[],
                    overall_passed=False,
                    failure_reason=f"SCHEMA_INVALID: {schema_err}",
                )

            # Step 3: Grounding validation via ContractGroundingValidator
            grounding_res = self.grounding_validator.validate(
                decision=decision,
                tool=tool,
                objective=task.human_description,
            )

            overall = grounding_res.is_valid
            fail_reason = None
            if not overall:
                fail_reason = f"GROUNDING_REJECTED: {'; '.join(grounding_res.errors)}"

            return ReplayOutcome(
                fixture_id=fixture_id,
                task_id=task_id,
                category=category,
                parsed_json_ok=True,
                schema_valid=True,
                schema_error=None,
                structured_decision=decision,
                grounding_valid=grounding_res.is_valid,
                grounding_errors=grounding_res.errors,
                overall_passed=overall,
                failure_reason=fail_reason,
            )
        finally:
            if store is not None:
                store.cleanup()

    def _diagnose_schema_error(self, raw: Dict[str, Any]) -> str:
        """Diagnose exact schema failure reason for diagnostic reporting."""
        if not isinstance(raw, dict):
            return "Payload is not a dict"
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
        return "Unknown schema rejection"


# ── Canonical Fixture Generators for Testing ─────────────────────────────────

def generate_valid_fixture(task_id: str) -> str:
    """Generate a canonical synthetically authored valid fixture for a task."""
    task = get_task(task_id)
    rid = task.required_request_id_field
    amount_str = f" of {task.prompt_numeric_amount}" if task.prompt_numeric_amount else ""
    
    entities_str = ""
    if task.prompt_source_id:
        entities_str += f" from {task.prompt_source_id}"
    if task.prompt_destination_id:
        entities_str += f" to {task.prompt_destination_id}"

    payload = {
        "tool": "dbops_database",
        "arguments": {
            "task_id": task.task_id,
            "fault_mode": "NO_FAULT",
        },
        "proposed_postconditions": [
            {
                "predicate": "STATUS_SUCCESS",
                "target": task.primary_table,
                "expected": task.required_request_id_field,
                "description": f"Execute {task.task_id} with request {task.required_request_id_field}{amount_str}{entities_str}",
                "required": True,
            },
            {
                "predicate": "KEY_EXISTS",
                "target": task.required_request_id_field,
                "expected": task.prompt_source_id or task.primary_table,
                "description": f"Verify trace key {task.required_request_id_field} for {task.primary_table}",
                "required": False,
            },
        ],
        "diagnosis_category": None,
        "diagnosis_confidence": 0.0,
        "diagnosis_reason": f"Grounded synthetic fixture for {task_id}",
        "recovery_strategy": None,
    }
    return json.dumps(payload, indent=2)


def generate_negative_fixtures(task_id: str) -> Dict[str, str]:
    """Generate diverse negative test fixtures for a task."""
    task = get_task(task_id)
    
    return {
        "MALFORMED_JSON": (
            "```json\n"
            "This is clearly not valid JSON output from a model"
        ),
        "SCHEMA_INVALID_PROVENANCE_INJECTED": json.dumps({
            "tool": "dbops_database",
            "arguments": {"task_id": task_id},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
            "provenance": "model_injected_forbidden_field",
        }),
        "SCHEMA_INVALID_UNKNOWN_KEYS": json.dumps({
            "tool": "dbops_database",
            "arguments": {"task_id": task_id},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
            "hallucinated_extra_key": "some_extra_data",
        }),
        "SCHEMA_INVALID_MISSING_TOOL": json.dumps({
            "arguments": {"task_id": task_id},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
        }),
        "SCHEMA_INVALID_EMPTY_TOOL": json.dumps({
            "tool": "   ",
            "arguments": {"task_id": task_id},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
        }),
        "STATUS_ONLY_WRITE": json.dumps({
            "tool": "dbops_database",
            "arguments": {"task_id": task_id},
            "proposed_postconditions": [
                {
                    "predicate": "STATUS_SUCCESS",
                    "target": task.primary_table,
                    "description": f"Execute {task.task_id}",
                    "required": True,
                }
            ],
        }),
        "UNGROUNDED_OBJECTIVE": json.dumps({
            "tool": "dbops_database",
            "arguments": {"task_id": task_id},
            "proposed_postconditions": [
                {
                    "predicate": "STATUS_SUCCESS",
                    "target": "arbitrary_other_table",
                    "description": "Completely unrelated operation about solar flares",
                    "required": True,
                },
                {
                    "predicate": "KEY_EXISTS",
                    "target": "solar_flare_intensity",
                    "description": "Magnetic field reading",
                    "required": True,
                },
            ],
        }),
        "WRONG_TASK_TARGET": json.dumps({
            "tool": "dbops_database",
            "arguments": {"task_id": "task_12_schema_migration" if task_id != "task_12_schema_migration" else "task_01_refund_issue"},
            "proposed_postconditions": [
                {
                    "predicate": "STATUS_SUCCESS",
                    "target": "schema_migrations" if task_id != "task_12_schema_migration" else "refunds",
                    "description": "Completely wrong target task postcondition",
                    "required": True,
                },
                {
                    "predicate": "KEY_EXISTS",
                    "target": "version" if task_id != "task_12_schema_migration" else "refund_id",
                    "description": "Wrong entity key",
                    "required": True,
                },
            ],
        }),
        "UNSUPPORTED_PREDICATE_FILESYSTEM": json.dumps({
            "tool": "dbops_database",
            "arguments": {"task_id": task_id},
            "proposed_postconditions": [
                {
                    "predicate": "FILE_EXISTS",
                    "target": "nonexistent_file.txt",
                    "description": "Checking file on disk for a database tool",
                    "required": True,
                },
            ],
        }),
    }
