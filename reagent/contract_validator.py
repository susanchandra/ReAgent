"""
ReAgent Contract Grounding & Semantic Validator (Final Production Release)

Enforces concrete, deterministic grounding rules BEFORE a SuccessContract is frozen
and BEFORE any tool invocation occurs.

Hard Rejection Rules:
  1. TARGET BINDING: A required file predicate targeting a different file from the
     validated tool file argument is REJECTED before tool execution.
  2. VACUITY REJECTION: A required predicate already satisfied on disk before action
     execution (e.g. FILE_EXISTS on an existing file, or FILE_CONTAINS with substring
     already present) makes the claimed postcondition vacuous and is REJECTED.
  3. OBJECTIVE GROUNDING: Matching only tool arguments is insufficient. At least one
     required predicate must be meaningfully grounded in the task objective text itself.
  4. TOOL COMPATIBILITY: FILE_* predicates require a tool compatible with filesystem operations.
  5. EFFECT CLASS: State-changing tools require >= 1 independent observable predicate
     (STATUS_SUCCESS alone is prohibited for state-changing tools).
  6. PREDICATE CAPS: 1 <= count <= MAX_PREDICATES (10).
  7. SUPPORTED PREDICATES: All predicates must belong to SUPPORTED_PREDICATES.
  8. PREDICATE FIELD REQUIREMENTS: KEY_EXISTS requires target; VALUE_EQUALS requires target
     and non-null expected; FILE_* requires valid non-empty path; HTTP_STATUS requires int code.
  9. GENERIC POSTCONDITION REJECTION: A generic STATUS_SUCCESS postcondition with null target
     and null expected ("Action must succeed") is rejected when the workflow requires concrete
     observable state verification.
  10. OBSERVER CAPABILITY: Postcondition targets must map to observable state properties
      of the tool/probe (e.g. ALLOWED_HOTEL_OBSERVABLE_KEYS for hotel_booking_http).
  11. TASK IDENTIFIER BINDING: Security-critical identifiers (booking_id, room_number, txn_id)
      must match trusted task context. Model proposals cannot alter or forge target entities.
  12. PROVENANCE FORGERY: Model cannot claim EXECUTED or VERIFIED provenance.

Read-only tools (effect_class == "read-only") remain valid with STATUS_SUCCESS or output-key checks
provided they do not fail predicate validation or vacuity checks.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Union

from reagent.models import Postcondition, StructuredDecision
from reagent.tools.base import BaseTool
from reagent.verification.verifier import DeterministicVerifier

# Predicate types that probe the filesystem (independent of the tool's self-report)
FILE_PREDICATE_TYPES = {"FILE_EXISTS", "FILE_CONTAINS", "FILE_SIZE_GTE"}

# Maximum allowed predicates per contract
MAX_PREDICATES = 10

# Predicate types that are purely status-based (not independently observable)
STATUS_ONLY_PREDICATES = {"STATUS_SUCCESS"}

# Supported predicates recognized by ReAgent's DeterministicVerifier
SUPPORTED_PREDICATES = frozenset({
    "STATUS_SUCCESS",
    "KEY_EXISTS",
    "VALUE_EQUALS",
    "FILE_EXISTS",
    "FILE_CONTAINS",
    "FILE_SIZE_GTE",
    "JSON_VALID",
    "JSON_PATH_EQUALS",
    "HTTP_STATUS",
    "NO_ERROR_PAYLOAD",
})

# Observable fields exposed by HotelBookingHttpTool / simulator probe
ALLOWED_HOTEL_OBSERVABLE_KEYS = frozenset({
    "booking_id",
    "booking_reference",
    "room_number",
    "guest_name",
    "amount",
    "status",
    "reservation_status",
    "committed_at",
    "audit_logged",
    "reservations",
})

# Generic descriptions that provide no meaningful observable assertion
GENERIC_STATUS_DESCRIPTIONS = frozenset({
    "",
    "action must succeed",
    "action succeeded",
    "tool must succeed",
    "tool succeeded",
    "success",
    "done",
    "ok",
    "true",
    "action success",
    "execution success",
    "must succeed",
})

# Allowed tools per workflow
WORKFLOW_ALLOWED_TOOLS: Dict[str, Set[str]] = {
    "hotel_booking": {"hotel_booking_http"},
    "wallet_transfer": {"wallet_transfer", "dbops_read_only", "dbops_database"},
    "database_update": {"dbops_database", "dbops_read_only"},
}


@dataclass
class ValidationResult:
    """Result of ContractGroundingValidator.validate() or validate_contract_proposal()."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MockToolWrapper(BaseTool):
    """Fallback tool wrapper for validating proposals when full tool instance is not provided."""

    def __init__(self, name: str, effect_class: str = "read-only"):
        self.name = name
        self.description = f"Proposal validator wrapper for tool {name}"
        self.effect_class = effect_class
        self.parameters_schema = {"type": "object", "properties": {}}

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class ContractGroundingValidator:
    """
    Validates that a proposed SuccessContract is grounded in the task objective,
    tool arguments, and independent observer capabilities before execution.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        """
        Args:
            workspace_root: Passed to DeterministicVerifier for sandbox / vacuity path checks.
        """
        self.workspace_root = workspace_root
        self._verifier = DeterministicVerifier(workspace_root=workspace_root)

    def validate(
        self,
        decision: StructuredDecision,
        tool: BaseTool,
        objective: str,
        task_context: Optional[Dict[str, Any]] = None,
        workflow: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate the proposed postconditions in a StructuredDecision.

        Rejects invalid contracts BEFORE tool execution.
        """
        errors: List[str] = []
        warnings: List[str] = []
        pcs = decision.proposed_postconditions
        args = decision.arguments or {}

        # ── CAP: min 1, max MAX_PREDICATES ────────────────────────────
        if len(pcs) == 0:
            errors.append("Contract must have at least 1 postcondition predicate.")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        if len(pcs) > MAX_PREDICATES:
            errors.append(
                f"Contract has {len(pcs)} predicates; maximum allowed is {MAX_PREDICATES}."
            )

        # ── PROVENANCE FORGERY CHECK ──────────────────────────────────
        prov = getattr(decision, "provenance", None)
        if prov in ("EXECUTED", "VERIFIED"):
            errors.append(
                f"ProvenanceForgeryRejected: model cannot claim provenance='{prov}'."
            )

        # ── WORKFLOW TOOL ALLOWLIST CHECK ─────────────────────────────
        if workflow and workflow in WORKFLOW_ALLOWED_TOOLS:
            allowed = WORKFLOW_ALLOWED_TOOLS[workflow]
            if tool.name not in allowed:
                errors.append(
                    f"DisallowedWorkflowTool: tool '{tool.name}' is not permitted for workflow '{workflow}'. "
                    f"Allowed tools: {sorted(allowed)}"
                )

        # ── TOOL ARGUMENTS SCHEMA VALIDATION ──────────────────────────
        if tool.name == "hotel_booking_http":
            if not isinstance(args, dict):
                errors.append("SchemaInvalidArguments: arguments must be a dict.")
            else:
                bkg = args.get("booking_id") or args.get("booking_reference")
                room = args.get("room_number")
                if not bkg or not str(bkg).strip():
                    errors.append("SchemaInvalidArguments: hotel_booking_http requires non-empty 'booking_id'.")
                if not room or not str(room).strip():
                    errors.append("SchemaInvalidArguments: hotel_booking_http requires non-empty 'room_number'.")
                if "amount" in args:
                    try:
                        amt = float(args["amount"])
                        if amt <= 0:
                            errors.append("SchemaInvalidArguments: amount must be positive.")
                    except (ValueError, TypeError):
                        errors.append("SchemaInvalidArguments: amount must be a number.")

        # ── TASK CONTEXT IDENTIFIER BINDING (ARGUMENTS) ───────────────
        if task_context and isinstance(args, dict):
            if "booking_id" in task_context:
                exp_bkg = str(task_context["booking_id"]).strip().lower()
                act_bkg = str(args.get("booking_id") or args.get("booking_reference") or "").strip().lower()
                if act_bkg and act_bkg != exp_bkg:
                    errors.append(
                        f"IdentifierBindingMismatch: proposed booking_id '{args.get('booking_id')}' "
                        f"does not match trusted task context '{task_context['booking_id']}'."
                    )
            if "room_number" in task_context:
                exp_room = str(task_context["room_number"]).strip().lower()
                act_room = str(args.get("room_number") or "").strip().lower()
                if act_room and act_room != exp_room:
                    errors.append(
                        f"IdentifierBindingMismatch: proposed room_number '{args.get('room_number')}' "
                        f"does not match trusted task context '{task_context['room_number']}'."
                    )
            if "txn_id" in task_context:
                exp_txn = str(task_context["txn_id"]).strip().lower()
                act_txn = str(args.get("txn_id") or "").strip().lower()
                if act_txn and act_txn != exp_txn:
                    errors.append(
                        f"IdentifierBindingMismatch: proposed txn_id '{args.get('txn_id')}' "
                        f"does not match trusted task context '{task_context['txn_id']}'."
                    )

        is_state_changing = getattr(tool, "effect_class", "state-changing") == "state-changing"

        # ── EFFECT CLASS: state-changing tools need >= 1 independent predicate ──
        if is_state_changing:
            independent = [
                p for p in pcs if p.predicate not in STATUS_ONLY_PREDICATES
            ]
            if not independent:
                errors.append(
                    f"Tool '{tool.name}' is state-changing but has no independent "
                    f"observable postcondition (only STATUS_SUCCESS). "
                    f"Add at least one FILE_EXISTS, FILE_CONTAINS, KEY_EXISTS, or similar predicate."
                )

        # ── TOOL COMPATIBILITY: FILE predicates require file-compatible tool ──
        has_file_predicate = any(p.predicate in FILE_PREDICATE_TYPES for p in pcs)
        tool_supports_files = (
            "file" in tool.name.lower()
            or any(k in args for k in ("relative_path", "path", "file_path", "filename", "target_path"))
            or any(k in tool.parameters_schema.get("properties", {}) for k in ("relative_path", "path", "file_path"))
        )
        if has_file_predicate and not tool_supports_files:
            errors.append(
                f"IncompatibleToolPredicate: contract contains filesystem predicates, "
                f"but tool '{tool.name}' does not declare or support filesystem operations."
            )

        # ── OBJECTIVE GROUNDING CHECK ─────────────────────────────────
        # For state-changing tools, at least one required predicate must be grounded
        # in the task objective itself — matching only tool arguments is insufficient.
        if is_state_changing:
            required_pcs = [p for p in pcs if getattr(p, "required", True)]
            objective_grounded = False
            for pc in required_pcs:
                if self._is_grounded_in_objective(pc, objective):
                    objective_grounded = True
                    break

            if required_pcs and not objective_grounded:
                errors.append(
                    f"UngroundedContract: no required postcondition predicate is meaningfully "
                    f"grounded in the task objective '{objective}'. Matching only tool arguments "
                    f"is insufficient to establish task relevance."
                )

        # ── PER-PREDICATE CHECKS ──────────────────────────────────────
        for pc in pcs:
            self._check_predicate(
                pc=pc,
                args=args,
                objective=objective,
                tool=tool,
                is_state_changing=is_state_changing,
                all_pcs=pcs,
                task_context=task_context,
                workflow=workflow,
                errors=errors,
                warnings=warnings,
            )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _is_grounded_in_objective(self, pc: Postcondition, objective: str) -> bool:
        """
        Check if a predicate has meaningful semantic grounding in the objective text.
        Extracts alphanumeric tokens (length >= 3) and checks for presence/overlap.
        """
        obj_clean = objective.lower()

        # Check target literal
        if pc.target:
            target_str = str(pc.target).lower().replace("\\", "/")
            basename = os.path.basename(target_str)
            if basename and basename in obj_clean:
                return True
            tokens = [t for t in re.findall(r"[a-zA-Z0-9_]{3,}", target_str)]
            if any(t in obj_clean for t in tokens):
                return True

        # Check expected literal
        if pc.expected is not None:
            exp_str = str(pc.expected).lower()
            if len(exp_str) >= 3 and exp_str in obj_clean:
                return True
            tokens = [t for t in re.findall(r"[a-zA-Z0-9_]{3,}", exp_str)]
            if any(t in obj_clean for t in tokens):
                return True

        # Check description
        if pc.description:
            desc_str = pc.description.lower()
            tokens = [t for t in re.findall(r"[a-zA-Z0-9_]{4,}", desc_str)]
            matched_tokens = [t for t in tokens if t in obj_clean and t not in ("must", "tool", "file", "action", "succ")]
            if len(matched_tokens) >= 2:
                return True

        return False

    def _check_predicate(
        self,
        pc: Postcondition,
        args: Dict[str, Any],
        objective: str,
        tool: BaseTool,
        is_state_changing: bool,
        all_pcs: List[Postcondition],
        task_context: Optional[Dict[str, Any]],
        workflow: Optional[str],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Evaluate predicate validity, target binding, observer capability, and vacuity."""
        is_required = getattr(pc, "required", True)

        # ── 1. SUPPORTED PREDICATE CHECK ─────────────────────────────
        if pc.predicate not in SUPPORTED_PREDICATES:
            errors.append(
                f"UnsupportedPredicate: '{pc.predicate}' is not a supported postcondition predicate. "
                f"Supported: {sorted(SUPPORTED_PREDICATES)}"
            )
            return

        # ── 2. PER-PREDICATE REQUIRED FIELD CHECKS ───────────────────
        if pc.predicate == "KEY_EXISTS":
            if not pc.target or not str(pc.target).strip():
                errors.append("PredicateFieldMissing: KEY_EXISTS requires a non-empty string target.")
        elif pc.predicate == "VALUE_EQUALS":
            if not pc.target or not str(pc.target).strip() or pc.expected is None:
                errors.append("PredicateFieldMissing: VALUE_EQUALS requires a non-empty string target and a non-null expected value.")
        elif pc.predicate == "FILE_EXISTS":
            if not (pc.target or pc.expected):
                errors.append("PredicateFieldMissing: FILE_EXISTS requires a non-empty path in target or expected.")
        elif pc.predicate == "FILE_CONTAINS":
            if not pc.target or pc.expected is None:
                errors.append("PredicateFieldMissing: FILE_CONTAINS requires a non-empty target path and non-null expected substring.")
        elif pc.predicate == "FILE_SIZE_GTE":
            if not pc.target or pc.expected is None:
                errors.append("PredicateFieldMissing: FILE_SIZE_GTE requires a non-empty target path and expected size.")
            else:
                try:
                    if int(pc.expected) < 0:
                        errors.append("InvalidPredicateField: FILE_SIZE_GTE expected size must be non-negative.")
                except (ValueError, TypeError):
                    errors.append("InvalidPredicateField: FILE_SIZE_GTE expected size must be an integer.")
        elif pc.predicate == "JSON_PATH_EQUALS":
            if not pc.target or pc.expected is None:
                errors.append("PredicateFieldMissing: JSON_PATH_EQUALS requires non-empty target path and non-null expected value.")
        elif pc.predicate == "HTTP_STATUS":
            if pc.expected is None:
                errors.append("PredicateFieldMissing: HTTP_STATUS requires expected status code integer.")
            else:
                try:
                    code = int(pc.expected)
                    if not (100 <= code <= 599):
                        errors.append("InvalidPredicateField: HTTP_STATUS expected code must be between 100 and 599.")
                except (ValueError, TypeError):
                    errors.append("InvalidPredicateField: HTTP_STATUS expected code must be an integer.")

        # ── 3. REJECTION OF GENERIC STATUS_SUCCESS ───────────────────
        if pc.predicate == "STATUS_SUCCESS":
            desc_clean = (pc.description or "").strip().lower()
            is_specific_generic_defect = (
                pc.target is None
                and pc.expected is None
                and desc_clean in ("action must succeed", "action succeeded", "must succeed")
            )
            if is_specific_generic_defect:
                all_status_only = all(p.predicate in STATUS_ONLY_PREDICATES for p in all_pcs)
                if (
                    all_status_only
                    or is_state_changing
                    or (tool and tool.name == "hotel_booking_http")
                    or (workflow == "hotel_booking")
                ):
                    errors.append(
                        f"GenericPostconditionRejected: postcondition 'STATUS_SUCCESS' with null target and "
                        f"null expected ('{pc.description or 'Action must succeed'}') lacks concrete observable state assertions."
                    )

        # ── 4. OBSERVER CAPABILITY / OBSERVABLE TARGETS ──────────────
        if tool and tool.name == "hotel_booking_http":
            if pc.target is not None:
                tgt_norm = str(pc.target).strip().lower()
                if tgt_norm not in ALLOWED_HOTEL_OBSERVABLE_KEYS:
                    errors.append(
                        f"UnobservablePostcondition: target '{pc.target}' is not an observable property "
                        f"of hotel_booking_http. Allowed observable fields: {sorted(ALLOWED_HOTEL_OBSERVABLE_KEYS)}"
                    )

        # ── 5. TASK CONTEXT IDENTIFIER BINDING (POSTCONDITIONS) ───────
        if task_context and pc.target:
            tgt_clean = str(pc.target).strip().lower()
            if tgt_clean in task_context and pc.expected is not None:
                exp_c = str(task_context[tgt_clean]).strip().lower()
                act_c = str(pc.expected).strip().lower()
                if act_c != exp_c:
                    errors.append(
                        f"IdentifierBindingMismatch: postcondition target '{pc.target}' expected '{pc.expected}' "
                        f"does not match trusted task context '{task_context[tgt_clean]}'."
                    )

        # ── 6. TARGET BINDING (State-changing filesystem tools) ───────
        if pc.predicate in FILE_PREDICATE_TYPES:
            tool_path_arg = self._get_tool_path_arg(args)
            predicate_path = pc.target or pc.expected
            if tool_path_arg and predicate_path:
                norm_tool = str(tool_path_arg).replace("\\", "/").strip("/").lower()
                norm_pred = str(predicate_path).replace("\\", "/").strip("/").lower()
                tool_base = os.path.basename(norm_tool)
                pred_base = os.path.basename(norm_pred)
                if norm_tool != norm_pred and tool_base != pred_base:
                    msg = (
                        f"TargetBindingMismatch: required predicate '{pc.predicate}' targets "
                        f"'{predicate_path}', but tool '{tool.name}' argument targets '{tool_path_arg}'."
                    )
                    if is_state_changing and is_required:
                        errors.append(msg)
                    else:
                        warnings.append(msg)

        # ── 7. FILE PREDICATE SANDBOX & VACUITY CHECKS ────────────────
        if pc.predicate in FILE_PREDICATE_TYPES:
            raw_path = pc.target or (str(pc.expected) if pc.expected else None)
            if raw_path:
                try:
                    safe = self._verifier._resolve_verified_path(raw_path)
                    if os.path.exists(safe):
                        if pc.predicate == "FILE_EXISTS":
                            msg = (
                                f"VacuousPostcondition: predicate 'FILE_EXISTS' on path '{raw_path}' "
                                f"is already true on disk before action execution. Proving file creation "
                                f"is impossible because the file already exists."
                            )
                            if is_state_changing and is_required:
                                errors.append(msg)
                            else:
                                warnings.append(msg)
                        elif pc.predicate == "FILE_CONTAINS":
                            substring = str(pc.expected or "")
                            if substring:
                                try:
                                    with open(safe, "r", encoding="utf-8", errors="replace") as fh:
                                        pre_content = fh.read(1024 * 1024)
                                    if substring in pre_content:
                                        msg = (
                                            f"VacuousPostcondition: predicate 'FILE_CONTAINS' on path '{raw_path}' "
                                            f"already contains expected content '{substring}' before action execution."
                                        )
                                        if is_state_changing and is_required:
                                            errors.append(msg)
                                        else:
                                            warnings.append(msg)
                                except OSError:
                                    pass
                except PermissionError:
                    errors.append(
                        f"Predicate '{pc.predicate}' path '{raw_path}' fails sandbox boundary check."
                    )

    def _get_tool_path_arg(self, args: Dict[str, Any]) -> Optional[str]:
        """Extract primary file path from tool arguments."""
        for key in ("relative_path", "path", "file_path", "filename", "target_path"):
            if key in args and isinstance(args[key], str):
                return args[key]
        return None


def validate_contract_proposal(
    contract_dict: Dict[str, Any],
    task_objective: str,
    workflow: Optional[str] = None,
    task_context: Optional[Dict[str, Any]] = None,
    tool: Optional[BaseTool] = None,
    workspace_root: Optional[str] = None,
) -> ValidationResult:
    """
    Validates a contract proposal dictionary before freezing or live execution.
    Checks:
      1. Structural schema (dict, required keys, no forged provenance).
      2. Tool allowlist for workflow.
      3. Argument validity and task context binding.
      4. Postcondition predicate support and required fields.
      5. Rejection of generic STATUS_SUCCESS (null target/expected).
      6. Observer-verifiable postcondition targets.
    """
    errors: List[str] = []

    if not isinstance(contract_dict, dict):
        return ValidationResult(is_valid=False, errors=["SchemaInvalid: contract must be a JSON object (dict)."])

    # Provenance forgery check
    prov = contract_dict.get("provenance")
    if prov in ("EXECUTED", "VERIFIED"):
        errors.append(f"ProvenanceForgeryRejected: model cannot claim provenance='{prov}'.")

    prov_str = str(prov) if prov is not None and str(prov).strip() else "advisory"

    tool_name = contract_dict.get("tool")
    if not tool_name or not isinstance(tool_name, str) or not tool_name.strip():
        errors.append("SchemaInvalid: 'tool' field is empty, missing, or not a string.")
        return ValidationResult(is_valid=False, errors=errors)

    tool_name = tool_name.strip()
    raw_args = contract_dict.get("arguments")
    if raw_args is None:
        args = {}
    elif isinstance(raw_args, dict):
        args = dict(raw_args)
    else:
        errors.append("SchemaInvalidArguments: 'arguments' must be a JSON object (dict).")
        return ValidationResult(is_valid=False, errors=errors)

    # Normalize booking_reference -> booking_id
    if "booking_reference" in args and "booking_id" not in args:
        args["booking_id"] = args["booking_reference"]

    # Resolve tool instance
    resolved_tool = tool
    if resolved_tool is None:
        if tool_name == "hotel_booking_http":
            from reagent.tools.hotel_tool import HotelBookingHttpTool
            resolved_tool = HotelBookingHttpTool()
        elif tool_name in ("dbops_database", "wallet_transfer"):
            resolved_tool = MockToolWrapper(name=tool_name, effect_class="state-changing")
        else:
            resolved_tool = MockToolWrapper(name=tool_name, effect_class="read-only")

    # Parse proposed postconditions
    raw_pcs = contract_dict.get("proposed_postconditions")
    if raw_pcs is None or not isinstance(raw_pcs, list):
        errors.append("SchemaInvalid: 'proposed_postconditions' must be a list.")
        return ValidationResult(is_valid=False, errors=errors)

    if len(raw_pcs) == 0:
        errors.append("Contract must have at least 1 postcondition predicate.")
        return ValidationResult(is_valid=False, errors=errors)

    pcs: List[Postcondition] = []
    for idx, p in enumerate(raw_pcs):
        if not isinstance(p, dict):
            errors.append(f"SchemaInvalid: postcondition [{idx}] must be a JSON object.")
            continue
        pred = p.get("predicate")
        if not pred or not isinstance(pred, str) or not pred.strip():
            errors.append(f"SchemaInvalid: postcondition [{idx}] has missing or invalid 'predicate'.")
            continue
        try:
            pc = Postcondition(
                predicate=pred.strip(),
                target=p.get("target"),
                expected=p.get("expected"),
                description=p.get("description", ""),
                required=bool(p.get("required", True)),
                postcondition_id=p.get("postcondition_id"),
            )
            pcs.append(pc)
        except Exception as exc:
            errors.append(f"SchemaInvalid: postcondition [{idx}] failed construction: {exc}")

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    decision = StructuredDecision(
        tool=tool_name,
        arguments=args,
        proposed_postconditions=pcs,
        diagnosis_category=contract_dict.get("diagnosis_category"),
        diagnosis_confidence=float(contract_dict.get("diagnosis_confidence", 0.0) or 0.0),
        diagnosis_reason=contract_dict.get("diagnosis_reason"),
        recovery_strategy=contract_dict.get("recovery_strategy"),
        provenance=prov_str,
    )

    validator = ContractGroundingValidator(workspace_root=workspace_root)
    return validator.validate(
        decision=decision,
        tool=resolved_tool,
        objective=task_objective,
        task_context=task_context,
        workflow=workflow,
    )
