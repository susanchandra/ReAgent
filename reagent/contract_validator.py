"""
ReAgent Contract Grounding Validator  (Milestone 4B revision)

Enforces concrete, deterministic grounding rules BEFORE a SuccessContract is frozen
and BEFORE any tool invocation occurs.

Hard Rejection Rules for state-changing tools:
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

Read-only tools (effect_class == "read-only") remain valid with STATUS_SUCCESS or output-key checks.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set

from reagent.models import Postcondition, StructuredDecision
from reagent.tools.base import BaseTool
from reagent.verification.verifier import DeterministicVerifier

# Predicate types that probe the filesystem (independent of the tool's self-report)
FILE_PREDICATE_TYPES = {"FILE_EXISTS", "FILE_CONTAINS", "FILE_SIZE_GTE"}

# Maximum allowed predicates per contract
MAX_PREDICATES = 10

# Predicate types that are purely status-based (not independently observable)
STATUS_ONLY_PREDICATES = {"STATUS_SUCCESS"}


@dataclass
class ValidationResult:
    """Result of ContractGroundingValidator.validate()."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ContractGroundingValidator:
    """
    Validates that a proposed SuccessContract is grounded in the task objective
    and tool arguments before execution.
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
    ) -> ValidationResult:
        """
        Validate the proposed postconditions in a StructuredDecision.

        Rejects invalid contracts BEFORE tool execution.
        """
        errors: List[str] = []
        warnings: List[str] = []
        pcs = decision.proposed_postconditions
        args = decision.arguments or {}

        # ── CAP: min 1, max MAX_PREDICATES ──────────────────────────────
        if len(pcs) == 0:
            errors.append("Contract must have at least 1 postcondition predicate.")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        if len(pcs) > MAX_PREDICATES:
            errors.append(
                f"Contract has {len(pcs)} predicates; maximum allowed is {MAX_PREDICATES}."
            )

        is_state_changing = getattr(tool, "effect_class", "state-changing") == "state-changing"

        # ── EFFECT CLASS: state-changing tools need >= 1 independent predicate ─
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

        # ── OBJECTIVE GROUNDING CHECK ─────────────────────────────────────
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

        # ── PER-PREDICATE CHECKS ──────────────────────────────────────────
        for pc in pcs:
            self._check_predicate(pc, args, objective, tool, is_state_changing, errors, warnings)

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
            # Basename check (e.g. "report.txt")
            basename = os.path.basename(target_str)
            if basename and basename in obj_clean:
                return True
            # Token overlap
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
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Evaluate target binding, sandbox boundaries, and vacuity for a predicate."""
        is_required = getattr(pc, "required", True)

        # ── TARGET BINDING (State-changing tools) ────────────────────────
        # A required file predicate targeting a different file from the tool argument is REJECTED.
        if pc.predicate in FILE_PREDICATE_TYPES:
            tool_path_arg = self._get_tool_path_arg(args)
            predicate_path = pc.target or pc.expected
            if tool_path_arg and predicate_path:
                norm_tool = str(tool_path_arg).replace("\\", "/").strip("/").lower()
                norm_pred = str(predicate_path).replace("\\", "/").strip("/").lower()
                # Check basename or path equivalence
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

        # ── FILE PREDICATE SANDBOX & VACUITY CHECKS ──────────────────────
        if pc.predicate in FILE_PREDICATE_TYPES:
            raw_path = pc.target or (str(pc.expected) if pc.expected else None)
            if raw_path:
                try:
                    safe = self._verifier._resolve_verified_path(raw_path)
                    # Vacuity check: If file already exists before state-changing execution
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
                            # Check if content is ALREADY in the file before execution
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
