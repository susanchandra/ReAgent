"""
ReAgent End-to-End Runner  (Milestone 3)

Exercises the complete frozen loop against real local tools (no Nebius required):
  TASK_RECEIVED -> CONTRACT_FROZEN -> POLICY_ALLOWED -> TOOL_EXECUTED ->
  EVIDENCE_COLLECTED -> VERIFIED_RECEIPT

Also exercises the recovery path:
  VERIFICATION_FAILED -> FAILURE_ATTRIBUTED -> RECOVERY_SELECTED ->
  RECOVERY_ALLOWED -> RECOVERY_EXECUTED -> REVERIFICATION -> VERIFIED_RECEIPT

Run with:
  python -m reagent.e2e

Optional Nebius integration (requires NEBIUS_API_KEY):
  NEBIUS_API_KEY=<key> python -m reagent.e2e --live
"""

import sys
import os
import json
import tempfile

from reagent.pipeline import ReAgentPipeline, ExecutionCycleResult
from reagent.models import TaskState, Postcondition, SuccessContract
from reagent.tools.registry import ToolRegistry
from reagent.tools.json_tool import JsonValidatorTool
from reagent.tools.data_tool import DataTransformTool
from reagent.tools.file_tool import SandboxedFileTool
from reagent.taxonomy import AgentStatus
from reagent.events import (
    PHASE_CONTRACT_FROZEN, PHASE_VERIFIED_RECEIPT, PHASE_VERIFICATION_FAILED,
    PHASE_RECOVERY_EXECUTED, PHASE_REVERIFICATION,
)


def _banner(text: str) -> None:
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def _check(label: str, cond: bool) -> None:
    mark = "[PASS]" if cond else "[FAIL]"
    print(f"  {mark} {label}")
    if not cond:
        raise AssertionError(f"E2E check failed: {label}")


def _print_timeline(result: ExecutionCycleResult) -> None:
    print("\n  Timeline:")
    for ev in result.timeline:
        print(f"    {ev.phase:<30} {ev.detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Happy path — JSON validation with STATUS_SUCCESS contract
# ─────────────────────────────────────────────────────────────────────────────

def run_happy_path() -> None:
    _banner("E2E #1: Happy Path (STATUS_SUCCESS contract)")

    registry = ToolRegistry()
    registry.register(JsonValidatorTool())
    registry.register(DataTransformTool())

    pipeline = ReAgentPipeline(tool_registry=registry)
    task = TaskState(objective="Validate a well-formed JSON payload.")

    result = pipeline.execute_with_contract(
        task=task,
        tool_name="json_validator",
        params={"json_payload": '{"status": "ok", "score": 42}'},
        postconditions=[
            Postcondition(predicate="STATUS_SUCCESS", description="Tool must succeed"),
            Postcondition(predicate="KEY_EXISTS", target="is_valid", description="'is_valid' key must be present"),
            Postcondition(predicate="VALUE_EQUALS", target="is_valid", expected=True, description="JSON must be valid"),
        ],
    )

    _print_timeline(result)
    _check("Contract frozen before execution", result.frozen_contract.frozen_at != "")
    _check("Tool executed successfully", result.action_result.success)
    _check("All postconditions verified", result.verification_result.verified)
    _check("VerifiedReceipt status = VERIFIED", result.verified_receipt.final_status == AgentStatus.VERIFIED)
    _check("CONTRACT_FROZEN in timeline", any(e.phase == PHASE_CONTRACT_FROZEN for e in result.timeline))
    _check("VERIFIED_RECEIPT in timeline", any(e.phase == PHASE_VERIFIED_RECEIPT for e in result.timeline))
    _check("Recovery count = 0", result.recovery_count == 0)

    print("\n  --> Happy path PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: False-success detection — tool reports success but postcondition fails
# ─────────────────────────────────────────────────────────────────────────────

def run_false_success_detection() -> None:
    _banner("E2E #2: False-Success Detection")

    registry = ToolRegistry()
    registry.register(JsonValidatorTool())

    pipeline = ReAgentPipeline(tool_registry=registry)
    task = TaskState(objective="Validate JSON and check specific field value.")

    result = pipeline.execute_with_contract(
        task=task,
        tool_name="json_validator",
        params={"json_payload": '{"status": "ok"}'},
        postconditions=[
            Postcondition(predicate="STATUS_SUCCESS", description="Tool succeeds"),
            # This assertion WILL fail — "score" key won't be in validator output
            Postcondition(predicate="KEY_EXISTS", target="score", description="Score must exist"),
        ],
        max_recovery_attempts=0,    # No recovery — let it escalate immediately
    )

    _print_timeline(result)
    _check("VERIFICATION_FAILED in timeline", any(e.phase == PHASE_VERIFICATION_FAILED for e in result.timeline))
    _check("Receipt not VERIFIED (false success detected)", result.verified_receipt.final_status != AgentStatus.VERIFIED)
    _check("Discrepancies recorded", len(result.verification_result.discrepancies) > 0)

    print("\n  --> False-success detection PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: FILE_EXISTS + FILE_CONTAINS contract with SandboxedFileTool
# ─────────────────────────────────────────────────────────────────────────────

def run_file_contract() -> None:
    _banner("E2E #3: FILE_EXISTS + FILE_CONTAINS Contract")

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ToolRegistry()
        file_tool = SandboxedFileTool(sandbox_root=tmpdir)
        registry.register(file_tool)

        pipeline = ReAgentPipeline(tool_registry=registry)
        target_file = os.path.join(tmpdir, "reagent_output.txt")
        task = TaskState(objective="Write a verified output file.")

        result = pipeline.execute_with_contract(
            task=task,
            tool_name="sandboxed_file",
            params={"action": "write", "relative_path": "reagent_output.txt", "content": "REAGENT_VERIFIED"},
            postconditions=[
                Postcondition(predicate="STATUS_SUCCESS", description="Write must succeed"),
                Postcondition(predicate="FILE_EXISTS", expected=target_file, description="Output file must exist"),
                Postcondition(predicate="FILE_CONTAINS", target=target_file, expected="REAGENT_VERIFIED",
                              description="File must contain verification marker"),
                Postcondition(predicate="FILE_SIZE_GTE", target=target_file, expected=1,
                              description="File must not be empty"),
            ],
        )

        _print_timeline(result)
        _check("File write succeeded", result.action_result.success)
        _check("FILE_EXISTS postcondition passed", result.verification_result.verified or
               all("FILE_EXISTS" not in d for d in result.verification_result.discrepancies))
        _check("All postconditions verified", result.verification_result.verified)
        _check("VERIFIED receipt", result.verified_receipt.final_status == AgentStatus.VERIFIED)

    print("\n  --> File contract PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Recoverable failure — bad JSON repaired on retry
# ─────────────────────────────────────────────────────────────────────────────

def run_recoverable_failure() -> None:
    _banner("E2E #4: Recoverable Failure (JSON repair via retry)")

    class FixingJsonTool(JsonValidatorTool):
        """First call fails (bad JSON). Second call succeeds (good JSON). Simulates recovery."""
        _call_count = 0

        def execute(self, params):
            self._call_count += 1
            if self._call_count == 1:
                return super().execute({"json_payload": "NOT VALID JSON {{{"})
            return super().execute({"json_payload": '{"repaired": true}'})

    registry = ToolRegistry()
    fixer = FixingJsonTool()
    fixer._name = "json_validator"
    registry.register(fixer)

    pipeline = ReAgentPipeline(tool_registry=registry)
    task = TaskState(objective="Validate JSON with one failure then recovery.")

    result = pipeline.execute_with_contract(
        task=task,
        tool_name="json_validator",
        params={"json_payload": "NOT VALID JSON {{{"},
        postconditions=[
            Postcondition(predicate="STATUS_SUCCESS", description="Tool must succeed"),
        ],
        max_recovery_attempts=1,
    )

    _print_timeline(result)
    phases = [e.phase for e in result.timeline]
    _check("Initial verification failed",    PHASE_VERIFICATION_FAILED in phases)
    _check("Recovery executed",              PHASE_RECOVERY_EXECUTED in phases)
    _check("Reverification fired",           PHASE_REVERIFICATION in phases)
    _check("Final status VERIFIED",          result.verified_receipt.final_status == AgentStatus.VERIFIED)
    _check("Recovery count = 1",             result.recovery_count == 1)

    print("\n  --> Recoverable failure PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Escalation — verification fails, max retries exhausted
# ─────────────────────────────────────────────────────────────────────────────

def run_escalation_path() -> None:
    _banner("E2E #5: Escalation (all recovery attempts exhausted)")

    registry = ToolRegistry()
    registry.register(JsonValidatorTool())

    pipeline = ReAgentPipeline(tool_registry=registry)
    task = TaskState(objective="Trigger escalation path.")

    result = pipeline.execute_with_contract(
        task=task,
        tool_name="json_validator",
        params={"json_payload": "ALWAYS_INVALID{{{{"},
        postconditions=[
            Postcondition(predicate="STATUS_SUCCESS", description="Must succeed -- but won't"),
        ],
        max_recovery_attempts=1,
    )

    _print_timeline(result)
    _check("VERIFIED_RECEIPT emitted",  any(e.phase == PHASE_VERIFIED_RECEIPT for e in result.timeline))
    _check("Final status ESCALATED",   result.verified_receipt.final_status == AgentStatus.ESCALATED)
    _check("Reliability escalations incremented",
           pipeline.reliability_state.escalations >= 1)

    print("\n  --> Escalation path PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Optional: Live Nebius integration
# ─────────────────────────────────────────────────────────────────────────────

def run_live_nebius() -> None:
    _banner("E2E LIVE: Nebius / Nemotron structured decision")
    api_key = os.environ.get("NEBIUS_API_KEY", "")
    if not api_key:
        print("  SKIPPED: NEBIUS_API_KEY not set.\n")
        return

    from reagent.providers.nebius import NebiusNemotronProvider
    provider = NebiusNemotronProvider()

    registry = ToolRegistry()
    registry.register(JsonValidatorTool())
    registry.register(DataTransformTool())

    pipeline = ReAgentPipeline(provider=provider, tool_registry=registry)

    # Ask Nemotron for a structured decision
    decision = provider.make_structured_decision(
        task_objective="Validate a JSON payload using the json_validator tool.",
        available_tools=registry.list_tool_names(),
    )
    _check("Nemotron returned a StructuredDecision", decision is not None)
    if decision:
        _check("Decision specifies a tool", bool(decision.tool))
        _check("Tool is in allowlist", decision.tool in registry.list_tool_names())
        print(f"  Nemotron chose tool: {decision.tool}")
        print(f"  Proposed postconditions: {[p.predicate for p in decision.proposed_postconditions]}")

    print("\n  --> Live Nebius PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    live = "--live" in sys.argv

    # Isolate E2E runner to a dedicated temporary ledger
    prev_ledger = os.environ.get("REAGENT_LEDGER_PATH")
    temp_e2e_dir = tempfile.mkdtemp(prefix="reagent_e2e_")
    os.environ["REAGENT_LEDGER_PATH"] = os.path.join(temp_e2e_dir, "ledger.json")

    failures = []
    try:
        for fn in [
            run_happy_path,
            run_false_success_detection,
            run_file_contract,
            run_recoverable_failure,
            run_escalation_path,
        ]:
            try:
                fn()
            except AssertionError as exc:
                failures.append(str(exc))
                print(f"  [FAIL] {exc}\n")
    finally:
        if prev_ledger is not None:
            os.environ["REAGENT_LEDGER_PATH"] = prev_ledger
        else:
            os.environ.pop("REAGENT_LEDGER_PATH", None)
        import shutil
        shutil.rmtree(temp_e2e_dir, ignore_errors=True)

    if live:
        try:
            run_live_nebius()
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"  [FAIL] {exc}\n")

    _banner(f"E2E COMPLETE: {5 - len(failures)}/5 local scenarios passed")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("  All local E2E scenarios PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
