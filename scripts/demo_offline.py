#!/usr/bin/env python3
"""
ReAgent Offline Mechanics Demo  -  No API key required
============================================================
Demonstrates deterministic pipeline mechanics, contract verification,
and recovery state transitions using the local DevelopmentLLMProvider.
This script demonstrates offline verification mechanics and does NOT
connect to or evaluate live models.

Run with:
    python scripts/demo_offline.py

Expected: 5/5 mechanics scenarios PASS, exit code 0.
"""

import os
import sys
import json
import tempfile
import textwrap
from typing import List

# -- Ensure D:\ReAgent is importable even if not installed ------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reagent.pipeline import ReAgentPipeline, ExecutionCycleResult
from reagent.models import TaskState, Postcondition
from reagent.tools.registry import ToolRegistry
from reagent.tools.json_tool import JsonValidatorTool
from reagent.tools.file_tool import SandboxedFileTool
from reagent.tools.data_tool import DataTransformTool
from reagent.taxonomy import AgentStatus

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def banner(title: str, n: int) -> None:
    print(f"\n{'=' * 64}")
    print(f"  Scenario {n}: {title}")
    print(f"{'=' * 64}")


def print_timeline(result: ExecutionCycleResult) -> None:
    print()
    for ev in result.timeline:
        detail_str = json.dumps(ev.detail, default=str)
        if len(detail_str) > 80:
            detail_str = detail_str[:77] + "..."
        print(f"  {ev.phase:<28}  {detail_str}")


def check(label: str, condition: bool) -> bool:
    mark = PASS if condition else FAIL
    print(f"  {mark}  {label}")
    return condition


def _run_scenarios() -> List[str]:
    failures = []

    # ----------------------------------------------------------------------
    # Scenario 1 - Happy Path: JSON Validation
    # ----------------------------------------------------------------------
    banner("Happy Path - JSON Validation with Full Verification", 1)
    print(textwrap.dedent("""\
    Agent validates a well-formed JSON payload. ReAgent freezes a 3-predicate
    success contract (STATUS_SUCCESS + KEY_EXISTS + VALUE_EQUALS), executes the
    tool, collects independent evidence, and verifies deterministically.
    Provider is never asked to grade its own output.
    """))

    registry = ToolRegistry()
    registry.register(JsonValidatorTool())
    pipeline = ReAgentPipeline(tool_registry=registry)
    task = TaskState(objective="Validate a well-formed JSON payload.")

    result = pipeline.execute_with_contract(
        task=task,
        tool_name="json_validator",
        params={"json_payload": '{"status": "ok", "score": 42}'},
        postconditions=[
            Postcondition(predicate="STATUS_SUCCESS", description="Tool must succeed"),
            Postcondition(predicate="KEY_EXISTS", target="is_valid",
                          description="'is_valid' key must be present"),
            Postcondition(predicate="VALUE_EQUALS", target="is_valid", expected=True,
                          description="JSON must be valid"),
        ],
    )

    print_timeline(result)
    ok = all([
        check("Contract frozen before execution",
              result.frozen_contract.frozen_at != ""),
        check("Tool executed successfully", result.action_result.success),
        check("All postconditions verified", result.verification_result.verified),
        check("Receipt status = VERIFIED",
              result.verified_receipt.final_status == AgentStatus.VERIFIED),
        check("Recovery count = 0", result.recovery_count == 0),
    ])
    if not ok:
        failures.append("Scenario 1: Happy Path")

    # ----------------------------------------------------------------------
    # Scenario 2 - False-Success Detection
    # ----------------------------------------------------------------------
    banner("False-Success Detection - Agent Claims Success, ReAgent Disagrees", 2)
    print(textwrap.dedent("""\
    Agent calls json_validator (succeeds, exit 0). But the frozen contract
    requires key 'score' in the output - and the validator doesn't return it.
    ReAgent's DeterministicVerifier catches the discrepancy and records
    ESCALATED - the agent's self-reported success is ignored.
    """))

    registry2 = ToolRegistry()
    registry2.register(JsonValidatorTool())
    pipeline2 = ReAgentPipeline(tool_registry=registry2)
    task2 = TaskState(objective="Validate JSON and check score key exists.")

    result2 = pipeline2.execute_with_contract(
        task=task2,
        tool_name="json_validator",
        params={"json_payload": '{"status": "ok"}'},
        postconditions=[
            Postcondition(predicate="STATUS_SUCCESS"),
            Postcondition(predicate="KEY_EXISTS", target="score",
                          description="Score key must exist - but it won't"),
        ],
        max_recovery_attempts=0,
    )

    print_timeline(result2)
    tool_initially_succeeded = any(
        e.phase == "TOOL_EXECUTED" and e.detail.get("success") is True
        for e in result2.timeline
    )
    ok2 = all([
        check("Tool execution succeeded (claimed success)", tool_initially_succeeded),
        check("VERIFICATION_FAILED emitted - false success caught",
              any(e.phase == "VERIFICATION_FAILED" for e in result2.timeline)),
        check("Discrepancies recorded",
              len(result2.verification_result.discrepancies) > 0),
        check("Receipt = ESCALATED (not VERIFIED)",
              result2.verified_receipt.final_status == AgentStatus.ESCALATED),
    ])
    if not ok2:
        failures.append("Scenario 2: False-Success Detection")

    # ----------------------------------------------------------------------
    # Scenario 3 - File Contract (FILE_EXISTS + FILE_CONTAINS)
    # ----------------------------------------------------------------------
    banner("File Contract - Sandboxed File Tool with Content Verification", 3)
    print(textwrap.dedent("""\
    Agent writes a file to a sandboxed workspace. Success contract requires:
    FILE_EXISTS, FILE_CONTAINS the marker string, and FILE_SIZE_GTE > 0.
    All predicates resolved against real filesystem state - not model claims.
    Sandbox prevents path traversal and write outside the workspace root.
    """))

    with tempfile.TemporaryDirectory() as tmpdir:
        file_tool = SandboxedFileTool(sandbox_root=tmpdir)
        registry3 = ToolRegistry()
        registry3.register(file_tool)
        pipeline3 = ReAgentPipeline(tool_registry=registry3)
        task3 = TaskState(objective="Write a verified output file.")
        target = os.path.join(tmpdir, "reagent_output.txt")

        result3 = pipeline3.execute_with_contract(
            task=task3,
            tool_name="sandboxed_file",
            params={"action": "write", "relative_path": "reagent_output.txt",
                    "content": "REAGENT_VERIFIED_MARKER"},
            postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="FILE_EXISTS", expected=target,
                              description="Output file must exist on disk"),
                Postcondition(predicate="FILE_CONTAINS", target=target,
                              expected="REAGENT_VERIFIED_MARKER"),
                Postcondition(predicate="FILE_SIZE_GTE", target=target, expected=1),
            ],
        )

    print_timeline(result3)
    ok3 = all([
        check("File written successfully", result3.action_result.success),
        check("All file postconditions verified", result3.verification_result.verified),
        check("Receipt = VERIFIED", result3.verified_receipt.final_status == AgentStatus.VERIFIED),
    ])
    if not ok3:
        failures.append("Scenario 3: File Contract")

    # ----------------------------------------------------------------------
    # Scenario 4 - Recoverable Failure + Auto-Diagnosis
    # ----------------------------------------------------------------------
    banner("Bounded Auto-Recovery - Tool Fails, Diagnosis, Retry Succeeds", 4)
    print(textwrap.dedent("""\
    First tool call is injected to fail (bad JSON). ReAgent's DevelopmentProvider
    diagnoses MODEL_OUTPUT failure, selects RETRY_MODIFIED_INPUT strategy,
    PolicyGuard approves, second call succeeds, reverification passes.
    Every step is recorded in the immutable ObservabilityTimeline.
    Recovery is bounded (max 1 attempt). If it fails, system escalates.
    """))

    class _FirstFailJson(JsonValidatorTool):
        _calls = 0
        def execute(self, params):
            self.__class__._calls += 1
            if self.__class__._calls == 1:
                return super().execute({"json_payload": "NOT VALID {{{}"})
            return super().execute({"json_payload": '{"repaired": true}'})

    registry4 = ToolRegistry()
    fixer = _FirstFailJson()
    fixer._name = "json_validator"
    registry4.register(fixer)
    pipeline4 = ReAgentPipeline(tool_registry=registry4)
    task4 = TaskState(objective="Validate JSON with simulated failure then recovery.")

    result4 = pipeline4.execute_with_contract(
        task=task4,
        tool_name="json_validator",
        params={"json_payload": "NOT VALID {{{"},
        postconditions=[
            Postcondition(predicate="STATUS_SUCCESS", description="Must succeed"),
        ],
        max_recovery_attempts=1,
    )

    print_timeline(result4)
    phases4 = [e.phase for e in result4.timeline]
    ok4 = all([
        check("Initial verification failed", "VERIFICATION_FAILED" in phases4),
        check("Failure attributed (FAILURE_ATTRIBUTED)", "FAILURE_ATTRIBUTED" in phases4),
        check("Recovery selected", "RECOVERY_SELECTED" in phases4),
        check("Recovery executed", "RECOVERY_EXECUTED" in phases4),
        check("Reverification fired", "REVERIFICATION" in phases4),
        check("Final status VERIFIED", result4.verified_receipt.final_status == AgentStatus.VERIFIED),
        check("Recovery count = 1", result4.recovery_count == 1),
    ])
    if not ok4:
        failures.append("Scenario 4: Recoverable Failure")

    # ----------------------------------------------------------------------
    # Scenario 5 - Escalation: All Recovery Attempts Exhausted
    # ----------------------------------------------------------------------
    banner("Escalation - Max Recovery Attempts Exhausted, Never Silent Failure", 5)
    print(textwrap.dedent("""\
    A tool that always fails is run with max_recovery_attempts=2.
    ReAgent attempts recovery twice, both reverifications fail, and the
    system escalates - it never silently marks the task 'done'.
    Reliability state tracks escalation count for observability.
    """))

    registry5 = ToolRegistry()
    registry5.register(JsonValidatorTool())
    pipeline5 = ReAgentPipeline(tool_registry=registry5)
    task5 = TaskState(objective="Trigger escalation path with always-failing input.")

    result5 = pipeline5.execute_with_contract(
        task=task5,
        tool_name="json_validator",
        params={"json_payload": "ALWAYS_INVALID{{{{{"},
        postconditions=[
            Postcondition(predicate="STATUS_SUCCESS", description="Must succeed - will never"),
        ],
        max_recovery_attempts=2,
    )

    print_timeline(result5)
    ok5 = all([
        check("VERIFIED_RECEIPT emitted",
              any(e.phase == "VERIFIED_RECEIPT" for e in result5.timeline)),
        check("Final status = ESCALATED",
              result5.verified_receipt.final_status == AgentStatus.ESCALATED),
        check("Escalation recorded in reliability state",
              pipeline5.reliability_state.escalations >= 1),
    ])
    if not ok5:
        failures.append("Scenario 5: Escalation")

    return failures


def run_demo() -> int:
    prev_ledger = os.environ.get("REAGENT_LEDGER_PATH")
    temp_demo_dir = tempfile.mkdtemp(prefix="reagent_demo_")
    os.environ["REAGENT_LEDGER_PATH"] = os.path.join(temp_demo_dir, "ledger.json")

    try:
        failures = _run_scenarios()
    finally:
        if prev_ledger is not None:
            os.environ["REAGENT_LEDGER_PATH"] = prev_ledger
        else:
            os.environ.pop("REAGENT_LEDGER_PATH", None)
        import shutil
        shutil.rmtree(temp_demo_dir, ignore_errors=True)

    passed = 5 - len(failures)
    print(f"\n{'=' * 64}")
    print(f"  ReAgent Offline Demo: {passed}/5 scenarios PASSED")
    print(f"{'=' * 64}")

    if failures:
        print("\n  FAILED:")
        for f in failures:
            print(f"    {FAIL}  {f}")
        print()
        return 1
    else:
        print(f"\n  {PASS}  All 5 scenarios passed. No API key required.\n")
        print("  Next step (supervised, billing confirmation required):")
        print("    python scripts/live_probe.py --confirm-billing-checked")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(run_demo())
