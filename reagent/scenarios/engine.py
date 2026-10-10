"""
ReAgent Scenario Runner
Executes automated deterministic failure scenarios and traces all lifecycle transitions:
initial state → failure → diagnosis → recovery → evidence → verification → final state
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import time
import os

from reagent.scenarios.definitions import ScenarioDefinition
from reagent.pipeline import ReAgentPipeline
from reagent.tools.registry import ToolRegistry
from reagent.tools.data_tool import DataTransformTool
from reagent.tools.json_tool import JsonValidatorTool
from reagent.tools.file_tool import SandboxedFileTool
from reagent.tools.http_tool import HttpRequestTool
from reagent.verification.evidence import EvidenceCollector
from reagent.verification.verifier import DeterministicVerifier
from reagent.models import FailureEvent, ActionResult, TaskState, AgentState
from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus


class ScenarioExecutionTrace(BaseModel):
    scenario_id: str
    name: str
    passed: bool
    initial_status: str
    detected_failure: Optional[str] = None
    diagnosed_failure_type: Optional[str] = None
    selected_strategy: Optional[str] = None
    policy_allowed: bool = True
    recovery_executed: bool = False
    evidence_collected: bool = False
    verification_passed: bool = False
    final_status: str
    discrepancies: List[str] = Field(default_factory=list)
    duration_ms: int = 0


class ScenarioRunner:
    """
    Automated benchmark runner for ReAgent reliability scenarios.
    """

    def __init__(self, pipeline: Optional[ReAgentPipeline] = None):
        self.pipeline = pipeline or ReAgentPipeline(agent_id="scenario-runner-agent")
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(DataTransformTool())
        self.tool_registry.register(JsonValidatorTool())
        self.tool_registry.register(SandboxedFileTool())
        self.tool_registry.register(HttpRequestTool())
        self.evidence_collector = EvidenceCollector()
        self.verifier = DeterministicVerifier()

    def run_scenario(self, scenario: ScenarioDefinition) -> ScenarioExecutionTrace:
        t0 = time.perf_counter()
        discrepancies: List[str] = []

        # Ensure clean state for Scenario F
        if scenario.scenario_id == "scenario_f_false_success_verification_failure":
            for rule in scenario.assertions:
                if rule.assertion_type == "FILE_EXISTS" and rule.expected_value and os.path.exists(rule.expected_value):
                    try:
                        os.remove(rule.expected_value)
                    except Exception:
                        pass

        # 1. Initial State
        self.pipeline.agent_state.status = scenario.initial_status
        task = TaskState(objective=scenario.description)

        # 2. Trigger or simulate failure
        action_res: Optional[ActionResult] = None
        failure_event: Optional[FailureEvent] = None

        if scenario.injected_error:
            failure_event = FailureEvent(
                task_id=task.task_id,
                step_index=1,
                tool_name=scenario.tool_name,
                error_message=scenario.injected_error,
            )
        else:
            # Execute tool directly to see outcome
            action_res = self.tool_registry.execute_tool(scenario.tool_name, scenario.initial_payload)
            if not action_res.success:
                failure_event = FailureEvent(
                    task_id=task.task_id,
                    step_index=1,
                    tool_name=scenario.tool_name,
                    error_message=action_res.error or "Unknown error",
                )

        # 3. If Scenario F (False Success check): tool succeeded, but verification must detect discrepancy
        if scenario.scenario_id == "scenario_f_false_success_verification_failure" and action_res:
            evidence = self.evidence_collector.collect_from_action(action_res)
            v_res = self.verifier.verify(recovery_decision_id="pre-recovery", evidence=evidence, assertions=scenario.assertions)

            # Post-condition fails as expected (detecting false success)
            if not v_res.verified:
                failure_event = FailureEvent(
                    task_id=task.task_id,
                    step_index=1,
                    tool_name=scenario.tool_name,
                    failure_type=FailureType.TOOL_FAILURE,
                    error_message=f"PostConditionVerificationFailed: {', '.join(v_res.discrepancies)}",
                )

        # 4. Reliability Pipeline Cycle (Detect -> Diagnose -> Select Recovery -> Guard)
        cycle_res = None
        if failure_event:
            cycle_res = self.pipeline.process_failure(
                failure=failure_event,
                available_tools=self.tool_registry.list_tool_names(),
                task=task,
            )

            # Assert expected diagnosis
            if cycle_res.diagnosis.failure_type != scenario.expected_failure_type:
                discrepancies.append(
                    f"Diagnosis mismatch: expected {scenario.expected_failure_type}, got {cycle_res.diagnosis.failure_type}"
                )

            # Assert expected strategy
            if cycle_res.effective_strategy != scenario.expected_strategy:
                discrepancies.append(
                    f"Strategy mismatch: expected {scenario.expected_strategy}, got {cycle_res.effective_strategy}"
                )

        # 5. Execute Recovery & Verify (if recovery strategy selected and not escalated)
        recovery_executed = False
        verification_passed = False

        if cycle_res and cycle_res.effective_strategy in (
            RecoveryStrategyType.RETRY_MODIFIED_INPUT,
            RecoveryStrategyType.WAIT_AND_RETRY,
            RecoveryStrategyType.FALLBACK_TOOL,
            RecoveryStrategyType.RETRY_SAME,
        ):
            exec_tool = scenario.fallback_tool_name or scenario.tool_name
            exec_payload = scenario.recovery_payload or scenario.initial_payload

            recovery_action_res = self.tool_registry.execute_tool(exec_tool, exec_payload)
            recovery_executed = True

            # Collect Evidence
            evidence = self.evidence_collector.collect_from_action(recovery_action_res)

            # Deterministic Verification
            if scenario.assertions:
                v_res = self.verifier.verify(
                    recovery_decision_id=cycle_res.recovery_decision.decision_id,
                    evidence=evidence,
                    assertions=scenario.assertions,
                )
                verification_passed = v_res.verified
                if not v_res.verified:
                    discrepancies.extend(v_res.discrepancies)
            else:
                verification_passed = recovery_action_res.success

            if verification_passed:
                self.pipeline.record_recovery_success(task.task_id)

        elif cycle_res and cycle_res.effective_strategy == RecoveryStrategyType.ESCALATE:
            # Escalated scenario
            verification_passed = True

        # 6. Final State Evaluation
        final_status = self.pipeline.agent_state.status
        if final_status != scenario.expected_final_status:
            discrepancies.append(
                f"Final status mismatch: expected {scenario.expected_final_status}, got {final_status}"
            )

        duration = int((time.perf_counter() - t0) * 1000)
        passed = len(discrepancies) == 0

        return ScenarioExecutionTrace(
            scenario_id=scenario.scenario_id,
            name=scenario.name,
            passed=passed,
            initial_status=scenario.initial_status.value,
            detected_failure=failure_event.error_message if failure_event else None,
            diagnosed_failure_type=cycle_res.diagnosis.failure_type.value if cycle_res else None,
            selected_strategy=cycle_res.effective_strategy.value if cycle_res else None,
            policy_allowed=cycle_res.policy_decision.is_allowed if cycle_res else True,
            recovery_executed=recovery_executed,
            evidence_collected=True,
            verification_passed=verification_passed,
            final_status=final_status.value,
            discrepancies=discrepancies,
            duration_ms=duration,
        )
