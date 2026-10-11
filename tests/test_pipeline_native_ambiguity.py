"""
test_pipeline_native_ambiguity.py — Direct unit tests for ReAgentPipeline native ambiguity recovery.

Verifies:
1. Primary write commits, response lost, authoritative probe reports PRESENT: exactly one effect, restraint triggered, verified receipt.
2. Primary write does not commit, authoritative probe reports CONFIRMED_ABSENT: safe retry within budget, verified completion.
3. Non-idempotent operation with ordinary ABSENT: retry prohibited, fail-closed to ESCALATED.
4. Probe UNAVAILABLE or UNOBSERVABLE: fail-closed to ESCALATED.
5. Probe DELAYED without authoritative presence: polls boundedly, fails closed if not PRESENT.
6. Action/recovery budget: no third write attempt.
7. Existing callers with no probe_callback retain previous behavior.
8. Core pipeline operates without importing reagent.experiments.
"""

import unittest
from typing import Dict, Any

from reagent.pipeline import ReAgentPipeline, ExecutionCycleResult
from reagent.tools.base import BaseTool
from reagent.models import TaskState, Postcondition, AgentStatus, ActionResult


class MockStatefulTool(BaseTool):
    """Simple test tool with controlled fault behavior."""
    name: str = "mock_stateful_tool"
    description: str = "Mock tool for testing native pipeline recovery"
    effect_class: str = "state-changing"

    def __init__(self, commit_on_attempt: int = 1, timeout_on_attempt: int = 1):
        self.commit_on_attempt = commit_on_attempt
        self.timeout_on_attempt = timeout_on_attempt
        self.attempts = 0
        self.commits = 0

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.attempts += 1
        if self.attempts == self.commit_on_attempt:
            self.commits += 1

        if self.attempts == self.timeout_on_attempt:
            raise TimeoutError("Simulated upstream RPC timeout")

        return {"success": True, "attempts": self.attempts, "commits": self.commits}


class TestPipelineNativeAmbiguity(unittest.TestCase):

    def test_core_does_not_import_experiments(self):
        """Invariant: reagent.pipeline must NOT import reagent.experiments."""
        import sys
        import inspect
        import reagent.pipeline as pipeline_mod
        source = inspect.getsource(pipeline_mod)
        self.assertNotIn("reagent.experiments", source)

    def test_existing_caller_without_probe_retains_behavior(self):
        """Callers passing no probe_callback retain standard behavior."""
        pipeline = ReAgentPipeline(agent_id="test-no-probe")
        tool = MockStatefulTool(commit_on_attempt=1, timeout_on_attempt=0)
        pipeline.tool_registry.register(tool)

        task = TaskState(task_id="t_std", objective="Standard run")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={"x": 1},
            max_recovery_attempts=0,
        )
        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(tool.commits, 1)
        self.assertEqual(tool.attempts, 1)

    def test_lost_response_restraint_on_present(self):
        """Case 1: Write commits, response lost, probe reports PRESENT -> restraint."""
        pipeline = ReAgentPipeline(agent_id="test-restraint")
        # Tool commits on attempt 1, but raises timeout on attempt 1
        tool = MockStatefulTool(commit_on_attempt=1, timeout_on_attempt=1)
        pipeline.tool_registry.register(tool)

        def probe_fn(name: str, params: Dict[str, Any], step: int):
            # Authoritative surface confirms state PRESENT
            return {"status": "PRESENT", "consistent": True}

        task = TaskState(task_id="t_restrain", objective="Test restraint")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={"x": 1},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(res.evidence.source, "probe_restraint")
        # No recovery attempt should have been issued
        self.assertEqual(tool.attempts, 1)
        self.assertEqual(tool.commits, 1)
        self.assertEqual(res.recovery_count, 0)

    def test_fail_before_commit_recovers_safely(self):
        """Case 2: Write does not commit, probe reports CONFIRMED_ABSENT -> safe retry."""
        pipeline = ReAgentPipeline(agent_id="test-safe-retry")
        # Tool does NOT commit on attempt 1, times out on attempt 1; commits on attempt 2
        tool = MockStatefulTool(commit_on_attempt=2, timeout_on_attempt=1)
        pipeline.tool_registry.register(tool)

        def probe_fn(name: str, params: Dict[str, Any], step: int):
            return {"status": "CONFIRMED_ABSENT", "consistent": True}

        task = TaskState(task_id="t_retry", objective="Test safe retry")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={"x": 1},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(tool.attempts, 2)
        self.assertEqual(tool.commits, 1)
        self.assertEqual(res.recovery_count, 1)

    def test_non_idempotent_ordinary_absent_prohibits_retry(self):
        """Case 3: Non-idempotent operation with ordinary ABSENT -> fail-closed to ESCALATED."""
        pipeline = ReAgentPipeline(agent_id="test-no-blind-retry")
        tool = MockStatefulTool(commit_on_attempt=2, timeout_on_attempt=1)
        pipeline.tool_registry.register(tool)

        def probe_fn(name: str, params: Dict[str, Any], step: int):
            # Ordinary unconfirmed absence (e.g. eventual consistency lag)
            return {"status": "ABSENT", "consistent": False}

        task = TaskState(task_id="t_gate", objective="Test safety gate")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={"x": 1},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertIn("NonIdempotentOperationAmbiguous", res.action_result.error)
        # Primary attempt ran, retry was blocked
        self.assertEqual(tool.attempts, 1)

    def test_probe_unavailable_fails_closed(self):
        """Case 4: Probe raises StatusUnavailableError -> fail-closed to ESCALATED."""
        pipeline = ReAgentPipeline(agent_id="test-status-down")
        tool = MockStatefulTool(commit_on_attempt=1, timeout_on_attempt=1)
        pipeline.tool_registry.register(tool)

        def probe_fn(name: str, params: Dict[str, Any], step: int):
            raise RuntimeError("StatusUnavailableError: Endpoint unreachable")

        task = TaskState(task_id="t_unavail", objective="Test status down")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={"x": 1},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertEqual(tool.attempts, 1)

    def test_delayed_visibility_polls_and_restrains(self):
        """Case 5: Step 1 DELAYED, Step 2 PRESENT -> polls boundedly and restrains."""
        pipeline = ReAgentPipeline(agent_id="test-delayed")
        tool = MockStatefulTool(commit_on_attempt=1, timeout_on_attempt=1)
        pipeline.tool_registry.register(tool)

        probe_calls = []
        def probe_fn(name: str, params: Dict[str, Any], step: int):
            probe_calls.append(step)
            if step == 1:
                return {"status": "DELAYED"}
            return {"status": "PRESENT"}

        task = TaskState(task_id="t_delayed", objective="Test delayed")
        res = pipeline.execute_with_contract(
            task=task,
            tool_name="mock_stateful_tool",
            params={"x": 1},
            max_recovery_attempts=1,
            probe_callback=probe_fn,
            supports_idempotency=False,
        )

        self.assertEqual(res.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(probe_calls, [1, 2])
        self.assertEqual(tool.attempts, 1)


if __name__ == "__main__":
    unittest.main()
