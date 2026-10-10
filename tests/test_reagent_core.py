"""
ReAgent Core Unit Tests
Validates domain models, failure taxonomy classification, recovery selection, policy guard, and state loop.
"""

import sys
import os
import unittest
import tempfile
import shutil

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_core_test_")
    _MODULE_PREV_LEDGER = os.environ.get("REAGENT_LEDGER_PATH")
    os.environ["REAGENT_LEDGER_PATH"] = os.path.join(_MODULE_TEMP_DIR, "ledger.json")


def tearDownModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    if _MODULE_PREV_LEDGER is not None:
        os.environ["REAGENT_LEDGER_PATH"] = _MODULE_PREV_LEDGER
    else:
        os.environ.pop("REAGENT_LEDGER_PATH", None)
    if _MODULE_TEMP_DIR and os.path.exists(_MODULE_TEMP_DIR):
        shutil.rmtree(_MODULE_TEMP_DIR, ignore_errors=True)


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reagent.models import (
    AgentState,
    TaskState,
    ExecutionEvent,
    FailureEvent,
    FailureDiagnosis,
    RecoveryDecision,
    PolicyDecision,
    ReliabilityState,
)
from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus
from reagent.providers.development import DevelopmentLLMProvider
from reagent.providers.nebius import NebiusNemotronProvider
from reagent.policy import PolicyGuard
from reagent.pipeline import ReAgentPipeline


class TestReAgentModels(unittest.TestCase):
    """Test 1: Domain models instantiation and validation."""

    def test_domain_models_creation(self):
        state = AgentState(agent_id="test-agent", status=AgentStatus.IDLE)
        self.assertEqual(state.agent_id, "test-agent")
        self.assertEqual(state.status, AgentStatus.IDLE)
        self.assertEqual(state.failure_streak, 0)

        failure = FailureEvent(
            task_id="task-101",
            step_index=1,
            tool_name="web_search",
            error_message="HTTP 504 Gateway Timeout",
        )
        self.assertEqual(failure.task_id, "task-101")
        self.assertEqual(failure.tool_name, "web_search")

        rel = ReliabilityState(agent_id="test-agent", total_actions=10, total_failures=2)
        self.assertEqual(rel.total_actions, 10)
        self.assertEqual(rel.total_failures, 2)


class TestDevelopmentProviderDiagnostics(unittest.TestCase):
    """Test 2: Deterministic diagnosis and strategy classification."""

    def setUp(self):
        self.provider = DevelopmentLLMProvider()

    def test_timeout_diagnosis_and_recovery(self):
        failure = FailureEvent(
            task_id="task-timeout",
            step_index=1,
            tool_name="db_query",
            error_message="Query timed out after 30000ms",
        )
        diag = self.provider.diagnose_failure(failure)
        self.assertEqual(diag.failure_type, FailureType.TIMEOUT)
        self.assertFalse(diag.is_agent_fault)
        self.assertTrue(diag.is_transient)

        rec = self.provider.select_recovery(diag, available_tools=["db_query_fallback"])
        self.assertEqual(rec.strategy, RecoveryStrategyType.WAIT_AND_RETRY)
        self.assertGreater(rec.backoff_seconds, 0)

    def test_auth_failure_diagnosis_and_escalation(self):
        failure = FailureEvent(
            task_id="task-auth",
            step_index=2,
            tool_name="cloud_api",
            error_message="403 Forbidden: Invalid bearer token permissions",
        )
        diag = self.provider.diagnose_failure(failure)
        self.assertEqual(diag.failure_type, FailureType.AUTHORIZATION)
        self.assertFalse(diag.is_agent_fault)

        rec = self.provider.select_recovery(diag, available_tools=["backup_api"])
        self.assertEqual(rec.strategy, RecoveryStrategyType.ESCALATE)

    def test_tool_failure_fallback_selection(self):
        failure = FailureEvent(
            task_id="task-tool",
            step_index=3,
            tool_name="search_engine_primary",
            error_message="Service unavailable: connection refused",
        )
        diag = self.provider.diagnose_failure(failure)
        self.assertEqual(diag.failure_type, FailureType.TOOL_FAILURE)

        rec = self.provider.select_recovery(diag, available_tools=["search_engine_secondary"])
        self.assertEqual(rec.strategy, RecoveryStrategyType.FALLBACK_TOOL)
        self.assertEqual(rec.fallback_tool_name, "search_engine_secondary")


class TestPolicyGuard(unittest.TestCase):
    """Test 3: Policy Guard retry thresholds and destructive payload interceptor."""

    def test_max_retries_enforcement(self):
        guard = PolicyGuard(max_retries=2)
        task_id = "task-loop-test"

        decision = RecoveryDecision(
            diagnosis_id="diag-1",
            strategy=RecoveryStrategyType.RETRY_SAME,
            rationale="Retrying same action",
        )

        # 1st attempt -> Allowed
        p1 = guard.check_recovery(task_id, decision)
        self.assertTrue(p1.is_allowed)

        # 2nd attempt -> Allowed
        p2 = guard.check_recovery(task_id, decision)
        self.assertTrue(p2.is_allowed)

        # 3rd attempt -> Blocked (exceeded max_retries=2)
        p3 = guard.check_recovery(task_id, decision)
        self.assertFalse(p3.is_allowed)
        self.assertTrue(any("Retry limit exceeded" in v for v in p3.violations))

    def test_destructive_payload_blocking(self):
        guard = PolicyGuard()
        decision = RecoveryDecision(
            diagnosis_id="diag-2",
            strategy=RecoveryStrategyType.RETRY_MODIFIED_INPUT,
            rationale="Retrying with script",
            modified_input={"command": "rm -rf /var/data"},
        )
        p = guard.check_recovery("task-danger", decision, proposed_payload=decision.modified_input)
        self.assertFalse(p.is_allowed)
        self.assertTrue(any("prohibited dangerous command" in v for v in p.violations))


class TestReAgentPipeline(unittest.TestCase):
    """Test 4: Full End-to-End Reliability Loop."""

    def test_pipeline_recovery_cycle(self):
        pipeline = ReAgentPipeline(agent_id="test-agent-core")

        # Simulate a transient failure
        failure = FailureEvent(
            task_id="task-pipe-1",
            step_index=1,
            tool_name="weather_api",
            error_message="HTTP 502 Bad Gateway from upstream service",
        )

        result = pipeline.process_failure(failure, available_tools=["backup_weather"])

        self.assertEqual(result.diagnosis.failure_type, FailureType.TRANSIENT_EXTERNAL)
        self.assertEqual(result.effective_strategy, RecoveryStrategyType.WAIT_AND_RETRY)
        self.assertEqual(result.agent_status, AgentStatus.RECOVERING)
        self.assertEqual(pipeline.reliability_state.total_failures, 1)

        # Now simulate successful recovery
        pipeline.record_recovery_success(task_id="task-pipe-1")
        self.assertEqual(pipeline.agent_state.status, AgentStatus.RUNNING)
        self.assertEqual(pipeline.agent_state.failure_streak, 0)
        self.assertEqual(pipeline.reliability_state.successful_recoveries, 1)

    def test_nebius_provider_unconfigured_error(self):
        # Temporarily ensure no API key is set in the environment
        original = os.environ.pop("NEBIUS_API_KEY", None)
        try:
            provider = NebiusNemotronProvider()
            self.assertFalse(provider.is_configured())
            with self.assertRaises(RuntimeError):
                provider.generate(prompt="Hello Nemotron")
        finally:
            if original is not None:
                os.environ["NEBIUS_API_KEY"] = original


if __name__ == "__main__":
    unittest.main()
