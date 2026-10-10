"""
ReAgent Deterministic Failure Scenarios Automated Benchmark Test Suite
Runs Scenarios A through F and asserts full lifecycle transitions.
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
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_scenarios_test_")
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

from reagent.scenarios.definitions import get_all_scenarios
from reagent.scenarios.engine import ScenarioRunner
from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus


class TestDeterministicScenarios(unittest.TestCase):
    """Executes the 6 core reliability failure scenarios."""

    def setUp(self):
        self.runner = ScenarioRunner()
        self.scenarios = {s.scenario_id: s for s in get_all_scenarios()}

    def test_scenario_a_malformed_input(self):
        scenario = self.scenarios["scenario_a_malformed_input"]
        trace = self.runner.run_scenario(scenario)
        self.assertTrue(trace.passed, f"Scenario A failed: {trace.discrepancies}")
        self.assertEqual(trace.diagnosed_failure_type, FailureType.INPUT_CONTEXT.value)
        self.assertEqual(trace.selected_strategy, RecoveryStrategyType.RETRY_MODIFIED_INPUT.value)
        self.assertEqual(trace.final_status, AgentStatus.RUNNING.value)

    def test_scenario_b_transient_failure(self):
        scenario = self.scenarios["scenario_b_transient_failure"]
        trace = self.runner.run_scenario(scenario)
        self.assertTrue(trace.passed, f"Scenario B failed: {trace.discrepancies}")
        self.assertEqual(trace.diagnosed_failure_type, FailureType.TRANSIENT_EXTERNAL.value)
        self.assertEqual(trace.selected_strategy, RecoveryStrategyType.WAIT_AND_RETRY.value)
        self.assertEqual(trace.final_status, AgentStatus.RUNNING.value)

    def test_scenario_c_unavailable_dependency(self):
        scenario = self.scenarios["scenario_c_unavailable_dependency"]
        trace = self.runner.run_scenario(scenario)
        self.assertTrue(trace.passed, f"Scenario C failed: {trace.discrepancies}")
        self.assertEqual(trace.diagnosed_failure_type, FailureType.ENVIRONMENT_FAILURE.value)
        self.assertEqual(trace.selected_strategy, RecoveryStrategyType.FALLBACK_TOOL.value)
        self.assertEqual(trace.final_status, AgentStatus.RUNNING.value)

    def test_scenario_d_invalid_model_output(self):
        scenario = self.scenarios["scenario_d_invalid_model_output"]
        trace = self.runner.run_scenario(scenario)
        self.assertTrue(trace.passed, f"Scenario D failed: {trace.discrepancies}")
        self.assertEqual(trace.diagnosed_failure_type, FailureType.MODEL_OUTPUT.value)
        self.assertEqual(trace.selected_strategy, RecoveryStrategyType.RETRY_MODIFIED_INPUT.value)
        self.assertEqual(trace.final_status, AgentStatus.RUNNING.value)

    def test_scenario_e_repeated_failure_escalation(self):
        scenario = self.scenarios["scenario_e_repeated_failure_escalation"]
        trace = self.runner.run_scenario(scenario)
        self.assertTrue(trace.passed, f"Scenario E failed: {trace.discrepancies}")
        self.assertEqual(trace.diagnosed_failure_type, FailureType.AUTHORIZATION.value)
        self.assertEqual(trace.selected_strategy, RecoveryStrategyType.ESCALATE.value)
        self.assertEqual(trace.final_status, AgentStatus.ESCALATED.value)

    def test_scenario_f_false_success_verification_failure(self):
        scenario = self.scenarios["scenario_f_false_success_verification_failure"]
        trace = self.runner.run_scenario(scenario)
        self.assertTrue(trace.passed, f"Scenario F failed: {trace.discrepancies}")
        self.assertEqual(trace.diagnosed_failure_type, FailureType.TOOL_FAILURE.value)
        self.assertTrue(trace.evidence_collected)


if __name__ == "__main__":
    unittest.main()
