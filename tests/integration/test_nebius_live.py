"""
ReAgent Live Nebius Integration Tests
Skipped automatically unless NEBIUS_API_KEY is present in the environment.

Run live tests:
  NEBIUS_API_KEY=<key> python -m pytest tests/integration/test_nebius_live.py -v

These tests make REAL API calls to Nebius Token Factory.
They are NEVER included in the standard unit test suite.
"""

import os
import unittest
import tempfile
import shutil

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_live_test_")
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


# Require BOTH a non-empty NEBIUS_API_KEY and REAGENT_ALLOW_LIVE_TESTS=1 to run live tests
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
HAS_KEY = bool(NEBIUS_API_KEY and NEBIUS_API_KEY.strip())
ALLOW_LIVE_TESTS = (os.environ.get("REAGENT_ALLOW_LIVE_TESTS") == "1")
LIVE_ENABLED = HAS_KEY and ALLOW_LIVE_TESTS

def get_skip_reason() -> str:
    if not HAS_KEY:
        return "NEBIUS_API_KEY not set — live integration tests skipped."
    if not ALLOW_LIVE_TESTS:
        return "REAGENT_ALLOW_LIVE_TESTS=1 not set — live integration tests require explicit opt-in."
    return ""

SKIP_REASON = get_skip_reason()


@unittest.skipUnless(LIVE_ENABLED, SKIP_REASON)
class TestNebiusLiveGenerate(unittest.TestCase):
    """Live tests for generate() against Nebius Token Factory."""

    def setUp(self):
        from reagent.providers.nebius import NebiusNemotronProvider
        self.provider = NebiusNemotronProvider()

    def test_is_configured(self):
        self.assertTrue(self.provider.is_configured())

    def test_basic_generate_returns_content(self):
        res = self.provider.generate(
            prompt="Respond with exactly the word: REAGENT",
            temperature=0.0,
            max_tokens=16,
        )
        self.assertIsInstance(res.content, str)
        self.assertGreater(len(res.content), 0)
        # Key must not appear in content (security)
        self.assertNotIn(NEBIUS_API_KEY[:8], res.content)

    def test_generate_with_json_schema(self):
        res = self.provider.generate(
            prompt='Return a JSON object with a single key "status" set to "ok".',
            temperature=0.0,
            max_tokens=64,
            response_schema={},
        )
        self.assertIsNotNone(res.parsed_json)
        self.assertIn("status", res.parsed_json)

    def test_raw_response_not_stored(self):
        """raw_response must be None — never log API responses that may echo keys."""
        res = self.provider.generate(prompt="Hello", max_tokens=8)
        self.assertIsNone(res.raw_response)

    def test_token_usage_populated(self):
        res = self.provider.generate(prompt="one word", max_tokens=8)
        self.assertIn("prompt_tokens", res.token_usage)
        self.assertIn("completion_tokens", res.token_usage)


@unittest.skipUnless(LIVE_ENABLED, SKIP_REASON)
class TestNebiusLiveStructuredDecision(unittest.TestCase):
    """Live tests for make_structured_decision() with schema validation."""

    def setUp(self):
        from reagent.providers.nebius import NebiusNemotronProvider
        from reagent.tools.registry import ToolRegistry
        from reagent.tools.json_tool import JsonValidatorTool
        from reagent.tools.data_tool import DataTransformTool
        self.provider = NebiusNemotronProvider()
        registry = ToolRegistry()
        registry.register(JsonValidatorTool())
        registry.register(DataTransformTool())
        self.tools = registry.list_tool_names()

    def test_make_structured_decision_returns_valid_schema(self):
        decision = self.provider.make_structured_decision(
            task_objective="Validate a JSON payload.",
            available_tools=self.tools,
        )
        self.assertIsNotNone(decision, "Nemotron returned unparseable structured decision")
        self.assertIsInstance(decision.tool, str)
        self.assertTrue(decision.tool.strip())

    def test_structured_decision_tool_is_in_allowlist(self):
        decision = self.provider.make_structured_decision(
            task_objective="Validate a JSON payload.",
            available_tools=self.tools,
        )
        if decision:
            self.assertIn(
                decision.tool, self.tools,
                f"Nemotron proposed tool '{decision.tool}' not in allowlist {self.tools}",
            )


@unittest.skipUnless(LIVE_ENABLED, SKIP_REASON)
class TestNebiusLiveDiagnoseAndRecover(unittest.TestCase):
    """Live tests for diagnose_failure() and select_recovery()."""

    def setUp(self):
        from reagent.providers.nebius import NebiusNemotronProvider
        self.provider = NebiusNemotronProvider()

    def test_diagnose_failure_returns_valid_diagnosis(self):
        from reagent.models import FailureEvent
        from reagent.taxonomy import FailureType
        failure = FailureEvent(
            task_id="live-test-task",
            step_index=1,
            tool_name="json_validator",
            failure_type=FailureType.UNKNOWN,
            error_message="JSONDecodeError: Expecting value at line 1 column 1",
        )
        diagnosis = self.provider.diagnose_failure(failure)
        self.assertIsNotNone(diagnosis)
        self.assertIsInstance(diagnosis.failure_type, FailureType)
        self.assertGreater(len(diagnosis.root_cause), 0)
        self.assertGreaterEqual(diagnosis.confidence_score, 0.0)
        self.assertLessEqual(diagnosis.confidence_score, 1.0)

    def test_select_recovery_returns_valid_strategy(self):
        from reagent.models import FailureDiagnosis
        from reagent.taxonomy import FailureType, RecoveryStrategyType
        diagnosis = FailureDiagnosis(
            failure_id="f1",
            failure_type=FailureType.MODEL_OUTPUT,
            root_cause="Model returned malformed JSON",
            confidence_score=0.9,
            is_agent_fault=True,
            is_transient=False,
        )
        decision = self.provider.select_recovery(
            diagnosis=diagnosis,
            available_tools=["json_validator", "data_transform"],
        )
        self.assertIsNotNone(decision)
        self.assertIsInstance(decision.strategy, RecoveryStrategyType)
        self.assertGreater(len(decision.rationale), 0)


@unittest.skipUnless(LIVE_ENABLED, SKIP_REASON)
class TestNebiusLiveFullLoop(unittest.TestCase):
    """Live test of execute_with_contract() using NebiusNemotronProvider."""

    def test_full_loop_happy_path_live(self):
        from reagent.providers.nebius import NebiusNemotronProvider
        from reagent.pipeline import ReAgentPipeline
        from reagent.tools.registry import ToolRegistry
        from reagent.tools.json_tool import JsonValidatorTool
        from reagent.models import TaskState, Postcondition
        from reagent.taxonomy import AgentStatus

        registry = ToolRegistry()
        registry.register(JsonValidatorTool())
        provider = NebiusNemotronProvider()
        pipeline = ReAgentPipeline(provider=provider, tool_registry=registry)

        task = TaskState(objective="Live E2E: validate JSON with Nebius provider")
        result = pipeline.execute_with_contract(
            task=task,
            tool_name="json_validator",
            params={"json_string": '{"live": true, "model": "nemotron"}'},
            postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="KEY_EXISTS", target="valid"),
            ],
        )
        self.assertTrue(result.verification_result.verified)
        self.assertEqual(result.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertGreater(len(result.timeline), 3)


if __name__ == "__main__":
    unittest.main()
