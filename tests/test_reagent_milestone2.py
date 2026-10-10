"""
ReAgent Milestone 2 Unit Tests
Validates ContextSelector, Allowlisted ToolRegistry, EvidenceCollector, and DeterministicVerifier.
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
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_m2_test_")
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

from reagent.models import AgentState, TaskState, ExecutionEvent, ActionResult, Evidence
from reagent.taxonomy import AgentStatus
from reagent.context import ContextSelector
from reagent.tools.registry import ToolRegistry
from reagent.tools.data_tool import DataTransformTool
from reagent.tools.json_tool import JsonValidatorTool
from reagent.tools.file_tool import SandboxedFileTool
from reagent.verification.evidence import EvidenceCollector
from reagent.verification.verifier import DeterministicVerifier, AssertionRule


class TestContextSelector(unittest.TestCase):
    """Test 1: ContextSelector minimal context construction & bounds."""

    def test_context_window_bounding_and_sanitization(self):
        selector = ContextSelector(max_history_items=2, max_payload_chars=50)
        agent = AgentState(agent_id="agent-007", status=AgentStatus.RUNNING, step_index=5, failure_streak=1)
        task = TaskState(objective="Fetch and transform telemetry data")

        # Create 4 history events (only last 2 should be included)
        history = [
            ExecutionEvent(task_id="t1", step_index=1, tool_name="tool_1", input_payload={"large_text": "A" * 100}),
            ExecutionEvent(task_id="t1", step_index=2, tool_name="tool_2", input_payload={"step": 2}),
            ExecutionEvent(task_id="t1", step_index=3, tool_name="tool_3", input_payload={"step": 3}),
            ExecutionEvent(task_id="t1", step_index=4, tool_name="tool_4", input_payload={"large_text": "B" * 100}),
        ]

        win = selector.select_context(
            agent_state=agent,
            task=task,
            execution_history=history,
            allowlisted_tools=[{"name": "tool_4", "description": "test tool"}],
        )

        self.assertEqual(len(win.bounded_history), 2)
        self.assertEqual(win.bounded_history[0]["step"], 3)
        self.assertEqual(win.bounded_history[1]["step"], 4)
        # Check sanitization / truncation
        self.assertTrue("...[TRUNCATED]" in win.bounded_history[1]["input"]["large_text"])
        self.assertEqual(len(win.allowlisted_tools), 1)


class TestAllowlistedTools(unittest.TestCase):
    """Test 2: Tool Registry & Concrete Allowlisted Tools."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.data_tool = DataTransformTool()
        self.json_tool = JsonValidatorTool()
        self.registry.register(self.data_tool)
        self.registry.register(self.json_tool)

    def test_unregistered_tool_blocked_by_allowlist(self):
        res = self.registry.execute_tool("unregistered_bad_tool", {"cmd": "ls"})
        self.assertFalse(res.success)
        self.assertIn("SecurityViolation", res.error)

    def test_data_transform_tool_execution(self):
        res = self.registry.execute_tool(
            "data_transform",
            {"source_data": {"a": 1, "b": 2, "c": 3}, "fields_to_extract": ["a", "c"]},
        )
        self.assertTrue(res.success)
        self.assertEqual(res.output["extracted_data"], {"a": 1, "c": 3})

    def test_json_validator_tool_execution(self):
        res = self.registry.execute_tool(
            "json_validator",
            {"json_payload": '{"status": "ok", "code": 200}', "required_keys": ["status", "code"]},
        )
        self.assertTrue(res.success)
        self.assertTrue(res.output["is_valid"])

    def test_sandboxed_file_tool_safety(self):
        temp_dir = tempfile.mkdtemp()
        try:
            file_tool = SandboxedFileTool(sandbox_root=temp_dir)

            # 1. Write file
            w_res = file_tool.run({"action": "write", "relative_path": "sub/test.txt", "content": "Hello Sandbox"})
            self.assertTrue(w_res.success)

            # 2. Read file
            r_res = file_tool.run({"action": "read", "relative_path": "sub/test.txt"})
            self.assertTrue(r_res.success)
            self.assertEqual(r_res.output["content"], "Hello Sandbox")

            # 3. Path traversal attack attempt -> should fail
            bad_res = file_tool.run({"action": "read", "relative_path": "../../etc/passwd"})
            self.assertFalse(bad_res.success)
            self.assertIn("SecurityViolation", bad_res.error)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestEvidenceAndVerifier(unittest.TestCase):
    """Test 3: Evidence Collector & Deterministic Verifier."""

    def setUp(self):
        self.collector = EvidenceCollector()
        self.verifier = DeterministicVerifier()

    def test_evidence_collection_from_action(self):
        act = ActionResult(tool_name="data_tool", success=True, output={"count": 5}, duration_ms=12)
        ev = self.collector.collect_from_action(act)
        self.assertEqual(ev.source, "tool_execution:data_tool")
        self.assertEqual(ev.facts["output"]["count"], 5)

    def test_verifier_detects_false_success(self):
        # Action claimed success (e.g. status 200), but embedded output contains error payload
        act = ActionResult(tool_name="api_tool", success=True, output={"status_code": 500, "error": "Internal Error"})
        ev = self.collector.collect_from_action(act)

        rules = [
            AssertionRule(assertion_type="STATUS_SUCCESS"),
            AssertionRule(assertion_type="NO_ERROR_PAYLOAD"),
        ]

        v_res = self.verifier.verify(recovery_decision_id="rec-1", evidence=ev, assertions=rules)
        self.assertFalse(v_res.verified)
        self.assertGreater(len(v_res.discrepancies), 0)
        self.assertTrue(any("embedded error payload" in d for d in v_res.discrepancies))

    def test_verifier_asserts_exact_value_match(self):
        act = ActionResult(tool_name="calc_tool", success=True, output={"result": 42})
        ev = self.collector.collect_from_action(act)

        rules = [
            AssertionRule(assertion_type="VALUE_EQUALS", key_path="result", expected_value=42),
        ]
        v_res = self.verifier.verify(recovery_decision_id="rec-2", evidence=ev, assertions=rules)
        self.assertTrue(v_res.verified)


if __name__ == "__main__":
    unittest.main()
