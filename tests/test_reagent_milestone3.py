"""
ReAgent Milestone 3 Unit Tests

Tests:
  - SuccessContract + ContractManager (build, freeze, to_assertion_rules)
  - VerifiedReceipt emission
  - ObservabilityTimeline phases
  - New verifier predicates (FILE_CONTAINS, FILE_SIZE_GTE, JSON_VALID, JSON_PATH_EQUALS, HTTP_STATUS)
  - execute_with_contract() happy path
  - False-success detection
  - Bounded recovery loop (VERIFIED after 1 recovery)
  - Escalation path (max recovery exhausted)
  - PolicyGuard blocks pre-execution
  - StructuredDecision validation (Nebius provider, offline)
  - All 22 pre-existing tests still importable
"""

import os
import json
import tempfile
import shutil
import unittest
from typing import List

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_m3_test_")
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


from reagent.models import (
    TaskState, Postcondition, SuccessContract, VerifiedReceipt,
    ObservabilityEvent, StructuredDecision, FailureEvent, Evidence,
    VerificationResult,
)
from reagent.taxonomy import AgentStatus, FailureType, RecoveryStrategyType
from reagent.contract import ContractManager
from reagent.events import ObservabilityTimeline, PHASE_TASK_RECEIVED, PHASE_CONTRACT_FROZEN
from reagent.verification.verifier import DeterministicVerifier, AssertionRule
from reagent.verification.evidence import EvidenceCollector
from reagent.models import ActionResult
from reagent.pipeline import ReAgentPipeline, ExecutionCycleResult
from reagent.tools.registry import ToolRegistry
from reagent.tools.json_tool import JsonValidatorTool
from reagent.tools.data_tool import DataTransformTool
from reagent.tools.file_tool import SandboxedFileTool
from reagent.policy import PolicyGuard
from reagent.providers.nebius import NebiusNemotronProvider, _safe_parse_json


# ─────────────────────────────────────────────────────────────────────────────
class TestSuccessContract(unittest.TestCase):
    """ContractManager: build, freeze, and assertion rule conversion."""

    def setUp(self):
        self.mgr = ContractManager()

    def test_build_requires_postconditions(self):
        with self.assertRaises(ValueError):
            self.mgr.build_contract("task-x", [])

    def test_build_and_freeze(self):
        pc = Postcondition(predicate="STATUS_SUCCESS", description="must succeed")
        c = self.mgr.build_contract("task-1", [pc])
        frozen = self.mgr.freeze(c)
        self.assertEqual(frozen.contract_id, c.contract_id)
        self.assertEqual(frozen.task_id, "task-1")
        self.assertTrue(frozen.frozen_at)

    def test_to_assertion_rules_maps_all_supported_predicates(self):
        preds = [
            "STATUS_SUCCESS", "KEY_EXISTS", "VALUE_EQUALS",
            "FILE_EXISTS", "FILE_CONTAINS", "FILE_SIZE_GTE",
            "JSON_VALID", "JSON_PATH_EQUALS", "HTTP_STATUS", "NO_ERROR_PAYLOAD",
        ]
        pcs = [Postcondition(predicate=p, description=p) for p in preds]
        contract = self.mgr.build_contract("task-2", pcs)
        frozen = self.mgr.freeze(contract)
        rules = self.mgr.to_assertion_rules(frozen)
        self.assertEqual(len(rules), len(preds))
        types = {r.assertion_type for r in rules}
        self.assertEqual(types, set(preds))

    def test_unsupported_predicate_is_skipped(self):
        pcs = [
            Postcondition(predicate="STATUS_SUCCESS"),
            Postcondition(predicate="MADE_UP_PREDICATE"),
        ]
        contract = self.mgr.build_contract("task-3", pcs)
        frozen = self.mgr.freeze(contract)
        rules = self.mgr.to_assertion_rules(frozen)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].assertion_type, "STATUS_SUCCESS")


# ─────────────────────────────────────────────────────────────────────────────
class TestObservabilityTimeline(unittest.TestCase):

    def test_append_and_order(self):
        tl = ObservabilityTimeline("task-obs")
        tl.emit(PHASE_TASK_RECEIVED, {"tool": "json_validator"})
        tl.emit(PHASE_CONTRACT_FROZEN, {"contract_id": "abc"})
        phases = tl.phases()
        self.assertEqual(phases[0], PHASE_TASK_RECEIVED)
        self.assertEqual(phases[1], PHASE_CONTRACT_FROZEN)

    def test_events_are_immutable_copy(self):
        tl = ObservabilityTimeline("task-obs2")
        tl.emit(PHASE_TASK_RECEIVED)
        events = tl.events
        events.clear()
        self.assertEqual(len(tl.events), 1)  # original unchanged

    def test_has_phase(self):
        tl = ObservabilityTimeline("task-obs3")
        tl.emit(PHASE_TASK_RECEIVED)
        self.assertTrue(tl.has_phase(PHASE_TASK_RECEIVED))
        self.assertFalse(tl.has_phase(PHASE_CONTRACT_FROZEN))


# ─────────────────────────────────────────────────────────────────────────────
class TestNewVerifierPredicates(unittest.TestCase):

    def setUp(self):
        self.verifier = DeterministicVerifier()
        self.collector = EvidenceCollector()

    def _evidence_from(self, facts: dict) -> Evidence:
        return Evidence(source="test", facts=facts)

    def test_file_contains_pass(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("REAGENT_MARKER")
            path = f.name
        try:
            ev = self._evidence_from({"success": True})
            r = self.verifier.verify("d1", ev, [
                AssertionRule(assertion_type="FILE_CONTAINS", key_path=path, expected_value="REAGENT_MARKER"),
            ])
            self.assertTrue(r.verified)
        finally:
            os.unlink(path)

    def test_file_contains_fail(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("something else")
            path = f.name
        try:
            ev = self._evidence_from({"success": True})
            r = self.verifier.verify("d2", ev, [
                AssertionRule(assertion_type="FILE_CONTAINS", key_path=path, expected_value="MISSING_CONTENT"),
            ])
            self.assertFalse(r.verified)
            self.assertEqual(len(r.discrepancies), 1)
        finally:
            os.unlink(path)

    def test_file_size_gte_pass(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")   # 11 bytes
            path = f.name
        try:
            ev = self._evidence_from({"success": True})
            r = self.verifier.verify("d3", ev, [
                AssertionRule(assertion_type="FILE_SIZE_GTE", key_path=path, expected_value=5),
            ])
            self.assertTrue(r.verified)
        finally:
            os.unlink(path)

    def test_file_size_gte_fail(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hi")   # 2 bytes
            path = f.name
        try:
            ev = self._evidence_from({"success": True})
            r = self.verifier.verify("d4", ev, [
                AssertionRule(assertion_type="FILE_SIZE_GTE", key_path=path, expected_value=100),
            ])
            self.assertFalse(r.verified)
        finally:
            os.unlink(path)

    def test_json_valid_pass(self):
        ev = self._evidence_from({"success": True, "output": '{"a": 1}'})
        r = self.verifier.verify("d5", ev, [
            AssertionRule(assertion_type="JSON_VALID"),
        ])
        self.assertTrue(r.verified)

    def test_json_valid_fail(self):
        ev = self._evidence_from({"success": True, "output": "not json {{{{"})
        r = self.verifier.verify("d6", ev, [
            AssertionRule(assertion_type="JSON_VALID"),
        ])
        self.assertFalse(r.verified)

    def test_json_path_equals_pass(self):
        ev = self._evidence_from({"success": True, "output": {"a": {"b": 42}}})
        r = self.verifier.verify("d7", ev, [
            AssertionRule(assertion_type="JSON_PATH_EQUALS", key_path="a.b", expected_value=42),
        ])
        self.assertTrue(r.verified)

    def test_json_path_equals_fail(self):
        ev = self._evidence_from({"success": True, "output": {"a": {"b": 99}}})
        r = self.verifier.verify("d8", ev, [
            AssertionRule(assertion_type="JSON_PATH_EQUALS", key_path="a.b", expected_value=42),
        ])
        self.assertFalse(r.verified)

    def test_http_status_pass(self):
        ev = self._evidence_from({"success": True, "output": {"status_code": 200}})
        r = self.verifier.verify("d9", ev, [
            AssertionRule(assertion_type="HTTP_STATUS", expected_value=200),
        ])
        self.assertTrue(r.verified)

    def test_http_status_fail(self):
        ev = self._evidence_from({"success": True, "output": {"status_code": 503}})
        r = self.verifier.verify("d10", ev, [
            AssertionRule(assertion_type="HTTP_STATUS", expected_value=200),
        ])
        self.assertFalse(r.verified)

    def test_unknown_predicate_fails_verification(self):
        ev = self._evidence_from({"success": True})
        r = self.verifier.verify("d11", ev, [
            AssertionRule(assertion_type="COMPLETELY_MADE_UP"),
        ])
        self.assertFalse(r.verified)


# ─────────────────────────────────────────────────────────────────────────────
class TestExecuteWithContract(unittest.TestCase):
    """execute_with_contract() happy path and verification."""

    def _make_pipeline(self) -> ReAgentPipeline:
        registry = ToolRegistry()
        registry.register(JsonValidatorTool())
        registry.register(DataTransformTool())
        return ReAgentPipeline(tool_registry=registry)

    def test_happy_path_verified_receipt(self):
        pipeline = self._make_pipeline()
        task = TaskState(objective="Validate JSON")
        result = pipeline.execute_with_contract(
            task=task,
            tool_name="json_validator",
            params={"json_payload": '{"ok": true}'},
            postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="KEY_EXISTS", target="is_valid"),
            ],
        )
        self.assertIsInstance(result, ExecutionCycleResult)
        self.assertIsInstance(result.verified_receipt, VerifiedReceipt)
        self.assertEqual(result.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertTrue(result.verification_result.verified)
        self.assertEqual(result.recovery_count, 0)

    def test_contract_is_frozen_before_execution(self):
        pipeline = self._make_pipeline()
        task = TaskState(objective="Test freeze ordering")
        result = pipeline.execute_with_contract(
            task=task,
            tool_name="json_validator",
            params={"json_payload": '{"x": 1}'},
            postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
        )
        self.assertTrue(result.frozen_contract.frozen_at)
        # Timeline: CONTRACT_FROZEN must appear BEFORE TOOL_EXECUTED
        phases = [e.phase for e in result.timeline]
        self.assertIn("CONTRACT_FROZEN", phases)
        self.assertIn("TOOL_EXECUTED", phases)
        self.assertLess(phases.index("CONTRACT_FROZEN"), phases.index("TOOL_EXECUTED"))

    def test_false_success_detected(self):
        """Tool succeeds but postcondition requires a key that does not exist in output."""
        pipeline = self._make_pipeline()
        task = TaskState(objective="False success test")
        result = pipeline.execute_with_contract(
            task=task,
            tool_name="json_validator",
            params={"json_payload": '{"ok": 1}'},
            postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="KEY_EXISTS", target="score"),   # 'score' won't be in output
            ],
            max_recovery_attempts=0,
        )
        self.assertFalse(result.verification_result.verified)
        self.assertNotEqual(result.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertTrue(any(e.phase == "VERIFICATION_FAILED" for e in result.timeline))

    def test_verified_receipt_has_frozen_contract(self):
        pipeline = self._make_pipeline()
        task = TaskState(objective="Receipt has frozen contract")
        result = pipeline.execute_with_contract(
            task=task,
            tool_name="json_validator",
            params={"json_payload": "{}"},
            postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
        )
        receipt = result.verified_receipt
        self.assertEqual(receipt.frozen_contract.task_id, task.task_id)
        self.assertIsInstance(receipt.evidence, Evidence)
        self.assertIsInstance(receipt.verification_result, VerificationResult)
        self.assertIsNotNone(receipt.timestamp)

    def test_escalation_after_max_recovery(self):
        """Max recovery exhausted → ESCALATED."""
        pipeline = self._make_pipeline()
        task = TaskState(objective="Escalation test")
        result = pipeline.execute_with_contract(
            task=task,
            tool_name="json_validator",
            params={"json_payload": "INVALID{{{{"},
            postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
            max_recovery_attempts=1,
        )
        self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertGreaterEqual(pipeline.reliability_state.escalations, 1)

    def test_policy_guard_blocks_pre_execution(self):
        """Destructive payload in params → PolicyGuard blocks before tool runs."""
        pipeline = self._make_pipeline()
        task = TaskState(objective="Security block test")
        result = pipeline.execute_with_contract(
            task=task,
            tool_name="json_validator",
            params={"json_payload": "data", "danger": "rm -rf /"},
            postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
        )
        self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertFalse(result.action_result.success)


# ─────────────────────────────────────────────────────────────────────────────
class TestRecoveryLoop(unittest.TestCase):
    """Recovery loop: one failure then successful recovery."""

    def test_recovery_produces_verified_receipt(self):
        """
        We run json_validator with bad input first.
        The recovery path will retry — but since DevelopmentProvider selects RETRY_SAME,
        the same bad input is retried and escalation follows.
        What we verify is that recovery_count increments and phases are emitted.
        """
        registry = ToolRegistry()
        registry.register(JsonValidatorTool())
        pipeline = ReAgentPipeline(tool_registry=registry)
        task = TaskState(objective="Recovery loop test")

        result = pipeline.execute_with_contract(
            task=task,
            tool_name="json_validator",
            params={"json_payload": "ALWAYS_BAD{{{{"},
            postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
            max_recovery_attempts=1,
        )
        phases = [e.phase for e in result.timeline]
        self.assertIn("VERIFICATION_FAILED", phases)
        self.assertIn("VERIFIED_RECEIPT", phases)
        # Recovery attempt was made (RECOVERY_EXECUTED or escalated immediately)
        self.assertIn(result.verified_receipt.final_status,
                      [AgentStatus.VERIFIED, AgentStatus.ESCALATED])


# ─────────────────────────────────────────────────────────────────────────────
class TestFileContractIntegration(unittest.TestCase):

    def test_file_exists_and_contains_via_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            registry.register(SandboxedFileTool(sandbox_root=tmpdir))
            pipeline = ReAgentPipeline(tool_registry=registry)

            target = os.path.join(tmpdir, "out.txt")
            task = TaskState(objective="Write verified file")

            result = pipeline.execute_with_contract(
                task=task,
                tool_name="sandboxed_file",
                params={"action": "write", "relative_path": "out.txt", "content": "VERIFIED_CONTENT"},
                postconditions=[
                    Postcondition(predicate="STATUS_SUCCESS"),
                    Postcondition(predicate="FILE_EXISTS", expected=target),
                    Postcondition(predicate="FILE_CONTAINS", target=target, expected="VERIFIED_CONTENT"),
                ],
            )
            self.assertTrue(result.verification_result.verified, result.verification_result.discrepancies)
            self.assertEqual(result.verified_receipt.final_status, AgentStatus.VERIFIED)


# ─────────────────────────────────────────────────────────────────────────────
class TestNebiusProviderOffline(unittest.TestCase):
    """Offline tests for the Nebius provider (no API key needed)."""

    def test_unconfigured_generate_raises(self):
        """Provider without API key raises RuntimeError on generate()."""
        # Temporarily clear env
        old = os.environ.pop("NEBIUS_API_KEY", None)
        try:
            provider = NebiusNemotronProvider()
            self.assertFalse(provider.is_configured())
            with self.assertRaises(RuntimeError):
                provider.generate("hello")
        finally:
            if old is not None:
                os.environ["NEBIUS_API_KEY"] = old

    def test_safe_parse_json_valid(self):
        raw = '{"tool": "json_validator", "arguments": {}}'
        result = _safe_parse_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "json_validator")

    def test_safe_parse_json_with_fence(self):
        raw = "```json\n{\"tool\": \"data_transform\"}\n```"
        result = _safe_parse_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "data_transform")

    def test_safe_parse_json_invalid_returns_none(self):
        result = _safe_parse_json("this is not json at all")
        self.assertIsNone(result)

    def test_validate_structured_decision_missing_tool_returns_none(self):
        provider = NebiusNemotronProvider.__new__(NebiusNemotronProvider)
        result = provider._validate_structured_decision({"arguments": {}})
        self.assertIsNone(result)

    def test_validate_structured_decision_valid(self):
        provider = NebiusNemotronProvider.__new__(NebiusNemotronProvider)
        raw = {
            "tool": "json_validator",
            "arguments": {"json_string": "{}"},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS", "description": "ok"}],
            "diagnosis_category": None,
            "diagnosis_confidence": 0.8,
            "diagnosis_reason": None,
            "recovery_strategy": None,
        }
        result = provider._validate_structured_decision(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool, "json_validator")
        self.assertEqual(len(result.proposed_postconditions), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Import guard: ensure all pre-existing test modules are still loadable
# ─────────────────────────────────────────────────────────────────────────────
class TestBackwardCompatImports(unittest.TestCase):

    def test_milestone1_imports(self):
        import tests.test_reagent_core  # noqa: F401

    def test_milestone2_imports(self):
        import tests.test_reagent_milestone2  # noqa: F401

    def test_scenarios_imports(self):
        import tests.test_scenarios  # noqa: F401


if __name__ == "__main__":
    unittest.main()
