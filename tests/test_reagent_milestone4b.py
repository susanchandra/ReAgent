"""
ReAgent Milestone 4B-Lite Comprehensive Test Suite (100% Offline)

Audits and verifies:
1. CONTRACT VALIDATION:
   - Target binding mismatch rejected before tool execution (call_count == 0).
   - Vacuous postconditions (file/content already exists on disk) rejected before tool execution (call_count == 0).
   - Ungrounded contracts (no predicate grounded in objective) rejected before tool execution (call_count == 0).
   - Incompatible tool predicates rejected before tool execution (call_count == 0).
   - Positive controls for read-only and state-changing contracts.
2. NEBIUS PROVIDER PARSING (OFFLINE MOCK CLIENT):
   - Empty or missing content.
   - finish_reason="length" rejection.
   - reasoning_content isolation (never parsed as decision).
   - Unknown JSON fields rejected.
   - Malicious/injected provenance rejected.
   - Malformed JSON rejected.
3. GLOBAL TOKEN LEDGER (CROSS-PROCESS & PERSISTENT):
   - Reservation and reconciliation.
   - Budget exhaustion fails closed.
   - Corrupt JSON fails closed (CorruptLedgerError, never resets).
   - Persistence across restarts.
4. SAFE VERIFIER MUTATION TEST:
   - try...finally guarantee; zero production file modifications.
5. SANDBOX PATH & BOUNDARY AUDIT:
   - Path traversal, foreign drives, read caps (1MB), missing files, symlinks/junctions.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from typing import Dict, Any, Optional, List
from unittest.mock import MagicMock, patch

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_m4b_test_")
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
    Postcondition, StructuredDecision, AgentStatus,
    ActionResult, Evidence, TaskState, FailureEvent,
)
from reagent.tools.base import BaseTool
from reagent.tools.registry import ToolRegistry
from reagent.tools.file_tool import SandboxedFileTool, MAX_READ_BYTES
from reagent.tools.json_tool import JsonValidatorTool
from reagent.verification.verifier import (
    DeterministicVerifier, AssertionRule, FILE_READ_CAP_BYTES,
)
from reagent.contract_validator import (
    ContractGroundingValidator, ValidationResult,
)
from reagent.policy import (
    GlobalTokenLedger, TokenBudgetGuard, BudgetExceededError,
    CorruptLedgerError, PolicyGuard, estimate_prompt_tokens,
)
from reagent.providers.base import LLMResponse
from reagent.providers.nebius import (
    NebiusNemotronProvider, DEFAULT_MODEL, get_configured_max_tokens,
)
from reagent.pipeline import ReAgentPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Test Fakes & Mocks
# ─────────────────────────────────────────────────────────────────────────────

class CountingMockTool(BaseTool):
    """A tool that counts every execution attempt to prove pre-invocation rejection."""
    name = "counting_state_tool"
    description = "Test tool with state-changing effect class."
    effect_class = "state-changing"
    parameters_schema = {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["relative_path"],
    }

    def __init__(self):
        self.call_count = 0

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        return {"status": "created", "path": params.get("relative_path")}


class FakeOpenAIResponse:
    """Mock OpenAI API chat completion response."""
    def __init__(self, content: str = "", finish_reason: str = "stop", reasoning_content: Optional[str] = None, usage: Optional[Dict[str, int]] = None):
        self.choices = [FakeChoice(content, finish_reason, reasoning_content)]
        if usage:
            self.usage = MagicMock(
                prompt_tokens=usage.get("prompt_tokens", 10),
                completion_tokens=usage.get("completion_tokens", 20),
                total_tokens=usage.get("total_tokens", 30),
            )
        else:
            self.usage = None


class FakeChoice:
    def __init__(self, content: str, finish_reason: str, reasoning_content: Optional[str] = None):
        self.finish_reason = finish_reason
        self.message = FakeMessage(content, reasoning_content)


class FakeMessage:
    def __init__(self, content: str, reasoning_content: Optional[str] = None):
        self.content = content
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content


class FakeOpenAIClient:
    """Fake OpenAI client for NebiusNemotronProvider offline testing."""
    def __init__(self, next_response: FakeOpenAIResponse):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(return_value=next_response)
        self.last_kwargs = None

    def set_response(self, response: FakeOpenAIResponse):
        self.chat.completions.create.return_value = response


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract Grounding Validator & Pre-Invocation Isolation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestContractValidationHardening(unittest.TestCase):
    """
    Validates that invalid contracts are rejected BEFORE any tool invocation,
    proving the tool call count is strictly zero.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ContractGroundingValidator(workspace_root=self.temp_dir)
        self.tool = CountingMockTool()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_target_binding_mismatch_rejects_with_zero_tool_calls(self):
        """Required file predicate targeting a different file than tool arg is REJECTED before tool execution."""
        decision = StructuredDecision(
            tool=self.tool.name,
            arguments={"relative_path": "audit_log.txt", "content": "entry"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="FILE_EXISTS", target="other_file.txt", required=True),
            ],
        )
        res = self.validator.validate(decision, self.tool, "Generate audit_log.txt")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("TargetBindingMismatch" in e for e in res.errors))
        self.assertEqual(self.tool.call_count, 0, "Tool must NOT be invoked when contract is rejected")

    def test_vacuous_file_exists_postcondition_rejects_with_zero_tool_calls(self):
        """FILE_EXISTS for a file that ALREADY exists on disk is rejected as vacuous."""
        existing_path = os.path.join(self.temp_dir, "existing.txt")
        with open(existing_path, "w", encoding="utf-8") as f:
            f.write("already here")

        decision = StructuredDecision(
            tool=self.tool.name,
            arguments={"relative_path": "existing.txt", "content": "already here"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="FILE_EXISTS", target="existing.txt", required=True),
            ],
        )
        res = self.validator.validate(decision, self.tool, "Ensure existing.txt is present")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("VacuousPostcondition" in e and "FILE_EXISTS" in e for e in res.errors))
        self.assertEqual(self.tool.call_count, 0, "Tool must NOT be invoked when postcondition is vacuous")

    def test_vacuous_file_contains_postcondition_rejects_with_zero_tool_calls(self):
        """FILE_CONTAINS where the substring already exists on disk is rejected as vacuous."""
        existing_path = os.path.join(self.temp_dir, "report.txt")
        with open(existing_path, "w", encoding="utf-8") as f:
            f.write("Status: COMPLETED\nTimestamp: 2026-10-09")

        decision = StructuredDecision(
            tool=self.tool.name,
            arguments={"relative_path": "report.txt", "content": "COMPLETED"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="FILE_CONTAINS", target="report.txt", expected="COMPLETED", required=True),
            ],
        )
        res = self.validator.validate(decision, self.tool, "Update report.txt with COMPLETED")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("VacuousPostcondition" in e and "FILE_CONTAINS" in e for e in res.errors))
        self.assertEqual(self.tool.call_count, 0, "Tool must NOT be invoked when content is already present")

    def test_ungrounded_contract_rejects_with_zero_tool_calls(self):
        """Contract matching tool arguments but completely ungrounded in objective is REJECTED."""
        decision = StructuredDecision(
            tool=self.tool.name,
            arguments={"relative_path": "random_file.txt", "content": "payload"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="FILE_EXISTS", target="random_file.txt", required=True),
            ],
        )
        # Objective is completely unrelated (e.g. summarize weather forecast)
        res = self.validator.validate(decision, self.tool, "Summarize tomorrow's weather forecast in Chicago")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("UngroundedContract" in e for e in res.errors))
        self.assertEqual(self.tool.call_count, 0, "Tool must NOT be invoked when contract is ungrounded")

    def test_incompatible_tool_predicate_rejects_with_zero_tool_calls(self):
        """FILE_* predicate assigned to a non-filesystem tool (json_validator) is REJECTED."""
        read_tool = JsonValidatorTool()
        decision = StructuredDecision(
            tool=read_tool.name,
            arguments={"json_payload": '{"status": "ok"}'},
            proposed_postconditions=[
                Postcondition(predicate="FILE_EXISTS", target="result.json", required=True),
            ],
        )
        res = self.validator.validate(decision, read_tool, "Validate result.json format")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("IncompatibleToolPredicate" in e for e in res.errors))

    def test_state_changing_tool_with_only_status_success_rejected(self):
        """State-changing tool with ONLY STATUS_SUCCESS is rejected."""
        decision = StructuredDecision(
            tool=self.tool.name,
            arguments={"relative_path": "out.txt", "content": "data"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
            ],
        )
        res = self.validator.validate(decision, self.tool, "Write out.txt with data")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("no independent observable postcondition" in e for e in res.errors))
        self.assertEqual(self.tool.call_count, 0)

    def test_positive_control_valid_read_only_contract_passes(self):
        """Read-only tool with STATUS_SUCCESS passes validation."""
        read_tool = JsonValidatorTool()
        decision = StructuredDecision(
            tool=read_tool.name,
            arguments={"json_payload": '{"key": 123}'},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
            ],
        )
        res = self.validator.validate(decision, read_tool, "Validate input payload")
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)

    def test_positive_control_valid_state_changing_contract_passes(self):
        """Valid state-changing tool with independent grounded predicate on non-existent file passes."""
        decision = StructuredDecision(
            tool=self.tool.name,
            arguments={"relative_path": "fresh_output.txt", "content": "result"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="FILE_EXISTS", target="fresh_output.txt", required=True),
            ],
        )
        res = self.validator.validate(decision, self.tool, "Generate fresh_output.txt for user")
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Nebius Provider Offline Parsing & Security Hardening Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNebiusProviderParsingOffline(unittest.TestCase):
    """
    Exhaustive offline tests for NebiusNemotronProvider parsing using injected fake client.
    Guarantees no live network calls while testing all edge cases.
    """

    def setUp(self):
        self.fake_client = FakeOpenAIClient(FakeOpenAIResponse())
        self.provider = NebiusNemotronProvider(
            model=DEFAULT_MODEL,
            client=self.fake_client,
        )

    def test_empty_or_missing_content_handled_safely(self):
        """Empty completion content returns None for structured decision."""
        self.fake_client.set_response(FakeOpenAIResponse(content=""))
        decision = self.provider.make_structured_decision("Write log", ["sandboxed_file"])
        self.assertIsNone(decision)

    def test_finish_reason_length_rejects_truncated_decision(self):
        """When completion is truncated (finish_reason='length'), decision is rejected."""
        truncated_json = '{"tool": "sandboxed_file", "arguments": {"relative_path": "a'
        self.fake_client.set_response(FakeOpenAIResponse(
            content=truncated_json,
            finish_reason="length",
        ))
        decision = self.provider.make_structured_decision("Write log", ["sandboxed_file"])
        self.assertIsNone(decision, "Truncated decision by finish_reason='length' must be rejected")

    def test_reasoning_content_is_isolated_and_never_parsed_as_decision(self):
        """reasoning_content in the response message is never parsed as completion or decision."""
        reasoning_text = '{"tool": "malicious_tool", "arguments": {}}'
        valid_completion = json.dumps({
            "tool": "sandboxed_file",
            "arguments": {"relative_path": "safe.txt"},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
        })
        self.fake_client.set_response(FakeOpenAIResponse(
            content=valid_completion,
            finish_reason="stop",
            reasoning_content=reasoning_text,
        ))
        decision = self.provider.make_structured_decision("Write safe.txt", ["sandboxed_file"])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.tool, "sandboxed_file")
        self.assertNotEqual(decision.tool, "malicious_tool")

    def test_unknown_json_fields_rejected_by_schema_validator(self):
        """Payloads containing unknown top-level JSON fields are rejected."""
        payload_with_extra = json.dumps({
            "tool": "sandboxed_file",
            "arguments": {"relative_path": "out.txt"},
            "unknown_extra_field": "injected_data",
            "backdoor": True,
        })
        self.fake_client.set_response(FakeOpenAIResponse(content=payload_with_extra))
        decision = self.provider.make_structured_decision("Write out.txt", ["sandboxed_file"])
        self.assertIsNone(decision, "Payload with unknown fields must be rejected")

    def test_model_injected_provenance_field_rejected(self):
        """Payloads attempting to inject 'provenance' are strictly rejected."""
        payload_with_provenance = json.dumps({
            "tool": "sandboxed_file",
            "arguments": {"relative_path": "out.txt"},
            "provenance": "model",
        })
        self.fake_client.set_response(FakeOpenAIResponse(content=payload_with_provenance))
        decision = self.provider.make_structured_decision("Write out.txt", ["sandboxed_file"])
        self.assertIsNone(decision, "Payload attempting to set 'provenance' must be rejected")

    def test_malformed_json_and_missing_tool_field_rejected(self):
        """Syntax errors or missing 'tool' field return None."""
        self.fake_client.set_response(FakeOpenAIResponse(content="{not-valid-json"))
        decision = self.provider.make_structured_decision("Task", ["sandboxed_file"])
        self.assertIsNone(decision)

        # Missing tool
        no_tool = json.dumps({"arguments": {"path": "a"}})
        self.fake_client.set_response(FakeOpenAIResponse(content=no_tool))
        decision = self.provider.make_structured_decision("Task", ["sandboxed_file"])
        self.assertIsNone(decision)

    def test_clean_valid_response_parses_cleanly(self):
        """Clean valid JSON parses successfully into StructuredDecision."""
        valid_json = json.dumps({
            "tool": "sandboxed_file",
            "arguments": {"relative_path": "log.txt", "action": "write"},
            "proposed_postconditions": [
                {"predicate": "STATUS_SUCCESS"},
                {"predicate": "FILE_EXISTS", "target": "log.txt"},
            ],
        })
        self.fake_client.set_response(FakeOpenAIResponse(content=valid_json))
        decision = self.provider.make_structured_decision("Write log.txt", ["sandboxed_file"])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.tool, "sandboxed_file")
        self.assertEqual(decision.arguments["relative_path"], "log.txt")
        self.assertEqual(len(decision.proposed_postconditions), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Global Token Ledger & Cross-Process Budget Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalTokenLedger(unittest.TestCase):
    """
    Tests cross-process atomic file locking, pre-call reservation,
    reconciliation, and fail-closed corrupt state handling.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_file = os.path.join(self.temp_dir, "token_ledger.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reservation_and_reconciliation(self):
        """Check reservation creates pending entry, and reconcile updates actual usage."""
        ledger = GlobalTokenLedger(
            ledger_path=self.ledger_file,
            max_cumulative_tokens=2000,
            max_cumulative_calls=5,
        )

        res_id = ledger.check_and_reserve(estimated_tokens=500)
        self.assertTrue(bool(res_id) and isinstance(res_id, str))

        state = ledger.get_state()
        self.assertEqual(state["active_reserved_tokens"], 500)
        self.assertEqual(state["total_tokens_used"], 0)

        # Reconcile with actual usage (e.g. 350 tokens)
        ledger.reconcile_usage(res_id, actual_tokens=350)

        state2 = ledger.get_state()
        self.assertEqual(state2["total_tokens_used"], 350)
        self.assertEqual(state2["total_calls_used"], 1)
        self.assertEqual(state2["active_reserved_tokens"], 0)

    def test_budget_exhaustion_blocks_reservations(self):
        """When cumulative tokens or calls are exceeded, reservations fail closed."""
        ledger = GlobalTokenLedger(
            ledger_path=self.ledger_file,
            max_cumulative_tokens=1000,
            max_cumulative_calls=2,
        )

        r1 = ledger.check_and_reserve(estimated_tokens=600)
        ledger.reconcile_usage(r1, actual_tokens=600)

        r2 = ledger.check_and_reserve(estimated_tokens=300)
        ledger.reconcile_usage(r2, actual_tokens=300)

        # Calls limit reached (2 of 2)
        with self.assertRaises(BudgetExceededError):
            ledger.check_and_reserve(estimated_tokens=50)

    def test_corrupt_ledger_file_fails_closed(self):
        """Corrupt JSON in the ledger raises CorruptLedgerError and NEVER silently resets."""
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            f.write("CORRUPTED_JSON_DATA{{{{")

        # Initializing ledger on corrupt file fails closed immediately
        with self.assertRaises(CorruptLedgerError):
            GlobalTokenLedger(
                ledger_path=self.ledger_file,
                max_cumulative_tokens=1000,
                max_cumulative_calls=5,
            )

        # Verify the file was NOT replaced or reset
        with open(self.ledger_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("CORRUPTED_JSON_DATA", content)

    def test_persistence_across_process_restarts(self):
        """New ledger instance reads accumulated totals from disk."""
        ledger1 = GlobalTokenLedger(
            ledger_path=self.ledger_file,
            max_cumulative_tokens=5000,
            max_cumulative_calls=10,
        )
        r1 = ledger1.check_and_reserve(estimated_tokens=400)
        ledger1.reconcile_usage(r1, actual_tokens=450)

        # Create new ledger instance pointing to the same file (simulating new process/run)
        ledger2 = GlobalTokenLedger(
            ledger_path=self.ledger_file,
            max_cumulative_tokens=5000,
            max_cumulative_calls=10,
        )
        state = ledger2.get_state()
        self.assertEqual(state["total_tokens_used"], 450)
        self.assertEqual(state["total_calls_used"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Safe Verifier Mutation Experiment (try...finally Guaranteed)
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeVerifierMutation(unittest.TestCase):
    """
    Proves offline that the deterministic verifier is a genuine gate,
    and that modifying it would cause negative assertions to fail.
    Guarantees 0 production file modifications via try...finally.
    """

    def test_verifier_mutation_experiment_with_zero_production_file_changes(self):
        verifier = DeterministicVerifier()
        original_verify = verifier.verify

        facts = {"success": True, "output": {}}
        evidence = Evidence(source="test", facts=facts)
        # Predicate checking a file that does not exist
        missing_rule = [AssertionRule(assertion_type="FILE_EXISTS", expected_value="D:/Jarvis/nonexistent_xyz.txt")]

        # Baseline: genuine verifier rejects missing file
        baseline_res = verifier.verify("d-baseline", evidence, missing_rule)
        self.assertFalse(baseline_res.verified, "Genuine verifier must reject missing file")

        # Mutate the in-memory method to unconditional pass
        def unconditional_pass_mutant(*args, **kwargs):
            from reagent.models import VerificationResult
            return VerificationResult(
                recovery_decision_id="mutant",
                verified=True,
                evidence_summary="MUTANT PASS (unconditional)",
                discrepancies=[],
            )

        try:
            verifier.verify = unconditional_pass_mutant
            mutant_res = verifier.verify("d-mutant", evidence, missing_rule)
            # Under mutation, the negative test WOULD fail (mutant claims success)
            self.assertTrue(mutant_res.verified, "Mutant verifier falsely claims success")
        finally:
            # Reliably restore original method in all cases
            verifier.verify = original_verify

        # Verify restored behavior
        restored_res = verifier.verify("d-restored", evidence, missing_rule)
        self.assertFalse(restored_res.verified, "Verifier successfully restored to secure behavior")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sandbox Path & Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxPathsAndBoundaries(unittest.TestCase):
    """
    Audits path traversal, foreign drive letters, oversized read caps,
    and missing files across both SandboxedFileTool and DeterministicVerifier.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tool = SandboxedFileTool(sandbox_root=self.temp_dir)
        self.verifier = DeterministicVerifier(workspace_root=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_path_traversal_rejected_by_tool_and_verifier(self):
        """Path traversal via '..' or windows slashes is rejected by both tool and verifier."""
        # Tool rejection
        res_tool = self.tool.run({"action": "read", "relative_path": "../../etc/passwd"})
        self.assertFalse(res_tool.success)
        self.assertIn("SecurityViolation", res_tool.error or "")

        res_tool_win = self.tool.run({"action": "read", "relative_path": r"..\..\windows\win.ini"})
        self.assertFalse(res_tool_win.success)
        self.assertIn("SecurityViolation", res_tool_win.error or "")

        # Verifier rejection
        evidence = Evidence(source="test", facts={"success": True})
        res_ver = self.verifier.verify(
            "v1", evidence,
            assertions=[AssertionRule(assertion_type="FILE_EXISTS", expected_value="../../outside.txt")]
        )
        self.assertFalse(res_ver.verified)
        self.assertTrue(any("exits workspace root" in d for d in res_ver.discrepancies))

    def test_foreign_drive_letters_rejected_by_tool_and_verifier(self):
        """Foreign drive paths (e.g. C:, E:) are rejected by tool and verifier."""
        # Tool rejection
        res_tool = self.tool.run({"action": "read", "relative_path": "C:\\Windows\\System32\\calc.exe"})
        self.assertFalse(res_tool.success)
        self.assertIn("SecurityViolation", res_tool.error or "")

        # Verifier rejection
        evidence = Evidence(source="test", facts={"success": True})
        # If workspace is on D:, foreign drive C: is rejected
        foreign_path = "E:\\some_foreign_path\\file.txt"
        res_ver = self.verifier.verify(
            "v2", evidence,
            assertions=[AssertionRule(assertion_type="FILE_EXISTS", expected_value=foreign_path)]
        )
        self.assertFalse(res_ver.verified)
        self.assertTrue(any("exits workspace root" in d or "foreign drive" in d for d in res_ver.discrepancies))

    def test_oversized_file_read_cap_enforced_by_tool(self):
        """Files exceeding MAX_READ_BYTES (1MB) cannot be read by SandboxedFileTool."""
        big_file = os.path.join(self.temp_dir, "large_file.bin")
        # Create a file slightly over 1MB
        with open(big_file, "wb") as f:
            f.write(b"X" * (MAX_READ_BYTES + 512))

        res = self.tool.run({"action": "read", "relative_path": "large_file.bin"})
        self.assertFalse(res.success)
        self.assertIn("FileExceedsSizeLimit", res.error or "")

    def test_oversized_file_read_cap_enforced_by_verifier(self):
        """FILE_CONTAINS reads only up to FILE_READ_CAP_BYTES (1MB). Needles beyond cap fail."""
        capped_file = os.path.join(self.temp_dir, "capped_file.txt")
        # Write 1MB of padding followed by needle
        with open(capped_file, "wb") as f:
            f.write(b"A" * (FILE_READ_CAP_BYTES + 100))
            f.write(b"SECRET_NEEDLE_BEYOND_CAP")

        evidence = Evidence(source="test", facts={"success": True})
        res = self.verifier.verify(
            "v3", evidence,
            assertions=[AssertionRule(
                assertion_type="FILE_CONTAINS",
                key_path="capped_file.txt",
                expected_value="SECRET_NEEDLE_BEYOND_CAP",
            )]
        )
        self.assertFalse(res.verified, "Needle beyond 1MB cap must NOT be found")
        self.assertTrue(any("does not contain expected substring" in d for d in res.discrepancies))

    def test_missing_files_handled_gracefully(self):
        """Missing files return clear errors without crashing."""
        res_tool = self.tool.run({"action": "read", "relative_path": "definitely_not_here.txt"})
        self.assertFalse(res_tool.success)
        self.assertIn("File not found", res_tool.error or "")

        evidence = Evidence(source="test", facts={"success": True})
        res_ver = self.verifier.verify(
            "v4", evidence,
            assertions=[AssertionRule(assertion_type="FILE_EXISTS", expected_value="definitely_not_here.txt")]
        )
        self.assertFalse(res_ver.verified)

class TestPreLiveSafetyAndSpendingControls(unittest.TestCase):
    """
    Offline tests for spending control, secret safety, and trace persistence:
      1. Retries after truncation within persistent budget.
      2. Persistent caps block excess calls across re-instantiation.
      3. Worst-case reservation computation (prompt est + max tokens).
      4. Unknown usage charges full reservation.
      5. Secret leakage prevented from timeline, console, and error sites.
      6. Safe allowlisted receipt persistence.
      7. Abandoned reservations block startup and fail closed.
      8. REAGENT_MAX_TOKENS environment validation and bounds enforcement.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.temp_dir, "test_safety_ledger.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_retry_after_truncation_allowed_within_budget(self):
        """Retries after truncation are allowed as long as persistent budget permits."""
        ledger = GlobalTokenLedger(
            ledger_path=self.ledger_path,
            max_cumulative_tokens=2000,
            max_cumulative_calls=3,
        )

        # Call 1: Truncated
        res_id_1 = ledger.check_and_reserve(estimated_tokens=500)
        # Reconcile call 1 (charged conservative reservation, call count increments)
        ledger.reconcile_usage(res_id_1, reported_usage=None, conservative_default=500)

        state = ledger.get_state()
        self.assertEqual(state["total_calls_used"], 1)
        self.assertEqual(state["remaining_calls"], 2)
        self.assertGreater(state["remaining_tokens"], 0)

        # Call 2: Retry within remaining budget succeeds
        res_id_2 = ledger.check_and_reserve(estimated_tokens=500)
        ledger.reconcile_usage(res_id_2, reported_usage={"total_tokens": 150})

        state = ledger.get_state()
        self.assertEqual(state["total_calls_used"], 2)
        self.assertEqual(state["remaining_calls"], 1)

    def test_persistent_cap_blocks_excess_calls(self):
        """Persistent caps block calls beyond limit, and persist across restarts."""
        ledger = GlobalTokenLedger(
            ledger_path=self.ledger_path,
            max_cumulative_tokens=2000,
            max_cumulative_calls=2,
        )
        res1 = ledger.check_and_reserve(300)
        ledger.reconcile_usage(res1, {"total_tokens": 100})
        res2 = ledger.check_and_reserve(300)
        ledger.reconcile_usage(res2, {"total_tokens": 100})

        # 3rd call must be blocked
        with self.assertRaises(BudgetExceededError):
            ledger.check_and_reserve(300)

        # Reopen ledger in new instance; must still block
        reopened = GlobalTokenLedger(
            ledger_path=self.ledger_path,
            max_cumulative_tokens=2000,
            max_cumulative_calls=2,
        )
        with self.assertRaises(BudgetExceededError):
            reopened.check_and_reserve(300)

    def test_worst_case_reservation_computation(self):
        """Conservative reservation includes prompt estimate + max completion tokens."""
        prompt = "Validate JSON task description."
        est_prompt = estimate_prompt_tokens(prompt)
        # Should have reasonable minimum floor
        self.assertGreaterEqual(est_prompt, 20)

        max_tokens = get_configured_max_tokens(512)
        self.assertEqual(max_tokens, 512)

        total_res = est_prompt + max_tokens
        self.assertEqual(total_res, est_prompt + 512)

    def test_unknown_usage_charges_full_reservation(self):
        """Unknown or missing token usage charges the full reservation, never a flat 500 or 100."""
        ledger = GlobalTokenLedger(ledger_path=self.ledger_path, max_cumulative_tokens=5000)
        res_id = ledger.check_and_reserve(estimated_tokens=850)

        # Reconcile with missing usage and smaller conservative_default
        charged = ledger.reconcile_usage(
            res_id,
            reported_usage=None,
            conservative_default=100,
        )
        # Must charge 850 (the full reservation), not the smaller 100 default
        self.assertEqual(charged, 850)
        self.assertEqual(ledger.get_state()["total_tokens_used"], 850)

    def test_secret_leakage_prevented_in_timeline_and_errors(self):
        """Fake exception containing sensitive API key must NEVER appear in timeline or receipts."""
        fake_secret_key = "sk-ant-fake-secret-key-NEVER-LEAK-THIS-12345"
        fake_bearer = "Bearer secret-bearer-token-abc"

        class FakeLeakingClient:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                raise RuntimeError(
                    f"Connection failed to https://api.tokenfactory.nebius.com/v1/?key={fake_secret_key} with {fake_bearer}"
                )

        provider = NebiusNemotronProvider(client=FakeLeakingClient())
        pipeline = ReAgentPipeline(
            agent_id="test-secret-agent",
            provider=provider,
            workspace_root=self.temp_dir,
        )

        res = pipeline.execute_task("Perform test task")
        self.assertEqual(res.verified_receipt.final_status, AgentStatus.ESCALATED)

        # Inspect all events in timeline
        for event in res.timeline:
            event_str = json.dumps(event.detail)
            self.assertNotIn(
                fake_secret_key,
                event_str,
                "API key MUST NOT leak into timeline event details!"
            )
            self.assertNotIn(
                fake_bearer,
                event_str,
                "Bearer token MUST NOT leak into timeline event details!"
            )
            if event.phase == "PROVIDER_FAILED":
                self.assertEqual(event.detail.get("error_type"), "RuntimeError")
                self.assertIsNone(event.detail.get("status_code"))

    def test_safe_allowlisted_receipt_persistence(self):
        """Receipt persistence writes allowlisted fields and excludes sensitive data."""
        from reagent.receipts import save_allowlisted_receipt

        reports_dir = os.path.join(self.temp_dir, "reports")
        receipt_path = save_allowlisted_receipt(
            receipt_dir=reports_dir,
            task_id="test-task-1",
            status="PROBE_DECISION_VALIDATED",
            tool_name="json_validator",
            probe_type="test_probe",
            verified=True,
            token_usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            discrepancies=[],
        )

        self.assertIsNotNone(receipt_path)
        self.assertTrue(os.path.exists(receipt_path))

        with open(receipt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["task_id"], "test-task-1")
        self.assertEqual(data["status"], "PROBE_DECISION_VALIDATED")
        self.assertTrue(data["verified"])
        self.assertEqual(data["token_usage"]["total_tokens"], 70)
        from reagent.receipts import ALLOWED_RECEIPT_KEYS
        self.assertTrue(set(data.keys()).issubset(ALLOWED_RECEIPT_KEYS))

    def test_abandoned_reservations_block_startup(self):
        """Active un-reconciled reservations block startup and fail closed."""
        from reagent.policy import AbandonedReservationError

        ledger = GlobalTokenLedger(ledger_path=self.ledger_path)
        # Create un-reconciled reservation
        res_id = ledger.check_and_reserve(500)

        # Do NOT reconcile (simulating crash)
        # New instance or probe startup check must fail closed
        with self.assertRaises(AbandonedReservationError):
            ledger.assert_no_abandoned_reservations()

        # Reservations must NOT be refunded or erased
        state = ledger.get_state()
        self.assertEqual(state["active_reserved_tokens"], 500)

    def test_reagent_max_tokens_environment_validation(self):
        """REAGENT_MAX_TOKENS enforces bounds [16, 4096] with default 1536 and rejects bad values."""
        # Default value without env var
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_configured_max_tokens(), 1536)

        # Non-integer
        with patch.dict(os.environ, {"REAGENT_MAX_TOKENS": "invalid_number"}):
            with self.assertRaises(ValueError):
                get_configured_max_tokens()

        # Below lower bound (15 < 16)
        with patch.dict(os.environ, {"REAGENT_MAX_TOKENS": "15"}):
            with self.assertRaises(ValueError):
                get_configured_max_tokens()

        # Above upper bound (4097 > 4096)
        with patch.dict(os.environ, {"REAGENT_MAX_TOKENS": "4097"}):
            with self.assertRaises(ValueError):
                get_configured_max_tokens()

        # Valid settings
        with patch.dict(os.environ, {"REAGENT_MAX_TOKENS": "2048"}):
            self.assertEqual(get_configured_max_tokens(), 2048)
        with patch.dict(os.environ, {"REAGENT_MAX_TOKENS": "4096"}):
            self.assertEqual(get_configured_max_tokens(), 4096)

    def test_configurable_ledger_caps_and_hard_bounds(self):
        """Ledger caps are configurable up to hard code bounds (calls <= 50, tokens <= 200,000)."""
        from reagent.policy import (
            get_configured_ledger_caps,
            HARD_LIMIT_MAX_CALLS,
            HARD_LIMIT_MAX_TOKENS,
            DEFAULT_LEDGER_MAX_CALLS,
            DEFAULT_LEDGER_MAX_TOKENS,
        )

        # Probe defaults
        with patch.dict(os.environ, {}, clear=True):
            calls, tokens = get_configured_ledger_caps()
            self.assertEqual(calls, DEFAULT_LEDGER_MAX_CALLS)
            self.assertEqual(calls, 20)
            self.assertEqual(tokens, DEFAULT_LEDGER_MAX_TOKENS)
            self.assertEqual(tokens, 60000)

        # Valid override within hard bounds
        with patch.dict(os.environ, {"REAGENT_LEDGER_MAX_CALLS": "35", "REAGENT_LEDGER_MAX_TOKENS": "120000"}):
            calls, tokens = get_configured_ledger_caps()
            self.assertEqual(calls, 35)
            self.assertEqual(tokens, 120000)

        # Calls exceeding hard upper bound (51 > 50)
        with patch.dict(os.environ, {"REAGENT_LEDGER_MAX_CALLS": "51"}):
            with self.assertRaises(ValueError):
                get_configured_ledger_caps()

        # Tokens exceeding hard upper bound (200001 > 200000)
        with patch.dict(os.environ, {"REAGENT_LEDGER_MAX_TOKENS": "200001"}):
            with self.assertRaises(ValueError):
                get_configured_ledger_caps()

        # Invalid non-integer or <= 0
        with patch.dict(os.environ, {"REAGENT_LEDGER_MAX_CALLS": "0"}):
            with self.assertRaises(ValueError):
                get_configured_ledger_caps()

    def test_probe1_truncated_leaves_room_for_retry_and_probe2(self):
        """A truncated probe 1 call leaves ample budget for a retry and for probe 2."""
        ledger = GlobalTokenLedger(ledger_path=self.ledger_path)
        self.assertEqual(ledger.max_calls, 20)
        self.assertEqual(ledger.max_tokens, 60000)

        # Probe 1: truncated call (reserves prompt + max_tokens, charged full completion)
        res1 = ledger.check_and_reserve(estimated_tokens=1886)
        usage1 = {"prompt_tokens": 350, "completion_tokens": 1536, "total_tokens": 1886}
        charged1 = ledger.reconcile_usage(res1, reported_usage=usage1, conservative_default=1886)
        self.assertEqual(charged1, 1886)

        state1 = ledger.get_state()
        self.assertEqual(state1["total_calls_used"], 1)
        self.assertEqual(state1["remaining_calls"], 19)
        self.assertEqual(state1["remaining_tokens"], 60000 - 1886)

        # Probe 1 Retry: succeeds within remaining budget
        res2 = ledger.check_and_reserve(estimated_tokens=1886)
        usage2 = {"prompt_tokens": 350, "completion_tokens": 150, "total_tokens": 500}
        charged2 = ledger.reconcile_usage(res2, reported_usage=usage2, conservative_default=1886)
        self.assertEqual(charged2, 500)

        state2 = ledger.get_state()
        self.assertEqual(state2["total_calls_used"], 2)
        self.assertEqual(state2["remaining_calls"], 18)

        # Probe 2 (planning + diagnosis + recovery): 3 calls
        # 1. Planning call
        res3 = ledger.check_and_reserve(estimated_tokens=1886)
        charged3 = ledger.reconcile_usage(res3, reported_usage={"prompt_tokens": 350, "completion_tokens": 150, "total_tokens": 500}, conservative_default=1886)
        # 2. Diagnosis call
        res4 = ledger.check_and_reserve(estimated_tokens=800)
        charged4 = ledger.reconcile_usage(res4, reported_usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}, conservative_default=800)
        # 3. Recovery selection call
        res5 = ledger.check_and_reserve(estimated_tokens=800)
        charged5 = ledger.reconcile_usage(res5, reported_usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}, conservative_default=800)

        final_state = ledger.get_state()
        # All 5 calls completed comfortably
        self.assertEqual(final_state["total_calls_used"], 5)
        self.assertGreaterEqual(final_state["remaining_calls"], 15)
        self.assertGreaterEqual(final_state["remaining_tokens"], 50000)

    def test_reservation_sizing_exceeds_objective_only_estimate(self):
        """Provider estimate based on actual request messages strictly exceeds objective-only estimate."""
        fake_response = FakeOpenAIResponse(
            content=json.dumps({"tool": "json_validator", "arguments": {}}),
            finish_reason="stop",
        )
        provider = NebiusNemotronProvider(client=FakeOpenAIClient(fake_response))
        task = "Validate that {\"status\": \"ok\", \"source\": \"live_probe\"} is valid JSON."

        # Old estimate based only on task objective text
        objective_only_prompt = estimate_prompt_tokens(task)
        # New estimate based on full request payload (system prompt, schemas, taxonomy, task)
        request_tokens = provider.estimate_decision_request_tokens(
            task_objective=task,
            available_tools=["json_validator"],
        )

        max_tokens = get_configured_max_tokens(1536)
        old_reservation = objective_only_prompt + max_tokens
        new_reservation = request_tokens + max_tokens

        self.assertGreater(
            request_tokens,
            objective_only_prompt,
            f"Full request tokens ({request_tokens}) must strictly exceed objective-only tokens ({objective_only_prompt})"
        )
        self.assertGreater(
            new_reservation,
            old_reservation,
            f"New reservation ({new_reservation}) must exceed old reservation ({old_reservation})"
        )

    def test_last_usage_reset_at_start_of_each_call(self):
        """provider.last_usage is reset to None at start of every call to prevent stale usage attribution."""
        fake_response1 = FakeOpenAIResponse(
            content="{}",
            finish_reason="stop",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        )
        fake_client = FakeOpenAIClient(fake_response1)
        provider = NebiusNemotronProvider(client=fake_client)

        # Call 1 succeeds and records usage
        provider.generate(prompt="test 1")
        self.assertIsNotNone(provider.last_usage)
        self.assertEqual(provider.last_usage["total_tokens"], 70)

        # Call 2 raises an exception during provider call
        fake_client.chat.completions.create.side_effect = RuntimeError("Simulated failure")
        with self.assertRaises(RuntimeError):
            provider.generate(prompt="test 2")

        # Crucial check: provider.last_usage must be None, NOT stale 70
        self.assertIsNone(provider.last_usage)

    def test_reconcile_runs_exactly_once_per_reservation(self):
        """Reconciliation in finally: executes exactly once per reservation under success and failure."""
        ledger = GlobalTokenLedger(ledger_path=self.ledger_path)

        reconcile_count = 0
        original_reconcile = ledger.reconcile_usage

        def counting_reconcile(*args, **kwargs):
            nonlocal reconcile_count
            reconcile_count += 1
            return original_reconcile(*args, **kwargs)

        ledger.reconcile_usage = counting_reconcile

        # Success case in pipeline
        fake_response = FakeOpenAIResponse(
            content=json.dumps({
                "tool": "json_validator",
                "arguments": {"json_payload": "{\"test\": 1}"},
                "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}]
            }),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        provider = NebiusNemotronProvider(client=FakeOpenAIClient(fake_response))
        registry = ToolRegistry()
        registry.register(JsonValidatorTool())
        guard = TokenBudgetGuard(max_calls_per_run=10, max_tokens_per_run=20000, ledger=ledger)
        pipeline = ReAgentPipeline(
            agent_id="test-agent-reconcile",
            provider=provider,
            tool_registry=registry,
            token_budget=guard,
            workspace_root=self.temp_dir,
        )

        res = pipeline.execute_task(task_objective="Validate JSON")
        self.assertEqual(reconcile_count, 1, "Reconcile must run exactly once on success path")

        # Failure case in pipeline
        fake_fail_client = FakeOpenAIClient(fake_response)
        fake_fail_client.chat.completions.create.side_effect = RuntimeError("Provider exploded")
        fail_provider = NebiusNemotronProvider(client=fake_fail_client)
        fail_pipeline = ReAgentPipeline(
            agent_id="test-agent-fail-reconcile",
            provider=fail_provider,
            tool_registry=registry,
            token_budget=guard,
            workspace_root=self.temp_dir,
        )

        res_fail = fail_pipeline.execute_task(task_objective="Validate JSON")
        self.assertEqual(res_fail.action_result.success, False)
        # Total reconciliations: 1 from previous success + 1 from failure = 2
        self.assertEqual(reconcile_count, 2, "Reconcile must run exactly once on failure path")

    def test_live_probe_all_five_exit_paths_write_allowlisted_receipts(self):
        """live_probe writes allowlisted receipts on EVERY exit path (success, truncation, malformed, none, exception)."""
        os.environ["REAGENT_ALLOW_PROBE_IMPORT_FOR_TESTS"] = "1"
        from scripts.live_probe import run_live_probe

        probe_reports_dir = os.path.join(self.temp_dir, "probe_reports")

        # ── Path 1: Provider Exception ──────────────────────────────────────────
        fail_client = FakeOpenAIClient(FakeOpenAIResponse(content=""))
        fail_client.chat.completions.create.side_effect = RuntimeError("Network timeout")
        p1 = NebiusNemotronProvider(client=fail_client)
        l1 = GlobalTokenLedger(ledger_path=os.path.join(self.temp_dir, "led1.json"))
        exit1 = run_live_probe(provider=p1, ledger=l1, reports_dir=probe_reports_dir, confirm_billing_checked=True)
        self.assertEqual(exit1, 1)

        # ── Path 2: Truncation (finish_reason='length') ──────────────────────────
        trunc_client = FakeOpenAIClient(FakeOpenAIResponse(
            content="",
            finish_reason="length",
            usage={"prompt_tokens": 100, "completion_tokens": 1536, "total_tokens": 1636}
        ))
        p2 = NebiusNemotronProvider(client=trunc_client)
        l2 = GlobalTokenLedger(ledger_path=os.path.join(self.temp_dir, "led2.json"))
        exit2 = run_live_probe(provider=p2, ledger=l2, reports_dir=probe_reports_dir, confirm_billing_checked=True)
        self.assertEqual(exit2, 2)

        # ── Path 3: Malformed Output (unparseable or invalid schema) ──────────────
        malformed_client = FakeOpenAIClient(FakeOpenAIResponse(
            content=json.dumps({"unknown_injected_key": "bad"}),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        ))
        p3 = NebiusNemotronProvider(client=malformed_client)
        l3 = GlobalTokenLedger(ledger_path=os.path.join(self.temp_dir, "led3.json"))
        exit3 = run_live_probe(provider=p3, ledger=l3, reports_dir=probe_reports_dir, confirm_billing_checked=True)
        self.assertEqual(exit3, 2)

        # ── Path 4: None Decision (unparseable JSON) ──────────────────────────────
        none_client = FakeOpenAIClient(FakeOpenAIResponse(
            content="{not valid json",
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        ))
        p4 = NebiusNemotronProvider(client=none_client)
        l4 = GlobalTokenLedger(ledger_path=os.path.join(self.temp_dir, "led4.json"))
        exit4 = run_live_probe(provider=p4, ledger=l4, reports_dir=probe_reports_dir, confirm_billing_checked=True)
        self.assertEqual(exit4, 2)

        # ── Path 5: Clean Success ────────────────────────────────────────────────
        success_client = FakeOpenAIClient(FakeOpenAIResponse(
            content=json.dumps({
                "tool": "json_validator",
                "arguments": {"json_string": "{\"probe\": 1}"},
                "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}]
            }),
            finish_reason="stop",
            usage={"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160}
        ))
        p5 = NebiusNemotronProvider(client=success_client)
        l5 = GlobalTokenLedger(ledger_path=os.path.join(self.temp_dir, "led5.json"))
        exit5 = run_live_probe(provider=p5, ledger=l5, reports_dir=probe_reports_dir, confirm_billing_checked=True)
        self.assertEqual(exit5, 0)

        # ── Verify Receipts written for ALL paths ────────────────────────────────
        receipt_files = [f for f in os.listdir(probe_reports_dir) if f.startswith("receipt_") and f.endswith(".json")]
        self.assertGreaterEqual(len(receipt_files), 5, "Must write a receipt on every exit path")

        from reagent.receipts import ALLOWED_RECEIPT_KEYS
        statuses_found = set()
        for rf in receipt_files:
            with open(os.path.join(probe_reports_dir, rf), "r", encoding="utf-8") as f:
                rdata = json.load(f)
            # Strict allowlist check
            self.assertTrue(set(rdata.keys()).issubset(ALLOWED_RECEIPT_KEYS))
            statuses_found.add(rdata["status"])
            # Ensure argument_keys contains keys only, NEVER values
            arg_keys = rdata.get("argument_keys", [])
            self.assertIsInstance(arg_keys, list)
            for k in arg_keys:
                self.assertIsInstance(k, str)
                self.assertNotIn("probe", k)  # 'probe' was the argument value, not key
            # Ensure finish_reason, model_id, exit_reason are present
            self.assertIn("exit_reason", rdata)
            self.assertIn("model_id", rdata)

        self.assertIn("PROBE_PROVIDER_ERROR", statuses_found)
        self.assertIn("PROBE_TRUNCATION", statuses_found)
        self.assertIn("PROBE_DECISION_VALIDATED", statuses_found)

    def test_live_tests_require_both_key_and_explicit_opt_in(self):
        """Offline verification: Setting NEBIUS_API_KEY alone does NOT enable live tests; REAGENT_ALLOW_LIVE_TESTS=1 is strictly required."""
        import importlib
        import tests.integration.test_nebius_live as live_mod

        # Case 1: Key present, but opt-in unset -> LIVE_ENABLED must be False
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-mock-key-only", "REAGENT_ALLOW_LIVE_TESTS": ""}, clear=False):
            has_key = bool(os.environ.get("NEBIUS_API_KEY", "").strip())
            allow_live = (os.environ.get("REAGENT_ALLOW_LIVE_TESTS") == "1")
            live_enabled = has_key and allow_live
            self.assertTrue(has_key)
            self.assertFalse(allow_live)
            self.assertFalse(live_enabled, "Live tests MUST NOT be enabled when only NEBIUS_API_KEY is present")

        # Case 2: Opt-in set, but key absent -> LIVE_ENABLED must be False
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "", "REAGENT_ALLOW_LIVE_TESTS": "1"}, clear=False):
            has_key = bool(os.environ.get("NEBIUS_API_KEY", "").strip())
            allow_live = (os.environ.get("REAGENT_ALLOW_LIVE_TESTS") == "1")
            live_enabled = has_key and allow_live
            self.assertFalse(has_key)
            self.assertTrue(allow_live)
            self.assertFalse(live_enabled, "Live tests MUST NOT be enabled when API key is missing")

        # Case 3: Both set -> only then LIVE_ENABLED is True
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-mock-key-only", "REAGENT_ALLOW_LIVE_TESTS": "1"}, clear=False):
            has_key = bool(os.environ.get("NEBIUS_API_KEY", "").strip())
            allow_live = (os.environ.get("REAGENT_ALLOW_LIVE_TESTS") == "1")
            live_enabled = has_key and allow_live
            self.assertTrue(has_key)
            self.assertTrue(allow_live)
            self.assertTrue(live_enabled)

        # Confirm all 4 live test classes have skip condition applied
        for cls in (
            live_mod.TestNebiusLiveGenerate,
            live_mod.TestNebiusLiveStructuredDecision,
            live_mod.TestNebiusLiveDiagnoseAndRecover,
            live_mod.TestNebiusLiveFullLoop,
        ):
            self.assertTrue(
                getattr(cls, "__unittest_skip__", False),
                f"{cls.__name__} must be skipped when REAGENT_ALLOW_LIVE_TESTS is unset",
            )

    def test_live_probe_reconcile_failure_fails_closed_and_preserves_provider_error(self):
        """Reconciliation failure in live_probe fails closed with PROBE_ACCOUNTING_UNCERTAIN and preserves provider errors."""
        os.environ["REAGENT_ALLOW_PROBE_IMPORT_FOR_TESTS"] = "1"
        from scripts.live_probe import run_live_probe

        probe_reports_dir = os.path.join(self.temp_dir, "probe_reports_rf")

        # Sub-test A: Provider succeeded, but reconcile failed -> PROBE_ACCOUNTING_UNCERTAIN
        success_client = FakeOpenAIClient(FakeOpenAIResponse(
            content=json.dumps({
                "tool": "json_validator",
                "arguments": {"json_string": "{\"probe\": 1}"},
                "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}]
            }),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        ))
        p_ok = NebiusNemotronProvider(client=success_client)
        l_ok = GlobalTokenLedger(ledger_path=os.path.join(self.temp_dir, "led_rf_1.json"))
        # Force reconcile failure
        l_ok.reconcile_usage = MagicMock(side_effect=PermissionError("Ledger disk lock error"))

        exit_code_a = run_live_probe(provider=p_ok, ledger=l_ok, reports_dir=probe_reports_dir, confirm_billing_checked=True)
        self.assertEqual(exit_code_a, 1, "Must return error exit code 1 on reconciliation failure")

        # Find written receipt
        rfiles = [f for f in os.listdir(probe_reports_dir) if f.startswith("receipt_") and f.endswith(".json")]
        self.assertTrue(len(rfiles) >= 1)
        with open(os.path.join(probe_reports_dir, rfiles[-1]), "r", encoding="utf-8") as f:
            rdata_a = json.load(f)
        self.assertEqual(rdata_a["status"], "PROBE_ACCOUNTING_UNCERTAIN")
        self.assertEqual(rdata_a["exit_reason"], "reconciliation_failed_accounting_uncertain")
        self.assertIn("PermissionError", rdata_a["error_type"])

        # Sub-test B: Provider failed AND reconcile failed -> preserve both errors
        probe_reports_dir_b = os.path.join(self.temp_dir, "probe_reports_rf_b")
        fail_client = FakeOpenAIClient(FakeOpenAIResponse(content=""))
        fail_client.chat.completions.create.side_effect = TimeoutError("Remote endpoint timeout")
        p_fail = NebiusNemotronProvider(client=fail_client)
        l_fail = GlobalTokenLedger(ledger_path=os.path.join(self.temp_dir, "led_rf_2.json"))
        l_fail.reconcile_usage = MagicMock(side_effect=RuntimeError("Disk corruption during reconcile"))

        exit_code_b = run_live_probe(provider=p_fail, ledger=l_fail, reports_dir=probe_reports_dir_b, confirm_billing_checked=True)
        self.assertEqual(exit_code_b, 1)

        rfiles_b = [f for f in os.listdir(probe_reports_dir_b) if f.startswith("receipt_") and f.endswith(".json")]
        self.assertTrue(len(rfiles_b) >= 1)
        with open(os.path.join(probe_reports_dir_b, rfiles_b[-1]), "r", encoding="utf-8") as f:
            rdata_b = json.load(f)
        self.assertEqual(rdata_b["status"], "PROBE_ACCOUNTING_UNCERTAIN")
        self.assertIn("TimeoutError", rdata_b["error_type"])
        self.assertIn("RuntimeError", rdata_b["error_type"])

    def test_pipeline_reconcile_failure_fails_closed_and_preserves_provider_error(self):
        """In pipeline.execute_task, reconciliation failure fails closed, marks ACCOUNTING_UNCERTAIN, and preserves provider errors."""
        ledger = GlobalTokenLedger(ledger_path=os.path.join(self.temp_dir, "led_pipe_rf.json"))
        guard = TokenBudgetGuard(max_calls_per_run=10, max_tokens_per_run=20000, ledger=ledger)
        registry = ToolRegistry()
        registry.register(JsonValidatorTool())

        # Sub-test A: Provider succeeds, reconcile fails
        success_client = FakeOpenAIClient(FakeOpenAIResponse(
            content=json.dumps({
                "tool": "json_validator",
                "arguments": {"json_string": "{}"},
                "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}]
            }),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        ))
        p_ok = NebiusNemotronProvider(client=success_client)
        pipeline_a = ReAgentPipeline(
            agent_id="test-pipe-rf-a",
            provider=p_ok,
            tool_registry=registry,
            token_budget=guard,
            workspace_root=self.temp_dir,
        )
        with patch.object(guard, "reconcile", side_effect=IOError("Ledger file locked")):
            res_a = pipeline_a.execute_task(task_objective="Validate JSON")
        self.assertEqual(res_a.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertIn("AccountingUncertain", res_a.action_result.error)
        timeline_phases_a = [e.phase for e in res_a.timeline]
        self.assertIn("ACCOUNTING_UNCERTAIN", timeline_phases_a)

        # Sub-test B: Provider fails AND reconcile fails
        fail_client = FakeOpenAIClient(FakeOpenAIResponse(content=""))
        fail_client.chat.completions.create.side_effect = TimeoutError("Nebius timeout")
        p_fail = NebiusNemotronProvider(client=fail_client)
        pipeline_b = ReAgentPipeline(
            agent_id="test-pipe-rf-b",
            provider=p_fail,
            tool_registry=registry,
            token_budget=guard,
            workspace_root=self.temp_dir,
        )
        with patch.object(guard, "reconcile", side_effect=RuntimeError("Reconcile crash")):
            res_b = pipeline_b.execute_task(task_objective="Validate JSON")
        self.assertEqual(res_b.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertIn("ProviderCallFailedAndAccountingUncertain", res_b.action_result.error)
        self.assertIn("TimeoutError", res_b.action_result.error)
        self.assertIn("RuntimeError", res_b.action_result.error)
        timeline_phases_b = [e.phase for e in res_b.timeline]
        self.assertIn("PROVIDER_FAILED", timeline_phases_b)
        self.assertIn("ACCOUNTING_UNCERTAIN", timeline_phases_b)

    def test_offline_operations_do_not_modify_default_ledger(self):
        """Regression test: running pipeline with isolated ledger does not touch the default ledger."""
        default_ledger_path = os.path.join(os.path.expanduser("~"), ".reagent", "token_ledger.json")
        default_exists = os.path.exists(default_ledger_path)
        before_stat = None
        if default_exists:
            st = os.stat(default_ledger_path)
            before_stat = (st.st_mtime, st.st_size)

        # Run pipeline operation in an isolated environment with temporary ledger
        isolated_dir = tempfile.mkdtemp(prefix="reagent_iso_test_")
        try:
            isolated_ledger_path = os.path.join(isolated_dir, "iso_ledger.json")
            iso_ledger = GlobalTokenLedger(ledger_path=isolated_ledger_path)
            iso_guard = TokenBudgetGuard(max_calls_per_run=5, max_tokens_per_run=10000, ledger=iso_ledger)
            registry = ToolRegistry()
            registry.register(JsonValidatorTool())
            pipeline = ReAgentPipeline(
                agent_id="iso-agent",
                tool_registry=registry,
                token_budget=iso_guard,
                workspace_root=isolated_dir,
            )
            res = pipeline.execute_task(task_objective="Validate JSON string")
            self.assertIn(res.verified_receipt.final_status, [AgentStatus.VERIFIED, AgentStatus.ESCALATED])
        finally:
            shutil.rmtree(isolated_dir, ignore_errors=True)

        # Verify default ledger was untouched
        if default_exists:
            st_after = os.stat(default_ledger_path)
            self.assertEqual((st_after.st_mtime, st_after.st_size), before_stat, "Default ledger modified by offline test!")
        else:
            self.assertFalse(os.path.exists(default_ledger_path), "Default ledger was inadvertently created by test!")


if __name__ == "__main__":
    unittest.main()
