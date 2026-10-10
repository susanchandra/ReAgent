"""
ReAgent Milestone 4A — Autonomous Planning Entry Point Offline Test Suite

Verifies:
  1. Happy path autonomous planning via execute_task() with provenance assigned by code.
  2. Schema validation rejection: unallowlisted tool proposed by provider fails closed.
  3. Grounding validator rejection: state-changing tool with status-only predicate rejected.
  4. Grounding validator path sandbox boundary check: out-of-sandbox predicate rejected.
  5. Target binding discrepancy warning between tool argument and predicate target.
  6. TokenBudgetGuard enforces call limits before provider invocation (zero network/model calls).
  7. TokenBudgetGuard tracks token usage and blocks calls when token cap is reached.
  8. Truncated output or None decision from provider fails closed to ESCALATED.
  9. Provenance assigned by code: model-injected provenance field is ignored/overwritten.
  10. Mutation sanity check: monkeypatching verifier to always PASS fails negative tests.
"""

import os
import shutil
import tempfile
import unittest
from typing import Dict, Any, List, Optional
from unittest.mock import patch

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_m4a_test_")
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

from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus
from reagent.models import (
    TaskState, Postcondition, SuccessContract, VerifiedReceipt,
    ObservabilityEvent, StructuredDecision, FailureEvent, FailureDiagnosis,
    RecoveryDecision, ActionResult, Evidence, VerificationResult,
)
from reagent.contract import ContractManager
from reagent.events import (
    ObservabilityTimeline, PHASE_TASK_RECEIVED, PHASE_CONTRACT_FROZEN,
    PHASE_MODEL_DECISION, PHASE_DECISION_VALIDATED, PHASE_VERIFIED_RECEIPT,
)
from reagent.verification.verifier import DeterministicVerifier, AssertionRule
from reagent.verification.evidence import EvidenceCollector
from reagent.contract_validator import ContractGroundingValidator, ValidationResult
from reagent.pipeline import ReAgentPipeline, ExecutionCycleResult
from reagent.tools.registry import ToolRegistry
from reagent.tools.json_tool import JsonValidatorTool
from reagent.tools.file_tool import SandboxedFileTool
from reagent.policy import PolicyGuard, TokenBudgetGuard
from reagent.providers.base import LLMProvider, LLMResponse
from reagent.providers.development import DevelopmentLLMProvider


class FakeStructuredDecisionProvider(LLMProvider):
    """
    Test fixture provider that injects a predetermined StructuredDecision
    to test the pipeline's handling of autonomous planning outputs.
    """

    def __init__(self, predetermined_decision: Optional[StructuredDecision] = None):
        self.predetermined_decision = predetermined_decision
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(model_name="fake-provider", content="mock-content")

    def diagnose_failure(self, failure: FailureEvent, **kwargs) -> FailureDiagnosis:
        return FailureDiagnosis(
            failure_id=failure.failure_id,
            failure_type=FailureType.TOOL_FAILURE,
            root_cause="Mock diagnosis",
            confidence_score=0.9,
            is_agent_fault=False,
            is_transient=False,
            provenance="fixture",
        )

    def select_recovery(self, diagnosis: FailureDiagnosis, available_tools: List[str], **kwargs) -> RecoveryDecision:
        return RecoveryDecision(
            diagnosis_id=diagnosis.diagnosis_id,
            strategy=RecoveryStrategyType.ESCALATE,
            rationale="Escalate mock",
            provenance="fixture",
        )

    def make_structured_decision(
        self,
        task_objective: str,
        available_tools: List[str],
        context_summary: Optional[str] = None,
    ) -> Optional[StructuredDecision]:
        self.call_count += 1
        return self.predetermined_decision


class ExplodingPlanningProvider(LLMProvider):
    """Explodes if make_structured_decision is called — proves budget guard blocks before call."""

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        raise RuntimeError("ExplodingPlanningProvider.generate() was invoked!")

    def diagnose_failure(self, failure: FailureEvent, **kwargs) -> FailureDiagnosis:
        raise RuntimeError("ExplodingPlanningProvider.diagnose_failure() was invoked!")

    def select_recovery(self, diagnosis: FailureDiagnosis, **kwargs) -> RecoveryDecision:
        raise RuntimeError("ExplodingPlanningProvider.select_recovery() was invoked!")

    def make_structured_decision(self, *args, **kwargs) -> Optional[StructuredDecision]:
        raise RuntimeError("ExplodingPlanningProvider.make_structured_decision() was invoked!")


class TestMilestone4APlanning(unittest.TestCase):
    """Offline test suite for Milestone 4A autonomous planning entry point."""

    def setUp(self):
        ledger_file = os.environ.get("REAGENT_LEDGER_PATH")
        if ledger_file and os.path.exists(ledger_file):
            try:
                os.remove(ledger_file)
            except OSError:
                pass
        self.registry = ToolRegistry()
        self.registry.register(JsonValidatorTool())

    # ── 1. Happy path autonomous planning via execute_task() ─────────────────
    def test_happy_path_execute_task(self):
        """
        Provider proposes json_validator with valid arguments and STATUS_SUCCESS.
        execute_task runs end-to-end, assigns fixture provenance by code, and produces VERIFIED receipt.
        """
        decision = StructuredDecision(
            tool="json_validator",
            arguments={"json_payload": '{"status": "ok", "value": 42}'},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS", description="Validation succeeds"),
            ],
        )
        provider = FakeStructuredDecisionProvider(predetermined_decision=decision)
        pipeline = ReAgentPipeline(provider=provider, tool_registry=self.registry)

        result = pipeline.execute_task("Validate JSON payload")

        self.assertEqual(result.verified_receipt.final_status, AgentStatus.VERIFIED)
        self.assertEqual(result.tool, "json_validator")
        self.assertEqual(result.recovery_count, 0)
        # Provenance must be assigned by code as 'fixture' because provider is not real Nebius
        self.assertEqual(result.frozen_contract.provenance, "fixture")
        self.assertTrue(result.verification_result.verified)

    # ── 2. Schema validation rejection: unallowlisted tool ───────────────────
    def test_unallowlisted_tool_rejected(self):
        """
        Provider proposes an unallowlisted tool ('arbitrary_shell_executor').
        Pipeline must reject before execution, emit DecisionValidationFailed, and escalate.
        """
        decision = StructuredDecision(
            tool="arbitrary_shell_executor",
            arguments={"command": "whoami"},
            proposed_postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
        )
        provider = FakeStructuredDecisionProvider(predetermined_decision=decision)
        pipeline = ReAgentPipeline(provider=provider, tool_registry=self.registry)

        result = pipeline.execute_task("Run arbitrary shell command")

        self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertIn("DecisionValidationFailed", result.action_result.error)
        self.assertFalse(result.verification_result.verified)

    # ── 3. Grounding validator: state-changing tool with status-only pc ─────
    def test_grounding_rejects_state_changing_tool_with_only_status_success(self):
        """
        Provider proposes sandboxed_file (state-changing) but provides ONLY STATUS_SUCCESS.
        ContractGroundingValidator must reject: state-changing tools require >=1 independent observable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            file_tool = SandboxedFileTool(sandbox_root=tmpdir)
            self.registry.register(file_tool)

            decision = StructuredDecision(
                tool="sandboxed_file",
                arguments={"action": "write", "relative_path": "report.txt", "content": "data"},
                proposed_postconditions=[
                    Postcondition(predicate="STATUS_SUCCESS", description="Tool claims success only"),
                ],
            )
            provider = FakeStructuredDecisionProvider(predetermined_decision=decision)
            pipeline = ReAgentPipeline(provider=provider, tool_registry=self.registry)

            result = pipeline.execute_task("Write report.txt to disk")

            self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
            self.assertIn("ContractGroundingFailed", result.action_result.error)
            self.assertIn("state-changing", result.action_result.error)

    # ── 4. Grounding validator: sandbox boundary check on predicate path ────
    def test_grounding_rejects_out_of_sandbox_predicate_path(self):
        """
        Provider proposes a postcondition path escaping the sandbox (e.g. '../../outside.txt').
        ContractGroundingValidator detects sandbox boundary failure and fails closed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            file_tool = SandboxedFileTool(sandbox_root=tmpdir)
            self.registry.register(file_tool)

            decision = StructuredDecision(
                tool="sandboxed_file",
                arguments={"action": "write", "relative_path": "report.txt", "content": "data"},
                proposed_postconditions=[
                    Postcondition(predicate="STATUS_SUCCESS"),
                    Postcondition(predicate="FILE_EXISTS", target="../../outside.txt"),
                ],
            )
            provider = FakeStructuredDecisionProvider(predetermined_decision=decision)
            pipeline = ReAgentPipeline(provider=provider, tool_registry=self.registry)

            result = pipeline.execute_task("Write report with path traversal predicate")

            self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
            self.assertIn("ContractGroundingFailed", result.action_result.error)
            self.assertIn("sandbox boundary", result.action_result.error)

    # ── 5. Target binding discrepancy warning ───────────────────────────────
    def test_target_binding_discrepancy_reported(self):
        """
        Provider proposes writing to 'out1.txt' but postcondition checks 'out2.txt'.
        ContractGroundingValidator reports a target binding discrepancy warning.
        """
        validator = ContractGroundingValidator(workspace_root="D:/Jarvis/workspace/sandbox")
        tool = SandboxedFileTool()
        decision = StructuredDecision(
            tool="sandboxed_file",
            arguments={"action": "write", "relative_path": "out1.txt", "content": "data"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS"),
                Postcondition(predicate="FILE_EXISTS", target="out2.txt"),
            ],
        )

        res = validator.validate(decision, tool, "Write data to out1.txt")

        # Discrepancy generates a blocking error about mismatched targets per Milestone 4B
        self.assertFalse(res.is_valid)
        self.assertTrue(any("TargetBindingMismatch" in e for e in res.errors))

    # ── 6. TokenBudgetGuard enforces call limits before provider call ───────
    def test_token_budget_guard_blocks_call_when_exhausted(self):
        """
        When TokenBudgetGuard has zero remaining calls, execute_task() halts
        BEFORE calling provider. ExplodingPlanningProvider proves zero calls occur.
        """
        budget = TokenBudgetGuard(max_calls_per_run=1)
        # Consume the budget
        budget.record_usage({"total_tokens": 50})
        self.assertFalse(budget.allow_call())

        # Exploding provider will raise if called
        exploding_provider = ExplodingPlanningProvider()
        pipeline = ReAgentPipeline(
            provider=exploding_provider,
            tool_registry=self.registry,
            token_budget=budget,
        )

        result = pipeline.execute_task("Any task objective")

        self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertIn("TokenBudgetExceeded", result.action_result.error)

    # ── 7. TokenBudgetGuard tracks token usage and limits ───────────────────
    def test_token_budget_tracks_tokens_and_enforces_cap(self):
        """
        TokenBudgetGuard accurately tallies tokens and rejects calls once
        the cumulative token ceiling is reached.
        """
        budget = TokenBudgetGuard(max_calls_per_run=10, max_tokens_per_run=1000)
        self.assertTrue(budget.allow_call())

        # Call 1: 600 tokens -> OK
        ok = budget.record_usage({"total_tokens": 600})
        self.assertTrue(ok)
        self.assertEqual(budget.tokens_used, 600)
        self.assertTrue(budget.allow_call())

        # Call 2: 500 tokens -> Total 1100 exceeds 1000 cap
        ok = budget.record_usage({"total_tokens": 500})
        self.assertFalse(ok)
        self.assertEqual(budget.tokens_used, 1100)
        # Next call is blocked
        self.assertFalse(budget.allow_call())

    # ── 8. Truncated output or None decision fails closed ────────────────────
    def test_provider_returning_none_fails_closed(self):
        """
        If provider returns None (unparseable / model output failure),
        execute_task() fails closed to ESCALATED without crashing.
        """
        provider = FakeStructuredDecisionProvider(predetermined_decision=None)
        pipeline = ReAgentPipeline(provider=provider, tool_registry=self.registry)

        result = pipeline.execute_task("Objective with unparseable model response")

        self.assertEqual(result.verified_receipt.final_status, AgentStatus.ESCALATED)
        self.assertIn("ProviderDecisionNone", result.action_result.error)

    # ── 9. Provenance assigned by code, not model ────────────────────────────
    def test_provenance_assigned_by_code_overrides_model_payload(self):
        """
        If model JSON maliciously or mistakenly claims 'provenance': 'model' or 'trusted',
        code overrides it to 'fixture' when running with a non-live provider.
        """
        decision = StructuredDecision(
            tool="json_validator",
            arguments={"json_payload": '{"key": "value"}'},
            proposed_postconditions=[Postcondition(predicate="STATUS_SUCCESS")],
            provenance="model",  # Model claims model provenance
        )
        provider = FakeStructuredDecisionProvider(predetermined_decision=decision)
        pipeline = ReAgentPipeline(provider=provider, tool_registry=self.registry)

        result = pipeline.execute_task("Validate JSON with model-injected provenance")

        # Must be overridden to 'fixture' by code
        self.assertEqual(result.frozen_contract.provenance, "fixture")

    # ── 10. Mutation experiment: monkeypatch verifier to verify negative test strength ─
    def test_mutation_experiment_proves_verifier_is_genuine_gate(self):
        """
        Proves that verification is not a no-op pass-through:
        When verifier is mutated to always PASS, a false-success scenario that normally
        FAILS would erroneously PASS. Restoring the real verifier restores proper FAILS.
        """
        verifier = DeterministicVerifier()
        bad_evidence = Evidence(source="test", facts={"success": False, "error": "Disk full"})
        rule = [AssertionRule(assertion_type="STATUS_SUCCESS")]

        # Under real verifier: MUST FAIL
        real_result = verifier.verify("real-test", bad_evidence, rule)
        self.assertFalse(real_result.verified, "Real verifier must fail when success=False")

        # Mutate verifier to always return verified=True (simulating broken/fake verifier)
        with patch.object(
            DeterministicVerifier,
            "verify",
            return_value=VerificationResult(
                recovery_decision_id="mutated",
                verified=True,
                evidence_summary="Fake mutated pass",
            ),
        ):
            mutated_result = verifier.verify("mutated-test", bad_evidence, rule)
            # The mutated verifier erroneously reports True
            self.assertTrue(mutated_result.verified)

        # After patch context exits, real verifier behavior is fully restored
        restored_result = verifier.verify("restored-test", bad_evidence, rule)
        self.assertFalse(restored_result.verified, "Restored verifier must fail on bad evidence")


if __name__ == "__main__":
    unittest.main()
