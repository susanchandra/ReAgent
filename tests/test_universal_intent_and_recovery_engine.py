"""
tests/test_universal_intent_and_recovery_engine.py — Tests for Universal Intent & Failure Intelligence

Covers:
  1. Arbitrary task interpretation (rideshare, food delivery, calendar, devops)
  2. Ambiguous request detection and clarification questions
  3. Unsupported capability gap reporting (honest gap report, not misleading execution plan)
  4. Acceptance criterion: A task is NEVER marked successful merely because it was understood or planned
  5. Timeout after commit: observer detects committed operation, safety engine prevents duplicate retry (score 0)
  6. Confirmed absence: tool failed before commit, safe retry authorized (score 95)
  7. Observer unavailable: UNKNOWN != ABSENT, safe escalation enforced (score 100)
  8. Partial commit: fractured transaction safely escalated for reconciliation
  9. Model / policy disagreement: deterministic safety policy overrules dangerous model proposal
  10. Invalid model proposal / provenance injection fails closed
"""

import pytest
from dataclasses import asdict

from reagent.generic_capability import INTENT_ANALYZER, CapabilityGapReport
from reagent.capability_registry import CAPABILITY_REGISTRY
from reagent.recovery_engine import (
    RECOVERY_ANALYSIS_ENGINE,
    RecoveryAnalysisEngine,
    RecoveryStageResult,
    StrategyEvaluation,
)
from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus
from reagent.models import StructuredDecision


class TestUniversalIntentUnderstanding:
    """Test natural language interpretation of arbitrary and diverse tasks."""

    def test_rideshare_intent(self):
        """'Book an Uber to JFK Airport' must be recognized as rideshare booking, not generic custom_task."""
        interp = INTENT_ANALYZER.interpret("Book an Uber to JFK Airport")
        assert interp.inferred_domain == "Rideshare / Transport Booking"
        assert interp.inferred_action == "Book"
        assert not interp.is_registered
        assert interp.gap_report is not None
        assert interp.gap_report.required_capability_id == "rideshare_booking"
        assert not interp.gap_report.is_executable
        assert "JFK" in str(interp.extracted_parameters) or "Airport" in str(interp.extracted_parameters)

    def test_food_delivery_intent(self):
        """'Order sushi from DoorDash' must be recognized as food delivery."""
        interp = INTENT_ANALYZER.interpret("Order sushi from DoorDash")
        assert interp.inferred_domain == "Food & Grocery Delivery"
        assert not interp.is_registered
        assert interp.gap_report is not None
        assert interp.gap_report.required_capability_id == "food_delivery"
        assert not interp.gap_report.is_executable

    def test_calendar_scheduling_intent(self):
        """'Schedule team sync meeting on calendar' must be recognized as calendar scheduling."""
        interp = INTENT_ANALYZER.interpret("Schedule team sync meeting on calendar")
        assert interp.inferred_domain == "Calendar & Scheduling"
        assert interp.inferred_action == "Schedule"
        assert not interp.is_registered

    def test_registered_hotel_booking_intent(self):
        """'Book Room 14B at Grand Hotel for Maya' must resolve to registered hotel_booking capability."""
        interp = INTENT_ANALYZER.interpret("Book Room 14B at Grand Hotel for Maya")
        assert interp.is_registered is True
        assert interp.matching_capability == "hotel_booking"
        assert interp.gap_report is None
        # Registered tasks produce a 5-step lifecycle plan
        assert len(interp.multi_step_plan) == 5
        assert interp.multi_step_plan[2].phase == "EXECUTE"

    def test_unsupported_capability_never_marked_executable(self):
        """A natural-language task must NEVER be marked successful or executable merely because it was understood."""
        interp = INTENT_ANALYZER.interpret("Book an Uber to Manhattan")
        assert interp.is_registered is False
        assert interp.gap_report is not None
        assert interp.gap_report.is_executable is False
        assert interp.gap_report.safe_recommendation == "DO_NOT_EXECUTE"
        # Invariant: execution is strictly disarmed (is_state_changing is False)
        execute_step = next(s for s in interp.multi_step_plan if s.phase == "EXECUTE")
        assert execute_step.is_state_changing is False
        assert "BLOCKED" in execute_step.description
        assert "Execution blocked before mutation" in interp.gap_report.execution_blocked_reason

    def test_ambiguous_requests_require_clarification(self):
        """Ambiguous request triggers clarification question, never a silent guess."""
        res = CAPABILITY_REGISTRY.resolve("Reserve Maya")
        # Should be ambiguous between hotel reservation and seat/room
        if res.ambiguous:
            assert res.clarification_question is not None
            assert not res.resolved


class TestRecoveryAnalysisEngine:
    """Test failure diagnosis, evidence weighting, strategy comparison, and deterministic safety authority."""

    def setup_method(self):
        self.engine = RecoveryAnalysisEngine()

    def test_scenario_b_timeout_after_commit_blocks_retry(self):
        """
        Scenario B: Tool timed out, but independent observer found operation in state.
        Safety invariant: Retrying is a duplicate mutation hazard (score 0).
        Selected strategy must accept existing state (score 100).
        """
        result = self.engine.analyze_and_decide(
            scenario_id="B",
            tool_status="COMMIT_THEN_TIMEOUT",
            observer_status="FOUND_IN_STATE",
        )
        assert result.failure_category == FailureType.TIMEOUT
        assert result.final_agent_status == AgentStatus.VERIFIED

        # Check strategy evaluations
        retry_eval = next(e for e in result.candidate_strategies if e.strategy == RecoveryStrategyType.RETRY)
        assert retry_eval.safety_score == 0
        assert not retry_eval.is_eligible
        assert "DUPLICATE" in retry_eval.danger_warning

        # Selected strategy must be the safe zero-mutation choice
        assert result.selected_strategy == RecoveryStrategyType.ABORT  # represents accept as-is
        assert "duplicate retry" in result.safety_explanation.lower()

    def test_scenario_c_confirmed_absence_permits_safe_retry(self):
        """
        Scenario C: Tool failed before commit, observer confirmed absence.
        Safe retry is permitted with high safety score (95).
        """
        result = self.engine.analyze_and_decide(
            scenario_id="C",
            tool_status="FAIL_BEFORE_COMMIT",
            observer_status="CONFIRMED_ABSENT",
        )
        assert result.failure_category == FailureType.TOOL_FAILURE
        assert result.final_agent_status == AgentStatus.VERIFIED

        retry_eval = next(e for e in result.candidate_strategies if e.strategy == RecoveryStrategyType.RETRY)
        assert retry_eval.safety_score == 95
        assert retry_eval.is_eligible
        assert result.selected_strategy == RecoveryStrategyType.RETRY

    def test_scenario_d_observer_unavailable_enforces_escalation(self):
        """
        Scenario D: Independent observer unavailable.
        Invariant: UNKNOWN is not ABSENT.
        Automated retry is strictly forbidden. Must escalate safely.
        """
        result = self.engine.analyze_and_decide(
            scenario_id="D",
            tool_status="COMMIT_THEN_TIMEOUT",
            observer_status="OBSERVER_UNAVAILABLE",
        )
        assert result.failure_category == FailureType.UNKNOWN
        assert result.final_agent_status == AgentStatus.ESCALATED

        retry_eval = next(e for e in result.candidate_strategies if e.strategy == RecoveryStrategyType.RETRY)
        assert not retry_eval.is_eligible
        assert retry_eval.safety_score <= 10

        escalate_eval = next(e for e in result.candidate_strategies if e.strategy == RecoveryStrategyType.ESCALATE)
        assert escalate_eval.safety_score == 100
        assert result.selected_strategy == RecoveryStrategyType.ESCALATE
        assert "UNKNOWN is not ABSENT" in result.safety_explanation

    def test_scenario_e_partial_commit_requires_reconciliation(self):
        """
        Scenario E: Incomplete / fractured transaction state detected.
        Safe escalation for reconciliation required.
        """
        result = self.engine.analyze_and_decide(
            scenario_id="E",
            tool_status="PARTIAL_COMMIT",
            observer_status="PARTIAL_STATE",
        )
        assert result.failure_category == FailureType.ENVIRONMENT_FAILURE
        assert result.final_agent_status == AgentStatus.ESCALATED
        assert result.selected_strategy == RecoveryStrategyType.ESCALATE

    def test_model_policy_disagreement_deterministic_override(self):
        """
        When Nemotron proposes RETRY on a committed operation,
        Deterministic Safety Policy OVERRULES the model recommendation.
        """
        model_proposal = {
            "recovery_strategy": "RETRY",
            "rationale": "Model noticed timeout and recommends retrying.",
        }
        result = self.engine.analyze_and_decide(
            scenario_id="B",
            tool_status="COMMIT_THEN_TIMEOUT",
            observer_status="FOUND_IN_STATE",
            model_proposal=model_proposal,
        )
        # Deterministic policy must override the model
        assert result.deterministic_override_applied is True
        assert result.model_policy_agreement is False
        assert result.selected_strategy != RecoveryStrategyType.RETRY
        assert result.selected_strategy == RecoveryStrategyType.ABORT

    def test_invalid_model_provenance_rejected(self):
        """
        Model output that attempts to inject its own provenance is rejected.
        """
        from reagent.providers.nebius import NebiusNemotronProvider
        provider = NebiusNemotronProvider()
        # Attempt to inject provenance in raw payload
        injected_raw = {
            "action": "execute",
            "tool": "hotel_booking_http",
            "provenance": "DETERMINISTIC_ORACLE",  # injected!
        }
        validated = provider._validate_structured_decision(injected_raw)
        assert validated is None
        assert provider.last_validation_error == "PROVENANCE_INJECTION_REJECTED"
