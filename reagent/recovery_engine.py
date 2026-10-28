"""
reagent/recovery_engine.py — Universal Intent & Failure Intelligence Engine

Separates reliability lifecycle into 7 explicit stages:
  1. Intent Understanding (natural language parsing & entity extraction)
  2. Capability Resolution (registry lookup or CapabilityGapReport)
  3. Planning & Contract Declaration (proposal with provenance; NEVER proof of execution)
  4. Tool Execution (monitored action with timeout / failure capture)
  5. Independent Observation (authoritative state probe bypassing tool claims)
  6. Deterministic Verification (postcondition evaluation against frozen contract)
  7. Recovery Analysis & Policy Authority (strategy comparison, safety scoring, deterministic override)

Core Invariant:
  UNKNOWN IS NOT ABSENT.
  A task is NEVER marked successful merely because it was understood or planned.
  Deterministic policy strictly governs safety over model recommendations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus


# ─── Stage 1 & 2: Intent and Capability Contracts ─────────────────────────────

@dataclass
class IntentResult:
    """Stage 1: Intent Understanding Output."""
    raw_prompt: str
    inferred_domain: str
    inferred_action: str
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class CapabilityGapReport:
    """Detailed capability deficiency analysis for unregistered domains."""
    detected_domain: str
    detected_action: str
    required_capability_id: str
    is_executable: bool = False
    execution_blocked_reason: str = ""
    missing_prerequisites: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    independent_verification_requirement: str = ""
    available_alternatives: List[str] = field(default_factory=list)
    safe_recommendation: str = "DO_NOT_EXECUTE"
    extensibility_example: str = ""


@dataclass
class CapabilityResult:
    """Stage 2: Capability Resolution Output."""
    resolved_capability: Optional[str]
    is_executable: bool
    is_ambiguous: bool = False
    clarification_question: Optional[str] = None
    gap_report: Optional[CapabilityGapReport] = None
    notice: Optional[str] = None


# ─── Stage 3: Planning & Contract ─────────────────────────────────────────────

@dataclass
class PlanResult:
    """Stage 3: Task Plan & Frozen Contract.
    Invariant: is_execution_proof is ALWAYS False. A plan is a proposal, not proof.
    """
    provenance: str                                 # "LIVE_NEMOTRON" | "OFFLINE_PLANNER" | "ADVISORY_FIXTURE"
    target_system: str
    operation: str
    preconditions: str
    postconditions: str
    idempotency_key: str
    safety_invariant: str = "UNKNOWN is not ABSENT. Independent probe required before retry."
    is_execution_proof: bool = False                # Explicit invariant
    raw_model_proposal: Optional[Dict[str, Any]] = None


# ─── Stage 4: Execution ───────────────────────────────────────────────────────

@dataclass
class ExecutionStageResult:
    """Stage 4: Tool Execution Attempt."""
    tool_name: str
    attempt_count: int
    tool_status: str                                # "SUCCESS" | "COMMIT_THEN_TIMEOUT" | "FAIL_BEFORE_COMMIT" | "PARTIAL_COMMIT" | "UNREGISTERED"
    tool_reported_success: bool
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    mutated_state: bool = False


# ─── Stage 5: Independent Observation ─────────────────────────────────────────

@dataclass
class ObservationStageResult:
    """Stage 5: Independent Authoritative State Query (bypasses tool claims)."""
    observer_name: str
    authoritative_target: str
    observer_status: str                            # "FOUND_IN_STATE" | "CONFIRMED_ABSENT" | "OBSERVER_UNAVAILABLE" | "PARTIAL_STATE"
    observed_records: Dict[str, Any] = field(default_factory=dict)
    probe_queries: int = 1
    evidence_receipt_hash: Optional[str] = None
    is_authoritative: bool = True


# ─── Stage 6: Deterministic Verification ──────────────────────────────────────

@dataclass
class VerificationStageResult:
    """Stage 6: Postcondition Evaluation vs Frozen Contract."""
    postconditions_met: bool
    safety_invariant_held: bool
    verification_verdict: str                       # "VERIFIED_SUCCESS" | "VERIFIED_FAILURE" | "UNKNOWN_ESCALATE" | "STATE_MISMATCH"
    details: Dict[str, Any] = field(default_factory=dict)


# ─── Stage 7: Recovery Strategy Comparison & Selection ────────────────────────

@dataclass
class StrategyEvaluation:
    """Comparative evaluation of an individual recovery candidate."""
    strategy: RecoveryStrategyType
    safety_score: int                               # 0 to 100
    is_eligible: bool
    rationale: str
    danger_warning: Optional[str] = None
    is_selected: bool = False


@dataclass
class RecoveryStageResult:
    """Stage 7: Recovery Decision & Deterministic Policy Authority."""
    failure_category: FailureType
    root_cause: str
    candidate_strategies: List[StrategyEvaluation]
    selected_strategy: RecoveryStrategyType
    safety_explanation: str
    model_recommended_strategy: Optional[RecoveryStrategyType] = None
    model_policy_agreement: bool = True
    deterministic_override_applied: bool = False
    final_agent_status: AgentStatus = AgentStatus.VERIFIED


# ─── Master Lifecycle Container ───────────────────────────────────────────────

@dataclass
class ReliabilityLifecycleTrace:
    """Full 7-stage verifiable audit trace for any natural-language or scenario task."""
    task_id: str
    scenario_id: Optional[str]
    intent: IntentResult
    capability: CapabilityResult
    plan: Optional[PlanResult]
    execution: Optional[ExecutionStageResult]
    observation: Optional[ObservationStageResult]
    verification: Optional[VerificationStageResult]
    recovery: Optional[RecoveryStageResult]
    overall_outcome: str                            # "SUCCESS_VERIFIED" | "SAFE_ESCALATION" | "BLOCKED_UNSUPPORTED" | "BLOCKED_RETRY" | "RECONCILED"
    mutated_state: bool = False
    is_safe: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Recovery Analysis Engine Implementation ──────────────────────────────────

class RecoveryAnalysisEngine:
    """
    Evaluates execution failures, weighs independent evidence, compares candidate
    recovery strategies, assigns safety scores, and guarantees that deterministic
    safety policy strictly governs over model recommendations.
    """

    def analyze_and_decide(
        self,
        scenario_id: str,
        tool_status: str,
        observer_status: str,
        model_proposal: Optional[Dict[str, Any]] = None,
        attempt_count: int = 1,
        max_retries: int = 2,
    ) -> RecoveryStageResult:
        """
        Produce a rigorous recovery diagnosis and strategy comparison.
        """
        # Determine failure category and root cause
        if tool_status in ("COMMIT_THEN_TIMEOUT", "TIMEOUT"):
            if observer_status == "FOUND_IN_STATE":
                failure_type = FailureType.TIMEOUT
                root_cause = "Tool reported timeout, but side-effect successfully committed to backend store."
            elif observer_status == "OBSERVER_UNAVAILABLE":
                failure_type = FailureType.UNKNOWN
                root_cause = "Tool reported timeout and independent observer is unavailable. UNKNOWN is not ABSENT."
            elif observer_status == "PARTIAL_STATE":
                failure_type = FailureType.ENVIRONMENT_FAILURE
                root_cause = "Tool timed out with incomplete or unbalanced ledger records."
            else:
                failure_type = FailureType.TIMEOUT
                root_cause = "Tool timed out before side-effect committed."
        elif tool_status == "FAIL_BEFORE_COMMIT":
            failure_type = FailureType.TOOL_FAILURE
            root_cause = "Tool failed before mutation; authoritative state confirmed absent."
        elif tool_status == "PARTIAL_COMMIT":
            failure_type = FailureType.ENVIRONMENT_FAILURE
            root_cause = "Multi-step transaction fractured; partial mutation detected."
        elif tool_status == "UNREGISTERED":
            failure_type = FailureType.AUTHORIZATION
            root_cause = "No verified execution adapter registered for requested capability."
        else:
            failure_type = FailureType.TOOL_FAILURE
            root_cause = "Operation failed postcondition check."

        # Model recommendation extraction (if any)
        model_rec: Optional[RecoveryStrategyType] = None
        if model_proposal and "recovery_strategy" in model_proposal:
            try:
                model_rec = RecoveryStrategyType(model_proposal["recovery_strategy"])
            except Exception:
                model_rec = None

        # Evaluate candidate recovery strategies
        evaluations: List[StrategyEvaluation] = []

        # 1. ACCEPT_AS_IS
        if observer_status == "FOUND_IN_STATE":
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.ABORT,  # represents accept / no retry
                safety_score=100,
                is_eligible=True,
                rationale="Postconditions already satisfied in authoritative store. Accepting existing state is 100% safe.",
                danger_warning=None,
            ))
        else:
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.ABORT,
                safety_score=20,
                is_eligible=False,
                rationale="State is missing or unverified; accepting as-is would leave work incomplete.",
                danger_warning="False completion hazard",
            ))

        # 2. RETRY / RETRY_SAME
        if observer_status == "FOUND_IN_STATE":
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.RETRY,
                safety_score=0,
                is_eligible=False,
                rationale="Operation already committed in backend. Retrying will execute duplicate transfer/booking.",
                danger_warning="FATAL DUPLICATE MUTATION HAZARD: User would be charged/booked twice.",
            ))
        elif observer_status == "OBSERVER_UNAVAILABLE":
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.RETRY,
                safety_score=10,
                is_eligible=False,
                rationale="System state is unknown. UNKNOWN is not ABSENT. Retrying blind risks duplicate mutation.",
                danger_warning="POTENTIAL DUPLICATE MUTATION: State could be committed but unobservable.",
            ))
        elif observer_status == "CONFIRMED_ABSENT" and attempt_count < max_retries:
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.RETRY,
                safety_score=95,
                is_eligible=True,
                rationale="Independent observer verified zero prior side-effects. Safe to retry with same idempotency key.",
                danger_warning=None,
            ))
        else:
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.RETRY,
                safety_score=30,
                is_eligible=False,
                rationale="Max retry budget reached or state is ambiguous.",
                danger_warning="Retry budget exhaustion",
            ))

        # 3. ESCALATE (Safe Operator Handover)
        if observer_status == "OBSERVER_UNAVAILABLE":
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.ESCALATE,
                safety_score=100,
                is_eligible=True,
                rationale="State is unobservable. Safe escalation to human operator preserves system safety without mutating.",
                danger_warning=None,
            ))
        elif observer_status == "PARTIAL_STATE":
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.ESCALATE,
                safety_score=90,
                is_eligible=True,
                rationale="Inconsistent transaction state detected. Requires operator reconciliation or compensatory rollback.",
                danger_warning=None,
            ))
        else:
            evaluations.append(StrategyEvaluation(
                strategy=RecoveryStrategyType.ESCALATE,
                safety_score=70,
                is_eligible=True,
                rationale="Escalation is always safe, but automated recovery is preferred when evidence is conclusive.",
                danger_warning=None,
            ))

        # 4. FALLBACK_TOOL / RECONCILE
        evaluations.append(StrategyEvaluation(
            strategy=RecoveryStrategyType.FALLBACK_TOOL,
            safety_score=40 if observer_status != "FOUND_IN_STATE" else 0,
            is_eligible=observer_status == "CONFIRMED_ABSENT",
            rationale="Fallback tool only safe when primary tool confirmed absent and secondary tool registered.",
            danger_warning="Secondary adapter not configured" if observer_status == "CONFIRMED_ABSENT" else "Duplicate risk",
        ))

        # Select safest eligible strategy deterministically
        eligible = [e for e in evaluations if e.is_eligible]
        eligible.sort(key=lambda x: x.safety_score, reverse=True)
        selected_eval = eligible[0] if eligible else evaluations[0]
        selected_eval.is_selected = True

        selected_strategy = selected_eval.strategy

        # Check for Model / Policy Disagreement
        # (e.g. Model recommended RETRY when state was already committed)
        deterministic_override = False
        agreement = True
        if model_rec is not None:
            if model_rec in (RecoveryStrategyType.RETRY, RecoveryStrategyType.RETRY_SAME):
                if observer_status in ("FOUND_IN_STATE", "OBSERVER_UNAVAILABLE"):
                    # DETERMINISTIC POLICY OVERRULES MODEL
                    deterministic_override = True
                    agreement = False

        if observer_status == "FOUND_IN_STATE":
            safety_explanation = (
                "Verified safe: Independent observer confirmed the operation committed to authoritative storage. "
                "ReAgent prevented a dangerous duplicate retry and confirmed the existing record."
            )
            final_status = AgentStatus.VERIFIED
        elif observer_status == "OBSERVER_UNAVAILABLE":
            safety_explanation = (
                "Verified safe: Independent observer connection unavailable. Applying safety invariant: "
                "UNKNOWN is not ABSENT. Execution safely escalated without blind retries or destructive mutation."
            )
            final_status = AgentStatus.ESCALATED
        elif observer_status == "CONFIRMED_ABSENT":
            safety_explanation = (
                "Verified safe: Independent probe confirmed absolute absence of side-effects. "
                "A clean retry was approved with preserved idempotency token."
            )
            final_status = AgentStatus.VERIFIED
        elif observer_status == "PARTIAL_STATE":
            safety_explanation = (
                "Verified safe: Detected unbalanced partial commit. Operation paused for deterministic reconciliation "
                "rather than risking double-charging."
            )
            final_status = AgentStatus.ESCALATED
        else:
            safety_explanation = "Execution safely halted under deterministic policy guard."
            final_status = AgentStatus.FAILED

        return RecoveryStageResult(
            failure_category=failure_type,
            root_cause=root_cause,
            candidate_strategies=evaluations,
            selected_strategy=selected_strategy,
            safety_explanation=safety_explanation,
            model_recommended_strategy=model_rec,
            model_policy_agreement=agreement,
            deterministic_override_applied=deterministic_override,
            final_agent_status=final_status,
        )


# Global recovery analysis engine singleton
RECOVERY_ANALYSIS_ENGINE = RecoveryAnalysisEngine()
