"""
ReAgent -- AI-Agent Reliability Layer
NVIDIA x Nebius Global AI Hackathon 2026

Core Loop:
TASK -> CONTEXT SELECTOR -> NEMOTRON / LLM -> STRUCTURED DECISION ->
POLICY GUARD -> FREEZE SUCCESS CONTRACT -> ALLOWLISTED TOOL ->
REAL ACTION -> INDEPENDENT OBSERVATION -> DETERMINISTIC VERIFIER ->
PASS: VERIFIED RECEIPT / FAIL: FAILURE ATTRIBUTION -> RECOVERY -> REVERIFY
"""

__version__ = "0.3.0"

from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus
from reagent.models import (
    AgentState,
    TaskState,
    ExecutionEvent,
    FailureEvent,
    FailureDiagnosis,
    RecoveryDecision,
    PolicyDecision,
    ActionResult,
    Evidence,
    VerificationResult,
    ReliabilityState,
    Postcondition,
    SuccessContract,
    VerifiedReceipt,
    ObservabilityEvent,
    StructuredDecision,
)
from reagent.pipeline import ReAgentPipeline, ExecutionCycleResult, ReliabilityCycleResult
from reagent.policy import PolicyGuard
from reagent.contract import ContractManager
from reagent.events import ObservabilityTimeline
