"""
reagent/generic_capability.py — Generic Capability Adapter Interface & Intent Engine

Phase 6B: Truly generic runtime architecture for ReAgent.
Separates:
  1. Natural-language intent understanding and parameter extraction
  2. Multi-step task plan representation and prerequisite analysis
  3. Pre-execution frozen contract declaration
  4. Capability discovery and registration via typed adapters
  5. State-changing execution under controlled fault modes (A–E)
  6. Independent observation of authoritative state
  7. Deterministic verification and safe-recovery policy

Design principles:
  - Open-ended natural language input: any task can be submitted.
  - Adding a capability requires zero changes to the central routing or classifiers.
  - Unregistered capabilities produce a complete interpretation and proposed action plan,
    disclosing missing integrations, permissions, and verification requirements.
  - Never substitute an unrelated operation.
  - Never claim verified success without independent observation.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

from reagent.capability_registry import CapabilityDescriptor, CapabilityRegistry, CAPABILITY_REGISTRY
from reagent.experiments.dbops.recovery_demo import DemoArtifact
from reagent.models import StructuredDecision
from reagent.recovery_engine import CapabilityGapReport


# ─── Data Contracts ───────────────────────────────────────────────────────────

@dataclass
class TaskPlanStep:
    """A discrete step in an agent's multi-step execution plan."""
    step_num: int
    name: str
    description: str
    phase: str                           # "PREFLIGHT", "CONTRACT", "EXECUTE", "OBSERVE", "DECIDE"
    is_state_changing: bool = False
    requires_independent_verification: bool = False


@dataclass
class SuccessContractSpec:
    """Specification of the success contract frozen before state-changing execution."""
    target_system: str
    operation: str
    preconditions: str
    postconditions: str
    safety_invariant: str = "A timeout is not a no; UNKNOWN != ABSENT. Independent probe required before retry."
    idempotency_key: Optional[str] = None


@dataclass
class MissingIntegrationSpec:
    """Explains why an unconfigured capability cannot execute and what is required to add it."""
    capability_id: str
    required_adapter_interface: str = "reagent.generic_capability.CapabilityAdapter"
    required_permissions: List[str] = field(default_factory=list)
    independent_verification_requirement: str = ""
    suggested_configuration_code: str = ""


@dataclass
class IntentInterpretation:
    """Structured interpretation of any natural-language request."""
    raw_prompt: str
    inferred_domain: str
    inferred_action: str
    extracted_parameters: Dict[str, Any]
    multi_step_plan: List[TaskPlanStep]
    missing_prerequisites: List[str]
    proposed_contract: SuccessContractSpec
    is_registered: bool
    matching_capability: Optional[str] = None
    missing_integration: Optional[MissingIntegrationSpec] = None
    gap_report: Optional[CapabilityGapReport] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        return d


# ─── Abstract Capability Adapter Interface ────────────────────────────────────

class CapabilityAdapter(ABC):
    """
    Extensible interface for capability execution backends.

    Any tool, service simulator, or database integration implements this interface
    to participate in ReAgent outcome verification without touching central server dispatch.
    """

    @property
    @abstractmethod
    def capability_id(self) -> str:
        """Unique identifier matching CapabilityDescriptor.name (e.g. 'hotel_booking')."""
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable name for UI badges and reporting."""
        pass

    @property
    @abstractmethod
    def descriptor(self) -> CapabilityDescriptor:
        """Descriptor for capability discovery and natural language resolution."""
        pass

    @property
    def supported_operations(self) -> List[str]:
        """List of supported operation names."""
        return ["execute"]

    @property
    def required_permissions(self) -> List[str]:
        """List of permissions or roles required by this adapter."""
        return []

    @property
    def independent_observer_description(self) -> str:
        """Description of how authoritative state is observed independently."""
        return "Independent read-only probe of backend state store"

    @abstractmethod
    def create_contract(
        self,
        prompt: str,
        scenario_id: str,
        run_id: str,
        entities: Dict[str, str],
        use_model_contract: bool = False,
    ) -> SuccessContractSpec:
        """Formulate and freeze the task-specific success contract before execution."""
        pass

    @abstractmethod
    def execute(
        self,
        scenario_id: str,
        timestamp_utc: Optional[float] = None,
        run_id: Optional[str] = None,
        use_model_contract: bool = False,
        decision: Optional[StructuredDecision] = None,
        **kwargs: Any,
    ) -> DemoArtifact:
        """
        Execute the scenario head-to-head (Naive vs ReAgent).
        Must independently observe authoritative state and return a DemoArtifact.
        """
        pass


# ─── Generic Intent Analyzer ──────────────────────────────────────────────────

class GenericIntentAnalyzer:
    """
    Interprets arbitrary natural language task input into structured intent,
    multi-step execution plan, and contract specification.

    Deterministic, offline, zero model dependency.
    """

    # Domain vocabulary patterns for intent recognition
    _DOMAIN_PATTERNS = [
        ("rideshare_booking", ["uber", "lyft", "ride", "rideshare", "taxi", "cab", "pickup", "dropoff", "driver", "auto"], "Rideshare / Transport Booking"),
        ("food_delivery", ["doordash", "ubereats", "grubhub", "delivery", "groceries", "grocery", "food", "restaurant", "meal", "biryani", "pizza", "burger", "order"], "Food & Grocery Delivery"),
        ("flight_booking", ["flight", "flights", "airline", "airlines", "plane", "airplane", "fly", "airport", "pnr"], "Flight Reservation"),
        ("hotel_booking", ["hotel", "room", "stay", "accommodation", "resort", "suite", "check-in", "hostel", "lodging"], "Hotel / Accommodation Booking"),
        ("wallet_transfer", ["transfer", "send money", "wallet", "wire", "remit", "pay", "payment", "rupees", "₹", "dollars", "$"], "Financial / Wallet Transfer"),
        ("database_update", ["refund", "database", "order", "update order", "reimburse", "rollback", "reversal"], "Database Record / Refund"),
        ("cloud_infrastructure", ["s3", "bucket", "kubernetes", "pod", "cluster", "deploy", "server", "vm", "container", "cloud"], "Cloud Infrastructure / DevOps"),
        ("calendar_scheduling", ["meeting", "calendar", "schedule", "appointment", "reschedule", "invite"], "Calendar & Scheduling"),
        ("email_communication", ["email", "emails", "inbox", "mail", "send message", "forward", "spam"], "Email & Communications"),
        ("code_execution", ["python", "script", "run code", "compile", "execute script", "benchmark", "test suite"], "Code & Workflow Execution"),
    ]

    def interpret(
        self,
        raw_prompt: str,
        registry: Optional[CapabilityRegistry] = None,
    ) -> IntentInterpretation:
        """Analyze raw prompt into structured intent, plan, and requirements."""
        if registry is None:
            registry = CAPABILITY_REGISTRY

        clean = (raw_prompt or "").strip()
        normalized = clean.lower()

        # Step 1: Detect Domain
        inferred_domain = "Custom / Unspecified Domain"
        domain_key = "custom_task"
        for key, keywords, label in self._DOMAIN_PATTERNS:
            if any(k in normalized for k in keywords):
                domain_key = key
                inferred_domain = label
                break

        # Step 2: Extract Entities & Parameters
        extracted_params: Dict[str, Any] = {}
        # Amounts
        amt_match = re.search(r"[₹$](\d+(?:\.\d+)?)", clean) or re.search(r"\b(\d+(?:\.\d+)?)\s*(?:dollars|rupees|usd|inr)\b", normalized)
        if amt_match:
            extracted_params["amount"] = amt_match.group(1)

        # Identifiers (ORD-*, CUST-*, REF-*, ACCT-*, TXN-*, BKG-*, etc.)
        ids = re.findall(r"\b[A-Z]{3,5}-\d+\b", clean)
        if ids:
            extracted_params["identifiers"] = ids

        # Room/seat codes like 14B, 12A
        room_match = re.search(r"\b(\d{1,3}[A-Za-z])\b", clean)
        if room_match:
            extracted_params["target_unit"] = room_match.group(1).upper()

        # Destination or recipient named entities
        to_match = re.search(r"\b(?:to|for|at|in)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:for|at|in|on|\.|$)|$)", clean)
        if to_match:
            extracted_params["target_entity"] = to_match.group(1).strip()

        # Step 3: Infer Action Verb
        action_match = re.search(r"\b(schedule|book|reserve|transfer|send|pay|refund|update|deploy|provision|create|delete|check|cancel|sync|order|deliver|request|dispatch)\b", normalized)
        inferred_action = action_match.group(1).capitalize() if action_match else "Execute Requested Action"

        # Step 4: Check if a registered capability matches
        res = registry.resolve(clean)
        is_registered = res.resolved and res.matched is not None
        matching_cap = res.matched if is_registered else None

        # Step 5 & 6: Plan Representation & Gap Analysis
        missing_prereqs: List[str] = []
        missing_integration: Optional[MissingIntegrationSpec] = None
        gap_report: Optional[CapabilityGapReport] = None

        # Multi-step plan representation (5 stages across lifecycle)
        multi_step_plan = [
            TaskPlanStep(
                step_num=1,
                name="Prerequisites & Pre-Flight Check",
                description=f"Inspect system status and verify prerequisites for '{inferred_action}'.",
                phase="PREFLIGHT",
                is_state_changing=False,
            ),
            TaskPlanStep(
                step_num=2,
                name="Contract Declaration & Freezing",
                description="Freeze task-specific success contract (preconditions, postconditions, idempotency key).",
                phase="CONTRACT",
                is_state_changing=False,
            ),
            TaskPlanStep(
                step_num=3,
                name="State-Changing Execution",
                description=f"Attempt '{inferred_action}' via designated tool with timeout and fault monitoring." if is_registered else f"BLOCKED: No execution adapter registered for '{domain_key}'. Refused before mutation.",
                phase="EXECUTE",
                is_state_changing=is_registered,
            ),
            TaskPlanStep(
                step_num=4,
                name="Independent Authoritative Observation",
                description="Probe authoritative system of record directly (bypassing tool return value).",
                phase="OBSERVE",
                is_state_changing=False,
                requires_independent_verification=True,
            ),
            TaskPlanStep(
                step_num=5,
                name="Deterministic Recovery Decision",
                description="Evaluate frozen contract against independent evidence: verify success, safely retry, or escalate.",
                phase="DECIDE",
                is_state_changing=False,
            ),
        ]

        if not is_registered:
            # Detailed capability deficiency analysis for unregistered domains
            if domain_key == "food_delivery":
                missing_prereqs = [
                    "Food delivery merchant / restaurant dispatch API access",
                    "Authoritative order state observer probe (kitchen / POS status)",
                    "Payment capture token with idempotency constraint",
                ]
                permissions = ["food:order_create", "pos:read_order_status"]
                verification_req = "Query merchant POS or dispatch system directly to confirm order status independently of tool response."
                ext_code = (
                    "# Register a Food Delivery Adapter:\n"
                    "from reagent.adapter_registry import ADAPTER_REGISTRY\n"
                    "from reagent.generic_capability import CapabilityAdapter\n\n"
                    "class FoodDeliveryAdapter(CapabilityAdapter):\n"
                    "    capability_id = 'food_delivery'\n"
                    "    label = 'Food & Grocery Delivery'\n\n"
                    "ADAPTER_REGISTRY.register('food_delivery', FoodDeliveryAdapter())\n"
                )
            elif domain_key == "rideshare_booking":
                missing_prereqs = [
                    "Rideshare Dispatch API credentials (e.g. Uber / Lyft OAuth token)",
                    "Independent ride telemetry / dispatch status observer probe",
                    "Pre-authorized payment method with idempotency keying",
                ]
                permissions = ["rideshare:dispatch", "rideshare:read_authoritative_status"]
                verification_req = "Authoritative probe must query ride dispatch service directly to confirm trip status and driver assignment independently."
                ext_code = (
                    "# Register a Rideshare Adapter:\n"
                    "from reagent.adapter_registry import ADAPTER_REGISTRY\n"
                    "from reagent.generic_capability import CapabilityAdapter\n\n"
                    "class UberAdapter(CapabilityAdapter):\n"
                    "    capability_id = 'rideshare_booking'\n"
                    "    label = 'Uber / Lyft Dispatch'\n"
                    "    # Implement create_contract, execute, observe\n\n"
                    "ADAPTER_REGISTRY.register('rideshare_booking', UberAdapter())\n"
                )
            elif domain_key == "flight_booking":
                missing_prereqs = [
                    "Airline GDS / Reservation API credentials",
                    "Independent PNR verification probe endpoint",
                    "Payment token or pre-authorized billing clearance",
                ]
                permissions = ["flight:reserve", "pnr:read_authoritative"]
                verification_req = "Authoritative probe must query airline PNR database directly, bypassing the tool response."
                ext_code = (
                    "# Register a flight booking adapter:\n"
                    "from reagent.adapter_registry import ADAPTER_REGISTRY\n"
                    "from reagent.capability_registry import CAPABILITY_REGISTRY, CapabilityDescriptor\n\n"
                    "class FlightAdapter(CapabilityAdapter):\n"
                    "    capability_id = 'flight_booking'\n"
                    "    label = 'Airline Flight Booking'\n"
                    "    # implement create_contract, execute, observe\n\n"
                    "ADAPTER_REGISTRY.register('flight_booking', FlightAdapter())\n"
                )
            elif domain_key == "cloud_infrastructure":
                missing_prereqs = [
                    "Cloud provider IAM credentials with write access",
                    "Independent resource state observer (e.g. AWS HeadBucket or K8s API)",
                ]
                permissions = ["infra:provision", "infra:read_status"]
                verification_req = "Query cloud resource management API directly for resource ARN/state."
                ext_code = (
                    "ADAPTER_REGISTRY.register('cloud_infrastructure', CloudInfraAdapter())\n"
                )
            else:
                missing_prereqs = [
                    f"Registered execution adapter for capability '{domain_key}'",
                    "Independent observer connection to authoritative state store",
                ]
                permissions = ["service:execute", "store:read_authoritative"]
                verification_req = "Read-only query mechanism into authoritative state of record."
                ext_code = f"ADAPTER_REGISTRY.register('{domain_key}', CustomAdapter())\n"

            missing_integration = MissingIntegrationSpec(
                capability_id=domain_key,
                required_permissions=permissions,
                independent_verification_requirement=verification_req,
                suggested_configuration_code=ext_code,
            )

            available_alts = ["hotel_booking", "wallet_transfer", "database_update"]
            gap_report = CapabilityGapReport(
                detected_domain=inferred_domain,
                detected_action=inferred_action,
                required_capability_id=domain_key,
                is_executable=False,
                execution_blocked_reason=(
                    f"Execution blocked before mutation. ReAgent strictly forbids blind execution "
                    f"of unverified tools or unconfigured capabilities ('{domain_key}'). "
                    f"A task is never marked successful merely because it was understood or planned."
                ),
                missing_prerequisites=missing_prereqs,
                required_permissions=permissions,
                independent_verification_requirement=verification_req,
                available_alternatives=available_alts,
                safe_recommendation="DO_NOT_EXECUTE",
                extensibility_example=ext_code,
            )

        # Step 7: Proposed Success Contract
        proposed_contract = SuccessContractSpec(
            target_system=inferred_domain,
            operation=inferred_action,
            preconditions=f"System ready; parameters validated: {extracted_params or 'standard defaults'}",
            postconditions=f"Authoritative record of '{inferred_action}' persisted and independently observable",
            safety_invariant="A timeout is not a no; UNKNOWN != ABSENT. Independent probe required before retry.",
            idempotency_key=extracted_params.get("identifiers", ["TXN-GENERIC"])[0] if extracted_params.get("identifiers") else "REQ-" + re.sub(r"[^A-Za-z0-9]", "", clean)[:12],
        )

        return IntentInterpretation(
            raw_prompt=clean,
            inferred_domain=inferred_domain,
            inferred_action=inferred_action,
            extracted_parameters=extracted_params,
            multi_step_plan=multi_step_plan,
            missing_prerequisites=missing_prereqs,
            proposed_contract=proposed_contract,
            is_registered=is_registered,
            matching_capability=matching_cap,
            missing_integration=missing_integration,
            gap_report=gap_report,
        )



# Global singleton intent analyzer
INTENT_ANALYZER = GenericIntentAnalyzer()
