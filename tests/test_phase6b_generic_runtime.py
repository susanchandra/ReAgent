"""
tests/test_phase6b_generic_runtime.py — Comprehensive Tests for Phase 6B Generic Runtime

Verifies:
  1. Arbitrary unforeseen natural-language tasks are interpreted into domain, action,
     extracted parameters, 5-phase multi-step plan, and frozen contract specification.
  2. Unregistered capabilities fail closed safely before mutation, providing:
     - Interpreted intent and extracted parameters
     - Proposed success contract
     - Missing integration & observer specifications
     - Python configuration path to register an adapter
  3. Dynamic adapter registration: a newly registered adapter works immediately
     through natural-language resolution and dispatch without modifying central dispatch
     or classifier if/elif chains.
  4. Strict safety against wrong-adapter execution: unregistered capabilities never
     substitute an unrelated operation or silently execute wallet_transfer.
  5. Independent observation invariant: no execution claims success without
     independent authoritative evidence.
  6. Core safety principles (A timeout isn't a no; UNKNOWN != ABSENT; duplicate prevention)
     remain fully enforced across all scenarios.
  7. Screen separation: Task Workbench, Reliability Lab (Scenarios), and Evidence views
     serve distinct, non-duplicated purposes.
"""

import json
import threading
import time
import unittest
import urllib.parse
import urllib.request
from http.server import HTTPServer
from typing import Dict, Any, Optional

from reagent.adapter_registry import ADAPTER_REGISTRY, AdapterRegistry
from reagent.capability_registry import (
    CAPABILITY_REGISTRY,
    CapabilityDescriptor,
    CapabilityRegistry,
)
from reagent.experiments.dbops.recovery_demo import DemoArtifact
from reagent.generic_capability import (
    CapabilityAdapter,
    GenericIntentAnalyzer,
    INTENT_ANALYZER,
    SuccessContractSpec,
)
from reagent.models import StructuredDecision
from reagent.ui_server import DashboardHTTPHandler, generate_dashboard_html


# ─── Mock Test Adapter for Dynamic Registration ───────────────────────────────

class MockCustomEchoAdapter(CapabilityAdapter):
    """
    Test adapter registered dynamically during tests to prove zero-change extensibility.
    Implements the CapabilityAdapter interface.
    """

    @property
    def capability_id(self) -> str:
        return "mock_echo_service"

    @property
    def label(self) -> str:
        return "Mock Echo & Audit Service"

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            name="mock_echo_service",
            label="Mock Echo & Audit Service",
            intent_verbs=frozenset({"echo", "audit", "ping", "test"}),
            entity_keywords=frozenset({"payload", "message", "echo", "audit"}),
            fixture_description="Echos payload to authoritative memory store with audit verification.",
            fixture_entities={"payload": "hello"},
            fixture_notice="Running mock echo service in memory.",
            supported_examples=["Echo test payload: hello world"],
        )

    def create_contract(
        self,
        prompt: str,
        scenario_id: str,
        run_id: str,
        entities: Dict[str, str],
        use_model_contract: bool = False,
    ) -> SuccessContractSpec:
        return SuccessContractSpec(
            target_system="Mock Echo Store",
            operation="ECHO_MESSAGE",
            preconditions="Echo memory store initialized",
            postconditions="Message payload persisted in authoritative echo log",
            idempotency_key=f"ECHO-{run_id[:8]}",
        )

    def execute(
        self,
        scenario_id: str,
        timestamp_utc: Optional[float] = None,
        run_id: Optional[str] = None,
        use_model_contract: bool = False,
        decision: Optional[StructuredDecision] = None,
        **kwargs: Any,
    ) -> DemoArtifact:
        ts = timestamp_utc or time.time()
        rid = run_id or f"mock_run_{int(ts)}"

        # Simulate ReAgent verification: independent observer queries memory log
        observed_record = {"status": "COMMITTED", "payload": "hello world", "records_found": 1}

        reagent_arm = {
            "strategy": "reagent_diagnosis_guided",
            "final_status": "VERIFIED",
            "verified": True,
            "duplicate_occurred": False,
            "tool_attempts": 1,
            "probe_queries": 1,
            "details": {
                "scenario": f"mock_echo_service_{scenario_id}",
                "observed_record": observed_record,
                "contract_hash": f"sha256:echo_{rid[:8]}",
            },
        }
        naive_arm = {
            "strategy": "naive_retry",
            "final_status": "UNVERIFIED",
            "verified": False,
            "duplicate_occurred": False,
            "tool_attempts": 1,
        }

        return DemoArtifact(
            demo_name="Mock Echo Demo",
            task_id="task_mock_echo",
            fault_mode="COMMIT_THEN_TIMEOUT" if scenario_id == "B" else "NONE",
            timestamp_utc=ts,
            provenance="DYNAMIC_TEST_ADAPTER",
            naive_retry_arm=naive_arm,
            reagent_arm=reagent_arm,
            comparison_summary="Dynamic test adapter executed and independently verified.",
            run_id=rid,
            execution_duration_ms=42.0,
        )


# ─── Test Suite ───────────────────────────────────────────────────────────────

class TestPhase6BGenericRuntime(unittest.TestCase):
    """Test suite validating generic runtime requirements."""

    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _post_json(self, url: str) -> dict:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── 1. Intent Understanding & Parameter Extraction ────────────────────────

    def test_01_unforeseen_flight_intent_interpreted(self):
        """Unforeseen flight booking is parsed into structured domain, plan, and requirements."""
        phrase = "Book a flight from NYC to London Heathrow on Dec 12"
        interp = INTENT_ANALYZER.interpret(phrase, CAPABILITY_REGISTRY)

        self.assertEqual(interp.inferred_domain, "Flight Reservation")
        self.assertEqual(interp.inferred_action, "Book")
        self.assertFalse(interp.is_registered)
        self.assertIsNotNone(interp.missing_integration)
        self.assertEqual(interp.missing_integration.capability_id, "flight_booking")
        self.assertIn("flight:reserve", interp.missing_integration.required_permissions)

        # Multi-step plan contains required phases
        phases = [step.phase for step in interp.multi_step_plan]
        self.assertIn("PREFLIGHT", phases)
        self.assertIn("CONTRACT", phases)
        self.assertIn("EXECUTE", phases)
        self.assertIn("OBSERVE", phases)
        self.assertIn("DECIDE", phases)

        # Proposed contract contains safety invariant
        self.assertIn("UNKNOWN != ABSENT", interp.proposed_contract.safety_invariant)

    def test_02_unforeseen_cloud_storage_intent_interpreted(self):
        """Unforeseen cloud infrastructure request is parsed with plan and missing credentials."""
        phrase = "Provision an S3 bucket named logs-backup-2026 for cluster audit"
        interp = INTENT_ANALYZER.interpret(phrase, CAPABILITY_REGISTRY)

        self.assertEqual(interp.inferred_domain, "Cloud Infrastructure / DevOps")
        self.assertEqual(interp.inferred_action, "Provision")
        self.assertFalse(interp.is_registered)
        self.assertIsNotNone(interp.missing_integration)
        self.assertIn("infra:provision", interp.missing_integration.required_permissions)

    def test_03_registered_capability_recognized_by_intent_engine(self):
        """Registered hotel and wallet tasks are correctly recognized as registered."""
        hotel_interp = INTENT_ANALYZER.interpret("Book Room 14B at Grand Hotel for Guest Maya")
        self.assertTrue(hotel_interp.is_registered)
        self.assertEqual(hotel_interp.matching_capability, "hotel_booking")

        wallet_interp = INTENT_ANALYZER.interpret("Send ₹500 to Asha for lunch")
        self.assertTrue(wallet_interp.is_registered)
        self.assertEqual(wallet_interp.matching_capability, "wallet_transfer")

    # ── 2. Unregistered Capability Safe Fail-Closed ────────────────────────────

    def test_04_unregistered_flight_returns_rich_plan_and_fails_closed(self):
        """
        POST with arbitrary flight request fails closed before mutation,
        returning HTTP 400 with UNSUPPORTED_TASK and rich plan metadata.
        """
        phrase = "Book a flight to India"
        url = self._url(f"/api/run_recovery_demo?scenario=B&task_input={urllib.parse.quote(phrase)}")
        req = urllib.request.Request(url, method="POST")

        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTP 400 for unconfigured capability")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            body = json.loads(e.read().decode("utf-8"))
            self.assertEqual(body.get("status"), "UNSUPPORTED_TASK")
            self.assertIn("Flights", body.get("error", ""))

            # Rich interpretation payload is attached
            interp = body.get("interpreted_intent", {})
            self.assertEqual(interp.get("inferred_domain"), "Flight Reservation")
            self.assertEqual(interp.get("inferred_action"), "Book")
            self.assertGreaterEqual(len(interp.get("multi_step_plan", [])), 5)

            # Proposed contract & missing integration specs are present
            contract = body.get("proposed_contract", {})
            self.assertIn("safety_invariant", contract)

            missing = body.get("missing_integration", {})
            self.assertEqual(missing.get("capability_id"), "flight_booking")
            self.assertIn("flight:reserve", missing.get("required_permissions", []))

            # ReAgent arm discloses safe non-mutation
            reagent_arm = body.get("reagent_arm", {})
            self.assertFalse(reagent_arm.get("mutated", True))
            self.assertIn("strictly forbids blind execution", reagent_arm.get("safe_explanation", ""))

    def test_05_unregistered_task_never_silently_substitutes_wallet(self):
        """
        Direct dispatch to an unregistered key raises KeyError.
        Zero silent fallback to wallet_transfer in ADAPTER_REGISTRY.
        """
        registry = AdapterRegistry()
        registry.register("wallet_transfer", lambda **kw: None)

        with self.assertRaises(KeyError) as ctx:
            registry.dispatch("unregistered_flight_service", scenario_id="B")
        self.assertIn("unregistered_flight_service", str(ctx.exception))
        self.assertIn("strictly forbids", str(ctx.exception))

    # ── 3. Dynamic Adapter Registration (Extensibility) ──────────────────────

    def test_06_dynamically_registered_adapter_executes_without_classifier_changes(self):
        """
        A newly registered adapter immediately resolves natural language,
        executes, and verifies without changing any central code.
        """
        adapter = MockCustomEchoAdapter()

        # Step 1: Register descriptor in CAPABILITY_REGISTRY
        CAPABILITY_REGISTRY.register(adapter.descriptor)
        # Step 2: Register execution adapter in ADAPTER_REGISTRY
        ADAPTER_REGISTRY.register("mock_echo_service", adapter)

        try:
            # NL resolution works directly through generic registry
            res = CAPABILITY_REGISTRY.resolve("Echo test payload: hello world")
            self.assertTrue(res.resolved)
            self.assertEqual(res.matched, "mock_echo_service")

            # Server executes via registered adapter
            url = self._url(
                f"/api/run_recovery_demo?scenario=B&task_template=mock_echo_service"
                f"&task_input={urllib.parse.quote('Echo test payload: hello world')}"
            )
            data = self._post_json(url)

            self.assertEqual(data.get("task_template"), "mock_echo_service")
            self.assertEqual(data.get("entity_label"), "Mock Echo & Audit Service")

            reagent_arm = data.get("reagent_arm", {})
            self.assertEqual(reagent_arm.get("final_status"), "VERIFIED")
            self.assertTrue(reagent_arm.get("verified"))
            self.assertIn("observed_record", reagent_arm.get("details", {}))
        finally:
            # Clean up
            CAPABILITY_REGISTRY.unregister("mock_echo_service")
            ADAPTER_REGISTRY._adapters.pop("mock_echo_service", None)

    # ── 4. Independent Observation Invariant ──────────────────────────────────

    def test_07_scenario_d_observer_unavailable_enforces_escalation(self):
        """
        Scenario D: When observer is unavailable, outcome is ESCALATED (UNKNOWN != ABSENT).
        No blind retry is permitted under uncertainty.
        """
        url = self._url("/api/run_recovery_demo?scenario=D&task_template=hotel_booking")
        data = self._post_json(url)

        reagent_arm = data.get("reagent_arm", {})
        self.assertEqual(reagent_arm.get("final_status"), "ESCALATED")
        self.assertFalse(reagent_arm.get("duplicate_occurred", True))

    def test_08_scenario_b_commit_then_timeout_blocks_duplicate(self):
        """
        Scenario B: Under timeout, independent probe finds committed state;
        redundant retry is suppressed.
        """
        url = self._url("/api/run_recovery_demo?scenario=B&task_template=wallet_transfer")
        data = self._post_json(url)

        reagent_arm = data.get("reagent_arm", {})
        naive_arm = data.get("naive_retry_arm", {})

        self.assertEqual(reagent_arm.get("final_status"), "VERIFIED")
        self.assertFalse(reagent_arm.get("duplicate_occurred", True))
        # Naive baseline attempted blind duplicate retry
        self.assertTrue(naive_arm.get("duplicate_occurred", False))

    # ── 5. Distinct Screen Purpose & UI Invariants ────────────────────────────

    def test_09_screen_separation_in_html(self):
        """HTML contains distinct sections for Workbench, Reliability Lab, and Evidence."""
        html = generate_dashboard_html()

        # Task Workbench view
        self.assertIn('id="view-workbench"', html)
        self.assertIn('TASK WORKBENCH', html)
        self.assertIn('wbTaskInput', html)
        self.assertIn('wbActiveFamilyBadge', html)

        # Scenarios / Reliability Lab view
        self.assertIn('id="view-scenarios"', html)
        self.assertIn('Reliability Lab', html)

        # Evidence & Activity view
        self.assertIn('id="view-evidence"', html)
        self.assertIn('Evidence &amp; Activity', html)
        self.assertIn('evidenceAuditList', html)

    def test_10_no_misleading_live_badge_when_offline(self):
        """When NEBIUS_API_KEY is not set, dashboard displays OFFLINE / REFERENCE mode."""
        html = generate_dashboard_html()
        # Verify offline disclosure is present
        self.assertIn('OFFLINE &bull; DETERMINISTIC', html)


if __name__ == "__main__":
    unittest.main()
