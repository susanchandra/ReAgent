"""
test_nemotron_hotel_integration.py — Phase 2 Nemotron Integration Test Suite.

Verifies the integration of NVIDIA Nemotron contract proposals into the
hotel-booking HTTP service demonstration across all reliability invariants:
  1. Offline execution with mock client injection (zero live network calls).
  2. Strict provenance protection (attacker cannot inject provenance).
  3. Grounding validator enforcement (state-changing requires independent postconditions).
  4. Contract freezing and deterministic verification in Scenario B (Commit-then-timeout).
  5. Contract freezing and safe recovery in Scenario C (Fail-before-commit).
  6. UI endpoint dispatch and fails-closed behavior.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from reagent.models import StructuredDecision, Postcondition
from reagent.providers.nebius import NebiusNemotronProvider
from reagent.tools.hotel_tool import HotelBookingHttpTool
from reagent.contract_validator import ContractGroundingValidator
from reagent.services.hotel_demo import run_hotel_booking_demo
from reagent.services.hotel_booking_service import HotelBookingServiceSimulator


def _make_mock_client(content: str, finish_reason: str = "stop", raise_exc=None):
    """Return a mock OpenAI-compatible client for offline provider tests."""
    client = MagicMock()
    if raise_exc is not None:
        client.chat.completions.create.side_effect = raise_exc
        return client

    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    response = SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=45, completion_tokens=55, total_tokens=100),
    )
    client.chat.completions.create.return_value = response
    return client


class TestNemotronHotelIntegration(unittest.TestCase):
    """Offline integration tests for Nemotron proposal and hotel-booking execution."""

    @classmethod
    def setUpClass(cls):
        cls.simulator = HotelBookingServiceSimulator()
        cls.simulator.start()

    @classmethod
    def tearDownClass(cls):
        cls.simulator.stop()

    def setUp(self):
        self.simulator.reset_database()

    def test_01_model_decision_generation_with_injected_mock_client(self):
        """Model proposes a valid hotel booking contract; returns typed StructuredDecision."""
        valid_hotel_json = json.dumps({
            "tool": "hotel_booking_http",
            "arguments": {
                "booking_id": "BKG-NM-9901",
                "guest_name": "Maya Lin",
                "room_number": "14B",
                "amount": 350.0,
            },
            "proposed_postconditions": [
                {
                    "predicate": "STATUS_SUCCESS",
                    "target": "reservations",
                    "expected": "CONFIRMED",
                    "description": "Reservation confirmed",
                    "required": True,
                },
                {
                    "predicate": "KEY_EXISTS",
                    "target": "booking_id",
                    "expected": "BKG-NM-9901",
                    "description": "Booking reference returned",
                    "required": True,
                },
                {
                    "predicate": "VALUE_EQUALS",
                    "target": "room_number",
                    "expected": "14B",
                    "description": "Room 14B allocated",
                    "required": True,
                },
            ],
            "diagnosis_category": None,
            "diagnosis_confidence": 0.95,
            "diagnosis_reason": "Pre-execution contract proposal for hotel reservation",
            "recovery_strategy": None,
        })
        client = _make_mock_client(valid_hotel_json)
        provider = NebiusNemotronProvider(client=client)

        decision = provider.make_structured_decision(
            task_objective="Book Room 14B at Grand Hotel for Guest Maya Lin",
            available_tools=["hotel_booking_http"],
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.tool, "hotel_booking_http")
        self.assertEqual(decision.arguments["booking_id"], "BKG-NM-9901")
        self.assertEqual(decision.arguments["room_number"], "14B")
        self.assertEqual(len(decision.proposed_postconditions), 3)

    def test_02_provenance_injection_rejection(self):
        """Model attempting to inject 'provenance': 'EXECUTED' is strictly rejected."""
        forged_json = json.dumps({
            "tool": "hotel_booking_http",
            "arguments": {"booking_id": "BKG-HACK-01", "room_number": "14B"},
            "proposed_postconditions": [
                {"predicate": "STATUS_SUCCESS", "description": "Done"}
            ],
            "provenance": "EXECUTED",
        })
        client = _make_mock_client(forged_json)
        provider = NebiusNemotronProvider(client=client)

        decision = provider.make_structured_decision(
            task_objective="Book Room 14B at Grand Hotel",
            available_tools=["hotel_booking_http"],
        )
        self.assertIsNone(decision, "Attempt to inject provenance must be rejected")
        self.assertEqual(provider.last_validation_error, "PROVENANCE_INJECTION_REJECTED")

    def test_03_missing_arguments_rejection(self):
        """Model payload with non-dict arguments is rejected."""
        malformed_json = json.dumps({
            "tool": "hotel_booking_http",
            "arguments": "not-a-dict",
            "proposed_postconditions": [
                {"predicate": "STATUS_SUCCESS", "description": "Done"}
            ],
        })
        client = _make_mock_client(malformed_json)
        provider = NebiusNemotronProvider(client=client)

        decision = provider.make_structured_decision(
            task_objective="Book Room 14B",
            available_tools=["hotel_booking_http"],
        )
        self.assertIsNone(decision)
        self.assertEqual(provider.last_validation_error, "SCHEMA_INVALID_ARGUMENTS")

    def test_04_grounding_validator_rejects_status_only_for_hotel_tool(self):
        """ContractGroundingValidator rejects STATUS_SUCCESS alone for state-changing hotel tool."""
        tool = HotelBookingHttpTool()
        validator = ContractGroundingValidator()

        decision = StructuredDecision(
            tool="hotel_booking_http",
            arguments={"booking_id": "BKG-1", "room_number": "14B"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS", description="Call returned 200")
            ],
        )

        res = validator.validate(decision, tool, objective="Book Room 14B at Grand Hotel")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("state-changing but has no independent observable postcondition" in err for err in res.errors))

    def test_05_grounding_validator_accepts_independent_observables(self):
        """ContractGroundingValidator accepts contracts with independent observables grounded in objective."""
        tool = HotelBookingHttpTool()
        validator = ContractGroundingValidator()

        decision = StructuredDecision(
            tool="hotel_booking_http",
            arguments={"booking_id": "BKG-1", "room_number": "14B"},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS", description="Confirmed"),
                Postcondition(
                    predicate="KEY_EXISTS",
                    target="room_number",
                    expected="14B",
                    description="Room 14B confirmed",
                    required=True,
                ),
            ],
        )

        res = validator.validate(decision, tool, objective="Book Room 14B at Grand Hotel")
        self.assertTrue(res.is_valid, f"Validation failed with errors: {res.errors}")

    def test_06_scenario_b_restraint_with_model_contract(self):
        """Scenario B executes with validated model contract: commits, times out, restrains, marks model provenance."""
        decision = StructuredDecision(
            tool="hotel_booking_http",
            arguments={"booking_id": "BKG-NM-9901", "room_number": "14B"},
            proposed_postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    target="reservations",
                    expected="CONFIRMED",
                    description="Reservation confirmed",
                    required=True,
                ),
                Postcondition(
                    predicate="KEY_EXISTS",
                    target="booking_id",
                    expected="BKG-NM-9901",
                    description="Booking reference present",
                    required=True,
                ),
                Postcondition(
                    predicate="VALUE_EQUALS",
                    target="room_number",
                    expected="14B",
                    description="Room 14B confirmed",
                    required=True,
                ),
            ],
        )

        artifact = run_hotel_booking_demo(
            scenario_id="B",
            simulator=self.simulator,
            use_model_contract=True,
            decision=decision,
        )

        # ReAgent verified restraint
        reagent = artifact.reagent_arm
        self.assertEqual(reagent.final_status, "VERIFIED")
        self.assertFalse(reagent.duplicate_occurred)
        self.assertEqual(reagent.provenance, "MODEL_PROPOSAL_VERIFIED")
        self.assertEqual(reagent.details.get("contract_source"), "NVIDIA_NEMOTRON_ADVISORY")
        self.assertEqual(artifact.provenance, "MODEL_PROPOSAL_VERIFIED")

        # Database state: exactly 1 reservation committed, zero duplication
        self.assertEqual(self.simulator.db.count_reservations(), 1)

    def test_07_scenario_c_safe_retry_with_model_contract(self):
        """Scenario C executes with model contract: fails before commit, verifies absence, retries safely."""
        decision = StructuredDecision(
            tool="hotel_booking_http",
            arguments={"booking_id": "BKG-NM-9902", "room_number": "14B"},
            proposed_postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    target="reservations",
                    expected="CONFIRMED",
                    description="Reservation confirmed",
                    required=True,
                ),
                Postcondition(
                    predicate="KEY_EXISTS",
                    target="booking_id",
                    expected="BKG-NM-9902",
                    description="Booking reference present",
                    required=True,
                ),
            ],
        )

        artifact = run_hotel_booking_demo(
            scenario_id="C",
            simulator=self.simulator,
            use_model_contract=True,
            decision=decision,
        )

        reagent = artifact.reagent_arm
        self.assertEqual(reagent.final_status, "VERIFIED")
        self.assertFalse(reagent.duplicate_occurred)
        self.assertEqual(reagent.provenance, "MODEL_PROPOSAL_VERIFIED")
        self.assertEqual(self.simulator.db.count_reservations(), 1)

    def test_08_ui_propose_endpoint_fails_closed_without_key(self):
        """POST /api/hotel/propose_contract fails closed when NEBIUS_API_KEY is not set."""
        from reagent.ui_server import DashboardHTTPHandler
        import io
        from unittest.mock import MagicMock

        handler = DashboardHTTPHandler.__new__(DashboardHTTPHandler)
        handler.path = "/api/hotel/propose_contract"
        handler.command = "POST"
        handler.wfile = io.BytesIO()

        # Mock send_response, send_header, end_headers
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        with patch.dict("os.environ", {"NEBIUS_API_KEY": ""}):
            handler.do_POST()

        handler.wfile.seek(0)
        output = json.loads(handler.wfile.read().decode("utf-8"))
        self.assertEqual(output.get("status"), "NOT_CONFIGURED")
        self.assertFalse(output.get("is_live"))
        self.assertEqual(output.get("task_template"), "hotel_booking")

    def test_09_public_space_entrypoint_locks_all_propose_routes(self):
        """PublicSpaceDashboardHandler locks both /api/nemotron_propose and /api/hotel/propose_contract."""
        from deploy.hf_space.entrypoint import PublicSpaceDashboardHandler
        import io
        from unittest.mock import MagicMock

        for route in ("/api/nemotron_propose", "/api/hotel/propose_contract"):
            handler = PublicSpaceDashboardHandler.__new__(PublicSpaceDashboardHandler)
            handler.path = route
            handler.command = "POST"
            handler.wfile = io.BytesIO()
            handler.send_response = MagicMock()
            handler.send_header = MagicMock()
            handler.end_headers = MagicMock()

            # Even if an API key were present in env, public mode MUST lock the route
            with patch.dict("os.environ", {"NEBIUS_API_KEY": "fake-key-for-test"}):
                handler.do_POST()

            handler.wfile.seek(0)
            output = json.loads(handler.wfile.read().decode("utf-8"))
            self.assertEqual(output.get("status"), "PUBLIC_DEMO_LOCKED", f"Route {route} was not locked")
            self.assertFalse(output.get("is_live"))

    def test_10_client_cannot_forge_verified_outcome(self):
        """Query parameters or body cannot force a verified outcome without execution."""
        from reagent.ui_server import DashboardHTTPHandler
        import io
        from unittest.mock import MagicMock

        handler = DashboardHTTPHandler.__new__(DashboardHTTPHandler)
        # Attempt to pass forged status in query string
        handler.path = "/api/run_recovery_demo?task_template=hotel_booking&scenario=D&final_status=VERIFIED"
        handler.command = "POST"
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_POST()

        handler.wfile.seek(0)
        output = json.loads(handler.wfile.read().decode("utf-8"))
        # In Scenario D (observer failure), status MUST be ESCALATED regardless of query params
        reagent = output.get("reagent_arm", {})
        self.assertEqual(reagent.get("final_status"), "ESCALATED")
        self.assertFalse(reagent.get("contract_postconditions_verified"))


if __name__ == "__main__":
    unittest.main()
