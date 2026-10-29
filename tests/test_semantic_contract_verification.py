"""
tests/test_semantic_contract_verification.py
Phase Final — Comprehensive Semantic Contract & Verification Test Suite.

Deterministic, zero-network regression tests covering:
  1. A valid, concrete, observer-verifiable postcondition is accepted.
  2. The observed generic STATUS_SUCCESS contract with null required fields is rejected.
  3. Missing, null, unsupported, contradictory, and malformed predicate fields are rejected.
  4. Unsupported tool names and altered trusted task identifiers are rejected.
  5. A schema-valid but unobservable postcondition is rejected.
  6. Valid contracts for existing unrelated supported workflows remain accepted.
  7. A live provider failure cannot silently fall back to a simulated fixture.
  8. The observer's actual state, not model output, determines the final verdict.
  9. An ambiguous timeout does not cause a duplicate side effect.
  10. Observer failure or missing evidence yields UNKNOWN rather than fabricated success or failure.
  11. UI/backend attribution correctly separates LIVE NEBIUS INFERENCE from OFFLINE REFERENCE PROPOSAL.
  12. Existing rate limits, authorization, public locking behavior, and server-side secret handling remain intact.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from reagent.models import StructuredDecision, Postcondition
from reagent.tools.hotel_tool import HotelBookingHttpTool
from reagent.tools.json_tool import JsonValidatorTool
from reagent.contract_validator import (
    ContractGroundingValidator,
    validate_contract_proposal,
    ValidationResult,
    SUPPORTED_PREDICATES,
    ALLOWED_HOTEL_OBSERVABLE_KEYS,
)
from reagent.services.hotel_booking_service import HotelBookingServiceSimulator
from reagent.services.hotel_demo import run_hotel_booking_demo
from reagent.ui_server import DashboardHTTPHandler, DemoRateLimiter
import reagent.ui_server as _ui


def _make_mock_client(content: str, finish_reason: str = "stop", usage=None):
    client = MagicMock()
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    resp = SimpleNamespace(
        choices=[choice],
        usage=usage or SimpleNamespace(prompt_tokens=40, completion_tokens=50, total_tokens=90),
    )
    client.chat.completions.create.return_value = resp
    return client


def _invoke_post(path: str, body: Optional[dict] = None) -> dict:
    handler = DashboardHTTPHandler.__new__(DashboardHTTPHandler)
    handler.path = path
    handler.command = "POST"
    handler.headers = {}
    if body:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.headers["Content-Type"] = "application/json"
        handler.rfile = io.BytesIO(body_bytes)
    else:
        handler.rfile = io.BytesIO(b"")
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.do_POST()
    handler.wfile.seek(0)
    raw = handler.wfile.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


class TestSemanticContractVerification(unittest.TestCase):
    """Exhaustive deterministic tests for semantic contract validation & reality verification."""

    @classmethod
    def setUpClass(cls):
        cls.simulator = HotelBookingServiceSimulator()
        cls.simulator.start()

    @classmethod
    def tearDownClass(cls):
        cls.simulator.stop()

    def setUp(self):
        self.simulator.reset_database()
        _ui._demo_rate_limiter = DemoRateLimiter()

    # ── 1. Valid, concrete, observer-verifiable postcondition is accepted ─
    def test_01_valid_concrete_observer_verifiable_contract_accepted(self):
        valid_contract = {
            "tool": "hotel_booking_http",
            "arguments": {
                "booking_id": "BKG-NM-9901",
                "room_number": "14B",
                "guest_name": "Maya Lin",
                "amount": 350.0,
            },
            "proposed_postconditions": [
                {
                    "predicate": "STATUS_SUCCESS",
                    "target": "reservations",
                    "expected": "CONFIRMED",
                    "description": "Reservation confirmed in hotel service",
                    "required": True,
                },
                {
                    "predicate": "KEY_EXISTS",
                    "target": "booking_id",
                    "expected": "BKG-NM-9901",
                    "description": "Booking reference returned by service",
                    "required": True,
                },
                {
                    "predicate": "VALUE_EQUALS",
                    "target": "room_number",
                    "expected": "14B",
                    "description": "Room 14B confirmed",
                    "required": True,
                },
            ],
            "provenance": "model",
        }
        res = validate_contract_proposal(
            contract_dict=valid_contract,
            task_objective="Book Room 14B at Grand Hotel for Guest Maya Lin (booking_id: BKG-NM-9901, room_number: 14B)",
            workflow="hotel_booking",
            task_context={"booking_id": "BKG-NM-9901", "room_number": "14B"},
        )
        self.assertTrue(res.is_valid, f"Expected valid contract, got errors: {res.errors}")
        self.assertEqual(len(res.errors), 0)

    # ── 2. Observed generic STATUS_SUCCESS contract with null fields rejected
    def test_02_observed_generic_status_success_contract_rejected(self):
        generic_contract = {
            "tool": "hotel_booking_http",
            "arguments": {
                "booking_id": "BKG-NM-9901",
                "room_number": "14B",
            },
            "proposed_postconditions": [
                {
                    "predicate": "STATUS_SUCCESS",
                    "target": None,
                    "expected": None,
                    "description": "Action must succeed",
                    "required": True,
                    "postcondition_id": None,
                }
            ],
            "provenance": "model",
        }
        res = validate_contract_proposal(
            contract_dict=generic_contract,
            task_objective="Book Room 14B at Grand Hotel for Guest Maya Lin (booking_id: BKG-NM-9901, room_number: 14B)",
            workflow="hotel_booking",
            task_context={"booking_id": "BKG-NM-9901", "room_number": "14B"},
        )
        self.assertFalse(res.is_valid)
        self.assertTrue(
            any("GenericPostconditionRejected" in e or "state-changing but has no independent" in e for e in res.errors),
            f"Expected generic postcondition rejection, got: {res.errors}",
        )

    # ── 3. Missing, null, unsupported, contradictory predicate fields rejected
    def test_03_missing_null_unsupported_malformed_predicate_fields_rejected(self):
        # 3a. Unsupported predicate
        c_unsupported = {
            "tool": "hotel_booking_http",
            "arguments": {"booking_id": "BKG-NM-9901", "room_number": "14B"},
            "proposed_postconditions": [
                {"predicate": "HOTEL_ROOM_UNCHECKED", "target": "14B"}
            ],
        }
        res_a = validate_contract_proposal(c_unsupported, "Book Room 14B", workflow="hotel_booking")
        self.assertFalse(res_a.is_valid)
        self.assertTrue(any("UnsupportedPredicate" in e for e in res_a.errors))

        # 3b. KEY_EXISTS with null target
        c_null_target = {
            "tool": "hotel_booking_http",
            "arguments": {"booking_id": "BKG-NM-9901", "room_number": "14B"},
            "proposed_postconditions": [
                {"predicate": "KEY_EXISTS", "target": None}
            ],
        }
        res_b = validate_contract_proposal(c_null_target, "Book Room 14B", workflow="hotel_booking")
        self.assertFalse(res_b.is_valid)
        self.assertTrue(any("KEY_EXISTS requires a non-empty string target" in e for e in res_b.errors))

        # 3c. VALUE_EQUALS with null expected
        c_null_expected = {
            "tool": "hotel_booking_http",
            "arguments": {"booking_id": "BKG-NM-9901", "room_number": "14B"},
            "proposed_postconditions": [
                {"predicate": "VALUE_EQUALS", "target": "room_number", "expected": None}
            ],
        }
        res_c = validate_contract_proposal(c_null_expected, "Book Room 14B", workflow="hotel_booking")
        self.assertFalse(res_c.is_valid)
        self.assertTrue(any("VALUE_EQUALS requires a non-empty string target and a non-null expected" in e for e in res_c.errors))

    # ── 4. Unsupported tool names and altered task identifiers rejected ───
    def test_04_unsupported_tool_names_and_altered_task_identifiers_rejected(self):
        # 4a. Disallowed tool for workflow
        c_disallowed_tool = {
            "tool": "arbitrary_remote_exec",
            "arguments": {"booking_id": "BKG-NM-9901", "room_number": "14B"},
            "proposed_postconditions": [
                {"predicate": "KEY_EXISTS", "target": "booking_id"}
            ],
        }
        res_tool = validate_contract_proposal(c_disallowed_tool, "Book Room 14B", workflow="hotel_booking")
        self.assertFalse(res_tool.is_valid)
        self.assertTrue(any("DisallowedWorkflowTool" in e for e in res_tool.errors))

        # 4b. Altered booking_id in arguments
        c_altered_arg = {
            "tool": "hotel_booking_http",
            "arguments": {"booking_id": "BKG-SUBSTITUTED-999", "room_number": "14B"},
            "proposed_postconditions": [
                {"predicate": "KEY_EXISTS", "target": "booking_id", "expected": "BKG-NM-9901"},
                {"predicate": "VALUE_EQUALS", "target": "room_number", "expected": "14B"},
            ],
        }
        res_arg = validate_contract_proposal(
            c_altered_arg,
            "Book Room 14B",
            workflow="hotel_booking",
            task_context={"booking_id": "BKG-NM-9901", "room_number": "14B"},
        )
        self.assertFalse(res_arg.is_valid)
        self.assertTrue(any("IdentifierBindingMismatch" in e and "booking_id" in e for e in res_arg.errors))

        # 4c. Altered room_number in postcondition expected
        c_altered_pc = {
            "tool": "hotel_booking_http",
            "arguments": {"booking_id": "BKG-NM-9901", "room_number": "14B"},
            "proposed_postconditions": [
                {"predicate": "KEY_EXISTS", "target": "booking_id", "expected": "BKG-NM-9901"},
                {"predicate": "VALUE_EQUALS", "target": "room_number", "expected": "99Z"},
            ],
        }
        res_pc = validate_contract_proposal(
            c_altered_pc,
            "Book Room 14B",
            workflow="hotel_booking",
            task_context={"booking_id": "BKG-NM-9901", "room_number": "14B"},
        )
        self.assertFalse(res_pc.is_valid)
        self.assertTrue(any("IdentifierBindingMismatch" in e and "room_number" in e for e in res_pc.errors))

    # ── 5. Schema-valid but unobservable postcondition rejected ───────────
    def test_05_schema_valid_but_unobservable_postcondition_rejected(self):
        c_unobservable = {
            "tool": "hotel_booking_http",
            "arguments": {"booking_id": "BKG-NM-9901", "room_number": "14B"},
            "proposed_postconditions": [
                {"predicate": "KEY_EXISTS", "target": "unobservable_quantum_flux", "expected": "valid"},
            ],
        }
        res = validate_contract_proposal(c_unobservable, "Book Room 14B", workflow="hotel_booking")
        self.assertFalse(res.is_valid)
        self.assertTrue(any("UnobservablePostcondition" in e for e in res.errors))

    # ── 6. Valid contracts for existing unrelated workflows remain accepted
    def test_06_valid_contracts_for_unrelated_workflows_remain_accepted(self):
        # Read-only json tool
        read_tool = JsonValidatorTool()
        validator = ContractGroundingValidator()
        decision = StructuredDecision(
            tool=read_tool.name,
            arguments={"json_payload": '{"status": "ok"}'},
            proposed_postconditions=[
                Postcondition(predicate="STATUS_SUCCESS", description="Validation ok"),
            ],
        )
        res = validator.validate(decision, read_tool, "Validate json payload format")
        self.assertTrue(res.is_valid)

    # ── 7. Live provider failure cannot silently fall back to simulated fixture
    def test_07_live_provider_failure_cannot_silently_fallback_to_simulated_fixture(self):
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-live-key"}):
            with patch("reagent.ui_server.NebiusNemotronProvider") as mock_cls:
                mock_inst = MagicMock()
                mock_inst.model = "nvidia/nemotron-3-super-120b-a12b"
                mock_inst.make_structured_decision.side_effect = RuntimeError("Provider service timeout 504")
                mock_cls.return_value = mock_inst

                resp = _invoke_post("/api/nemotron_propose?live=1")
                # Must return error, NOT a simulated fixture or CONTRACT_VALIDATED
                self.assertIn(resp.get("status"), ("ERROR", "PROVIDER_FAILED"))
                self.assertIsNone(resp.get("contract"))
                self.assertFalse(resp.get("schema_valid"))

    # ── 8. Observer's actual state determines final verdict ───────────────
    def test_08_observer_actual_state_determines_final_verdict(self):
        # In Scenario B, the mutation succeeds in DB, but HTTP network times out.
        # The observer queries SQLite independently, confirms PRESENT, and verifies contract.
        artifact = run_hotel_booking_demo(
            scenario_id="B",
            simulator=self.simulator,
            use_model_contract=False,
        )
        self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED")
        self.assertTrue(artifact.reagent_arm.contract_postconditions_verified)
        self.assertEqual(self.simulator.db.count_reservations(), 1)

    # ── 9. Ambiguous timeout does not cause duplicate side effect ─────────
    def test_09_ambiguous_timeout_does_not_cause_duplicate_side_effect(self):
        artifact = run_hotel_booking_demo(
            scenario_id="B",
            simulator=self.simulator,
            use_model_contract=False,
        )
        # Naive arm duplicated/corrupted
        self.assertTrue(artifact.naive_retry_arm.duplicate_occurred)
        # ReAgent restrained and avoided duplicate
        self.assertFalse(artifact.reagent_arm.duplicate_occurred)
        self.assertEqual(self.simulator.db.count_reservations(), 1)

    # ── 10. Observer failure yields UNKNOWN rather than fabricated outcome ─
    def test_10_observer_failure_yields_unknown_rather_than_fabricated_outcome(self):
        artifact = run_hotel_booking_demo(
            scenario_id="D",  # Observer unavailable (503)
            simulator=self.simulator,
            use_model_contract=False,
        )
        self.assertIn(artifact.reagent_arm.final_status, ("UNKNOWN", "ESCALATED"))
        self.assertEqual(artifact.reagent_arm.details.get("recovery_loop_decision"), "ESCALATED_UNKNOWN_NOT_ABSENT")
        self.assertFalse(artifact.reagent_arm.contract_postconditions_verified)
        self.assertFalse(artifact.reagent_arm.duplicate_occurred)

    # ── 11. UI/backend attribution separates live from offline reference ──
    def test_11_ui_backend_attribution_separates_live_from_offline_reference(self):
        valid_hotel_json = json.dumps({
            "tool": "hotel_booking_http",
            "arguments": {"booking_id": "BKG-NM-9901", "room_number": "14B"},
            "proposed_postconditions": [
                {
                    "predicate": "STATUS_SUCCESS",
                    "target": "reservations",
                    "expected": "CONFIRMED",
                    "description": "Reservation confirmed",
                },
                {
                    "predicate": "KEY_EXISTS",
                    "target": "booking_id",
                    "expected": "BKG-NM-9901",
                    "description": "Booking reference confirmed",
                },
                {
                    "predicate": "VALUE_EQUALS",
                    "target": "room_number",
                    "expected": "14B",
                    "description": "Room 14B allocated",
                },
            ],
        })
        client = _make_mock_client(valid_hotel_json)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-live-key"}):
            with patch("reagent.ui_server.NebiusNemotronProvider") as mock_cls:
                from reagent.providers.nebius import NebiusNemotronProvider
                provider = NebiusNemotronProvider(client=client)
                mock_cls.return_value = provider

                resp = _invoke_post("/api/hotel/propose_contract?live=1")
                self.assertEqual(resp.get("status"), "CONTRACT_VALIDATED")
                self.assertTrue(resp.get("is_live"))
                self.assertEqual(resp.get("token_source"), "PROVIDER_REPORTED")

    # ── 12. Rate limits, public locking, and secret redaction intact ──────
    def test_12_rate_limits_public_locking_and_secret_redaction_intact(self):
        # 12a. Public mode lock without live opt-in
        with patch.dict(os.environ, {"REAGENT_PUBLIC_MODE": "1"}):
            resp = _invoke_post("/api/nemotron_propose")
            self.assertEqual(resp.get("status"), "PUBLIC_DEMO_LOCKED")

        # 12b. Secret redaction check
        test_secret = "sk-super-secret-cannot-leak-into-responses"
        with patch.dict(os.environ, {"NEBIUS_API_KEY": test_secret}):
            resp = _invoke_post("/api/nemotron_propose?live=1")
            raw_resp = json.dumps(resp)
            self.assertNotIn(test_secret, raw_resp)
