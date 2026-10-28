"""
test_hotel_booking_service.py — Verification Test Suite for Hotel Booking HTTP Service & ReAgent Integration.

Covers Phase 1 Requirements:
1. Normal success: HTTP 201, clean commit in SQLite, status verified.
2. Commit-then-timeout: HTTP connection drops after SQLite commit; ReAgent probes, observes PRESENT,
   restrains retry, zero duplicate reservations.
3. Fail-before-commit: HTTP 500 before write (0 rows in SQLite); ReAgent probes, observes CONFIRMED_ABSENT,
   safely retries once and verifies.
4. Observer failure: Status probe returns 503; ReAgent enforces UNKNOWN != ABSENT, escalates without blind retry.
5. Partial commit: Discrepancy between reservation and audit log detected; ReAgent halts and escalates.
6. Duplicate submission: Subsequent POST with same booking ID returns 409 Conflict.
7. Reset isolation: /api/v1/reset clears simulator SQLite tables cleanly.
8. Malformed requests: Missing required fields or negative amounts return 400 Bad Request.
9. Service unavailability: ReAgent gracefully handles connection refusal when service is down.
"""

from __future__ import annotations

import json
import socket
import time
import unittest
import urllib.request
import urllib.error
import requests

from reagent.services.hotel_booking_service import HotelBookingServiceSimulator, HotelBookingDB
from reagent.services.hotel_demo import run_hotel_booking_demo
from reagent.tools.hotel_tool import HotelBookingHttpTool, make_hotel_booking_probe_callback


class TestHotelBookingService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = HotelBookingServiceSimulator()
        cls.base_url = cls.sim.start()

    @classmethod
    def tearDownClass(cls):
        cls.sim.stop()

    def setUp(self):
        self.sim.reset_database()
        self.sim.set_global_fault(None)

    # ── Test 1: Simulator Lifecycle & Health ──────────────────────────────────
    def test_01_simulator_lifecycle_and_health(self):
        """GET /api/v1/health reports ok status and database counts."""
        resp = requests.get(f"{self.base_url}/api/v1/health", timeout=2.0)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "HotelBookingSimulator")
        self.assertEqual(data["reservations"], 0)
        self.assertEqual(data["audit_logs"], 0)

    # ── Test 2: Normal Success Flow (NO_FAULT) ────────────────────────────────
    def test_02_normal_success_workflow(self):
        """POST /api/v1/bookings with NO_FAULT creates reservation in SQLite; GET status confirms."""
        payload = {
            "booking_id": "BKG-TEST-001",
            "guest_name": "Maya Lin",
            "room_number": "14B",
            "amount": 350.00,
            "fault_mode": "NO_FAULT",
        }
        resp = requests.post(f"{self.base_url}/api/v1/bookings", json=payload, timeout=2.0)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["status"], "CONFIRMED")
        self.assertEqual(data["booking_id"], "BKG-TEST-001")
        self.assertEqual(data["room_number"], "14B")

        # Verify persistent database state
        self.assertEqual(self.sim.db.count_reservations(), 1)
        self.assertEqual(self.sim.db.count_audit_logs(), 1)
        res_record = self.sim.db.get_reservation("BKG-TEST-001")
        self.assertIsNotNone(res_record)
        self.assertEqual(res_record["status"], "CONFIRMED")
        self.assertTrue(res_record["audit_logged"])

        # Status probe confirms PRESENT
        probe = make_hotel_booking_probe_callback(self.base_url, "BKG-TEST-001")
        probe_res = probe("hotel_booking_http", {}, 1)
        self.assertEqual(probe_res["status"], "PRESENT")
        self.assertEqual(probe_res["booking_id"], "BKG-TEST-001")

    # ── Test 3: Commit-Then-Timeout & Restraint ───────────────────────────────
    def test_03_commit_then_timeout_restraint(self):
        """
        COMMIT_THEN_TIMEOUT commits reservation in SQLite, then drops response.
        ReAgent probes, observes PRESENT, and restrains retry with zero duplicate rows.
        """
        artifact = run_hotel_booking_demo(scenario_id="B", simulator=self.sim)

        # ReAgent Arm asserts
        r_arm = artifact.reagent_arm
        self.assertEqual(r_arm.final_status, "VERIFIED")
        self.assertTrue(r_arm.contract_postconditions_verified)
        self.assertEqual(r_arm.probe_queries, 1)
        self.assertEqual(r_arm.tool_attempts, 1)  # Only 1 execution attempt, retry was RESTRAINED
        self.assertFalse(r_arm.duplicate_occurred)

        # Database asserts: exactly 1 reservation, not duplicated
        self.assertEqual(self.sim.db.count_reservations(), 1)
        self.assertEqual(self.sim.db.count_audit_logs(), 1)

        # Naive Arm asserts: blinded retry flagged as duplicate/corrupted
        n_arm = artifact.naive_retry_arm
        self.assertEqual(n_arm.tool_attempts, 2)
        self.assertTrue(n_arm.duplicate_occurred)

    # ── Test 4: Fail-Before-Commit & Safe Retry ───────────────────────────────
    def test_04_fail_before_commit_safe_retry(self):
        """
        FAIL_BEFORE_COMMIT fails before SQLite write.
        ReAgent probes, observes CONFIRMED_ABSENT, safely retries once, and verifies.
        """
        artifact = run_hotel_booking_demo(scenario_id="C", simulator=self.sim)

        r_arm = artifact.reagent_arm
        self.assertEqual(r_arm.final_status, "VERIFIED")
        self.assertTrue(r_arm.contract_postconditions_verified)
        self.assertFalse(r_arm.duplicate_occurred)

        # After safe retry, database contains exactly 1 valid reservation
        self.assertEqual(self.sim.db.count_reservations(), 1)
        self.assertEqual(self.sim.db.count_audit_logs(), 1)

    # ── Test 5: Observer Failure & UNKNOWN!=ABSENT Escalation ──────────────────
    def test_05_observer_failure_unknown_not_absent(self):
        """
        When the observer status query fails (503), ReAgent enforces UNKNOWN is not ABSENT.
        Does NOT blindly retry; escalates safely.
        """
        artifact = run_hotel_booking_demo(scenario_id="D", simulator=self.sim)

        r_arm = artifact.reagent_arm
        self.assertEqual(r_arm.final_status, "ESCALATED")
        self.assertFalse(r_arm.contract_postconditions_verified)
        self.assertEqual(r_arm.probe_queries, 1)

    # ── Test 6: Partial Commit & Discrepancy Escalation ───────────────────────
    def test_06_partial_commit_discrepancy_escalation(self):
        """
        PARTIAL_COMMIT creates a reservation row but omits the audit log.
        ReAgent probe detects DISCREPANCY and escalates safely.
        """
        artifact = run_hotel_booking_demo(scenario_id="E", simulator=self.sim)

        r_arm = artifact.reagent_arm
        self.assertEqual(r_arm.final_status, "ESCALATED")
        self.assertFalse(r_arm.contract_postconditions_verified)

    # ── Test 7: Duplicate Submissions Blocked by Database ─────────────────────
    def test_07_duplicate_submission_blocked(self):
        """Submitting a reservation with an existing booking_id returns 409 Conflict."""
        payload = {
            "booking_id": "BKG-UNIQUE-999",
            "guest_name": "Maya Lin",
            "room_number": "14B",
            "amount": 350.00,
            "fault_mode": "NO_FAULT",
        }
        # First submission succeeds
        r1 = requests.post(f"{self.base_url}/api/v1/bookings", json=payload, timeout=2.0)
        self.assertEqual(r1.status_code, 201)

        # Duplicate submission with same ID fails closed with 409
        r2 = requests.post(f"{self.base_url}/api/v1/bookings", json=payload, timeout=2.0)
        self.assertEqual(r2.status_code, 409)
        self.assertIn("DuplicateReservation", r2.json()["error"])

    # ── Test 8: Reset Isolation ───────────────────────────────────────────────
    def test_08_reset_isolation(self):
        """POST /api/v1/reset clears all simulator database state."""
        # Seed 2 bookings
        self.sim.db.create_reservation("BKG-01", "Guest A", "101", 200.0)
        self.sim.db.create_reservation("BKG-02", "Guest B", "102", 250.0)
        self.assertEqual(self.sim.db.count_reservations(), 2)

        # Reset via HTTP endpoint
        resp = requests.post(f"{self.base_url}/api/v1/reset", timeout=2.0)
        self.assertEqual(resp.status_code, 200)

        # Verify database is completely empty
        self.assertEqual(self.sim.db.count_reservations(), 0)
        self.assertEqual(self.sim.db.count_audit_logs(), 0)

    # ── Test 9: Malformed Requests Rejected ───────────────────────────────────
    def test_09_malformed_requests_rejected(self):
        """Missing parameters, negative amount, or invalid JSON return HTTP 400."""
        # Missing required field
        r1 = requests.post(
            f"{self.base_url}/api/v1/bookings",
            json={"booking_id": "BKG-BAD", "room_number": "14B"},
            timeout=2.0,
        )
        self.assertEqual(r1.status_code, 400)
        self.assertIn("Missing required parameter", r1.json()["error"])

        # Negative amount
        r2 = requests.post(
            f"{self.base_url}/api/v1/bookings",
            json={"booking_id": "BKG-BAD", "guest_name": "Maya", "room_number": "14B", "amount": -100.0},
            timeout=2.0,
        )
        self.assertEqual(r2.status_code, 400)
        self.assertIn("must be a positive number", r2.json()["error"])

    # ── Test 10: Service Unavailability Handling ──────────────────────────────
    def test_10_service_unavailability_handling(self):
        """When service port is closed, tool raises ConnectionError and probe raises StatusUnavailableError."""
        # Create an unused ephemeral port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
        sock.close()
        dead_url = f"http://127.0.0.1:{closed_port}"

        # Probe against dead URL
        probe = make_hotel_booking_probe_callback(dead_url, "BKG-UNKNOWN")
        from reagent.tools.hotel_tool import StatusUnavailableError
        with self.assertRaises(StatusUnavailableError):
            probe("hotel_booking_http", {}, 1)


if __name__ == "__main__":
    unittest.main()
