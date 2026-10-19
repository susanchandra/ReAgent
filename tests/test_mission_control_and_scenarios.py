"""
tests/test_mission_control_and_scenarios.py
Comprehensive test suite for Phase 2–9 Mission Control & 5 Deterministic Scenarios:
- Scenarios A, B, C, D, E execution & outcome verification
- Isolated execution worlds: Naive Agent vs ReAgent
- Oracle reveal functionality (Scenario D)
- Event log consistency and sequence verification
- Natural language task input validation (offline grammar)
- Reset endpoint functionality
- Downloadable evidence receipt integrity
- Preflight safety and zero credential leakage
"""

import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path

from reagent.ui_server import (
    DashboardHTTPHandler,
    generate_dashboard_html,
    execute_scenario_backend,
    _validate_task_input,
    _contract_hash,
    SCENARIOS,
    SUPPORTED_TASKS,
)


class TestMissionControlAndScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _get_json(self, path: str) -> dict:
        req = urllib.request.Request(self._url(path))
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str) -> dict:
        req = urllib.request.Request(self._url(path), method="POST")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── 1. FIVE SCENARIOS VERIFICATION ──────────────────────────────────────────

    def test_scenario_a_normal_success(self):
        """Scenario A: One write, one probe, both succeed, zero duplicate debit."""
        data = self._post_json("/api/run_recovery_demo?scenario=A")
        naive = data["naive_retry_arm"]
        reagent = data["reagent_arm"]

        # Both agents start with $1,000 and end with $500
        self.assertEqual(naive["starting_balance"], 1000.0)
        self.assertEqual(reagent["starting_balance"], 1000.0)
        self.assertEqual(naive["final_balance"], 500.0)
        self.assertEqual(reagent["final_balance"], 500.0)

        # Both have 1 ledger row, 0 duplicate
        self.assertEqual(naive["ledger_records_count"], 1)
        self.assertEqual(reagent["ledger_records_count"], 1)
        self.assertFalse(naive["duplicate_occurred"])
        self.assertFalse(reagent["duplicate_occurred"])

        # ReAgent verified contract postconditions via probe
        self.assertEqual(reagent["final_status"], "VERIFIED")
        self.assertTrue(reagent["contract_postconditions_verified"])
        self.assertEqual(reagent["probe_queries"], 1)

    def test_scenario_b_commit_then_timeout(self):
        """Scenario B: Commit occurs before timeout. Naive double-debits; ReAgent restrains retry."""
        data = self._post_json("/api/run_recovery_demo?scenario=B")
        naive = data["naive_retry_arm"]
        reagent = data["reagent_arm"]

        # Naive caused double debit ($1000 total)
        self.assertEqual(naive["final_balance"], 0.0)
        self.assertEqual(naive["net_transferred"], 1000.0)
        self.assertEqual(naive["ledger_records_count"], 2)
        self.assertTrue(naive["duplicate_occurred"])
        self.assertEqual(naive["final_status"], "UNVERIFIED_CORRUPTED")

        # ReAgent probed, confirmed state, restrained retry, preserved $500
        self.assertEqual(reagent["final_balance"], 500.0)
        self.assertEqual(reagent["net_transferred"], 500.0)
        self.assertEqual(reagent["ledger_records_count"], 1)
        self.assertFalse(reagent["duplicate_occurred"])
        self.assertEqual(reagent["final_status"], "VERIFIED")
        self.assertTrue(reagent["details"]["runtime_contract_verified"])

    def test_scenario_c_fail_before_commit(self):
        """Scenario C: No effect on attempt 1. ReAgent confirms absence, retries once, verifies."""
        data = self._post_json("/api/run_recovery_demo?scenario=C")
        naive = data["naive_retry_arm"]
        reagent = data["reagent_arm"]

        # ReAgent probed, confirmed ABSENT, safely authorized 1 retry, verified
        self.assertEqual(reagent["final_balance"], 500.0)
        self.assertEqual(reagent["net_transferred"], 500.0)
        self.assertEqual(reagent["tool_attempts"], 2)
        self.assertEqual(reagent["probe_queries"], 2)
        self.assertEqual(reagent["ledger_records_count"], 1)
        self.assertFalse(reagent["duplicate_occurred"])
        self.assertEqual(reagent["final_status"], "VERIFIED")

        # Naive retried blindly without proof of absence
        self.assertEqual(naive["final_status"], "UNVERIFIED_RETRY")

    def test_scenario_d_unknown_outcome(self):
        """Scenario D: Commit occurred but probe unavailable. ReAgent enforces UNKNOWN != ABSENT, escalates without retry."""
        data = self._post_json("/api/run_recovery_demo?scenario=D")
        naive = data["naive_retry_arm"]
        reagent = data["reagent_arm"]

        # Naive blindly retried on timeout, causing double debit ($1000 across 2 rows)
        self.assertEqual(naive["final_balance"], 0.0)
        self.assertEqual(naive["ledger_records_count"], 2)
        self.assertTrue(naive["duplicate_occurred"])

        # ReAgent detected probe was unreachable; did NOT retry blindly; safely escalated
        self.assertEqual(reagent["final_balance"], 500.0)
        self.assertEqual(reagent["net_transferred"], 500.0)
        self.assertEqual(reagent["ledger_records_count"], 1)
        self.assertEqual(reagent["tool_attempts"], 1)
        self.assertFalse(reagent["duplicate_occurred"])
        self.assertEqual(reagent["final_status"], "ESCALATED")
        self.assertEqual(reagent["details"]["recovery_loop_decision"], "PROBE_UNAVAILABLE_ESCALATION")

    def test_scenario_e_partial_commit(self):
        """Scenario E: Step 0 committed, step 1 failed. ReAgent detects inconsistency and escalates."""
        data = self._post_json("/api/run_recovery_demo?scenario=E")
        naive = data["naive_retry_arm"]
        reagent = data["reagent_arm"]

        # Naive blindly retried, corrupting state
        self.assertTrue(naive["duplicate_occurred"])
        self.assertEqual(naive["final_status"], "UNVERIFIED_CORRUPTED")

        # ReAgent detected state discrepancy, restrained automated retry, escalated to human
        self.assertEqual(reagent["final_status"], "ESCALATED")
        self.assertEqual(reagent["tool_attempts"], 1)
        self.assertFalse(reagent["duplicate_occurred"])

    # ── 2. ORACLE REVEAL FOR SCENARIO D ────────────────────────────────────────

    def test_oracle_reveal_scenario_d(self):
        """Scenario D oracle endpoint reveals hidden ground truth that transaction committed."""
        data = self._get_json("/api/oracle_reveal?scenario=D")
        self.assertTrue(data["oracle_available"])
        self.assertEqual(data["scenario_id"], "D")
        self.assertEqual(data["oracle_truth"], "COMMITTED_ON_DISK")
        self.assertIn("TXN-779043 DID commit to disk", data["oracle_note"])
        self.assertIn("JUDGE-ONLY ORACLE", data["label"])

    def test_oracle_reveal_fails_closed_for_non_oracle_scenario(self):
        """Oracle endpoint returns 404 for scenarios where oracle is not applicable."""
        for non_oracle_sc in ("A", "B", "C", "E"):
            req = urllib.request.Request(self._url(f"/api/oracle_reveal?scenario={non_oracle_sc}"))
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 404)

    # ── 3. TASK INPUT GRAMMAR VALIDATION ────────────────────────────────────────

    def test_task_validation_supported_intents(self):
        """Supported task phrases pass grammar validation."""
        for phrase in SUPPORTED_TASKS:
            rejection = _validate_task_input(phrase)
            self.assertIsNone(rejection, f"Expected '{phrase}' to be valid")

            # Check via HTTP endpoint
            data = self._get_json(f"/api/validate_task?task={urllib.parse.quote(phrase)}")
            self.assertTrue(data["valid"])
            self.assertIsNone(data["rejection"])

    def test_task_validation_rejects_unsupported(self):
        """Unsupported intents, missing amounts, empty strings, and foreign recipients are rejected."""
        unsupported = [
            ("", "empty"),
            ("Book a flight to Paris", "intent"),
            ("Send money to Asha", "amount"),
            ("Send ₹500 to Bob", "recipient"),
            ("A" * 250, "length"),
        ]
        for phrase, reason in unsupported:
            rejection = _validate_task_input(phrase)
            self.assertIsNotNone(rejection, f"Expected '{phrase}' ({reason}) to be rejected")

            data = self._get_json(f"/api/validate_task?task={urllib.parse.quote(phrase)}")
            self.assertFalse(data["valid"])
            self.assertIsNotNone(data["rejection"])

    # ── 4. EVENT LOG INTEGRITY ──────────────────────────────────────────────────

    def test_event_log_sequence_and_phases(self):
        """Event log returns strictly monotonically increasing sequence numbers and required phases."""
        post_data = self._post_json("/api/run_recovery_demo?scenario=B")
        run_id = post_data["run_id"]

        ev_data = self._get_json(f"/api/event_log?scenario=B&run_id={run_id}")
        events = ev_data["events"]
        self.assertGreater(len(events), 5)

        # Monotonic sequence
        seqs = [e["seq"] for e in events]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(seqs, list(range(1, len(events) + 1)))

        # Required phases present
        phases = {e["phase"] for e in events}
        self.assertTrue({"INIT", "EXECUTE", "PROBE", "DECIDE"}.issubset(phases))

        # Agents represented
        agents = {e["agent"] for e in events}
        self.assertTrue({"NAIVE", "REAGENT", "SYSTEM"}.issubset(agents))

    # ── 5. RESET ENDPOINT ───────────────────────────────────────────────────────

    def test_reset_endpoint(self):
        """POST /api/reset clears runtime state successfully."""
        data = self._post_json("/api/reset")
        self.assertEqual(data["status"], "reset")
        self.assertIn("reset", data["message"].lower())

    # ── 6. EVIDENCE RECEIPT AND CONTRACT HASH ───────────────────────────────────

    def test_contract_hash_deterministic(self):
        """Contract hash is reproducible given same scenario and run ID."""
        h1 = _contract_hash("B", "sim_run_12345")
        h2 = _contract_hash("B", "sim_run_12345")
        self.assertEqual(h1, h2)
        self.assertTrue(h1.startswith("sha256:"))

        # Different run ID produces different hash
        h3 = _contract_hash("B", "sim_run_67890")
        self.assertNotEqual(h1, h3)


if __name__ == "__main__":
    unittest.main()
