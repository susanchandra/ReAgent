"""
tests/test_phase6_capability_registry.py — Tests for Phase 6 Generic Capability Resolution

Verifies:
  1. CapabilityRegistry resolves natural paraphrases to the correct capability.
  2. Entity extraction pulls amount, room, IDs, recipient, guest.
  3. Unsupported actions (flights, pizza, crypto) fail closed with informative rejections.
  4. Ambiguous prompts flag ambiguity and provide clarification questions.
  5. Fixture notice is populated when entity values deviate from fixture defaults.
  6. Empty or excessively long prompts return structured rejections.
  7. Extensibility: registering a new capability dynamically works cleanly.
"""

import unittest
from reagent.capability_registry import (
    CapabilityRegistry,
    CapabilityDescriptor,
    ResolutionResult,
    CAPABILITY_REGISTRY,
)


class TestCapabilityRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = CAPABILITY_REGISTRY

    def test_hotel_canonical_phrase(self):
        res = self.registry.resolve("Book Room 14B at Grand Hotel for Guest Maya")
        self.assertTrue(res.resolved)
        self.assertEqual(res.matched, "hotel_booking")
        self.assertEqual(res.entities.get("room"), "14B")
        self.assertEqual(res.entities.get("guest"), "maya")
        self.assertIsNone(res.rejection_reason)

    def test_hotel_natural_paraphrases(self):
        phrases = [
            "Reserve room 14B for Maya",
            "Book a hotel room for Maya",
            "Reserve a room at Grand Hotel",
            "I need accommodation for Maya at the Grand Hotel",
        ]
        for p in phrases:
            res = self.registry.resolve(p)
            self.assertTrue(res.resolved, f"Failed to resolve: {p}")
            self.assertEqual(res.matched, "hotel_booking", f"Mismatched for: {p}")

    def test_wallet_canonical_and_paraphrases(self):
        # All phrases include a numeric amount OR the ACCT-3390 token (which contains digits).
        # "Transfer money to Asha" is intentionally excluded: it contains no digit, so the
        # fixture boundary guard correctly rejects it (consistent with test_task_validation_rejects_unsupported).
        phrases = [
            "Send ₹500 to Asha for lunch",
            "Transfer $500 to Asha",
            "Pay Asha ₹500",
            "Send 500 to Asha",
            "Send payment to Asha ACCT-3390",
        ]
        for p in phrases:
            res = self.registry.resolve(p)
            self.assertTrue(res.resolved, f"Failed to resolve: {p}")
            self.assertEqual(res.matched, "wallet_transfer", f"Mismatched for: {p}")
            self.assertIn("asha", res.entities.get("recipient", ""))

    def test_database_update_canonical_and_paraphrases(self):
        phrases = [
            "Update Order ORD-8821: Issue $250.00 refund for CUST-1042",
            "Issue refund for order ORD-8821",
            "Refund $250 for customer CUST-1042",
            "Process refund for order 8821",
        ]
        for p in phrases:
            res = self.registry.resolve(p)
            self.assertTrue(res.resolved, f"Failed to resolve: {p}")
            self.assertEqual(res.matched, "database_update", f"Mismatched for: {p}")

    def test_unsupported_flight_rejection(self):
        res = self.registry.resolve("Book a flight to India")
        self.assertFalse(res.resolved)
        self.assertIsNone(res.matched)
        self.assertIsNotNone(res.rejection_reason)
        self.assertIn("flight", res.rejection_reason.lower())
        self.assertIn("hotel booking", res.rejection_reason.lower())

    def test_unsupported_general_actions(self):
        unsupported = [
            "Order a pepperoni pizza for delivery",
            "Check my unread emails and delete spam",
            "Buy 10 shares of Apple stock",
            "What is the weather in Tokyo today?",
        ]
        for p in unsupported:
            res = self.registry.resolve(p)
            self.assertFalse(res.resolved, f"Should be unresolved: {p}")
            self.assertIsNone(res.matched)
            self.assertIsNotNone(res.rejection_reason)
            self.assertIn("Unsupported task", res.rejection_reason)

    def test_empty_and_whitespace_input(self):
        for raw in ["", "   ", "\t\n"]:
            res = self.registry.resolve(raw)
            self.assertFalse(res.resolved)
            self.assertIn("empty", res.rejection_reason.lower())

    def test_overlong_input(self):
        raw = "Book room " + ("14B " * 100)
        res = self.registry.resolve(raw)
        self.assertFalse(res.resolved)
        self.assertIn("exceeds maximum length", res.rejection_reason.lower())

    def test_fixture_notice_generated_on_deviating_values(self):
        # Specifying Room 12C instead of fixture 14B
        res = self.registry.resolve("Reserve room 12C at Grand Hotel for Maya")
        self.assertTrue(res.resolved)
        self.assertEqual(res.matched, "hotel_booking")
        self.assertIsNotNone(res.fixture_notice)
        self.assertIn("12C", res.fixture_notice)
        self.assertIn("14B", res.fixture_notice)

    def test_custom_capability_registration(self):
        custom_registry = CapabilityRegistry()
        desc = CapabilityDescriptor(
            name="cloud_storage_backup",
            label="Cloud Storage Backup",
            intent_verbs=frozenset({"backup", "archive", "snapshot", "sync"}),
            entity_keywords=frozenset({"bucket", "s3", "storage", "archive"}),
            fixture_description="Simulates snapshot backup of primary object store.",
            fixture_entities={"bucket": "prod-data"},
            fixture_notice="Simulated prod-data bucket backup.",
            supported_examples=["Backup storage bucket prod-data"],
        )
        custom_registry.register(desc)
        res = custom_registry.resolve("Snapshot archive of storage bucket prod-data")
        self.assertTrue(res.resolved)
        self.assertEqual(res.matched, "cloud_storage_backup")


import json
import threading
import urllib.parse
import urllib.request
from http.server import HTTPServer
from reagent.ui_server import DashboardHTTPHandler
from reagent.adapter_registry import ADAPTER_REGISTRY


class TestPhase6ServerIntegration(unittest.TestCase):
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

    def test_paraphrase_hotel_execution(self):
        phrase = "Reserve room 14B for Maya at Grand Hotel"
        url = self._url(f"/api/run_recovery_demo?scenario=B&task_input={urllib.parse.quote(phrase)}")
        data = self._post_json(url)
        self.assertEqual(data.get("task_template"), "hotel_booking")
        self.assertEqual(data.get("reagent_arm", {}).get("final_status"), "VERIFIED")

    def test_paraphrase_wallet_execution(self):
        phrase = "Transfer $500 to Asha"
        url = self._url(f"/api/run_recovery_demo?scenario=B&task_input={urllib.parse.quote(phrase)}")
        data = self._post_json(url)
        self.assertEqual(data.get("task_template"), "wallet_transfer")
        self.assertEqual(data.get("reagent_arm", {}).get("final_status"), "VERIFIED")

    def test_fixture_notice_included_in_response(self):
        phrase = "Reserve room 12C for Maya at Grand Hotel"
        url = self._url(f"/api/run_recovery_demo?scenario=B&task_input={urllib.parse.quote(phrase)}")
        data = self._post_json(url)
        self.assertIn("fixture_notice", data)
        self.assertIn("12C", data["fixture_notice"])

    def test_adapter_registry_dispatch(self):
        self.assertTrue(ADAPTER_REGISTRY.is_registered("hotel_booking"))
        self.assertTrue(ADAPTER_REGISTRY.is_registered("wallet_transfer"))
        self.assertTrue(ADAPTER_REGISTRY.is_registered("database_update"))


if __name__ == "__main__":
    unittest.main()

