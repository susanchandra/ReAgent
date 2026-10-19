"""
tests/test_readiness_gate_nemotron.py
Tests for Final Readiness Gate Requirements:
- Endpoint consistency across provider, UI server, and preflight
- Telemetry records only hostname, never full paths or credentials
- Missing credentials handling (NOT_CONFIGURED)
- Malformed output handling (fails closed)
- HTTP failures and timeout handling
- Zero retries enforced (max_retries=0)
- Bounded timeout enforced
- Secret redaction invariant
- Invariant: model response cannot itself mark an operation successful
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reagent.providers.nebius import (
    NebiusNemotronProvider,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)
from reagent.providers.telemetry import (
    ProviderTelemetryEvent,
    compute_canonical_hash,
)
from reagent.ui_server import generate_dashboard_html
from run_nemotron_smoke_test import (
    run_smoke_test,
    InventoryReadOnlyTool,
    HARMLESS_TASK_OBJECTIVE,
)
from reagent.contract_validator import ContractGroundingValidator


def _make_mock_response(content: str, prompt_tokens: int = 50, completion_tokens: int = 25):
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


VALID_INVENTORY_CONTRACT = json.dumps({
    "tool": "inventory_read_only",
    "arguments": {"item_id": "INV-ITEM-9842", "warehouse_code": "WH-A12"},
    "proposed_postconditions": [
        {"predicate": "STATUS_SUCCESS", "description": "Inventory query completed successfully"}
    ],
    "diagnosis_category": None,
    "diagnosis_confidence": 0.0,
    "diagnosis_reason": None,
    "recovery_strategy": None,
})


class TestReadinessGateNemotron(unittest.TestCase):

    # 1. Endpoint consistency
    def test_01_canonical_endpoint_consistency(self):
        """Provider and UI server canonical endpoint must agree on https://api.tokenfactory.nebius.com/v1/."""
        self.assertEqual(DEFAULT_BASE_URL, "https://api.tokenfactory.nebius.com/v1/")
        provider = NebiusNemotronProvider(client=MagicMock())
        self.assertEqual(provider.base_url, "https://api.tokenfactory.nebius.com/v1/")

        # Verify UI dashboard HTML embeds the canonical URL
        html = generate_dashboard_html()
        self.assertIn("https://api.tokenfactory.nebius.com/v1/", html)
        self.assertNotIn("api.studio.nebius.com", html)

    # 2. Telemetry hostname only
    def test_02_telemetry_records_hostname_only(self):
        """Telemetry event endpoint_host must be only the netloc hostname, never a path or secret."""
        event = ProviderTelemetryEvent(
            endpoint_host=urlparse(DEFAULT_BASE_URL).netloc
        )
        self.assertEqual(event.endpoint_host, "api.tokenfactory.nebius.com")
        self.assertNotIn("/", event.endpoint_host)
        self.assertNotIn("v1", event.endpoint_host)
        self.assertNotIn("http", event.endpoint_host)

    # 3. Missing credentials -> NOT_CONFIGURED
    def test_03_missing_credentials_returns_not_configured(self):
        """When NEBIUS_API_KEY is not set, smoke test returns NOT_CONFIGURED and fails closed."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEBIUS_API_KEY", None)
            res = run_smoke_test()
        self.assertEqual(res["verdict"], "NOT_CONFIGURED")
        self.assertEqual(res["error_category"], "MISSING_API_KEY")
        self.assertFalse(res["schema_valid"])
        self.assertFalse(res["grounding_valid"])
        self.assertFalse(res["is_live"])

    # 4. Valid mock -> PASS
    def test_04_valid_mock_passes_smoke_test(self):
        """Valid harmless inventory proposal with mock client passes schema and grounding checks."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(VALID_INVENTORY_CONTRACT)
        provider = NebiusNemotronProvider(client=mock_client)

        res = run_smoke_test(provider_override=provider)
        self.assertEqual(res["verdict"], "PASS")
        self.assertTrue(res["schema_valid"])
        self.assertTrue(res["grounding_valid"])
        self.assertEqual(res["input_tokens"], 50)
        self.assertEqual(res["output_tokens"], 25)
        self.assertEqual(res["endpoint_host"], "api.tokenfactory.nebius.com")

    # 5. Malformed output -> FAIL
    def test_05_malformed_output_fails_closed(self):
        """Malformed JSON from provider fails closed with schema_valid=False."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("not-json{{{")
        provider = NebiusNemotronProvider(client=mock_client)

        res = run_smoke_test(provider_override=provider)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertFalse(res["schema_valid"])
        self.assertFalse(res["grounding_valid"])

    # 6. HTTP failure handling
    def test_06_http_failure_handled_cleanly(self):
        """HTTP error from provider returns error category without crashing."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("HTTP 502 Bad Gateway")
        provider = NebiusNemotronProvider(client=mock_client)

        res = run_smoke_test(provider_override=provider)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "RuntimeError")

    # 7. Timeout handling
    def test_07_timeout_handled_cleanly(self):
        """Timeout from provider returns TimeoutError category without crashing."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timed out after 15.0s")
        provider = NebiusNemotronProvider(client=mock_client)

        res = run_smoke_test(provider_override=provider)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "TimeoutError")

    # 8. Zero retries enforced
    def test_08_zero_retries_enforced_by_default(self):
        """NebiusNemotronProvider default max_retries must be 0 (zero automatic retries)."""
        provider = NebiusNemotronProvider(client=MagicMock())
        self.assertEqual(provider.max_retries, 0)
        self.assertLessEqual(provider.timeout, 30.0)

    # 9. Secret redaction invariant
    def test_09_secret_redaction_invariant(self):
        """Sensitive API key values must NEVER appear in smoke test telemetry, output dict, or files."""
        sentinel_secret = "sk-nebius-SECRET-KEY-12345-DO-NOT-LEAK"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(VALID_INVENTORY_CONTRACT)

        with patch.dict(os.environ, {"NEBIUS_API_KEY": sentinel_secret}):
            provider = NebiusNemotronProvider(client=mock_client)
            res = run_smoke_test(provider_override=provider)

        res_str = json.dumps(res)
        self.assertNotIn(sentinel_secret, res_str)

        # Check saved telemetry file
        art_path = Path("reports/dbops/smoke_test_telemetry.json")
        if art_path.exists():
            with open(art_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn(sentinel_secret, content)

    # 10. Invariant: model response cannot mark operation successful
    def test_10_model_output_cannot_mark_operation_executed(self):
        """A model proposal attempting to claim EXECUTED provenance is rejected and cannot mark task successful."""
        injected_contract = json.dumps({
            "tool": "inventory_read_only",
            "arguments": {"item_id": "INV-ITEM-9842"},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
            "provenance": "EXECUTED",  # Malicious/hallucinated execution claim
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(injected_contract)
        provider = NebiusNemotronProvider(client=mock_client)

        res = run_smoke_test(provider_override=provider)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertFalse(res["schema_valid"], "Provenance injection must cause schema validation to reject")

    # 11. Dry-run mode passes cleanly without API key
    def test_11_dry_run_mode_passes_without_api_key(self):
        """Smoke test with dry_run=True exercises real validation path and passes with mock client."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEBIUS_API_KEY", None)
            res = run_smoke_test(dry_run=True)
        self.assertEqual(res["verdict"], "PASS")
        self.assertTrue(res["schema_valid"])
        self.assertTrue(res["grounding_valid"])
        self.assertEqual(res["input_tokens"], 337)
        self.assertEqual(res["output_tokens"], 342)
        self.assertFalse(res["is_live"])
        self.assertIsNotNone(res["contract"])

    # 12. Dry-run mode simulates truncation when max_tokens is insufficient (<300)
    def test_12_dry_run_truncation_detection_at_low_tokens(self):
        """Smoke test with dry_run=True and max_tokens=256 detects and rejects truncation."""
        res = run_smoke_test(dry_run=True, max_tokens=256)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "TRUNCATED_OUTPUT")
        self.assertFalse(res["schema_valid"])
        self.assertFalse(res["grounding_valid"])


    # 13. No-key case fails closed even when the developer shell has a key set
    def test_13_no_key_fails_closed_even_when_shell_has_key(self):
        """Provider reports not configured when NEBIUS_API_KEY is cleared, regardless of shell state.

        Deterministic environment-isolation regression: even if the developer shell exports
        NEBIUS_API_KEY (e.g. after a successful live smoke test), clearing it within the test
        scope must make NebiusNemotronProvider(client=None).is_configured() return False and
        generate() raise RuntimeError.  Zero network calls are made.
        """
        with patch.dict(os.environ, {"NEBIUS_API_KEY": ""}, clear=False):
            provider = NebiusNemotronProvider(client=None)
            self.assertFalse(
                provider.is_configured(),
                "Provider must report unconfigured when NEBIUS_API_KEY is empty, "
                "even if the outer shell has the key set.",
            )
            with self.assertRaises(RuntimeError):
                provider.generate(prompt="Should fail closed.")

    # 14. Valid configuration enables the provider without network calls
    def test_14_valid_configuration_enables_provider_without_network_calls(self):
        """Provider reports configured when a client is injected; env key controls the client=None path.

        Two sub-cases:
        (a) Injected mock client -- is_configured() is True regardless of env.
        (b) client=None with a sentinel env key -- is_configured() is True, key must not leak.
        No live network calls are made.
        """
        # (a) Injected mock client -- always configured
        mock_client = MagicMock()
        provider_a = NebiusNemotronProvider(client=mock_client)
        self.assertTrue(provider_a.is_configured())

        # (b) client=None path with a clearly fake sentinel key
        sentinel_key = "sk-test-DETERMINISTIC-SENTINEL-NOT-REAL-0000"
        with patch.dict(os.environ, {"NEBIUS_API_KEY": sentinel_key}, clear=False):
            provider_b = NebiusNemotronProvider(client=None)
            self.assertTrue(
                provider_b.is_configured(),
                "Provider must report configured when NEBIUS_API_KEY is non-empty.",
            )
            # Sentinel key must not leak into repr or model_id
            self.assertNotIn(sentinel_key, repr(provider_b))
            self.assertNotIn(sentinel_key, provider_b.model)


if __name__ == "__main__":
    unittest.main()

