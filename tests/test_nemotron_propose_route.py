"""
tests/test_nemotron_propose_route.py
Phase 6 — Tests for /api/nemotron_propose POST endpoint in ui_server.py.

Tests use a fully injected mock HTTP client so NO live provider calls occur,
NO secrets are accessed, NO network traffic is generated, and NO pilot ledger
is written.

Coverage:
  01  Missing credentials -> NOT_CONFIGURED (safe fail)
  02  Injected valid mock  -> CONTRACT_VALIDATED, contract returned
  03  Injected malformed JSON -> SCHEMA_INVALID, no side effect
  04  Injected empty-tool mock -> SCHEMA_INVALID (empty tool rejected)
  05  Injected unknown-field mock -> SCHEMA_INVALID (unknown field rejected)
  06  Provider HTTP 500 -> ERROR returned, not crash
  07  Token counts recorded from mock usage object
  08  Secret never appears in response body (paranoid check)
  09  Model output cannot mark operation as executed (no provenance key)
  10  run_id present and unique on every call
  11  latency_ms is non-negative int
  12  is_live always False when key is absent
"""

from __future__ import annotations

import io
import json
import os
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reagent.providers.nebius import NebiusNemotronProvider


# ---------------------------------------------------------------------------
# Minimal fake HTTP infrastructure — avoids opening a real socket
# ---------------------------------------------------------------------------

class _FakeWFile:
    def __init__(self):
        self.written = b""
    def write(self, data):
        self.written += data
    def flush(self):
        pass


class _FakeRFile:
    def __init__(self):
        self.stream = io.BytesIO(b"")
    def makefile(self, *args, **kwargs):
        return self.stream


def _invoke_post(path: str) -> Dict[str, Any]:
    """
    Invoke DashboardHTTPHandler.do_POST for a given path.
    Returns the parsed JSON response body.
    """
    from reagent.ui_server import DashboardHTTPHandler

    handler = DashboardHTTPHandler.__new__(DashboardHTTPHandler)
    handler.path = path
    handler.headers = {}
    handler.wfile = _FakeWFile()
    handler.rfile = _FakeRFile()
    handler.server = MagicMock()

    def _send_response(code, message=None):
        handler._response_code = code
    def _send_header(k, v):
        pass
    def _end_headers():
        pass
    handler.send_response = _send_response
    handler.send_header = _send_header
    handler.end_headers = _end_headers

    handler.do_POST()

    raw = handler.wfile.written
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Mock client factories
# ---------------------------------------------------------------------------

def _make_completion(content: str, prompt_tokens: int = 42, completion_tokens: int = 88):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _client_returning(content: str, prompt_tokens: int = 42, completion_tokens: int = 88):
    client = MagicMock()
    client.chat.completions.create.return_value = _make_completion(
        content, prompt_tokens, completion_tokens
    )
    return client


VALID_DECISION_JSON = json.dumps({
    "tool": "dbops_read_only",
    "arguments": {"txn_id": "TXN-20261001-002"},
    "proposed_postconditions": [
        {"predicate": "STATUS_SUCCESS", "description": "Read completed"}
    ],
    "diagnosis_category": None,
    "diagnosis_confidence": 0.0,
    "diagnosis_reason": None,
    "recovery_strategy": None,
})

MALFORMED_DECISION_JSON = "not_valid_json{{{"

EMPTY_TOOL_JSON = json.dumps({
    "tool": "   ",   # whitespace-only — rejected
    "arguments": {},
    "proposed_postconditions": [],
    "diagnosis_category": None,
    "diagnosis_confidence": 0.0,
    "diagnosis_reason": None,
    "recovery_strategy": None,
})

UNKNOWN_FIELD_JSON = json.dumps({
    "tool": "dbops_read_only",
    "arguments": {},
    "proposed_postconditions": [],
    "diagnosis_category": None,
    "diagnosis_confidence": 0.0,
    "diagnosis_reason": None,
    "recovery_strategy": None,
    "EXTRA_INJECTED_FIELD": "malicious_value",
})

PROVENANCE_INJECTION_JSON = json.dumps({
    "tool": "dbops_read_only",
    "arguments": {},
    "proposed_postconditions": [],
    "diagnosis_category": None,
    "diagnosis_confidence": 0.0,
    "diagnosis_reason": None,
    "recovery_strategy": None,
    "provenance": "EXECUTED",
})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNemotronProposeRoute(unittest.TestCase):

    def setUp(self):
        """Reset the global rate limiter before each test to prevent cross-test bleed."""
        import reagent.ui_server as _ui
        from reagent.ui_server import DemoRateLimiter
        _ui._demo_rate_limiter = DemoRateLimiter()

    # --- 01 Missing credentials -> NOT_CONFIGURED -------------------------

    def test_01_missing_key_returns_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEBIUS_API_KEY", None)
            result = _invoke_post("/api/nemotron_propose")
        self.assertEqual(result["status"], "NOT_CONFIGURED")
        self.assertFalse(result.get("schema_valid", True))
        self.assertIsNone(result.get("grounding_valid"))
        self.assertFalse(result.get("is_live", True))

    # --- 02 Injected valid mock -> CONTRACT_VALIDATED ----------------------

    def test_02_valid_decision_returns_contract_validated(self):
        mock_client = _client_returning(VALID_DECISION_JSON)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        self.assertEqual(result["status"], "CONTRACT_VALIDATED")
        self.assertTrue(result["schema_valid"])
        self.assertTrue(result.get("grounding_valid"), "grounding_valid must be True for CONTRACT_VALIDATED")
        self.assertIn("contract", result)
        self.assertIsNotNone(result["contract"])
        self.assertEqual(result["contract"]["tool"], "dbops_read_only")
        self.assertTrue(result["is_live"])

    # --- 03 Malformed JSON -> SCHEMA_INVALID, no side effect ---------------

    def test_03_malformed_json_returns_schema_invalid(self):
        mock_client = _client_returning(MALFORMED_DECISION_JSON)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        self.assertEqual(result["status"], "SCHEMA_INVALID")
        self.assertFalse(result["schema_valid"])
        self.assertIsNone(result.get("contract"))

    # --- 04 Empty tool -> SCHEMA_INVALID -----------------------------------

    def test_04_empty_tool_rejected_as_schema_invalid(self):
        mock_client = _client_returning(EMPTY_TOOL_JSON)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        self.assertEqual(result["status"], "SCHEMA_INVALID")
        self.assertIsNone(result.get("contract"))

    # --- 05 Unknown field -> SCHEMA_INVALID --------------------------------

    def test_05_unknown_field_rejected_as_schema_invalid(self):
        mock_client = _client_returning(UNKNOWN_FIELD_JSON)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        self.assertEqual(result["status"], "SCHEMA_INVALID")
        self.assertIsNone(result.get("contract"))

    # --- 06 Provider raises exception -> ERROR, no crash -------------------

    def test_06_provider_exception_returns_error_not_crash(self):
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", side_effect=RuntimeError("Injected failure")):
                result = _invoke_post("/api/nemotron_propose")
        self.assertEqual(result["status"], "ERROR")
        self.assertFalse(result.get("schema_valid", True))
        self.assertIn("error", result)

    # --- 07 Token counts recorded from mock usage -------------------------

    def test_07_token_counts_recorded(self):
        mock_client = _client_returning(VALID_DECISION_JSON, prompt_tokens=77, completion_tokens=33)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        self.assertEqual(result["status"], "CONTRACT_VALIDATED")
        self.assertIn("input_tokens", result)
        self.assertIn("output_tokens", result)
        self.assertEqual(result["input_tokens"], 77)
        self.assertEqual(result["output_tokens"], 33)

    # --- 08 Secret never appears in response body -------------------------

    def test_08_secret_never_in_response(self):
        sentinel = "sk-SUPER-SECRET-SENTINEL-XYZ-9999"
        mock_client = _client_returning(VALID_DECISION_JSON)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": sentinel}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        result_str = json.dumps(result)
        self.assertNotIn(sentinel, result_str, "API key must NEVER appear in response body")

    # --- 09 Model provenance injection rejected ---------------------------

    def test_09_provenance_injection_rejected(self):
        mock_client = _client_returning(PROVENANCE_INJECTION_JSON)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        self.assertEqual(result["status"], "SCHEMA_INVALID")
        self.assertIsNone(result.get("contract"))

    # --- 10 run_id present and unique on each call ------------------------

    def test_10_run_id_present_and_unique(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEBIUS_API_KEY", None)
            r1 = _invoke_post("/api/nemotron_propose")
            r2 = _invoke_post("/api/nemotron_propose")
        self.assertIn("run_id", r1)
        self.assertIn("run_id", r2)
        self.assertNotEqual(r1["run_id"], r2["run_id"],
                            "Each call must produce a unique run_id")

    # --- 11 latency_ms is non-negative int --------------------------------

    def test_11_latency_ms_non_negative(self):
        mock_client = _client_returning(VALID_DECISION_JSON)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        self.assertIn("latency_ms", result)
        self.assertIsInstance(result["latency_ms"], int)
        self.assertGreaterEqual(result["latency_ms"], 0)

    # --- 12 is_live False when key absent ---------------------------------

    def test_12_is_live_false_when_key_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEBIUS_API_KEY", None)
            result = _invoke_post("/api/nemotron_propose")
        self.assertFalse(result.get("is_live", True),
                         "is_live must be False when API key is absent")

    # --- 13 Grounding gate rejects schema-valid contract with disallowed tool --

    def test_13_grounding_failed_for_disallowed_tool(self):
        """A schema-valid contract proposing an unknown tool triggers GROUNDING_FAILED."""
        disallowed_tool_json = json.dumps({
            "tool": "arbitrary_unknown_tool",
            "arguments": {"x": 1},
            "proposed_postconditions": [
                {"predicate": "STATUS_SUCCESS", "description": "Done"}
            ],
            "diagnosis_category": None,
            "diagnosis_confidence": 0.0,
            "diagnosis_reason": None,
            "recovery_strategy": None,
        })
        mock_client = _client_returning(disallowed_tool_json)
        provider = NebiusNemotronProvider(client=mock_client)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "sk-test-safe-mock"}):
            with patch("reagent.ui_server.NebiusNemotronProvider", return_value=provider):
                result = _invoke_post("/api/nemotron_propose")
        # Schema passes; grounding gate must reject
        self.assertEqual(result["status"], "GROUNDING_FAILED",
                         f"Expected GROUNDING_FAILED but got {result.get('status')}. "
                         f"failure_reason={result.get('failure_reason')}")
        self.assertTrue(result.get("schema_valid"), "Schema should still be valid")
        self.assertFalse(result.get("grounding_valid"), "Grounding must be False for disallowed tool")
        self.assertIn("failure_reason", result)
        self.assertIsNotNone(result["contract"], "Contract dict should still be present")


if __name__ == "__main__":
    unittest.main(verbosity=2)
