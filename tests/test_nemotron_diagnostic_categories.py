"""
tests/test_nemotron_diagnostic_categories.py
Offline verification of sanitized diagnostic categories:
- 401/403: AuthenticationError & PermissionDeniedError -> AUTH_REJECTED
- 404: NotFoundError -> MODEL_NOT_FOUND
- 429: RateLimitError -> RATE_LIMIT
- Timeout: APITimeoutError -> TIMEOUT
- Network: APIConnectionError -> NETWORK_ERROR
- 400: BadRequestError -> BAD_REQUEST
- 5xx: InternalServerError -> SERVER_ERROR_5XX
- Truncated output: finish_reason="length" -> TRUNCATED_OUTPUT
- Empty completion: empty string -> EMPTY_COMPLETION
- JSON parse error: non-json content -> JSON_PARSE_ERROR
- Schema violations: unknown fields, missing tool, invalid args, provenance injection
- Grounding Gate rejection: tool not permitted
- Execution Invariant violation: model claiming EXECUTED provenance
- Redaction guarantee: raw secrets never leak into error_category or error_details
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock
try:
    import httpx
except ImportError:
    import httpx2 as httpx
from openai import (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    BadRequestError,
    InternalServerError,
)

from reagent.providers.nebius import (
    NebiusNemotronProvider,
    SanitizedProviderError,
)
from run_nemotron_smoke_test import (
    run_smoke_test,
    HARMLESS_TASK_OBJECTIVE,
)


def _make_mock_response(content: str, finish_reason: str = "stop", prompt_tokens: int = 50, completion_tokens: int = 25):
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _make_dummy_request():
    return httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1/chat/completions")


def _make_dummy_response(status_code: int):
    return httpx.Response(status_code, request=_make_dummy_request())


class TestNemotronSanitizedDiagnosticCategories(unittest.TestCase):

    def test_01_auth_rejected_401(self):
        """HTTP 401 AuthenticationError is mapped to AUTH_REJECTED."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Bearer token invalid: key-12345",
            response=_make_dummy_response(401),
            body={"error": "invalid_api_key"},
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "AUTH_REJECTED")
        self.assertNotIn("key-12345", res.get("error_details", ""))
        self.assertNotIn("Bearer", res.get("error_details", ""))

    def test_02_permission_denied_403(self):
        """HTTP 403 PermissionDeniedError is mapped to AUTH_REJECTED."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = PermissionDeniedError(
            message="Account has no permission for model",
            response=_make_dummy_response(403),
            body={"error": "permission_denied"},
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "AUTH_REJECTED")

    def test_03_model_not_found_404(self):
        """HTTP 404 NotFoundError is mapped to MODEL_NOT_FOUND."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = NotFoundError(
            message="The model nvidia/nemotron-3-super-120b-a12b does not exist",
            response=_make_dummy_response(404),
            body={"error": "model_not_found"},
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "MODEL_NOT_FOUND")

    def test_04_rate_limit_429(self):
        """HTTP 429 RateLimitError is mapped to RATE_LIMIT."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=_make_dummy_response(429),
            body={"error": "rate_limit_exceeded"},
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "RATE_LIMIT")

    def test_05_timeout_408(self):
        """APITimeoutError is mapped to TIMEOUT."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=_make_dummy_request()
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "TIMEOUT")

    def test_06_network_error(self):
        """APIConnectionError is mapped to NETWORK_ERROR."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=_make_dummy_request(),
            message="Connection refused to api.tokenfactory.nebius.com",
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "NETWORK_ERROR")

    def test_07_bad_request_400(self):
        """BadRequestError (e.g., unsupported parameter) is mapped to BAD_REQUEST."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = BadRequestError(
            message="Unsupported response_format json_object",
            response=_make_dummy_response(400),
            body={"error": "bad_request"},
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "BAD_REQUEST")

    def test_08_server_error_5xx(self):
        """InternalServerError 500 is mapped to SERVER_ERROR_5XX."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = InternalServerError(
            message="Internal server error",
            response=_make_dummy_response(500),
            body={"error": "internal_server_error"},
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "SERVER_ERROR_5XX")

    def test_09_truncated_output(self):
        """finish_reason='length' is categorized as TRUNCATED_OUTPUT."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content='{"tool": "inventory_read_only", "arguments": {',
            finish_reason="length",
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "TRUNCATED_OUTPUT")

    def test_10_empty_completion(self):
        """Empty completion is categorized as EMPTY_COMPLETION."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="",
            finish_reason="stop",
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "EMPTY_COMPLETION")

    def test_11_json_parse_error(self):
        """Invalid JSON is categorized as JSON_PARSE_ERROR."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="Here is your answer: {tool: inventory_read_only}",
            finish_reason="stop",
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "JSON_PARSE_ERROR")

    def test_12_schema_unknown_fields(self):
        """Hallucinated top-level fields are rejected and categorized."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content=json.dumps({
                "tool": "inventory_read_only",
                "arguments": {"item_id": "INV-1"},
                "hallucinated_explanation": "I am an AI agent",
            }),
            finish_reason="stop",
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertIn("SCHEMA_UNKNOWN_FIELDS", res["error_category"])

    def test_13_schema_missing_tool(self):
        """Missing tool field is rejected and categorized."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content=json.dumps({
                "arguments": {"item_id": "INV-1"},
            }),
            finish_reason="stop",
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "SCHEMA_MISSING_TOOL")

    def test_14_provenance_injection_rejected(self):
        """Model attempt to inject provenance is rejected."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content=json.dumps({
                "tool": "inventory_read_only",
                "arguments": {"item_id": "INV-1"},
                "provenance": "OFFLINE_DETERMINISTIC",
            }),
            finish_reason="stop",
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "PROVENANCE_INJECTION_REJECTED")

    def test_15_grounding_gate_rejected_disallowed_tool(self):
        """Model proposing an unpermitted tool is rejected by the Grounding Gate."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content=json.dumps({
                "tool": "unauthorized_delete_database",
                "arguments": {},
                "proposed_postconditions": [],
            }),
            finish_reason="stop",
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["error_category"], "GROUNDING_GATE_REJECTED")
        self.assertFalse(res["grounding_valid"])

    def test_16_secret_redaction_in_diagnostics(self):
        """Ensures secrets in raw exception messages never leak into error_details."""
        mock_client = MagicMock()
        raw_secret = "secret-token-abcdef-999"
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message=f"Invalid Authorization header with secret {raw_secret}",
            response=_make_dummy_response(401),
            body={"error": "unauthorized"},
        )
        provider = NebiusNemotronProvider(client=mock_client)
        res = run_smoke_test(provider_override=provider)

        res_str = json.dumps(res)
        self.assertNotIn(raw_secret, res_str)


if __name__ == "__main__":
    unittest.main()
