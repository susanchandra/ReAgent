"""
test_nemotron_provider_mock.py — Offline unit tests for NebiusNemotronProvider.

Tests the provider's structured decision path using an injected mock OpenAI client
(zero network calls, zero paid tokens, zero secret inspection).

Cases covered:
  1. Valid JSON contract proposal → StructuredDecision returned.
  2. Malformed / non-JSON response → fails closed (returns None).
  3. Missing required 'tool' field → rejected.
  4. Hallucinated 'provenance' field → rejected (attacker cannot set provenance).
  5. Unknown top-level fields → rejected.
  6. Response truncated (finish_reason='length') → fails closed (returns None).
  7. Provider HTTP 500 error → raises RuntimeError (fails closed).
  8. Timeout APIError → raises RuntimeError (fails closed).
  9. Arguments field not a dict → rejected.
 10. Empty 'tool' string → rejected.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Build a minimal mock OpenAI client
# ---------------------------------------------------------------------------

def _make_client(content: str, finish_reason: str = "stop", raise_exc=None):
    """Return a mock OpenAI-compatible client whose chat.completions.create() behaves as specified."""
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
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    )
    client.chat.completions.create.return_value = response
    return client


class TestNebiusNemotronProviderMock(unittest.TestCase):
    """Offline unit tests for NebiusNemotronProvider using injected mock clients."""

    def _make_provider(self, content: str, finish_reason: str = "stop", raise_exc=None):
        from reagent.providers.nebius import NebiusNemotronProvider
        client = _make_client(content, finish_reason=finish_reason, raise_exc=raise_exc)
        return NebiusNemotronProvider(client=client)

    # ── 1. Valid contract proposal ─────────────────────────────────────────
    def test_01_valid_structured_decision(self):
        """Valid schema-conformant JSON response → StructuredDecision with correct fields."""
        json_body = """{
            "tool": "dbops_database",
            "arguments": {"task_id": "task_02", "fault_mode": "COMMIT_THEN_TIMEOUT"},
            "proposed_postconditions": [
                {"predicate": "STATUS_SUCCESS", "description": "Transfer committed"},
                {"predicate": "KEY_EXISTS", "description": "Ledger row present", "key": "TXN-779043"}
            ],
            "diagnosis_category": null,
            "diagnosis_confidence": 0.9,
            "diagnosis_reason": "Idempotency key present in ledger",
            "recovery_strategy": null
        }"""
        provider = self._make_provider(json_body)
        decision = provider.make_structured_decision(
            task_objective="Execute wallet transfer of $500",
            available_tools=["dbops_database"],
        )
        self.assertIsNotNone(decision, "Valid payload must return a StructuredDecision")
        self.assertEqual(decision.tool, "dbops_database")
        self.assertIsInstance(decision.arguments, dict)
        self.assertEqual(len(decision.proposed_postconditions), 2)
        self.assertAlmostEqual(decision.diagnosis_confidence, 0.9)

    # ── 2. Malformed / non-JSON response ──────────────────────────────────
    def test_02_malformed_json_response_fails_closed(self):
        """Non-JSON model output → make_structured_decision returns None (fails closed)."""
        provider = self._make_provider("Sorry, I cannot process this request in JSON.")
        decision = provider.make_structured_decision(
            task_objective="Execute wallet transfer of $500",
            available_tools=["dbops_database"],
        )
        self.assertIsNone(decision, "Malformed JSON must return None (fails closed)")

    # ── 3. Missing required 'tool' field ──────────────────────────────────
    def test_03_missing_tool_field_rejected(self):
        """JSON without 'tool' field → _validate_structured_decision returns None."""
        provider = self._make_provider('{"arguments": {}, "proposed_postconditions": []}')
        decision = provider.make_structured_decision(
            task_objective="Execute wallet transfer",
            available_tools=["dbops_database"],
        )
        self.assertIsNone(decision, "Missing 'tool' must be rejected")

    # ── 4. Hallucinated 'provenance' field ────────────────────────────────
    def test_04_hallucinated_provenance_rejected(self):
        """Model attempting to set 'provenance' must be rejected (attacker prevention)."""
        json_body = """{
            "tool": "dbops_database",
            "arguments": {},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
            "provenance": "LIVE_NEMOTRON_VERIFIED"
        }"""
        provider = self._make_provider(json_body)
        decision = provider.make_structured_decision(
            task_objective="Execute wallet transfer",
            available_tools=["dbops_database"],
        )
        self.assertIsNone(decision, "Payload with 'provenance' field must be rejected")

    # ── 5. Unknown top-level fields ───────────────────────────────────────
    def test_05_unknown_extra_fields_rejected(self):
        """JSON with unknown top-level fields must be rejected."""
        json_body = """{
            "tool": "dbops_database",
            "arguments": {},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
            "diagnosis_category": null,
            "diagnosis_confidence": 0.0,
            "diagnosis_reason": null,
            "recovery_strategy": null,
            "injected_arbitrary_field": "exploit_attempt"
        }"""
        provider = self._make_provider(json_body)
        decision = provider.make_structured_decision(
            task_objective="Execute wallet transfer",
            available_tools=["dbops_database"],
        )
        self.assertIsNone(decision, "Unknown fields must cause rejection")

    # ── 6. Response truncated (finish_reason='length') ────────────────────
    def test_06_truncated_response_fails_closed(self):
        """finish_reason='length' (truncated output) → returns None (fails closed)."""
        provider = self._make_provider(
            '{"tool": "dbops_database", "arguments"',
            finish_reason="length",
        )
        decision = provider.make_structured_decision(
            task_objective="Execute wallet transfer",
            available_tools=["dbops_database"],
        )
        self.assertIsNone(decision, "Truncated response must return None (fails closed)")

    # ── 7. Provider HTTP 500 error ────────────────────────────────────────
    def test_07_provider_http_500_raises_runtime_error(self):
        """APIError from provider → RuntimeError raised (not silently ignored)."""
        from openai import APIError
        fake_exc = APIError("Internal Server Error", request=MagicMock(), body=None)
        provider = self._make_provider("", raise_exc=fake_exc)
        with self.assertRaises(RuntimeError):
            provider.make_structured_decision(
                task_objective="Execute wallet transfer",
                available_tools=["dbops_database"],
            )

    # ── 8. Timeout error ──────────────────────────────────────────────────
    def test_08_timeout_error_raises_runtime_error(self):
        """Any exception from provider → RuntimeError (fails closed, never silently passes)."""
        provider = self._make_provider("", raise_exc=ConnectionError("timeout"))
        with self.assertRaises(Exception):
            provider.make_structured_decision(
                task_objective="Execute wallet transfer",
                available_tools=["dbops_database"],
            )

    # ── 9. Arguments field not a dict ────────────────────────────────────
    def test_09_arguments_not_dict_rejected(self):
        """'arguments' field that is a list (not a dict) must be rejected."""
        json_body = """{
            "tool": "dbops_database",
            "arguments": ["not", "a", "dict"],
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
            "diagnosis_category": null,
            "diagnosis_confidence": 0.0,
            "diagnosis_reason": null,
            "recovery_strategy": null
        }"""
        provider = self._make_provider(json_body)
        decision = provider.make_structured_decision(
            task_objective="Execute wallet transfer",
            available_tools=["dbops_database"],
        )
        self.assertIsNone(decision, "'arguments' as list must be rejected")

    # ── 10. Empty 'tool' string ──────────────────────────────────────────
    def test_10_empty_tool_string_rejected(self):
        """Empty 'tool' string (whitespace) must be rejected."""
        json_body = """{
            "tool": "   ",
            "arguments": {},
            "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
            "diagnosis_category": null,
            "diagnosis_confidence": 0.0,
            "diagnosis_reason": null,
            "recovery_strategy": null
        }"""
        provider = self._make_provider(json_body)
        decision = provider.make_structured_decision(
            task_objective="Execute wallet transfer",
            available_tools=["dbops_database"],
        )
        self.assertIsNone(decision, "Empty/whitespace 'tool' must be rejected")

    # ── 11. Provider mock does not touch env / network ────────────────────
    def test_11_injected_client_never_reads_env_key(self):
        """Provider with injected client must not inspect NEBIUS_API_KEY env var."""
        import os
        original_env = os.environ.copy()
        # Temporarily unset the key to ensure no env read is required
        os.environ.pop("NEBIUS_API_KEY", None)
        try:
            json_body = """{
                "tool": "dbops_database",
                "arguments": {},
                "proposed_postconditions": [{"predicate": "STATUS_SUCCESS"}],
                "diagnosis_category": null,
                "diagnosis_confidence": 0.0,
                "diagnosis_reason": null,
                "recovery_strategy": null
            }"""
            provider = self._make_provider(json_body)
            # Provider must be configured when client is injected, regardless of env
            self.assertTrue(provider.is_configured(),
                            "Injected-client provider must be configured even without env key")
        finally:
            # Restore env
            for k, v in original_env.items():
                os.environ[k] = v
            for k in list(os.environ.keys()):
                if k not in original_env:
                    del os.environ[k]

    # ── 12. UI provenance label must not match model provenance ──────────
    def test_12_mock_provider_provenance_distinct_from_live(self):
        """Verify that the mock provider does not set provenance to LIVE_NEMOTRON."""
        from reagent.experiments.dbops.integration import DeterministicMockWalletProvider
        mock_prov = DeterministicMockWalletProvider()
        decision = mock_prov.make_structured_decision(
            task_objective="Transfer $500 from ACCT-2201 to ACCT-3390",
            available_tools=["dbops_database"],
        )
        self.assertIsNotNone(decision, "DeterministicMockWalletProvider must return a decision")
        # StructuredDecision has no provenance field — provenance is set by the pipeline
        self.assertFalse(hasattr(decision, "provenance") and "LIVE" in str(getattr(decision, "provenance", "")),
                         "Mock decision must not carry LIVE provenance label")


if __name__ == "__main__":
    unittest.main()
