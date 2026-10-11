"""
nemotron_fixtures.py — Offline fixtures and mock clients for NVIDIA Nemotron / Nebius Token Factory.

Ensures NebiusNemotronProvider can be exercised completely offline without
network access or API keys, testing:
1. Strict schema validation
2. Chain-of-thought stripping
3. Fail-closed behavior on malformed or truncated responses
"""

from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock
from reagent.providers.nebius import NebiusNemotronProvider


def make_offline_nebius_client(
    response_content: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 150,
    completion_tokens: int = 50,
) -> MagicMock:
    """Creates a mock OpenAI client returning a synthetic Nemotron response."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_content
    mock_choice.finish_reason = finish_reason

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = prompt_tokens + completion_tokens

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


OFFLINE_DECISION_PAYLOAD = """{
  "tool": "sandboxed_file",
  "arguments": {"action": "write", "path": "output.txt", "content": "hello"},
  "proposed_postconditions": [
    {"predicate": "STATUS_SUCCESS", "description": "Write succeeded"}
  ],
  "diagnosis_category": null,
  "diagnosis_confidence": 0.0,
  "diagnosis_reason": null,
  "recovery_strategy": null
}"""

OFFLINE_DIAGNOSIS_PAYLOAD = """{
  "failure_type": "TOOL_FAILURE",
  "root_cause": "Remote service timed out before response was received",
  "confidence_score": 0.95,
  "is_agent_fault": false,
  "is_transient": true
}"""

OFFLINE_RECOVERY_PAYLOAD = """{
  "strategy": "RETRY",
  "rationale": "State restraint confirmed absence of side effect; safe to retry",
  "modified_input": {},
  "fallback_tool_name": null,
  "backoff_seconds": 0.5
}"""
