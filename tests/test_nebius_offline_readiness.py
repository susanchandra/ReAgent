"""
test_nebius_offline_readiness.py — Offline test suite verifying NVIDIA Nemotron / Nebius integration.
Checks:
- Structured schema validation
- Chain-of-thought isolation
- Fail-closed error handling without any live network access
"""

import unittest
from reagent.providers.nebius import NebiusNemotronProvider
from reagent.models import FailureEvent
from reagent.taxonomy import FailureType
from reagent.experiments.ambiguity.nemotron_fixtures import (
    make_offline_nebius_client,
    OFFLINE_DECISION_PAYLOAD,
    OFFLINE_DIAGNOSIS_PAYLOAD,
    OFFLINE_RECOVERY_PAYLOAD,
)


class TestNebiusOfflineReadiness(unittest.TestCase):

    def test_structured_decision_parsing(self):
        client = make_offline_nebius_client(OFFLINE_DECISION_PAYLOAD)
        provider = NebiusNemotronProvider(client=client)
        decision = provider.make_structured_decision(
            task_objective="Write hello to output.txt",
            available_tools=["sandboxed_file"],
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.tool, "sandboxed_file")
        self.assertEqual(decision.arguments["path"], "output.txt")
        self.assertEqual(len(decision.proposed_postconditions), 1)

    def test_diagnosis_parsing(self):
        client = make_offline_nebius_client(OFFLINE_DIAGNOSIS_PAYLOAD)
        provider = NebiusNemotronProvider(client=client)
        failure = FailureEvent(
            task_id="t-1",
            step_index=1,
            tool_name="http_post",
            error_message="Gateway timeout",
        )
        diag = provider.diagnose_failure(failure)
        self.assertEqual(diag.failure_type, FailureType.TOOL_FAILURE)
        self.assertEqual(diag.provenance, "model")
        self.assertTrue(diag.is_transient)

    def test_recovery_selection_parsing(self):
        from reagent.models import FailureDiagnosis
        client = make_offline_nebius_client(OFFLINE_RECOVERY_PAYLOAD)
        provider = NebiusNemotronProvider(client=client)
        diag = FailureDiagnosis(
            failure_id="f-1",
            failure_type=FailureType.TOOL_FAILURE,
            root_cause="Timeout",
            confidence_score=0.9,
            is_agent_fault=False,
            is_transient=True,
        )
        rec = provider.select_recovery(diag, available_tools=["sandboxed_file"])
        self.assertEqual(rec.strategy.value, "RETRY")
        self.assertEqual(rec.provenance, "model")

    def test_fail_closed_on_malformed_json(self):
        client = make_offline_nebius_client("Not a JSON object")
        provider = NebiusNemotronProvider(client=client)
        decision = provider.make_structured_decision("Do task", ["sandboxed_file"])
        self.assertIsNone(decision)
        self.assertEqual(provider.last_validation_error, "unparseable_json")


if __name__ == "__main__":
    unittest.main()
