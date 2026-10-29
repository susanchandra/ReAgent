"""
tests/test_nemotron_evidence_record.py — Deterministic Tests for Nemotron Evidence Records

Verifies:
1. Secret and API key sanitization (never stored in records).
2. Honest attribution: fixtures vs. live inference.
3. Validation evidence capture (schema, grounding, status).
4. Missing metadata and failure handling.
5. Explicit 4-stage role separation.
6. Safe CLI execution refusal without explicit opt-in.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch

from reagent.evidence_record import (
    NemotronEvidenceRecord,
    create_evidence_record,
    save_sanitized_evidence,
    sanitize_evidence_payload,
    load_sanitized_evidence,
)


class TestNemotronEvidenceRecord(unittest.TestCase):

    def test_01_sanitization_removes_api_keys_and_secrets(self):
        """Sanitizer must redact sk- keys, Bearer tokens, and sensitive dictionary keys."""
        dirty_payload = {
            "tool": "hotel_booking_http",
            "arguments": {
                "booking_id": "BKG-123",
                "api_key": "sk-real-super-secret-key-1234567890",
                "auth_token": "secret_token_abc",
                "password": "my_password_xyz",
            },
            "headers": {
                "Authorization": "Bearer sk-live-token-abcdef12345678",
            },
            "proposed_postconditions": [
                {"predicate": "KEY_EXISTS", "target": "booking_id", "expected": "BKG-123"}
            ],
        }

        sanitized = sanitize_evidence_payload(dirty_payload)

        # Sensitive keys redacted
        self.assertEqual(sanitized["arguments"]["api_key"], "[REDACTED_SECRET]")
        self.assertEqual(sanitized["arguments"]["auth_token"], "[REDACTED_SECRET]")
        self.assertEqual(sanitized["arguments"]["password"], "[REDACTED_SECRET]")
        self.assertEqual(sanitized["headers"]["Authorization"], "[REDACTED_SECRET]")

        # Safe keys preserved intact
        self.assertEqual(sanitized["tool"], "hotel_booking_http")
        self.assertEqual(sanitized["arguments"]["booking_id"], "BKG-123")
        self.assertEqual(sanitized["proposed_postconditions"][0]["predicate"], "KEY_EXISTS")

    def test_02_fixture_cannot_be_labeled_as_genuine_live(self):
        """Integrity invariant: A deterministic fixture must NEVER be labeled as GENUINE_LIVE_INFERENCE."""
        with self.assertRaises(ValueError):
            NemotronEvidenceRecord(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                provider_name="nebius_nemotron",
                timestamp_utc=1700000000.0,
                iso_timestamp="2023-11-14T22:13:20Z",
                run_id="run_test",
                is_live=False,  # <--- Fixture
                invocation_mode="GENUINE_LIVE_INFERENCE",  # <--- Contradiction! Must raise!
                task_objective="Test task",
                workflow="hotel_booking",
                sanitized_proposal_output={},
                schema_valid=True,
                grounding_valid=True,
                contract_validation_status="CONTRACT_VALIDATED",
                failure_reason=None,
                latency_ms=10,
                token_usage={},
            )

    def test_03_live_attribution_honestly_distinguished(self):
        """Factory must automatically label live vs fixture invocation modes accurately."""
        rec_live = create_evidence_record(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            provider_name="nebius_nemotron",
            run_id="live_01",
            is_live=True,
            task_objective="Hotel task",
            workflow="hotel_booking",
            raw_proposal={"tool": "hotel_booking_http"},
            schema_valid=True,
            grounding_valid=True,
            contract_validation_status="CONTRACT_VALIDATED",
        )
        self.assertEqual(rec_live.invocation_mode, "GENUINE_LIVE_INFERENCE")
        self.assertTrue(rec_live.is_live)

        rec_fixture = create_evidence_record(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            provider_name="deterministic_mock",
            run_id="fix_01",
            is_live=False,
            task_objective="Hotel task",
            workflow="hotel_booking",
            raw_proposal={"tool": "hotel_booking_http"},
            schema_valid=True,
            grounding_valid=True,
            contract_validation_status="CONTRACT_VALIDATED",
        )
        self.assertEqual(rec_fixture.invocation_mode, "DETERMINISTIC_FIXTURE")
        self.assertFalse(rec_fixture.is_live)

    def test_04_validation_evidence_capture(self):
        """Record must faithfully capture validation status, schema_valid, and grounding_valid."""
        rec = create_evidence_record(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            provider_name="nebius_nemotron",
            run_id="val_01",
            is_live=True,
            task_objective="Hotel task",
            workflow="hotel_booking",
            raw_proposal={"tool": "hotel_booking_http"},
            schema_valid=True,
            grounding_valid=False,
            contract_validation_status="GROUNDING_FAILED",
            failure_reason="Generic STATUS_SUCCESS rejected",
            latency_ms=120,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        self.assertEqual(rec.contract_validation_status, "GROUNDING_FAILED")
        self.assertTrue(rec.schema_valid)
        self.assertFalse(rec.grounding_valid)
        self.assertEqual(rec.failure_reason, "Generic STATUS_SUCCESS rejected")
        self.assertEqual(rec.token_usage["prompt_tokens"], 100)

    def test_05_missing_metadata_handled_gracefully(self):
        """Missing or None metadata must default safely without crashing."""
        rec = create_evidence_record(
            model_id="",
            provider_name="",
            run_id="",
            is_live=False,
            task_objective="",
            workflow="",
            raw_proposal=None,
            schema_valid=False,
            grounding_valid=None,
            contract_validation_status="NOT_CONFIGURED",
        )
        self.assertEqual(rec.model_id, "unknown-model")
        self.assertEqual(rec.provider_name, "nebius_nemotron")
        self.assertIsNone(rec.sanitized_proposal_output)
        self.assertIsNone(rec.grounding_valid)
        self.assertEqual(rec.contract_validation_status, "NOT_CONFIGURED")

    def test_06_failure_handling_records_error(self):
        """Provider failures and unparseable responses must be captured honestly."""
        rec = create_evidence_record(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            provider_name="nebius_nemotron",
            run_id="err_01",
            is_live=True,
            task_objective="Hotel task",
            workflow="hotel_booking",
            raw_proposal=None,
            schema_valid=False,
            grounding_valid=None,
            contract_validation_status="SCHEMA_INVALID",
            failure_reason="JSONDecodeError: Unterminated string",
            latency_ms=45,
        )
        self.assertFalse(rec.schema_valid)
        self.assertEqual(rec.contract_validation_status, "SCHEMA_INVALID")
        self.assertIn("JSONDecodeError", rec.failure_reason)

    def test_07_four_stage_pipeline_roles_present(self):
        """Record must explicitly serialize the 4 distinct pipeline roles to prevent self-grading confusion."""
        rec = create_evidence_record(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            provider_name="nebius_nemotron",
            run_id="roles_01",
            is_live=True,
            task_objective="Hotel task",
            workflow="hotel_booking",
            raw_proposal={"tool": "hotel_booking_http"},
            schema_valid=True,
            grounding_valid=True,
            contract_validation_status="CONTRACT_VALIDATED",
        )
        roles = rec.role_separation
        self.assertIn("stage_1_model_proposal", roles)
        self.assertIn("stage_2_contract_validation", roles)
        self.assertIn("stage_3_observed_state", roles)
        self.assertIn("stage_4_final_verdict", roles)

        # Verify explanations state that model is advisory only
        self.assertIn("Advisory proposal only", roles["stage_1_model_proposal"])
        self.assertIn("UNKNOWN is never treated as ABSENT", roles["stage_4_final_verdict"])

    def test_08_save_sanitized_evidence_file_integrity(self):
        """Saving evidence record to disk produces sanitized, readable JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_evidence.json")
            rec = create_evidence_record(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                provider_name="nebius_nemotron",
                run_id="disk_01",
                is_live=True,
                task_objective="Book Room 14B",
                workflow="hotel_booking",
                raw_proposal={
                    "tool": "hotel_booking_http",
                    "secret_api_key": "sk-leak-test-12345",
                    "arguments": {"booking_id": "BKG-99"},
                },
                schema_valid=True,
                grounding_valid=True,
                contract_validation_status="CONTRACT_VALIDATED",
                endpoint_host="https://api.tokenfactory.nebius.com/v1?token=secret123",
            )

            saved = save_sanitized_evidence(rec, out_file)
            self.assertEqual(saved, out_file)
            self.assertTrue(os.path.exists(out_file))

            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Ensure secret was redacted
            self.assertNotIn("sk-leak-test-12345", content)
            self.assertNotIn("token=secret123", content)
            self.assertIn("[REDACTED_SECRET]", content)

            loaded = load_sanitized_evidence(out_file)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["model_id"], "nvidia/nemotron-3-super-120b-a12b")
            self.assertEqual(loaded["contract_validation_status"], "CONTRACT_VALIDATED")

    def test_09_cli_blocks_without_live_opt_in_flag(self):
        """CLI script must fail closed with exit code 1 if --live-run-opt-in is not specified."""
        import subprocess
        import sys

        cmd = [sys.executable, "scripts/record_live_evidence.py"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("--live-run-opt-in is required", res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()
