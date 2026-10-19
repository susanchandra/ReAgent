"""
test_dbops_pilot_analysis.py — Focused offline tests for M9 PilotEvidenceAnalyzer.

Tests:
  1.  All 6 samples NOT_RUN (empty ledger — file does not exist)
  2.  All 6 samples NOT_RUN (ledger exists but is empty)
  3.  All 6 valid gold-equivalent results (synthetic COMPLETED fixtures)
  4.  Partial results (3/6 run, 2 VALID_CONTRACT, 1 PROVIDER_ERROR)
  5.  One PROVIDER_ERROR sample
  6.  Ambiguous dispatch (DISPATCHED status without terminal)
  7.  Malformed JSON in ledger line → ValueError
  8.  Fingerprint mismatch → detected and reported in findings
  9.  Duplicate terminal sample IDs → detected
 10.  Unexpected sample ID in ledger → detected
 11.  Missing token telemetry → UNKNOWN/NOT_REPORTED (never 0)
 12.  Idempotent: run analyzer twice → identical substantive output
 13.  Overall gate verdict PASS when >= 2 tasks conservative-pass
 14.  Overall gate verdict FAIL when < 2 tasks conservative-pass
 15.  INDETERMINATE when ambiguous dispatch remains

Zero network calls. All fixtures use tempfile.mkdtemp(). Never mutates real ledger.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Module under test ──────────────────────────────────────────────────────────
from reagent.experiments.dbops.pilot_analysis import (
    PilotEvidenceAnalyzer,
    PilotLedgerReader,
    OUTCOME_NOT_RUN,
    OUTCOME_DISPATCHED_AMBIGUOUS,
    OUTCOME_PROVIDER_ERROR,
    OUTCOME_REFUSAL,
    OUTCOME_PARSE_ERROR,
    OUTCOME_SCHEMA_ERROR,
    OUTCOME_GROUNDING_ERROR,
    OUTCOME_INVALID_CONTRACT,
    OUTCOME_VALID_CONTRACT,
    EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_SCHEMA_FINGERPRINT,
    EXPECTED_PROMPT_HASHES,
    PILOT_ANALYZER_VERSION,
)

# ── Shared test fixtures ───────────────────────────────────────────────────────

TASK_IDS = [
    "task_01_refund_issue",
    "task_02_wallet_transfer",
    "task_05_order_cancellation",
]
SAMPLES_PER_TASK = 2

PLANNED_SAMPLE_IDS = [
    f"pilot_{t}_s{s}"
    for t in TASK_IDS
    for s in range(SAMPLES_PER_TASK)
]  # 6 entries

FROZEN_CONFIG: Dict[str, Any] = {
    "pilot_manifest_version":   "1.0.0-pilot-nemotron-offline-prepared",
    "task_ids":                 TASK_IDS,
    "target_model":             "nvidia/nemotron-3-super-120b-a12b",
    "temperature":              0.0,
    "max_tokens":               512,
    "samples_per_task":         SAMPLES_PER_TASK,
    "total_samples":            6,
    "schema_version":           "structured_decision_v1",
    "prompt_hashes":            EXPECTED_PROMPT_HASHES,
    "schema_fingerprint":       EXPECTED_SCHEMA_FINGERPRINT,
    "config_fingerprint":       EXPECTED_CONFIG_FINGERPRINT,
    "acceptance_criteria": {
        "per_sample": {
            "parsed_json_ok":                  True,
            "schema_valid":                    True,
            "grounding_valid":                 True,
            "unsupported_tables_or_columns":   [],
            "semantic_gold_agreement":         True,
        },
        "per_task_conservative_threshold":  "2 / 2",
        "per_task_relaxed_threshold":       "1 / 2",
        "overall_gate_threshold":           "2 / 3",
    },
    "stopping_conditions": {
        "hard_request_ceiling":          6,
        "zero_automatic_retries":        True,
        "fail_closed_on_error":          True,
        "fail_closed_on_refusal":        True,
        "fail_closed_on_malformed_json": True,
        "abort_on_config_drift":         True,
        "abort_on_budget_exhaustion":    True,
    },
}


def _make_completed_record(
    sample_id: str,
    task_id: str,
    semantic_gold_agreement: bool = True,
    schema_valid: bool = True,
    grounding_valid: bool = True,
    generation_parse_success: bool = True,
    is_live_call: bool = True,
    token_telemetry: Optional[Dict[str, int]] = None,
    config_fingerprint: str = EXPECTED_CONFIG_FINGERPRINT,
) -> Dict[str, Any]:
    """Build a synthetic COMPLETED ledger record (fixture only)."""
    idx = int(sample_id[-1])
    return {
        "sample_id":                       sample_id,
        "request_id":                      f"req_{sample_id}",
        "task_id":                         task_id,
        "sample_index":                    idx,
        "model_identifier":                "nvidia/nemotron-3-super-120b-a12b",
        "config_fingerprint":              config_fingerprint,
        "prompt_fingerprint":              EXPECTED_PROMPT_HASHES.get(task_id, "aabbcc"),
        "schema_fingerprint":              EXPECTED_SCHEMA_FINGERPRINT,
        "timestamp_utc":                   1700000000.0,
        "timestamp_iso":                   "2026-01-01T00:00:00+00:00",
        "is_live_call":                    is_live_call,
        "status":                          "COMPLETED",
        "latency_ms":                      450,
        "raw_response_artifact_path":      None,
        "generation_parse_success":        generation_parse_success,
        "schema_valid":                    schema_valid,
        "schema_error":                    None,
        "grounding_valid":                 grounding_valid,
        "grounding_errors":                [],
        "semantic_gold_agreement":         semantic_gold_agreement,
        "missing_required_postconditions": [],
        "extra_unsupported_predicates":    [],
        "unsupported_tables_or_columns":   [],
        "uniqueness_respected":            True,
        "invalid_contract_rejected":       not semantic_gold_agreement,
        "fallback_used":                   False,
        "rejection_reason":                None,
        "token_telemetry":                 token_telemetry,
        "cost_telemetry_usd":              None,
    }


def _make_terminal_record(
    sample_id: str,
    task_id: str,
    status: str,
    is_live_call: bool = True,
    config_fingerprint: str = EXPECTED_CONFIG_FINGERPRINT,
) -> Dict[str, Any]:
    """Build a synthetic non-COMPLETED terminal ledger record."""
    base = _make_completed_record(
        sample_id=sample_id,
        task_id=task_id,
        is_live_call=is_live_call,
        config_fingerprint=config_fingerprint,
    )
    base["status"]                   = status
    base["generation_parse_success"] = False
    base["schema_valid"]             = False
    base["grounding_valid"]          = False
    base["semantic_gold_agreement"]  = False
    return base


def _make_dispatched_record(
    sample_id: str,
    task_id: str,
    config_fingerprint: str = EXPECTED_CONFIG_FINGERPRINT,
) -> Dict[str, Any]:
    """Build a DISPATCHED ledger record (ambiguous — no terminal follows)."""
    return {
        "sample_id":         sample_id,
        "request_id":        f"req_{sample_id}_disp",
        "task_id":           task_id,
        "sample_index":      0,
        "model_identifier":  "nvidia/nemotron-3-super-120b-a12b",
        "config_fingerprint":config_fingerprint,
        "prompt_fingerprint":EXPECTED_PROMPT_HASHES.get(task_id, "aabbcc"),
        "schema_fingerprint":EXPECTED_SCHEMA_FINGERPRINT,
        "timestamp_utc":     1700000000.0,
        "timestamp_iso":     "2026-01-01T00:00:00+00:00",
        "is_live_call":      True,
        "status":            "DISPATCHED",
        "latency_ms":        0,
        "raw_response_artifact_path": None,
        "generation_parse_success":   False,
        "schema_valid":               False,
        "schema_error":               None,
        "grounding_valid":            False,
        "grounding_errors":           [],
        "semantic_gold_agreement":    False,
        "missing_required_postconditions": [],
        "extra_unsupported_predicates":    [],
        "unsupported_tables_or_columns":   [],
        "uniqueness_respected":            True,
        "invalid_contract_rejected":       False,
        "fallback_used":                   False,
        "rejection_reason":                None,
        "token_telemetry":                 None,
        "cost_telemetry_usd":              None,
    }


def _write_config(tmp_dir: str, cfg: Dict[str, Any] = FROZEN_CONFIG) -> Path:
    p = Path(tmp_dir) / "pilot_config_frozen.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _write_ledger(tmp_dir: str, records: List[Dict[str, Any]]) -> Path:
    p = Path(tmp_dir) / "pilot_ledger.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return p


# ── Test cases ─────────────────────────────────────────────────────────────────

class TestAllSamplesNotRun(unittest.TestCase):
    """Test 1: ledger file does not exist → all 6 samples NOT_RUN."""

    def test_no_ledger_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            ledger_path = Path(tmp) / "nonexistent_ledger.jsonl"
            self.assertFalse(ledger_path.exists())

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertEqual(report.n_not_run,          6)
            self.assertEqual(report.n_attempted,        0)
            self.assertEqual(report.n_gold_equiv,       0)
            self.assertEqual(report.overall_gate_verdict, "NOT_RUN")
            self.assertFalse(report.pilot_executed)
            self.assertIsNone(report.gold_equivalence_rate)
            for sr in report.sample_results:
                self.assertEqual(sr.outcome_category, OUTCOME_NOT_RUN)
            self.assertIn("NOT_COMPARABLE", report.comparability_warning)


class TestEmptyLedger(unittest.TestCase):
    """Test 2: ledger file exists but is empty → all 6 samples NOT_RUN."""

    def test_empty_ledger_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            ledger_path = _write_ledger(tmp, [])

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertTrue(report.ledger_exists)
            self.assertEqual(report.n_not_run, 6)
            self.assertEqual(report.overall_gate_verdict, "NOT_RUN")
            self.assertFalse(report.pilot_executed)


class TestAllSixValid(unittest.TestCase):
    """Test 3: all 6 COMPLETED + gold-equiv → gate PASS."""

    def test_all_six_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            records = [
                _make_completed_record(f"pilot_{t}_s{s}", t)
                for t in TASK_IDS
                for s in range(SAMPLES_PER_TASK)
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertEqual(report.n_gold_equiv,         6)
            self.assertEqual(report.n_schema_valid,       6)
            self.assertEqual(report.n_grounded,           6)
            self.assertEqual(report.n_parsed,             6)
            self.assertEqual(report.overall_gate_verdict, "PASS")
            self.assertTrue(report.pilot_executed)
            # All 3 tasks conservative-pass
            self.assertEqual(report.n_passing_tasks_conservative, 3)
            for ts in report.task_summaries:
                self.assertTrue(ts.conservative_pass)
            # Check gold equiv rate
            self.assertEqual(report.gold_equivalence_rate, "6/6 (100.0%)")


class TestPartialResults(unittest.TestCase):
    """Test 4: 3/6 run, 2 VALID_CONTRACT, 1 PROVIDER_ERROR, 3 NOT_RUN."""

    def test_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            # task_01: both samples completed + gold-equiv
            # task_02: s0 PROVIDER_ERROR, s1 NOT_RUN
            # task_05: both NOT_RUN
            records = [
                _make_completed_record("pilot_task_01_refund_issue_s0", "task_01_refund_issue"),
                _make_completed_record("pilot_task_01_refund_issue_s1", "task_01_refund_issue"),
                _make_terminal_record("pilot_task_02_wallet_transfer_s0", "task_02_wallet_transfer", "PROVIDER_ERROR"),
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertEqual(report.n_gold_equiv, 2)
            self.assertEqual(report.n_not_run,    3)
            self.assertEqual(report.n_attempted,  3)
            # task_01 passes conservative; task_02 and task_05 do not
            self.assertEqual(report.n_passing_tasks_conservative, 1)
            self.assertEqual(report.overall_gate_verdict, "FAIL")

            outcomes = {sr.sample_id: sr.outcome_category for sr in report.sample_results}
            self.assertEqual(outcomes["pilot_task_01_refund_issue_s0"], OUTCOME_VALID_CONTRACT)
            self.assertEqual(outcomes["pilot_task_01_refund_issue_s1"], OUTCOME_VALID_CONTRACT)
            self.assertEqual(outcomes["pilot_task_02_wallet_transfer_s0"], OUTCOME_PROVIDER_ERROR)
            self.assertEqual(outcomes["pilot_task_02_wallet_transfer_s1"], OUTCOME_NOT_RUN)
            self.assertEqual(outcomes["pilot_task_05_order_cancellation_s0"], OUTCOME_NOT_RUN)


class TestProviderError(unittest.TestCase):
    """Test 5: one PROVIDER_ERROR sample classified correctly."""

    def test_provider_error_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            records = [
                _make_terminal_record(
                    "pilot_task_01_refund_issue_s0",
                    "task_01_refund_issue",
                    "PROVIDER_ERROR",
                )
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            sr = next(r for r in report.sample_results if r.sample_id == "pilot_task_01_refund_issue_s0")
            self.assertEqual(sr.outcome_category, OUTCOME_PROVIDER_ERROR)
            self.assertEqual(sr.status, "PROVIDER_ERROR")
            self.assertTrue(report.pilot_executed)  # live call attempted


class TestAmbiguousDispatch(unittest.TestCase):
    """Test 6: DISPATCHED record without terminal → DISPATCHED_AMBIGUOUS, gate INDETERMINATE."""

    def test_ambiguous_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            records = [
                _make_dispatched_record(
                    "pilot_task_01_refund_issue_s0",
                    "task_01_refund_issue",
                )
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            ambig = next(
                r for r in report.sample_results
                if r.sample_id == "pilot_task_01_refund_issue_s0"
            )
            self.assertEqual(ambig.outcome_category, OUTCOME_DISPATCHED_AMBIGUOUS)
            self.assertEqual(report.n_dispatched_ambiguous, 1)
            self.assertEqual(report.overall_gate_verdict, "INDETERMINATE")


class TestMalformedLedgerLine(unittest.TestCase):
    """Test 7: malformed JSON in ledger → ValueError raised."""

    def test_malformed_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            bad_ledger  = Path(tmp) / "pilot_ledger.jsonl"
            bad_ledger.write_text('{"sample_id": "x"}\nNOT_JSON\n', encoding="utf-8")

            analyzer = PilotEvidenceAnalyzer()
            with self.assertRaises(ValueError):
                analyzer.analyze(config_path, bad_ledger)


class TestFingerprintMismatch(unittest.TestCase):
    """Test 8: ledger record has wrong config_fingerprint → detected in report."""

    def test_fingerprint_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            wrong_fp    = "00" * 32  # 64 hex chars of zeros
            records     = [
                _make_completed_record(
                    "pilot_task_01_refund_issue_s0",
                    "task_01_refund_issue",
                    config_fingerprint=wrong_fp,
                )
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertFalse(report.fingerprint_check.all_ok)
            self.assertTrue(len(report.fingerprint_check.mismatches) > 0)
            mismatch_text = " ".join(report.fingerprint_check.mismatches)
            self.assertIn(wrong_fp, mismatch_text)

    def test_config_file_fingerprint_mismatch(self):
        """Config file itself has wrong fingerprint."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_cfg = dict(FROZEN_CONFIG)
            bad_cfg["config_fingerprint"] = "deadbeef" * 8
            config_path = _write_config(tmp, bad_cfg)
            ledger_path = _write_ledger(tmp, [])

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertFalse(report.fingerprint_check.config_fingerprint_ok)
            self.assertFalse(report.fingerprint_check.all_ok)


class TestDuplicateSampleIds(unittest.TestCase):
    """Test 9: two terminal records with same sample_id → detected."""

    def test_duplicate_ids_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            records = [
                _make_completed_record("pilot_task_01_refund_issue_s0", "task_01_refund_issue"),
                _make_completed_record("pilot_task_01_refund_issue_s0", "task_01_refund_issue"),  # duplicate
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertIn("pilot_task_01_refund_issue_s0", report.duplicate_sample_ids_detected)


class TestUnexpectedSampleId(unittest.TestCase):
    """Test 10: sample_id in ledger that is not in planned list → detected."""

    def test_unexpected_sample_id_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            # Use a sample_id that is not in the 6 planned IDs
            records = [
                _make_completed_record("pilot_task_99_ghost_s0", "task_99_ghost"),
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertIn("pilot_task_99_ghost_s0", report.unexpected_sample_ids_detected)


class TestMissingTokenTelemetry(unittest.TestCase):
    """Test 11: absent token_telemetry → UNKNOWN/NOT_REPORTED, never 0."""

    def test_token_telemetry_none_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            # token_telemetry=None (default in _make_completed_record)
            records = [
                _make_completed_record(
                    "pilot_task_01_refund_issue_s0",
                    "task_01_refund_issue",
                    token_telemetry=None,
                )
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            sr = next(r for r in report.sample_results if r.sample_id == "pilot_task_01_refund_issue_s0")
            self.assertFalse(sr.token_telemetry_reported)
            self.assertIsNone(sr.token_telemetry)
            self.assertIn("NOT_REPORTED", report.token_telemetry_note)

    def test_token_telemetry_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            records = [
                _make_completed_record(
                    "pilot_task_01_refund_issue_s0",
                    "task_01_refund_issue",
                    token_telemetry={"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
                )
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            sr = next(r for r in report.sample_results if r.sample_id == "pilot_task_01_refund_issue_s0")
            self.assertTrue(sr.token_telemetry_reported)
            self.assertEqual(sr.token_telemetry["total_tokens"], 200)


class TestIdempotent(unittest.TestCase):
    """Test 12: running analyzer twice on the same ledger → identical substantive output."""

    def test_idempotent_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            records = [
                _make_completed_record(f"pilot_{t}_s{s}", t)
                for t in TASK_IDS
                for s in range(SAMPLES_PER_TASK)
            ]
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            r1 = analyzer.analyze(config_path, ledger_path)
            r2 = analyzer.analyze(config_path, ledger_path)

            # Substantive fields must match (exclude generated_at_utc which is a timestamp)
            self.assertEqual(r1.n_gold_equiv,                r2.n_gold_equiv)
            self.assertEqual(r1.n_not_run,                   r2.n_not_run)
            self.assertEqual(r1.overall_gate_verdict,        r2.overall_gate_verdict)
            self.assertEqual(r1.fingerprint_check.all_ok,    r2.fingerprint_check.all_ok)
            self.assertEqual(r1.planned_sample_ids,          r2.planned_sample_ids)
            self.assertEqual(
                [(s.sample_id, s.outcome_category) for s in r1.sample_results],
                [(s.sample_id, s.outcome_category) for s in r2.sample_results],
            )


class TestGateVerdictPass(unittest.TestCase):
    """Test 13: >= 2 tasks with all samples gold-equiv → PASS."""

    def test_gate_pass_two_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            # task_01 and task_02: both samples gold-equiv; task_05: both PROVIDER_ERROR
            records = []
            for t in ["task_01_refund_issue", "task_02_wallet_transfer"]:
                for s in range(2):
                    records.append(_make_completed_record(f"pilot_{t}_s{s}", t))
            for s in range(2):
                records.append(
                    _make_terminal_record(
                        f"pilot_task_05_order_cancellation_s{s}",
                        "task_05_order_cancellation",
                        "PROVIDER_ERROR",
                    )
                )
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertEqual(report.n_passing_tasks_conservative, 2)
            self.assertEqual(report.overall_gate_verdict, "PASS")


class TestGateVerdictFail(unittest.TestCase):
    """Test 14: < 2 tasks with all samples gold-equiv → FAIL."""

    def test_gate_fail_one_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            # Only task_01: both gold-equiv; task_02 and task_05: PROVIDER_ERROR
            records = []
            for s in range(2):
                records.append(
                    _make_completed_record(f"pilot_task_01_refund_issue_s{s}", "task_01_refund_issue")
                )
            for t in ["task_02_wallet_transfer", "task_05_order_cancellation"]:
                for s in range(2):
                    records.append(
                        _make_terminal_record(f"pilot_{t}_s{s}", t, "PROVIDER_ERROR")
                    )
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertEqual(report.n_passing_tasks_conservative, 1)
            self.assertEqual(report.overall_gate_verdict, "FAIL")


class TestGateVerdictIndeterminate(unittest.TestCase):
    """Test 15: ambiguous dispatch → INDETERMINATE gate."""

    def test_indeterminate_when_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            # All 6 would be gold-equiv, except one is stuck at DISPATCHED
            records = []
            for t in TASK_IDS:
                for s in range(2):
                    sid = f"pilot_{t}_s{s}"
                    if t == "task_01_refund_issue" and s == 0:
                        records.append(_make_dispatched_record(sid, t))
                    else:
                        records.append(_make_completed_record(sid, t))
            ledger_path = _write_ledger(tmp, records)

            analyzer = PilotEvidenceAnalyzer()
            report   = analyzer.analyze(config_path, ledger_path)

            self.assertEqual(report.n_dispatched_ambiguous, 1)
            self.assertEqual(report.overall_gate_verdict, "INDETERMINATE")


class TestJsonReportWritten(unittest.TestCase):
    """Verify write_json_report produces valid parseable JSON."""

    def test_json_report_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            ledger_path = _write_ledger(tmp, [])

            analyzer   = PilotEvidenceAnalyzer()
            report     = analyzer.analyze(config_path, ledger_path)
            json_path  = Path(tmp) / "pilot_analysis_report.json"
            analyzer.write_json_report(report, json_path)

            self.assertTrue(json_path.exists())
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["analyzer_version"], PILOT_ANALYZER_VERSION)
            self.assertEqual(data["n_not_run"], 6)
            self.assertEqual(data["overall_gate_verdict"], "NOT_RUN")


class TestMarkdownReportWritten(unittest.TestCase):
    """Verify write_markdown_report produces a non-empty Markdown file."""

    def test_markdown_report_contains_key_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            ledger_path = _write_ledger(tmp, [])

            analyzer  = PilotEvidenceAnalyzer()
            report    = analyzer.analyze(config_path, ledger_path)
            md_path   = Path(tmp) / "pilot_analysis_report.md"
            analyzer.write_markdown_report(report, md_path)

            self.assertTrue(md_path.exists())
            content = md_path.read_text(encoding="utf-8")
            self.assertIn("DBOPS Nemotron Pilot", content)
            self.assertIn("Fingerprint Integrity", content)
            self.assertIn("Overall Metrics", content)
            self.assertIn("NOT_COMPARABLE", content)
            self.assertIn("NOT_REPORTED", content)


class TestAnalyzerVersion(unittest.TestCase):
    """Verify PILOT_ANALYZER_VERSION is set correctly."""

    def test_version_constant(self):
        self.assertEqual(PILOT_ANALYZER_VERSION, "1.0.0")


class TestComparabilityWarning(unittest.TestCase):
    """Verify comparability warning appears in report and JSON output."""

    def test_not_comparable_in_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(tmp)
            ledger_path = _write_ledger(tmp, [])

            analyzer  = PilotEvidenceAnalyzer()
            report    = analyzer.analyze(config_path, ledger_path)
            json_path = Path(tmp) / "pilot_analysis_report.json"
            analyzer.write_json_report(report, json_path)

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("NOT_COMPARABLE", data["comparability_warning"])
            self.assertIn("SYNTHETIC_OFFLINE_FIXTURES", data["comparability_warning"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
