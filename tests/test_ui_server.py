"""
Unit tests for ReAgent M6 UI Server (reagent/ui_server.py).
Verifies:
  - HTML export produces correct self-contained output
  - Data normalization produces correct structure from benchmark records
  - All displayed metrics derive from actual benchmark records (no fabrication)
  - Export data and normalization data are consistent
  - Missing fields are handled without fabrication
  - Provenance is correctly labeled OFFLINE
  - New BenchmarkRunTrace enrichment fields are present and typed correctly
"""

import os
import json
import unittest
import tempfile
import shutil

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_ui_test_")
    _MODULE_PREV_LEDGER = os.environ.get("REAGENT_LEDGER_PATH")
    os.environ["REAGENT_LEDGER_PATH"] = os.path.join(_MODULE_TEMP_DIR, "ledger.json")


def tearDownModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    if _MODULE_PREV_LEDGER is not None:
        os.environ["REAGENT_LEDGER_PATH"] = _MODULE_PREV_LEDGER
    else:
        os.environ.pop("REAGENT_LEDGER_PATH", None)
    if _MODULE_TEMP_DIR and os.path.exists(_MODULE_TEMP_DIR):
        shutil.rmtree(_MODULE_TEMP_DIR, ignore_errors=True)


from reagent.benchmark import run_benchmark_scenarios, BenchmarkRunTrace, BenchmarkComparisonReport
from reagent.ui_server import generate_dashboard_html, _normalize_report, _status_meta


class TestStatusMeta(unittest.TestCase):
    """Tests that status metadata correctly maps to UI classes and labels."""

    def test_verified(self):
        m = _status_meta("VERIFIED")
        self.assertEqual(m["css"], "verified")
        self.assertEqual(m["label"], "VERIFIED")
        self.assertIn("icon", m)

    def test_escalated(self):
        m = _status_meta("ESCALATED")
        self.assertEqual(m["css"], "escalated")
        self.assertEqual(m["label"], "ESCALATED")

    def test_failed(self):
        m = _status_meta("FAILED")
        self.assertEqual(m["css"], "failed")

    def test_false_success(self):
        m = _status_meta("FALSE_SUCCESS")
        self.assertEqual(m["css"], "false-success")
        self.assertIn("FALSE", m["label"])

    def test_unknown(self):
        m = _status_meta("")
        self.assertEqual(m["css"], "unknown")

    def test_case_insensitive(self):
        m = _status_meta("verified")
        self.assertEqual(m["css"], "verified")


class TestNormalizeReport(unittest.TestCase):
    """Tests that _normalize_report produces a complete, accurate UI data structure."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_benchmark_scenarios()
        cls.data = _normalize_report(cls.report)

    def test_provenance_is_offline(self):
        self.assertIn("OFFLINE", self.data["provenance"])
        self.assertIn("DETERMINISTIC", self.data["provenance"])

    def test_provenance_note_present(self):
        self.assertIsInstance(self.data["provenance_note"], str)
        self.assertGreater(len(self.data["provenance_note"]), 10)

    def test_scenarios_evaluated_matches_report(self):
        self.assertEqual(self.data["scenarios_evaluated"], self.report.scenarios_evaluated)

    def test_reagent_verified_matches_report(self):
        self.assertEqual(self.data["reagent_verified_count"], self.report.reagent_verified_count)

    def test_reagent_escalated_matches_report(self):
        self.assertEqual(self.data["reagent_escalated_count"], self.report.reagent_escalated_count)

    def test_reagent_false_success_matches_report(self):
        self.assertEqual(self.data["reagent_false_success_count"], self.report.reagent_false_success_count)

    def test_baseline_false_success_matches_report(self):
        self.assertEqual(self.data["baseline_false_success_count"], self.report.baseline_false_success_count)

    def test_runs_count(self):
        self.assertEqual(len(self.data["runs"]), 10)

    def test_reagent_runs_have_strategy_label(self):
        reagent_runs = [r for r in self.data["runs"] if r["strategy"] == "reagent_diagnosis_guided"]
        for r in reagent_runs:
            self.assertEqual(r["strategy_label"], "ReAgent")

    def test_baseline_runs_have_strategy_label(self):
        baseline_runs = [r for r in self.data["runs"] if r["strategy"] == "naive_fixed_retry"]
        for r in baseline_runs:
            self.assertIn("Naive", r["strategy_label"])

    def test_each_run_has_status_meta(self):
        for r in self.data["runs"]:
            self.assertIn("status_meta", r)
            sm = r["status_meta"]
            self.assertIn("css", sm)
            self.assertIn("icon", sm)
            self.assertIn("label", sm)

    def test_scenario_ids_present(self):
        ids = {r["scenario_id"] for r in self.data["runs"] if r["strategy"] == "reagent_diagnosis_guided"}
        expected_ids = {
            "SCENARIO_1_TRANSIENT_TOOL_FAILURE",
            "SCENARIO_2_INVALID_MODEL_OUTPUT",
            "SCENARIO_3_INCORRECT_ACTION_RESULT",
            "SCENARIO_4_ENVIRONMENT_DEPENDENCY_FAILURE",
            "SCENARIO_5_AUTHENTICATION_SECURITY_ESCALATION",
        }
        self.assertEqual(ids, expected_ids)

    def test_run_has_required_fields(self):
        for r in self.data["runs"]:
            for field in ("scenario_id", "scenario_name", "fault_type", "strategy",
                          "attempts_executed", "final_status", "verified",
                          "duration_ms", "timeline_phases", "discrepancies"):
                self.assertIn(field, r, f"Field '{field}' missing from run {r.get('scenario_id', '?')}")

    def test_baseline_false_success_count_equals_one(self):
        """Regression: baseline must produce exactly one FALSE_SUCCESS on scenario 3."""
        self.assertEqual(self.data["baseline_false_success_count"], 1)

    def test_reagent_false_success_count_equals_zero(self):
        """Regression: ReAgent must never produce a FALSE_SUCCESS."""
        self.assertEqual(self.data["reagent_false_success_count"], 0)


class TestBenchmarkTraceEnrichment(unittest.TestCase):
    """Tests that M6 enrichment fields are correctly populated in BenchmarkRunTrace."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_benchmark_scenarios()
        cls.reagent_runs = [r for r in cls.report.runs if r.strategy == "reagent_diagnosis_guided"]
        cls.baseline_runs = [r for r in cls.report.runs if r.strategy == "naive_fixed_retry"]

    def test_reagent_runs_have_claimed_outcome(self):
        for r in self.reagent_runs:
            self.assertIsNotNone(r.claimed_outcome, f"claimed_outcome missing for {r.scenario_id}")
            self.assertIn("tool_reported_success", r.claimed_outcome)
            self.assertIn("tool_name", r.claimed_outcome)

    def test_reagent_runs_have_observed_outcome(self):
        for r in self.reagent_runs:
            self.assertIsNotNone(r.observed_outcome, f"observed_outcome missing for {r.scenario_id}")
            self.assertIn("verifier_verdict", r.observed_outcome)
            self.assertIn("evidence_source", r.observed_outcome)

    def test_reagent_runs_have_frozen_contract_summary(self):
        for r in self.reagent_runs:
            if not r.contract_applicable:
                self.assertIsNone(r.frozen_contract_summary)
                continue
            self.assertIsNotNone(r.frozen_contract_summary,
                                 f"frozen_contract_summary missing for {r.scenario_id}")
            fc = r.frozen_contract_summary
            self.assertIn("contract_id", fc)
            self.assertIn("provenance", fc)
            self.assertIn("predicates", fc)
            self.assertIsInstance(fc["predicates"], list)
            self.assertGreater(len(fc["predicates"]), 0)

    def test_reagent_runs_have_timeline_events(self):
        for r in self.reagent_runs:
            self.assertIsInstance(r.timeline_events, list)
            self.assertGreater(len(r.timeline_events), 0,
                               f"timeline_events empty for {r.scenario_id}")
            for ev in r.timeline_events:
                self.assertIn("phase", ev)
                self.assertIn("detail", ev)

    def test_reagent_runs_have_safe_receipt(self):
        for r in self.reagent_runs:
            if not r.contract_applicable:
                self.assertIsNone(r.safe_receipt)
                continue
            self.assertIsNotNone(r.safe_receipt, f"safe_receipt missing for {r.scenario_id}")
            sr = r.safe_receipt
            for key in ("receipt_id", "task_id", "tool", "final_status", "verified"):
                self.assertIn(key, sr)


    def test_safe_receipt_contains_no_secrets(self):
        """Verify safe_receipt does not contain raw error messages, model reasoning, or credential fields."""
        forbidden_keys = {"api_key", "token", "authorization", "password", "secret",
                          "raw_output", "model_reasoning", "exception_message"}
        for r in self.reagent_runs:
            if r.safe_receipt:
                keys = {k.lower() for k in r.safe_receipt.keys()}
                for fk in forbidden_keys:
                    self.assertNotIn(fk, keys,
                                     f"Forbidden key '{fk}' found in safe_receipt for {r.scenario_id}")

    def test_claimed_observed_agreement_for_pipeline_verified_runs(self):
        """
        For pipeline-cycle VERIFIED runs (Sc1, Sc2): tool claimed success and verifier agrees.
        Sc4 is excluded — it is VERIFIED via registry fallback, not execute_with_contract,
        so the primary tool's claimed_success is False while the run is still VERIFIED.
        """
        pipeline_verified = [
            r for r in self.reagent_runs
            if r.final_status == "VERIFIED"
            and r.scenario_id != "SCENARIO_4_ENVIRONMENT_DEPENDENCY_FAILURE"
        ]
        for r in pipeline_verified:
            self.assertTrue(r.claimed_outcome["tool_reported_success"],
                            f"Verified run {r.scenario_id} has claimed_success=False")
            self.assertEqual(r.observed_outcome["verifier_verdict"], "PASS",
                             f"Verified run {r.scenario_id} has verifier_verdict != PASS")


    def test_scenario3_false_success_correctly_labeled(self):
        """Scenario 3 baseline must be FALSE_SUCCESS; ReAgent must be ESCALATED or not VERIFIED."""
        sc3_reagent = next((r for r in self.reagent_runs if r.scenario_id == "SCENARIO_3_INCORRECT_ACTION_RESULT"), None)
        sc3_baseline = next((r for r in self.baseline_runs if r.scenario_id == "SCENARIO_3_INCORRECT_ACTION_RESULT"), None)
        self.assertIsNotNone(sc3_reagent)
        self.assertIsNotNone(sc3_baseline)
        self.assertNotEqual(sc3_reagent.final_status, "VERIFIED",
                            "Scenario 3 ReAgent should NOT be VERIFIED — required key is missing")
        self.assertEqual(sc3_baseline.final_status, "FALSE_SUCCESS",
                         "Scenario 3 baseline should be FALSE_SUCCESS")

    def test_timeline_events_no_secrets(self):
        """Timeline event details must not contain credential-like fields."""
        forbidden = {"api_key", "token", "authorization", "password", "bearer"}
        for r in self.reagent_runs:
            for ev in r.timeline_events:
                for k in ev.get("detail", {}).keys():
                    self.assertNotIn(k.lower(), forbidden,
                                     f"Forbidden key '{k}' in timeline event {ev['phase']} for {r.scenario_id}")

    def test_baseline_claimed_outcome_present(self):
        for r in self.baseline_runs:
            self.assertIsNotNone(r.claimed_outcome,
                                 f"claimed_outcome missing for baseline {r.scenario_id}")
            self.assertIn("tool_reported_success", r.claimed_outcome)

    def test_baseline_observed_outcome_has_no_verifier(self):
        for r in self.baseline_runs:
            self.assertIsNotNone(r.observed_outcome)
            src = r.observed_outcome.get("evidence_source", "")
            self.assertIn("no_verifier", src,
                          f"Baseline {r.scenario_id} evidence_source should indicate no verifier: got '{src}'")


class TestHTMLExport(unittest.TestCase):
    """Tests that HTML export produces valid, complete, self-contained output."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_benchmark_scenarios()
        cls.html = generate_dashboard_html(cls.report)

    def test_html_is_string(self):
        self.assertIsInstance(self.html, str)

    def test_html_length_reasonable(self):
        # A minimal complete UI should be at least 50KB
        self.assertGreater(len(self.html), 50_000)

    def test_html_is_complete(self):
        self.assertIn("<!DOCTYPE html>", self.html)
        self.assertIn("</html>", self.html)

    def test_report_json_embedded(self):
        self.assertIn("reagent_verified_count", self.html)
        self.assertIn("OFFLINE", self.html)

    def test_provenance_note_embedded(self):
        self.assertIn("OFFLINE", self.html)
        self.assertIn("DETERMINISTIC", self.html)

    def test_verified_count_correct(self):
        """The embedded REPORT JSON must reflect actual benchmark counts."""
        import re
        match = re.search(r'"reagent_verified_count":(\d+)', self.html)
        self.assertIsNotNone(match, "reagent_verified_count not found in HTML")
        self.assertEqual(int(match.group(1)), self.report.reagent_verified_count)

    def test_baseline_false_success_correct(self):
        import re
        match = re.search(r'"baseline_false_success_count":(\d+)', self.html)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), self.report.baseline_false_success_count)

    def test_no_external_cdn_dependencies(self):
        """Dashboard must not load external resources."""
        for cdn in ("cdn.jsdelivr", "unpkg.com", "cloudflare", "googleapis", "bootstrapcdn"):
            self.assertNotIn(cdn, self.html, f"External CDN dependency found: {cdn}")

    def test_no_api_keys_in_html(self):
        """HTML export must not contain API key patterns."""
        import re
        # Nebius keys start with v1. — ensure nothing matching that pattern is present
        self.assertNotRegex(self.html, r'\bv1\.[A-Za-z0-9+/=]{20,}')

    def test_three_tabs_present(self):
        self.assertIn('tab-runs', self.html)
        self.assertIn('tab-benchmark', self.html)
        self.assertIn('tab-method', self.html)

    def test_run_inspector_sections_present(self):
        for section_id in ('cvo-container', 'predicate-table-body', 'timeline-list', 'step-inspector', 'receipt-body'):
            self.assertIn(section_id, self.html, f"Section '{section_id}' missing from HTML")

    def test_method_and_limits_content(self):
        """Method & Limits tab must include honest limitation disclosures."""
        self.assertIn("FALSE_SUCCESS", self.html)
        self.assertIn("ESCALATED", self.html)
        self.assertIn("illustrative sample size", self.html)


    def test_html_export_to_file(self):
        """Test full file write round-trip."""
        out_dir = tempfile.mkdtemp(prefix="reagent_export_test_")
        try:
            out_path = os.path.join(out_dir, "test_export.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(self.html)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as fh:
                read_back = fh.read()
            self.assertEqual(read_back[:100], self.html[:100])
            self.assertEqual(read_back[-100:], self.html[-100:])
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_scenario_names_in_html(self):
        for name_fragment in ("Transient", "Invalid Model", "False-Success", "Dependency", "Authentication"):
            self.assertIn(name_fragment, self.html,
                          f"Expected scenario fragment '{name_fragment}' not in HTML")


class TestDataConsistency(unittest.TestCase):
    """Tests that displayed summary counts match underlying run records."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_benchmark_scenarios()
        cls.data = _normalize_report(cls.report)

    def test_reagent_verified_from_runs(self):
        """reagent_verified_count in report must equal count of VERIFIED runs in reagent strategy."""
        reagent_runs = [r for r in self.data["runs"] if r["strategy"] == "reagent_diagnosis_guided"]
        computed = sum(1 for r in reagent_runs if r["final_status"] == "VERIFIED")
        self.assertEqual(computed, self.data["reagent_verified_count"],
                         f"Computed {computed} != reported {self.data['reagent_verified_count']}")

    def test_reagent_escalated_from_runs(self):
        reagent_runs = [r for r in self.data["runs"] if r["strategy"] == "reagent_diagnosis_guided"]
        computed = sum(1 for r in reagent_runs if r["final_status"] == "ESCALATED")
        self.assertEqual(computed, self.data["reagent_escalated_count"])

    def test_baseline_false_success_from_runs(self):
        baseline_runs = [r for r in self.data["runs"] if r["strategy"] == "naive_fixed_retry"]
        computed = sum(1 for r in baseline_runs if r["final_status"] == "FALSE_SUCCESS")
        self.assertEqual(computed, self.data["baseline_false_success_count"])

    def test_run_count_is_double_scenarios(self):
        """Total runs = 2x scenarios (one reagent, one baseline per scenario)."""
        self.assertEqual(len(self.data["runs"]), self.data["scenarios_evaluated"] * 2)


class TestBenchmarkTraceEnrichmentVerifiedAgreement(unittest.TestCase):
    """
    Separate class for the claimed-vs-observed agreement test so it can be
    scoped correctly. Sc4 is VERIFIED via a registry-level fallback after the
    primary tool failed — so claimed_success can be False on a VERIFIED run.
    We only assert agreement for runs where the pipeline used execute_with_contract
    (Sc1, Sc2 are VERIFIED; Sc3 is ESCALATED; Sc5 is ESCALATED).
    """

    @classmethod
    def setUpClass(cls):
        cls.report = run_benchmark_scenarios()
        cls.reagent_runs = [r for r in cls.report.runs if r.strategy == "reagent_diagnosis_guided"]

    def test_pipeline_verified_runs_have_matching_claimed_and_observed(self):
        """
        For runs that used execute_with_contract and are VERIFIED:
        tool_reported_success should be True and verifier_verdict should be PASS.
        Scenario 4 uses a registry fallback (not execute_with_contract) so is excluded.
        """
        pipeline_verified = [
            r for r in self.reagent_runs
            if r.final_status == "VERIFIED"
            and r.scenario_id != "SCENARIO_4_ENVIRONMENT_DEPENDENCY_FAILURE"
        ]
        for r in pipeline_verified:
            self.assertTrue(
                r.claimed_outcome["tool_reported_success"],
                f"Pipeline-verified run {r.scenario_id} has claimed_success=False"
            )
            self.assertEqual(
                r.observed_outcome["verifier_verdict"], "PASS",
                f"Pipeline-verified run {r.scenario_id} verifier_verdict != PASS"
            )

    def test_sc4_fallback_verified_but_primary_failed(self):
        """
        Scenario 4 is VERIFIED via fallback after primary tool was unregistered.
        The claimed_outcome reflects the primary tool's failure (False),
        while the run is still marked VERIFIED.
        """
        sc4 = next((r for r in self.reagent_runs
                    if r.scenario_id == "SCENARIO_4_ENVIRONMENT_DEPENDENCY_FAILURE"), None)
        self.assertIsNotNone(sc4)
        self.assertEqual(sc4.final_status, "VERIFIED")
        # Primary tool claimed failure (it was unregistered)
        self.assertFalse(sc4.claimed_outcome["tool_reported_success"])
        # Observed outcome reflects the successful fallback
        self.assertEqual(sc4.observed_outcome["verifier_verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
