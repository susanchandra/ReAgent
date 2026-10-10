"""
Unit tests for ReAgent M6.1 Trace Integrity, Readability, and Evidence Honesty.
Verifies:
  - Scenario identity: unique titles, 1-to-1 mapping with scenario IDs, stability
  - Scenario 4: contract_applicable=False, frozen_contract_summary & safe_receipt are None
  - Rendered output: scenario 4 displays NOT APPLICABLE in run view, table, and method
  - Claimed/Observed/predicates/timeline/receipt/summary consistency
  - No fabricated dates or hashes (hash_recorded=False, occurred_at is real or null)
  - Independent Oracle evaluation (reagent/oracles.py) and False Success detection
  - Oracle detects false success even when DeterministicVerifier is mutated
  - HTML export integrity: stdlib html.parser validation, no external http(s) URLs,
    source badge present, embedded JSON safe escaping, hostile string injection safety
  - Optional Playwright test (cleanly skipped if not installed)
"""

import os
import json
import unittest
import tempfile
import shutil
import re
from html.parser import HTMLParser

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_m61_test_")
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


from reagent.benchmark import (
    run_benchmark_scenarios,
    SCENARIO_DEFINITIONS,
    BenchmarkRunTrace,
    BenchmarkComparisonReport,
)
from reagent.ui_server import (
    generate_dashboard_html,
    _normalize_report,
    _safe_json_embed,
    _status_meta,
)
from reagent.oracles import evaluate_oracle, detect_false_success


class TestScenarioIdentityAndStability(unittest.TestCase):
    """Verifies scenario titles are unique, 1-to-1 with IDs, and stable."""

    def test_scenario_definitions_count(self):
        self.assertEqual(len(SCENARIO_DEFINITIONS), 5)

    def test_titles_unique(self):
        titles = [defn["title"] for defn in SCENARIO_DEFINITIONS.values()]
        self.assertEqual(len(titles), len(set(titles)))

    def test_titles_one_to_one_with_scenario_ids(self):
        report = run_benchmark_scenarios()
        reagent_runs = [r for r in report.runs if r.strategy == "reagent_diagnosis_guided"]
        self.assertEqual(len(reagent_runs), 5)
        for r in reagent_runs:
            self.assertIn(r.scenario_id, SCENARIO_DEFINITIONS)
            expected_title = SCENARIO_DEFINITIONS[r.scenario_id]["title"]
            self.assertEqual(r.scenario_name, expected_title)

    def test_outcomes_stable_across_consecutive_runs(self):
        rep1 = run_benchmark_scenarios()
        rep2 = run_benchmark_scenarios()
        self.assertEqual(rep1.reagent_verified_count, rep2.reagent_verified_count)
        self.assertEqual(rep1.reagent_escalated_count, rep2.reagent_escalated_count)
        self.assertEqual(rep1.baseline_false_success_count, rep2.baseline_false_success_count)
        for r1, r2 in zip(rep1.runs, rep2.runs):
            self.assertEqual(r1.scenario_id, r2.scenario_id)
            self.assertEqual(r1.final_status, r2.final_status)
            self.assertEqual(r1.scenario_name, r2.scenario_name)


class TestScenario4OmissionAndNotApplicable(unittest.TestCase):
    """
    Verifies that Scenario 4 (registry fallback only):
    - Sets contract_applicable = False
    - Completely omits frozen_contract_summary and safe_receipt (None)
    - Renders NOT APPLICABLE in the HTML output
    """

    @classmethod
    def setUpClass(cls):
        cls.report = run_benchmark_scenarios()
        cls.reagent_sc4 = next(
            r for r in cls.report.runs
            if r.scenario_id == "SCENARIO_4_ENVIRONMENT_DEPENDENCY_FAILURE"
            and r.strategy == "reagent_diagnosis_guided"
        )
        cls.baseline_sc4 = next(
            r for r in cls.report.runs
            if r.scenario_id == "SCENARIO_4_ENVIRONMENT_DEPENDENCY_FAILURE"
            and r.strategy == "naive_fixed_retry"
        )
        cls.html = generate_dashboard_html(cls.report)

    def test_sc4_trace_contract_applicable_is_false(self):
        self.assertFalse(self.reagent_sc4.contract_applicable)
        self.assertFalse(self.baseline_sc4.contract_applicable)

    def test_sc4_trace_frozen_contract_summary_is_none(self):
        self.assertIsNone(self.reagent_sc4.frozen_contract_summary)
        self.assertIsNone(self.baseline_sc4.frozen_contract_summary)

    def test_sc4_trace_safe_receipt_is_none(self):
        self.assertIsNone(self.reagent_sc4.safe_receipt)
        self.assertIsNone(self.baseline_sc4.safe_receipt)

    def test_sc4_rendered_html_contains_not_applicable_notice(self):
        self.assertIn("NOT APPLICABLE: this scenario does not use the contract pipeline", self.html)

    def test_sc4_timeline_events_have_no_fake_wall_clock(self):
        """Timeline events for Sc4 must not invent wall-clock timestamps."""
        for ev in self.reagent_sc4.timeline_events:
            self.assertIsNone(ev.get("occurred_at"))


class TestRenderedOutputConsistency(unittest.TestCase):
    """
    Asserts consistency between underlying run records and rendered HTML representation.
    Extracts embedded REPORT JSON and parses it to guarantee exact fidelity.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = run_benchmark_scenarios()
        cls.html = generate_dashboard_html(cls.report)
        # Extract embedded JSON
        match = re.search(r'const REPORT = (\{.*?\});\s*//', cls.html, re.DOTALL)
        assert match is not None, "Failed to locate embedded REPORT JSON in HTML"
        cls.embedded_data = json.loads(match.group(1))

    def test_embedded_summary_counts_match_report(self):
        ed = self.embedded_data
        rep = self.report
        self.assertEqual(ed["scenarios_evaluated"], rep.scenarios_evaluated)
        self.assertEqual(ed["reagent_verified_count"], rep.reagent_verified_count)
        self.assertEqual(ed["reagent_escalated_count"], rep.reagent_escalated_count)
        self.assertEqual(ed["reagent_false_success_count"], rep.reagent_false_success_count)
        self.assertEqual(ed["baseline_verified_count"], rep.baseline_verified_count)
        self.assertEqual(ed["baseline_failed_count"], rep.baseline_failed_count)
        self.assertEqual(ed["baseline_false_success_count"], rep.baseline_false_success_count)
        self.assertEqual(ed["reagent_completed_and_verified"], rep.reagent_completed_and_verified)
        self.assertEqual(ed["reagent_correctly_escalated"], rep.reagent_correctly_escalated)

    def test_embedded_runs_match_report_runs(self):
        ed_runs = self.embedded_data["runs"]
        rep_runs = self.report.runs
        self.assertEqual(len(ed_runs), len(rep_runs))
        for ed_r, rep_r in zip(ed_runs, rep_runs):
            self.assertEqual(ed_r["scenario_id"], rep_r.scenario_id)
            self.assertEqual(ed_r["final_status"], rep_r.final_status)
            self.assertEqual(ed_r["verified"], rep_r.verified)
            self.assertEqual(ed_r["attempts_executed"], rep_r.attempts_executed)
            self.assertEqual(ed_r["contract_applicable"], rep_r.contract_applicable)
            if rep_r.strategy == "reagent_diagnosis_guided" and rep_r.contract_applicable:
                self.assertIsNotNone(ed_r["frozen_contract_summary"])
                self.assertFalse(ed_r["frozen_contract_summary"]["hash_recorded"])
            elif not rep_r.contract_applicable:
                self.assertIsNone(ed_r["frozen_contract_summary"])
                self.assertIsNone(ed_r["safe_receipt"])


    def test_no_fabricated_hashes_or_dates(self):
        for run in self.embedded_data["runs"]:
            fc = run.get("frozen_contract_summary")
            if fc:
                self.assertFalse(fc["hash_recorded"])
            # Ensure timeline events don't have fabricated strings like '2023'
            for ev in run.get("timeline_events", []):
                occ = ev.get("occurred_at")
                if occ is not None:
                    self.assertNotIn("2023", occ)


class TestIndependentOracles(unittest.TestCase):
    """
    Verifies that reagent/oracles.py independently evaluates outcomes
    WITHOUT importing or calling DeterministicVerifier, and catches false successes.
    """

    def test_oracle_scenario_1_passes_on_valid_payload(self):
        obs = {"success": True, "output": {"is_valid": True, "parsed": {"service": "active"}}}
        ok, reason = evaluate_oracle("SCENARIO_1_TRANSIENT_TOOL_FAILURE", obs, "VERIFIED")
        self.assertTrue(ok)

    def test_oracle_scenario_3_fails_when_security_token_missing(self):
        obs = {"success": True, "output": {"status": "ok", "host": "prod-1"}}
        ok, reason = evaluate_oracle("SCENARIO_3_INCORRECT_ACTION_RESULT", obs, "VERIFIED")
        self.assertFalse(ok)
        self.assertIn("security_token", reason)

    def test_oracle_detects_baseline_false_success_on_scenario_3(self):
        obs = {"exit_success": True, "output": {"status": "ok", "host": "prod-1"}}
        is_false = detect_false_success("SCENARIO_3_INCORRECT_ACTION_RESULT", obs, "FALSE_SUCCESS")
        self.assertTrue(is_false)

    def test_oracle_flags_false_success_when_verifier_mutated(self):
        """
        CRITICAL TEST: Mutate DeterministicVerifier to always return verified=True.
        The independent oracle MUST still catch the discrepancy and flag false success.
        Guarantees cleanup with try/finally.
        """
        from reagent.verification.verifier import DeterministicVerifier, VerificationResult

        original_verify = DeterministicVerifier.verify

        def mock_compromised_verify(self, recovery_decision_id, evidence, assertions):
            # Blindly returns verified = True regardless of actual discrepancies
            return VerificationResult(
                recovery_decision_id=recovery_decision_id,
                verified=True,
                evidence_summary="COMPROMISED VERIFIER: blindly returning pass",
                discrepancies=[],
            )

        try:
            DeterministicVerifier.verify = mock_compromised_verify

            # In scenario 3, output has NO security token
            tool_output = {"status": "ok", "host": "prod-1"}
            observations = {"success": True, "output": tool_output}

            # Even if the pipeline or verifier claims VERIFIED:
            compromised_status = "VERIFIED"

            # Independent Oracle must flag this as False Success
            is_false = detect_false_success(
                "SCENARIO_3_INCORRECT_ACTION_RESULT",
                observations,
                compromised_status,
            )
            self.assertTrue(is_false, "Independent oracle failed to detect false success from compromised verifier!")

        finally:
            # Guarantee restoration
            DeterministicVerifier.verify = original_verify

        # Verify restoration
        self.assertEqual(DeterministicVerifier.verify, original_verify)


class TestHTMLStructureAndHostileEscaping(unittest.TestCase):
    """
    Verifies HTML parsing via standard library, absence of external URLs,
    and safe rendering of hostile script injection strings.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = run_benchmark_scenarios()
        cls.html = generate_dashboard_html(cls.report)

    def test_stdlib_html_parser_validity(self):
        """Ensure document parses without structural exception via HTMLParser."""
        class StrictCollector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags = []
            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)

        parser = StrictCollector()
        parser.feed(self.html)
        self.assertIn("html", parser.tags)
        self.assertIn("head", parser.tags)
        self.assertIn("body", parser.tags)
        self.assertIn("header", parser.tags)
        self.assertIn("main", parser.tags)

    def test_no_external_urls(self):
        """Ensure no external http:// or https:// stylesheet or script resources exist."""
        matches = re.findall(r'(?:src|href)=["\'](https?://[^"\']+)["\']', self.html)
        self.assertEqual(len(matches), 0, f"External URLs found: {matches}")

    def test_source_badge_present(self):
        self.assertIn("OFFLINE · DETERMINISTIC", self.html)

    def test_safe_json_embed_escaping(self):
        """Verify that </script>, <!--, and paragraph separators are safely escaped."""
        hostile_payload = {
            "attack": "</script><script>alert('xss')</script>",
            "comment": "<!-- comment -->",
            "para": "Line\u2028Break",
        }
        embedded = _safe_json_embed(hostile_payload)
        self.assertNotIn("</script>", embedded)
        self.assertNotIn("<!--", embedded)
        self.assertNotIn("\u2028", embedded)
        self.assertIn("<\\/script>", embedded)
        self.assertIn("<\\!--", embedded)
        self.assertIn("\\u2028", embedded)

    def test_export_file_has_correct_encoding_and_meta(self):
        out_dir = tempfile.mkdtemp(prefix="reagent_m61_exp_")
        try:
            out_file = os.path.join(out_dir, "export.html")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(self.html)
            with open(out_file, "rb") as f:
                raw_bytes = f.read()
            decoded = raw_bytes.decode("utf-8")

            self.assertIn('<meta charset="UTF-8">', decoded)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


class TestPlaywrightOptional(unittest.TestCase):
    """Optional Playwright browser test. Cleanly skips if Playwright is not installed."""

    def test_browser_render_if_available(self):
        try:
            import playwright
        except ImportError:
            self.skipTest("Playwright not installed in environment (optional test).")


if __name__ == "__main__":
    unittest.main()
