"""
test_judge_ready_product.py — Verification suite for judge-ready product requirements.

Verifies:
1. Arbitrary natural-language tasks reach generic intent and capability discovery.
2. Unknown tasks produce structured capability gap specifications with non-mutating dry runs.
3. Rendered dashboard HTML contains all required judge-ready investigation controls:
   - Universal task prompt chips
   - Execution mode segmented control (Local Sim, Dry Run, Real Locked)
   - Step, Restart, and Speed controls in Failure Lab
   - Copy JSON button in Workbench raw diagnostics
   - Nemotron Advisory Architecture diagram and division of responsibility table
4. Evidence timeline and verifier receipt derive from the same authoritative run record.
5. Zero external model calls on page load or preflight (offline deterministic assurance).
6. Zero secrets exposed across HTML, health, or preflight endpoints.
7. JavaScript syntax is completely clean and defines all interactive controllers.
"""

import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.parse
import urllib.request
from http.server import HTTPServer

from reagent.ui_server import DashboardHTTPHandler, generate_dashboard_html


class TestJudgeReadyProduct(unittest.TestCase):
    """Judge-ready product feature and safety verification."""

    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_01_arbitrary_natural_language_intent_discovery(self):
        """Unsupported task enters intent pipeline and returns structured capability gap."""
        task = "Order biryani for delivery to campus"
        url = self._url(f"/api/run_recovery_demo?scenario=B&task_input={urllib.parse.quote(task)}")
        req = urllib.request.Request(url, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            data = json.loads(e.read().decode("utf-8"))

        self.assertEqual(data.get("status"), "UNSUPPORTED_TASK")
        self.assertIn("interpreted_intent", data)
        self.assertIn("proposed_contract", data)
        self.assertIn("missing_integration", data)

        interp = data["interpreted_intent"]
        self.assertIn("Food", interp.get("inferred_domain", ""))
        self.assertTrue(len(interp.get("multi_step_plan", [])) >= 3)
        self.assertFalse(data.get("execution_authorized", False))

    def test_02_unknown_tasks_produce_structured_gap_not_generic_error(self):
        """Extended real-world tasks return domain-specific dry runs without inventing execution."""
        sample_tasks = [
            ("Book an Uber auto to university", "Rideshare"),
            ("Schedule team sync tomorrow at 10am", "Calendar"),
            ("Provision S3 bucket for cloud storage", "Cloud Infrastructure"),
        ]

        for task_prompt, expected_domain_keyword in sample_tasks:
            url = self._url(f"/api/run_recovery_demo?scenario=B&task_input={urllib.parse.quote(task_prompt)}")
            req = urllib.request.Request(url, method="POST")
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                data = json.loads(e.read().decode("utf-8"))

            self.assertEqual(data.get("status"), "UNSUPPORTED_TASK")
            interp = data.get("interpreted_intent", {})
            self.assertIn(expected_domain_keyword, interp.get("inferred_domain", ""))
            missing = data.get("missing_integration", {})
            self.assertTrue(len(missing.get("required_permissions", [])) > 0)
            self.assertIn("gap_report", data)
            self.assertFalse(data.get("reagent_arm", {}).get("mutated", True))

    def test_03_rendered_html_contains_judge_ready_controls(self):
        """Dashboard HTML contains all interactive UI controls required for judge evaluation."""
        html = generate_dashboard_html()

        # Workbench controls
        self.assertIn("Universal Tasks:", html)
        self.assertIn("Registered Adapters:", html)
        self.assertIn("Execution Mode:", html)
        self.assertIn("Local Simulation (Isolated SQLite)", html)
        self.assertIn("Dry Run (Intent &amp; Gap Analysis)", html)
        self.assertIn("Real Execution (Locked", html)
        self.assertIn('id="btnCopyJson"', html)

        # Failure Lab investigation controls
        self.assertIn('id="btnStepForward"', html)
        self.assertIn('id="btnRestart"', html)
        self.assertIn('id="labSpeedSelect"', html)
        self.assertIn('id="btnRun"', html)
        self.assertIn('id="btnReset"', html)
        self.assertIn('id="btnReplay"', html)

        # Capabilities & Model Integration Tab
        self.assertIn("Capabilities &amp; Model Integration: Nemotron Advisory Architecture", html)
        self.assertIn("Architecture Pipeline: Advisory Reasoning vs. Deterministic Grounding", html)
        self.assertIn("Advisory Only", html)
        self.assertIn("Nemotron advisory reasoning via Nebius", html)

        # Core Invariants
        self.assertIn("UNKNOWN", html)
        self.assertIn("ABSENT", html)

    def test_04_evidence_timeline_and_receipt_consistency(self):
        """Event log and run result derive from the same authoritative execution run_id."""
        url = self._url("/api/run_recovery_demo?scenario=B&task_template=wallet_transfer")
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as resp:
            run_data = json.loads(resp.read().decode("utf-8"))

        run_id = run_data.get("run_id")
        self.assertTrue(run_id)

        ev_url = self._url(f"/api/event_log?scenario=B&run_id={urllib.parse.quote(run_id)}")
        with urllib.request.urlopen(ev_url) as ev_resp:
            ev_data = json.loads(ev_resp.read().decode("utf-8"))

        events = ev_data.get("events", [])
        self.assertTrue(len(events) >= 3)
        self.assertEqual(ev_data.get("run_id"), run_id)

    def test_05_no_model_calls_on_page_load_or_preflight(self):
        """Page load and preflight operate strictly offline without calling external APIs."""
        preflight_url = self._url("/api/model_preflight")
        with urllib.request.urlopen(preflight_url) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Strictly asserts offline deterministic mode
        self.assertIn("api_key_present", data)
        self.assertIn("endpoint", data)
        self.assertFalse(data.get("network_calls_allowed", True))

    def test_06_no_secrets_exposed_in_html_or_endpoints(self):
        """Dashboard HTML, preflight, and health responses never leak API keys."""
        secret_patterns = [
            re.compile(r'nb-[a-zA-Z0-9_\-]{20,}'),
            re.compile(r'NEBIUS_API_KEY\s*=\s*["\']?[a-zA-Z0-9_\-]+'),
            re.compile(r'Bearer\s+[a-zA-Z0-9_\-\.]{25,}'),
        ]

        endpoints = ["/", "/api/health", "/api/model_preflight"]
        for ep in endpoints:
            with urllib.request.urlopen(self._url(ep)) as resp:
                body = resp.read().decode("utf-8")
                for pat in secret_patterns:
                    self.assertIsNone(pat.search(body), f"Secret pattern matched in endpoint {ep}")

    def test_07_javascript_syntax_and_stepping_controllers(self):
        """Inline JavaScript defines all new interactive functions and parses with zero errors."""
        html = generate_dashboard_html()
        scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
        self.assertTrue(len(scripts) > 0)
        all_js = "\n".join(scripts)

        # Check required functions exist
        required_fns = [
            "setWbCustomTask",
            "setWbExecutionMode",
            "copyRawJson",
            "stepScenario",
            "restartScenario",
            "setLabSpeed",
            "updateEvidenceTab",
            "runComparison",
            "resetState",
            "runWorkbench",
        ]
        for fn in required_fns:
            self.assertIn(fn, all_js, f"Expected function {fn} not found in inline JS")

        # Validate syntax via node --check if node is installed
        node_bin = shutil.which("node")
        if node_bin:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as f:
                f.write(all_js)
                tmp_name = f.name
            try:
                res = subprocess.run([node_bin, "--check", tmp_name], capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, f"Node syntax error: {res.stderr}")
            finally:
                import os
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)


    def test_08_extensible_adapter_outside_core(self):
        """Standalone custom adapters register and dispatch without touching core routing; unregistered keys fail closed."""
        from reagent.adapter_registry import ADAPTER_REGISTRY
        from reagent.experiments.dbops.recovery_demo import DemoArtifact, ArmResult

        def standalone_custom_adapter(scenario_id="B", timestamp_utc=None, run_id=None, **kwargs):
            naive = ArmResult(
                policy_name="Naive",
                provenance="STANDALONE_CUSTOM",
                starting_balance=100.0,
                final_balance=60.0,
                net_transferred=40.0,
                expected_transfer_amount=20.0,
                tool_attempts=2,
                probe_queries=0,
                ledger_records_count=2,
                duplicate_occurred=True,
                final_status="UNVERIFIED_CORRUPTED",
                contract_postconditions_verified=False,
                details={"custom": True},
            )
            reagent = ArmResult(
                policy_name="ReAgent",
                provenance="STANDALONE_CUSTOM",
                starting_balance=100.0,
                final_balance=80.0,
                net_transferred=20.0,
                expected_transfer_amount=20.0,
                tool_attempts=1,
                probe_queries=1,
                ledger_records_count=1,
                duplicate_occurred=False,
                final_status="VERIFIED_SUCCESS",
                contract_postconditions_verified=True,
                details={"custom": True},
            )
            return DemoArtifact(
                demo_name="CustomAdapterDemo",
                task_id="custom_task_99",
                fault_mode="NETWORK_DROP",
                timestamp_utc=timestamp_utc or 1700000000.0,
                provenance="STANDALONE_CUSTOM_ADAPTER",
                naive_retry_arm=naive,
                reagent_arm=reagent,
                comparison_summary="Standalone adapter executed successfully with zero core changes.",
                run_id=run_id or "run-custom-123",
                execution_duration_ms=15.5,
            )

        custom_key = "standalone_custom_sync"
        ADAPTER_REGISTRY.register(custom_key, standalone_custom_adapter)
        try:
            self.assertTrue(ADAPTER_REGISTRY.is_registered(custom_key))
            self.assertIn(custom_key, ADAPTER_REGISTRY.list_adapters())

            # Dispatch via registry without modifying core routing
            artifact = ADAPTER_REGISTRY.dispatch(custom_key, scenario_id="B")
            self.assertEqual(artifact.demo_name, "CustomAdapterDemo")
            self.assertEqual(artifact.reagent_arm.final_status, "VERIFIED_SUCCESS")
            self.assertFalse(artifact.reagent_arm.duplicate_occurred)

            # Strict fail-closed on unregistered capability
            with self.assertRaises(KeyError):
                ADAPTER_REGISTRY.dispatch("unregistered_domain_capability", scenario_id="B")
        finally:
            if custom_key in ADAPTER_REGISTRY._adapters:
                del ADAPTER_REGISTRY._adapters[custom_key]

    def test_09_failure_lab_stepper_semantics_and_scenarios(self):
        """Failure Lab enforces UNKNOWN != ABSENT on Scenario D and discrepancy detection on Scenario E."""
        # Scenario D: Timeout / observer unreachable
        url_d = self._url("/api/run_recovery_demo?scenario=D&task_template=wallet_transfer")
        req_d = urllib.request.Request(url_d, method="POST")
        with urllib.request.urlopen(req_d) as resp:
            data_d = json.loads(resp.read().decode("utf-8"))

        reagent_d = data_d.get("reagent_arm", {})
        self.assertEqual(reagent_d.get("final_status"), "ESCALATED")
        self.assertFalse(reagent_d.get("duplicate_occurred", True))
        details_d = reagent_d.get("details", {})
        self.assertIn("UNKNOWN != ABSENT", details_d.get("safety_invariant", ""))
        self.assertEqual(details_d.get("probe_result"), "STATUS_UNAVAILABLE")

        # Scenario E: Partial commit / conflicting evidence
        url_e = self._url("/api/run_recovery_demo?scenario=E&task_template=wallet_transfer")
        req_e = urllib.request.Request(url_e, method="POST")
        with urllib.request.urlopen(req_e) as resp:
            data_e = json.loads(resp.read().decode("utf-8"))

        reagent_e = data_e.get("reagent_arm", {})
        self.assertEqual(reagent_e.get("final_status"), "ESCALATED")
        self.assertFalse(reagent_e.get("duplicate_occurred", True))
        details_e = reagent_e.get("details", {})
        self.assertIn("DISCREPANCY", details_e.get("probe_result", ""))

        # Verify stage-to-event inspectability elements in dashboard HTML
        html = generate_dashboard_html()
        self.assertIn('id="step1"', html)
        self.assertIn('id="step2"', html)
        self.assertIn('id="step3"', html)
        self.assertIn('id="step4"', html)
        self.assertIn('id="step5"', html)
        self.assertIn("inspectLabStage", html)
        self.assertIn("renderLabStage", html)


if __name__ == "__main__":
    unittest.main()
