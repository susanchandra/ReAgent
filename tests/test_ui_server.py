"""
test_ui_server.py — Integration tests for ReAgent Judge UI Server.
Tests HTTP handler endpoints, security boundaries, and recovery demo integration.
"""

import json
import threading
import time
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from reagent.ui_server import DashboardHTTPHandler, generate_dashboard_html


class TestUIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Bind to port 0 to get an ephemeral free port
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_01_index_html_renders_required_elements(self):
        """GET / returns HTTP 200, required headline, problem statement, and safety invariant."""
        req = urllib.request.Request(self._url("/"))
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get("Content-Type", "")
            self.assertIn("text/html", content_type)
            html_text = resp.read().decode("utf-8")

        self.assertIn("ReAgent — Verify outcomes. Recover safely.", html_text)
        self.assertIn("A tool can fail to report success even after its side effect has committed", html_text)
        self.assertIn("UNKNOWN is not ABSENT", html_text)
        self.assertIn("COMMIT_THEN_TIMEOUT", html_text)
        self.assertIn("480-Trial Comparative Benchmark", html_text)

    def test_02_health_endpoint(self):
        """GET /api/health returns HTTP 200 and offline deterministic metadata."""
        req = urllib.request.Request(self._url("/api/health"))
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["mode"], "OFFLINE_DETERMINISTIC")
        self.assertFalse(data["network_calls_allowed"])
        self.assertIn("source_git_revision", data)

    def test_03_recovery_demo_post_execution(self):
        """POST /api/run_recovery_demo executes fresh demo and returns side-by-side comparison."""
        req = urllib.request.Request(self._url("/api/run_recovery_demo"), method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertIn("naive_retry_arm", data)
        self.assertIn("reagent_arm", data)

        naive = data["naive_retry_arm"]
        reagent = data["reagent_arm"]

        # Starting balances equal
        self.assertEqual(naive["starting_balance"], 1000.0)
        self.assertEqual(reagent["starting_balance"], 1000.0)

        # Naive double debit
        self.assertEqual(naive["tool_attempts"], 2)
        self.assertEqual(naive["ledger_records_count"], 2)
        self.assertEqual(naive["net_transferred"], 1000.0)
        self.assertEqual(naive["final_balance"], 0.0)
        self.assertTrue(naive["duplicate_occurred"])
        self.assertEqual(naive["final_status"], "UNVERIFIED_CORRUPTED")

        # ReAgent single debit preserved
        self.assertEqual(reagent["tool_attempts"], 1)
        self.assertEqual(reagent["probe_queries"], 1)
        self.assertEqual(reagent["ledger_records_count"], 1)
        self.assertEqual(reagent["net_transferred"], 500.0)
        self.assertEqual(reagent["final_balance"], 500.0)
        self.assertFalse(reagent["duplicate_occurred"])
        self.assertEqual(reagent["final_status"], "VERIFIED")

    def test_04_benchmark_endpoint(self):
        """GET /api/benchmark returns 480-trial benchmark data."""
        req = urllib.request.Request(self._url("/api/benchmark"))
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertEqual(data["total_paired_configurations"], 480)
        self.assertIn("REAGENT_CONTRACT", data["arm_aggregates"])

    def test_05_security_and_404_handling(self):
        """Non-existent paths and traversal attempts fail closed with HTTP 404."""
        for bad_path in ("/unknown_route", "/../../../etc/passwd", "/reagent/../.env"):
            req = urllib.request.Request(self._url(bad_path))
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 404)

    def test_06_generate_dashboard_html_function(self):
        """generate_dashboard_html produces standalone string without errors."""
        html_out = generate_dashboard_html()
        self.assertIsInstance(html_out, str)
        self.assertGreater(len(html_out), 1000)

    def test_07_html_contains_offline_button_label(self):
        """GET / dashboard contains 'Run Recovery Simulation' button label (offline mode)."""
        req = urllib.request.Request(self._url("/"))
        with urllib.request.urlopen(req) as resp:
            html_text = resp.read().decode("utf-8")
        self.assertIn("Run Recovery Simulation", html_text)
        # Old label 'Run Live Simulation' must not appear as the button default label
        # (it may still appear in JS reset code — we specifically check the HTML button span)
        self.assertNotIn("Run Live Simulation</span>", html_text)

    def test_08_successive_posts_return_unique_run_ids(self):
        """Two successive POST /api/run_recovery_demo calls return different run_ids."""
        def do_post():
            req = urllib.request.Request(self._url("/api/run_recovery_demo"), method="POST")
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))

        result1 = do_post()
        time.sleep(0.05)  # ensure clock advances
        result2 = do_post()

        self.assertIn("run_id", result1, "run_id missing from first response")
        self.assertIn("run_id", result2, "run_id missing from second response")
        self.assertNotEqual(result1["run_id"], result2["run_id"],
                            "Successive calls must return distinct run_id values")

    def test_09_run_id_format_and_duration(self):
        """POST /api/run_recovery_demo returns run_id in expected format and non-negative duration."""
        req = urllib.request.Request(self._url("/api/run_recovery_demo"), method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        run_id = data.get("run_id", "")
        self.assertTrue(run_id.startswith("sim_run_"),
                        f"run_id '{run_id}' must start with 'sim_run_'")
        # Must have timestamp segment and uuid-hex suffix
        parts = run_id.split("_")
        self.assertGreaterEqual(len(parts), 3, "run_id must have at least 3 underscore-separated parts")

        duration = data.get("execution_duration_ms")
        self.assertIsNotNone(duration, "execution_duration_ms missing from response")
        self.assertGreaterEqual(duration, 0, "execution_duration_ms must be >= 0")

    def test_10_preflight_endpoint_fields(self):
        """GET /api/model_preflight returns safe boolean fields without secret values."""
        req = urllib.request.Request(self._url("/api/model_preflight"))
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        # Required fields
        self.assertIn("api_key_present", data)
        self.assertIn("model", data)
        self.assertIn("endpoint", data)
        self.assertIn("network_calls_allowed", data)
        self.assertIn("pilot_ledger_exists", data)

        # api_key_present must be bool — never a string containing key material
        self.assertIsInstance(data["api_key_present"], bool,
                              "api_key_present must be a bool, not a string")
        # network_calls_allowed must be False in offline mode
        self.assertFalse(data["network_calls_allowed"],
                         "network_calls_allowed must be False in offline mode")

    def test_11_preflight_does_not_expose_secret_value(self):
        """GET /api/model_preflight response body must not contain any API key value."""
        req = urllib.request.Request(self._url("/api/model_preflight"))
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
        # The response must not contain any non-empty NEBIUS_API_KEY value
        import os
        api_key_raw = os.environ.get("NEBIUS_API_KEY", "")
        if api_key_raw and api_key_raw.strip():
            self.assertNotIn(api_key_raw.strip(), body,
                             "Secret NEBIUS_API_KEY value must not appear in preflight response")

    def test_12_template_selector_buttons_in_html(self):
        """GET / dashboard contains all 3 task template buttons and supported-demo labels."""
        req = urllib.request.Request(self._url("/"))
        with urllib.request.urlopen(req) as resp:
            html_text = resp.read().decode("utf-8")

        self.assertIn("id=\"tpl_wallet_transfer\"", html_text)
        self.assertIn("id=\"tpl_hotel_booking\"", html_text)
        self.assertIn("id=\"tpl_database_update\"", html_text)
        self.assertIn("[Supported Task Template Demo]", html_text)
        self.assertIn("TASK_TEMPLATES_META", html_text)

    def test_13_hotel_booking_task_template_demo(self):
        """POST /api/run_recovery_demo?task_template=hotel_booking executes real isolated booking fixture."""
        req = urllib.request.Request(self._url("/api/run_recovery_demo?task_template=hotel_booking"), method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertEqual(data.get("task_template"), "hotel_booking")
        self.assertIn("naive_retry_arm", data)
        self.assertIn("reagent_arm", data)

        naive = data["naive_retry_arm"]
        reagent = data["reagent_arm"]

        # Entity label reflects flight seat booking
        ent_desc = reagent.get("details", {}).get("entity_description", "") + data.get("entity_description", "")
        self.assertIn("FL-2047", ent_desc)
        self.assertIn("14B", ent_desc)

        # ReAgent discovers existing booking via probe, restrains retry
        self.assertEqual(reagent["tool_attempts"], 1)
        self.assertEqual(reagent["probe_queries"], 1)
        self.assertEqual(reagent["ledger_records_count"], 2)
        self.assertFalse(reagent["duplicate_occurred"])
        self.assertEqual(reagent["final_status"], "VERIFIED")
        self.assertTrue(reagent["contract_postconditions_verified"])

        # Naive attempts blind retry without probe verification
        self.assertEqual(naive["tool_attempts"], 2)
        self.assertEqual(naive["probe_queries"], 0)
        self.assertFalse(naive["contract_postconditions_verified"])

    def test_14_database_update_task_template_demo(self):
        """POST /api/run_recovery_demo?task_template=database_update executes real isolated refund fixture."""
        req = urllib.request.Request(self._url("/api/run_recovery_demo?task_template=database_update"), method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertEqual(data.get("task_template"), "database_update")
        self.assertIn("naive_retry_arm", data)
        self.assertIn("reagent_arm", data)

        reagent = data["reagent_arm"]
        naive = data["naive_retry_arm"]
        ent_desc = reagent.get("details", {}).get("entity_description", "") + data.get("entity_description", "")
        self.assertIn("ORD-8821", ent_desc)
        self.assertEqual(reagent["tool_attempts"], 1)
        self.assertEqual(reagent["probe_queries"], 1)
        self.assertEqual(reagent["ledger_records_count"], 2)
        self.assertFalse(reagent["duplicate_occurred"])
        self.assertEqual(reagent["final_status"], "VERIFIED")
        self.assertTrue(reagent["contract_postconditions_verified"])

        # Naive attempts blind retry without probe verification
        self.assertEqual(naive["tool_attempts"], 2)
        self.assertEqual(naive["probe_queries"], 0)
        self.assertFalse(naive["contract_postconditions_verified"])

    def test_15_task_template_reconciled_benchmark_tab(self):
        """GET / dashboard contains reconciled benchmark duplicate accounting."""
        req = urllib.request.Request(self._url("/"))
        with urllib.request.urlopen(req) as resp:
            html_text = resp.read().decode("utf-8")

        self.assertIn("Duplicate Writes (Primary / Audit)", html_text)
        self.assertIn("24 unsafe retries produced 4 duplicate primary records and 5 duplicate audit log rows", html_text)

    def test_16_model_contract_proposal_integration(self):
        """POST /api/run_recovery_demo?use_model_contract=1 reflects model contract proposal in provenance and event log."""
        req = urllib.request.Request(self._url("/api/run_recovery_demo?scenario=B&use_model_contract=1"), method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertEqual(data.get("provenance"), "MODEL_PROPOSAL_VERIFIED")
        reagent = data["reagent_arm"]
        self.assertEqual(reagent.get("provenance"), "MODEL_PROPOSAL_VERIFIED")
        self.assertEqual(reagent.get("details", {}).get("contract_source"), "NVIDIA_NEMOTRON_ADVISORY")

        # Verify event log reflects the model proposal source
        ev_req = urllib.request.Request(self._url(f"/api/event_log?scenario=B&run_id={data.get('run_id')}"))
        with urllib.request.urlopen(ev_req) as ev_resp:
            self.assertEqual(ev_resp.status, 200)
            ev_data = json.loads(ev_resp.read().decode("utf-8"))
            events = ev_data.get("events", [])
            freeze_events = [e for e in events if e.get("action") == "FREEZE_CONTRACT"]
            self.assertTrue(len(freeze_events) > 0)
            self.assertIn("from NVIDIA Nemotron proposal", freeze_events[0].get("detail", ""))


if __name__ == "__main__":
    unittest.main()
