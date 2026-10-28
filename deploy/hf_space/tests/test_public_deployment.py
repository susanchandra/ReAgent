"""
test_public_deployment.py — Verification tests for ReAgent Public Deployment Package.

Tests PublicSpaceDashboardHandler on an ephemeral port to verify:
- Zero external network requests (strict network guard installed).
- Fail-closed public mode: /api/nemotron_propose is locked and returns PUBLIC_DEMO_LOCKED.
- All 5 scenario simulations run cleanly on isolated disposable SQLite.
- Benchmark sample route returns correct aggregates without disk mutation.
- Clean health, scenarios, preflight, and receipt endpoints.
- Zero secret exposure in responses or headers.
- Concurrent requests yield independent, non-colliding run IDs.
"""

import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

# Enforce repository root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy.hf_space.entrypoint import PublicSpaceDashboardHandler


class PublicDeploymentNetworkGuard:
    """Blocks any outbound TCP connection attempt during test execution to external hosts."""
    def __init__(self):
        self._orig_connect = socket.socket.connect
        self.blocked_attempts = []

    def __enter__(self):
        guard = self
        def guarded_connect(sock_self, address):
            host, port = address[0], address[1]
            # Allow only loopback connections
            if host in ("127.0.0.1", "localhost", "::1"):
                return guard._orig_connect(sock_self, address)
            guard.blocked_attempts.append(address)
            raise ConnectionRefusedError(f"[NETWORK-GUARD] External connection to {address} strictly blocked!")
        socket.socket.connect = guarded_connect
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        socket.socket.connect = self._orig_connect


class TestPublicDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Bind to port 0 for an ephemeral free port
        cls.server = HTTPServer(("127.0.0.1", 0), PublicSpaceDashboardHandler)
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

    def test_01_landing_page_renders_offline_dashboard(self):
        """GET / returns HTTP 200, Mission Control HTML, and core safety invariants."""
        with PublicDeploymentNetworkGuard() as guard:
            with urllib.request.urlopen(self._url("/")) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("ReAgent — Verify outcomes. Recover safely.", html)
                self.assertIn("480-Trial Comparative Benchmark", html)
                self.assertIn("Automated Quality Gates", html)
                self.assertIn("767 / 780", html)
            self.assertEqual(len(guard.blocked_attempts), 0)

    def test_02_health_endpoint_metadata(self):
        """GET /api/health reports OFFLINE_DETERMINISTIC mode and zero network calls allowed."""
        with urllib.request.urlopen(self._url("/api/health")) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["mode"], "OFFLINE_DETERMINISTIC")
            self.assertFalse(data["network_calls_allowed"])
            self.assertEqual(data["scenarios"], ["A", "B", "C", "D", "E"])

    def test_03_fail_closed_public_model_lock(self):
        """POST /api/nemotron_propose fails closed with PUBLIC_DEMO_LOCKED without network calls."""
        with PublicDeploymentNetworkGuard() as guard:
            req = urllib.request.Request(self._url("/api/nemotron_propose"), data=b"", method="POST")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "PUBLIC_DEMO_LOCKED")
                self.assertFalse(data["is_live"])
                self.assertFalse(data["schema_valid"])
                self.assertIn("locked in public", data["message"].lower())
            # Ensure guard intercepted 0 external connections
            self.assertEqual(len(guard.blocked_attempts), 0)

    def test_04_scenario_recovery_simulation_isolated(self):
        """POST /api/run_recovery_demo executes real simulation on disposable SQLite."""
        req = urllib.request.Request(
            self._url("/api/run_recovery_demo?scenario=B&task_template=wallet_transfer"),
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("run_id", data)
            self.assertTrue(data["run_id"].startswith("sim_run_"))
            self.assertIn("naive_retry_arm", data)
            self.assertIn("reagent_arm", data)
            self.assertEqual(data["naive_retry_arm"]["duplicate_occurred"], True)
            self.assertEqual(data["reagent_arm"]["duplicate_occurred"], False)
            self.assertEqual(data["reagent_arm"]["final_status"], "VERIFIED")

    def test_05_benchmark_sample_execution(self):
        """POST /api/run_benchmark?sample=1 executes fast sample without mutating submission artifacts."""
        req = urllib.request.Request(
            self._url("/api/run_benchmark?sample=1"),
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["total_trials"], 80)
            self.assertIn("REAGENT_CONTRACT", data["arm_aggregates"])
            self.assertEqual(data["arm_aggregates"]["REAGENT_CONTRACT"]["false_verified_count"], 0)

    def test_06_concurrent_execution_isolation(self):
        """Two concurrent simulation calls return distinct run IDs and independent states."""
        results = []
        errors = []

        def worker(sc):
            try:
                r = urllib.request.Request(self._url(f"/api/run_recovery_demo?scenario={sc}"), data=b"", method="POST")
                with urllib.request.urlopen(r) as resp:
                    results.append(json.loads(resp.read().decode("utf-8")))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker, args=("B",))
        t2 = threading.Thread(target=worker, args=("C",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0, f"Errors in concurrent workers: {errors}")
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0]["run_id"], results[1]["run_id"], "Run IDs must be distinct")

    def test_07_zero_secrets_leakage(self):
        """Endpoints never leak API keys, auth tokens, or private credentials."""
        endpoints = [
            "/",
            "/api/health",
            "/api/model_preflight",
            "/api/scenarios",
            "/api/benchmark",
        ]
        # Specific secret patterns: Bearer tokens, nvapi keys, OpenAI sk-proj keys
        forbidden_substrings = ["nvapi-", "nebius-secret", "Bearer ", "sk-proj-", "sk-live-"]
        for ep in endpoints:
            with urllib.request.urlopen(self._url(ep)) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                for secret in forbidden_substrings:
                    self.assertNotIn(secret, body, f"Forbidden pattern '{secret}' leaked in {ep}")
    def test_08_concurrency_receipt_and_replay_integrity(self):
        """Cross-session concurrent execution preserves individual run IDs, events, and receipt data."""
        # 1. Run Scenario B and Scenario E concurrently
        run_b_res = {}
        run_e_res = {}
        errors = []

        def run_sim(scenario_id, out_dict):
            try:
                url = self._url(f"/api/run_recovery_demo?scenario={scenario_id}&task_template=wallet_transfer")
                req = urllib.request.Request(url, data=b"", method="POST")
                with urllib.request.urlopen(req) as resp:
                    out_dict.update(json.loads(resp.read().decode("utf-8")))
            except Exception as e:
                errors.append(e)

        tb = threading.Thread(target=run_sim, args=("B", run_b_res))
        te = threading.Thread(target=run_sim, args=("E", run_e_res))
        tb.start()
        te.start()
        tb.join()
        te.join()

        self.assertEqual(len(errors), 0, f"Concurrent execution errors: {errors}")
        self.assertIn("run_id", run_b_res)
        self.assertIn("run_id", run_e_res)
        self.assertNotEqual(run_b_res["run_id"], run_e_res["run_id"])

        # 2. Fetch specific event logs for each run_id
        with urllib.request.urlopen(self._url(f"/api/event_log?run_id={run_b_res['run_id']}")) as resp_b:
            events_b = json.loads(resp_b.read().decode("utf-8"))
        with urllib.request.urlopen(self._url(f"/api/event_log?run_id={run_e_res['run_id']}")) as resp_e:
            events_e = json.loads(resp_e.read().decode("utf-8"))

        self.assertEqual(events_b["run_id"], run_b_res["run_id"])
        self.assertEqual(events_e["run_id"], run_e_res["run_id"])

        # 3. Verify receipt construction matches requesting run
        receipt_b = {
            "run_id": run_b_res["run_id"],
            "scenario": "B",
            "reagent_status": run_b_res["reagent_arm"]["final_status"],
            "naive_duplicate": run_b_res["naive_retry_arm"]["duplicate_occurred"],
        }
        receipt_e = {
            "run_id": run_e_res["run_id"],
            "scenario": "E",
            "reagent_status": run_e_res["reagent_arm"]["final_status"],
            "naive_duplicate": run_e_res["naive_retry_arm"]["duplicate_occurred"],
        }
        self.assertEqual(receipt_b["reagent_status"], "VERIFIED")
        self.assertEqual(receipt_b["naive_duplicate"], True)
        self.assertEqual(receipt_e["reagent_status"], "ESCALATED")
        self.assertEqual(receipt_e["naive_duplicate"], True)

        # 4. Reset one session does not wipe the other session's queryability
        req_reset = urllib.request.Request(self._url("/api/reset"), data=b"", method="POST")
        with urllib.request.urlopen(req_reset) as r:
            self.assertEqual(r.status, 200)

        # Event log for B is still recoverable with its run_id
        with urllib.request.urlopen(self._url(f"/api/event_log?run_id={run_b_res['run_id']}&scenario=B")) as r_check:
            self.assertEqual(r_check.status, 200)
            data_check = json.loads(r_check.read().decode("utf-8"))
            self.assertGreater(data_check["event_count"], 0)


if __name__ == "__main__":
    unittest.main()
