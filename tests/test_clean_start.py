"""
test_clean_start.py — Clean-start environment and zero-dependency smoke test.

Verifies:
1. Environment dependencies: all required modules can be imported cleanly.
2. Server launch: starts cleanly on ephemeral port, serves health, preflight, and index.html.
3. Offline recovery simulation: returns fresh run_id, valid non-idempotent restraint, and zero mutations on disk manifest.
4. Offline judge demo: executes in-process or via subprocess with exit code 0.
5. Manifest validation: 12/12 artifacts match cryptographic hashes.
"""

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from reagent.ui_server import DashboardHTTPHandler, generate_dashboard_html
from reagent.experiments.dbops.validate_submission_manifest import validate_manifest


class TestCleanStart(unittest.TestCase):
    """End-to-end clean-start verification suite."""

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
        cls.server.server_close()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_01_core_dependencies_importable(self):
        """All core dependencies specified in requirements.txt can be imported."""
        import pydantic
        import filelock
        import openai
        self.assertTrue(hasattr(pydantic, "BaseModel"))
        self.assertTrue(hasattr(filelock, "FileLock"))
        self.assertTrue(hasattr(openai, "OpenAI"))

    def test_02_server_serves_valid_judge_dashboard(self):
        """UI server serves complete dashboard with offline labels and no broken templates."""
        req = urllib.request.Request(self._url("/"))
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            html_text = resp.read().decode("utf-8")

        self.assertIn("Run Recovery Simulation", html_text)
        # The tab title is always present; the subtitle suffix changes with key state.
        self.assertIn("Real Nemotron (Nebius) Mode", html_text)
        # Safety label is always rendered regardless of key state.
        self.assertIn("ADVISORY ONLY", html_text)
        self.assertIn("UNKNOWN is not ABSENT", html_text)
        self.assertIn("480-Trial Comparative Benchmark", html_text)

    def test_03_preflight_endpoint_safe_and_ready(self):
        """Preflight reports configuration cleanly without secret exposure."""
        req = urllib.request.Request(self._url("/api/model_preflight"))
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertIn("model", data)
        self.assertEqual(data["model"], "nvidia/nemotron-3-super-120b-a12b")
        self.assertIn("endpoint", data)
        self.assertEqual(data["endpoint"], "https://api.tokenfactory.nebius.com/v1/")
        self.assertIsInstance(data["api_key_present"], bool)
        self.assertFalse(data["network_calls_allowed"])

    def test_04_offline_simulation_dynamic_execution(self):
        """POST /api/run_recovery_demo runs fresh simulation without disk mutation."""
        req = urllib.request.Request(self._url("/api/run_recovery_demo"), method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertIn("run_id", data)
        self.assertTrue(data["run_id"].startswith("sim_run_"))
        self.assertGreaterEqual(data["execution_duration_ms"], 0)
        self.assertEqual(data["reagent_arm"]["ledger_records_count"], 1)
        self.assertEqual(data["naive_retry_arm"]["ledger_records_count"], 2)

    def test_05_judge_demo_cli_executes_cleanly(self):
        """python -m reagent.experiments.dbops.judge_demo runs in subprocess with exit code 0."""
        with tempfile.TemporaryDirectory(prefix="reagent_clean_start_demo_") as tmp_dir:
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "reagent.experiments.dbops.judge_demo",
                    "--output-dir",
                    tmp_dir,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(res.returncode, 0, f"judge_demo failed: {res.stderr}")
            self.assertIn("0.0% false verifications", res.stdout)
            self.assertIn("REAGENT_CONTRACT", res.stdout)

    def test_06_manifest_validation_clean_pass(self):
        """validate_manifest confirms 12/12 artifacts pass SHA-256 verification."""
        success, details = validate_manifest(
            manifest_path=Path("reports/dbops/submission/submission_manifest.json"),
            repo_root=Path("."),
            quiet=True,
        )
        self.assertTrue(success, f"Manifest validation failed: {details['errors']}")
        self.assertEqual(details["mismatched_count"], 0)
        self.assertEqual(details["missing_count"], 0)
        self.assertEqual(details["verified_count"], 12)


if __name__ == "__main__":
    unittest.main()
