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

    def test_17_dashboard_inline_js_syntax_clean(self):
        """Dashboard inline JavaScript parses cleanly with 0 syntax errors and 0 duplicate declarations."""
        import collections
        import os
        import re
        import shutil
        import subprocess
        import tempfile

        html_text = generate_dashboard_html()
        start = html_text.find("<script>")
        end = html_text.rfind("</script>")
        self.assertGreater(start, 0, "No <script> tag found in dashboard HTML")
        self.assertGreater(end, start, "Malformed <script> tag in dashboard HTML")
        js_code = html_text[start + 8 : end]

        # Check for duplicate top-level let/const declarations (prevents Identifier already declared SyntaxError)
        decls = collections.defaultdict(list)
        for i, line in enumerate(js_code.splitlines(), 1):
            m = re.match(r"^(let|const)\s+([_a-zA-Z][_a-zA-Z0-9]*)", line)
            if m:
                decls[m.group(2)].append((i, m.group(1), line.strip()[:80]))
        dups = {k: v for k, v in decls.items() if len(v) > 1}
        self.assertEqual(dups, {}, f"Duplicate top-level let/const declarations found in inline JS: {dups}")

        # If node is installed, run official syntax check
        node_bin = shutil.which("node")
        if node_bin:
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
                f.write(js_code)
                tmp_js = f.name
            try:
                proc = subprocess.run([node_bin, "--check", tmp_js], capture_output=True, text=True, timeout=10)
                self.assertEqual(
                    proc.returncode,
                    0,
                    f"node --check failed on inline JS:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
                )
            finally:
                try:
                    os.unlink(tmp_js)
                except OSError:
                    pass

    def test_18_browser_interactivity_and_clicks(self):
        """In a real headless browser, clicks on navigation, scenarios, simulation, replay, reset succeed with 0 console errors."""
        import json
        import os
        import shutil
        import subprocess
        import tempfile

        node_bin = shutil.which("node")
        chrome_exe = None
        for candidate in [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            shutil.which("google-chrome"),
            shutil.which("chromium"),
            shutil.which("msedge"),
        ]:
            if candidate and os.path.exists(candidate):
                chrome_exe = candidate
                break

        if not (node_bin and chrome_exe):
            self.skipTest("node or chrome/edge not found; skipping headless browser interaction test")

        js_script = f"""
const {{ spawn }} = require('child_process');

async function run() {{
  const chrome = spawn({json.dumps(chrome_exe)}, [
    '--headless=new',
    '--remote-debugging-port=0',
    '--disable-gpu',
    '--no-sandbox',
    'about:blank'
  ], {{ stdio: ['ignore', 'pipe', 'pipe'] }});

  let wsUrl = null;
  const portMatchPromise = new Promise((resolve, reject) => {{
    const timer = setTimeout(() => reject(new Error('Chrome launch timeout')), 10000);
    let buf = '';
    chrome.stderr.on('data', chunk => {{
      buf += chunk.toString();
      const m = buf.match(/DevTools listening on ws:\\/\\/127\\.0\\.0\\.1:(\\d+)\\//);
      if (m) {{
        clearTimeout(timer);
        resolve(m[1]);
      }}
    }});
  }});

  const debugPort = await portMatchPromise;
  await new Promise(r => setTimeout(r, 500));
  const tResp = await fetch('http://127.0.0.1:' + debugPort + '/json');
  const targets = await tResp.json();
  const pageTarget = targets.find(t => t.type === 'page') || targets[0];
  const ws = new WebSocket(pageTarget.webSocketDebuggerUrl);
  let id = 1;
  const pending = new Map();
  const exceptions = [];

  function send(method, params = {{}}) {{
    return new Promise((resolve, reject) => {{
      const mid = id++;
      pending.set(mid, {{ resolve, reject }});
      ws.send(JSON.stringify({{ id: mid, method, params }}));
    }});
  }}

  ws.onmessage = e => {{
    const d = JSON.parse(e.data);
    if (d.id && pending.has(d.id)) {{
      pending.get(d.id).resolve(d.result);
      pending.delete(d.id);
    }}
    if (d.method === 'Runtime.exceptionThrown') {{
      exceptions.push(d.params.exceptionDetails);
    }}
  }};

  await new Promise(r => ws.onopen = r);
  await send('Page.enable');
  await send('Runtime.enable');

  await send('Page.navigate', {{ url: 'http://127.0.0.1:{self.port}/' }});
  await new Promise(r => setTimeout(r, 1500));

  // 1. Navigation clicks
  const navResults = await send('Runtime.evaluate', {{
    expression: `(() => {{
      const r = {{}};
      for (const id of ['nav-mc', 'nav-scenarios', 'nav-evidence', 'nav-benchmark', 'nav-settings']) {{
        const btn = document.getElementById(id);
        if (btn) {{
          btn.click();
          r[id] = document.querySelector('.app-view.active')?.id;
        }}
      }}
      return r;
    }})()`,
    returnByValue: true
  }});

  // 2. Scenario selector clicks
  await send('Runtime.evaluate', {{ expression: `navigateTo('view-scenarios')` }});
  const scResults = await send('Runtime.evaluate', {{
    expression: `(() => {{
      const r = {{}};
      for (const sc of ['A', 'B', 'C', 'D', 'E']) {{
        const pill = document.getElementById('pill-sc-' + sc);
        if (pill) {{
          pill.click();
          r[sc] = _currentScenario;
        }}
      }}
      return r;
    }})()`,
    returnByValue: true
  }});

  // 3. Run Recovery Simulation
  await send('Runtime.evaluate', {{ expression: `selectScenario('B')` }});
  await send('Runtime.evaluate', {{ expression: `document.getElementById('btnRun')?.click()` }});
  await new Promise(r => setTimeout(r, 1800));

  const runResults = await send('Runtime.evaluate', {{
    expression: `(() => {{
      return {{
        runResultPresent: !!_runResult,
        runId: _runResult?.run_id,
        bannerText: document.getElementById('verificationBigBanner')?.innerText?.slice(0, 50),
        eventCount: _eventLog?.length
      }};
    }})()`,
    returnByValue: true
  }});

  // 4. Replay and Stop
  const replayStart = await send('Runtime.evaluate', {{
    expression: `(() => {{
      const btn = document.getElementById('btnReplay');
      if (btn) btn.click();
      return {{ isReplaying: _isReplaying, btnText: btn?.innerText }};
    }})()`,
    returnByValue: true
  }});

  await new Promise(r => setTimeout(r, 800));

  const replayStop = await send('Runtime.evaluate', {{
    expression: `(() => {{
      const btn = document.getElementById('btnReplay');
      if (btn) btn.click();
      return {{ isReplaying: _isReplaying, btnText: btn?.innerText }};
    }})()`,
    returnByValue: true
  }});

  // 5. Diagnostics toggle
  const diagRes = await send('Runtime.evaluate', {{
    expression: `(() => {{
      const btn = document.getElementById('btnToggleDiagnostics');
      const wrap = document.getElementById('advancedDiagnosticsWrap');
      const before = wrap?.style.display;
      if (btn) btn.click();
      const after = wrap?.style.display;
      return {{ before, after }};
    }})()`,
    returnByValue: true
  }});

  // 6. Reset
  const resetRes = await send('Runtime.evaluate', {{
    expression: `(() => {{
      document.getElementById('btnReset')?.click();
      return {{ runResult: _runResult, eventCount: _eventLog?.length }};
    }})()`,
    returnByValue: true
  }});

  ws.close();
  chrome.kill();

  console.log(JSON.stringify({{
    exceptions,
    navResults: navResults?.result?.value,
    scResults: scResults?.result?.value,
    runResults: runResults?.result?.value,
    replayStart: replayStart?.result?.value,
    replayStop: replayStop?.result?.value,
    diagRes: diagRes?.result?.value,
    resetRes: resetRes?.result?.value
  }}));
}}

run().catch(err => {{
  console.error(err);
  process.exit(1);
}});
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(js_script)
            tmp_runner = f.name

        try:
            proc = subprocess.run([node_bin, tmp_runner], capture_output=True, text=True, encoding="utf-8", timeout=30)
            if proc.returncode != 0:
                self.fail(f"Node runner failed (code {proc.returncode}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")
            try:
                res = json.loads(proc.stdout.strip().splitlines()[-1])
            except Exception as e:
                self.fail(f"JSON parse error: {e}\nFull STDOUT:\n{proc.stdout}\nFull STDERR:\n{proc.stderr}")
            self.assertEqual(len(res["exceptions"]), 0, f"Uncaught exceptions in browser: {res['exceptions']}")
            self.assertEqual(res["navResults"]["nav-mc"], "view-mission-control")
            self.assertEqual(res["navResults"]["nav-scenarios"], "view-scenarios")
            self.assertEqual(res["navResults"]["nav-evidence"], "view-evidence")
            self.assertEqual(res["navResults"]["nav-benchmark"], "view-benchmark")
            self.assertEqual(res["scResults"], {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E"})
            self.assertTrue(res["runResults"]["runResultPresent"])
            self.assertEqual(res["runResults"]["eventCount"], 12)
            self.assertTrue(res["replayStart"]["isReplaying"])
            self.assertFalse(res["replayStop"]["isReplaying"])
            self.assertIsNone(res["resetRes"]["runResult"])
            self.assertEqual(res["resetRes"]["eventCount"], 0)
        finally:
            try:
                os.unlink(tmp_runner)
            except OSError:
                pass


    def test_19_benchmark_endpoint_sample(self):
        """POST /api/run_benchmark?sample=1 returns ok status, 80 trials, and all 5 arm aggregates."""
        req = urllib.request.Request(
            self._url("/api/run_benchmark?sample=1"),
            data=b"",
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=120) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertEqual(data["status"], "ok", f"Benchmark returned non-ok: {data.get('error')}")
        self.assertGreaterEqual(data["total_trials"], 80)
        self.assertIn("arm_aggregates", data)
        self.assertIn("trials", data)
        self.assertTrue(data["is_sample"])

        expected_arms = {"REAGENT_CONTRACT", "SHARED_OBSERVATION", "TOOL_CLAIM_ONLY", "NAIVE_RETRY", "ALWAYS_ESCALATE"}
        self.assertEqual(set(data["arm_aggregates"].keys()), expected_arms)

        reagent = data["arm_aggregates"]["REAGENT_CONTRACT"]
        self.assertEqual(reagent["false_verified_count"], 0,
                         "REAGENT_CONTRACT must have 0 false verifications")
        self.assertEqual(reagent["unsafe_retry_count"], 0,
                         "REAGENT_CONTRACT must have 0 unsafe retries")

        self.assertGreater(len(data["trials"]), 0)
        first_trial = data["trials"][0]
        for field_name in ("trial_id", "task_id", "fault_mode", "arm", "oracle_verdict"):
            self.assertIn(field_name, first_trial,
                          f"Trial missing expected field: {field_name}")

        # Verify submission artifact is unchanged
        from pathlib import Path
        artifact_path = Path("reports/dbops/comparative_evaluation_results.json")
        if artifact_path.exists():
            import hashlib
            with open(artifact_path, "rb") as f:
                content = f.read()
            # Just verify it is still valid JSON (not corrupted by the run)
            json.loads(content)


if __name__ == "__main__":
    unittest.main()
