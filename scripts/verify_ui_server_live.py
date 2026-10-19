"""
verify_ui_server_live.py — Programmatic validation of ReAgent Mission Control server.
Exercises root HTML rendering, preflight configuration, 5-scenario execution,
oracle reveal, and reset endpoint.
"""
import json
import threading
import time
import urllib.request
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from http.server import HTTPServer
from reagent.ui_server import DashboardHTTPHandler

def run_checks():
    server = HTTPServer(('127.0.0.1', 8080), DashboardHTTPHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    time.sleep(0.5)

    try:
        print("=== 1. VERIFY ROOT GET / ===")
        with urllib.request.urlopen("http://127.0.0.1:8080/") as r:
            html = r.read().decode("utf-8")
            assert "512 tokens" not in html, "Stale '512 tokens' found in HTML"
            assert "768 tokens" in html, "768 tokens missing from HTML"
            assert "Checking…" not in html and "Checking..." not in html, "Stale Checking string found in HTML"
            assert "Run Recovery Simulation" in html, "Run Recovery Simulation button missing"
            assert "0 live calls on page load" in html, "0 live calls notice missing"
            print("  [PASS] HTML verified: 768 tokens present, 512 absent, no Checking flicker, offline labels clear.")

        print("\n=== 2. VERIFY /api/model_preflight ===")
        with urllib.request.urlopen("http://127.0.0.1:8080/api/model_preflight") as r:
            pre = json.loads(r.read().decode("utf-8"))
            print("  Preflight response:", json.dumps(pre, indent=2))
            assert pre["max_tokens"] == 768, f"Expected 768, got {pre['max_tokens']}"
            assert isinstance(pre["api_key_present"], bool)
            assert pre["network_calls_allowed"] is False
            print("  [PASS] Preflight endpoint clean and accurate (max_tokens=768).")

        print("\n=== 3. VERIFY SCENARIOS A, B, C, D, E SIMULATION ===")
        for sc in ["A", "B", "C", "D", "E"]:
            req = urllib.request.Request(f"http://127.0.0.1:8080/api/run_recovery_demo?scenario={sc}", method="POST")
            with urllib.request.urlopen(req) as r:
                res = json.loads(r.read().decode("utf-8"))
                naive = res["naive_retry_arm"]
                reagent = res["reagent_arm"]
                print(f"  Scenario {sc}: Naive={naive['final_status']} (bal=${naive['final_balance']:.2f}, dup={naive['duplicate_occurred']}) | ReAgent={reagent['final_status']} (bal=${reagent['final_balance']:.2f}, dup={reagent['duplicate_occurred']})")
                assert "run_id" in res

        print("\n=== 4. VERIFY SCENARIO D ORACLE REVEAL ===")
        with urllib.request.urlopen("http://127.0.0.1:8080/api/oracle_reveal?scenario=D") as r:
            orc = json.loads(r.read().decode("utf-8"))
            print("  Oracle D truth:", orc.get("oracle_truth"))
            assert orc.get("oracle_available") is True

        print("\n=== 5. VERIFY TASK TEMPLATES (HOTEL BOOKING & DATABASE UPDATE) ===")
        for tpl in ["hotel_booking", "database_update"]:
            req = urllib.request.Request(f"http://127.0.0.1:8080/api/run_recovery_demo?task_template={tpl}", method="POST")
            with urllib.request.urlopen(req) as r:
                res = json.loads(r.read().decode("utf-8"))
                reagent = res["reagent_arm"]
                naive = res["naive_retry_arm"]
                assert res.get("task_template") == tpl
                assert reagent["final_status"] == "VERIFIED"
                assert reagent["contract_postconditions_verified"] is True
                assert reagent["tool_attempts"] == 1
                assert reagent["probe_queries"] == 1
                assert naive["tool_attempts"] == 2
                print(f"  [PASS] Task template '{tpl}': ReAgent VERIFIED (attempts=1, probes=1), Naive (attempts=2, probes=0).")

        print("\n=== 6. VERIFY BENCHMARK RECONCILIATION ENDPOINT ===")
        with urllib.request.urlopen("http://127.0.0.1:8080/api/benchmark") as r:
            bm = json.loads(r.read().decode("utf-8"))
            assert bm["total_paired_configurations"] == 480
            assert "REAGENT_CONTRACT" in bm["arm_aggregates"]
            print(f"  [PASS] Benchmark endpoint returned {bm['total_paired_configurations']} paired configurations.")

        print("\n=== 7. VERIFY RESET ENDPOINT ===")
        req = urllib.request.Request("http://127.0.0.1:8080/api/reset", method="POST")
        with urllib.request.urlopen(req) as r:
            rst = json.loads(r.read().decode("utf-8"))
            print("  Reset status:", rst)
            assert rst.get("status") == "reset"

        print("\n>>> ALL 7 ENDPOINT AND TEMPLATE CHECKS PASSED WITH 100% FIDELITY! <<<")
    finally:
        server.shutdown()
        server.server_close()

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_checks()

