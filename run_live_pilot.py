"""
run_live_pilot.py — Supervised live pilot execution launcher.

USAGE:
    # 1. Offline dry-run preflight (default, zero network calls):
    python run_live_pilot.py

    # 2. Live execution (requires explicit opt-in flag AND environment key):
    python run_live_pilot.py --live-run-opt-in

HARD SAFETY CONSTRAINTS:
1. Pure import: Importing this module executes NO network calls, NO subprocesses,
   and NO top-level logic.
2. Explicit Opt-In: Live execution requires the explicit CLI flag `--live-run-opt-in`.
   Live mode is NEVER inferred from the presence of NEBIUS_API_KEY in the environment.
3. Preflight Enforcement: An offline dry-run preflight is ALWAYS executed first and
   must succeed with PREFLIGHT_READY before any live provider call can occur.
4. Fail-Closed: If NEBIUS_API_KEY is unset or unconfigured, execution fails closed
   immediately before any dispatch.
5. Zero Retries & 6-Request Ceiling: Enforced by the underlying durable ledger.
6. Secret Sanitization: API keys are never printed, logged, or included in reports.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reagent.experiments.dbops.pilot_eval import (
    PilotEvaluationRunner,
    PILOT_TOTAL_SAMPLES,
    PILOT_TASK_IDS,
    PILOT_TARGET_MODEL,
    PILOT_TEMPERATURE,
    PILOT_MAX_TOKENS,
)
from reagent.experiments.dbops.pilot_analysis import PilotEvidenceAnalyzer
from reagent.providers.nebius import NebiusNemotronProvider


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Supervised Runner for DBOPS Nemotron Pilot Evaluation."
    )
    parser.add_argument(
        "--live-run-opt-in",
        action="store_true",
        default=False,
        help="Explicit CLI opt-in required to execute live model requests. "
             "Without this flag, runs offline preflight with zero network calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Explicitly request offline dry-run preflight.",
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("reports/dbops"),
        help="Directory for frozen configuration, append-only ledger, and reports.",
    )
    parser.add_argument(
        "--skip-countdown",
        action="store_true",
        default=False,
        help="Skip the 10-second abort countdown (for automated tests only).",
    )

    args = parser.parse_args(argv)

    # ── Path 1: Default / Dry-Run (Zero Network Calls) ──────────────────────
    if not args.live_run_opt_in or args.dry_run:
        print("[PILOT-LAUNCHER] Mode: OFFLINE DRY-RUN PREFLIGHT (zero network calls)")
        preflight_runner = PilotEvaluationRunner(pilot_dir=args.pilot_dir, live_opt_in=False)
        summary = preflight_runner.run_pilot()
        preflight = summary.task_summaries.get("PREFLIGHT_ONLY", {})
        status = preflight.get("status", "UNKNOWN")
        print(f"[PILOT-LAUNCHER] Preflight status: {status}")
        print(f"[PILOT-LAUNCHER] Config fingerprint: {preflight.get('config_fingerprint')}")
        print(f"[PILOT-LAUNCHER] Network calls made: {preflight.get('network_calls_made', 0)}")
        print(f"[PILOT-LAUNCHER] Paid tokens consumed: {preflight.get('paid_tokens_consumed', 0)}")
        print(f"[PILOT-LAUNCHER] Total planned samples: {preflight.get('total_planned_samples', 0)}")

        if status != "PREFLIGHT_READY":
            print(f"[PILOT-LAUNCHER] ERROR: Preflight failed with status: {status}")
            return 1

        print("[PILOT-LAUNCHER] Preflight check completed cleanly. Ready for authorization.")
        return 0

    # ── Path 2: Live Execution (Explicit Opt-In Required) ───────────────────
    print("[PILOT-LAUNCHER] Mode: LIVE EXECUTION REQUESTED (--live-run-opt-in provided)")

    # Guard 1: Verify API key is present in environment without logging value
    api_key_present = bool(os.environ.get("NEBIUS_API_KEY", "").strip())
    if not api_key_present:
        print("ERROR: NEBIUS_API_KEY environment variable is not set.")
        print("Live execution refused. Fails closed.")
        return 1

    print("[PILOT-LAUNCHER] NEBIUS_API_KEY is present in environment (value redacted).")
    print(f"[PILOT-LAUNCHER] Target model: {PILOT_TARGET_MODEL}")
    print(f"[PILOT-LAUNCHER] Sampling: temp={PILOT_TEMPERATURE}, max_tokens={PILOT_MAX_TOKENS}")
    print(f"[PILOT-LAUNCHER] Hard request ceiling: {PILOT_TOTAL_SAMPLES} dispatch attempts (zero automatic retries)")
    print("[PILOT-LAUNCHER] Planned sample IDs:")
    for task_id in PILOT_TASK_IDS:
        for s in range(2):
            print(f"  pilot_{task_id}_s{s}")

    # Guard 2: Run offline preflight FIRST and verify clean result
    print("\n[PILOT-LAUNCHER] Step 1: Running mandatory offline preflight verification...")
    preflight_runner = PilotEvaluationRunner(pilot_dir=args.pilot_dir, live_opt_in=False)
    summary = preflight_runner.run_pilot()
    preflight = summary.task_summaries.get("PREFLIGHT_ONLY", {})
    if preflight.get("status") != "PREFLIGHT_READY":
        print(f"[PILOT-LAUNCHER] FATAL: Mandatory preflight check failed: {preflight}")
        return 1
    if preflight.get("network_calls_made", -1) != 0:
        print("[PILOT-LAUNCHER] FATAL: Preflight made non-zero network calls!")
        return 1
    print("[PILOT-LAUNCHER] Preflight check PASSED.")

    # Guard 3: Verify provider configuration
    print("\n[PILOT-LAUNCHER] Step 2: Verifying provider configuration...")
    provider = NebiusNemotronProvider()
    if not provider.is_configured():
        print("[PILOT-LAUNCHER] FATAL: Provider reports is_configured=False. Fails closed.")
        return 1
    print("[PILOT-LAUNCHER] Provider is_configured: True")

    # Safety countdown (can be interrupted by operator)
    if not args.skip_countdown:
        print("\n[PILOT-LAUNCHER] WARNING: Live paid requests will begin in 10 seconds.")
        print("[PILOT-LAUNCHER] Press Ctrl+C NOW to abort without any provider calls...")
        try:
            for remaining in range(10, 0, -1):
                sys.stdout.write(f"\r[PILOT-LAUNCHER] Starting in {remaining}s... ")
                sys.stdout.flush()
                time.sleep(1)
            print("\r[PILOT-LAUNCHER] Countdown complete. Dispatching requests.        ")
        except KeyboardInterrupt:
            print("\n[PILOT-LAUNCHER] Execution ABORTED by operator. Zero live requests dispatched.")
            return 130

    # Guard 4: Execute guarded live pilot
    print(f"\n[PILOT-LAUNCHER] Step 3: Executing {PILOT_TOTAL_SAMPLES}-sample live pilot...")
    t0 = time.time()
    live_runner = PilotEvaluationRunner(pilot_dir=args.pilot_dir, live_opt_in=True, provider=provider)
    live_summary = live_runner.run_pilot()
    elapsed = time.time() - t0

    print(f"\n[PILOT-LAUNCHER] Pilot completed in {elapsed:.1f}s.")
    print(f"[PILOT-LAUNCHER] Total samples attempted: {live_summary.total_samples_attempted}")
    print(f"[PILOT-LAUNCHER] Schema valid samples:    {live_summary.schema_valid_samples}")
    print(f"[PILOT-LAUNCHER] Grounded samples:        {live_summary.grounded_samples}")
    print(f"[PILOT-LAUNCHER] Gold equivalent samples: {live_summary.gold_equivalent_samples}")
    print(f"[PILOT-LAUNCHER] Passing tasks count:     {live_summary.passing_tasks_count}")
    print(f"[PILOT-LAUNCHER] Overall gate passed:     {live_summary.overall_gate_passed}")

    # Guard 5: Run read-only analyzer on the resulting ledger
    print("\n[PILOT-LAUNCHER] Step 4: Running M9 PilotEvidenceAnalyzer...")
    analyzer = PilotEvidenceAnalyzer()
    config_file = args.pilot_dir / "pilot_config_frozen.json"
    ledger_file = args.pilot_dir / "pilot_ledger.jsonl"
    report = analyzer.analyze(config_path=config_file, ledger_path=ledger_file)

    report_json = args.pilot_dir / "pilot_analysis_report.json"
    report_md = args.pilot_dir / "pilot_analysis_report.md"
    analyzer.write_json_report(report, report_json)
    analyzer.write_markdown_report(report, report_md)

    print(f"[PILOT-LAUNCHER] Analysis report JSON: {report_json}")
    print(f"[PILOT-LAUNCHER] Analysis report MD:   {report_md}")
    print(f"[PILOT-LAUNCHER] Gate verdict:         {report.overall_gate_verdict}")
    print(f"[PILOT-LAUNCHER] Pilot executed:       {report.pilot_executed}")

    return 0 if live_summary.overall_gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
