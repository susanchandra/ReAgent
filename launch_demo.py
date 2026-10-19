#!/usr/bin/env python3
"""
launch_demo.py — Unified Entry Point for ReAgent Evaluation & Demos.

Usage:
    # 1. Launch Interactive Mission Control Dashboard (Default):
    python launch_demo.py

    # 2. Launch Mission Control and open default browser:
    python launch_demo.py --open

    # 3. Run Fast Terminal Judge Demo (CLI, sub-4 seconds):
    python launch_demo.py --cli

    # 4. Run Release Verification Suite (Clean-start + UI tests + Manifest):
    python launch_demo.py --check

Guarantees:
    - 100% Offline & Deterministic: Zero external network requests on load.
    - Zero Secret Exposure: Never logs, displays, or transmits API keys.
    - Zero Silent Paid API Calls: Fails closed to offline deterministic mode.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

# Enforce repository root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def check_prerequisites() -> bool:
    """Validate runtime environment and core dependencies."""
    if sys.version_info < (3, 10):
        print(
            f"[ERROR] Python 3.10 or higher is required. Detected: {sys.version.split()[0]}",
            file=sys.stderr,
        )
        return False

    missing = []
    for pkg in ["pydantic", "filelock"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(
            f"[ERROR] Missing required packages: {', '.join(missing)}\n"
            f"[INFO] Install dependencies via: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return False

    return True


def run_cli_demo() -> int:
    """Execute the sub-4-second terminal judge demonstration."""
    print("\n--- Launching ReAgent Unified Terminal Judge Demo ---\n")
    try:
        from reagent.experiments.dbops.judge_demo import run_judge_demo
        run_judge_demo(regenerate_benchmark=False)
        return 0
    except Exception as exc:
        print(f"[ERROR] Terminal demo execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def run_checks() -> int:
    """Execute release-blocking verification checks."""
    import unittest

    print("\n==================================================================")
    print("      ReAgent: Release Verification Gate Checks")
    print("==================================================================")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load UI server tests, clean start tests, and manifest validator
    for test_module in [
        "tests.test_ui_server",
        "tests.test_clean_start",
        "tests.test_submission_manifest_validator",
    ]:
        try:
            mod_suite = loader.loadTestsFromName(test_module)
            suite.addTests(mod_suite)
        except Exception as exc:
            print(f"[ERROR] Could not load test module {test_module}: {exc}", file=sys.stderr)
            return 1

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def start_mission_control(port: int = 8080, open_browser: bool = False) -> int:
    """Launch the Mission Control web server."""
    from reagent.ui_server import start_server

    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        start_server(port=port)
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to start Mission Control server: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ReAgent: Outcome Verification & Safe Recovery for AI Agent Actions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve Mission Control UI (default: 8080)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open Mission Control in default web browser",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the fast sub-4-second terminal judge demonstration",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run core release verification checks (UI, clean-start, manifest)",
    )

    args = parser.parse_args()

    if not check_prerequisites():
        return 1

    if args.cli:
        return run_cli_demo()

    if args.check:
        return run_checks()

    return start_mission_control(port=args.port, open_browser=args.open)


if __name__ == "__main__":
    sys.exit(main())
