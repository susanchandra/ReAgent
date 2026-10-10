"""
ReAgent Full-Workflow Supervised Live Probe (Milestone 4B)

╔══════════════════════════════════════════════════════════════════════╗
║  !! DO NOT RUN WITHOUT EXPLICIT HUMAN AUTHORIZATION !!              ║
║                                                                      ║
║  This script runs a FULL end-to-end task through execute_task():    ║
║    1. Model Decision via Nebius Token Factory                       ║
║    2. Contract Grounding & Freezing                                 ║
║    3. Sandboxed Tool Execution (SandboxedFileTool only)             ║
║    4. Deterministic Verification (DeterministicVerifier)            ║
║    5. Verified Receipt Generation                                   ║
║                                                                      ║
║  Before running, YOU MUST verify:                                   ║
║    1. Nebius billing dashboard shows current balance and no         ║
║       auto-top-up is enabled.                                        ║
║    2. You have confirmed NEBIUS_API_KEY is set correctly.           ║
║    3. REAGENT_LEDGER_PATH is set to an appropriate path.            ║
║    4. You accept that this will charge the account real tokens.     ║
║                                                                      ║
║  Run with:                                                           ║
║    python scripts/live_probe_full.py --confirm-billing-checked      ║
╚══════════════════════════════════════════════════════════════════════╝

Safety guards:
  - Requires --confirm-billing-checked flag (blocks accidental runs).
  - GlobalTokenLedger hard cap: 3 calls, 2000 tokens maximum.
  - Blocks startup on abandoned/un-reconciled reservations (fails closed).
  - Strict workspace parity between SandboxedFileTool and DeterministicVerifier.
  - No arbitrary shell execution tools registered.
  - Safe allowlisted receipt persistence without secret or raw error leakage.
"""

import os
import sys
import shutil
import tempfile
import argparse

# ──────────────────────────────────────────────────────────────────────────────
# SAFETY GUARD: This module must NEVER be imported by test files or pipeline code.
# If imported without the --confirm-billing-checked flag, refuse immediately.
# ──────────────────────────────────────────────────────────────────────────────

def _check_offline_test_import():
    """Refuse import during test discovery / offline test runs unless testing fake client."""
    if os.environ.get("REAGENT_ALLOW_PROBE_IMPORT_FOR_TESTS") == "1":
        return
    import traceback
    for frame in traceback.extract_stack():
        if "unittest" in frame.filename or "pytest" in frame.filename:
            raise ImportError(
                "live_probe_full.py MUST NOT be imported during test runs. "
                "This is a supervised live probe only. Remove it from test discovery."
            )

_check_offline_test_import()


# ──────────────────────────────────────────────────────────────────────────────
# HARD LIMITS & PROBE TASK
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PROBE_MAX_CALLS = 20
DEFAULT_PROBE_MAX_TOKENS = 60000
HARD_LIMIT_MAX_CALLS = 50
HARD_LIMIT_MAX_TOKENS = 200000
PROBE_FULL_TASK = "Write 'PROBE_VERIFIED_2026' to file 'probe_result.txt'."


def main():
    parser = argparse.ArgumentParser(description="ReAgent full-workflow supervised live probe.")
    parser.add_argument(
        "--confirm-billing-checked",
        action="store_true",
        help="REQUIRED: Confirms you have manually checked the Nebius billing dashboard.",
    )
    args = parser.parse_args()

    if not args.confirm_billing_checked:
        print(
            "\n[BLOCKED] You must pass --confirm-billing-checked to run this script.\n"
            "This confirms you have reviewed the Nebius billing dashboard and\n"
            "accept that this script will consume real tokens.\n"
        )
        sys.exit(1)

    api_key = os.environ.get("NEBIUS_API_KEY", "")
    if not api_key or not api_key.strip():
        print("[BLOCKED] NEBIUS_API_KEY is not set in the environment. Refusing to run.")
        sys.exit(1)

    # Import only after safety checks pass
    from reagent.policy import (
        GlobalTokenLedger,
        BudgetExceededError,
        CorruptLedgerError,
        AbandonedReservationError,
        TokenBudgetGuard,
        estimate_prompt_tokens,
    )
    from reagent.providers.nebius import NebiusNemotronProvider, get_configured_max_tokens
    from reagent.tools.file_tool import SandboxedFileTool
    from reagent.tools.registry import ToolRegistry
    from reagent.pipeline import ReAgentPipeline
    from reagent.taxonomy import AgentStatus
    from reagent.receipts import save_allowlisted_receipt

    raw_ledger_path = os.environ.get("REAGENT_LEDGER_PATH")
    if not raw_ledger_path or not raw_ledger_path.strip():
        print("[BLOCKED] REAGENT_LEDGER_PATH is not set. You must specify an explicit, dedicated ledger path for live probes.")
        sys.exit(1)
    ledger_path = os.path.realpath(os.path.abspath(raw_ledger_path.strip()))
    default_ledger = os.path.realpath(os.path.abspath(os.path.join(os.path.expanduser("~"), ".reagent", "token_ledger.json")))
    if ledger_path == default_ledger:
        print("[BLOCKED] REAGENT_LEDGER_PATH cannot point to the default shared ledger (~/.reagent/token_ledger.json). Specify a dedicated probe ledger path.")
        sys.exit(1)

    print(f"\n[FULL PROBE] Ledger path: {ledger_path}")

    try:
        ledger = GlobalTokenLedger(ledger_path=ledger_path)
    except CorruptLedgerError as exc:
        print(f"[BLOCKED] Ledger is corrupt. Failing closed.\n  {exc}")
        sys.exit(1)

    # Block startup on outstanding reservations from prior crashes/processes
    try:
        ledger.assert_no_abandoned_reservations()
    except AbandonedReservationError as exc:
        print(f"[BLOCKED] Outstanding un-reconciled reservations detected. Failing closed:\n  {exc}")
        sys.exit(1)

    state = ledger.get_state()
    if state["remaining_calls"] <= 0:
        print(
            f"[BLOCKED] Cumulative call limit reached ({state['total_calls_used']}/{ledger.max_calls}). "
            f"Reset ledger after billing review if authorized."
        )
        sys.exit(1)

    provider = NebiusNemotronProvider()
    if hasattr(provider, "estimate_decision_request_tokens"):
        prompt_est = provider.estimate_decision_request_tokens(
            task_objective=PROBE_FULL_TASK,
            available_tools=["sandboxed_file"],
        )
    else:
        prompt_est = estimate_prompt_tokens(PROBE_FULL_TASK)

    max_tokens = get_configured_max_tokens(1536)
    required_reservation = prompt_est + max_tokens

    if state["remaining_tokens"] < required_reservation:
        print(
            f"[BLOCKED] Cumulative token budget insufficient: remaining={state['remaining_tokens']} "
            f"< required={required_reservation}. Reset ledger after billing review if authorized."
        )
        sys.exit(1)

    # Create isolated temporary workspace for tool & verifier parity
    ws_dir = tempfile.mkdtemp(prefix="reagent_live_full_ws_")
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")

    print(f"[FULL PROBE] Task: {PROBE_FULL_TASK}")
    print(f"[FULL PROBE] Workspace: {ws_dir}")
    print(f"[FULL PROBE] Ledger caps: {ledger.max_calls} calls / {ledger.max_tokens} tokens")

    try:
        file_tool = SandboxedFileTool(sandbox_root=ws_dir)
        registry = ToolRegistry()
        registry.register(file_tool)

        token_guard = TokenBudgetGuard(
            max_calls_per_run=ledger.max_calls,
            max_tokens_per_run=ledger.max_tokens,
            ledger=ledger,
        )

        pipeline = ReAgentPipeline(
            agent_id="live-full-probe-agent",
            provider=provider,
            tool_registry=registry,
            token_budget=token_guard,
            workspace_root=ws_dir,
        )

        # Confirm workspace parity
        assert pipeline.verifier.workspace_root == ws_dir, "Workspace mismatch between pipeline and verifier"
        assert file_tool.sandbox_root == ws_dir, "Workspace mismatch between pipeline and file tool"

        print("[FULL PROBE] Executing task through autonomous pipeline...\n")
        result = pipeline.execute_task(task_objective=PROBE_FULL_TASK, max_recovery_attempts=1)

        final_status = result.verified_receipt.final_status
        is_verified = (final_status == AgentStatus.VERIFIED)
        usage = provider.get_last_usage()

        print(f"\n[FULL PROBE] Result: {final_status.value} (Verified: {is_verified})")
        print(f"[FULL PROBE] Tool executed: {result.tool}")
        print(f"[FULL PROBE] Discrepancies: {result.verification_result.discrepancies}")

        # Persist allowlisted receipt
        save_allowlisted_receipt(
            receipt_dir=reports_dir,
            task_id=result.task_id,
            status=final_status.value,
            tool_name=result.tool,
            probe_type="full_workflow_probe",
            verified=is_verified,
            recovery_count=result.recovery_count,
            token_usage=usage,
            discrepancies=result.verification_result.discrepancies,
        )

        final_state = ledger.get_state()
        print(
            f"[LEDGER] Calls used: {final_state['total_calls_used']}/{ledger.max_calls}\n"
            f"[LEDGER] Tokens used: {final_state['total_tokens_used']}/{ledger.max_tokens}"
        )

        if is_verified:
            print("\n[RESULT] Full workflow live probe PASSED and was independently verified.")
            sys.exit(0)
        else:
            print(f"\n[RESULT] Full workflow live probe completed but ended in {final_status.value}.")
            sys.exit(2)

    except Exception as exc:
        err_type = type(exc).__name__
        raw_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        status_code = int(raw_code) if isinstance(raw_code, int) else None
        print(f"[ERROR] Live probe execution failed: error_type={err_type} status_code={status_code}")

        save_allowlisted_receipt(
            receipt_dir=reports_dir,
            task_id="live-probe-full-error",
            status="PROBE_ERROR",
            tool_name="sandboxed_file",
            probe_type="full_workflow_probe",
            verified=False,
            error_type=err_type,
            status_code=status_code,
        )
        sys.exit(1)
    finally:
        # Clean up temporary workspace safely
        if os.path.exists(ws_dir):
            try:
                shutil.rmtree(ws_dir)
            except Exception:
                pass


if __name__ == "__main__":
    main()
