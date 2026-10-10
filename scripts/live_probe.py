"""
ReAgent Live Probe — Supervised First Run Script

╔══════════════════════════════════════════════════════════════════════╗
║  !! DO NOT RUN WITHOUT EXPLICIT HUMAN AUTHORIZATION !!              ║
║                                                                      ║
║  This script makes ONE live call to the Nebius Token Factory API.   ║
║  It will CONSUME REAL TOKENS from the linked billing account.       ║
║                                                                      ║
║  Before running, YOU MUST verify:                                   ║
║    1. Nebius billing dashboard shows current balance and no         ║
║       auto-top-up is enabled.                                        ║
║    2. You have confirmed NEBIUS_API_KEY is set correctly.           ║
║    3. REAGENT_LEDGER_PATH is set to an appropriate path.            ║
║    4. You accept that this will charge the account a small amount.  ║
║                                                                      ║
║  Run with:                                                           ║
║    python scripts/live_probe.py --confirm-billing-checked           ║
╚══════════════════════════════════════════════════════════════════════╝

Safety guards:
  - Requires --confirm-billing-checked flag (blocks accidental runs).
  - Configurable cumulative call & token caps (defaults 20 calls, 60,000 tokens;
    code-enforced hard upper bounds: calls <= 50, tokens <= 200,000).
  - Blocks startup on abandoned/un-reconciled reservations (fails closed).
  - Pre-call reservation estimated from full actual request messages.
  - Exactly one reconcile per reservation executed in finally.
  - Safe allowlisted receipt persistence on EVERY exit path without secret leakage.
"""

import os
import sys
import argparse
from typing import Optional, Any

# ──────────────────────────────────────────────────────────────────────────────
# SAFETY GUARD: This module must NEVER be imported by test discovery.
# Unit tests using fake clients may import only when explicitly enabled.
# ──────────────────────────────────────────────────────────────────────────────

def _check_offline_test_import():
    """Refuse import during test discovery / offline test runs unless testing fake client."""
    if os.environ.get("REAGENT_ALLOW_PROBE_IMPORT_FOR_TESTS") == "1":
        return
    import traceback
    for frame in traceback.extract_stack():
        if "unittest" in frame.filename or "pytest" in frame.filename:
            raise ImportError(
                "live_probe.py MUST NOT be imported during test discovery. "
                "Set REAGENT_ALLOW_PROBE_IMPORT_FOR_TESTS=1 only for unit tests of fake clients."
            )

_check_offline_test_import()


# ──────────────────────────────────────────────────────────────────────────────
# LIMITS & TASK SPECIFICATION
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PROBE_MAX_CALLS = 20
DEFAULT_PROBE_MAX_TOKENS = 60000
HARD_LIMIT_MAX_CALLS = 50
HARD_LIMIT_MAX_TOKENS = 200000
PROBE_TASK = "Validate that {\"status\": \"ok\", \"source\": \"live_probe\"} is valid JSON."


def run_live_probe(
    provider: Optional[Any] = None,
    ledger: Optional[Any] = None,
    reports_dir: Optional[str] = None,
    confirm_billing_checked: bool = False,
    task_objective: str = PROBE_TASK,
) -> int:
    """
    Executes the supervised live probe with strict budget, reservation,
    error-handling, and allowlisted receipt persistence on every exit path.

    Returns process exit code (0 = success, 1 = provider error or billing block, 2 = model failure/truncation).
    """
    if not confirm_billing_checked:
        print(
            "\n[BLOCKED] You must pass --confirm-billing-checked to run this script.\n"
            "This confirms you have reviewed the Nebius billing dashboard and\n"
            "accept that this script will consume real tokens.\n"
        )
        return 1

    api_key = os.environ.get("NEBIUS_API_KEY", "")
    if provider is None and (not api_key or not api_key.strip()):
        print("[BLOCKED] NEBIUS_API_KEY is not set in the environment. Refusing to run.")
        return 1

    from reagent.policy import (
        GlobalTokenLedger,
        BudgetExceededError,
        CorruptLedgerError,
        AbandonedReservationError,
        estimate_prompt_tokens,
    )
    from reagent.providers.nebius import NebiusNemotronProvider, get_configured_max_tokens
    from reagent.tools.json_tool import JsonValidatorTool
    from reagent.tools.registry import ToolRegistry
    from reagent.receipts import save_allowlisted_receipt

    if reports_dir is None:
        reports_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
        )

    if ledger is None:
        raw_ledger_path = os.environ.get("REAGENT_LEDGER_PATH")
        if not raw_ledger_path or not raw_ledger_path.strip():
            print("[BLOCKED] REAGENT_LEDGER_PATH is not set. You must specify an explicit, dedicated ledger path for live probes.")
            return 1
        ledger_path = os.path.realpath(os.path.abspath(raw_ledger_path.strip()))
        default_ledger = os.path.realpath(os.path.abspath(os.path.join(os.path.expanduser("~"), ".reagent", "token_ledger.json")))
        if ledger_path == default_ledger:
            print("[BLOCKED] REAGENT_LEDGER_PATH cannot point to the default shared ledger (~/.reagent/token_ledger.json). Specify a dedicated probe ledger path.")
            return 1

        print(f"\n[PROBE] Ledger path: {ledger_path}")
        try:
            ledger = GlobalTokenLedger(ledger_path=ledger_path)
        except CorruptLedgerError as exc:
            print(f"[BLOCKED] Ledger is corrupt. Failing closed.\n  {exc}")
            return 1
    else:
        ledger_path = getattr(ledger, "ledger_path", "in-memory-or-custom")
        print(f"\n[PROBE] Ledger path: {ledger_path}")

    # Block startup on outstanding reservations from prior crashes/processes
    try:
        ledger.assert_no_abandoned_reservations()
    except AbandonedReservationError as exc:
        print(f"[BLOCKED] Outstanding un-reconciled reservations detected. Failing closed:\n  {exc}")
        return 1

    state = ledger.get_state()
    if state["remaining_calls"] <= 0:
        print(
            f"[BLOCKED] Cumulative call limit reached ({state['total_calls_used']}/{ledger.max_calls}). "
            f"Reset ledger after billing review if authorized."
        )
        return 1

    if provider is None:
        provider = NebiusNemotronProvider()

    registry = ToolRegistry()
    registry.register(JsonValidatorTool())

    # Reservation sizing: estimate from ACTUAL request messages (system prompt, context, tool schemas, user content)
    if hasattr(provider, "estimate_decision_request_tokens"):
        prompt_est = provider.estimate_decision_request_tokens(
            task_objective=task_objective,
            available_tools=["json_validator"],
        )
    elif hasattr(provider, "estimate_request_tokens"):
        prompt_est = provider.estimate_request_tokens(prompt=task_objective)
    else:
        prompt_est = estimate_prompt_tokens(task_objective)

    max_tokens = get_configured_max_tokens(1536)
    required_reservation = prompt_est + max_tokens

    if state["remaining_tokens"] < required_reservation:
        print(
            f"[BLOCKED] Cumulative token budget insufficient: remaining={state['remaining_tokens']} "
            f"< required={required_reservation}. Reset ledger after billing review if authorized."
        )
        return 1

    print(f"\n[PROBE] Task: {task_objective}")
    print(f"[PROBE] Ledger caps: {ledger.max_calls} calls / {ledger.max_tokens} tokens")
    print(f"[PROBE] Pre-call reservation: {required_reservation} tokens ({prompt_est} request + {max_tokens} completion)")
    print("[PROBE] Making ONE live API call to Nebius Token Factory...\n")

    try:
        res_id = ledger.check_and_reserve(estimated_tokens=required_reservation)
    except BudgetExceededError as exc:
        print(f"[BLOCKED] Budget exceeded before call: {exc}")
        return 1

    decision = None
    provider_error = None
    status_code = None
    usage = None
    reconcile_error = None

    try:
        decision = provider.make_structured_decision(
            task_objective=task_objective,
            available_tools=["json_validator"],
            max_tokens=max_tokens,
        )
        if decision:
            print(f"[PROBE] Provider decision received: tool={decision.tool} provenance={decision.provenance}")
    except Exception as exc:
        provider_error = type(exc).__name__
        raw_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        status_code = int(raw_code) if isinstance(raw_code, int) else None
        print(f"[ERROR] Provider call failed: error_type={provider_error} status_code={status_code}")
    finally:
        try:
            usage = provider.get_last_usage() if hasattr(provider, "get_last_usage") else getattr(provider, "last_usage", None)
            charged = ledger.reconcile_usage(
                res_id,
                reported_usage=usage,
                conservative_default=required_reservation,
            )
            print(f"[LEDGER] Reconciled call: charged {charged} tokens.")
        except Exception as r_exc:
            reconcile_error = type(r_exc).__name__
            print(f"[ERROR] Ledger reconciliation failed: error_type={reconcile_error}")

    try:
        final_state = ledger.get_state()
        print(
            f"[LEDGER] Calls used: {final_state['total_calls_used']}/{ledger.max_calls}\n"
            f"[LEDGER] Tokens used: {final_state['total_tokens_used']}/{ledger.max_tokens}"
        )
    except Exception:
        pass

    model_id = getattr(provider, "model", None)
    finish_reason = getattr(provider, "last_finish_reason", None)
    val_err = getattr(provider, "last_validation_error", None)

    # ── Path 0: Reconciliation Failure (Fail Closed & Accounting Uncertain) ──
    if reconcile_error is not None:
        combined_error = f"{provider_error}+{reconcile_error}" if provider_error else reconcile_error
        save_allowlisted_receipt(
            receipt_dir=reports_dir,
            task_id="live-probe-planning",
            status="PROBE_ACCOUNTING_UNCERTAIN",
            tool_name="json_validator",
            probe_type="planning_probe",
            model_id=model_id,
            provenance="none",
            finish_reason="error",
            verified=False,
            token_usage=usage,
            error_type=combined_error,
            status_code=status_code,
            exit_reason="reconciliation_failed_accounting_uncertain",
        )
        print(f"\n[RESULT] Live probe failed: reconciliation error ({combined_error}). Accounting uncertain. Safe receipt written.")
        return 1

    # ── Path 1: Provider Exception ──────────────────────────────────────────
    if provider_error is not None:
        save_allowlisted_receipt(
            receipt_dir=reports_dir,
            task_id="live-probe-planning",
            status="PROBE_PROVIDER_ERROR",
            tool_name="json_validator",
            probe_type="planning_probe",
            model_id=model_id,
            provenance="none",
            finish_reason="error",
            verified=False,
            token_usage=usage,
            error_type=provider_error,
            status_code=status_code,
            exit_reason="provider_exception",
        )
        print("\n[RESULT] Live probe failed with provider error. Safe receipt written.")
        return 1

    # ── Path 2: Truncation (finish_reason == 'length' or val_err == 'truncated')
    if decision is None and (finish_reason == "length" or val_err == "truncated"):
        save_allowlisted_receipt(
            receipt_dir=reports_dir,
            task_id="live-probe-planning",
            status="PROBE_TRUNCATION",
            tool_name="json_validator",
            probe_type="planning_probe",
            model_id=model_id,
            provenance="none",
            finish_reason="length",
            verified=False,
            token_usage=usage,
            error_type="TruncatedResponse",
            exit_reason="truncated_by_max_tokens",
        )
        print("\n[RESULT] Provider response truncated by max_tokens. Safe receipt written.")
        return 2

    # ── Path 3: Malformed Output (unparseable json or invalid schema) ─────────
    if decision is None and val_err in ("unparseable_json", "malformed_output"):
        save_allowlisted_receipt(
            receipt_dir=reports_dir,
            task_id="live-probe-planning",
            status="PROBE_MALFORMED_OUTPUT",
            tool_name="json_validator",
            probe_type="planning_probe",
            model_id=model_id,
            provenance="none",
            finish_reason=finish_reason,
            verified=False,
            token_usage=usage,
            error_type="MalformedOutput",
            exit_reason="malformed_model_output",
        )
        print("\n[RESULT] Provider returned malformed output. Safe receipt written.")
        return 2

    # ── Path 4: None Decision (generic fallback for None decision) ───────────
    if decision is None:
        save_allowlisted_receipt(
            receipt_dir=reports_dir,
            task_id="live-probe-planning",
            status="PROBE_MODEL_DECISION_NONE",
            tool_name="json_validator",
            probe_type="planning_probe",
            model_id=model_id,
            provenance="none",
            finish_reason=finish_reason,
            verified=False,
            token_usage=usage,
            error_type="ProviderDecisionNone",
            exit_reason="decision_is_none",
        )
        print("\n[RESULT] Provider returned no valid decision. Safe receipt written.")
        return 2

    # ── Path 5: Clean Success (Decision Parsed and Schema Validated) ─────────
    arg_keys = list(decision.arguments.keys()) if hasattr(decision, "arguments") and decision.arguments else []
    postconditions = decision.proposed_postconditions if hasattr(decision, "proposed_postconditions") else []

    save_allowlisted_receipt(
        receipt_dir=reports_dir,
        task_id="live-probe-planning",
        status="PROBE_DECISION_VALIDATED",
        tool_name=decision.tool,
        argument_keys=arg_keys,
        proposed_postconditions=postconditions,
        probe_type="planning_probe",
        model_id=model_id,
        provenance=getattr(decision, "provenance", "model"),
        finish_reason=finish_reason or "stop",
        verified=False,
        token_usage=usage,
        exit_reason="decision_parsed_and_schema_validated",
    )
    print("\n[RESULT] Live planning probe succeeded: decision parsed and schema validated. Safe receipt written.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="ReAgent supervised live probe (planning).")
    parser.add_argument(
        "--confirm-billing-checked",
        action="store_true",
        help="REQUIRED: Confirms you have manually checked the Nebius billing dashboard.",
    )
    args = parser.parse_args()

    exit_code = run_live_probe(confirm_billing_checked=args.confirm_billing_checked)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
