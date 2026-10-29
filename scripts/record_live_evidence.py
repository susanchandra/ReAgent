"""
scripts/record_live_evidence.py — Supervised Live Nemotron Evidence Recorder

Allows the project owner to explicitly perform ONE genuine live inference call
to Nebius Token Factory (model: nvidia/nemotron-3-super-120b-a12b), validate
the proposed contract against ReAgent's semantic rules, and save an auditable,
sanitized evidence record to reports/evidence/nemotron_live_evidence.json.

Safety Guarantees:
- REQUIRES the explicit flag --live-run-opt-in.
- Does NOT run automatically.
- Sanitizes all secrets, API keys, and authorization headers before saving.
- Labels provenance honestly (GENUINE_LIVE_INFERENCE).
"""

import os
import sys
import argparse
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reagent.providers.nebius import NebiusNemotronProvider, DEFAULT_MODEL, DEFAULT_BASE_URL
from reagent.contract_validator import validate_contract_proposal
from reagent.evidence_record import create_evidence_record, save_sanitized_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Record sanitized live Nemotron contract evidence.")
    parser.add_argument(
        "--live-run-opt-in",
        action="store_true",
        help="Explicit human opt-in to authorize exactly ONE live Nebius API inference call.",
    )
    parser.add_argument(
        "--output",
        default="reports/evidence/nemotron_live_evidence.json",
        help="Output path for the sanitized evidence record.",
    )
    args = parser.parse_args()

    if not args.live_run_opt_in:
        print("[-] Live execution refused: --live-run-opt-in is required to make live network calls.")
        print("    Run: python scripts/record_live_evidence.py --live-run-opt-in")
        return 1

    api_key = os.environ.get("NEBIUS_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        print("[-] NEBIUS_API_KEY environment variable is not set. Cannot perform live inference.")
        return 1

    print("[*] Initiating single live Nemotron contract proposal request via Nebius Token Factory...")
    print(f"    Target Model: {DEFAULT_MODEL}")
    print(f"    Target Endpoint: {DEFAULT_BASE_URL}")

    t0 = time.time()
    task_objective = (
        "Book Room 14B at Grand Hotel for Guest Maya Lin (booking_id: BKG-NM-9901, room_number: 14B, Amount: $350.00). "
        "Propose success contract with tool arguments booking_id and room_number, and independent postconditions "
        "including KEY_EXISTS (target: booking_id, expected: BKG-NM-9901) and VALUE_EQUALS (target: room_number, expected: 14B). "
        "Generic STATUS_SUCCESS without concrete observable state properties is strictly forbidden."
    )
    trusted_context = {"booking_id": "BKG-NM-9901", "room_number": "14B"}

    try:
        provider = NebiusNemotronProvider(max_retries=0, timeout=20.0)
        decision = provider.make_structured_decision(
            task_objective=task_objective,
            available_tools=["hotel_booking_http"],
            max_tokens=768,
        )
        latency_ms = int((time.time() - t0) * 1000)

        schema_valid = decision is not None
        contract_dict = None
        grounding_valid = None
        grounding_errors = None

        if decision is not None:
            contract_dict = decision.model_dump() if hasattr(decision, "model_dump") else dict(decision)
            val_res = validate_contract_proposal(
                contract_dict=contract_dict,
                task_objective=task_objective,
                workflow="hotel_booking",
                task_context=trusted_context,
            )
            grounding_valid = val_res.is_valid
            if not grounding_valid:
                grounding_errors = "; ".join(val_res.errors)

        validation_status = (
            "CONTRACT_VALIDATED" if (schema_valid and grounding_valid)
            else "GROUNDING_FAILED" if schema_valid
            else "SCHEMA_INVALID"
        )

        record = create_evidence_record(
            model_id=provider.model,
            provider_name="nebius_nemotron",
            run_id=f"live_nemotron_{int(time.time())}",
            is_live=True,
            task_objective=task_objective,
            workflow="hotel_booking",
            raw_proposal=contract_dict,
            schema_valid=schema_valid,
            grounding_valid=grounding_valid,
            contract_validation_status=validation_status,
            failure_reason=grounding_errors or provider.last_validation_error,
            latency_ms=latency_ms,
            token_usage=provider.last_usage or {},
            endpoint_host=DEFAULT_BASE_URL,
        )

        saved_path = save_sanitized_evidence(record, args.output)
        print(f"[+] Successfully captured and saved sanitized live evidence to: {saved_path}")
        print(f"    Validation Status: {validation_status}")
        print(f"    Latency: {latency_ms} ms")
        print(f"    Model: {record.model_id}")
        return 0

    except Exception as exc:
        latency_ms = int((time.time() - t0) * 1000)
        print(f"[-] Live inference failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
