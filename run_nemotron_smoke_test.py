"""
run_nemotron_smoke_test.py — Dedicated One-Shot Live Inference Smoke Test.

Fulfills Final Readiness Gate Requirements:
1. Harmless task: Proposes a structured, read-only verification contract
   for a fictional inventory record (INV-ITEM-9842 in warehouse WH-A12, count 42).
   Zero mentions of wallets, transfers, balances, or financial transactions.
2. Zero side effects: No tools are executed. Strictly read-only schema proposal.
3. Bounded resources: timeout=15.0s, max_tokens=768 (calibrated for complete structured
   contract output; 256 caused TRUNCATED_OUTPUT as real model returned 342 tokens), temperature=0.0.
4. Zero automatic retries: max_retries=0 at SDK client and application level.
5. Invariant enforcement: Model output cannot mark operation as executed.
   Validated through ReAgent ContractGroundingValidator.
6. Secret protection: API key is never printed, logged, or recorded.
   Telemetry records only the endpoint hostname and sanitized metrics.
7. Fails closed: Exits with NOT_CONFIGURED when NEBIUS_API_KEY is not set.

USAGE:
    # 1. Offline dry run (mock client, zero cost, verifies complete flow):
    python run_nemotron_smoke_test.py --dry-run

    # 2. Live one-shot inference (requires NEBIUS_API_KEY set in environment):
    $env:NEBIUS_API_KEY = "your-key-here"
    python run_nemotron_smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Dict, Any

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reagent.providers.nebius import (
    NebiusNemotronProvider,
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    SanitizedProviderError,
)
from reagent.providers.telemetry import (
    ProviderTelemetryEvent,
    compute_canonical_hash,
    TOKEN_SOURCE_PROVIDER,
    TOKEN_SOURCE_UNKNOWN,
)
from reagent.contract_validator import ContractGroundingValidator
from reagent.tools.base import BaseTool
from reagent.models import StructuredDecision, Postcondition


# ---------------------------------------------------------------------------
# Fictional Harmless Inventory Read-Only Tool Definition
# ---------------------------------------------------------------------------

class InventoryReadOnlyTool(BaseTool):
    """Fictional read-only tool for warehouse stock inspection. Zero side effects."""
    name: str = "inventory_read_only"
    description: str = "Query current in-stock quantity for a warehouse item."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "item_id": {"type": "string", "description": "Fictional item SKU (e.g., INV-ITEM-9842)"},
            "warehouse_code": {"type": "string", "description": "Warehouse location code (e.g., WH-A12)"},
            "location": {"type": "string", "description": "Warehouse location code (e.g., WH-A12)"},
        },
        "required": ["item_id"],
    }
    effect_class: str = "read-only"

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Not invoked during contract proposal. Raises if executed."""
        raise NotImplementedError("InventoryReadOnlyTool is for contract validation only.")


# ---------------------------------------------------------------------------
# Harmless Task Objective
# ---------------------------------------------------------------------------

HARMLESS_TASK_OBJECTIVE = (
    "Verify read-only stock record for fictional inventory item INV-ITEM-9842 "
    "in warehouse location WH-A12. Expected quantity is exactly 42 units with "
    "status IN_STOCK. Read-only verification only."
)
HARMLESS_AVAILABLE_TOOLS = ["inventory_read_only"]


def run_smoke_test(
    provider_override: Optional[NebiusNemotronProvider] = None,
    dry_run: bool = False,
    timeout: float = 15.0,
    max_tokens: int = 768,
) -> Dict[str, Any]:
    """
    Executes a single, bounded contract proposal smoke test.
    Never retries, never executes side effects, never exposes secrets.
    """
    run_id = f"smoke_{int(time.time())}_{os.urandom(4).hex()}"
    t0 = time.perf_counter()

    # 1. Check environment credentials or initialize dry-run mock
    if provider_override is None:
        if dry_run:
            from unittest.mock import MagicMock
            mock_client = MagicMock()
            if max_tokens < 300:
                # Simulate truncation when max_tokens is insufficient (< 300 tokens)
                choice = MagicMock()
                choice.message.content = '{"tool": "inventory_read_only", "arguments": {"item_id": "INV-ITEM-9842"'
                choice.finish_reason = "length"
                mock_resp = MagicMock()
                mock_resp.choices = [choice]
                mock_resp.usage = MagicMock(prompt_tokens=337, completion_tokens=max_tokens)
            else:
                # Complete valid structured decision matching harmless inventory task
                valid_contract = json.dumps({
                    "tool": "inventory_read_only",
                    "arguments": {"item_id": "INV-ITEM-9842", "location": "WH-A12"},
                    "proposed_postconditions": [
                        {
                            "predicate": "STATUS_SUCCESS",
                            "description": "Inventory query completed successfully",
                        }
                    ],
                    "diagnosis_category": None,
                    "diagnosis_confidence": 0.0,
                    "diagnosis_reason": None,
                    "recovery_strategy": None,
                })
                choice = MagicMock()
                choice.message.content = valid_contract
                choice.finish_reason = "stop"
                mock_resp = MagicMock()
                mock_resp.choices = [choice]
                mock_resp.usage = MagicMock(prompt_tokens=337, completion_tokens=342)

            mock_client.chat.completions.create.return_value = mock_resp
            provider = NebiusNemotronProvider(
                client=mock_client,
                timeout=timeout,
                max_retries=0,
            )
        else:
            api_key = os.environ.get("NEBIUS_API_KEY", "")
            if not (api_key and api_key.strip()):
                return {
                    "verdict": "NOT_CONFIGURED",
                    "run_id": run_id,
                    "model_id": os.environ.get("NEBIUS_NEMOTRON_MODEL", DEFAULT_MODEL),
                    "endpoint_host": urlparse(os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL)).netloc,
                    "latency_ms": 0,
                    "input_tokens": None,
                    "output_tokens": None,
                    "token_source": TOKEN_SOURCE_UNKNOWN,
                    "schema_valid": False,
                    "grounding_valid": False,
                    "error_category": "MISSING_API_KEY",
                    "message": "NEBIUS_API_KEY environment variable is not set. Live inference refused.",
                    "is_live": False,
                }
            provider = NebiusNemotronProvider(
                timeout=timeout,
                max_retries=0,  # Strict: zero automatic retries
            )
    else:
        provider = provider_override

    endpoint_host = urlparse(provider.base_url).netloc
    model_id = provider.model

    # 2. Invoke structured decision proposal
    decision: Optional[StructuredDecision] = None
    error_category: Optional[str] = None
    sanitized_error_details: Optional[str] = None
    try:
        decision = provider.make_structured_decision(
            task_objective=HARMLESS_TASK_OBJECTIVE,
            available_tools=HARMLESS_AVAILABLE_TOOLS,
            max_tokens=max_tokens,
        )
        if decision is None:
            raw_err = provider.last_validation_error or "MALFORMED_OUTPUT"
            sanitized_error_details = raw_err
            err_map = {
                "unparseable_json": "JSON_PARSE_ERROR",
                "truncated": "TRUNCATED_OUTPUT",
                "empty_completion": "EMPTY_COMPLETION",
                "malformed_output": "SCHEMA_VALIDATION_ERROR",
            }
            error_category = err_map.get(raw_err, raw_err)
    except SanitizedProviderError as exc:
        error_category = exc.category
        sanitized_error_details = exc.details
    except Exception as exc:
        error_category = type(exc).__name__
        sanitized_error_details = str(exc)

    latency_ms = int((time.perf_counter() - t0) * 1000)

    # 3. Token usage attribution (genuine provider-reported usage only)
    input_tokens = None
    output_tokens = None
    token_source = TOKEN_SOURCE_UNKNOWN
    if provider.last_usage:
        usage = provider.last_usage
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
        else:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is not None:
            token_source = TOKEN_SOURCE_PROVIDER

    # 4. Schema validation check
    schema_valid = decision is not None

    # 5. Contract Grounding Gate validation
    grounding_valid = False
    grounding_errors = []
    if decision is not None:
        validator = ContractGroundingValidator()
        tool = InventoryReadOnlyTool()
        vres = validator.validate(decision, tool, HARMLESS_TASK_OBJECTIVE)
        grounding_valid = vres.is_valid
        grounding_errors = vres.errors

    # 6. Safety Invariant Check:
    # A model output cannot mark an operation as executed or set provenance to executed
    model_marked_executed = False
    if decision is not None:
        prov = getattr(decision, "provenance", None)
        if prov == "EXECUTED":
            model_marked_executed = True
            grounding_valid = False
            grounding_errors.append("SafetyViolation: model attempted to claim EXECUTED provenance.")

    # Assign category for validation / invariant failures if not already an exception
    if error_category is None:
        if model_marked_executed:
            error_category = "EXECUTION_INVARIANT_VIOLATION"
        elif not grounding_valid and decision is not None:
            error_category = "GROUNDING_GATE_REJECTED"

    # 7. Final verdict
    if schema_valid and grounding_valid and not model_marked_executed:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    contract_dict = None
    if decision is not None:
        if hasattr(decision, "model_dump"):
            contract_dict = decision.model_dump()
        elif hasattr(decision, "dict"):
            contract_dict = decision.dict()
        else:
            contract_dict = dict(decision)

    canonical_hash = compute_canonical_hash(contract_dict)

    # 8. Sanitized telemetry event
    telemetry = ProviderTelemetryEvent(
        run_id=run_id,
        provider_name="nebius_nemotron",
        model_id=model_id,
        endpoint_host=endpoint_host,
        http_status_category="2xx" if verdict == "PASS" else "error",
        latency_ms=latency_ms,
        input_tokens_reported=input_tokens,
        output_tokens_reported=output_tokens,
        token_source=token_source,
        schema_valid=schema_valid,
        grounding_valid=grounding_valid,
        canonical_output_hash=canonical_hash,
        is_live=not dry_run and provider_override is None,
        failure_reason=error_category,
    )

    out_artifact = {
        "verdict": verdict,
        "run_id": run_id,
        "model_id": model_id,
        "endpoint_host": endpoint_host,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_source": token_source,
        "schema_valid": schema_valid,
        "grounding_valid": grounding_valid,
        "error_category": error_category,
        "error_details": sanitized_error_details,
        "canonical_output_hash": canonical_hash,
        "grounding_errors": grounding_errors,
        "is_live": not dry_run and provider_override is None,
        "contract": contract_dict,
    }

    # Save artifact to reports/dbops/smoke_test_telemetry.json
    try:
        art_path = Path("reports/dbops/smoke_test_telemetry.json")
        art_path.parent.mkdir(parents=True, exist_ok=True)
        with open(art_path, "w", encoding="utf-8") as f:
            json.dump(out_artifact, f, indent=2)
    except Exception:
        pass

    return out_artifact


def main():
    parser = argparse.ArgumentParser(description="ReAgent Nemotron One-Shot Smoke Test")
    parser.add_argument("--dry-run", action="store_true", help="Run offline dry-run test with mock client")
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds (default 15.0)")
    parser.add_argument("--max-tokens", type=int, default=768, help="Max tokens (default 768, calibrated for complete structured contracts)")
    args = parser.parse_args()

    print("==================================================================")
    print("      ReAgent NVIDIA Nemotron: One-Shot Smoke Test Gate")
    print("==================================================================")

    res = run_smoke_test(
        dry_run=args.dry_run,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    verdict = res["verdict"]
    print(f" Verdict          : {verdict}")
    print(f" Model ID         : {res['model_id']}")
    print(f" Endpoint Host    : {res['endpoint_host']}")
    print(f" Latency          : {res['latency_ms']} ms")
    print(f" Input Tokens     : {res['input_tokens']} ({res['token_source']})")
    print(f" Output Tokens    : {res['output_tokens']}")
    print(f" Schema Valid     : {res['schema_valid']}")
    print(f" Grounding Valid  : {res['grounding_valid']}")
    print(f" Error Category   : {res.get('error_category') or 'None'}")
    print(f" Is Live Request  : {res['is_live']}")
    if res.get("canonical_output_hash"):
        print(f" Output Hash      : {res['canonical_output_hash'][:24]}...")
    print("------------------------------------------------------------------")

    if verdict == "NOT_CONFIGURED":
        print("[NOTICE] NEBIUS_API_KEY is not configured in environment.")
        print("[INSTRUCTION] To run live inference privately, execute in your local shell:")
        print("    $env:NEBIUS_API_KEY = \"your-actual-api-key\"")
        print("    python run_nemotron_smoke_test.py")
        sys.exit(0)
    elif verdict == "PASS":
        print("[SUCCESS] Smoke test passed: Valid structured contract and grounding verified.")
        sys.exit(0)
    else:
        print(f"[FAILED] Smoke test failed: {res.get('error_category')}")
        if res.get("grounding_errors"):
            print(f"Grounding errors: {res['grounding_errors']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
