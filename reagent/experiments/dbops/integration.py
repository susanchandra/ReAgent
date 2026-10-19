"""
integration.py — ReAgent Pipeline Integration Bridge for DBOPS V1.

Bridges the existing ReAgentPipeline, ToolRegistry, and recovery loop to the
authoritative DBOPS execution, store, and oracle machinery.

Key Architecture:
1. DbopsDatabaseTool(BaseTool):
   - Registered in ToolRegistry.
   - Parameterized execution only (task_id, fault_mode). No arbitrary SQL.
   - Delegates directly to execute_task() using a disposable DbopsStore.
2. DbopsProbeCallback:
   - Matches the pipeline's Callable[[str, Dict[str, Any], int], Dict[str, Any]] signature.
   - Queries through DbopsOracle on a fresh observer connection.
   - Enforces contract verification: checks that primary and secondary postconditions
     hold before reporting PRESENT. If fields are mutated or partial, reports DISCREPANCY.
3. DeterministicMockProvider:
   - Generates grounded StructuredDecisions for Task 02 without network/model calls.

Zero network access. Zero model provider calls. Zero arbitrary SQL.
"""

from typing import Dict, Any, Optional, Callable
import json

from reagent.tools.base import BaseTool
from reagent.models import (
    ActionResult,
    StructuredDecision,
    Postcondition,
)
from reagent.providers.base import LLMProvider, LLMResponse
from reagent.experiments.dbops.store import DbopsStore
from reagent.experiments.dbops.executor import execute_task, ExecutionResult
from reagent.experiments.dbops.oracle import DbopsOracle
from reagent.experiments.dbops.validator import GoldContractEvaluator
from reagent.experiments.dbops.types import VerdictType
from reagent.experiments.dbops.tasks import get_task


class DbopsDatabaseTool(BaseTool):
    """
    Allowlisted database execution tool that connects ReAgentPipeline to the
    existing DBOPS SQLite store and task executor.
    """
    name: str = "dbops_database"
    description: str = "Executes pre-registered database tasks in an isolated SQLite environment."
    effect_class: str = "state-changing"
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The manifest task ID to execute."},
            "fault_mode": {"type": "string", "description": "Injected fault mode."},
        },
        "required": ["task_id"],
    }

    def __init__(self, store: DbopsStore):
        self.store = store
        self.last_execution_result: Optional[ExecutionResult] = None
        self.execution_count: int = 0

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task_id = params["task_id"]
        fault_mode = params.get("fault_mode", "NO_FAULT")

        self.execution_count += 1
        exec_res = execute_task(self.store, task_id, fault_mode)
        self.last_execution_result = exec_res

        if exec_res.agent_visible_error is not None:
            # Raise exception so BaseTool captures it as a failed ActionResult
            raise TimeoutError(exec_res.agent_visible_error)

        # Check if the execution resulted in an unverified/corrupted state
        oracle = DbopsOracle(self.store, exec_res)
        obs = oracle.evaluate()
        if obs.verdict in (VerdictType.WRONG_CONTENT, VerdictType.PARTIAL, VerdictType.UNKNOWN):
            raise RuntimeError(f"DatabaseStateDiscrepancy: verdict={obs.verdict.value}")

        return {
            "status": "SUCCESS",
            "task_id": task_id,
            "committed_steps": exec_res.committed_steps,
        }


def make_dbops_probe_callback(
    store: DbopsStore,
    task_id: str,
    fault_mode: str,
) -> Callable[[str, Dict[str, Any], int], Dict[str, Any]]:
    """
    Factory creating a probe callback for ReAgentPipeline._run_recovery_loop.
    Enforces the fresh connection rule by creating a DbopsOracle connected
    to the store's committed state.
    """
    def probe_callback(tool_name: str, params: Dict[str, Any], step: int) -> Dict[str, Any]:
        # Build dummy ExecutionResult representing the completed execution
        exec_res = ExecutionResult(
            task_id=task_id,
            fault_mode=fault_mode,
            agent_visible_error=None,
            committed_steps=[0, 1],
            wrong_content_injected=(fault_mode == "WRONG_CONTENT"),
        )
        oracle = DbopsOracle(store, exec_res)

        if fault_mode == "DELAYED_VISIBILITY":
            p = oracle.probe(call_number=step)
            if p.status == "DELAYED":
                return {"status": "DELAYED"}
            elif p.status == "PRESENT":
                obs = oracle.evaluate()
                if obs.verdict == VerdictType.EXPECTED_PRESENT:
                    return {"status": "PRESENT", "verdict": obs.verdict.value}
                return {"status": "DISCREPANCY", "verdict": obs.verdict.value}
            return {"status": p.status}

        if fault_mode == "STATUS_UNAVAILABLE":
            raise ConnectionError("StatusProbeUnavailable: probe surface temporarily unreachable.")

        if fault_mode == "UNOBSERVABLE":
            raise RuntimeError("UnobservableEvidence: no observation surface exists.")

        # Full ground truth evaluation on fresh observer connection
        obs = oracle.evaluate()

        if obs.verdict == VerdictType.EXPECTED_PRESENT:
            return {"status": "PRESENT", "verdict": obs.verdict.value}
        elif obs.verdict == VerdictType.CONFIRMED_ABSENT:
            return {"status": "CONFIRMED_ABSENT", "verdict": obs.verdict.value}
        elif obs.verdict in (VerdictType.PARTIAL, VerdictType.WRONG_CONTENT):
            return {"status": "DISCREPANCY", "verdict": obs.verdict.value}
        else:
            return {"status": "UNKNOWN", "verdict": obs.verdict.value}

    return probe_callback


class DeterministicMockWalletProvider(LLMProvider):
    """
    Offline mock provider returning a valid, grounded StructuredDecision for Task 02.
    """
    def __init__(self):
        self.last_usage = {"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100}

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(content="{}", parsed_json={}, model_name="mock-wallet-provider")

    def diagnose_failure(self, failure: Any, execution_history: Optional[list] = None) -> Any:
        from reagent.models import FailureDiagnosis
        from reagent.taxonomy import FailureType
        return FailureDiagnosis(
            failure_id=failure.failure_id,
            failure_type=FailureType.TRANSIENT_TOOL_ERROR,
            root_cause="Mock failure diagnosis",
            confidence_score=0.9,
            is_agent_fault=False,
            is_transient=True,
            provenance="fixture",
        )

    def select_recovery(self, diagnosis: Any, available_tools: list[str], policy_rules: Optional[list] = None) -> Any:
        from reagent.models import RecoveryDecision
        from reagent.taxonomy import RecoveryStrategyType
        return RecoveryDecision(
            diagnosis_id=diagnosis.diagnosis_id,
            strategy=RecoveryStrategyType.RETRY,
            rationale="Mock recovery decision",
            provenance="fixture",
        )

    def make_structured_decision(
        self,
        task_objective: str,
        available_tools: list[str],
    ) -> StructuredDecision:
        return StructuredDecision(
            tool="dbops_database",
            arguments={"task_id": "task_02_wallet_transfer", "fault_mode": "COMMIT_THEN_TIMEOUT"},
            proposed_postconditions=[
                Postcondition(
                    predicate="STATUS_SUCCESS",
                    target="wallet_transfer_ledger",
                    expected="TXN-779043",
                    description="Transfer TXN-779043 of 500.00 from ACCT-2201 to ACCT-3390",
                    required=True,
                ),
                Postcondition(
                    predicate="KEY_EXISTS",
                    target="task_id",
                    expected="task_02_wallet_transfer",
                    description="Task identifier exists in result",
                    required=False,
                ),
            ],
            provenance="fixture",
        )
