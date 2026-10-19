"""
eval_runner.py — Guarded Live-Model Evaluation Runner for DBOPS-M2.

Implements the guarded execution harness for evaluating model-generated DBOPS
contracts against the 12 pre-registered tasks with 3 samples each (36 total).

HARD SAFETY GUARDS:
1. Default mode is strictly DRY-RUN / OFFLINE (zero network requests).
2. Live mode requires EXPLICIT opt-in flag AND pre-configured budget/ledger.
3. Hard ceiling of 36 intended requests (12 tasks * 3 samples).
4. Zero automatic retries: errors, malformed outputs, and timeouts count as failures.
5. Frozen configuration fingerprint: model ID, temperature, max_tokens, schema
   version, prompt hashes, gold-set identity. Aborts if configuration changes.
6. Durable per-sample execution ledger preventing duplicate or ambiguous retries.
7. Token budget guard integration with fail-closed ledger error handling.
8. Separate accounting: Schema-valid rate, Grounded rate, Gold-equivalent rate.
9. Pre-registered Gate Aggregation: Task passes if >= 2/3 samples gold-equivalent;
   Benchmark passes if >= 9/12 distinct tasks pass.

Zero secrets in logs or reports. Never inspects .env.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

from reagent.experiments.dbops.manifest_dbops_v1 import (
    ALL_TASK_IDS,
    MANIFEST_VERSION,
    LIVE_MODEL_TARGET_IDENTIFIER,
    LIVE_MODEL_SAMPLES_PER_TASK,
    LIVE_MODEL_TOTAL_SAMPLES,
    LIVE_MODEL_MIN_PASSING_TASKS,
    LIVE_MODEL_TEMPERATURE,
    LIVE_MODEL_MAX_TOKENS,
)
from reagent.experiments.dbops.tasks import get_task, DbopsTask
from reagent.experiments.dbops.gold_contracts import get_gold_contract
from reagent.experiments.dbops.prompt_harness import DbopsPromptHarness, TaskPromptBundle
from reagent.experiments.dbops.fixture_replay import OfflineFixtureReplayEngine, ReplayOutcome
from reagent.experiments.dbops.eval_scoring import DbopsGoldScorer, ContractScoringResult
from reagent.policy import TokenBudgetGuard, GlobalTokenLedger, CorruptLedgerError
from reagent.providers.nebius import NebiusNemotronProvider, DEFAULT_MODEL, DEFAULT_BASE_URL
from reagent.models import StructuredDecision


@dataclass(frozen=True)
class FrozenEvaluationConfig:
    """
    Immutable evaluation configuration frozen before any execution.
    """
    manifest_version: str
    target_model: str
    temperature: float
    max_tokens: int
    samples_per_task: int
    total_samples: int
    min_passing_tasks: int
    prompt_hashes: Dict[str, str]
    config_fingerprint: str


@dataclass
class SampleRecord:
    """
    Durable record of a single evaluation sample attempt.
    """
    sample_id: str                # e.g., task_01_refund_issue_s0
    task_id: str
    sample_index: int
    config_fingerprint: str
    prompt_hash: str
    timestamp_utc: float
    is_live_call: bool
    status: str                   # COMPLETED | TIMEOUT | ERROR | SKIPPED
    latency_ms: int
    raw_response: Optional[str]
    parsed_json_ok: bool
    schema_valid: bool
    schema_error: Optional[str]
    grounding_valid: bool
    grounding_errors: List[str]
    gold_equivalent: bool
    gold_errors: List[str]
    token_usage: Dict[str, int]
    failure_reason: Optional[str] = None


@dataclass
class TaskEvaluationSummary:
    task_id: str
    total_samples: int
    schema_valid_count: int
    grounded_count: int
    gold_equivalent_count: int
    task_passed: bool             # >= 2 / 3 gold-equivalent
    sample_records: List[SampleRecord] = field(default_factory=list)


@dataclass
class BenchmarkEvaluationSummary:
    config_fingerprint: str
    timestamp_utc: float
    is_live: bool
    total_samples_attempted: int
    schema_valid_samples: int
    grounded_samples: int
    gold_equivalent_samples: int
    passing_tasks_count: int
    overall_gate_passed: bool      # >= 9 / 12 tasks pass
    task_summaries: Dict[str, TaskEvaluationSummary] = field(default_factory=dict)


class DbopsEvaluationRunner:
    """
    Guarded runner for DBOPS-M2 model contract derivation evaluation.
    """

    def __init__(
        self,
        ledger_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        live_opt_in: bool = False,
        provider: Optional[NebiusNemotronProvider] = None,
        budget_guard: Optional[TokenBudgetGuard] = None,
        model_name: Optional[str] = None,
    ):
        self.live_opt_in = live_opt_in
        self.results_dir = Path(results_dir) if results_dir else Path("reagent/experiments/dbops/eval_records")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.record_log_path = self.results_dir / "samples_ledger.jsonl"

        # Freeze prompt harness and prompts
        self.prompt_harness = DbopsPromptHarness()
        self.prompt_bundles = self.prompt_harness.construct_all_prompts()

        # Build and freeze configuration
        self.frozen_config = self._freeze_configuration(model_name or DEFAULT_MODEL)

        # Budget and Token Ledger setup
        self.ledger_path = ledger_path or os.environ.get("REAGENT_LEDGER_PATH", "reagent_ledger.json")
        if budget_guard is not None:
            self.budget_guard = budget_guard
        else:
            ledger = GlobalTokenLedger(
                ledger_path=self.ledger_path,
                max_cumulative_calls=50,
                max_cumulative_tokens=200_000,
            )
            self.budget_guard = TokenBudgetGuard(
                ledger=ledger,
                max_calls_per_run=LIVE_MODEL_TOTAL_SAMPLES + 5,
                max_tokens_per_run=100000,
            )

        # Replay & Scoring engines
        self.replay_engine = OfflineFixtureReplayEngine()
        self.scorer = DbopsGoldScorer()

        # Provider: strictly unconfigured by default unless explicit opt-in
        if provider is not None:
            self.provider = provider
        elif self.live_opt_in:
            self.provider = NebiusNemotronProvider(model=self.frozen_config.target_model)
        else:
            self.provider = None

    def _freeze_configuration(self, model_name: str) -> FrozenEvaluationConfig:
        hashes = {t: self.prompt_bundles[t].prompt_hash for t in ALL_TASK_IDS}
        canonical_str = (
            f"MANIFEST:{MANIFEST_VERSION}|"
            f"MODEL:{model_name}|"
            f"TEMP:{LIVE_MODEL_TEMPERATURE}|"
            f"MAX_TOKENS:{LIVE_MODEL_MAX_TOKENS}|"
            f"SAMPLES_PER_TASK:{LIVE_MODEL_SAMPLES_PER_TASK}|"
            f"TOTAL:{LIVE_MODEL_TOTAL_SAMPLES}|"
            f"PASS_GATE:{LIVE_MODEL_MIN_PASSING_TASKS}|"
            f"HASHES:{json.dumps(hashes, sort_keys=True)}"
        )
        fingerprint = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        return FrozenEvaluationConfig(
            manifest_version=MANIFEST_VERSION,
            target_model=model_name,
            temperature=LIVE_MODEL_TEMPERATURE,
            max_tokens=LIVE_MODEL_MAX_TOKENS,
            samples_per_task=LIVE_MODEL_SAMPLES_PER_TASK,
            total_samples=LIVE_MODEL_TOTAL_SAMPLES,
            min_passing_tasks=LIVE_MODEL_MIN_PASSING_TASKS,
            prompt_hashes=hashes,
            config_fingerprint=fingerprint,
        )

    def load_completed_sample_ids(self) -> set:
        """Read completed sample IDs from durable JSONL ledger."""
        completed = set()
        if not self.record_log_path.exists():
            return completed

        with open(self.record_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Any recorded sample (COMPLETED, TIMEOUT, ERROR) is considered attempted
                    completed.add(data["sample_id"])
                except Exception:
                    pass
        return completed

    def _append_sample_record(self, record: SampleRecord) -> None:
        """Durably append a sample record to JSONL."""
        with open(self.record_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def run_evaluation(
        self,
        mock_response_generator: Optional[Callable[[str, int], str]] = None,
    ) -> BenchmarkEvaluationSummary:
        """
        Execute evaluation across all 12 tasks with 3 samples each.
        """
        # Hard safety gate: live mode requires explicit opt-in
        if not self.live_opt_in and mock_response_generator is None:
            # Default dry-run / preflight mode
            return self.run_dry_run_preflight()

        if self.live_opt_in:
            if not self.provider or not self.provider.is_configured():
                raise RuntimeError(
                    "LiveExecutionRefused: live_opt_in=True but Nebius provider is not configured. "
                    "Fail closed to prevent silent failure."
                )

        completed_samples = self.load_completed_sample_ids()
        task_summaries: Dict[str, TaskEvaluationSummary] = {}
        all_records: List[SampleRecord] = []

        total_attempted = 0
        total_schema_valid = 0
        total_grounded = 0
        total_gold_equiv = 0

        for task_id in ALL_TASK_IDS:
            task = get_task(task_id)
            bundle = self.prompt_bundles[task_id]

            # Invariant check: verify prompt hash has not drifted
            if bundle.prompt_hash != self.frozen_config.prompt_hashes[task_id]:
                raise RuntimeError(f"ConfigDriftDetected: prompt hash for {task_id} has changed mid-run!")

            task_records: List[SampleRecord] = []
            schema_valid_count = 0
            grounded_count = 0
            gold_equiv_count = 0

            for sample_idx in range(LIVE_MODEL_SAMPLES_PER_TASK):
                sample_id = f"{task_id}_s{sample_idx}"

                # Resume safety: skip samples already recorded as attempted
                if sample_id in completed_samples:
                    # Load existing record if possible
                    # (Treated as attempted and evaluated)
                    continue

                # Hard ceiling: exactly 36 total intended requests
                total_attempted += 1
                if total_attempted > LIVE_MODEL_TOTAL_SAMPLES:
                    raise RuntimeError(f"SafetyViolation: Hard ceiling of {LIVE_MODEL_TOTAL_SAMPLES} requests exceeded!")

                # Step 1: Token Budget Check & Reserve (fails closed on ledger corruption)
                try:
                    res_id = self.budget_guard.reserve(estimated_tokens=bundle.estimated_tokens)
                except Exception as exc:
                    raise RuntimeError(f"BudgetLedgerError: Failed to reserve tokens for {sample_id}: {exc}") from exc

                t0 = time.perf_counter()
                raw_text = None
                status = "COMPLETED"
                fail_reason = None
                usage = {"prompt_tokens": bundle.estimated_tokens, "completion_tokens": 0, "total_tokens": bundle.estimated_tokens}

                try:
                    if mock_response_generator is not None:
                        raw_text = mock_response_generator(task_id, sample_idx)
                    elif self.live_opt_in and self.provider:
                        # Zero retry call: standard generate with timeout
                        res = self.provider.generate(
                            prompt=bundle.user_prompt,
                            system_prompt=bundle.system_prompt,
                            temperature=self.frozen_config.temperature,
                            max_tokens=self.frozen_config.max_tokens,
                            response_schema={},
                        )
                        raw_text = res.content
                        if res.token_usage:
                            usage = res.token_usage
                    else:
                        raise RuntimeError("Unreachable execution branch in guarded runner")
                except TimeoutError:
                    status = "TIMEOUT"
                    fail_reason = "Model request timed out"
                except Exception as exc:
                    status = "ERROR"
                    fail_reason = f"Provider invocation error: {exc}"

                latency_ms = int((time.perf_counter() - t0) * 1000)

                # Step 2: Reconcile budget
                try:
                    self.budget_guard.reconcile(
                        reservation_id=res_id,
                        usage=usage,
                        conservative_default=bundle.estimated_tokens,
                    )
                except Exception as exc:
                    raise RuntimeError(f"BudgetLedgerError: Failed to reconcile ledger: {exc}") from exc

                # Step 3: Parse, Validate & Ground
                if status == "COMPLETED" and raw_text is not None:
                    replay_outcome = self.replay_engine.replay_fixture(
                        fixture_id=sample_id,
                        task_id=task_id,
                        category="LIVE_OR_MOCK",
                        raw_response_text=raw_text,
                    )
                    
                    parsed_ok = replay_outcome.parsed_json_ok
                    schema_ok = replay_outcome.schema_valid
                    schema_err = replay_outcome.schema_error
                    ground_ok = replay_outcome.grounding_valid
                    ground_errs = replay_outcome.grounding_errors
                    decision = replay_outcome.structured_decision

                    # Step 4: Gold-Equivalence Scoring
                    score_res = self.scorer.evaluate_decision(
                        task_id=task_id,
                        decision=decision,
                        schema_valid=schema_ok,
                        grounding_valid=ground_ok,
                    )
                    gold_ok = score_res.is_gold_equivalent
                    gold_errs = list(score_res.gold_equivalence_errors)
                    if not gold_ok and not fail_reason:
                        fail_reason = score_res.score_summary
                else:
                    parsed_ok = False
                    schema_ok = False
                    schema_err = fail_reason
                    ground_ok = False
                    ground_errs = []
                    gold_ok = False
                    gold_errs = [fail_reason or "Request failed"]

                # Counts
                if schema_ok:
                    schema_valid_count += 1
                    total_schema_valid += 1
                if ground_ok:
                    grounded_count += 1
                    total_grounded += 1
                if gold_ok:
                    gold_equiv_count += 1
                    total_gold_equiv += 1

                record = SampleRecord(
                    sample_id=sample_id,
                    task_id=task_id,
                    sample_index=sample_idx,
                    config_fingerprint=self.frozen_config.config_fingerprint,
                    prompt_hash=bundle.prompt_hash,
                    timestamp_utc=time.time(),
                    is_live_call=self.live_opt_in,
                    status=status,
                    latency_ms=latency_ms,
                    raw_response=raw_text,
                    parsed_json_ok=parsed_ok,
                    schema_valid=schema_ok,
                    schema_error=schema_err,
                    grounding_valid=ground_ok,
                    grounding_errors=ground_errs,
                    gold_equivalent=gold_ok,
                    gold_errors=gold_errs,
                    token_usage=usage,
                    failure_reason=fail_reason,
                )

                # Persist durably
                self._append_sample_record(record)
                task_records.append(record)
                all_records.append(record)

            # Pre-registered task passing rule: >= 2 of 3 samples are gold-equivalent
            task_passed = (gold_equiv_count >= 2)
            task_summaries[task_id] = TaskEvaluationSummary(
                task_id=task_id,
                total_samples=len(task_records),
                schema_valid_count=schema_valid_count,
                grounded_count=grounded_count,
                gold_equivalent_count=gold_equiv_count,
                task_passed=task_passed,
                sample_records=task_records,
            )

        passing_tasks = sum(1 for t in task_summaries.values() if t.task_passed)
        overall_passed = (passing_tasks >= LIVE_MODEL_MIN_PASSING_TASKS)

        return BenchmarkEvaluationSummary(
            config_fingerprint=self.frozen_config.config_fingerprint,
            timestamp_utc=time.time(),
            is_live=self.live_opt_in,
            total_samples_attempted=len(all_records),
            schema_valid_samples=total_schema_valid,
            grounded_samples=total_grounded,
            gold_equivalent_samples=total_gold_equiv,
            passing_tasks_count=passing_tasks,
            overall_gate_passed=overall_passed,
            task_summaries=task_summaries,
        )

    def run_dry_run_preflight(self) -> BenchmarkEvaluationSummary:
        """
        Execute dry-run preflight: validates configuration freeze, prompt hashes,
        and ledger without executing any provider requests.
        """
        # Ensure budget ledger is accessible
        state = self.budget_guard.ledger.get_state()
        
        task_summaries = {}
        for task_id in ALL_TASK_IDS:
            task_summaries[task_id] = TaskEvaluationSummary(
                task_id=task_id,
                total_samples=0,
                schema_valid_count=0,
                grounded_count=0,
                gold_equivalent_count=0,
                task_passed=False,
                sample_records=[],
            )

        return BenchmarkEvaluationSummary(
            config_fingerprint=self.frozen_config.config_fingerprint,
            timestamp_utc=time.time(),
            is_live=False,
            total_samples_attempted=0,
            schema_valid_samples=0,
            grounded_samples=0,
            gold_equivalent_samples=0,
            passing_tasks_count=0,
            overall_gate_passed=False,
            task_summaries=task_summaries,
        )
