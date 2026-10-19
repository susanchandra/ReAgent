"""
pilot_eval.py — Controlled Offline Pilot Preparation for Nemotron DBOPS Evaluation.

Prepares a reproducible, controlled evaluation of model-derived contracts
on three representative DBOPS tasks with two independent generations per task (6 samples).

HARD SAFETY GUARDS & CONSTRAINTS:
1. Default mode is strictly DRY-RUN / OFFLINE (zero network requests, zero provider charges).
2. Live execution is guarded by explicit opt-in, pre-frozen configuration, and token ledger.
3. Hard ceiling of 6 intended sample requests (3 tasks * 2 generations).
4. Strictly zero automatic retries: refusals, malformed outputs, and timeouts fail closed.
5. Append-only JSONL result ledger with resume safety (never duplicate paid requests).
6. Missing telemetry is explicitly stored as None, never masked as zero.
7. Token caps are local usage guards to prevent runaway loops, not provider billing guarantees.
8. Prompts are isolated from gold contracts and fault modes (verified pre-execution).
9. Full integration with CandidateContractValidationGate; no parallel evaluators.

Zero secrets in logs or reports. Never inspects .env or prints credentials.
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Callable

from filelock import FileLock, Timeout

from reagent.experiments.dbops.manifest_dbops_v1 import (
    MANIFEST_VERSION,
    LIVE_MODEL_TARGET_IDENTIFIER,
)
from reagent.experiments.dbops.tasks import get_task, DbopsTask
from reagent.experiments.dbops.prompt_harness import DbopsPromptHarness, TaskPromptBundle
from reagent.experiments.dbops.candidate_gate import (
    CandidateContractValidationGate,
    CandidateEvaluationRecord,
)
from reagent.experiments.dbops.eval_scoring import DbopsGoldScorer
from reagent.policy import TokenBudgetGuard, GlobalTokenLedger, CorruptLedgerError
from reagent.providers.nebius import (
    NebiusNemotronProvider,
    STRUCTURED_DECISION_SCHEMA,
    ALLOWED_STRUCTURED_DECISION_KEYS,
    DEFAULT_MODEL,
)
from reagent.models import StructuredDecision


# ── Security & Sanitization Helpers ─────────────────────────────────────────

SECRET_PATTERNS = [
    (re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]+", re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{8,}['\"]?"), r"\1=[REDACTED]"),
    (re.compile(r"https?://[^:\s]+:[^@\s]+@"), "http://[REDACTED]@"),
]

MAX_RAW_ARTIFACT_BYTES = 65536


def sanitize_text(text: Optional[str], max_bytes: int = MAX_RAW_ARTIFACT_BYTES) -> Optional[str]:
    """
    Sanitizes raw text or error messages against accidental credential/secret leakage
    and bounds maximum payload length.
    """
    if text is None:
        return None
    cleaned = text
    for pattern, repl in SECRET_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    if len(cleaned.encode("utf-8")) > max_bytes:
        cleaned = cleaned[:max_bytes] + "\n... [TRUNCATED_MAX_BYTES_EXCEEDED]"
    return cleaned


class AmbiguousDispatchedRequestError(RuntimeError):
    """
    Raised when a request was dispatched to a provider but the process crashed before
    recording a terminal result. To prevent duplicate billing on a potentially charged
    request, automated replay is strictly prohibited without explicit resolution.
    """
    pass


class ConfigFingerprintMismatchError(CorruptLedgerError):
    """
    Raised when ledger contains records with an unexpected config or manifest fingerprint.
    Prevents silent cross-contamination across different models or prompt versions.
    """
    pass


TERMINAL_STATUSES: Set[str] = {
    "COMPLETED",
    "REFUSAL",
    "PARSE_ERROR",
    "SCHEMA_ERROR",
    "GROUNDING_ERROR",
    "SEMANTIC_ERROR",
    "TIMEOUT",
    "PROVIDER_ERROR",
    "SKIPPED",
}


# ── Pilot Configuration Constants ──────────────────────────────────────────

PILOT_MANIFEST_VERSION = "1.0.0-pilot-nemotron-offline-prepared"

# Three representative tasks covering distinct protection classes:
# 1. task_01_refund_issue: KEY_PROTECTED (single unique key `refund_requests.refund_ref`)
# 2. task_02_wallet_transfer: UNPROTECTED_NUMERIC (non-idempotent numeric mutation, $500 balance debit)
# 3. task_05_order_cancellation: KEY_PROTECTED with composite business key `order_event_log(order_id, event_type)`
PILOT_TASK_IDS: Tuple[str, ...] = (
    "task_01_refund_issue",
    "task_02_wallet_transfer",
    "task_05_order_cancellation",
)

PILOT_SAMPLES_PER_TASK = 2
PILOT_TOTAL_SAMPLES = len(PILOT_TASK_IDS) * PILOT_SAMPLES_PER_TASK  # Exactly 6 samples

PILOT_TARGET_MODEL = DEFAULT_MODEL
PILOT_TEMPERATURE = 0.0          # Greedy deterministic decoding for contract derivation
PILOT_MAX_TOKENS = 512
PILOT_SCHEMA_VERSION = "structured_decision_v1"

# Acceptance criteria
PILOT_CONSERVATIVE_MIN_PASSING_SAMPLES_PER_TASK = 2  # 100% gold equivalence per task
PILOT_RELAXED_MIN_PASSING_SAMPLES_PER_TASK = 1       # >= 50% exploratory threshold
PILOT_MIN_PASSING_TASKS = 2                          # >= 2 / 3 tasks pass pilot gate


# ── Frozen Data Structures ──────────────────────────────────────────────────

@dataclass(frozen=True)
class FrozenPilotConfig:
    """
    Immutable specification of the pilot evaluation frozen before any request.
    """
    pilot_manifest_version: str
    task_ids: Tuple[str, ...]
    target_model: str
    temperature: float
    max_tokens: int
    samples_per_task: int
    total_samples: int
    schema_version: str
    prompt_hashes: Dict[str, str]
    schema_fingerprint: str
    config_fingerprint: str
    acceptance_criteria: Dict[str, Any]
    stopping_conditions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pilot_manifest_version": self.pilot_manifest_version,
            "task_ids": list(self.task_ids),
            "target_model": self.target_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "samples_per_task": self.samples_per_task,
            "total_samples": self.total_samples,
            "schema_version": self.schema_version,
            "prompt_hashes": self.prompt_hashes,
            "schema_fingerprint": self.schema_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "acceptance_criteria": self.acceptance_criteria,
            "stopping_conditions": self.stopping_conditions,
        }


@dataclass
class PilotSampleLedgerRecord:
    """
    Durable append-only ledger record for one pilot evaluation sample.
    """
    sample_id: str                      # e.g., pilot_task_01_refund_issue_s0
    request_id: str                    # Unique per-request identifier
    task_id: str
    sample_index: int                  # 0 or 1
    model_identifier: str
    config_fingerprint: str
    prompt_fingerprint: str
    schema_fingerprint: str
    timestamp_utc: float
    timestamp_iso: str
    is_live_call: bool
    status: str                        # COMPLETED | REFUSAL | PARSE_ERROR | SCHEMA_ERROR | GROUNDING_ERROR | SEMANTIC_ERROR | TIMEOUT | PROVIDER_ERROR | SKIPPED
    latency_ms: int
    raw_response_artifact_path: Optional[str]
    generation_parse_success: bool
    schema_valid: bool
    schema_error: Optional[str]
    grounding_valid: bool
    grounding_errors: List[str]
    semantic_gold_agreement: bool
    missing_required_postconditions: List[str]
    extra_unsupported_predicates: List[str]
    unsupported_tables_or_columns: List[str]
    uniqueness_respected: bool
    invalid_contract_rejected: bool
    fallback_used: bool                # MUST always be False
    rejection_reason: Optional[str]
    token_telemetry: Optional[Dict[str, int]]  # None if missing (not masked as 0)
    cost_telemetry_usd: Optional[float]        # None if unmetered/unavailable
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PilotEvaluationSummary:
    """
    Summary of pilot execution across the 3 tasks (6 samples).
    """
    config_fingerprint: str
    timestamp_utc: float
    is_live: bool
    total_samples_attempted: int
    schema_valid_samples: int
    grounded_samples: int
    gold_equivalent_samples: int
    passing_tasks_count: int
    overall_gate_passed: bool
    task_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sample_records: List[PilotSampleLedgerRecord] = field(default_factory=list)


# ── Append-Only Result Ledger ───────────────────────────────────────────────

class PilotLedger:
    """
    Append-only result ledger with integrity validation, file locking, and crash/resume safety.
    """

    def __init__(self, ledger_file: Path, raw_artifacts_dir: Path):
        self.ledger_file = ledger_file
        self.raw_artifacts_dir = raw_artifacts_dir
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        self.raw_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.ledger_file) + ".lock", timeout=15.0)

    def load_completed_sample_ids(
        self, expected_config_fingerprint: Optional[str] = None
    ) -> Set[str]:
        """
        Loads all sample IDs that have reached a terminal status in the ledger.
        Fails closed on:
        1. Corrupted ledger lines (CorruptLedgerError)
        2. Config fingerprint mismatch (ConfigFingerprintMismatchError)
        3. Ambiguous dispatched requests without terminal resolution (AmbiguousDispatchedRequestError)
        """
        with self.lock:
            if not self.ledger_file.exists():
                return set()

            completed: Set[str] = set()
            dispatched: Dict[str, str] = {}  # sample_id -> request_id

            with open(self.ledger_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    try:
                        data = json.loads(clean_line)
                    except json.JSONDecodeError as exc:
                        raise CorruptLedgerError(
                            f"Corrupt JSON on line {line_no} of {self.ledger_file}: {exc}"
                        ) from exc

                    sample_id = data.get("sample_id")
                    if not sample_id:
                        raise CorruptLedgerError(
                            f"Line {line_no} missing sample_id in {self.ledger_file}"
                        )

                    rec_cfg_fp = data.get("config_fingerprint")
                    if expected_config_fingerprint and rec_cfg_fp and rec_cfg_fp != expected_config_fingerprint:
                        raise ConfigFingerprintMismatchError(
                            f"ConfigFingerprintMismatch: Line {line_no} for sample '{sample_id}' "
                            f"has config_fingerprint '{rec_cfg_fp}' which differs from expected '{expected_config_fingerprint}'."
                        )

                    status = data.get("status")
                    if status == "DISPATCHED":
                        dispatched[sample_id] = data.get("request_id", "unknown_req")
                    elif status in TERMINAL_STATUSES:
                        if sample_id in completed:
                            raise CorruptLedgerError(
                                f"DuplicateTerminalRecord: sample_id '{sample_id}' has multiple terminal records in {self.ledger_file}"
                            )
                        completed.add(sample_id)
                        dispatched.pop(sample_id, None)

            if dispatched:
                ambiguous_list = [f"sample '{s}' (request_id '{r}')" for s, r in dispatched.items()]
                raise AmbiguousDispatchedRequestError(
                    f"AmbiguousDispatchedRequest: The following requests were dispatched but never reached a terminal state "
                    f"before process termination: {', '.join(ambiguous_list)}. "
                    f"They may have been billed by the provider. Automated replay is prohibited to prevent duplicate billing."
                )

            return completed

    def record_dispatch_intent(
        self,
        sample_id: str,
        request_id: str,
        task_id: str,
        sample_index: int,
        model_identifier: str,
        config_fingerprint: str,
        prompt_fingerprint: str,
        schema_fingerprint: str,
        is_live_call: bool,
        max_global_samples: int = PILOT_TOTAL_SAMPLES,
    ) -> None:
        """
        Durably logs intent to dispatch a request BEFORE contacting the provider.
        Under process-level file lock:
        1. Checks whether sample_id is already completed (raises ValueError).
        2. Checks whether sample_id is already in-flight (raises AmbiguousDispatchedRequestError).
        3. Checks whether total distinct claimed samples >= max_global_samples (raises RuntimeError).
        4. Flushes and fsyncs the DISPATCHED record to disk before lock release.
        """
        with self.lock:
            completed_samples: Set[str] = set()
            dispatched_samples: Dict[str, str] = {}
            if self.ledger_file.exists():
                with open(self.ledger_file, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        clean_line = line.strip()
                        if not clean_line:
                            continue
                        try:
                            d = json.loads(clean_line)
                        except json.JSONDecodeError as exc:
                            raise CorruptLedgerError(
                                f"Corrupt JSON on line {line_no} of {self.ledger_file}: {exc}"
                            ) from exc

                        s_id = d.get("sample_id")
                        st = d.get("status")
                        if st == "DISPATCHED":
                            dispatched_samples[s_id] = d.get("request_id", "unknown_req")
                        elif st in TERMINAL_STATUSES:
                            completed_samples.add(s_id)
                            dispatched_samples.pop(s_id, None)

            if sample_id in completed_samples:
                raise ValueError(
                    f"SampleAlreadyCompleted: sample_id '{sample_id}' has already reached terminal state."
                )

            if sample_id in dispatched_samples:
                raise AmbiguousDispatchedRequestError(
                    f"SampleAlreadyClaimed: sample_id '{sample_id}' is currently dispatched "
                    f"(request_id '{dispatched_samples[sample_id]}'). Concurrent duplicate dispatch is prohibited."
                )

            total_claimed = completed_samples.union(set(dispatched_samples.keys()))
            if len(total_claimed) >= max_global_samples:
                raise RuntimeError(
                    f"SafetyViolation: Global ceiling of {max_global_samples} requests reached across all processes "
                    f"({len(total_claimed)} active or completed). Cannot dispatch additional requests."
                )

            record = PilotSampleLedgerRecord(
                sample_id=sample_id,
                request_id=request_id,
                task_id=task_id,
                sample_index=sample_index,
                model_identifier=model_identifier,
                config_fingerprint=config_fingerprint,
                prompt_fingerprint=prompt_fingerprint,
                schema_fingerprint=schema_fingerprint,
                timestamp_utc=time.time(),
                timestamp_iso=datetime.now(timezone.utc).isoformat(),
                is_live_call=is_live_call,
                status="DISPATCHED",
                latency_ms=0,
                raw_response_artifact_path=None,
                generation_parse_success=False,
                schema_valid=False,
                schema_error=None,
                grounding_valid=False,
                grounding_errors=[],
                semantic_gold_agreement=False,
                missing_required_postconditions=[],
                extra_unsupported_predicates=[],
                unsupported_tables_or_columns=[],
                uniqueness_respected=False,
                invalid_contract_rejected=False,
                fallback_used=False,
                rejection_reason=None,
                token_telemetry=None,
                cost_telemetry_usd=None,
                failure_reason=None,
            )

            data = record.to_dict()
            line = json.dumps(data, ensure_ascii=False)
            with open(self.ledger_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def append_record(self, record: PilotSampleLedgerRecord, allow_dispatched: bool = False) -> None:
        """
        Durably appends a record to the ledger under process-level file lock.
        Fails closed on duplicate terminal records.
        """
        with self.lock:
            # If a terminal record, verify that this sample doesn't already have a terminal record
            if record.status in TERMINAL_STATUSES:
                existing_completed = set()
                if self.ledger_file.exists():
                    with open(self.ledger_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                d = json.loads(line)
                                if d.get("status") in TERMINAL_STATUSES:
                                    existing_completed.add(d.get("sample_id"))
                            except Exception:
                                pass
                if record.sample_id in existing_completed:
                    raise ValueError(
                        f"DuplicateRecordAttempt: sample_id '{record.sample_id}' already has a terminal record in ledger."
                    )

            data = record.to_dict()
            line = json.dumps(data, ensure_ascii=False)
            with open(self.ledger_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def save_raw_response_artifact(self, sample_id: str, raw_text: str) -> str:
        """
        Saves raw response text to a separate artifact file, sanitized against credential leaks.
        Returns the relative artifact path.
        """
        sanitized = sanitize_text(raw_text) or ""
        artifact_path = self.raw_artifacts_dir / f"{sample_id}.txt"
        with self.lock:
            with open(artifact_path, "w", encoding="utf-8") as f:
                f.write(sanitized)
                f.flush()
                os.fsync(f.fileno())
        return str(artifact_path)

    def validate_ledger_integrity(
        self, expected_config_fingerprint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates ledger integrity and returns summary metrics under file lock.
        """
        with self.lock:
            sample_ids = self.load_completed_sample_ids(expected_config_fingerprint=expected_config_fingerprint)
            record_count = 0
            if self.ledger_file.exists():
                with open(self.ledger_file, "r", encoding="utf-8") as f:
                    record_count = sum(1 for line in f if line.strip())

            return {
                "ledger_path": str(self.ledger_file),
                "total_records": record_count,
                "unique_terminal_samples": len(sample_ids),
                "is_valid": True,
            }


# ── Pilot Evaluation Runner ─────────────────────────────────────────────────

class PilotEvaluationRunner:
    """
    Guarded runner for the 6-sample DBOPS Nemotron pilot evaluation.
    """

    def __init__(
        self,
        pilot_dir: Optional[Path] = None,
        live_opt_in: bool = False,
        provider: Optional[NebiusNemotronProvider] = None,
        budget_guard: Optional[TokenBudgetGuard] = None,
        target_model: str = PILOT_TARGET_MODEL,
        task_ids: Tuple[str, ...] = PILOT_TASK_IDS,
    ):
        self.pilot_dir = pilot_dir or Path("reports/dbops")
        self.pilot_dir.mkdir(parents=True, exist_ok=True)
        self.live_opt_in = live_opt_in
        self.target_model = target_model
        self.task_ids = task_ids

        # Ledger setup
        ledger_path = self.pilot_dir / "pilot_ledger.jsonl"
        raw_artifacts_dir = self.pilot_dir / "pilot_raw_responses"
        self.ledger = PilotLedger(ledger_path, raw_artifacts_dir)

        # Harness and validation gate
        self.prompt_harness = DbopsPromptHarness()
        self.validation_gate = CandidateContractValidationGate()

        # Build and freeze configuration
        self.frozen_config = self._build_frozen_config()

        # Token Budget Guard
        if budget_guard is not None:
            self.budget_guard = budget_guard
        else:
            ledger_file = os.environ.get("REAGENT_LEDGER_PATH", "reagent_ledger.json")
            gt_ledger = GlobalTokenLedger(
                ledger_path=ledger_file,
                max_cumulative_calls=50,
                max_cumulative_tokens=200_000,
            )
            self.budget_guard = TokenBudgetGuard(
                ledger=gt_ledger,
                max_calls_per_run=PILOT_TOTAL_SAMPLES + 2,
                max_tokens_per_run=20_000,
            )

        # Provider: strictly unconfigured by default unless explicit live opt-in
        if provider is not None:
            self.provider = provider
        elif self.live_opt_in:
            self.provider = NebiusNemotronProvider(model=self.frozen_config.target_model)
        else:
            self.provider = None

    def _build_frozen_config(self) -> FrozenPilotConfig:
        """
        Constructs and freezes the pilot configuration with SHA-256 fingerprints.
        """
        prompt_hashes: Dict[str, str] = {}
        for t in self.task_ids:
            bundle = self.prompt_harness.construct_task_prompt(t)
            prompt_hashes[t] = bundle.prompt_hash

        schema_fp = hashlib.sha256(STRUCTURED_DECISION_SCHEMA.encode("utf-8")).hexdigest()

        canonical_payload = (
            f"PILOT_MANIFEST:{PILOT_MANIFEST_VERSION}|"
            f"TASKS:{json.dumps(list(self.task_ids))}|"
            f"MODEL:{self.target_model}|"
            f"TEMP:{PILOT_TEMPERATURE}|"
            f"MAX_TOKENS:{PILOT_MAX_TOKENS}|"
            f"SAMPLES_PER_TASK:{PILOT_SAMPLES_PER_TASK}|"
            f"TOTAL_SAMPLES:{PILOT_TOTAL_SAMPLES}|"
            f"SCHEMA_FP:{schema_fp}|"
            f"PROMPT_HASHES:{json.dumps(prompt_hashes, sort_keys=True)}"
        )
        config_fp = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

        acceptance_criteria = {
            "per_sample": {
                "parsed_json_ok": True,
                "schema_valid": True,
                "grounding_valid": True,
                "unsupported_tables_or_columns": [],
                "semantic_gold_agreement": True,
            },
            "per_task_conservative_threshold": f"{PILOT_CONSERVATIVE_MIN_PASSING_SAMPLES_PER_TASK} / {PILOT_SAMPLES_PER_TASK}",
            "per_task_relaxed_threshold": f"{PILOT_RELAXED_MIN_PASSING_SAMPLES_PER_TASK} / {PILOT_SAMPLES_PER_TASK}",
            "overall_gate_threshold": f"{PILOT_MIN_PASSING_TASKS} / {len(self.task_ids)}",
        }

        stopping_conditions = {
            "hard_request_ceiling": PILOT_TOTAL_SAMPLES,
            "zero_automatic_retries": True,
            "fail_closed_on_error": True,
            "fail_closed_on_refusal": True,
            "fail_closed_on_malformed_json": True,
            "abort_on_config_drift": True,
            "abort_on_budget_exhaustion": True,
        }

        return FrozenPilotConfig(
            pilot_manifest_version=PILOT_MANIFEST_VERSION,
            task_ids=self.task_ids,
            target_model=self.target_model,
            temperature=PILOT_TEMPERATURE,
            max_tokens=PILOT_MAX_TOKENS,
            samples_per_task=PILOT_SAMPLES_PER_TASK,
            total_samples=PILOT_TOTAL_SAMPLES,
            schema_version=PILOT_SCHEMA_VERSION,
            prompt_hashes=prompt_hashes,
            schema_fingerprint=schema_fp,
            config_fingerprint=config_fp,
            acceptance_criteria=acceptance_criteria,
            stopping_conditions=stopping_conditions,
        )

    def save_frozen_config_artifact(self, output_path: Optional[Path] = None) -> Path:
        """
        Saves the frozen configuration to an immutable JSON artifact.
        """
        target = output_path or (self.pilot_dir / "pilot_config_frozen.json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.frozen_config.to_dict(), f, indent=2)
        return target

    def verify_pilot_prompt_isolation(self) -> Dict[str, Dict[str, Any]]:
        """
        Verifies that none of the pilot tasks leak gold contracts or fault modes.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for t in self.task_ids:
            results[t] = self.validation_gate.verify_prompt_isolation(t)
            if not results[t]["is_clean"]:
                raise RuntimeError(
                    f"PromptLeakageDetected: Task {t} leaked fault/gold metadata: {results[t]}"
                )
        return results

    def run_dry_run_preflight(self) -> Dict[str, Any]:
        """
        Executes an offline preflight check with zero network calls:
        1. Verifies prompt isolation.
        2. Validates schema and configuration fingerprints.
        3. Checks ledger readiness.
        4. Saves the frozen config artifact.
        """
        isolation_report = self.verify_pilot_prompt_isolation()
        ledger_status = self.ledger.validate_ledger_integrity()
        artifact_path = self.save_frozen_config_artifact()

        return {
            "status": "PREFLIGHT_READY",
            "is_live": False,
            "network_calls_made": 0,
            "paid_tokens_consumed": 0,
            "config_fingerprint": self.frozen_config.config_fingerprint,
            "tasks_verified": list(self.task_ids),
            "total_planned_samples": PILOT_TOTAL_SAMPLES,
            "frozen_config_artifact": str(artifact_path),
            "isolation_results": isolation_report,
            "ledger_status": ledger_status,
        }

    def run_pilot(
        self,
        mock_response_generator: Optional[Callable[[str, int], str]] = None,
    ) -> PilotEvaluationSummary:
        """
        Executes the 6-sample pilot evaluation.
        Defaults to dry-run preflight if neither live_opt_in nor mock generator is provided.
        """
        # Hard safety guard: live mode requires explicit opt-in
        if not self.live_opt_in and mock_response_generator is None:
            preflight = self.run_dry_run_preflight()
            return PilotEvaluationSummary(
                config_fingerprint=self.frozen_config.config_fingerprint,
                timestamp_utc=time.time(),
                is_live=False,
                total_samples_attempted=0,
                schema_valid_samples=0,
                grounded_samples=0,
                gold_equivalent_samples=0,
                passing_tasks_count=0,
                overall_gate_passed=False,
                task_summaries={"PREFLIGHT_ONLY": preflight},
            )

        if self.live_opt_in:
            if not self.provider or not self.provider.is_configured():
                raise RuntimeError(
                    "LiveExecutionRefused: live_opt_in=True but Nebius provider is unconfigured. "
                    "Fail closed to prevent silent failure."
                )

        completed_samples = self.ledger.load_completed_sample_ids(
            expected_config_fingerprint=self.frozen_config.config_fingerprint
        )
        task_summaries: Dict[str, Dict[str, Any]] = {}
        all_records: List[PilotSampleLedgerRecord] = []

        total_attempted = len(completed_samples)
        total_schema_valid = 0
        total_grounded = 0
        total_gold_equiv = 0
        passing_tasks = 0

        for task_id in self.task_ids:
            bundle = self.prompt_harness.construct_task_prompt(task_id)

            # Invariant check: verify prompt hash has not drifted
            if bundle.prompt_hash != self.frozen_config.prompt_hashes[task_id]:
                raise RuntimeError(f"ConfigDriftDetected: prompt hash for {task_id} drifted mid-run!")

            task_records: List[PilotSampleLedgerRecord] = []
            task_schema_valid = 0
            task_grounded = 0
            task_gold_equiv = 0

            for sample_idx in range(PILOT_SAMPLES_PER_TASK):
                sample_id = f"pilot_{task_id}_s{sample_idx}"

                # Resume safety: skip samples already completed
                if sample_id in completed_samples:
                    continue

                # Hard ceiling: strictly at most 6 requests
                if total_attempted >= PILOT_TOTAL_SAMPLES:
                    raise RuntimeError(f"SafetyViolation: Hard ceiling of {PILOT_TOTAL_SAMPLES} requests exceeded!")

                total_attempted += 1

                # Two-phase dispatch logging: atomically claim under FileLock before contacting provider
                request_id = f"req_{hashlib.sha256(f'{sample_id}_{time.time()}'.encode()).hexdigest()[:12]}"
                try:
                    self.ledger.record_dispatch_intent(
                        sample_id=sample_id,
                        request_id=request_id,
                        task_id=task_id,
                        sample_index=sample_idx,
                        model_identifier=self.target_model,
                        config_fingerprint=self.frozen_config.config_fingerprint,
                        prompt_fingerprint=bundle.prompt_hash,
                        schema_fingerprint=self.frozen_config.schema_fingerprint,
                        is_live_call=self.live_opt_in,
                        max_global_samples=PILOT_TOTAL_SAMPLES,
                    )
                except ValueError:
                    # Sample was already completed concurrently by another process
                    continue

                # Token budget reservation (local safety guard)
                try:
                    res_id = self.budget_guard.reserve(estimated_tokens=bundle.estimated_tokens)
                except Exception as exc:
                    raise RuntimeError(f"BudgetLedgerError: Failed to reserve tokens for {sample_id}: {exc}") from exc

                t0 = time.perf_counter()
                raw_text: Optional[str] = None
                status = "COMPLETED"
                fail_reason: Optional[str] = None
                token_usage: Optional[Dict[str, int]] = None

                try:
                    if mock_response_generator is not None:
                        raw_text = mock_response_generator(task_id, sample_idx)
                        # In mock mode, token telemetry is None unless mocked
                        token_usage = None
                    elif self.live_opt_in and self.provider:
                        res = self.provider.generate(
                            prompt=bundle.user_prompt,
                            system_prompt=bundle.system_prompt,
                            temperature=self.frozen_config.temperature,
                            max_tokens=self.frozen_config.max_tokens,
                            response_schema={},
                        )
                        raw_text = res.content
                        token_usage = res.token_usage
                    else:
                        raise RuntimeError("UnreachableExecutionBranch")
                except TimeoutError:
                    status = "TIMEOUT"
                    fail_reason = "Model request timed out"
                except Exception as exc:
                    status = "PROVIDER_ERROR"
                    fail_reason = sanitize_text(f"Provider invocation error: {exc}")

                latency_ms = int((time.perf_counter() - t0) * 1000)

                # Reconcile budget
                try:
                    self.budget_guard.reconcile(
                        reservation_id=res_id,
                        usage=token_usage or {"prompt_tokens": bundle.estimated_tokens, "completion_tokens": 0, "total_tokens": bundle.estimated_tokens},
                        conservative_default=bundle.estimated_tokens,
                    )
                except Exception as exc:
                    raise RuntimeError(f"BudgetLedgerError: Failed to reconcile ledger: {exc}") from exc

                # Safe artifact storage
                raw_artifact_ref: Optional[str] = None
                if raw_text is not None:
                    raw_artifact_ref = self.ledger.save_raw_response_artifact(sample_id, raw_text)

                # Validation & Scoring via CandidateContractValidationGate
                if status == "COMPLETED" and raw_text is not None:
                    eval_rec = self.validation_gate.validate_candidate_raw(
                        task_id=task_id,
                        raw_response_text=raw_text,
                        provenance=f"PILOT_{self.target_model}",
                    )
                    parse_ok = eval_rec.generation_parse_success
                    schema_ok = eval_rec.schema_valid
                    schema_err = eval_rec.schema_error
                    ground_ok = eval_rec.grounding_valid
                    ground_errs = list(eval_rec.grounding_errors)
                    gold_ok = eval_rec.semantic_gold_agreement
                    missing_pcs = list(eval_rec.missing_required_postconditions)
                    extra_pcs = list(eval_rec.extra_unsupported_predicates)
                    unsupp_schema = list(eval_rec.unsupported_tables_or_columns)
                    uniqueness_ok = eval_rec.uniqueness_respected
                    rejected = eval_rec.invalid_contract_rejected

                    if not parse_ok:
                        status = "PARSE_ERROR" if "{" in raw_text else "REFUSAL"
                        fail_reason = eval_rec.score_summary
                    elif not schema_ok:
                        status = "SCHEMA_ERROR"
                        fail_reason = eval_rec.score_summary
                    elif not ground_ok:
                        status = "GROUNDING_ERROR"
                        fail_reason = eval_rec.score_summary
                    elif not gold_ok:
                        status = "SEMANTIC_ERROR"
                        fail_reason = eval_rec.score_summary
                else:
                    parse_ok = False
                    schema_ok = False
                    schema_err = fail_reason
                    ground_ok = False
                    ground_errs = []
                    gold_ok = False
                    missing_pcs = []
                    extra_pcs = []
                    unsupp_schema = []
                    uniqueness_ok = False
                    rejected = True

                if schema_ok:
                    task_schema_valid += 1
                    total_schema_valid += 1
                if ground_ok:
                    task_grounded += 1
                    total_grounded += 1
                if gold_ok:
                    task_gold_equiv += 1
                    total_gold_equiv += 1

                record = PilotSampleLedgerRecord(
                    sample_id=sample_id,
                    request_id=request_id,
                    task_id=task_id,
                    sample_index=sample_idx,
                    model_identifier=self.target_model,
                    config_fingerprint=self.frozen_config.config_fingerprint,
                    prompt_fingerprint=bundle.prompt_hash,
                    schema_fingerprint=self.frozen_config.schema_fingerprint,
                    timestamp_utc=time.time(),
                    timestamp_iso=datetime.now(timezone.utc).isoformat(),
                    is_live_call=self.live_opt_in,
                    status=status,
                    latency_ms=latency_ms,
                    raw_response_artifact_path=raw_artifact_ref,
                    generation_parse_success=parse_ok,
                    schema_valid=schema_ok,
                    schema_error=schema_err,
                    grounding_valid=ground_ok,
                    grounding_errors=ground_errs,
                    semantic_gold_agreement=gold_ok,
                    missing_required_postconditions=missing_pcs,
                    extra_unsupported_predicates=extra_pcs,
                    unsupported_tables_or_columns=unsupp_schema,
                    uniqueness_respected=uniqueness_ok,
                    invalid_contract_rejected=rejected,
                    fallback_used=False,
                    rejection_reason=fail_reason if not gold_ok else None,
                    token_telemetry=token_usage,
                    cost_telemetry_usd=None,  # Provider unmetered / no pricing endpoint
                    failure_reason=fail_reason,
                )

                self.ledger.append_record(record)
                task_records.append(record)
                all_records.append(record)

            task_passed = task_gold_equiv >= PILOT_CONSERVATIVE_MIN_PASSING_SAMPLES_PER_TASK
            if task_passed:
                passing_tasks += 1

            task_summaries[task_id] = {
                "task_id": task_id,
                "samples_attempted": len(task_records),
                "schema_valid_count": task_schema_valid,
                "grounded_count": task_grounded,
                "gold_equivalent_count": task_gold_equiv,
                "task_passed": task_passed,
            }

        overall_gate_passed = passing_tasks >= PILOT_MIN_PASSING_TASKS

        return PilotEvaluationSummary(
            config_fingerprint=self.frozen_config.config_fingerprint,
            timestamp_utc=time.time(),
            is_live=self.live_opt_in,
            total_samples_attempted=len(all_records),
            schema_valid_samples=total_schema_valid,
            grounded_samples=total_grounded,
            gold_equivalent_samples=total_gold_equiv,
            passing_tasks_count=passing_tasks,
            overall_gate_passed=overall_gate_passed,
            task_summaries=task_summaries,
            sample_records=all_records,
        )


# ── CLI Entrypoint ──────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entrypoint for pilot evaluation runner.
    Default invocation (no flags): runs offline dry-run preflight (0 network calls).
    Live execution requires explicit --live-run-opt-in flag.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Controlled Pilot Evaluation for Nemotron DBOPS Contracts."
    )
    parser.add_argument(
        "--live-run-opt-in",
        action="store_true",
        default=False,
        help="Explicit opt-in required to execute live model requests. Without this flag, defaults to offline preflight.",
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("reports/dbops"),
        help="Directory for frozen configuration and append-only ledger.",
    )
    args = parser.parse_args(argv)

    if not args.live_run_opt_in:
        print("[PILOT] Running offline dry-run preflight (zero network calls)...")
        runner = PilotEvaluationRunner(pilot_dir=args.pilot_dir, live_opt_in=False)
        summary = runner.run_pilot()
        preflight = summary.task_summaries.get("PREFLIGHT_ONLY", {})
        print(f"[PILOT] Status: {preflight.get('status', 'UNKNOWN')}")
        print(f"[PILOT] Config fingerprint: {preflight.get('config_fingerprint')}")
        print(f"[PILOT] Network calls made: {preflight.get('network_calls_made', 0)}")
        print(f"[PILOT] Paid tokens consumed: {preflight.get('paid_tokens_consumed', 0)}")
        print(f"[PILOT] Total planned samples: {preflight.get('total_planned_samples', 0)}")
        print(f"[PILOT] Preflight check completed cleanly.")
        return 0

    print("[PILOT] Live execution mode requested with --live-run-opt-in.")
    runner = PilotEvaluationRunner(pilot_dir=args.pilot_dir, live_opt_in=True)
    summary = runner.run_pilot()
    print(f"[PILOT] Execution complete: {summary.total_samples_attempted} samples attempted.")
    print(f"[PILOT] Schema valid: {summary.schema_valid_samples}/{summary.total_samples_attempted}")
    print(f"[PILOT] Grounded: {summary.grounded_samples}/{summary.total_samples_attempted}")
    print(f"[PILOT] Gold equivalent: {summary.gold_equivalent_samples}/{summary.total_samples_attempted}")
    print(f"[PILOT] Overall gate passed: {summary.overall_gate_passed}")
    return 0 if summary.overall_gate_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

