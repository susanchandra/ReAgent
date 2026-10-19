"""
pilot_analysis.py — M9 Pilot Evidence & Scientific Analyzer.

Reads the immutable frozen pilot config and append-only pilot ledger, then
produces a deterministic JSON + Markdown analysis report.

SAFETY CONSTRAINTS:
  - Read-only: never calls append_record, record_dispatch_intent, or any ledger-mutating method.
  - Never calls any network/provider endpoint.
  - Never inspects .env, credentials, or API keys.
  - Explicitly labels every metric NOT_COMPARABLE or SYNTHETIC_OFFLINE_FIXTURE until a
    human-authorized live pilot has been executed.
  - Token / cost telemetry reported as UNKNOWN / NOT_REPORTED when absent — never substituted as 0.
  - Running the analyzer twice produces identical substantive output for the same ledger state.

Usage (default paths):
    python -m reagent.experiments.dbops.pilot_analysis

Usage (custom paths):
    python -m reagent.experiments.dbops.pilot_analysis \\
        --config path/to/pilot_config_frozen.json \\
        --ledger path/to/pilot_ledger.jsonl \\
        --output-dir path/to/output/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

# ── Version ────────────────────────────────────────────────────────────────────
PILOT_ANALYZER_VERSION = "1.0.0"

# ── Expected frozen fingerprints (from pilot_config_frozen.json) ───────────────
EXPECTED_CONFIG_FINGERPRINT = "c6e5c8532377f005e385aa1e47f9d52f9139c9a5a6a6f16bd6d1ae109e95a6c1"
EXPECTED_SCHEMA_FINGERPRINT = "0cf054673dbde67697f9ae588decfe40e83cfc5da4b23533bfd0087e0e126fea"
EXPECTED_PROMPT_HASHES: Dict[str, str] = {
    "task_01_refund_issue":      "bfed8cec24ca1e74f848b3bafdc10232a1fad306f81d2e792e2e8a3cce06aa49",
    "task_02_wallet_transfer":   "ba46c21351e0ea51a546a5c24453c870b550f943fee5c42c5d7d51742e4bf468",
    "task_05_order_cancellation":"b7366be38ec8b5207c0abd1fb4036cc4a8c65c2e165da612990d61e61ca610cd",
}

# ── Sample outcome taxonomy (ordered by severity) ──────────────────────────────
OUTCOME_NOT_RUN                 = "NOT_RUN"
OUTCOME_DISPATCHED_AMBIGUOUS    = "DISPATCHED_AMBIGUOUS"
OUTCOME_PROVIDER_ERROR          = "PROVIDER_ERROR"
OUTCOME_REFUSAL                 = "REFUSAL"
OUTCOME_PARSE_ERROR             = "PARSE_ERROR"
OUTCOME_SCHEMA_ERROR            = "SCHEMA_ERROR"
OUTCOME_GROUNDING_ERROR         = "GROUNDING_ERROR"
OUTCOME_INVALID_CONTRACT        = "INVALID_CONTRACT"
OUTCOME_VALID_CONTRACT          = "VALID_CONTRACT"

# Statuses that map to specific outcomes
_TERMINAL_TO_OUTCOME: Dict[str, str] = {
    "PROVIDER_ERROR": OUTCOME_PROVIDER_ERROR,
    "TIMEOUT":        OUTCOME_PROVIDER_ERROR,
    "REFUSAL":        OUTCOME_REFUSAL,
    "PARSE_ERROR":    OUTCOME_PARSE_ERROR,
    "SCHEMA_ERROR":   OUTCOME_SCHEMA_ERROR,
    "GROUNDING_ERROR":OUTCOME_GROUNDING_ERROR,
    "SEMANTIC_ERROR": OUTCOME_INVALID_CONTRACT,
    "SKIPPED":        OUTCOME_NOT_RUN,
    # COMPLETED is evaluated further: VALID_CONTRACT if semantic_gold_agreement else INVALID_CONTRACT
}

TERMINAL_STATUSES: Set[str] = {
    "COMPLETED", "REFUSAL", "PARSE_ERROR", "SCHEMA_ERROR",
    "GROUNDING_ERROR", "SEMANTIC_ERROR", "TIMEOUT", "PROVIDER_ERROR", "SKIPPED",
}

# Default report paths
_REPO_ROOT       = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG   = _REPO_ROOT / "reports" / "dbops" / "pilot_config_frozen.json"
DEFAULT_LEDGER   = _REPO_ROOT / "reports" / "dbops" / "pilot_ledger.jsonl"
DEFAULT_OUT_DIR  = _REPO_ROOT / "reports" / "dbops"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FingerprintCheckResult:
    """Result of comparing loaded config fingerprints against expected values."""
    config_fingerprint_ok:      bool
    schema_fingerprint_ok:      bool
    prompt_hashes_ok:           bool
    mismatches:                 List[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return self.config_fingerprint_ok and self.schema_fingerprint_ok and self.prompt_hashes_ok


@dataclass
class SampleAnalysisResult:
    """Analysis result for a single planned sample."""
    sample_id:                          str
    task_id:                            str
    outcome_category:                   str
    ledger_record:                      Optional[Dict[str, Any]]
    # Ledger-derived fields (None when NOT_RUN)
    status:                             Optional[str]
    is_live_call:                       Optional[bool]
    generation_parse_success:           Optional[bool]
    schema_valid:                       Optional[bool]
    grounding_valid:                    Optional[bool]
    semantic_gold_agreement:            Optional[bool]
    missing_required_postconditions:    List[str]
    extra_unsupported_predicates:       List[str]
    unsupported_tables_or_columns:      List[str]
    uniqueness_respected:               Optional[bool]
    token_telemetry_reported:           bool    # True only if token_telemetry != None
    token_telemetry:                    Optional[Dict[str, Any]]
    cost_telemetry_usd:                 Optional[float]
    latency_ms:                         Optional[int]


@dataclass
class TaskAnalysisSummary:
    """Per-task aggregated metrics."""
    task_id:                str
    planned_samples:        int
    samples_run:            int   # any terminal status
    samples_parsed:         int   # generation_parse_success=True
    samples_schema_valid:   int
    samples_grounded:       int
    samples_gold_equiv:     int
    outcomes:               List[str]
    conservative_pass:      bool  # all planned samples gold-equiv
    relaxed_pass:           bool  # at least 1 sample gold-equiv


@dataclass
class PilotAnalysisReport:
    """Complete M9 analysis report."""
    analyzer_version:               str
    generated_at_utc:               str
    config_path:                    str
    ledger_path:                    str
    ledger_exists:                  bool
    pilot_executed:                 bool   # True if at least 1 terminal record exists (that was a live call)
    fingerprint_check:              FingerprintCheckResult
    planned_sample_ids:             List[str]
    sample_results:                 List[SampleAnalysisResult]
    task_summaries:                 List[TaskAnalysisSummary]
    # Overall metrics (all with explicit denominators)
    n_planned:                      int
    n_attempted:                    int   # terminal status exists
    n_parsed:                       int
    n_schema_valid:                 int
    n_grounded:                     int
    n_gold_equiv:                   int
    n_not_run:                      int
    n_dispatched_ambiguous:         int
    schema_parse_success_rate:      Optional[str]  # N/M string or None when no attempts
    schema_validity_rate:           Optional[str]
    grounding_validity_rate:        Optional[str]
    gold_equivalence_rate:          Optional[str]  # primary KPI
    overall_gate_verdict:           str  # PASS | FAIL | INDETERMINATE | NOT_RUN
    n_passing_tasks_conservative:   int
    n_passing_tasks_relaxed:        int
    token_telemetry_note:           str
    cost_note:                      str
    comparability_warning:          str
    duplicate_sample_ids_detected:  List[str]
    unexpected_sample_ids_detected: List[str]


# ── Read-only ledger scanner ───────────────────────────────────────────────────

class PilotLedgerReader:
    """
    Read-only scanner of the append-only pilot JSONL ledger.
    Never mutates any file. Raises on corruption, but collects
    duplicate/unexpected IDs as analysis findings rather than hard errors.
    """

    def __init__(self, ledger_path: Path) -> None:
        self._path = ledger_path

    def read_all_records(
        self,
        planned_sample_ids: Set[str],
        expected_config_fingerprint: Optional[str] = None,
    ) -> Tuple[
        List[Dict[str, Any]],   # raw records in order
        List[str],              # duplicate terminal sample_ids
        List[str],              # unexpected (not in planned_ids) terminal sample_ids
        List[str],              # fingerprint mismatch warnings
    ]:
        """
        Scans ledger lines and returns all records plus integrity findings.
        Raises ValueError for malformed JSON lines.
        Does NOT raise on duplicates or unexpected IDs — collects them for reporting.
        """
        if not self._path.exists():
            return [], [], [], []

        raw_records: List[Dict[str, Any]] = []
        terminal_seen: Dict[str, int] = {}   # sample_id -> count of terminal records
        duplicates: List[str] = []
        unexpected: List[str] = []
        fp_warnings: List[str] = []

        with open(self._path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    data = json.loads(clean)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON on line {line_no} of {self._path}: {exc}"
                    ) from exc

                sample_id = data.get("sample_id", "")
                status    = data.get("status", "")

                # Fingerprint mismatch check
                rec_cfg_fp = data.get("config_fingerprint")
                if (
                    expected_config_fingerprint
                    and rec_cfg_fp
                    and rec_cfg_fp != expected_config_fingerprint
                ):
                    fp_warnings.append(
                        f"Line {line_no} sample_id='{sample_id}' "
                        f"config_fingerprint='{rec_cfg_fp}' expected='{expected_config_fingerprint}'"
                    )

                raw_records.append(data)

                # Track terminal records for duplicate / unexpected detection
                if status in TERMINAL_STATUSES:
                    terminal_seen[sample_id] = terminal_seen.get(sample_id, 0) + 1
                    if terminal_seen[sample_id] > 1 and sample_id not in duplicates:
                        duplicates.append(sample_id)
                    if sample_id not in planned_sample_ids and sample_id not in unexpected:
                        unexpected.append(sample_id)

        return raw_records, duplicates, unexpected, fp_warnings


# ── Main analyzer ──────────────────────────────────────────────────────────────

class PilotEvidenceAnalyzer:
    """
    Deterministic, read-only analyzer for the Nemotron DBOPS pilot.

    Produces:
      - reports/dbops/pilot_analysis_report.json
      - reports/dbops/pilot_analysis_report.md
    """

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _check_fingerprints(self, cfg: Dict[str, Any]) -> FingerprintCheckResult:
        mismatches: List[str] = []

        cfg_fp_ok = cfg.get("config_fingerprint") == EXPECTED_CONFIG_FINGERPRINT
        if not cfg_fp_ok:
            mismatches.append(
                f"config_fingerprint: got '{cfg.get('config_fingerprint')}' "
                f"expected '{EXPECTED_CONFIG_FINGERPRINT}'"
            )

        schema_fp_ok = cfg.get("schema_fingerprint") == EXPECTED_SCHEMA_FINGERPRINT
        if not schema_fp_ok:
            mismatches.append(
                f"schema_fingerprint: got '{cfg.get('schema_fingerprint')}' "
                f"expected '{EXPECTED_SCHEMA_FINGERPRINT}'"
            )

        actual_ph = cfg.get("prompt_hashes", {})
        ph_ok = True
        for task_id, expected_hash in EXPECTED_PROMPT_HASHES.items():
            got = actual_ph.get(task_id)
            if got != expected_hash:
                ph_ok = False
                mismatches.append(
                    f"prompt_hash[{task_id}]: got '{got}' expected '{expected_hash}'"
                )

        return FingerprintCheckResult(
            config_fingerprint_ok=cfg_fp_ok,
            schema_fingerprint_ok=schema_fp_ok,
            prompt_hashes_ok=ph_ok,
            mismatches=mismatches,
        )

    def _build_planned_sample_ids(self, task_ids: List[str], samples_per_task: int) -> List[str]:
        ids: List[str] = []
        for task_id in task_ids:
            for s in range(samples_per_task):
                # Convention matches pilot_eval.py: pilot_{task_id}_s{index}
                ids.append(f"pilot_{task_id}_s{s}")
        return ids

    def _classify_sample(
        self,
        sample_id: str,
        task_id: str,
        latest_terminal: Optional[Dict[str, Any]],
        dispatched_without_terminal: bool,
    ) -> SampleAnalysisResult:
        """Classify one sample into an outcome category using ledger data."""

        def _pick(key: str) -> Any:
            return latest_terminal.get(key) if latest_terminal else None

        if latest_terminal is None and not dispatched_without_terminal:
            # Sample was never touched
            return SampleAnalysisResult(
                sample_id=sample_id,
                task_id=task_id,
                outcome_category=OUTCOME_NOT_RUN,
                ledger_record=None,
                status=None,
                is_live_call=None,
                generation_parse_success=None,
                schema_valid=None,
                grounding_valid=None,
                semantic_gold_agreement=None,
                missing_required_postconditions=[],
                extra_unsupported_predicates=[],
                unsupported_tables_or_columns=[],
                uniqueness_respected=None,
                token_telemetry_reported=False,
                token_telemetry=None,
                cost_telemetry_usd=None,
                latency_ms=None,
            )

        if dispatched_without_terminal:
            return SampleAnalysisResult(
                sample_id=sample_id,
                task_id=task_id,
                outcome_category=OUTCOME_DISPATCHED_AMBIGUOUS,
                ledger_record=latest_terminal,
                status="DISPATCHED",
                is_live_call=_pick("is_live_call"),
                generation_parse_success=None,
                schema_valid=None,
                grounding_valid=None,
                semantic_gold_agreement=None,
                missing_required_postconditions=[],
                extra_unsupported_predicates=[],
                unsupported_tables_or_columns=[],
                uniqueness_respected=None,
                token_telemetry_reported=False,
                token_telemetry=None,
                cost_telemetry_usd=None,
                latency_ms=None,
            )

        # We have a terminal record
        assert latest_terminal is not None
        status = latest_terminal.get("status", "")
        sem_agree = latest_terminal.get("semantic_gold_agreement", False)

        if status == "COMPLETED" and sem_agree:
            outcome = OUTCOME_VALID_CONTRACT
        elif status == "COMPLETED" and not sem_agree:
            outcome = OUTCOME_INVALID_CONTRACT
        else:
            outcome = _TERMINAL_TO_OUTCOME.get(status, OUTCOME_INVALID_CONTRACT)

        tt = _pick("token_telemetry")

        return SampleAnalysisResult(
            sample_id=sample_id,
            task_id=task_id,
            outcome_category=outcome,
            ledger_record=latest_terminal,
            status=status,
            is_live_call=_pick("is_live_call"),
            generation_parse_success=_pick("generation_parse_success"),
            schema_valid=_pick("schema_valid"),
            grounding_valid=_pick("grounding_valid"),
            semantic_gold_agreement=sem_agree,
            missing_required_postconditions=latest_terminal.get("missing_required_postconditions") or [],
            extra_unsupported_predicates=latest_terminal.get("extra_unsupported_predicates") or [],
            unsupported_tables_or_columns=latest_terminal.get("unsupported_tables_or_columns") or [],
            uniqueness_respected=_pick("uniqueness_respected"),
            token_telemetry_reported=(tt is not None),
            token_telemetry=tt,
            cost_telemetry_usd=_pick("cost_telemetry_usd"),
            latency_ms=_pick("latency_ms"),
        )

    def _build_task_summary(self, task_id: str, results: List[SampleAnalysisResult]) -> TaskAnalysisSummary:
        planned = len(results)
        run = sum(1 for r in results if r.outcome_category not in (OUTCOME_NOT_RUN, OUTCOME_DISPATCHED_AMBIGUOUS))
        parsed = sum(1 for r in results if r.generation_parse_success is True)
        sv    = sum(1 for r in results if r.schema_valid is True)
        gv    = sum(1 for r in results if r.grounding_valid is True)
        ge    = sum(1 for r in results if r.outcome_category == OUTCOME_VALID_CONTRACT)

        return TaskAnalysisSummary(
            task_id=task_id,
            planned_samples=planned,
            samples_run=run,
            samples_parsed=parsed,
            samples_schema_valid=sv,
            samples_grounded=gv,
            samples_gold_equiv=ge,
            outcomes=[r.outcome_category for r in results],
            conservative_pass=(ge == planned and planned > 0),
            relaxed_pass=(ge >= 1),
        )

    def _rate(self, num: int, den: int) -> Optional[str]:
        if den == 0:
            return None
        return f"{num}/{den} ({100.0 * num / den:.1f}%)"

    def analyze(
        self,
        config_path: Path,
        ledger_path: Path,
    ) -> PilotAnalysisReport:
        """
        Main analysis entry point. Read-only; produces a PilotAnalysisReport.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Load config
        cfg = self._load_config(config_path)
        task_ids: List[str]          = cfg.get("task_ids", [])
        samples_per_task: int        = cfg.get("samples_per_task", 2)
        planned_ids                  = self._build_planned_sample_ids(task_ids, samples_per_task)
        planned_ids_set: Set[str]    = set(planned_ids)

        # 2. Fingerprint check
        fp_check = self._check_fingerprints(cfg)

        # 3. Read ledger (read-only)
        reader = PilotLedgerReader(ledger_path)
        raw_records, duplicates, unexpected, fp_warnings = reader.read_all_records(
            planned_sample_ids=planned_ids_set,
            expected_config_fingerprint=cfg.get("config_fingerprint"),
        )

        if fp_warnings:
            fp_check.mismatches.extend(
                [f"LedgerFingerprintMismatch: {w}" for w in fp_warnings]
            )
            fp_check.config_fingerprint_ok = False

        ledger_exists = ledger_path.exists()

        # 4. Build per-sample state maps from raw records
        # For each sample_id: latest terminal record (if any), dispatched-without-terminal flag
        latest_terminal: Dict[str, Dict[str, Any]] = {}
        dispatched_only: Set[str] = set()

        for rec in raw_records:
            sid    = rec.get("sample_id", "")
            status = rec.get("status", "")
            if status in TERMINAL_STATUSES:
                latest_terminal[sid] = rec
                dispatched_only.discard(sid)
            elif status == "DISPATCHED":
                if sid not in latest_terminal:
                    dispatched_only.add(sid)

        # 5. Classify every planned sample
        sample_results: List[SampleAnalysisResult] = []
        for sid in planned_ids:
            task_id = self._sample_id_to_task_id(sid, task_ids)
            result = self._classify_sample(
                sample_id=sid,
                task_id=task_id,
                latest_terminal=latest_terminal.get(sid),
                dispatched_without_terminal=(sid in dispatched_only),
            )
            sample_results.append(result)

        # 6. Per-task summaries
        task_results_map: Dict[str, List[SampleAnalysisResult]] = {t: [] for t in task_ids}
        for r in sample_results:
            if r.task_id in task_results_map:
                task_results_map[r.task_id].append(r)

        task_summaries = [
            self._build_task_summary(tid, task_results_map[tid])
            for tid in task_ids
        ]

        # 7. Overall metrics
        n_planned   = len(planned_ids)
        n_nr        = sum(1 for r in sample_results if r.outcome_category == OUTCOME_NOT_RUN)
        n_amb       = sum(1 for r in sample_results if r.outcome_category == OUTCOME_DISPATCHED_AMBIGUOUS)
        n_attempted = n_planned - n_nr   # includes DISPATCHED_AMBIGUOUS as attempted
        n_parsed    = sum(1 for r in sample_results if r.generation_parse_success is True)
        n_sv        = sum(1 for r in sample_results if r.schema_valid is True)
        n_gv        = sum(1 for r in sample_results if r.grounding_valid is True)
        n_ge        = sum(1 for r in sample_results if r.outcome_category == OUTCOME_VALID_CONTRACT)

        n_cons_pass = sum(1 for ts in task_summaries if ts.conservative_pass)
        n_relax_pass = sum(1 for ts in task_summaries if ts.relaxed_pass)

        # Pilot executed only if at least one live-call terminal record exists
        pilot_executed = any(
            r.is_live_call is True and r.status in TERMINAL_STATUSES
            for r in sample_results
        )

        # Gate verdict
        if n_nr == n_planned:
            gate_verdict = "NOT_RUN"
        elif n_amb > 0:
            gate_verdict = "INDETERMINATE"
        else:
            min_passing = cfg.get("acceptance_criteria", {}).get("overall_gate_threshold", "2 / 3")
            # Parse "2 / 3" → numerator
            try:
                req_pass = int(str(min_passing).split("/")[0].strip())
            except (IndexError, ValueError):
                req_pass = 2
            gate_verdict = "PASS" if n_cons_pass >= req_pass else "FAIL"

        return PilotAnalysisReport(
            analyzer_version=PILOT_ANALYZER_VERSION,
            generated_at_utc=now_iso,
            config_path=str(config_path),
            ledger_path=str(ledger_path),
            ledger_exists=ledger_exists,
            pilot_executed=pilot_executed,
            fingerprint_check=fp_check,
            planned_sample_ids=planned_ids,
            sample_results=sample_results,
            task_summaries=task_summaries,
            n_planned=n_planned,
            n_attempted=n_attempted,
            n_parsed=n_parsed,
            n_schema_valid=n_sv,
            n_grounded=n_gv,
            n_gold_equiv=n_ge,
            n_not_run=n_nr,
            n_dispatched_ambiguous=n_amb,
            schema_parse_success_rate=self._rate(n_parsed, n_attempted) if n_attempted else None,
            schema_validity_rate=self._rate(n_sv, n_parsed) if n_parsed else None,
            grounding_validity_rate=self._rate(n_gv, n_sv) if n_sv else None,
            gold_equivalence_rate=self._rate(n_ge, n_attempted) if n_attempted else None,
            overall_gate_verdict=gate_verdict,
            n_passing_tasks_conservative=n_cons_pass,
            n_passing_tasks_relaxed=n_relax_pass,
            token_telemetry_note=(
                "NOT_REPORTED — no live samples completed."
                if not pilot_executed
                else "PARTIAL — see per-sample token_telemetry fields. "
                     "NOT_REPORTED / None values indicate telemetry was absent for that sample; "
                     "never substituted as 0."
            ),
            cost_note="NOT_REPORTED — pricing endpoint not queried; cost telemetry unavailable.",
            comparability_warning=(
                "NOT_COMPARABLE — no live Nemotron samples have been executed. "
                "All current metrics reflect the pilot ledger state only. "
                "M5 synthetic benchmark results (29 true verifications, 0 false verifications "
                "for REAGENT_CONTRACT) are SYNTHETIC_OFFLINE_FIXTURES and are not comparable "
                "to this pilot analysis until a human-authorized live pilot is executed."
                if not pilot_executed
                else "PARTIAL_LIVE_EVIDENCE — some or all samples were executed against the "
                     "live provider. Compare gold-equivalence rates to M5 baseline only after "
                     "all 6 samples reach terminal state."
            ),
            duplicate_sample_ids_detected=duplicates,
            unexpected_sample_ids_detected=unexpected,
        )

    def _sample_id_to_task_id(self, sample_id: str, task_ids: List[str]) -> str:
        """Extract task_id from sample_id: pilot_{task_id}_s{n}."""
        for tid in task_ids:
            if sample_id.startswith(f"pilot_{tid}_s"):
                return tid
        # Fallback: try splitting
        parts = sample_id.split("_s")
        if len(parts) >= 2:
            candidate = parts[0].removeprefix("pilot_")
            if candidate in task_ids:
                return candidate
        return "unknown"

    def write_json_report(self, report: PilotAnalysisReport, output_path: Path) -> None:
        """Serialize report to JSON (read-only ledger state; deterministic)."""

        def _convert(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        data = _convert(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def write_markdown_report(self, report: PilotAnalysisReport, output_path: Path) -> None:
        """Generate human-readable Markdown analysis report."""
        lines: List[str] = []

        def h(level: int, text: str) -> None:
            lines.append(f"\n{'#' * level} {text}\n")

        def row(*cells: str) -> str:
            return "| " + " | ".join(cells) + " |"

        h(1, "DBOPS Nemotron Pilot — Evidence Analysis Report (M9)")
        lines.append(f"**Analyzer Version**: {report.analyzer_version}  ")
        lines.append(f"**Generated At (UTC)**: `{report.generated_at_utc}`  ")
        lines.append(f"**Config**: `{report.config_path}`  ")
        lines.append(f"**Ledger**: `{report.ledger_path}`  ")
        lines.append(f"**Ledger exists**: `{report.ledger_exists}`  ")
        lines.append(f"**Live pilot executed**: `{report.pilot_executed}`  \n")

        lines.append("> **⚠ COMPARABILITY WARNING**")
        lines.append(f"> {report.comparability_warning}\n")

        h(2, "Fingerprint Integrity")
        fp = report.fingerprint_check
        lines.append(f"- config_fingerprint OK: `{fp.config_fingerprint_ok}`")
        lines.append(f"- schema_fingerprint OK: `{fp.schema_fingerprint_ok}`")
        lines.append(f"- prompt_hashes OK:      `{fp.prompt_hashes_ok}`")
        if fp.mismatches:
            lines.append("\n**Mismatches detected:**")
            for m in fp.mismatches:
                lines.append(f"  - {m}")
        else:
            lines.append("\n✅ All fingerprints match expected frozen values.")

        h(2, "Planned Sample IDs")
        for sid in report.planned_sample_ids:
            lines.append(f"- `{sid}`")

        h(2, "Overall Metrics")
        lines.append(row("Metric", "Value", "Denominator"))
        lines.append(row("---", "---", "---"))
        lines.append(row("Planned samples",      str(report.n_planned),           "—"))
        lines.append(row("Attempted (terminal)", str(report.n_attempted),          f"of {report.n_planned}"))
        lines.append(row("Not run",              str(report.n_not_run),            f"of {report.n_planned}"))
        lines.append(row("Ambiguous dispatch",   str(report.n_dispatched_ambiguous), f"of {report.n_planned}"))
        lines.append(row("Parse successes",      str(report.n_parsed),            f"of {report.n_attempted} attempted"))
        lines.append(row("Schema valid",         str(report.n_schema_valid),       f"of {report.n_parsed} parsed"))
        lines.append(row("Grounding valid",      str(report.n_grounded),           f"of {report.n_schema_valid} schema-valid"))
        lines.append(row("Gold-equivalent",      str(report.n_gold_equiv),         f"of {report.n_attempted} attempted"))
        lines.append("")
        lines.append(f"- **Schema parse success rate**: {report.schema_parse_success_rate or 'N/A — no attempts'}")
        lines.append(f"- **Schema validity rate**:      {report.schema_validity_rate or 'N/A — no parsed results'}")
        lines.append(f"- **Grounding validity rate**:   {report.grounding_validity_rate or 'N/A — no schema-valid results'}")
        lines.append(f"- **Gold equivalence rate (KPI)**:  {report.gold_equivalence_rate or 'N/A — no attempts'}")
        lines.append(f"- **Overall gate verdict**:      `{report.overall_gate_verdict}`")
        lines.append(f"- **Passing tasks (conservative / 2-of-2)**: {report.n_passing_tasks_conservative} / {len(report.task_summaries)}")
        lines.append(f"- **Passing tasks (relaxed / 1-of-2)**:      {report.n_passing_tasks_relaxed} / {len(report.task_summaries)}")

        h(2, "Per-Task Summaries")
        for ts in report.task_summaries:
            h(3, ts.task_id)
            lines.append(row("Metric", "Value"))
            lines.append(row("---", "---"))
            lines.append(row("Planned",          str(ts.planned_samples)))
            lines.append(row("Run",              str(ts.samples_run)))
            lines.append(row("Parsed",           str(ts.samples_parsed)))
            lines.append(row("Schema valid",     str(ts.samples_schema_valid)))
            lines.append(row("Grounded",         str(ts.samples_grounded)))
            lines.append(row("Gold-equiv",       str(ts.samples_gold_equiv)))
            lines.append(row("Conservative pass",str(str(ts.conservative_pass))))
            lines.append(row("Relaxed pass",     str(str(ts.relaxed_pass))))
            lines.append(f"\nOutcomes: {ts.outcomes}")

        h(2, "Per-Sample Results")
        for sr in report.sample_results:
            h(3, sr.sample_id)
            lines.append(f"- **Outcome**: `{sr.outcome_category}`")
            lines.append(f"- **Task**: `{sr.task_id}`")
            lines.append(f"- **Status**: `{sr.status}`")
            lines.append(f"- **Live call**: `{sr.is_live_call}`")
            lines.append(f"- **Parse success**: `{sr.generation_parse_success}`")
            lines.append(f"- **Schema valid**: `{sr.schema_valid}`")
            lines.append(f"- **Grounding valid**: `{sr.grounding_valid}`")
            lines.append(f"- **Gold agreement**: `{sr.semantic_gold_agreement}`")
            lines.append(f"- **Token telemetry reported**: `{sr.token_telemetry_reported}`")
            if sr.token_telemetry is not None:
                lines.append(f"- **Token telemetry**: `{sr.token_telemetry}`")
            else:
                lines.append(f"- **Token telemetry**: `UNKNOWN / NOT_REPORTED`")
            lines.append(f"- **Cost (USD)**: `{sr.cost_telemetry_usd if sr.cost_telemetry_usd is not None else 'NOT_REPORTED'}`")
            lines.append(f"- **Latency (ms)**: `{sr.latency_ms}`")
            if sr.missing_required_postconditions:
                lines.append(f"- **Missing postconditions**: {sr.missing_required_postconditions}")
            if sr.extra_unsupported_predicates:
                lines.append(f"- **Extra unsupported predicates**: {sr.extra_unsupported_predicates}")
            if sr.unsupported_tables_or_columns:
                lines.append(f"- **Unsupported tables/columns**: {sr.unsupported_tables_or_columns}")

        h(2, "Integrity Findings")
        if report.duplicate_sample_ids_detected:
            lines.append(f"⚠ **Duplicate terminal sample IDs detected**: {report.duplicate_sample_ids_detected}")
        else:
            lines.append("✅ No duplicate terminal sample IDs.")
        if report.unexpected_sample_ids_detected:
            lines.append(f"⚠ **Unexpected sample IDs detected**: {report.unexpected_sample_ids_detected}")
        else:
            lines.append("✅ No unexpected sample IDs in ledger.")

        h(2, "Telemetry & Cost")
        lines.append(f"- **Token telemetry**: {report.token_telemetry_note}")
        lines.append(f"- **Cost**: {report.cost_note}")

        h(2, "M5 Synthetic Baseline Reference (NOT COMPARABLE)")
        lines.append(
            "The following M5 results are **SYNTHETIC_OFFLINE_FIXTURES** from the 480-trial "
            "deterministic benchmark. They are **NOT COMPARABLE** to live Nemotron pilot results "
            "until the pilot has been executed."
        )
        lines.append("\n| Policy | True Verifications | False Verifications | False Rate |")
        lines.append("| --- | --- | --- | --- |")
        lines.append("| REAGENT_CONTRACT      | 29 | 0  | 0.0%  |")
        lines.append("| SHARED_OBSERVATION    | 29 | 43 | 44.8% |")
        lines.append("| TOOL_CLAIM_ONLY       | 0  | 45 | 46.9% |")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M9 Pilot Evidence Analyzer — read-only, offline."
    )
    parser.add_argument("--config",     type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ledger",     type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    analyzer = PilotEvidenceAnalyzer()
    print(f"[M9] Loading config from:  {args.config}")
    print(f"[M9] Reading ledger from:  {args.ledger}")

    report = analyzer.analyze(args.config, args.ledger)

    json_out = args.output_dir / "pilot_analysis_report.json"
    md_out   = args.output_dir / "pilot_analysis_report.md"

    analyzer.write_json_report(report, json_out)
    analyzer.write_markdown_report(report, md_out)

    print(f"[M9] JSON report written:  {json_out}")
    print(f"[M9] MD   report written:  {md_out}")
    print(f"[M9] Gate verdict:         {report.overall_gate_verdict}")
    print(f"[M9] Pilot executed:       {report.pilot_executed}")
    print(f"[M9] NOT_RUN samples:      {report.n_not_run} / {report.n_planned}")
    if not report.fingerprint_check.all_ok:
        print(f"[M9] ⚠ FINGERPRINT MISMATCHES: {report.fingerprint_check.mismatches}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
