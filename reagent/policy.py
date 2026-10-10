"""
ReAgent Policy Guard and Global Token Ledger (Milestone 4B revision)
Enforces safety boundaries, retry limits, authorization checks, and persistent token budgeting.

IMPORTANT NOTICE:
The GlobalTokenLedger is a LOCAL usage guard to prevent runaway loops and accidental overspend.
It is NOT a billing guarantee. It cannot guarantee the provider's actual charge or prevent
all provider-side billing outcomes.
"""

import os
import re
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from filelock import FileLock, Timeout

from reagent.models import RecoveryDecision, PolicyDecision
from reagent.taxonomy import RecoveryStrategyType

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_DIR = os.path.join(os.path.expanduser("~"), ".reagent")
DEFAULT_LEDGER_PATH = os.path.join(DEFAULT_LEDGER_DIR, "token_ledger.json")


class CorruptLedgerError(RuntimeError):
    """Raised when the persistent token ledger is malformed or corrupted. Fails closed."""
    pass


class BudgetExceededError(RuntimeError):
    """Raised when the global cumulative token budget or call count is exhausted."""
    pass


class AbandonedReservationError(RuntimeError):
    """Raised when the persistent token ledger contains un-reconciled active reservations from prior processes."""
    pass


HARD_LIMIT_MAX_CALLS = 50
HARD_LIMIT_MAX_TOKENS = 200_000
DEFAULT_LEDGER_MAX_CALLS = 20
DEFAULT_LEDGER_MAX_TOKENS = 60_000


def get_configured_ledger_caps(
    calls_override: Optional[int] = None,
    tokens_override: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Read ledger caps from overrides or environment:
      REAGENT_LEDGER_MAX_CALLS
      REAGENT_LEDGER_MAX_TOKENS
    with default offline probe limits (20 calls, 60,000 tokens)
    and HARD code-enforced upper limits:
      calls <= 50
      tokens <= 200,000
    Rejects negative, non-integer, or out-of-bounds values with ValueError.
    """
    calls = calls_override
    if calls is None:
        raw_c = os.environ.get("REAGENT_LEDGER_MAX_CALLS")
        if raw_c is not None and raw_c.strip():
            try:
                calls = int(raw_c.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid REAGENT_LEDGER_MAX_CALLS '{raw_c}': must be integer.") from exc
    if calls is None:
        calls = DEFAULT_LEDGER_MAX_CALLS

    tokens = tokens_override
    if tokens is None:
        raw_t = os.environ.get("REAGENT_LEDGER_MAX_TOKENS")
        if raw_t is not None and raw_t.strip():
            try:
                tokens = int(raw_t.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid REAGENT_LEDGER_MAX_TOKENS '{raw_t}': must be integer.") from exc
    if tokens is None:
        tokens = DEFAULT_LEDGER_MAX_TOKENS

    if not isinstance(calls, int) or calls <= 0 or calls > HARD_LIMIT_MAX_CALLS:
        raise ValueError(
            f"REAGENT_LEDGER_MAX_CALLS {calls} out of bounds: must be 1..{HARD_LIMIT_MAX_CALLS}."
        )

    if not isinstance(tokens, int) or tokens <= 0 or tokens > HARD_LIMIT_MAX_TOKENS:
        raise ValueError(
            f"REAGENT_LEDGER_MAX_TOKENS {tokens} out of bounds: must be 1..{HARD_LIMIT_MAX_TOKENS}."
        )

    return calls, tokens


def estimate_prompt_tokens(
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> int:
    """
    Conservative token estimator for complete provider request payloads.

    Estimates tokens across prompt strings or full message arrays.
    Rounds up at ~3 characters per token + framing overhead + floor of 20 tokens.

    LIMITATIONS AND DESIGN TRADEOFFS:
      1. Character-Ratio Heuristic: Approximates token count at 1 token per 3 characters
         (conservative compared to OpenAI's typical English ratio of ~1:4).
      2. Non-English & Dense Code Variance: In languages with multibyte UTF-8 characters
         or dense punctuation/code/regexes, actual tokenization can exceed 1:3.
         The estimator adds a minimum floor and message framing to compensate.
      3. BPE Ignorant: Does not execute exact Byte-Pair Encoding subword tokenization
         (to avoid heavyweight tokenizer dependencies and network downloads offline).
      4. Conservative Over-Estimation: May reserve up to 20-30% more tokens than
         actually consumed; actual usage is reconciled downward post-invocation.
    """
    try:
        total_chars = 0
        msg_count = 0
        if messages:
            for m in messages:
                content = m.get("content", "") or ""
                total_chars += len(content)
                msg_count += 1
        if prompt:
            total_chars += len(prompt)
            msg_count += 1
        if system_prompt:
            total_chars += len(system_prompt)
            msg_count += 1

        tokens = (total_chars + 2) // 3 + (msg_count * 4) + 10
        return max(tokens, 20)
    except Exception as exc:
        logger.warning("Prompt estimation failed (%s); using conservative default 500", type(exc).__name__)
        return 500


class GlobalTokenLedger:
    """
    Persistent cross-process JSON ledger tracking cumulative token usage.

    Enforces cumulative token and call caps across multiple runs and process restarts.
    Uses cross-process file locking (via filelock) for safe concurrent updates.
    Fails closed if the ledger file is corrupted, unreadable, or unwritable.
    """

    def __init__(
        self,
        ledger_path: Optional[str] = None,
        max_cumulative_tokens: Optional[int] = None,
        max_cumulative_calls: Optional[int] = None,
        lock_timeout_seconds: float = 5.0,
    ):
        configured_calls, configured_tokens = get_configured_ledger_caps(
            calls_override=max_cumulative_calls,
            tokens_override=max_cumulative_tokens,
        )
        self.ledger_path = os.path.abspath(
            ledger_path or os.environ.get("REAGENT_LEDGER_PATH", DEFAULT_LEDGER_PATH)
        )
        self.lock_path = f"{self.ledger_path}.lock"
        self.max_cumulative_tokens = configured_tokens
        self.max_cumulative_calls = configured_calls
        self.max_tokens = configured_tokens
        self.max_calls = configured_calls
        self.lock_timeout = lock_timeout_seconds

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)

        # Initialize ledger if not present; if present, validate without corrupting
        self._ensure_initialized()

    def _get_lock(self) -> FileLock:
        return FileLock(self.lock_path, timeout=self.lock_timeout)

    def _ensure_initialized(self) -> None:
        """Create empty ledger file if non-existent, or fail closed if corrupt."""
        with self._get_lock():
            if not os.path.exists(self.ledger_path):
                initial_data = {
                    "total_tokens_used": 0,
                    "total_calls_used": 0,
                    "active_reservations": {},
                }
                self._atomic_write(initial_data)
            else:
                # Validate existing ledger data. Will raise CorruptLedgerError if bad.
                self._read_ledger()

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        """Write JSON data atomically using a temp file."""
        tmp_path = f"{self.ledger_path}.tmp.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_path, self.ledger_path)
        except Exception as exc:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"LedgerWriteFailure: could not update ledger at '{self.ledger_path}': {exc}") from exc

    def _read_ledger(self) -> Dict[str, Any]:
        """Read and strictly validate the ledger JSON. Fails closed on corruption."""
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            raise CorruptLedgerError(
                f"LedgerStateCorrupt: failed to read JSON ledger at '{self.ledger_path}': {exc}. "
                f"Failing closed to protect budget. Do not silently reset."
            ) from exc

        # Strict validation
        if not isinstance(data, dict):
            raise CorruptLedgerError(f"LedgerStateCorrupt: root object is not a dict in '{self.ledger_path}'.")
        for req_key in ("total_tokens_used", "total_calls_used", "active_reservations"):
            if req_key not in data:
                raise CorruptLedgerError(f"LedgerStateCorrupt: missing required key '{req_key}' in '{self.ledger_path}'.")
        if not isinstance(data["total_tokens_used"], int) or not isinstance(data["total_calls_used"], int):
            raise CorruptLedgerError(f"LedgerStateCorrupt: token/call counters must be integers in '{self.ledger_path}'.")
        if not isinstance(data["active_reservations"], dict):
            raise CorruptLedgerError(f"LedgerStateCorrupt: active_reservations must be a dict in '{self.ledger_path}'.")

        return data

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of current ledger totals."""
        with self._get_lock():
            data = self._read_ledger()
            reserved = sum(data["active_reservations"].values())
            return {
                "total_tokens_used": data["total_tokens_used"],
                "total_calls_used": data["total_calls_used"],
                "active_reserved_tokens": reserved,
                "effective_tokens_used": data["total_tokens_used"] + reserved,
                "max_cumulative_tokens": self.max_cumulative_tokens,
                "max_cumulative_calls": self.max_cumulative_calls,
                "remaining_tokens": max(0, self.max_cumulative_tokens - (data["total_tokens_used"] + reserved)),
                "remaining_calls": max(0, self.max_cumulative_calls - data["total_calls_used"]),
            }

    def check_and_reserve(self, estimated_tokens: int = 500) -> str:
        """
        Check budget and create a reservation before invoking a provider.
        Must be called BEFORE every LLM call (planning, diagnosis, recovery).

        Returns:
            reservation_id string to be reconciled after call completion.

        Raises:
            BudgetExceededError: if cumulative token or call limit would be breached.
            CorruptLedgerError: if ledger is corrupt.
        """
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be non-negative.")

        with self._get_lock():
            data = self._read_ledger()
            curr_used = data["total_tokens_used"]
            curr_calls = data["total_calls_used"]
            curr_reserved = sum(data["active_reservations"].values())

            if curr_calls + 1 > self.max_cumulative_calls:
                raise BudgetExceededError(
                    f"GlobalTokenBudgetExceeded: call limit reached ({curr_calls}/{self.max_cumulative_calls})."
                )

            effective = curr_used + curr_reserved + estimated_tokens
            if effective > self.max_cumulative_tokens:
                raise BudgetExceededError(
                    f"GlobalTokenBudgetExceeded: token cap would be breached "
                    f"({effective}/{self.max_cumulative_tokens} tokens)."
                )

            res_id = str(uuid.uuid4())
            data["active_reservations"][res_id] = estimated_tokens
            self._atomic_write(data)
            return res_id

    def get_active_reservations(self) -> Dict[str, int]:
        """Return a snapshot of un-reconciled active reservations."""
        with self._get_lock():
            data = self._read_ledger()
            return dict(data.get("active_reservations", {}))

    def assert_no_abandoned_reservations(self) -> None:
        """
        Block startup if there are un-reconciled active reservations from prior processes/crashes.
        Never auto-resets or refunds abandoned reservations.
        """
        active = self.get_active_reservations()
        if active:
            raise AbandonedReservationError(
                f"AbandonedReservationsDetected: ledger contains {len(active)} active un-reconciled "
                f"reservation(s) ({sum(active.values())} tokens). Startup blocked to protect budget."
            )

    def reconcile_usage(
        self,
        reservation_id: str,
        reported_usage: Optional[Dict[str, int]] = None,
        conservative_default: Optional[int] = None,
        actual_tokens: Optional[int] = None,
    ) -> int:
        """
        Reconcile a reservation with actual reported token usage.

        If reported_usage is absent, empty, or missing 'total_tokens',
        unknown usage is NOT treated as zero: we conservatively charge
        the full reserved amount (or conservative_default if higher).
        """
        with self._get_lock():
            data = self._read_ledger()
            if reservation_id not in data["active_reservations"]:
                logger.warning("reconcile_usage: reservation '%s' not found.", reservation_id)
                reserved = max(500, conservative_default or 0)
            else:
                reserved = data["active_reservations"].pop(reservation_id)

            actual = None
            if actual_tokens is not None:
                actual = actual_tokens
            elif reported_usage and isinstance(reported_usage, dict):
                actual = reported_usage.get("total_tokens")

            if actual is None or actual < 0:
                # Unknown usage must charge the full reservation or more, never less
                actual_charge = max(reserved, conservative_default or 0)
                logger.info(
                    "Token usage unknown; conservatively charging full reservation %d tokens for '%s'.",
                    actual_charge, reservation_id
                )
            else:
                actual_charge = int(actual)

            data["total_tokens_used"] += actual_charge
            data["total_calls_used"] += 1
            self._atomic_write(data)
            return actual_charge

    def release_reservation(self, reservation_id: str) -> None:
        """Release a reservation without charging if call was cancelled before remote invocation."""
        with self._get_lock():
            data = self._read_ledger()
            if reservation_id in data["active_reservations"]:
                data["active_reservations"].pop(reservation_id)
                self._atomic_write(data)


class PolicyGuard:
    """
    Evaluates proposed recovery decisions against active safety constraints.
    Prevents infinite retry loops, unauthorized actions, and destructive tool usage.
    """

    DEFAULT_MAX_RETRIES = 3
    FORBIDDEN_COMMAND_PATTERNS = [
        r"\brm\s+-rf\b",
        r"\bformat\s+[a-z]:\b",
        r"\bdel\s+/[fqs]+\b",
        r"\bdrop\s+database\b",
        r"\bshutdown\b",
        r"\bmkfs\b",
    ]

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        forbidden_tools: Optional[List[str]] = None,
    ):
        self.max_retries = max_retries
        self.forbidden_tools = set(forbidden_tools or [])
        self._retry_counts: Dict[str, int] = {}

    def check_recovery(
        self,
        task_id: str,
        decision: RecoveryDecision,
        proposed_payload: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """Evaluate a proposed recovery decision against safety policies."""
        violations: List[str] = []
        rules_checked: List[str] = ["MAX_RETRY_LIMIT", "FORBIDDEN_TOOLS", "DESTRUCTIVE_PAYLOAD_CHECK"]

        # 1. Retry count check
        current_retries = self._retry_counts.get(task_id, 0)
        if decision.strategy in (
            RecoveryStrategyType.RETRY_SAME,
            RecoveryStrategyType.RETRY_MODIFIED_INPUT,
            RecoveryStrategyType.WAIT_AND_RETRY,
        ):
            if current_retries >= self.max_retries:
                violations.append(
                    f"Retry limit exceeded for task {task_id} ({current_retries}/{self.max_retries}). Must escalate."
                )

        # 2. Forbidden tool check
        if decision.fallback_tool_name and decision.fallback_tool_name in self.forbidden_tools:
            violations.append(f"Fallback tool '{decision.fallback_tool_name}' is blocked by security policy.")

        # 3. Dangerous payload check (regex with word boundaries)
        if proposed_payload:
            payload_str = str(proposed_payload).lower()
            for bad_pat in self.FORBIDDEN_COMMAND_PATTERNS:
                if re.search(bad_pat, payload_str):
                    violations.append(f"Payload contains prohibited dangerous command pattern: '{bad_pat}'.")

        is_allowed = len(violations) == 0
        if is_allowed:
            self._retry_counts[task_id] = current_retries + 1

        return PolicyDecision(
            is_allowed=is_allowed,
            violations=violations,
            checked_rules=rules_checked,
        )

    def reset_task(self, task_id: str):
        """Reset retry counters when a task succeeds or terminates."""
        self._retry_counts.pop(task_id, None)


class TokenBudgetGuard:
    """
    Coordinates local per-run caps and wraps the persistent GlobalTokenLedger.

    Enforced before EVERY provider call (planning, diagnosis, recovery).
    """

    def __init__(
        self,
        max_calls_per_run: int = 5,
        max_tokens_per_run: int = 4000,
        ledger: Optional[GlobalTokenLedger] = None,
        ledger_path: Optional[str] = None,
    ):
        self.max_calls_per_run = max_calls_per_run
        self.max_tokens_per_run = max_tokens_per_run
        self._calls: int = 0
        self._tokens: int = 0
        self.ledger = ledger or GlobalTokenLedger(ledger_path=ledger_path)

    @property
    def calls_used(self) -> int:
        return self._calls

    @property
    def tokens_used(self) -> int:
        return self._tokens

    def budget_remaining(self) -> Dict[str, Any]:
        global_state = self.ledger.get_state()
        return {
            "run_calls_remaining": max(0, self.max_calls_per_run - self._calls),
            "run_tokens_remaining": max(0, self.max_tokens_per_run - self._tokens),
            "global_tokens_remaining": global_state["remaining_tokens"],
            "global_calls_remaining": global_state["remaining_calls"],
        }

    def allow_call(self, estimated_tokens: int = 0) -> bool:
        """
        Check if a call is allowed under both per-run and global limits.
        """
        if self._calls >= self.max_calls_per_run:
            return False
        if self._tokens + estimated_tokens > self.max_tokens_per_run:
            return False
        try:
            state = self.ledger.get_state()
            if state["remaining_calls"] <= 0:
                return False
            if state["remaining_tokens"] < estimated_tokens:
                return False
            return True
        except Exception as exc:
            logger.error("allow_call check failed closed due to ledger error: %s", exc)
            return False


    def reserve(self, estimated_tokens: int = 500) -> str:
        """Reserve tokens in the global ledger before provider call."""
        if not self.allow_call(estimated_tokens):
            raise BudgetExceededError("Run or global token budget exceeded.")
        return self.ledger.check_and_reserve(estimated_tokens=estimated_tokens)

    def reserve_for_call(
        self,
        prompt: str,
        max_completion_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Conservative reservation: estimated prompt tokens + configured max completion tokens.
        """
        prompt_est = estimate_prompt_tokens(prompt, system_prompt)
        total_estimate = prompt_est + max_completion_tokens
        res_id = self.reserve(estimated_tokens=total_estimate)
        return res_id, total_estimate

    def reconcile(
        self,
        reservation_id: str,
        usage: Optional[Dict[str, int]] = None,
        conservative_default: Optional[int] = None,
    ) -> int:
        """Reconcile reported usage and record run-level tokens."""
        self._calls += 1
        charged = self.ledger.reconcile_usage(
            reservation_id=reservation_id,
            reported_usage=usage,
            conservative_default=conservative_default,
        )
        self._tokens += charged
        return charged

    def record_usage(self, usage: Dict[str, int]) -> bool:
        """Legacy helper for backward compatibility."""
        self._calls += 1
        total = usage.get("total_tokens", 0)
        self._tokens += total
        return self._tokens < self.max_tokens_per_run

    def record_truncated(self) -> str:
        self._calls += 1
        return "fallback"

    def reset(self) -> None:
        self._calls = 0
        self._tokens = 0
