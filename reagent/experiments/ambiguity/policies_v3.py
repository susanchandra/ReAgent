"""
policies_v3.py — Six policy arms for Ambiguity Benchmark V3.

DESIGN PRINCIPLES
-----------------
1. All arms operate strictly through AgentView. No arm receives the simulator,
   oracle, or private ledger as a runtime input.
2. Budget enforcement is performed by AgentView (max 2 actions, max 2 probes).
   Policies do not need to track counts explicitly but must not exceed budgets.
3. CONFIRMED_ABSENT authority:
   - For APPLY_TRANSFER (non-idempotent, no server-side idempotency key):
     an observation alone is NOT treated as safe-retry authorization.
     SHARED_OBSERVATION_POLICY escalates unless the structured probe surface
     explicitly returns CONFIRMED_ABSENT (which APPLY_TRANSFER supports via
     transaction-log-consistent probing in the simulator).
   - NAIVE_RETRY does not probe and always retries — recorded without endorsement.
4. Arm 6 (REAGENT_CONTRACT_POLICY) defers to run_pipeline_arm6_v3 from
   pipeline_adapter_v3. The adapter owns probe/retry/escalation decisions;
   receipts are generated through the pipeline's verifier machinery.
5. Arms 1–4 are structurally identical to V2 baselines but import V3 constants.
   Documented differences where budgets or safety semantics differ are noted.
"""

from typing import Dict, Any, Optional

from reagent.experiments.ambiguity.agent_view import AgentView
from reagent.experiments.ambiguity.manifest_v3 import (
    POLICY_ALWAYS_ESCALATE,
    POLICY_NEVER_RETRY,
    POLICY_NAIVE_RETRY,
    POLICY_IDEMPOTENCY_KEY,
    POLICY_SHARED_OBSERVATION,
    POLICY_REAGENT_CONTRACT,
    IDEMPOTENCY_KEY_SUPPORT,
)
from reagent.experiments.ambiguity.faults import (
    ToolExecutionTimeoutError,
    StatusUnavailableError,
    UnobservableEvidenceError,
)
from reagent.experiments.ambiguity.pipeline_adapter_v3 import run_pipeline_arm6_v3


class PolicyRunResult:
    """Structured result record for a single policy arm execution in V3."""

    def __init__(
        self,
        policy_name: str,
        operation: str,
        agent_status: str,              # "COMPLETED", "ESCALATED", "FAILED"
        claimed_complete_verified: bool,
        claimed_complete_unverified: bool,
        escalated: bool,
        failed_terminal: bool,
        retried: bool,
        observed_existing_state: bool,
        escalation_reason: Optional[str] = None,
    ):
        self.policy_name = policy_name
        self.operation = operation
        self.agent_status = agent_status
        self.claimed_complete_verified = claimed_complete_verified
        self.claimed_complete_unverified = claimed_complete_unverified
        self.escalated = escalated
        self.failed_terminal = failed_terminal
        self.retried = retried
        self.observed_existing_state = observed_existing_state
        self.escalation_reason = escalation_reason

    def agent_claim_partition(self) -> Dict[str, bool]:
        """Returns the four mutually-exclusive Agent Claim partition fields."""
        return {
            "claimed_complete_verified": self.claimed_complete_verified,
            "claimed_complete_unverified": self.claimed_complete_unverified,
            "escalated": self.escalated,
            "failed_terminal": self.failed_terminal,
        }


def execute_policy_v3(
    policy_name: str,
    agent_view: AgentView,
    op_name: str,
    params: Dict[str, Any],
    fault_mode: str,
    task_id: str,
) -> PolicyRunResult:
    """
    Dispatches to the correct policy arm for V3 Ambiguity Benchmark.
    All arms interact exclusively through AgentView.

    Parameters
    ----------
    policy_name    : One of the six V3 POLICY_* constants.
    agent_view     : Fresh AgentView backed by a fresh StatefulSimulator.
    op_name        : One of the three V3 OP_* operation names.
    params         : Operation parameters (request_id, resource_id, etc.).
    fault_mode     : Active fault mode for this trial (from FAULT_* constants).
    task_id        : Unique trial identifier for the pipeline adapter.
    """

    # ── ARM 1: ALWAYS_ESCALATE ────────────────────────────────────────────────
    # Baseline: Immediately escalate on any failure; never retry.
    # Advantage: Zero duplicate risk.
    # Disadvantage: Maximum lost effects on transient failures.
    if policy_name == POLICY_ALWAYS_ESCALATE:
        try:
            agent_view.execute(op_name, params)
            # Success on first attempt
            return PolicyRunResult(
                policy_name, op_name, "COMPLETED",
                claimed_complete_verified=True,
                claimed_complete_unverified=False,
                escalated=False, failed_terminal=False,
                retried=False, observed_existing_state=False,
            )
        except ToolExecutionTimeoutError:
            return PolicyRunResult(
                policy_name, op_name, "ESCALATED",
                claimed_complete_verified=False,
                claimed_complete_unverified=False,
                escalated=True, failed_terminal=False,
                retried=False, observed_existing_state=False,
                escalation_reason="PolicyEscalatedImmediately",
            )

    # ── ARM 2: NEVER_RETRY ────────────────────────────────────────────────────
    # Conservative: Accept failure terminal on timeout; never retry or escalate.
    # Advantage: Zero duplicate risk (like ALWAYS_ESCALATE).
    # Disadvantage: Caller receives hard failure instead of escalation signal.
    elif policy_name == POLICY_NEVER_RETRY:
        try:
            agent_view.execute(op_name, params)
            return PolicyRunResult(
                policy_name, op_name, "COMPLETED",
                claimed_complete_verified=True,
                claimed_complete_unverified=False,
                escalated=False, failed_terminal=False,
                retried=False, observed_existing_state=False,
            )
        except ToolExecutionTimeoutError:
            return PolicyRunResult(
                policy_name, op_name, "FAILED",
                claimed_complete_verified=False,
                claimed_complete_unverified=False,
                escalated=False, failed_terminal=True,
                retried=False, observed_existing_state=False,
                escalation_reason="ToolTimeoutNeverRetry",
            )

    # ── ARM 3: NAIVE_RETRY ────────────────────────────────────────────────────
    # Adversarial baseline: Blindly retry on any timeout without probing.
    # Advantage: Recovers lost effects on FAIL_BEFORE_COMMIT.
    # Danger: Causes duplicate commits on LOST_RESPONSE and DELAYED_VISIBILITY.
    elif policy_name == POLICY_NAIVE_RETRY:
        try:
            agent_view.execute(op_name, params)
            return PolicyRunResult(
                policy_name, op_name, "COMPLETED",
                claimed_complete_verified=False,
                claimed_complete_unverified=True,
                escalated=False, failed_terminal=False,
                retried=False, observed_existing_state=False,
            )
        except ToolExecutionTimeoutError:
            # Blind retry — no probing, no idempotency key
            try:
                agent_view.execute(op_name, params)
                return PolicyRunResult(
                    policy_name, op_name, "COMPLETED",
                    claimed_complete_verified=False,
                    claimed_complete_unverified=True,
                    escalated=False, failed_terminal=False,
                    retried=True, observed_existing_state=False,
                )
            except Exception as e:
                return PolicyRunResult(
                    policy_name, op_name, "FAILED",
                    claimed_complete_verified=False,
                    claimed_complete_unverified=False,
                    escalated=False, failed_terminal=True,
                    retried=True, observed_existing_state=False,
                    escalation_reason=str(e),
                )

    # ── ARM 4: IDEMPOTENCY_KEY ────────────────────────────────────────────────
    # Informed baseline: Retry using a client-generated idempotency key.
    # Works for operations with server-side key deduplication (create_resource,
    # reserve_then_link). APPLY_TRANSFER lacks upstream key support: the second
    # call is still a raw duplicate and may commit a second time.
    # This difference is documented in IDEMPOTENCY_KEY_SUPPORT in manifest_v3.
    elif policy_name == POLICY_IDEMPOTENCY_KEY:
        idem_key = f"idem-{params.get('request_id', 'req-001')}"
        try:
            agent_view.execute(op_name, params, idempotency_key=idem_key)
            return PolicyRunResult(
                policy_name, op_name, "COMPLETED",
                claimed_complete_verified=False,
                claimed_complete_unverified=True,
                escalated=False, failed_terminal=False,
                retried=False, observed_existing_state=False,
            )
        except ToolExecutionTimeoutError:
            try:
                agent_view.execute(op_name, params, idempotency_key=idem_key)
                return PolicyRunResult(
                    policy_name, op_name, "COMPLETED",
                    claimed_complete_verified=False,
                    claimed_complete_unverified=True,
                    escalated=False, failed_terminal=False,
                    retried=True, observed_existing_state=False,
                )
            except Exception as e:
                return PolicyRunResult(
                    policy_name, op_name, "FAILED",
                    claimed_complete_verified=False,
                    claimed_complete_unverified=False,
                    escalated=False, failed_terminal=True,
                    retried=True, observed_existing_state=False,
                    escalation_reason=str(e),
                )

    # ── ARM 5: SHARED_OBSERVATION_POLICY ─────────────────────────────────────
    # Symmetric baseline: Probes environment before retry.
    # Uses probe_structured() to distinguish PRESENT / CONFIRMED_ABSENT /
    # DELAYED / errors. Escalates when evidence is unavailable or unobservable.
    #
    # CONFIRMED_ABSENT gate for non-idempotent ops:
    #   APPLY_TRANSFER does NOT support server-side idempotency keys. A retry
    #   is only authorized if probe_structured returns CONFIRMED_ABSENT (meaning
    #   the transaction log shows no commit on a strongly-consistent surface).
    #   If the probe returns any other status, this arm escalates for safety.
    elif policy_name == POLICY_SHARED_OBSERVATION:
        try:
            agent_view.execute(op_name, params)
            return PolicyRunResult(
                policy_name, op_name, "COMPLETED",
                claimed_complete_verified=True,
                claimed_complete_unverified=False,
                escalated=False, failed_terminal=False,
                retried=False, observed_existing_state=False,
            )
        except ToolExecutionTimeoutError:
            # Probe Step 1: Structured observation
            try:
                probe_res = agent_view.probe_structured(op_name, params, probe_step=1)
            except StatusUnavailableError as e:
                return PolicyRunResult(
                    policy_name, op_name, "ESCALATED",
                    claimed_complete_verified=False,
                    claimed_complete_unverified=False,
                    escalated=True, failed_terminal=False,
                    retried=False, observed_existing_state=False,
                    escalation_reason=f"StatusUnavailable: {e}",
                )
            except UnobservableEvidenceError as e:
                return PolicyRunResult(
                    policy_name, op_name, "ESCALATED",
                    claimed_complete_verified=False,
                    claimed_complete_unverified=False,
                    escalated=True, failed_terminal=False,
                    retried=False, observed_existing_state=False,
                    escalation_reason=f"EvidenceUnobservable: {e}",
                )

            probe_status = probe_res.get("status")

            # Effect is already present: restrain (no retry needed)
            if probe_status == "PRESENT":
                return PolicyRunResult(
                    policy_name, op_name, "COMPLETED",
                    claimed_complete_verified=True,
                    claimed_complete_unverified=False,
                    escalated=False, failed_terminal=False,
                    retried=False, observed_existing_state=True,
                )

            # Visibility delayed: consume probe step 2 within budget
            if probe_status == "DELAYED":
                try:
                    probe_res_2 = agent_view.probe_structured(op_name, params, probe_step=2)
                    if probe_res_2.get("status") == "PRESENT":
                        return PolicyRunResult(
                            policy_name, op_name, "COMPLETED",
                            claimed_complete_verified=True,
                            claimed_complete_unverified=False,
                            escalated=False, failed_terminal=False,
                            retried=False, observed_existing_state=True,
                        )
                    # Step 2 also not present: fall through to retry logic below
                    probe_status = probe_res_2.get("status", "CONFIRMED_ABSENT")
                except (StatusUnavailableError, UnobservableEvidenceError, Exception) as e:
                    return PolicyRunResult(
                        policy_name, op_name, "ESCALATED",
                        claimed_complete_verified=False,
                        claimed_complete_unverified=False,
                        escalated=True, failed_terminal=False,
                        retried=False, observed_existing_state=False,
                        escalation_reason=f"DelayedPollFailed: {e}",
                    )

            # Safety gate for non-idempotent operations:
            # Only permit retry if probe explicitly confirms CONFIRMED_ABSENT
            supports_idempotency = IDEMPOTENCY_KEY_SUPPORT.get(op_name, False)
            is_confirmed_absent = (probe_status == "CONFIRMED_ABSENT")

            if not supports_idempotency and not is_confirmed_absent:
                # APPLY_TRANSFER with ambiguous outcome: fail closed
                return PolicyRunResult(
                    policy_name, op_name, "ESCALATED",
                    claimed_complete_verified=False,
                    claimed_complete_unverified=False,
                    escalated=True, failed_terminal=False,
                    retried=False, observed_existing_state=False,
                    escalation_reason=(
                        f"NonIdempotentOperationAmbiguous: {op_name} "
                        f"probe_status={probe_status}; retry prohibited."
                    ),
                )

            # Effect is absent (confirmed) or op is idempotent: safe retry
            try:
                agent_view.execute(op_name, params)
                return PolicyRunResult(
                    policy_name, op_name, "COMPLETED",
                    claimed_complete_verified=True,
                    claimed_complete_unverified=False,
                    escalated=False, failed_terminal=False,
                    retried=True, observed_existing_state=False,
                )
            except Exception as e:
                return PolicyRunResult(
                    policy_name, op_name, "FAILED",
                    claimed_complete_verified=False,
                    claimed_complete_unverified=False,
                    escalated=False, failed_terminal=True,
                    retried=True, observed_existing_state=False,
                    escalation_reason=str(e),
                )

    # ── ARM 6: REAGENT_CONTRACT_POLICY ───────────────────────────────────────
    # Pipeline-integrated arm: Uses ReAgentPipeline for primary attempt,
    # contract freezing, and deterministic receipt verification.
    # NOTE: The adapter (pipeline_adapter_v3.run_pipeline_arm6_v3) owns
    # probe/retry/escalation decisions. This is NOT claimed as native
    # pipeline behavior; receipts are produced by the pipeline's verifier.
    elif policy_name == POLICY_REAGENT_CONTRACT:
        res = run_pipeline_arm6_v3(agent_view, op_name, params, fault_mode, task_id)
        return PolicyRunResult(
            policy_name=policy_name,
            operation=op_name,
            agent_status=res["agent_status"],
            claimed_complete_verified=res["claimed_complete_verified"],
            claimed_complete_unverified=False,
            escalated=res["escalated"],
            failed_terminal=(res["agent_status"] == "FAILED"),
            retried=res["retried"],
            observed_existing_state=res["observed_existing_state"],
            escalation_reason=res.get("escalation_reason"),
        )

    raise ValueError(f"Unknown policy arm: {policy_name!r}")
