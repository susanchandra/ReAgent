"""
manifest_v2.py — Pre-registered specification for Ambiguity Benchmark V2.

PRE-REGISTRATION METADATA
-------------------------
Pre-registration Timestamp: 2026-10-11T01:16:00Z
Base Commit: f581442fee5eda1b364a7177df72abe8eb842128
Manifest Version: 2.0.0-preregistered

EXPERIMENTAL DESIGN & PRINCIPLES
--------------------------------
1. Central Problem:
   When a state-changing tool call times out or encounters an ambiguous failure,
   the agent cannot know from the response alone whether the change committed.
2. Identifiability Invariant:
   In matched pairs of FAIL_BEFORE_COMMIT and LOST_RESPONSE, the agent-visible
   exception is canonically bit-for-bit identical (type, message, codes, fields).
3. Fair Baselines:
   All observation-capable policies receive identical observation budgets,
   action budgets, and timing semantics via an AgentView facade. No policy
   has direct access to simulator internals or the oracle.
4. Dual-View Partition Scoring:
   - World State (Independent Oracle): mutually exclusive partition summing to total trials:
     [exactly_one_effect, duplicate_effect, lost_effect, no_effect_expected]
   - Agent Claim: mutually exclusive partition summing to total trials:
     [claimed_complete_verified, claimed_complete_unverified, escalated, failed_terminal]
   - Rate metrics (e.g. correct_escalation_rate) are reported as ratios (num / denom).
5. Controls:
   Includes NO_FAULT controls to measure false blocking and needless escalation.
"""

from typing import Dict, Any, List

# ── Manifest Constants ────────────────────────────────────────────────────────
MANIFEST_VERSION = "2.0.0-preregistered"
PREREGISTRATION_BASE_COMMIT = "f581442fee5eda1b364a7177df72abe8eb842128"
PREREGISTRATION_TIMESTAMP = "2026-10-11T01:16:00Z"

# ── Fault Modes ───────────────────────────────────────────────────────────────
FAULT_NO_FAULT             = "NO_FAULT"
FAULT_FAIL_BEFORE_COMMIT   = "FAIL_BEFORE_COMMIT"
FAULT_LOST_RESPONSE        = "LOST_RESPONSE"
FAULT_DELAYED_VISIBILITY   = "DELAYED_VISIBILITY"
FAULT_STATUS_UNAVAILABLE   = "STATUS_UNAVAILABLE"
FAULT_UNOBSERVABLE         = "UNOBSERVABLE"

ALL_FAULT_MODES = [
    FAULT_NO_FAULT,
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
    FAULT_DELAYED_VISIBILITY,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_UNOBSERVABLE,
]

# ── Observability Classes ─────────────────────────────────────────────────────
OBS_FULLY_OBSERVABLE       = "FULLY_OBSERVABLE"
OBS_DELAYED                = "DELAYED"
OBS_STATUS_DOWN            = "STATUS_DOWN"
OBS_UNOBSERVABLE           = "UNOBSERVABLE"

FAULT_TO_OBSERVABILITY_CLASS = {
    FAULT_NO_FAULT: OBS_FULLY_OBSERVABLE,
    FAULT_FAIL_BEFORE_COMMIT: OBS_FULLY_OBSERVABLE,
    FAULT_LOST_RESPONSE: OBS_FULLY_OBSERVABLE,
    FAULT_DELAYED_VISIBILITY: OBS_DELAYED,
    FAULT_STATUS_UNAVAILABLE: OBS_STATUS_DOWN,
    FAULT_UNOBSERVABLE: OBS_UNOBSERVABLE,
}

# ── Operations ────────────────────────────────────────────────────────────────
OP_CREATE_RESOURCE   = "create_resource"
OP_APPLY_TRANSFER    = "apply_transfer"
OP_RESERVE_THEN_LINK = "reserve_then_link"

ALL_OPERATIONS = [
    OP_CREATE_RESOURCE,
    OP_APPLY_TRANSFER,
    OP_RESERVE_THEN_LINK,
]

# ── Policy Arms ───────────────────────────────────────────────────────────────
POLICY_ALWAYS_ESCALATE          = "ALWAYS_ESCALATE"
POLICY_NEVER_RETRY              = "NEVER_RETRY"
POLICY_NAIVE_RETRY              = "NAIVE_RETRY"
POLICY_IDEMPOTENCY_KEY          = "IDEMPOTENCY_KEY"
POLICY_SHARED_OBSERVATION       = "SHARED_OBSERVATION_POLICY"
POLICY_REAGENT_CONTRACT         = "REAGENT_CONTRACT_POLICY"

ALL_POLICY_ARMS = [
    POLICY_ALWAYS_ESCALATE,
    POLICY_NEVER_RETRY,
    POLICY_NAIVE_RETRY,
    POLICY_IDEMPOTENCY_KEY,
    POLICY_SHARED_OBSERVATION,
    POLICY_REAGENT_CONTRACT,
]

# ── Upstream Idempotency Key Support Configuration ───────────────────────────
# In real-world distributed architectures, only certain endpoints support idempotency keys.
# We document and test this explicitly as a configuration property:
IDEMPOTENCY_KEY_SUPPORT: Dict[str, bool] = {
    OP_CREATE_RESOURCE: True,       # REST PUT/POST with Idempotency-Key header supported
    OP_APPLY_TRANSFER: False,      # Legacy core banking RPC lacks key deduplication
    OP_RESERVE_THEN_LINK: True,    # Two-phase reservation endpoint supports idempotency
}

# ── Sample Size & Case Mix Specification ─────────────────────────────────────
# Minimum 60 paired trial units per arm:
# 6 fault modes * 3 operations * 4 deterministic seeds = 72 trial units per arm.
# 72 > 60 minimum requirement.
# Total trials across 6 arms: 72 * 6 = 432 trials.
# Fixed seeds ensure exact replicability across all arms.
FIXED_SEEDS = [101, 202, 303, 404]

# ── Scoring Partitions (Mutually Exclusive) ──────────────────────────────────
WORLD_STATE_CATEGORIES = [
    "exactly_one_effect",
    "duplicate_effect",
    "lost_effect",
    "no_effect_expected",
]

AGENT_CLAIM_CATEGORIES = [
    "claimed_complete_verified",
    "claimed_complete_unverified",
    "escalated",
    "failed_terminal",
]

# ── Pre-Registered Hypotheses & Expected Outcomes (Per Fault Mode) ───────────
# Stated strictly from scenario semantics BEFORE benchmark execution:
#
# 1. NO_FAULT:
#    All policies complete without duplicates or lost effects.
#    Measures false blocking or needless escalation.
#
# 2. FAIL_BEFORE_COMMIT:
#    - ALWAYS_ESCALATE / NEVER_RETRY: lost effect (operation not completed).
#    - NAIVE_RETRY / IDEMPOTENCY_KEY: retries safely, achieves exactly_one_effect.
#    - SHARED_OBSERVATION / REAGENT_CONTRACT: observes absence, retries safely, achieves exactly_one_effect.
#
# 3. LOST_RESPONSE:
#    - ALWAYS_ESCALATE / NEVER_RETRY: effect committed once, no duplicate.
#    - NAIVE_RETRY: unconditional retry causes DUPLICATE side effect.
#    - IDEMPOTENCY_KEY:
#        * OP_CREATE_RESOURCE & OP_RESERVE_THEN_LINK: deduplicated (exactly_one_effect).
#        * OP_APPLY_TRANSFER: upstream lacks key support -> DUPLICATE side effect.
#    - SHARED_OBSERVATION & REAGENT_CONTRACT:
#        * Probe observes committed state, restrains retry -> exactly_one_effect without duplicate.
#
# 4. DELAYED_VISIBILITY:
#    - With bounded poll budget (step 2 observation):
#      Both SHARED_OBSERVATION and REAGENT_CONTRACT observe committed state upon poll -> exactly_one_effect.
#    - NAIVE_RETRY retries before delay clears -> DUPLICATE side effect.
#
# 5. STATUS_UNAVAILABLE:
#    - Probes fail. SHARED_OBSERVATION and REAGENT_CONTRACT fail closed -> ESCALATED.
#    - NAIVE_RETRY retries blindly -> DUPLICATE side effect.
#
# 6. UNOBSERVABLE:
#    - Observation is impossible.
#    - SHARED_OBSERVATION (if following strict fail-closed specification) & REAGENT_CONTRACT escalate.
#    - NAIVE_RETRY duplicates.
