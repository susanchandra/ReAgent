"""
manifest_v3.py — Pre-registered specification for Ambiguity Benchmark V3.

PRE-REGISTRATION METADATA
-------------------------
Pre-registration Timestamp: 2026-10-11T14:48:00Z
Base Commit: 3270ca8038c65bda4d6ef5ae2bbe103eb858d0a6
Checkpoint Heritage: f581442fee5eda1b364a7177df72abe8eb842128
Manifest Version: 3.0.0-preregistered

EXPERIMENTAL DESIGN & PRINCIPLES
--------------------------------
1. Central Problem:
   When a state-changing tool call times out or encounters an ambiguous failure,
   the agent cannot know from the response alone whether the change committed.
2. Identifiability Invariant:
   In matched pairs of FAIL_BEFORE_COMMIT and LOST_RESPONSE, caller-visible
   exceptions are canonically identical in serialization and hash.
3. Authority of Absence Observations:
   Ordinary ABSENT (e.g. unobservable, eventual consistency lag) is NOT treated
   as CONFIRMED_ABSENT. Only strongly consistent read-your-writes probe surfaces
   can return CONFIRMED_ABSENT. If an observation is merely ABSENT without
   authoritative confirmation, or is UNAVAILABLE / UNOBSERVABLE, safety invariants
   require the policy to FAIL CLOSED (ESCALATE) rather than retry non-idempotent operations.
4. Unit of Analysis:
   - Primary analytical unit: 18 distinct scenario conditions
     (6 fault modes * 3 operations).
   - Replicate structure: 4 deterministic parameter variants per condition
     yielding 72 trial executions per policy arm.
   - Total executions across 6 policy arms: 18 * 4 * 6 = 432 executions.
5. Shared Budgets:
   - Action budget: 1 primary execution + at most 1 recovery retry.
   - Observation budget: at most 2 probe calls.
6. Dual-View Partition Scoring:
   - World State (Independent Oracle): [exactly_one_effect, duplicate_effect, lost_effect, no_effect_expected]
     sums strictly to total trials per arm.
   - Agent Claim: [claimed_complete_verified, claimed_complete_unverified, escalated, failed_terminal]
     sums strictly to total trials per arm.
"""

from typing import Dict, Any, List

# ── Manifest Constants ────────────────────────────────────────────────────────
MANIFEST_VERSION = "3.0.0-preregistered"
PREREGISTRATION_BASE_COMMIT = "3270ca8038c65bda4d6ef5ae2bbe103eb858d0a6"
CHECKPOINT_ROLLBACK_ANCHOR = "f581442fee5eda1b364a7177df72abe8eb842128"
PREREGISTRATION_TIMESTAMP = "2026-10-11T14:48:00Z"

# ── Fault Modes (6 Modes) ─────────────────────────────────────────────────────
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

# ── Operations (3 Operations) ─────────────────────────────────────────────────
OP_CREATE_RESOURCE   = "create_resource"
OP_APPLY_TRANSFER    = "apply_transfer"
OP_RESERVE_THEN_LINK = "reserve_then_link"

ALL_OPERATIONS = [
    OP_CREATE_RESOURCE,
    OP_APPLY_TRANSFER,
    OP_RESERVE_THEN_LINK,
]

# ── Policy Arms (6 Arms) ──────────────────────────────────────────────────────
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
IDEMPOTENCY_KEY_SUPPORT: Dict[str, bool] = {
    OP_CREATE_RESOURCE: True,
    OP_APPLY_TRANSFER: False,      # Non-idempotent; upstream RPC lacks key deduplication
    OP_RESERVE_THEN_LINK: True,
}

# ── Probe Authority & Consistency Classification ─────────────────────────────
# Maps operation to whether its probe observation is strongly authoritative
PROBE_AUTHORITY_MAPPING: Dict[str, str] = {
    OP_CREATE_RESOURCE: "AUTHORITATIVE_CONSISTENT",
    OP_APPLY_TRANSFER: "TRANSACTION_LOG_CONSISTENT",
    OP_RESERVE_THEN_LINK: "AUTHORITATIVE_CONSISTENT",
}

# ── Unit of Analysis & Parameter Variants ────────────────────────────────────
# 18 distinct scenario conditions (6 fault modes * 3 operations).
# 4 fixed parameter variants per condition:
FIXED_SEEDS = [101, 202, 303, 404]

DISTINCT_CONDITIONS_COUNT = len(ALL_FAULT_MODES) * len(ALL_OPERATIONS)  # 18
EXECUTIONS_PER_ARM = DISTINCT_CONDITIONS_COUNT * len(FIXED_SEEDS)        # 72
TOTAL_BENCHMARK_EXECUTIONS = EXECUTIONS_PER_ARM * len(ALL_POLICY_ARMS)   # 432

# ── Shared Budgets ────────────────────────────────────────────────────────────
MAX_ACTION_ATTEMPTS = 2      # 1 primary attempt + at most 1 recovery retry
MAX_PROBE_OBSERVATIONS = 2   # at most 2 probe calls

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

# ── Pre-Registered Expected Outcomes (Per Fault Mode & Condition) ────────────
# Stated strictly prior to execution:
#
# 1. NO_FAULT (3 conditions, 12 runs per arm):
#    - All arms achieve exactly_one_effect (12/12).
#    - Observation arms achieve claimed_complete_verified (12/12).
#    - Measures false blocking (expected: 0%).
#
# 2. FAIL_BEFORE_COMMIT (3 conditions, 12 runs per arm):
#    - ALWAYS_ESCALATE / NEVER_RETRY: lost_effect (12/12).
#    - NAIVE_RETRY: if retry is transiently allowed, exactly_one_effect (12/12);
#      if retry persists failure, lost_effect (12/12).
#    - SHARED_OBSERVATION & REAGENT_CONTRACT: observes absence; if absence is
#      CONFIRMED_ABSENT, retry recovers safely (12/12 exactly_one_effect).
#
# 3. LOST_RESPONSE (3 conditions, 12 runs per arm):
#    - ALWAYS_ESCALATE / NEVER_RETRY: exactly_one_effect (12/12, no retry).
#    - NAIVE_RETRY: duplicate_effect (12/12, unconditional retry duplicates).
#    - IDEMPOTENCY_KEY:
#        * OP_CREATE_RESOURCE & OP_RESERVE_THEN_LINK: exactly_one_effect (8/12).
#        * OP_APPLY_TRANSFER: duplicate_effect (4/12, lacks upstream key support).
#    - SHARED_OBSERVATION & REAGENT_CONTRACT:
#        * Probe confirms PRESENT; restraint triggered -> exactly_one_effect (12/12, 0 duplicates).
#
# 4. DELAYED_VISIBILITY (3 conditions, 12 runs per arm):
#    - NAIVE_RETRY: duplicate_effect (12/12).
#    - SHARED_OBSERVATION & REAGENT_CONTRACT:
#        * Probe step 1 absent; probe step 2 present; restraint triggered -> exactly_one_effect (12/12).
#
# 5. STATUS_UNAVAILABLE (3 conditions, 12 runs per arm):
#    - NAIVE_RETRY: duplicate_effect (12/12).
#    - SHARED_OBSERVATION & REAGENT_CONTRACT:
#        * Probe throws StatusUnavailable; fail-closed -> escalated (12/12, 0 duplicates).
#
# 6. UNOBSERVABLE (3 conditions, 12 runs per arm):
#    - NAIVE_RETRY: duplicate_effect (12/12).
#    - SHARED_OBSERVATION & REAGENT_CONTRACT:
#        * Probe unobservable; fail-closed -> escalated (12/12, 0 duplicates).
