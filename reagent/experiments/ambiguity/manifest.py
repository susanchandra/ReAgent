"""
manifest.py — Pre-registered expectations for the ambiguity benchmark.

IMPORTANT: this file is written BEFORE the benchmark is run.
Every expected outcome is a falsifiable assertion, not a post-hoc rationalization.

Experiment design
-----------------
Central question:
  When a side-effecting tool call raises an exception, does ReAgent's
  observation-and-contract policy avoid duplicate side effects and missed
  effects MORE reliably than naive retry?

Three simulated operations (ordered by real-world frequency):
  1. create_resource   — creates a resource; not idempotent by default
  2. apply_transfer    — transfers a balance; NOT idempotent at all
  3. reserve_then_link — two-phase commit: reserve slot, then link object

Five deterministic fault modes (agent-visible error is indistinguishable
between FAIL_BEFORE_COMMIT and LOST_RESPONSE — see identifiability test):
  FAIL_BEFORE_COMMIT    — exception raised before write commits; world unchanged
  LOST_RESPONSE         — write commits, but response is dropped; world changed
  DELAYED_VISIBILITY    — write commits but not yet readable; transient
  STATUS_UNAVAILABLE    — status endpoint down; write may or may not have committed
  UNOBSERVABLE          — side effect committed to an external system with no probe

Six policy arms:
  ALWAYS_ESCALATE         — never retry, always ask operator
  NEVER_RETRY             — return error immediately, do nothing
  NAIVE_RETRY             — always retry once (duplicates committed effects!)
  IDEMPOTENCY_KEY         — retry with idempotency key (prevents duplicates IF server honours it)
  SHARED_OBSERVATION_POLICY — probe environment before deciding to retry (baseline)
  REAGENT_CONTRACT_POLICY — ReAgent: probe + frozen contract postconditions + recovery guard

Metrics (per trial):
  correct_outcome       — did the agent achieve exactly-once effect?
  duplicate_side_effect — was the side effect applied more than once?
  lost_side_effect      — did the agent fail to apply an effect that should have applied?
  unnecessary_escalation — did the agent escalate when recovery was safe?
  trials                — number of trials in this cell

=======================================================================
PRE-REGISTERED HYPOTHESES (must be checked with ZERO modification after data collected)
=======================================================================
"""

# ── Fault mode constants ─────────────────────────────────────────────────────
FAULT_FAIL_BEFORE_COMMIT   = "FAIL_BEFORE_COMMIT"
FAULT_LOST_RESPONSE        = "LOST_RESPONSE"
FAULT_DELAYED_VISIBILITY   = "DELAYED_VISIBILITY"
FAULT_STATUS_UNAVAILABLE   = "STATUS_UNAVAILABLE"
FAULT_UNOBSERVABLE         = "UNOBSERVABLE"

ALL_FAULT_MODES = [
    FAULT_FAIL_BEFORE_COMMIT,
    FAULT_LOST_RESPONSE,
    FAULT_DELAYED_VISIBILITY,
    FAULT_STATUS_UNAVAILABLE,
    FAULT_UNOBSERVABLE,
]

# ── Operation constants ──────────────────────────────────────────────────────
OP_CREATE_RESOURCE   = "create_resource"
OP_APPLY_TRANSFER    = "apply_transfer"
OP_RESERVE_THEN_LINK = "reserve_then_link"

ALL_OPERATIONS = [OP_CREATE_RESOURCE, OP_APPLY_TRANSFER, OP_RESERVE_THEN_LINK]

# ── Policy arm constants ─────────────────────────────────────────────────────
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

# ── Identifiability constraint (tested before running policies) ──────────────
# For matched FAIL_BEFORE_COMMIT and LOST_RESPONSE trials:
# - The agent-visible exception MUST be identical: same type, message, fields.
# - A canonical serialization hash of the agent-visible error MUST match.
# - The oracle MUST see different world states.
# If any pair fails the hash test, it is INVALID and must be fixed before running.
IDENTIFIABILITY_REQUIRED = True

# ── Pre-registered expected outcomes (per policy arm, per fault mode) ─────────
# Format: (policy_arm, fault_mode) → {
#     "duplicate_side_effect": bool or None (None = ambiguous / untestable),
#     "lost_side_effect": bool or None,
#     "correct_outcome": bool or None,
#     "rationale": str
# }
#
# Notation:
#   True  = we predict this WILL occur
#   False = we predict this will NOT occur
#   None  = outcome is genuinely ambiguous for this arm+fault combination
#           (both outcomes are possible; we do not penalise the benchmark for this)
#
EXPECTED_OUTCOMES: dict = {

    # ── ALWAYS_ESCALATE ──────────────────────────────────────────────────────
    # Never retries. No duplicate. But also never recovers from FAIL_BEFORE_COMMIT
    # (a lost effect). Always safe; never useful.
    (POLICY_ALWAYS_ESCALATE, FAULT_FAIL_BEFORE_COMMIT):   {
        "duplicate_side_effect": False,
        "lost_side_effect": True,      # escalation is NOT task completion
        "correct_outcome": False,
        "rationale": "Escalation leaves task incomplete when retry was safe.",
    },
    (POLICY_ALWAYS_ESCALATE, FAULT_LOST_RESPONSE):        {
        "duplicate_side_effect": False,
        "lost_side_effect": False,     # effect already committed; no retry = no duplicate
        "correct_outcome": True,       # effect committed; escalation is overcautious but not wrong
        "rationale": "Effect committed. Escalation avoids duplicate but reports incorrectly.",
    },
    (POLICY_ALWAYS_ESCALATE, FAULT_DELAYED_VISIBILITY):   {
        "duplicate_side_effect": False,
        "lost_side_effect": None,      # depends on whether caller retries after escalation
        "correct_outcome": None,
        "rationale": "Delayed visibility: effect committed but escalation outcome depends on caller.",
    },
    (POLICY_ALWAYS_ESCALATE, FAULT_STATUS_UNAVAILABLE):   {
        "duplicate_side_effect": False,
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Status unavailable: escalation is rational but task may remain incomplete.",
    },
    (POLICY_ALWAYS_ESCALATE, FAULT_UNOBSERVABLE):         {
        "duplicate_side_effect": False,
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Unobservable: cannot determine whether effect committed.",
    },

    # ── NEVER_RETRY ─────────────────────────────────────────────────────────
    # Returns error immediately. FAIL_BEFORE_COMMIT → correct (no effect = no retry needed).
    # LOST_RESPONSE → lost effect observed as an error by caller.
    (POLICY_NEVER_RETRY, FAULT_FAIL_BEFORE_COMMIT):       {
        "duplicate_side_effect": False,
        "lost_side_effect": False,     # nothing committed; returning error is correct
        "correct_outcome": False,      # task not completed
        "rationale": "Safe but incomplete: task fails without retry.",
    },
    (POLICY_NEVER_RETRY, FAULT_LOST_RESPONSE):            {
        "duplicate_side_effect": False,
        "lost_side_effect": False,     # effect already committed
        "correct_outcome": True,       # effect committed; caller sees error but state is correct
        "rationale": "Effect committed despite error; no retry = no duplicate.",
    },
    (POLICY_NEVER_RETRY, FAULT_DELAYED_VISIBILITY):       {
        "duplicate_side_effect": False,
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Depends on whether delayed effect eventually becomes visible.",
    },
    (POLICY_NEVER_RETRY, FAULT_STATUS_UNAVAILABLE):       {
        "duplicate_side_effect": False,
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Cannot determine without status; returning error may be appropriate.",
    },
    (POLICY_NEVER_RETRY, FAULT_UNOBSERVABLE):             {
        "duplicate_side_effect": False,
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Unobservable commit; returning error is conservative.",
    },

    # ── NAIVE_RETRY ──────────────────────────────────────────────────────────
    # Retries unconditionally. Safe for FAIL_BEFORE_COMMIT; DANGEROUS for LOST_RESPONSE.
    (POLICY_NAIVE_RETRY, FAULT_FAIL_BEFORE_COMMIT):       {
        "duplicate_side_effect": False,  # effect not committed; retry applies it once
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Retry after true failure correctly applies the effect.",
    },
    (POLICY_NAIVE_RETRY, FAULT_LOST_RESPONSE):            {
        "duplicate_side_effect": True,   # KEY PREDICTION: unconditional retry duplicates!
        "lost_side_effect": False,
        "correct_outcome": False,
        "rationale": "Effect already committed; naive retry causes DUPLICATE side effect.",
    },
    (POLICY_NAIVE_RETRY, FAULT_DELAYED_VISIBILITY):       {
        "duplicate_side_effect": True,   # retry before visibility resolves → duplicate
        "lost_side_effect": False,
        "correct_outcome": False,
        "rationale": "Delayed visibility: retry fires before first effect is readable → duplicate.",
    },
    (POLICY_NAIVE_RETRY, FAULT_STATUS_UNAVAILABLE):       {
        "duplicate_side_effect": None,   # depends on whether first attempt committed
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Cannot predict without knowing whether first commit landed.",
    },
    (POLICY_NAIVE_RETRY, FAULT_UNOBSERVABLE):             {
        "duplicate_side_effect": True,   # unobservable commit → retry blindly → duplicate
        "lost_side_effect": False,
        "correct_outcome": False,
        "rationale": "Unobservable commit + naive retry = duplicate side effect.",
    },

    # ── IDEMPOTENCY_KEY ──────────────────────────────────────────────────────
    # Sends a client-generated idempotency key with the request.
    # Server deduplication prevents duplicate *if* the server honours the key.
    # In our simulator, the server honours idempotency keys for create_resource
    # but NOT for apply_transfer (mimicking real-world partial support).
    (POLICY_IDEMPOTENCY_KEY, FAULT_FAIL_BEFORE_COMMIT):   {
        "duplicate_side_effect": False,
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Idempotency key safe to retry after true failure.",
    },
    (POLICY_IDEMPOTENCY_KEY, FAULT_LOST_RESPONSE):        {
        "duplicate_side_effect": False,  # server deduplicates for create_resource
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Server deduplication prevents duplicate for supported operations.",
    },
    # NOTE: for apply_transfer (no server-side idempotency), the outcome is the same as NAIVE_RETRY.
    # The runner will mark these per-operation.

    (POLICY_IDEMPOTENCY_KEY, FAULT_DELAYED_VISIBILITY):   {
        "duplicate_side_effect": False,
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Idempotency key survives delayed visibility window.",
    },
    (POLICY_IDEMPOTENCY_KEY, FAULT_STATUS_UNAVAILABLE):   {
        "duplicate_side_effect": False,
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Retry with key is safe when status is unavailable.",
    },
    (POLICY_IDEMPOTENCY_KEY, FAULT_UNOBSERVABLE):         {
        "duplicate_side_effect": None,  # depends on whether server honours key for this op
        "lost_side_effect": False,
        "correct_outcome": None,
        "rationale": "Unobservable: key may or may not protect depending on server support.",
    },

    # ── SHARED_OBSERVATION_POLICY ─────────────────────────────────────────────
    # Probes the environment (reads world state) before deciding to retry.
    # Avoids duplicate if the probe can detect the committed effect.
    # Fails for UNOBSERVABLE (no probe surface) and STATUS_UNAVAILABLE (probe fails).
    (POLICY_SHARED_OBSERVATION, FAULT_FAIL_BEFORE_COMMIT): {
        "duplicate_side_effect": False,
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Probe confirms no effect → safe to retry → correct outcome.",
    },
    (POLICY_SHARED_OBSERVATION, FAULT_LOST_RESPONSE):      {
        "duplicate_side_effect": False,
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Probe detects committed effect → skip retry → no duplicate.",
    },
    (POLICY_SHARED_OBSERVATION, FAULT_DELAYED_VISIBILITY): {
        "duplicate_side_effect": None,  # race: probe may fire before effect is readable
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Race between probe read and delayed effect visibility.",
    },
    (POLICY_SHARED_OBSERVATION, FAULT_STATUS_UNAVAILABLE): {
        "duplicate_side_effect": None,  # probe also fails → policy falls back
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Probe unavailable: observation policy cannot make informed decision.",
    },
    (POLICY_SHARED_OBSERVATION, FAULT_UNOBSERVABLE):       {
        "duplicate_side_effect": True,  # no probe surface → observation cannot detect commit
        "lost_side_effect": False,
        "correct_outcome": False,
        "rationale": "Unobservable commit: probe always returns 'no effect' → retry → duplicate.",
    },

    # ── REAGENT_CONTRACT_POLICY ───────────────────────────────────────────────
    # Uses frozen success contract postconditions + environment probe + recovery guard.
    # Applies the restraint check from pipeline._run_recovery_loop():
    #   if postcondition state already holds → skip retry → no duplicate.
    # Advantages over SHARED_OBSERVATION_POLICY:
    #   1. Contract specifies WHAT to check (not just "did something happen").
    #   2. Guard enforces policy limits (idempotency check, duplicate detection).
    #   3. Receipts record evidence so the outcome is auditable.
    (POLICY_REAGENT_CONTRACT, FAULT_FAIL_BEFORE_COMMIT):   {
        "duplicate_side_effect": False,
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Contract postcondition not satisfied → retry → correct.",
    },
    (POLICY_REAGENT_CONTRACT, FAULT_LOST_RESPONSE):        {
        "duplicate_side_effect": False,
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Restraint check: postcondition already holds → retry suppressed → no duplicate.",
    },
    (POLICY_REAGENT_CONTRACT, FAULT_DELAYED_VISIBILITY):   {
        "duplicate_side_effect": False,  # contract waits for postcondition; no early retry
        "lost_side_effect": False,
        "correct_outcome": True,
        "rationale": "Contract poll: postcondition must hold before marking success; prevents race.",
    },
    (POLICY_REAGENT_CONTRACT, FAULT_STATUS_UNAVAILABLE):   {
        "duplicate_side_effect": False,
        "lost_side_effect": None,       # escalates when evidence unavailable
        "correct_outcome": None,
        "rationale": "Escalates conservatively when contract cannot be verified.",
    },
    (POLICY_REAGENT_CONTRACT, FAULT_UNOBSERVABLE):         {
        "duplicate_side_effect": False,
        "lost_side_effect": None,
        "correct_outcome": None,
        "rationale": "Escalates when postcondition has no observable surface.",
    },
}

# ── Summary of key differences (used in report) ──────────────────────────────
KEY_DIFFERENTIATORS = {
    "NAIVE_RETRY vs REAGENT_CONTRACT (LOST_RESPONSE)": (
        "NAIVE_RETRY predicts duplicate; REAGENT_CONTRACT predicts no duplicate "
        "because the restraint check detects the committed state."
    ),
    "SHARED_OBSERVATION vs REAGENT_CONTRACT (UNOBSERVABLE)": (
        "SHARED_OBSERVATION predicts duplicate (no probe surface); "
        "REAGENT_CONTRACT escalates instead of guessing."
    ),
    "SHARED_OBSERVATION vs REAGENT_CONTRACT (DELAYED_VISIBILITY)": (
        "SHARED_OBSERVATION has ambiguous race outcome; "
        "REAGENT_CONTRACT contract-poll prevents premature retry."
    ),
}

# ── Small-sample warning ──────────────────────────────────────────────────────
# This is a deterministic offline benchmark, NOT a statistical study.
# With n=1 per cell (deterministic simulator), results are replicable but
# cannot support probability estimates.
# All conclusions must be reported as "behavioural correctness on defined scenarios"
# not as "probability of duplicate in production".
SAMPLE_SIZE_WARNING = (
    "SMALL SAMPLE: n=1 per (policy, fault_mode, operation) cell. "
    "Results are deterministic and replicable but not statistically generalisable. "
    "Report correctness counts, not percentages."
)
