# ReAgent Architecture: Verification & Reliability Layer

## 1. System Overview

ReAgent sits between the LLM cognitive layer and external system tools, intercepting state-changing operations to guarantee verified execution.

```
       +-------------------------------------------------------+
       |               LLM Agent (e.g., Nemotron)              |
       +-------------------------------------------------------+
                                  |
                   1. Structured Decision Request
                                  v
       +-------------------------------------------------------+
       |                  ReAgent Pipeline                     |
       |  +-------------------------------------------------+  |
       |  | 2. Freeze SuccessContract                       |  |
       |  |    - Preconditions (Starting balances)          |  |
       |  |    - Postconditions (Target values, deltas)     |  |
       |  |    - Invariants (Unique keys, audit records)    |  |
       |  +-------------------------------------------------+  |
       |                          |                            |
       |       3. Action Attempt  |                            |
       |                          v                            |
       |              +-----------------------+                |
       |              | State-Changing Tool   |                |
       |              +-----------------------+                |
       |                          |                            |
       |    4. Tool Returns Timeout / Ambiguity                |
       |                          |                            |
       |  +-----------------------v-------------------------+  |
       |  | 5. Independent Read-Only Observer               |  |
       |  |    (Isolated connection, zero mutation grants)  |  |
       |  +-------------------------------------------------+  |
       |                          |                            |
       |  +-----------------------v-------------------------+  |
       |  | 6. Deterministic Contract Evaluation            |  |
       |  |    - All postconditions met?                    |  |
       |  |    - Effect already committed?                  |  |
       |  +-------------------------------------------------+  |
       |                          |                            |
       |  +-----------------------v-------------------------+  |
       |  | 7. Recovery Policy Engine                       |  |
       |  |    - VERIFIED_SUCCESS -> Restrain retry, finish |  |
       |  |    - NOT_COMMITTED    -> Safe retry             |  |
       |  |    - CORRUPTED/UNKNOWN-> Fail-closed escalation |  |
       |  +-------------------------------------------------+  |
       +-------------------------------------------------------+
```

---

## 2. Core Architectural Components

### A. Frozen Success Contracts (`reagent.models.StructuredDecision`)
Before executing any state-changing action, ReAgent freezes a cryptographically grounded `SuccessContract`:
- **Preconditions**: Expected initial balances, record existence, and account statuses.
- **Postconditions**: Exact required post-execution states (e.g., source balance decreased by exactly $500, target balance increased by $500, status equals `'CONFIRMED'`).
- **Invariants**: Idempotency keys, audit trail references, and conservation of value.
- **Immutability Guarantee**: Once frozen, the contract cannot be rewritten or diluted during recovery loops.

### B. Independent Read-Only Observer (`reagent.pipeline.make_dbops_probe_callback`)
When an action returns an ambiguous outcome (timeout, dropped connection, unconfirmed status):
- ReAgent dispatches a probe through an **independent read-only observer channel** completely decoupled from the primary tool session.
- The observer uses dedicated connection pooling with read-only transaction isolation (`PRAGMA query_only = ON` in SQLite).
- It cannot execute writes, rollbacks, or retries.

### C. Deterministic Contract Validator (`reagent.contract_validator`)
The validator evaluates observed facts against the frozen contract using formal relational operators:
- Checks column-level value equivalence (not merely row existence).
- Verifies mathematical deltas (`final_balance == initial_balance - transfer_amount`).
- Verifies multi-table consistency (e.g., ledger record present AND balance log row present).

### D. Recovery Policy Engine (`reagent.experiments.dbops.policies`)
Based on the contract verification results and tool capabilities, the engine selects one of four deterministic actions:
1. **`VERIFIED_SUCCESS`**: All postconditions are fully satisfied. The engine **restrains any retry attempt**, returning a verified success receipt.
2. **`SAFE_RETRY`**: The observer proves that the state was untouched and the action is either idempotent or uncommitted.
3. **`WAIT_AND_REPROBE`**: The operation is in-flight (e.g., async background job) and within bounded timeout limits.
4. **`SAFE_ESCALATION`**: State cannot be observed, status is unavailable, or partial/corrupted writes occurred. The engine **fails closed**, alerting human operators without corrupting storage.

### E. Token Budget Guard & Protocol Integrity (`reagent.policy.TokenBudgetGuard`)
- Enforces strict token and call budgets per run and across runs via `GlobalTokenLedger`.
- Prevents infinite loops or runaway model queries during ambiguous recovery loops.
- Emits transparent structured audit trails with zero sensitive credential exposure.
