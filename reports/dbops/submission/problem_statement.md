# Problem Statement: The "Done" Illusion in Autonomous Agents

## 1. The Real-World Ambiguity Problem

Autonomous AI agents are increasingly entrusted with state-changing tools:
- Database mutations (inserting ledger transactions, updating records).
- Financial operations (initiating refunds, transferring balances).
- Cloud and DevOps actions (deploying containers, migrating schemas).

In distributed systems and network environments, tools do not simply succeed or fail cleanly. Instead, agents face **operational ambiguity**:
1. **Network Drops / Timeouts (`COMMIT_THEN_TIMEOUT`)**: The database engine commits a record to durable storage, but the socket connection terminates before the HTTP response or tool result returns to the agent.
2. **Partial Commits (`PARTIAL_COMMIT`)**: A composite operation commits a primary record but fails before secondary balance reconciliation or audit logs complete.
3. **Corrupted Content (`WRONG_CONTENT`)**: The tool writes a record, but with wrong column values, mismatched recipient IDs, or corrupted currency values.
4. **Transient Outages (`STATUS_UNAVAILABLE`, `UNOBSERVABLE`)**: The status check API is temporarily unavailable or the external system lacks an observable query interface.

---

## 2. Why Conventional Agents Fail

Today's agent architectures handle these ambiguous outcomes using naive heuristics that produce silent catastrophes:

### Failure Mode A: The Naive Retry Hazard
When a tool returns a timeout or unacknowledged error, standard agents retry the tool execution.
- If the endpoint is **non-idempotent** (such as standard SQL `INSERT`, wallet transfers, inventory decrements), the retry executes the operation a second time.
- **Consequence**: Permanent financial loss (double charging a customer) or corrupted inventory states.

### Failure Mode B: The "Presence != Correctness" Trap
More sophisticated agents attempt basic observation: they query `SELECT COUNT(*) FROM ledger WHERE id = 'TXN-123'`.
- If the row exists, the agent concludes that the task succeeded.
- However, if the operation wrote `$50.00` instead of `$500.00`, transferred funds to the wrong recipient, or failed to update the secondary balance ledger, the presence check still reports `TRUE`.
- In our empirical benchmark across 96 trials, this shared-observation baseline falsely verified **44.8% of corrupted or partial outcomes**.

### Failure Mode C: Tool-Claim Blindness
Agents that evaluate success based solely on tool exit codes or response text are vulnerable to tool hallucinations and silent database rollbacks.
- In our benchmark, tool-claim-only agents falsely accepted corrupted outcomes in **46.9% of trials**.

---

## 3. The Database Operations (DBOPS) Evaluation Suite

To systematically quantify agent reliability, we preregistered the **DBOPS v1.3 Benchmark**:
- **12 Real-World Tasks** spanning four foundational protection classes:
  1. `UNPROTECTED_NUMERIC`: Non-idempotent balance debits and ledger inserts where retries create duplicate financial effects.
  2. `AUDIT_LOGGED`: Multi-table mutations requiring primary row commit plus matching immutable audit trail records.
  3. `IDEMPOTENT_KEYED`: Deduplicated transactions protected by unique constraints or idempotency tokens.
  4. `STATE_MACHINE`: Ordered state transitions (e.g., `PENDING` -> `COMMITTED` -> `RECONCILED`).
- **8 System Fault Modes**:
  `NO_FAULT`, `IMMEDIATE_FAILURE`, `COMMIT_THEN_TIMEOUT`, `PARTIAL_COMMIT`, `WRONG_CONTENT`, `AUDIT_ONLY`, `STATUS_UNAVAILABLE`, and `UNOBSERVABLE`.

Any production-grade AI agent layer must be capable of distinguishing verified success from partial commits, restraining duplicate writes, and failing closed when state is unknowable.

This is precisely the failure regime addressed by **ReAgent**: "ReAgent doesn't trust 'done.' It verifies it." By freezing declarative success contracts before execution, independently probing actual state after ambiguity, and enforcing formal contract postconditions, ReAgent eliminates false verification and restrains dangerous duplicate actions.
