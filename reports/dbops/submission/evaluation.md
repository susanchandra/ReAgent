# DBOPS Comparative Evaluation: 480-Trial Empirical Benchmark

## 1. Experimental Methodology

To rigorously prove the value of contract-driven state verification, we evaluated ReAgent against four baseline policies across the entire **DBOPS v1.3 benchmark suite**:
- **Paired Design**: All policies evaluated under identical initial database states, schemas, parameters, and injected fault modes.
- **Trial Scale**: 12 tasks × 8 fault modes × 5 policy arms = **480 paired configurations**.
- **Isolation**: Fresh, disposable SQLite databases created and cleaned per trial. Zero cross-trial leakage.

### Compared Policy Arms
1. **`REAGENT_CONTRACT`**: Frozen contract verification via independent read-only observer. Restrains retries when postconditions are met; fails closed when corrupted or unobservable.
2. **`SHARED_OBSERVATION`**: Uses the same independent probe connection, but relies on shallow presence checks (`COUNT(*) > 0`), ignoring column-level value correctness.
3. **`TOOL_CLAIM_ONLY`**: Evaluates success purely based on tool exit status and error codes without independent probing.
4. **`NAIVE_RETRY`**: Blindly retries any action that returns a timeout or unacknowledged error code.
5. **`ALWAYS_ESCALATE`**: Fails safe by escalating all ambiguous or error outcomes directly to human operators.

---

## 2. Aggregate Benchmark Results (N = 480 Paired Trials)

| Policy Arm | Total Trials | True Verified | False Verified | **False Verification Rate** | Safe Escalations | Unnec. Escalations | Unsafe Retries | **Duplicate Primary Writes** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **REAGENT_CONTRACT** | **96** | **29** | **0** | **0.0%** | **67** | **0** | **0** | **0** |
| SHARED_OBSERVATION | 96 | 29 | 43 | **44.8%** | 24 | 0 | 0 | 0 |
| TOOL_CLAIM_ONLY | 96 | 15 | 45 | **46.9%** | 29 | 7 | 0 | 0 |
| NAIVE_RETRY | 96 | 32 | 64 | **66.7%** | 0 | 0 | 24 | **4** |
| ALWAYS_ESCALATE | 96 | 0 | 0 | **0.0%** | 74 | 22 | 0 | 0 |

> **Accounting Invariant & Denominator Integrity**:
> Every raw trial is mapped into exactly one mutually exclusive outcome:
> \\text{true\\_verified} + \\text{false\\_verified} + \\text{safe\\_escalations} + \\text{unnecessary\\_escalations} = \\text{total\\_trials} = 96
> In previous reporting tables, Unnecessary Escalations was omitted from display, causing TOOL_CLAIM_ONLY (+45+29=89$) and ALWAYS_ESCALATE (+0+74=74$) to appear incomplete. Including Unnecessary Escalations ($ and $, respectively) fully reconciles all 96 trials per arm.

---

## 3. Key Empirical Findings

### Finding 1: Elimination of False Verification Claims (0.0% vs. 44.8%)
In systems dealing with partial writes (`PARTIAL_COMMIT`) and corrupted values (`WRONG_CONTENT`), shallow presence checks fail catastrophically:
- `SHARED_OBSERVATION` verifies that a row was inserted, falsely claiming success in **44.8% of runs**.
- `REAGENT_CONTRACT` evaluates column values, numeric balance deltas, and multi-table audit logs against the frozen contract, achieving **0.0% false verifications** across all 96 trials.

### Finding 2: Prevention of Financial & Data Corruption
Under `COMMIT_THEN_TIMEOUT` faults on non-idempotent tasks (e.g., Task 02 $500 wallet transfer):
- `NAIVE_RETRY` executes 24 unsafe retries across the benchmark, generating **4 permanent duplicate primary writes** (double debits).
- `REAGENT_CONTRACT` identifies that the transfer already committed to disk, restrains the retry, and records **0 duplicate writes** and **0 unsafe retries**.

### Finding 3: Guaranteed Fail-Closed Safety Under Unobservable States
When external services suffer outages (`STATUS_UNAVAILABLE`) or lack query interfaces (`UNOBSERVABLE`):
- `REAGENT_CONTRACT` never hallucinates task success; it safely escalates to human operators **100% of the time**.
- Unlike naive agents that retry blindly or tool-claim agents that misclassify errors, ReAgent guarantees operational containment.

---

## 4. Breakdown by Protection Class

| Protection Class | Tasks Included | ReAgent False Ver. | Shared Obs False Ver. | Naive Retry Duplicates |
| :--- | :--- | :---: | :---: | :---: |
| `UNPROTECTED_NUMERIC` | Tasks 01, 02, 05 | **0.0%** | 50.0% | **4 (Double Debits)** |
| `AUDIT_LOGGED` | Tasks 03, 04, 06 | **0.0%** | 41.7% | 0 (Blocked by retry) |
| `IDEMPOTENT_KEYED` | Tasks 07, 08, 09 | **0.0%** | 41.7% | 0 (Deduplicated) |
| `STATE_MACHINE` | Tasks 10, 11, 12 | **0.0%** | 45.8% | 0 (State reject) |

*Full machine-readable data stored in `reports/dbops/comparative_evaluation_results.json`.*
