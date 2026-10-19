# ReAgent: Two-Minute Judge Walkthrough Script

This document provides an exact, step-by-step walkthrough for evaluating ReAgent during judging.

---

## Step 1: Execute the Unified Judge Demo (Time: 10 seconds)

Open a Windows PowerShell terminal in the repository root and run:

```powershell
python -m reagent.experiments.dbops.judge_demo
```

### What Happens:
- The command executes immediately (sub-second runtime).
- It spins up fresh disposable SQLite databases in memory/temp storage.
- It makes **zero network requests** and requires **zero API keys**.

---

## Step 2: Review Part 1 — The Live Recovery Demonstration (Time: 40 seconds)

Observe the output in **`[PART 1: LIVE DETERMINISTIC RECOVERY DEMONSTRATION]`**:

```
SCENARIO: Task 02 - High-Value Wallet Transfer ($500.00 from ACCT-2201 to ACCT-3390)
INITIAL STATE: Starting account balance is $1,000.00. Exactly one $500 debit allowed.
INJECTED FAULT: COMMIT_THEN_TIMEOUT
```

### Side-by-Side Comparison:
- **Look at `Total Debited Amount`**:
  - `Naive Retry Baseline`: **`$1000.00`** (2 attempts, 2 ledger rows) -> **`YES (DOUBLE DEBIT)`**.
  - `ReAgent Layer`: **`$500.00`** (1 attempt, 1 probe, 1 ledger row) -> **`NO (PREVENTED)`**.
- **Look at `Final Account Balance`**:
  - `Naive Retry Baseline`: **`$0.00`** (Account drained due to blind retry).
  - `ReAgent Layer`: **`$500.00`** (Account balance preserved correctly).
- **Why this happens**:
  - When the tool timed out, Naive Retry assumed failure and tried again. Because the wallet transfer endpoint was non-idempotent, both operations committed.
  - ReAgent dispatched an independent read-only probe, verified that transaction `TXN-779043` had already been debited, and **restrained the retry**.

---

## Step 3: Review Part 2 — The Comparative Benchmark (Time: 40 seconds)

Look at the table in **`[PART 2: 480-TRIAL PAIRED COMPARATIVE EVALUATION RESULTS]`**:

| Policy Arm | False Ver. Rate | Safe Escalations | Unsafe Retries | Duplicate Writes |
| :--- | :---: | :---: | :---: | :---: |
| **`REAGENT_CONTRACT`** | **0.0%** | **67** | **0** | **0** |
| `SHARED_OBSERVATION` | **44.8%** | 24 | 0 | 0 |
| `NAIVE_RETRY` | **66.7%** | 0 | 24 | 4 |

### What to check:
1. **Notice the False Verification drop**: `SHARED_OBSERVATION` falsely claims success 44.8% of the time because it only tests if a database row exists. ReAgent drops this to **0.0%** because it verifies specific column values and balance deltas against the frozen contract.
2. **Notice the safety restraint**: Naive Retry executes 24 unsafe retries and creates 4 duplicate writes; ReAgent creates 0.

---

## Step 4: Verify Test Integrity & Artifact Hashes (Time: 30 seconds)

Run the focused test suite to verify test passing:

```powershell
python -m unittest tests/test_dbops_m6_submission.py
```

All tests pass deterministically. The hashes and provenance tags in `reports/dbops/submission/submission_manifest.json` cryptographically match all local artifacts.
