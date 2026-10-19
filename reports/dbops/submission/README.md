# ReAgent: AI Agent Reliability & Verification Layer

> *"ReAgent doesn't trust 'done.' It verifies it."*

**Hackathon Track**: Personal AI / AI Agent Reliability  
**Target Architecture**: NVIDIA Nemotron / Local & Cloud AI Infrastructure  
**Core Innovation**: Contract-Driven State Verification & Non-Idempotent Action Restraint  
**Source Revision**: `e1292b57abcccda57c0070fdbbe6a80bae1b8834`  
**Provenance**: `SYNTHETIC / DETERMINISTIC OFFLINE BENCHMARK` (Zero live model calls required for evaluation)

---

## Executive Summary

When AI agents execute state-changing real-world actions (e.g., database updates, financial transfers, webhook triggers), tool executions frequently encounter **ambiguous outcomes**:
- The tool drops the connection before returning success (`COMMIT_THEN_TIMEOUT`).
- The database writes partial data or incorrect content (`WRONG_CONTENT`, `PARTIAL_COMMIT`).
- The tool reports an unhandled exception or status unavailable (`STATUS_UNAVAILABLE`).

Standard AI agents handle these situations catastrophically:
1. **Naive Retry** blindly re-invokes the action, causing **double charges** or duplicate records on non-idempotent endpoints.
2. **Shared Observation** checks whether any row exists (`COUNT > 0`), falsely claiming success **44.8% of the time** on corrupted or partial writes.
3. **Tool-Claim-Only** agents trust unverified tool return codes, falsely claiming success **46.9% of the time**.

**ReAgent** introduces a reliability layer for autonomous agents:
- **Freezes Success Contracts**: Declarative preconditions and postconditions established *prior* to action execution.
- **Independent Read-Only Probing**: Inspects system state via an isolated, read-only observer path.
- **Restrains Unsafe Retries**: Proves whether the side effect already committed, preventing duplicate execution.
- **Fails Closed**: When states are unobservable or ambiguous, ReAgent safely escalates to human operators with zero hallucinations.

---

## Two-Minute Judge Quickstart

Run the deterministic judge demo directly in Windows PowerShell:

```powershell
python -m reagent.experiments.dbops.judge_demo
```

### What You Will See:
1. **Part 1 — Live Recovery Demonstration**: Executes a $500 wallet transfer under an injected `COMMIT_THEN_TIMEOUT` fault on disposable SQLite databases.
   - **Naive Retry** causes an irreversible **double debit ($1,000.00)**.
   - **ReAgent** independently probes the ledger, verifies the $500 debit against the frozen contract, restrains retry, and preserves the single $500 debit.
2. **Part 2 — 480-Trial Comparative Benchmark**: Presents paired evaluation results across 12 tasks, 8 fault modes, and 5 policy arms.
   - **0.0% False Verification** for ReAgent vs. **44.8%** for Shared Observation.
   - **0 Unsafe Retries** for ReAgent vs. **24 Unsafe Retries (4 Permanent Duplicates)** for Naive Retry.
3. **Zero Network Requirement**: Runs completely offline in approximately 4 seconds in the tested environment with zero API keys or network dependencies.

---

## Submission Documentation Structure

| Document | Description |
| :--- | :--- |
| [`problem_statement.md`](problem_statement.md) | The failure mode: Ambiguity under state-changing operations and why simple presence checks fail. |
| [`architecture.md`](architecture.md) | Technical architecture: Frozen contracts, read-only observer, deterministic policy engine, and token budget guard. |
| [`evaluation.md`](evaluation.md) | 480-trial empirical benchmark comparing ReAgent against 4 baselines across 8 fault modes. |
| [`demo_script.md`](demo_script.md) | Step-by-step 2-minute walkthrough guide for evaluators and judges. |
| [`limitations.md`](limitations.md) | Explicit statement of operational boundaries, unobservable states, and offline fixture vs. live model status. |
| [`submission_manifest.json`](submission_manifest.json) | Cryptographic SHA-256 manifest of all submitted artifacts, source revisions, and timestamps. |

---

## Repository & Test Commands

To verify test coverage and benchmark reproducibility locally:

```powershell
# 1. Run the unified judge demo (<4s)
python -m reagent.experiments.dbops.judge_demo

# 2. Launch the interactive web application
python reagent/ui_server.py --port 8080

# 3. Validate cryptographic submission manifest (12/12 verified)
python -m reagent.experiments.dbops.validate_submission_manifest

# 4. Run the focused submission tests
python -m unittest tests/test_dbops_m6_submission.py

# 3. Optional: Regenerate the full 480-trial benchmark from scratch (~30s)
python -m reagent.experiments.dbops.judge_demo --regenerate-benchmark
```
