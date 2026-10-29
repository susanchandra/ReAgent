# ReAgent: Outcome Verification & Safe Recovery for AI Agent Actions

> **Headline:** *"A timeout isn't a no."*<br>
> **Supporting Line:** *"ReAgent checks what actually happened before an AI agent retries, so a lost response doesn't have to become a double charge."*<br>
> **Technical Principle:** *"Unknown is not absent."*<br>
> **Category:** Outcome verification and safe recovery for AI-agent actions<br>
> **Target Track:** Best Apps & Agents — NVIDIA × Nebius Global AI Hackathon 2026<br>
> **Target Architecture:** NVIDIA Nemotron (`nvidia/nemotron-3-super-120b-a12b`) on Nebius Token Factory<br>
> **Live Production Demo:** [https://reagent-app-nwif.onrender.com/](https://reagent-app-nwif.onrender.com/)<br>
> **Default Evaluation Mode:** SYNTHETIC / DETERMINISTIC OFFLINE FIXTURES (0 network calls required for offline judging)<br>
> **Live Inference Mode:** Opt-in real Nebius Token Factory inference (`?live=1`) with server-side rate limits & strict semantic contract validation<br>
> **License:** MIT License (Open Source)

---

## 1. The Real Problem & Intended Users

### Who ReAgent is For
ReAgent is designed for engineers deploying **autonomous AI agents in production environments** that perform real-world, state-changing actions—such as executing SQL database mutations, initiating financial ledger transfers, processing customer orders, or triggering external webhooks.

### The Real Problem: Post-Commit Ambiguity
When an AI agent executes a state-changing tool and the network connection drops or times out:
1. **The tool timed out before reaching the database** (action never committed).
2. **The database committed the transaction, but the network connection timed out before the tool could return a success response** (`COMMIT_THEN_TIMEOUT`).

To the agent, both cases look identical: an unhandled timeout error.

### Why Standard Solutions Fail Catastrophically
- **Naive Retry Loops**: The agent assumes timeout equals failure and blindly retries. On non-idempotent endpoints (such as high-value wallet debits), this causes **irreversible double charges, duplicate records, and financial corruption**.
- **Shallow Presence Observation (`COUNT > 0`)**: Some frameworks check if a record exists. In our 480-trial benchmark, shallow presence checking suffered a **44.8% false verification rate** because it mistakenly verifies partial commits, corrupted columns, or wrong debit amounts.
- **Self-Grading (Evaluator Bias)**: Asking an LLM to evaluate its own tool calls creates systematic confirmation bias where broken states are hallucinated as successful.

---

## 2. Core ReAgent Architecture & Guarantees

```
                   USER TASK OBJECTIVE
                            │
                            ▼
      NEBIUS TOKEN FACTORY (nvidia/nemotron-3-super-120b-a12b)
       Derives structured precondition & postcondition contract
                            │
                            ▼
          OFFLINE CANDIDATE VALIDATION GATE
     Strict Pydantic schema allowlist & task grounding check
                            │
                            ▼
                 FREEZE SUCCESS CONTRACT
          (Immutable: cannot be relaxed by LLM)
                            │
                            ▼
             TOKEN BUDGET GUARD & POLICY CHECK
            (Pre-call reservation against ledger)
                            │
                            ▼
                SANDBOXED TOOL EXECUTION
          (e.g., Database write or ledger transfer)
                            │
                            ▼
          INDEPENDENT READ-ONLY PROBE QUERY
     (query_only=ON: Fresh isolated connection; zero mutations)
                            │
                            ▼
                 DETERMINISTIC VERIFIER
        (Evaluates exact column invariants & deltas)
              /                            \
             /                              \
      [Postconditions Met]             [Ambiguous / Corrupted]
            │                                     │
            ▼                                     ▼
     VERIFIED RECEIPT                     BOUNDED RECOVERY LOOP
   Retry restrained!                     1. Discrepancy diagnosis
   Duplicate mutation prevented          2. Idempotency safety check
                                         3. Safe retry iff CONFIRMED ABSENT
                                         4. FAIL-CLOSED ESCALATION if UNKNOWN
```

### Architectural Guarantees
1. **Model Proposes; Deterministic Verifier Decides**: NVIDIA Nemotron derives the declarative contract. An independent, deterministic rule engine evaluates actual filesystem and database state. The model never grades its own work.
2. **Frozen Success Contracts**: Preconditions and postconditions are permanently frozen before tool execution. Recovery logic cannot loosen criteria to force a false pass.
3. **Mandatory Read-Only Probing**: Independent observation uses fresh connections with `query_only=ON` write-blocking enforced. Probing can never mutate application state.
4. **Retry Safety Gate**: `UNKNOWN` is never treated as `ABSENT`. If probe status is `DISCREPANCY`, `UNKNOWN`, `UNAVAILABLE`, or `UNOBSERVABLE`, retries are strictly blocked and ReAgent fails closed to `ESCALATED`.
5. **Cross-Process Token Ledger**: A file-locked `GlobalTokenLedger` tracks cumulative token and call usage across process restarts with conservative pre-call reservations.

---

## 3. The Role of NVIDIA Nemotron & Nebius Token Factory

### What NVIDIA Nemotron Does
ReAgent leverages **`nvidia/nemotron-3-super-120b-a12b`** hosted on **Nebius Token Factory**. Its role is strictly defined:
- **Contract Derivation**: Translates user task specifications and database schemas into structured JSON contracts specifying exact primary table predicates, expected column values, and multi-table audit assertions.
- **Candidate Grounding & Semantic Validation**: Candidate contracts are passed through an offline candidate validation gate (`validate_contract_proposal`) that strictly enforces:
  1. Valid JSON adhering to contract schema.
  2. Supported tool allowlist per workflow (`hotel_booking_http`, `dbops_read_only`, `dbops_database`, `wallet_transfer`).
  3. Context binding: Task identifiers (`booking_id`, `room_number`, `txn_id`) must match trusted context; model output cannot redirect actions or invent arbitrary targets.
  4. Predicate requirements: Required target/expected values per predicate; generic `STATUS_SUCCESS` ("Action must succeed") is rejected wherever concrete state must be observed.
  5. Observer capability: Postcondition targets must map to observable state properties.
  6. Provenance protection: Model output is advisory; provenance cannot be forged as executed or verified.
- **Fail-Closed Isolation**: If the model produces unparseable JSON, generic ungrounded assertions, or violates schemas, ReAgent rejects the proposal and fails closed (`SCHEMA_INVALID` or `GROUNDING_FAILED`)—it never falls back to an unverified action.

### How Nebius Token Factory is Used
- Nebius Token Factory serves as the high-performance inference backend, accessed via standard OpenAI-compatible endpoints with pre-flight parameter guards (`temperature=0.0`, `max_tokens=512`).
- **Two Distinct Modes**:
  - **Live Inference**: In the interactive Mission Control / Task Workbench, users can opt-in to live Nebius inference via the UI toggle or `?live=1` query parameter. Calls are rate-limited, tracked in a durable token ledger, guarded against unbounded inputs, and server-side authenticated (API keys are never exposed).
  - **Offline Deterministic Fixtures (Default)**: To protect developer credits and ensure 100% reproducible benchmark evaluation, all automated test suites and offline judging flows use deterministic fixtures with explicit `OFFLINE` provenance labeling. Live model failures never silently fall back to simulated fixtures under a live label.

---

## 4. Quickstart: 2-Minute Judge Walkthrough

ReAgent is built to run and verify completely offline right out of the box with zero network access, zero billing, and no API keys required.

### Prerequisites
- Python 3.10+ (Windows / Linux / macOS)
- Standard library or optional virtual environment:
```powershell
python -m venv reagent_env
.\reagent_env\Scripts\activate
pip install -r requirements.txt
```

### 1. Launch Interactive Mission Control (One-Command Quickstart)
```powershell
# Windows PowerShell:
.\launch_demo.ps1

# Or cross-platform Python:
python launch_demo.py
```
Open your browser at **`http://127.0.0.1:8080/`**.
- To automatically launch and open your default browser: `python launch_demo.py --open`
- To run the fast 4-second terminal judge demo: `python launch_demo.py --cli`
- To run release-gate verification checks: `python launch_demo.py --check`

### 2. Alternative Direct Commands
```powershell
# Direct UI server launch:
python reagent/ui_server.py --port 8080

# Direct CLI judge demo:
python -m reagent.experiments.dbops.judge_demo
```

#### Five Recovery Scenarios Demonstrated:
- **Scenario A (Normal Success)**: Clean execution. Both agents execute once; ReAgent independently observes and verifies postconditions.
- **Scenario B (Commit-Then-Timeout)**: Transaction commits on disk, but network response is lost. Naive retry causes duplicate mutations; ReAgent probes, observes committed state, and restrains retry.
- **Scenario C (Fail-Before-Commit)**: Network drop occurs before commit; database is untouched. ReAgent proves absence via independent probe, retries safely exactly once, and verifies.
- **Scenario D (Unknown Outcome)**: Commit occurs but read/probe path is unreachable. ReAgent enforces *"Unknown is not absent"*, refusing to guess or retry blindly, and escalates. An operator **Judge Oracle Reveal** button discloses the hidden ground truth.
- **Scenario E (Partial Commit)**: Multi-step transaction partially writes (ledger row commits, but balance update fails). ReAgent detects the discrepancy against frozen postconditions and halts/escalates to an operator rather than repeating the corrupt action.

#### Three Isolated Task Template Demos:
- **💳 Wallet Transfer** (`UNPROTECTED_NUMERIC`): $500 transfer (ACCT-2201 → ACCT-3390). Unprotected balance decrement exposes double debits under naive retry.
- **🏨 Hotel / Seat Booking** (`KEY_PROTECTED`): Seat 14B reservation on Flight FL-2047. `UNIQUE(flight_id, seat_num)` constraint blocks physical duplication, but naive retry crashes and misclassifies state.
- **🗄 Database Update / Refund** (`KEY_PROTECTED` on primary, unconstrained on audit): Refund REF-550291. Primary key protects refund table while secondary audit log duplicates under blind retry.

To export a standalone portable HTML report without running a server:
```powershell
python reagent/ui_server.py --export-html reports/dbops/reagent_judge_dashboard.html
```

### 3. Validate Cryptographic Submission Manifest (12/12 Verified)
```powershell
python -m reagent.experiments.dbops.validate_submission_manifest
```
Recomputes SHA-256 hashes from actual disk bytes and validates all 12 indexed submission artifacts.

---

## 5. DBOPS Comparative Benchmark: 480-Trial Empirical Evidence

Evaluated across **12 tasks × 8 fault modes × 5 policy arms = 480 paired trials** on fresh disposable SQLite databases:

| Policy Arm | Total Trials | True Verified | False Verified | False Ver. Rate | Safe Escalations | Unnec. Escalations | Unsafe Retries | Duplicate Writes (Primary / Audit) | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`REAGENT_CONTRACT`** | **96** | **29** | **0** | **0.0%** | **67** | **0** | **0** | **0 / 0** | 13.98 ms |
| `SHARED_OBSERVATION` | **96** | 29 | 43 | **44.8%** | 24 | 0 | 0 | 0 / 3 | 14.10 ms |
| `TOOL_CLAIM_ONLY` | **96** | 15 | 45 | **46.9%** | 29 | 7 | 0 | 0 / 0 | 8.45 ms |
| `NAIVE_RETRY` | **96** | 32 | 64 | **66.7%** | 0 | 0 | **24** | **4 / 5** | 15.65 ms |
| `ALWAYS_ESCALATE` | **96** | 0 | 0 | **0.0%** | 74 | 22 | 0 | 0 / 0 | 7.92 ms |

### Exact Mathematical Accounting Invariant
Every trial is mapped to exactly one mutually exclusive outcome:
$$\text{true\_verified} + \text{false\_verified} + \text{safe\_escalations} + \text{unnecessary\_escalations} = \text{total\_trials} = 96$$

### Duplicate Write Accounting Reconciliation (from 480 Raw Trial Records)
- **24 Unsafe Retries**: Trials where the Naive Agent executed a blind retry after an ambiguous error (`COMMIT_THEN_TIMEOUT`, `PARTIAL_COMMIT`, `STATUS_UNAVAILABLE`, `UNOBSERVABLE`).
- **4 Duplicate Primary Records**: Duplicate financial balance debits (`duplicate_primary_count: 4`) in unprotected numeric tables.
- **5 Duplicate Audit Log Rows**: Secondary audit log duplications on unconstrained append-only tables (`duplicate_audit_count: 5`).
- **Constraint Prevention**: In the remaining 15 unsafe retries, table-level uniqueness constraints caught the collision and prevented physical table duplication, but the agent still claimed false verification (64 false verifications total).
- **ReAgent Guarantee**: ReAgent had **0 unsafe retries, 0 duplicate primary writes, and 0 duplicate audit writes** across all 96 trials.

---

## 6. Running the Automated Test Suite

ReAgent includes comprehensive automated test suites covering recovery invariants, adversarial faults, schema validation, accounting integrity, mock provider integration, clean-start smoke tests, UI server endpoints, strict semantic contract verification, and final readiness gate constraints:

```powershell
# Run full comprehensive test suite across tests/ and deploy/hf_space/tests/ (874 passed, 13 skipped, 20 subtests passed):
python -m pytest tests/ deploy/hf_space/tests/ -q

# Run focused semantic contract and observer verification test suite:
python -m pytest tests/test_semantic_contract_verification.py -v

# Run focused UI and integration test suite:
python -m pytest tests/test_ui_server.py -v
```

**Results:**
- **874 Tests Passing** (13 skipped, 20 subtests passed)
- **0 Failures / 0 Errors**
- **Network Calls:** **ZERO** (Strictly offline by default)

---

## 7. Supervised Live Pilot Protocol (Optional / Human Authorization)

To connect to live Nebius Token Factory endpoints, ReAgent enforces a strict human-in-the-loop safety protocol:
1. **Mandatory Explicit Opt-In**: Requires the `--live-run-opt-in` CLI flag. Live execution is **never inferred** from environment variables alone.
2. **Preflight Verification**: An offline dry-run preflight check is always executed first.
3. **Hard Ceiling**: Hard global cap of 6 dispatch attempts enforced by a durable ledger (`reports/dbops/pilot_ledger.jsonl`).
4. **Secret Protection**: Keys are never printed, logged, or included in reports.

```powershell
# 1. Offline preflight dry-run (0 network calls, 0 cost):
python run_live_pilot.py

# 2. Live execution (requires explicit human authorization and valid API key):
$env:NEBIUS_API_KEY = "your-key-here"
python run_live_pilot.py --live-run-opt-in
```

---

## 8. Limitations & Honest Engineering Disclosure

- **Synthetic vs. Live Evidence**: All benchmark metrics (480 trials) and default demonstration replays are derived from deterministic SQLite fixtures with simulated fault injection. This rigorously isolates and validates the algorithmic recovery semantics and retry restraint across all 8 fault modes.
- **Model-Derived Contract Arm**: Real live inference using `nvidia/nemotron-3-super-120b-a12b` via Nebius Token Factory is fully operational on user opt-in (`?live=1` or the UI toggle). It requires an explicit user action and is safeguarded by token rate limits and candidate semantic validation. Offline reference proposals remain the default for repeatable zero-cost evaluation.
- **Advisory Model vs. Authoritative Observer**: Nemotron is strictly advisory—it derives proposed contracts, but an independent read-only observer probe querying actual SQLite/API state retains sole authority over whether an action succeeded.
- **Safety Trade-off**: ReAgent intentionally prioritizes safety over blind completion. Under ambiguous unobservable states (`UNKNOWN`), it escalates to an operator rather than retrying blindly (*"Unknown is not absent"*), resulting in a higher escalation rate under severe faults but **100% protection against duplicate mutations and financial data corruption**.

---

## 9. Submission Artifacts Directory

| Document / Artifact | Description |
| :--- | :--- |
| [`reports/dbops/submission/README.md`](reports/dbops/submission/README.md) | Submission guide and judge quickstart |
| [`reports/dbops/submission/problem_statement.md`](reports/dbops/submission/problem_statement.md) | Technical problem statement and failure analysis |
| [`reports/dbops/submission/architecture.md`](reports/dbops/submission/architecture.md) | Deep architecture: contracts, observer, and policy engine |
| [`reports/dbops/submission/evaluation.md`](reports/dbops/submission/evaluation.md) | 480-trial empirical benchmark results |
| [`reports/dbops/submission/demo_script.md`](reports/dbops/submission/demo_script.md) | 2-minute walkthrough guide for evaluators |
| [`reports/dbops/submission/limitations.md`](reports/dbops/submission/limitations.md) | Operational boundaries and synthetic vs live status |
| [`reports/dbops/submission/video_script.md`](reports/dbops/submission/video_script.md) | YouTube video script and shot plan (<3 min) |
| [`reports/dbops/submission/devpost_submission.md`](reports/dbops/submission/devpost_submission.md) | Complete Devpost submission draft and sponsor feedback |
| [`reports/dbops/submission/submission_manifest.json`](reports/dbops/submission/submission_manifest.json) | Cryptographic SHA-256 index of all submission files |

---

## 10. License

MIT License — see [LICENSE](LICENSE) for full details.
