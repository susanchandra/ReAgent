# ReAgent: Deterministic AI Agent Reliability Layer

> **Tagline:** *"ReAgent doesn't trust 'done.' It verifies it."*
> **Pitch:** ReAgent makes AI agents self-recovering: it diagnoses failures, separates agent errors from environment failures, and autonomously recovers and verifies observable outcomes.
> **Target Track:** Best Apps & Agents — NVIDIA x Nebius Global AI Hackathon 2026


---

## 1. The Problem: Why Agents Fail Silently

Contemporary LLM agents frequently declare tasks completed when they are broken:
- **False-Success Illusion:** An agent runs a tool, receives an exit code 0 or empty output, and hallucinates that the user's objective was achieved.
- **Evaluator Bias (Self-Grading):** When an LLM is asked to grade its own output, it suffers from systematic confirmation bias and repeatedly validates broken artifacts.
- **Unbounded Retry Loops:** When an API fails, naive agents retry in tight loops, exhausting rate limits and racking up billable token costs without diagnosing root causes.
- **Silent Degradation:** When an agent cannot fix an error, it often outputs a plausible explanation and marks the ticket "resolved" rather than escalating cleanly.

ReAgent solves this by enforcing an **independent, deterministic verification and recovery barrier** between the LLM and the real world.

---

## 2. Core Architecture & Guarantees

```
                   USER TASK OBJECTIVE
                            |
                            v
       PROVIDER DECISION ENGINE (Nebius Nemotron)
                            |
                            v
           STRICT SCHEMA & ALLOWLIST VALIDATION
                            |
                            v
              CONTRACT GROUNDING VALIDATOR
     (Rejects vacuous, mismatched, or invalid contracts)
                            |
                            v
                  FREEZE SUCCESS CONTRACT
            (Immutable: cannot be relaxed by LLM)
                            |
                            v
           TOKEN BUDGET GUARD & POLICY CHECK
            (Pre-call reservation against ledger)
                            |
                            v
                 SANDBOXED TOOL EXECUTION
                            |
                            v
             INDEPENDENT EVIDENCE COLLECTION
                            |
                            v
                 DETERMINISTIC VERIFIER
        (LLM has ZERO influence over pass/fail)
              /                                \
             /                                  \
      [All Passed]                       [Discrepancy Found]
           |                                      |
           v                                      v
    VERIFIED RECEIPT                     BOUNDED RECOVERY LOOP
                                         1. Failure Attribution
                                         2. Policy-Guarded Strategy
                                         3. Reverification
                                         4. ESCALATE if exhausted
```

### Architectural Guarantees

1. **Provider Proposes; Deterministic Verifier Decides:** The LLM proposes actions and verifiable postconditions. An offline, deterministic rule engine evaluates actual filesystem and environment state. The model never grades its own homework.
2. **Frozen Success Contracts:** Success criteria are captured in a `SuccessContract` and permanently frozen before tool execution. Recovery logic cannot loosen or alter postconditions to achieve a false pass.
3. **Pre-Invocation Grounding Checks:** Contracts are validated before tool execution. If target paths mismatch tool arguments, postconditions are vacuous (target already exists), or predicates lack grounding in the objective, execution is rejected immediately (call count remains 0).
4. **Sandboxed File Boundaries:** All file predicates resolve inside a strictly guarded `workspace_root`. Directory traversal (`..`), drive letter escapes (`C:`, `D:`), and symlinks escaping the boundary are blocked with `PermissionError`. File reads are capped at 1 MB.
5. **Cross-Process Token Ledger:** A persistent, file-locked `GlobalTokenLedger` tracks cumulative token and call usage across process restarts. Unknown token usage charges the conservative reservation. Corrupt ledgers fail closed (`CorruptLedgerError`).
6. **Immutable Audit Receipts:** Every execution path (success, truncation, provider failure, escalation) persists a strictly allowlisted, sanitized audit receipt without leaking API keys, credentials, or internal thoughts.

---

## 3. Quick Start (100% Offline — No API Key Required)

ReAgent is built to run and verify completely offline right out of the box using its built-in `DevelopmentLLMProvider`.

### Prerequisites
- Python 3.10+ (Windows / Linux / macOS)
- Virtual environment with requirements installed:
```powershell
python -m venv reagent_env
.\reagent_env\Scripts\activate
pip install -r requirements.txt
```

### Run the Offline Mechanics Demo
The offline demo exercises all 5 core reliability mechanics scenarios using the local `DevelopmentLLMProvider` with zero network access, zero billing, and no API keys:
```powershell
python scripts/demo_offline.py
```

> **Note on Demo Scope:** This script is an **offline mechanics demonstration** of ReAgent's contract-freezing, deterministic verification, and bounded recovery loop logic using rule-based mocks. It demonstrates the reliability framework's internal mechanics, not live model inference. Live model integration is handled separately via `NebiusNemotronProvider` under strict human supervision.

Expected output:
```
================================================================
  ReAgent Offline Demo: 5/5 scenarios PASSED
================================================================
  [PASS]  All 5 scenarios passed. No API key required.
```

### Run the Benchmark Comparison (ReAgent vs Illustrative Naive Baseline)
Execute the benchmark evaluating ReAgent against an illustrative naive fixed-retry loop across 5 seeded deterministic scenarios (n=5):
```powershell
python scripts/run_benchmark.py
```
Output full JSON traces:
```powershell
python scripts/run_benchmark.py --json
```

### Launch the Judge-Facing Reliability Dashboard
Launch the local web dashboard to visually inspect timelines, failure attribution, and head-to-head comparison:
```powershell
python reagent/ui_server.py --port 8080
```
Then navigate to `http://127.0.0.1:8080/`.

To export a standalone, portable HTML report for submission:
```powershell
python reagent/ui_server.py --export-html reports/benchmark_dashboard.html
```

### Run the Milestone E2E Runner
```powershell
python -m reagent.e2e
```

---

## 4. Demonstrable Scenarios

ReAgent includes 6 formal failure scenarios in `reagent/scenarios/definitions.py` and benchmark tests in `tests/test_scenarios.py`:

| Scenario ID | Name | Trigger Condition | ReAgent Diagnosis | Recovery Strategy | Verification Outcome |
|---|---|---|---|---|---|
| **Scenario A** | Malformed Input | Tool called with missing required parameters | `INPUT_CONTEXT` | `RETRY_MODIFIED_INPUT` | Parameter corrected, reverified |
| **Scenario B** | Transient Failure | 502 Bad Gateway / upstream network error | `TRANSIENT_EXTERNAL` | `WAIT_AND_RETRY` | Backoff wait, retry verified |
| **Scenario C** | Unavailable Dependency | Model asks for un-allowlisted tool | `ENVIRONMENT_FAILURE` | `FALLBACK_TOOL` | Fallback allowlisted tool selected and verified |
| **Scenario D** | Model Output Error | Model returns malformed / unparseable JSON | `MODEL_OUTPUT` | `RETRY_MODIFIED_INPUT` | Schema re-enforced, retry verified |
| **Scenario E** | Repeated Auth Failure | 403 Forbidden exhausts retry budget | `AUTHORIZATION` | `ESCALATE` | PolicyGuard blocks infinite loop, cleanly escalates |
| **Scenario F** | False Success Detection | Tool returns status 0 but expected output file missing | `TOOL_FAILURE` | `RETRY_MODIFIED_INPUT` | Verifier catches discrepancy, triggers recovery |

---

## 5. Offline Test Suite

ReAgent features a comprehensive offline test suite covering contract validation, provider isolation, token budget accounting, and deterministic verification:

```powershell
# Ensure live tests remain disabled
Remove-Item env:REAGENT_ALLOW_LIVE_TESTS -ErrorAction SilentlyContinue

# Run test discovery
python -m unittest discover -s tests -p "test_*.py" -v
```

**Results:**
- **132 Tests Passing**
- **10 Skipped** (live Nebius integration tests — require explicit opt-in)
- **0 Failures / 0 Errors**
- **Total Execution Time:** ~1.6 seconds

---

## 6. Configuration & Environment Variables

| Variable | Purpose | Default | Permitted Bounds |
|---|---|---|---|
| `NEBIUS_API_KEY` | Canonical API key for Nebius Token Factory | None | Non-empty string |
| `NEBIUS_NEMOTRON_MODEL` | Model ID on Nebius Token Factory | `nvidia/nemotron-3-super-120b-a12b` | Any valid model ID |
| `NEBIUS_BASE_URL` | Nebius Token Factory endpoint | `https://api.tokenfactory.nebius.com/v1/` | Valid URL |
| `REAGENT_MAX_TOKENS` | Completion token limit per provider call | `1536` | Code-enforced: [16, 4096] |
| `REAGENT_LEDGER_MAX_CALLS` | Persistent cumulative call limit | `20` | Code-enforced: <= 50 |
| `REAGENT_LEDGER_MAX_TOKENS`| Persistent cumulative token limit | `60000` | Code-enforced: <= 200,000 |
| `REAGENT_LEDGER_PATH` | Path to persistent ledger JSON file | `~/.reagent/token_ledger.json` | Valid filepath |
| `REAGENT_ALLOW_LIVE_TESTS` | Explicit opt-in gate for integration tests | Unset | Must be `1` to run live tests |

> **Security Note:** Never store API keys in source control. If you have a key, pass it via environment variable in your session.

---

## 7. Supervised Live Probe Protocol (Human-in-the-Loop)

ReAgent includes a minimal, supervised live probe script (`scripts/live_probe.py`) designed for careful verification of remote Nebius Token Factory connectivity.

### Strict Safety Guards Enforced:
1. **Mandatory Confirmation Flag:** Must be explicitly invoked with `--confirm-billing-checked`.
2. **Single Call Cap:** Makes exactly ONE live call for a read-only task (`json_validator`).
3. **Dual Opt-In Required:** Integration tests require BOTH `NEBIUS_API_KEY` and `REAGENT_ALLOW_LIVE_TESTS=1`. Setting the key alone will never trigger live tests.
4. **Pre-Call Reservation:** Full request prompt tokens plus configured completion tokens are reserved before making the network request.
5. **Fail-Closed Reconciliation:** If ledger reconciliation encounters an error, the run fails closed with `PROBE_ACCOUNTING_UNCERTAIN` and exit code 1.

```powershell
# Example supervised invocation (requires human authorization):
python scripts/live_probe.py --confirm-billing-checked
```

---

## 8. Limitations & Engineering Tradeoffs

- **Token Estimation Heuristic:** ReAgent uses a character-to-token ratio heuristic (~1 token per 3 characters) with message framing padding and floors to remain 100% dependency-free offline. In code-dense or non-English prompts, conservative over-reservation occurs and is reconciled post-call.
- **Local Ledger Scope:** The `GlobalTokenLedger` relies on OS file locking (`filelock`). It enforces local machine boundaries across processes, but is not a distributed cluster ledger.
- **No Provider-Side Billing Guarantee:** Local token counters prevent local runaway loops; they do not control provider-side subscription tiers or minimum billing increments.

---

## 9. License

MIT License — see [LICENSE](LICENSE) for details.
