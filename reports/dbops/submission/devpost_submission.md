# ReAgent: Devpost Submission Draft

> **Submission Status**: **DRAFT PREPARED FOR SUBMISSION**  
> *(Note: This document is a generated submission draft for the project owner to review and paste into Devpost. It does NOT claim that the Devpost form has already been submitted online).*

---

## 1. Basic Project Information

- **Project Title**: ReAgent: Outcome Verification & Safe Recovery for AI Agent Actions
- **Elevator Pitch / Tagline**: *"A timeout isn't a no."*
  - *Supporting Line*: ReAgent checks what actually happened before an AI agent retries, so a lost response doesn't have to become a double charge.
  - *Technical Principle*: *"Unknown is not absent."*
  - *Full Pitch*: An independent outcome-verification and safe-recovery layer for autonomous AI agents that eliminates false completion claims, restrains duplicate mutations on timeout ambiguity, and enforces fail-closed recovery.
- **Target Track**: Best Apps and Agents
- **Target Competition**: NVIDIA × Nebius Global AI Hackathon 2026
- **License**: MIT License (Open Source, verified in root `LICENSE`)
- **Public Repository URL**: `https://github.com/[OWNER]/ReAgent` *(Placeholder: replace with actual published GitHub URL)*
- **Demo / Hosted URL**: `http://127.0.0.1:8080/` *(Local interactive Mission Control dashboard)*
- **YouTube Demonstration Video URL**: `https://youtu.be/[VIDEO_ID]` *(Placeholder: replace with uploaded unlisted/public video link)*

---

## 2. Project Description

### Inspiration
When LLM agents are deployed to interact with enterprise databases, payment processors, and cloud infrastructure, they suffer from a dangerous failure mode: **network drops and ambiguous timeouts after partial or committed state changes**. 
When a tool call times out or throws an unhandled socket exception, existing agent frameworks either:
1. **Blindly retry**, causing double-charges, duplicate inventory deductions, and silent ledger corruption on non-idempotent endpoints.
2. **Shallowly check presence** (`COUNT(*) > 0`), which falsely verifies corrupted or partial records up to 44.8% of the time.
3. **Hallucinate success**, trusting unverified return codes.

We built **ReAgent** to provide what agents lack: an **independent, deterministic verification barrier** that inspects actual state before declaring success or triggering retries.

### What It Does
ReAgent acts as an intermediary reliability layer between LLM agents and underlying execution environments:
1. **Derives & Freezes Success Contracts**: Before an action is dispatched, ReAgent derives an immutable `SuccessContract` specifying expected database columns, value invariants, and audit requirements.
2. **Independent Read-Only Probing**: When an action drops connection or times out (`COMMIT_THEN_TIMEOUT`), ReAgent probes the underlying system using a fresh connection with `query_only=ON` write-blocking enforced.
3. **Restrains Redundant Retries**: If the probe reveals the side effect already successfully committed, ReAgent restrains retry, preventing catastrophic double debits.
4. **Fails Closed**: Under unobservable states or data discrepancies (`WRONG_CONTENT`, `PARTIAL_COMMIT`), ReAgent immediately escalates rather than retrying blindly. **UNKNOWN is never treated as ABSENT.**
5. **Interactive Mission Control**: Demonstrates 5 recovery scenarios (Normal Success, Commit-Then-Timeout, Fail-Before-Commit, Unknown Outcome with Judge Oracle Reveal, and Partial Commit) across 3 isolated fixture-backed task templates (Wallet Transfer, Hotel Booking, Database Update).

### How We Built It
- **LLM Intelligence**: Powered by `nvidia/nemotron-3-super-120b-a12b` via **Nebius Token Factory** for structured contract generation.
- **Offline Candidate Validation Gate**: Strict Pydantic-based schema validation and task-grounding rules that reject ungrounded or vacuous contracts before execution.
- **Deterministic Engine**: Built with pure Python, SQLite, and file-locked token ledgers (`GlobalTokenLedger`), ensuring 100% deterministic offline evaluation and zero external dependencies during offline judging.
- **Evaluation Benchmark (DBOPS v1.3)**: 480 paired configurations (12 tasks &times; 8 fault modes &times; 5 policy arms) measuring false verifications, unsafe retries, and duplicate writes with mathematical denominator reconciliation.
- **Judge-Facing Application**: A pure Python web server (`reagent/ui_server.py`) serving an interactive Mission Control dashboard, event timeline replay, receipt download, and full benchmark explorer.

### Challenges We Ran Into
- **Post-Commit Ambiguity**: In distributed systems, a timeout does not mean failure. Distinguishing between a connection drop before commit versus after commit required designing isolated read-only observers that cannot contaminate the database.
- **Denominator & Mutual Exclusivity Accounting**: In comparative benchmarking, tracking 480 trials across 5 arms required establishing strict invariant accounting where every trial maps to exactly one outcome (`true_verified + false_verified + safe_escalations + unnecessary_escalations == 96`).
- **Secret & Ledger Safety**: Ensuring zero accidental provider token consumption by enforcing offline preflight verifications and strict `--live-run-opt-in` CLI barriers.

### Accomplishments That We're Proud Of
- **0.0% False Verification Rate**: Across 480 paired trials, ReAgent achieved zero false verifications, compared to a **44.8% false verification rate** for shallow presence checking (`SHARED_OBSERVATION`) and **66.7%** for Naive Retry.
- **Complete Elimination of Duplicate Debits**: While Naive Retry caused 24 unsafe retries resulting in 4 primary duplicate debits and 5 audit duplicates, ReAgent caused **zero duplicate writes**.
- **583 Tests Passing (11 skipped)**: Comprehensive test suite covering recovery invariants, pipeline contracts, UI endpoints, and adversarial review.
- **Zero-Dependency Judge Application**: An interactive browser UI and terminal demo that run in ~3.8 seconds without requiring API keys or external networks.

### What We Learned
- Presence does not equal correctness: testing whether a row exists is completely inadequate for enterprise reliability.
- Models must not grade their own work: independent, deterministic rule engines must evaluate actual environmental state.
- Safety requires fail-closed restraint: when state is unobservable, the safest response is clean human escalation.

### What's Next for ReAgent
- Extending ReAgent probe adapters to PostgreSQL, MongoDB, Stripe, and Kubernetes APIs.
- Autonomous multi-turn recovery negotiation for distributed microservices.
- Production deployment as a lightweight proxy sidecar for enterprise agent frameworks (LangChain, AutoGen, CrewAI).

---

## 3. Technology Stack & Technologies Used

- **NVIDIA Model**: `nvidia/nemotron-3-super-120b-a12b` (used for deriving structured verification contracts and postcondition assertions).
- **Nebius AI Cloud / Token Factory**: Used as the high-throughput inference endpoint for Nemotron models via OpenAI-compatible APIs.
- **Backend & Verification**: Python 3.10+, SQLite, Pydantic v2, `filelock`.
- **UI & Visualization**: Pure Python `http.server`, HTML5, CSS3, vanilla JavaScript.

---

## 4. Sponsor Technology Feedback

### Feedback for Nebius Token Factory
- **What worked well**: The OpenAI-compatible API format made provider integration clean and predictable. Token streaming and response latency for large models like Nemotron-3-Super-120B were competitive and reliable.
- **Areas for improvement**: 
  - Having built-in native support for strict JSON schema output (`response_format: {"type": "json_object", "schema": ...}`) directly enforced by the sampling engine would prevent schema hallucinations on structured contract derivations.
  - Adding granular per-request spending caps and budget alerting directly in the Token Factory dashboard would provide extra safety for developer experimentation.

### Feedback for NVIDIA Nemotron Models
- **What worked well**: `nvidia/nemotron-3-super-120b-a12b` showed strong reasoning capabilities in decomposing complex database transaction prompts into discrete precondition and postcondition SQL predicates.
- **Areas for improvement**: In edge cases involving composite multi-table keys, Nemotron occasionally included extraneous explanation keys outside the requested JSON schema; enforcing token-level grammar constraints at inference time would make structured extraction even more robust.

---

## 5. Judge Quickstart & Reproduction Commands

```powershell
# 1. Run the interactive offline judge demo (<4 seconds):
python -m reagent.experiments.dbops.judge_demo

# 2. Launch the local interactive web UI:
python reagent/ui_server.py --port 8080
# Open browser at: http://127.0.0.1:8080/

# 3. Validate cryptographic submission manifest (12/12 verified):
python -m reagent.experiments.dbops.validate_submission_manifest

# 4. Run the complete focused test suite (151 tests, all passing):
python -W default -m unittest tests/test_dbops_m14_submission_validation.py tests/test_dbops_m13_adversarial.py tests/test_dbops_m12_invariants.py tests/test_pipeline_native_ambiguity.py tests/test_dbops_m4a_recovery_demo.py tests/test_dbops_candidate_validation_gate.py tests/test_dbops_pipeline_vertical_slice.py tests/test_dbops_pilot_eval.py tests/test_dbops_pilot_analysis.py tests/test_dbops_launcher_safety.py tests/test_dbops_m6_submission.py tests/test_ui_server.py tests/test_nemotron_provider_mock.py tests/test_clean_start.py tests/test_submission_manifest_validator.py tests/test_m5_benchmark_accounting.py
```

---
*Draft prepared for NVIDIA x Nebius Global AI Hackathon 2026.*
