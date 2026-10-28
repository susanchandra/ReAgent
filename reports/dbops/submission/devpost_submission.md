# ReAgent: Devpost Submission Draft

> **Submission Status**: **DRAFT PREPARED FOR SUBMISSION**  
> *(Note: This document is a generated submission draft for the project owner to review, complete personal attestations, and paste into Devpost. It does NOT claim that the Devpost form has already been submitted online).*

---

## 1. Basic Project Information

- **Project Title**: ReAgent: Outcome Verification & Safe Recovery for AI Agent Actions
- **Elevator Pitch / Tagline**: *"A timeout isn't a no. Unknown is not absent."*
  - *Supporting Line*: ReAgent checks what actually happened before an AI agent retries, so a lost response does not become an unsafe duplicate side-effect.
  - *Technical Principle*: Deterministic environmental verification, not model prose, retains authority over execution outcomes.
  - *Full Pitch*: An independent outcome-verification and safe-recovery layer for autonomous AI agents that eliminates false completion claims, restrains duplicate mutations on timeout ambiguity, and enforces fail-closed recovery.
- **Target Track**: Best Apps and Agents
- **Target Competition**: NVIDIA × Nebius Global AI Hackathon 2026
- **License**: MIT License (Open Source, verified in root `LICENSE`)
- **Public Repository URL**: `https://github.com/susanchandra/ReAgent` *(TODO: confirm repository visibility and final URL before submission)*
- **Demo / Hosted URL**: `http://127.0.0.1:8080/` *(Local interactive Task Workbench & Mission Control dashboard)*
- **Hugging Face Space**: `https://huggingface.co/spaces/...` *(TODO: confirm public Space URL if publishing; public demo endpoint is locked fail-closed without credentials)*
- **Demonstration Video URL**: `https://youtu.be/[TODO_VIDEO_ID]` *(TODO: record sub-3-minute video using the provided demo script and paste URL here)*

---

## 2. Project Description

### The Problem: Post-Commit Ambiguity
When LLM-driven agents interact with real-world services—such as hotel reservation APIs, database transactions, or cloud infrastructure—they face a fundamental distributed systems dilemma: **network drops and socket timeouts after state has already changed**.

When a tool call times out or throws a connection error, existing agent frameworks typically make one of three fatal assumptions:
1. **Blind retry on timeout**: The agent assumes timeout equals failure and re-executes the tool. On non-idempotent endpoints, this causes duplicate hotel bookings, double debits, or corrupted inventory.
2. **Shallow presence check (`COUNT(*) > 0`)**: Checking solely whether a record exists falsely verifies corrupt, mismatched, or partial records up to 44.8% of the time.
3. **Model self-evaluation**: Asking the LLM whether the tool succeeded allows model hallucinations to declare completion without environmental evidence.

### Core Principle
> **"A timeout isn't a no. Unknown is not absent."**  
> An unreceived acknowledgment does not mean the mutation failed. When outcome state cannot be authoritatively observed, automated retry must be restrained and the task safely escalated.

### What ReAgent Does: The Recovery Flow
ReAgent introduces a deterministic verification barrier between the LLM and real-world tools:
1. **Contract Derivation & Freezing**: Before execution, a `SuccessContract` is frozen with explicit postconditions (exact primary IDs, target entity states, expected amounts, and audit requirements). Nemotron acts as an *advisory proposal engine*; the contract is schema-validated, frozen, and immutable during execution.
2. **Tool Invocation**: The action is dispatched to the tool (e.g. an HTTP booking service).
3. **Independent Read-Only Probing**: When an action drops connection or times out (`COMMIT_THEN_TIMEOUT`), ReAgent does *not* blindly retry. It dispatches a fresh, independent read-only observer query across the service boundary.
4. **Deterministic Contract Verification**: The verifier compares observed postconditions directly against the frozen contract.
5. **Authoritative Recovery Policy**:
   - **Prior Commit Confirmed**: Side-effect verified in state &rarr; Redundant retry is **RESTRAINED** &rarr; `VERIFIED`. Zero duplicate mutations.
   - **Authoritative Absence Proved**: Probe confirms 0 records exist &rarr; Exactly one safe retry authorized &rarr; `VERIFIED`.
   - **Observer Failure (Unknown State)**: Probe is unavailable (HTTP 503) &rarr; `UNKNOWN` is not `ABSENT` &rarr; Blind retry **BLOCKED** &rarr; `ESCALATED`.
   - **Partial / Conflicting State**: Multi-table mismatch detected &rarr; Recovery **HALTED** &rarr; `ESCALATED`.

---

## 3. Five Flagship Scenarios (A through E)

ReAgent demonstrates these recovery semantics across a dedicated local HTTP Hotel Booking Service simulator (with isolated SQLite persistence) and enterprise DBOPS tasks:

- **Scenario A (Normal Success)**: Both commit and response succeed. ReAgent independently probes and verifies postconditions. Naive agent succeeds unverified.
- **Scenario B (Commit-Then-Timeout)**: The booking commits to SQLite, but the HTTP connection drops before acknowledgment. Naive retry blindly attempts a second booking (producing a duplicate or triggering key conflicts). ReAgent's independent probe discovers the committed booking, restrains retry, and confirms success with **zero duplicate bookings**.
- **Scenario C (Fail-Before-Commit)**: The service fails with HTTP 500 before writing to disk. ReAgent probes, authoritatively proves absence (0 records), and authorizes exactly one safe retry under policy bounds.
- **Scenario D (Observer Unavailable)**: Tool times out and the status probe endpoint returns HTTP 503. Because outcome is genuinely `UNKNOWN`, ReAgent enforces the invariant: *UNKNOWN is not ABSENT*. It blocks automated retry and escalates cleanly.
- **Scenario E (Partial / Conflicting State)**: A split-brain fault commits the reservation but drops the audit log. ReAgent detects the multi-table discrepancy and halts automated recovery to prevent state corruption.

---

## 4. Architecture & Nemotron Integration

- **NVIDIA Nemotron (`nvidia/nemotron-3-super-120b-a12b`)**:
  - Used via **Nebius Token Factory** for structured proposal of contracts and failure diagnoses.
  - **Zero-Trust Boundary**: Model prose is never treated as verification evidence. The model advises; the deterministic verifier and policy engine decide.
  - **Offline Reference vs. Live Mode**: The entire test suite and demonstration run 100% offline and deterministically without requiring an API key. When `NEBIUS_API_KEY` is present, Nemotron provides live contract proposals; in offline mode, pre-validated reference contracts are used with explicit `OFFLINE` provenance labeling.
  - **Public Deployment Lockout**: The Hugging Face Space deployment (`deploy/hf_space/entrypoint.py`) locks inference endpoints (`PUBLIC_DEMO_LOCKED`), preventing unauthenticated live token consumption.
- **HTTP Hotel Booking Simulator (`reagent/services/hotel_booking_service.py`)**:
  - Runs on a real local HTTP server on an ephemeral port.
  - Owns its own independent SQLite database (`reservations` and `reservation_audit_log` tables).
  - Deterministically injects socket timeouts, HTTP 500s, HTTP 503s, and partial commits.
- **Interactive Task Workbench (`reagent/ui_server.py`)**:
  - Single-screen judge dashboard with real-time prompt input, example task chips, live execution trace, side-by-side Naive vs. ReAgent comparison, and collapsible raw evidence inspector.

---

## 5. Comparative Evaluation & Benchmark

Across the 480-trial paired DBOPS benchmark (12 enterprise tasks &times; 8 fault modes &times; 5 policy arms):
- **False Verification Rate**:
  - Naive Retry Baseline: **66.7%**
  - Shallow Presence Check (`SHARED_OBSERVATION`): **44.8%**
  - ReAgent Contract Verification: **0.0%**
- **Duplicate Writes Under Ambiguous Timeouts**:
  - Naive Retry: 24 unsafe retries resulting in duplicate ledger debits / state corruption.
  - ReAgent: **0 duplicate debits** (all redundant retries safely restrained).

---

## 6. Sponsor Technology Feedback

### Feedback for Nebius Token Factory
- **Strengths**: High inference throughput on large Nemotron models; standard OpenAI-compatible REST endpoint simplified integration; predictable streaming behavior.
- **Constructive Recommendations**:
  1. Native server-side schema adherence (`response_format: {"type": "json_schema", "schema": ...}`) directly enforced by constrained decoding would eliminate schema deviations on complex composite-key tasks.
  2. Built-in per-token request budget caps configurable in headers would provide extra safeguards for agent loop development.

### Feedback for NVIDIA Nemotron Models
- **Strengths**: Strong reasoning capability in translating human task objectives into structured precondition and postcondition SQL predicates.
- **Constructive Recommendations**:
  1. On multi-table constraint prompts, Nemotron occasionally appended conversational explanation outside the requested JSON boundaries; grammar-guided sampling helps ensure strict JSON output.

---

## 7. Reproduction & Quickstart Commands

```powershell
# 1. Clone repository
git clone https://github.com/susanchandra/ReAgent.git
cd ReAgent

# 2. Run automated release verification gates (30/30 preflight checks):
python launch_demo.py --check

# 3. Run canonical test suite (657 passed, 13 skipped, 10 subtests passed):
# NOTE: Run with 'tests/' directory argument to prevent collecting root test_output.txt
python -m pytest tests/ -q --tb=line

# 4. Run focused Phase 3 & 4 Workbench test suite (51 passed, 2 skipped):
python -m pytest tests/test_phase3_workbench.py -v --tb=short

# 5. Launch local interactive Task Workbench & Mission Control:
python reagent/ui_server.py --port 8080
# Open browser at: http://127.0.0.1:8080/
```

---

## 8. Required Personal Declarations & Attestations

*(Note: In accordance with hackathon rules, personal attestations must be reviewed and confirmed directly by the participant).*

- [ ] **Originality & Prior Work**: I confirm that ReAgent was developed for the NVIDIA × Nebius Global AI Hackathon 2026. Any third-party libraries or reference implementations used are open-source and listed in `requirements.txt` and project documentation.
- [ ] **Sponsor Technology Usage**: I confirm that NVIDIA Nemotron and Nebius Token Factory were evaluated and integrated as described in the architecture.
- [ ] **Video & Code Submission**: I confirm that the recorded demonstration video and repository links are accessible and unlisted/public for judging.
