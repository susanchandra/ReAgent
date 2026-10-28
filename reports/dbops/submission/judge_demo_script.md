# ReAgent: Sub-Three-Minute Judge Demonstration Script

**Target Competition:** NVIDIA × Nebius Global AI Hackathon 2026 — Track: Best Apps and Agents  
**Target Duration:** 2 minutes 40 seconds (Hard limit: Under 3 Minutes)  
**Format:** Screen recording of local web UI (`http://127.0.0.1:8080/`) + clear voiceover  
**Prerequisites:**  
1. Start local UI server: `python reagent/ui_server.py --port 8080`  
2. Open browser to `http://127.0.0.1:8080/?view=view-workbench` (or click `Task Workbench` in sidebar)  
3. Zero internet or API keys required — runs 100% locally and offline.

---

## Shot-by-Shot Timeline & Spoken Narration

### Part 1: The Core Problem & Value Proposition (0:00 – 0:35)

- **[0:00 – 0:15] Visual:** Start on the **Task Workbench** screen. The hero banner is visible:  
  *"A timeout is not a no. Blind retries corrupt state. ReAgent independently probes external service state before choosing to verify, retry, or escalate."*  
  Badge: `OFFLINE · DETERMINISTIC DEMO`.
  - **Spoken:**  
    > "When autonomous AI agents interact with real-world APIs and databases, they face a dangerous problem: ambiguous network timeouts. If a tool call drops its socket connection after making a state change, existing agent frameworks assume the action failed and blindly retry. On non-idempotent endpoints, this creates duplicate hotel bookings, double debits, and corrupted state."

- **[0:15 – 0:35] Visual:** Highlight the Task Input pre-filled with the flagship task:  
  `Book room 14B for Maya Angelou, check-in 2026-11-01, check-out 2026-11-05`.  
  Point to the scenario selector showing **Scenario B: Commit-Then-Timeout**.
  - **Spoken:**  
    > "Our core principle is: *A timeout isn't a no. Unknown is not absent.*  
    > Here, our agent is booking Room 14B at Grand Hotel across a real local HTTP service simulator with its own isolated SQLite database. In Scenario B, the reservation commits to disk, but the HTTP connection drops before the client receives an acknowledgment."

---

### Part 2: Scenario B Live Run & Evidence Trace (0:35 – 1:25)

- **[0:35 – 0:50] Visual:** Click the blue **▶ Run** button.  
  - Button switches to `⏳ Running…` then `✓ Complete`.  
  - The **Outcome Banner** turns green: `VERIFIED`.  
  - Caption: *"The booking existed despite the timeout. ReAgent verified it and blocked a duplicate retry."*  
  - Run ID and Provenance (`GENUINE_LOCAL_HTTP_SERVICE`) appear.
  - **Spoken:**  
    > "Let's run Scenario B. Watch what happens. Instead of guessing or retrying, ReAgent's outcome verification layer steps in. The status is immediately confirmed as VERIFIED, and redundant retry is blocked."

- **[0:50 – 1:10] Visual:** Scroll slightly to the **Live Independent Evidence Trace**.  
  - Point to event sequence:  
    `1. INIT (SYSTEM) Seed state`  
    `2. INIT (REAGENT) FREEZE_CONTRACT (room=14B, guest=Maya Lin, amount=$350)`  
    `3. EXECUTE (TOOL_CALL) hotel_booking_http`  
    `4. FAULT (SYSTEM) COMMIT_THEN_TIMEOUT`  
    `5. PROBE (REAGENT) OBSERVER_QUERY: GET /reservations/BKG-REAGENT-HOTEL`  
    `6. PROBE (REAGENT) PROBE_RESULT: status=CONFIRMED | room=14B — PRESENT`  
    `7. DECIDE (REAGENT) RESTRAIN_RETRY -> VERIFIED`
  - **Spoken:**  
    > "Here is the exact evidence trace from this run. Before execution, ReAgent froze an immutable SuccessContract. When the timeout fault occurred, ReAgent dispatched a fresh read-only query directly to the hotel service's status endpoint. The probe found the reservation already committed. ReAgent verified postconditions against the frozen contract and safely restrained retry."

- **[1:10 – 1:25] Visual:** Point to the **Decision Inspector** comparing Naive vs. ReAgent:  
  - Left card (Naive Retry): `DUPLICATE_CORRUPTED ⚠ DUPLICATE` — *2 attempts — unsafe duplicate retry attempted (UNIQUE key conflict)*.  
  - Right card (ReAgent): `VERIFIED` — *1 independent probe — evidence-based decision*.  
  - Bottom callout: *Safe because: the booking was found in persisted hotel SQLite state via an independent read-only probe. ReAgent suppressed the redundant retry.*
  - **Spoken:**  
    > "Look at the contrast: the naive agent saw the timeout, assumed failure, and issued a blind second booking request that caused a database key conflict. ReAgent achieved clean verification with zero duplicate bookings."

---

### Part 3: Unknown Outcome & Failing Closed (1:25 – 2:05)

- **[1:25 – 1:45] Visual:** Select **Scenario D (Observer Unavailable)** from the dropdown. Click **▶ Run**.  
  - Outcome Banner turns amber: `ESCALATED`.  
  - Status Badge: `ESCALATED`.  
  - Caption: *"Observer unavailable — outcome genuinely UNKNOWN. UNKNOWN is not ABSENT. ReAgent escalated safely without guessing."*  
  - Evidence trace shows: `PROBE_FAILED: HTTP 503 Service Unavailable -> ESCALATE`.
  - **Spoken:**  
    > "Now let's switch to Scenario D: what happens if the observer itself is down? A naive agent retries anyway, risking disaster. But ReAgent strictly enforces our safety invariant: *UNKNOWN is not ABSENT.* When state cannot be authoritatively proven, automated retry is restrained and the action is safely escalated to an operator."

- **[1:45 – 2:05] Visual:** Switch scenario to **Scenario C (Fail-Before-Commit)**. Click **▶ Run**.  
  - Outcome: `VERIFIED`.  
  - Trace: `PROBE_RESULT: rows=0 — CONFIRMED_ABSENT -> SAFE_RETRY -> Post-retry probe: VERIFIED`.
  - **Spoken:**  
    > "In Scenario C, the tool failed before writing to disk. ReAgent probes, authoritatively proves absence—zero records exist—and only then authorizes exactly one safe retry, reaching verified success."

---

### Part 4: Nemotron Integration, Benchmark & Reproducibility (2:05 – 2:40)

- **[2:05 – 2:25] Visual:** Click **Mission Control** or **480-Trial Benchmark** in the sidebar.  
  - Show the 480-trial evaluation matrix:  
    - ReAgent Contract: **0.0% False Verification Rate**, **0 Unsafe Retries**, **0 Duplicate Debits**.  
    - Naive Retry: **66.7% False Verification Rate**, **24 Unsafe Retries**.
  - **Spoken:**  
    > "We evaluated ReAgent across 480 paired trials spanning 12 enterprise tasks. While naive retry suffered a 66.7% false verification rate and repeated duplicate writes, ReAgent achieved a zero percent false verification rate with zero duplicate writes.  
    > ReAgent integrates NVIDIA Nemotron via Nebius Token Factory for structured contract proposal. But critically: model prose is never treated as execution evidence. The deterministic verifier maintains final authority."

- **[2:25 – 2:40] Visual:** Switch to terminal showing test suite:  
  `python -m pytest tests/ -q --tb=line`  
  Output: `739 passed, 13 skipped, 20 subtests passed in ~137s`.
  - **Spoken:**  
    > "The entire test suite—739 tests—passes completely offline with zero external dependencies. The judge demo, hotel simulator, and Task Workbench are completely reproducible in seconds. Thank you."

---

## Recording Tips for the Presenter
- Keep pacing deliberate: total spoken words is ~375 words, which takes ~2 minutes 30 seconds at a normal speaking rate of 140–150 wpm.
- Keep the mouse cursor on the active visual element being narrated (the Run button, the Trace row, the Naive vs ReAgent card).
- Do not edit or speed up the video footage; the UI interactions respond in sub-second time.
