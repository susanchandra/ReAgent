# ReAgent: YouTube Demonstration Video Script & Shot Plan

**Target Duration**: 2 minutes 45 seconds (Strictly under 3 minutes)  
**Tone**: Technical, clear, authoritative, engineering-focused  
**Visual Setting**: Split screen with Browser UI (`http://127.0.0.1:8080/`) and Windows PowerShell Terminal  
**Audio**: Clear spoken English voiceover, zero distracting background music  

---

## Shot Plan & Storyboard

### Scene 1: The Core Problem — Why AI Agents Corrupt State (0:00 – 0:30)
- **Visual**: 
  - Slide/Terminal title: `ReAgent: Outcome Verification & Safe Recovery for AI Agent Actions`.
  - Headline: *"A timeout isn't a no."*
  - Supporting line: *"ReAgent checks what actually happened before an AI agent retries, so a lost response doesn't have to become a double charge."*
  - Technical principle: *"Unknown is not absent."*
  - Diagram showing an agent calling an external tool, the tool committing to the database, and the network dropping before the return confirmation (`COMMIT_THEN_TIMEOUT`).
- **Narration**:
  > "Autonomous AI agents are being connected to enterprise databases, payment gateways, and cloud APIs. But there is a silent, catastrophic flaw: network drops and timeouts. 
  > When an agent calls a state-changing tool and the connection drops after the database commits, the agent faces an ambiguous outcome: did the action succeed or fail? 
  > A timeout isn't a no. But naive agents treat timeouts as failure and blindly retry. On non-idempotent operations, this causes catastrophic duplicate mutations and double charges. 
  > ReAgent is an independent reliability and state verification layer that enforces a core principle: Unknown is not absent. It checks what actually happened before any retry."

---

### Scene 2: The Failure Demonstration — Mission Control (0:30 – 1:15)
- **Visual**:
  - Browser UI (`http://127.0.0.1:8080/`) showing the **Mission Control Dashboard**.
  - Showcases the 3 isolated task templates: Wallet Transfer, Hotel Booking, and Database Update.
  - Initial balance: $1,000.00. Intended transfer: $500.00 from ACCT-2201 to ACCT-3390.
  - Presenter selects Scenario B: Commit-Then-Timeout and clicks **"▶ Run Recovery Simulation"**.
  - Side-by-side comparison table updates dynamically; event timeline and receipt download become active.
  - Next, presenter briefly shows Scenario D (Unknown Outcome) with the **🔮 Judge Oracle Reveal** button, demonstrating how ReAgent escalates when unobservable while the oracle proves the committed reality.
- **Narration**:
  > "Here is our interactive Mission Control dashboard. We support three fixture-backed task demo templates: Wallet Transfer, Hotel Booking, and Database Update.
  > Let's test Scenario B: Commit-Then-Timeout on our $500 wallet transfer fixture starting at $1,000. 
  > We inject a network timeout after commit. Look at the Naive Retry baseline: it assumes failure, retries blindly, writes a second ledger row, and deducts $1,000, draining the customer's balance to zero.
  > Now look at ReAgent: instead of blindly retrying, ReAgent executes an isolated read-only observer query on a fresh connection with write-blocking enforced. It verifies that the $500 debit already occurred against a frozen contract, restrains retry, and preserves the customer's funds.
  > In Scenario D, when the observer itself cannot reach the database, ReAgent refuses to guess or retry blindly and safely escalates, while our Judge Oracle Reveal reveals the hidden truth for post-incident analysis."

---

### Scene 3: NVIDIA Nemotron & Nebius Token Factory Integration (1:15 – 1:55)
- **Visual**:
  - Architecture diagram transitioning to code editor showing `reagent/providers/nebius.py` and `reagent/contract_validator.py`.
  - Flow: `Task Context -> Nebius Token Factory (nvidia/nemotron-3-super-120b-a12b) -> Structured Success Contract Proposal -> Candidate Validation Gate -> Deterministic Verifier`.
- **Narration**:
  > "How does ReAgent integrate with NVIDIA and Nebius?
  > We utilize NVIDIA's Nemotron-3-Super-120B hosted on Nebius Token Factory. The model's role is critical and strictly defined: it derives declarative success contracts—specifying preconditions, expected column values, and multi-table invariants—before tool execution.
  > Crucially, ReAgent never lets the model grade its own homework. Candidate contracts are passed through an offline candidate validation gate that strictly checks Pydantic schemas and schema grounding. 
  > Once validated, the success contract is frozen. If the provider goes down or outputs malformed predicates, ReAgent fails closed, preventing ungrounded contracts from reaching production."

---

### Scene 4: 480-Trial Benchmark Evidence & Denominator Integrity (1:55 – 2:25)
- **Visual**:
  - Switch to Browser UI **Tab 2: 480-Trial Comparative Benchmark (DBOPS v1.3)**.
  - Highlighting the aggregate results table:
    - `REAGENT_CONTRACT`: 0.0% False Verification Rate, 0 Unsafe Retries, 0 Duplicate Writes.
    - `SHARED_OBSERVATION`: 44.8% False Verification Rate.
    - `NAIVE_RETRY`: 24 Unsafe Retries resulting in 4 Primary Duplicate Debits and 5 Secondary Audit Log Duplications.
  - Pointing out the accounting formula: $\text{True Ver} + \text{False Ver} + \text{Safe Esc} + \text{Unnec Esc} = 96$ for every arm.
- **Narration**:
  > "To evaluate ReAgent systematically, we developed DBOPS v1.3: a 480-trial paired benchmark across 12 tasks, 8 fault modes, and 5 policy arms.
  > The results are definitive:
  > First, shallow presence checks like `COUNT > 0` fail 44.8% of the time because row presence does not equal semantic correctness. ReAgent eliminates false verifications entirely, achieving 0.0%.
  > Second, Naive Retry executed 24 unsafe retries, causing 4 permanent duplicate primary debits and 5 duplicate audit rows. ReAgent caused zero duplicate writes across all 96 trials.
  > Every single trial reconciles cleanly across all 96 paired trials per arm with zero missing data."

---

### Scene 5: Reproducibility, Honest Limitations & Conclusion (2:25 – 2:45)
- **Visual**:
  - Windows PowerShell running the offline judge demo:
    `python -m reagent.experiments.dbops.judge_demo`
  - Showing clean sub-4-second execution and cryptographic artifact SHA-256 validation.
  - Final screen with GitHub repository link and hackathon summary.
- **Narration**:
  > "ReAgent runs 100% offline and is completely reproducible with a single PowerShell command: `python -m reagent.experiments.dbops.judge_demo`.
  > In the spirit of honest engineering, our benchmark numbers are derived from deterministic synthetic SQLite fixtures, proving the algorithmic recovery semantics. Live model generation is isolated under supervised human-in-the-loop controls.
  > ReAgent transforms brittle agents into resilient, verifiable systems that enterprise engineers can trust. Thank you."

---
*Script certified for NVIDIA x Nebius Global AI Hackathon 2026.*
