# Operational Boundaries & Limitations

In the interest of scientific rigor and full transparency, this document articulates the current operational boundaries, assumptions, and known limitations of the ReAgent system.

---

## 1. Provenance: Deterministic Fixtures vs. Live Model Derivation

### Current State:
- The recovery demonstrations and 480-trial comparative benchmarks in this submission are generated using **deterministic offline mock providers and registered golden fixtures** (`SYNTHETIC / DETERMINISTIC OFFLINE FIXTURES`).
- The evaluation harness, guarded runner (`eval_runner.py`), and scoring pipeline (`eval_scoring.py`) for live NVIDIA Nemotron derivation have been fully implemented with:
  - Hard 36-request ceilings (3 requests × 12 tasks).
  - Explicit budget enforcement guards.
  - Zero automatic retries.
  - Configuration freeze mechanisms.
- However, per hackathon safety instructions, **zero live network calls, token consumption, or cloud credits were incurred** during this implementation phase.
- Therefore, **the capability of raw LLM prompts to synthesize gold-equivalent contracts autonomously remains unproven on live endpoints until authorized live credit execution.**

---

## 2. Observability Boundaries: The "Unobservable" Limit

### Fail-Closed by Design:
- ReAgent fundamentally relies on an independent read-only observer to query post-execution state.
- If a system component or third-party API provides **no observable query interface** (`UNOBSERVABLE`) or if the query endpoint is unreachable (`STATUS_UNAVAILABLE`), ReAgent **cannot verify state**.
- In these scenarios, ReAgent does not guess or retry; it strictly **fails closed to `ESCALATED`**.
- *Practical implication*: Workflows that interact with write-only "black box" webhooks cannot be verified automatically and will require human intervention upon timeout.

---

## 3. Overhead & Latency

### Independent Probing Cost:
- ReAgent introduces a secondary read-only query following any ambiguous tool execution.
- In our benchmark, this added an average of **1.23 probe queries** and approximately **17 ms** of local processing overhead per ambiguous execution.
- For high-throughput systems operating under microseconds, this read-only round-trip must be balanced against the cost of financial corruption or duplicate records.

---

## 4. Schema Compatibility

### Relational vs. Unstructured Systems:
- The current DBOPS benchmark focuses on relational SQL storage engines (SQLite) with strict typing, foreign keys, and tables.
- Extending contracts to unstructured document stores (e.g., MongoDB, DynamoDB) or event streams (Kafka) requires implementing domain-specific observer connectors.
