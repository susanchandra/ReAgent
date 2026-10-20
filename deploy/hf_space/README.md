---
title: ReAgent Mission Control
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ReAgent — AI Agent Reliability & Verification Layer
### NVIDIA × Nebius Global AI Hackathon 2026 — Best Apps and Agents Track

> **Core Principle:** *"A timeout isn't a no. An action can commit even when its tool response is lost."*

ReAgent freezes an expected-outcome contract before execution, independently probes resulting system state after ambiguous errors (e.g., HTTP timeouts, process drops), and deterministically verifies postconditions before allowing retries or escalations.

---

## Public Demonstration Security & Guardrails

1. **100% Offline & Deterministic:** All interactive simulations (Scenarios A through E) run against isolated, disposable SQLite sandbox environments seeded on the fly.
2. **Fail-Closed Public Mode:** External network calls and live model inference routes are locked in this public demo environment (`PUBLIC_DEMO_LOCKED`). Zero unauthenticated API calls, zero billing risk.
3. **No Secrets in Container:** Operates cleanly without any API keys or credentials.
4. **Independent Oracle:** Ground truth for Scenario D (UNKNOWN outcome) is evaluated via a separate, blinded oracle that is strictly separated from the agent's decision logic.

---

## Key Scenarios Demonstrated

* **Scenario A — Normal Success:** Clean execution; probe confirms state on first attempt.
* **Scenario B — Commit-Then-Timeout (Lead Scenario):** The database write commits, but the tool response times out. Naive retry causes duplicate credit/debit corruption ($1,000 double debit). ReAgent independently probes, verifies the committed state, and halts safely ($500 preserved, 0 duplicates).
* **Scenario C — Fail-Before-Commit:** Write failed prior to commit. Probe detects absence of expected row; ReAgent safely authorizes retry.
* **Scenario D — Unknown Outcome:** Commit occurs, but observation probe is unreachable. Enforces **"UNKNOWN is not ABSENT"** — escalates to human operator rather than hallucinating success or blindly retrying.
* **Scenario E — Partial Commit:** Multi-table partial write discrepancy detected; halts and alerts operator.

---

## 480-Trial Comparative Benchmark

Evaluated across 5 policy arms with 96 paired trials per arm on disposable SQLite fixtures:
* **REAGENT_CONTRACT:** 0.0% false verification rate, 0 unsafe retries, 0 duplicate database records.
* **NAIVE_RETRY:** 66.7% false verification rate, 24 unsafe retries, 9 duplicate database rows (4 primary entity debits, 5 audit rows).
