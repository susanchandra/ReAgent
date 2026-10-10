"""
ReAgent UI Dashboard Server — Milestone 6.1
Evidence-first, judge-ready product interface with strict trace integrity.

Architecture: Pure Python http.server — zero external dependencies.
All HTML/CSS/JS is embedded inline; the server serves a single self-contained page.
Static HTML export via --export-html preserves all data and interactivity.

Provenance: All data is derived from run_benchmark_scenarios() which executes
the actual deterministic benchmark pipeline. No values are fabricated.
"""

import os
import sys
import json
import html
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reagent.benchmark import run_benchmark_scenarios, BenchmarkComparisonReport


# ---------------------------------------------------------------------------
# Data normalization & safe JSON serialization helpers
# ---------------------------------------------------------------------------

def _status_meta(status: str) -> Dict[str, str]:
    """Return CSS class, icon, and label for a final_status string."""
    s = (status or "").upper()
    if s == "VERIFIED":
        return {"css": "verified", "icon": "✓", "label": "VERIFIED"}
    elif s == "ESCALATED":
        return {"css": "escalated", "icon": "⚠", "label": "ESCALATED"}
    elif s == "FALSE_SUCCESS":
        return {"css": "false-success", "icon": "✗", "label": "FALSE SUCCESS"}
    elif s in ("FAILED", "SUCCESS_UNVERIFIED"):
        return {"css": "failed", "icon": "✗", "label": s.replace("_", " ")}
    else:
        return {"css": "unknown", "icon": "?", "label": s or "UNKNOWN"}


def _safe_json_embed(data: Any) -> str:
    """
    Safely serialize data to JSON for embedding inside a <script> tag.
    Prevents script injection and syntax termination by escaping:
      - '</' -> '<\\/'
      - '<!--' -> '<\\!--'
      - U+2028 (Line Separator) -> '\\u2028'
      - U+2029 (Paragraph Separator) -> '\\u2029'
    """
    raw_json = json.dumps(data, indent=None, separators=(',', ':'), ensure_ascii=False)
    safe = (
        raw_json
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return safe


def _normalize_report(report: BenchmarkComparisonReport) -> Dict[str, Any]:
    """
    Produce a JSON-serializable dict of all UI data.
    Every displayed value is derived from an actual benchmark record.
    Missing fields are explicitly labeled or None — never invented.
    """
    runs_out = []
    for r in report.runs:
        sm = _status_meta(r.final_status)
        is_reagent = (r.strategy == "reagent_diagnosis_guided")
        runs_out.append({
            "scenario_id": r.scenario_id,
            "scenario_name": r.scenario_name,
            "fault_type": r.fault_type,
            "strategy": r.strategy,
            "strategy_label": "ReAgent" if is_reagent else "Illustrative Naive Baseline",
            "attempts_executed": r.attempts_executed,
            "final_status": r.final_status,
            "status_meta": sm,
            "verified": r.verified,
            "expected_terminal_outcome": r.expected_terminal_outcome,
            "contract_applicable": r.contract_applicable,
            "diagnosed_failure_type": r.diagnosed_failure_type or "Not recorded",
            "selected_recovery": r.selected_recovery or "None",
            "discrepancies": r.discrepancies,
            "timeline_phases": r.timeline_phases,
            "duration_ms": r.duration_ms,
            "tool_observations": r.tool_observations,
            "claimed_outcome": r.claimed_outcome,
            "observed_outcome": r.observed_outcome,
            "frozen_contract_summary": r.frozen_contract_summary,
            "timeline_events": r.timeline_events,
            "safe_receipt": r.safe_receipt,
        })

    return {
        "provenance": "OFFLINE · DETERMINISTIC",
        "provenance_note": (
            "All results derived from seeded, deterministic offline benchmark with DevelopmentLLMProvider. "
            "No live API calls performed."
        ),
        "scenarios_evaluated": report.scenarios_evaluated,
        "reagent_verified_count": report.reagent_verified_count,
        "reagent_escalated_count": report.reagent_escalated_count,
        "reagent_false_success_count": report.reagent_false_success_count,
        "baseline_verified_count": report.baseline_verified_count,
        "baseline_failed_count": report.baseline_failed_count,
        "baseline_false_success_count": report.baseline_false_success_count,
        "expected_verifiable_count": report.expected_verifiable_count,
        "expected_escalate_count": report.expected_escalate_count,
        "reagent_completed_and_verified": report.reagent_completed_and_verified,
        "reagent_correctly_escalated": report.reagent_correctly_escalated,
        "runs": runs_out,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_dashboard_html(report: BenchmarkComparisonReport) -> str:
    """Generates a complete self-contained HTML dashboard — no CDN dependencies."""
    data = _normalize_report(report)
    report_json_safe = _safe_json_embed(data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ReAgent — Evidence-First Agent Reliability</title>
  <style>
    /* ── Design Tokens ──────────────────────────────────────────────── */
    :root {{
      --bg:         #0E1013;
      --panel:      #15181D;
      --surface:    #1B1F26;
      --border:     #262B33;
      --border-hi:  #343B47;
      --text:       #ECE8E1;
      --text-2:     #A39E94;
      --text-3:     #6B6760;
      --cobalt:     #5B8CFF;
      --cobalt-dim: #2B4099;
      --verified:   #4CC38A;
      --verified-dim:#1B4F35;
      --failed:     #E5645B;
      --failed-dim: #5A1A17;
      --escalated:  #E3A94B;
      --escalated-dim: #5A3E0F;
      --false-success: #E5645B;
      --mono:       "JetBrains Mono", "Fira Mono", "Cascadia Code", "Consolas", monospace;
      --sans:       -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }}

    /* ── Reset & Base ───────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ height: 100%; scroll-behavior: smooth; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 13px;
      line-height: 1.55;
      min-height: 100vh;
    }}
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{ transition: none !important; animation: none !important; }}
    }}

    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--panel); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border-hi); border-radius: 3px; }}

    /* ── Typography ─────────────────────────────────────────────────── */
    h1 {{ font-size: 16px; font-weight: 700; letter-spacing: -0.2px; }}
    h2 {{ font-size: 13px; font-weight: 600; color: var(--text-2); text-transform: uppercase; letter-spacing: 0.8px; }}
    h3 {{ font-size: 13px; font-weight: 600; color: var(--text); }}
    code, pre, .mono {{
      font-family: var(--mono);
      font-size: 11.5px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    a {{ color: var(--cobalt); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    a:focus-visible {{ outline: 2px solid var(--cobalt); outline-offset: 2px; border-radius: 2px; }}

    /* ── Layout ─────────────────────────────────────────────────────── */
    #app {{ display: flex; flex-direction: column; min-height: 100vh; }}

    /* ── Top Bar ────────────────────────────────────────────────────── */
    #topbar {{
      display: flex;
      align-items: center;
      gap: 20px;
      padding: 0 20px;
      height: 48px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 100;
      flex-shrink: 0;
    }}
    .wordmark {{
      display: flex; align-items: center; gap: 8px; flex-shrink: 0;
    }}
    .wordmark-logo {{
      width: 28px; height: 28px;
      background: var(--cobalt);
      border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 800; font-size: 13px; color: #fff; letter-spacing: -0.5px;
      flex-shrink: 0;
    }}
    .wordmark-name {{ font-weight: 700; font-size: 15px; color: var(--text); letter-spacing: -0.3px; }}
    .wordmark-tagline {{ font-size: 11px; color: var(--text-3); white-space: nowrap; display: none; }}
    @media (min-width: 640px) {{ .wordmark-tagline {{ display: block; }} }}
    .topbar-divider {{ width: 1px; height: 24px; background: var(--border); flex-shrink: 0; }}
    .provenance-badge {{
      font-family: var(--mono); font-size: 10px; font-weight: 600;
      padding: 3px 8px; border-radius: 3px;
      background: #1A1F28; border: 1px solid var(--border-hi);
      color: var(--text-2); white-space: nowrap; letter-spacing: 0.5px;
    }}
    .topbar-nav {{ display: flex; gap: 4px; flex: 1; }}
    .nav-btn {{
      padding: 6px 12px;
      border: none; border-radius: 5px;
      font-size: 12px; font-weight: 500;
      cursor: pointer; transition: background 0.15s, color 0.15s;
      background: transparent; color: var(--text-2);
    }}
    .nav-btn:hover {{ background: var(--surface); color: var(--text); }}
    .nav-btn.active {{ background: var(--cobalt-dim); color: var(--cobalt); font-weight: 600; }}
    .nav-btn:focus-visible {{ outline: 2px solid var(--cobalt); outline-offset: 1px; }}
    .topbar-actions {{ display: flex; gap: 8px; margin-left: auto; flex-shrink: 0; }}
    .action-btn {{
      padding: 5px 10px; border-radius: 4px;
      border: 1px solid var(--border); background: var(--surface);
      color: var(--text-2); font-size: 11px; cursor: pointer;
      transition: border-color 0.15s, color 0.15s;
    }}
    .action-btn:hover {{ border-color: var(--cobalt); color: var(--cobalt); }}
    .action-btn:focus-visible {{ outline: 2px solid var(--cobalt); outline-offset: 1px; }}

    /* ── Main Content ───────────────────────────────────────────────── */
    #main {{ flex: 1; padding: 20px; max-width: 1400px; margin: 0 auto; width: 100%; }}

    /* ── Tab Panels ─────────────────────────────────────────────────── */
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}

    /* ── Status Pills / Badges ──────────────────────────────────────── */
    .pill {{
      display: inline-flex; align-items: center; gap: 4px;
      padding: 2px 7px; border-radius: 3px;
      font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
      border: 1px solid transparent; white-space: nowrap;
    }}
    .pill.verified  {{ background: var(--verified-dim);  color: var(--verified);  border-color: var(--verified); }}
    .pill.escalated {{ background: var(--escalated-dim); color: var(--escalated); border-color: var(--escalated); }}
    .pill.failed    {{ background: var(--failed-dim);    color: var(--failed);    border-color: var(--failed); }}
    .pill.false-success {{ background: var(--failed-dim); color: var(--failed);  border-color: var(--failed); }}
    .pill.unknown   {{ background: var(--surface); color: var(--text-2); border-color: var(--border); }}
    .pill.cobalt    {{ background: rgba(91,140,255,0.12); color: var(--cobalt); border-color: var(--cobalt-dim); }}
    .pill.neutral   {{ background: var(--surface); color: var(--text-2); border-color: var(--border); }}

    /* ── Cards / Panels ─────────────────────────────────────────────── */
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .card-header {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
    }}
    .card-body {{ padding: 14px 16px; }}
    .card-section-label {{
      font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
      color: var(--text-3); margin-bottom: 10px;
    }}

    /* ── KV Rows ────────────────────────────────────────────────────── */
    .kv-row {{
      display: flex; gap: 8px; padding: 6px 0;
      border-bottom: 1px solid var(--border); font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .kv-row:last-child {{ border-bottom: none; }}
    .kv-key {{ color: var(--text-2); min-width: 140px; flex-shrink: 0; }}
    .kv-val {{ color: var(--text); word-break: break-word; overflow-wrap: anywhere; }}
    .kv-val.mono {{ font-family: var(--mono); font-size: 11px; }}

    /* ── Tables ─────────────────────────────────────────────────────── */
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .data-table th {{
      padding: 8px 12px; text-align: left;
      background: var(--surface); color: var(--text-2);
      font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.7px;
      border-bottom: 1px solid var(--border); position: sticky; top: 48px; z-index: 10;
    }}
    .data-table td {{
      padding: 9px 12px; border-bottom: 1px solid var(--border);
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    .data-table tr:last-child td {{ border-bottom: none; }}
    .data-table tbody tr {{ cursor: pointer; transition: background 0.1s; }}
    .data-table tbody tr:hover {{ background: var(--surface); }}
    .data-table tbody tr.selected {{ background: rgba(91,140,255,0.08); border-left: 2px solid var(--cobalt); }}

    /* ── Claimed vs Observed ────────────────────────────────────────── */
    .cvo-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    @media (max-width: 768px) {{ .cvo-grid {{ grid-template-columns: 1fr; }} }}
    .cvo-panel {{
      border-radius: 6px; overflow: hidden;
      border: 1px solid var(--border);
    }}
    .cvo-header {{
      padding: 8px 12px;
      font-size: 10px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
      border-bottom: 1px solid var(--border);
    }}
    .cvo-claimed .cvo-header {{ background: rgba(91,140,255,0.08); color: var(--cobalt); }}
    .cvo-observed .cvo-header {{ background: rgba(76,195,138,0.06); color: var(--verified); }}
    .cvo-observed.mismatch .cvo-header {{ background: rgba(229,100,91,0.08); color: var(--failed); }}
    .cvo-body {{ padding: 12px; background: var(--panel); }}
    .cvo-verdict {{
      font-size: 18px; font-weight: 800; font-family: var(--mono);
      margin-bottom: 8px;
    }}
    .cvo-verdict.pass {{ color: var(--verified); }}
    .cvo-verdict.fail {{ color: var(--failed); }}
    .cvo-verdict.skip {{ color: var(--text-3); }}
    .cvo-discrepancy {{
      margin-top: 8px; padding: 8px 10px;
      background: rgba(229,100,91,0.07); border-left: 2px solid var(--failed);
      border-radius: 0 4px 4px 0; font-size: 11.5px; color: var(--text-2);
      overflow-wrap: anywhere;
    }}

    /* ── Timeline ───────────────────────────────────────────────────── */
    #timeline-list {{ list-style: none; padding: 0; }}
    #timeline-list li {{
      display: flex; align-items: flex-start; gap: 10px;
      padding: 7px 10px; border-radius: 5px;
      cursor: pointer; transition: background 0.1s;
      border: 1px solid transparent; margin-bottom: 3px;
    }}
    #timeline-list li:hover {{ background: var(--surface); }}
    #timeline-list li.selected {{
      background: rgba(91,140,255,0.08);
      border-color: var(--cobalt-dim);
    }}
    .tl-seq {{
      min-width: 22px; height: 22px; border-radius: 50%;
      background: var(--surface); border: 1px solid var(--border);
      display: flex; align-items: center; justify-content: center;
      font-size: 10px; font-weight: 700; color: var(--text-2);
      flex-shrink: 0; margin-top: 1px;
    }}
    li.selected .tl-seq {{
      background: var(--cobalt-dim); border-color: var(--cobalt); color: var(--cobalt);
    }}
    .tl-phase {{
      font-family: var(--mono); font-size: 11px; font-weight: 600;
      color: var(--text); line-height: 1.4;
    }}
    .tl-time {{
      font-size: 10px; color: var(--text-3); margin-top: 1px; font-family: var(--mono);
      overflow-wrap: anywhere;
    }}

    /* ── Step Inspector ─────────────────────────────────────────────── */
    #step-inspector {{
      min-height: 120px;
      display: flex; flex-direction: column; gap: 6px;
    }}
    .inspector-empty {{
      color: var(--text-3); font-size: 12px; text-align: center;
      padding: 24px 0; font-style: italic;
    }}

    /* ── Contract Predicates ────────────────────────────────────────── */
    .predicate-row {{
      display: grid;
      grid-template-columns: 32px 140px 180px 1fr 60px;
      gap: 6px 12px; align-items: start;
      padding: 8px 0; border-bottom: 1px solid var(--border);
      font-size: 11.5px;
      overflow-wrap: anywhere;
    }}
    .predicate-row:last-child {{ border-bottom: none; }}
    .predicate-row.passed .pred-result {{ color: var(--verified); }}
    .predicate-row.failed .pred-result {{ color: var(--failed); }}
    @media (max-width: 768px) {{
      .predicate-row {{
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 10px 0;
      }}
    }}
    .pred-name {{ font-family: var(--mono); font-size: 11px; color: var(--cobalt); overflow-wrap: anywhere; }}
    .pred-target {{ font-family: var(--mono); font-size: 11px; color: var(--text-2); overflow-wrap: anywhere; }}
    .pred-expected {{ color: var(--text-2); overflow-wrap: anywhere; }}
    .pred-evidence {{ font-size: 11px; color: var(--text-3); overflow-wrap: anywhere; }}
    .pred-result {{ font-weight: 700; font-size: 11px; font-family: var(--mono); }}

    /* ── Run Selector ───────────────────────────────────────────────── */
    #run-selector-list {{ list-style: none; padding: 0; display: flex; flex-direction: column; gap: 3px; }}
    #run-selector-list li {{
      padding: 9px 12px; border-radius: 5px; cursor: pointer;
      border: 1px solid var(--border); background: var(--panel);
      transition: background 0.1s, border-color 0.1s;
    }}
    #run-selector-list li:hover {{ background: var(--surface); }}
    #run-selector-list li.active {{
      background: rgba(91,140,255,0.09);
      border-color: var(--cobalt-dim);
    }}
    #run-selector-list li:focus-visible {{ outline: 2px solid var(--cobalt); outline-offset: 1px; }}
    .run-item-name {{ font-size: 12px; font-weight: 600; color: var(--text); }}
    .run-item-meta {{ font-size: 10px; color: var(--text-3); margin-top: 2px; display: flex; gap: 8px; }}

    /* ── Inspector Layout ───────────────────────────────────────────── */
    #inspector-layout {{
      display: grid;
      grid-template-columns: 240px 1fr;
      gap: 16px;
      align-items: start;
    }}
    @media (max-width: 768px) {{
      #inspector-layout {{ grid-template-columns: 1fr; }}
    }}

    /* ── Benchmark Lab ──────────────────────────────────────────────── */
    .scenario-pair {{
      border: 1px solid var(--border); border-radius: 8px;
      overflow: hidden; margin-bottom: 12px;
    }}
    .scenario-pair-hdr {{
      padding: 10px 16px;
      background: var(--surface);
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      border-bottom: 1px solid var(--border);
    }}
    .scenario-pair-body {{ display: grid; grid-template-columns: 1fr 1fr; }}
    @media (max-width: 768px) {{ .scenario-pair-body {{ grid-template-columns: 1fr; }} }}
    .system-col {{
      padding: 12px 16px;
    }}
    .system-col:first-child {{ border-right: 1px solid var(--border); }}
    @media (max-width: 768px) {{ .system-col:first-child {{ border-right: none; border-bottom: 1px solid var(--border); }} }}
    .system-col-header {{
      font-size: 10px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;
      margin-bottom: 10px;
    }}
    .system-col.reagent-col .system-col-header {{ color: var(--cobalt); }}
    .system-col.baseline-col .system-col-header {{ color: var(--text-3); }}

    /* ── Summary Bar ────────────────────────────────────────────────── */
    .summary-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px; margin-bottom: 20px;
    }}
    .summary-stat {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 7px; padding: 12px 14px;
    }}
    .summary-stat-val {{
      font-size: 26px; font-weight: 800; font-family: var(--mono); line-height: 1.1;
      margin-bottom: 4px;
    }}
    .summary-stat-val.v {{ color: var(--verified); }}
    .summary-stat-val.e {{ color: var(--escalated); }}
    .summary-stat-val.f {{ color: var(--failed); }}
    .summary-stat-val.c {{ color: var(--cobalt); }}
    .summary-stat-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.7px; color: var(--text-2); }}
    .summary-stat-sub {{ font-size: 10px; color: var(--text-3); margin-top: 3px; }}

    /* ── Sections ───────────────────────────────────────────────────── */
    .section {{ margin-bottom: 20px; }}
    .section-hdr {{
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 10px;
    }}

    /* ── Grid helpers ───────────────────────────────────────────────── */
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .gap-12 {{ gap: 12px; display: flex; flex-direction: column; }}

    /* ── Utility ────────────────────────────────────────────────────── */
    .text-2  {{ color: var(--text-2); }}
    .text-3  {{ color: var(--text-3); }}
    .text-sm {{ font-size: 11.5px; }}
    .font-mono {{ font-family: var(--mono); font-size: 11px; overflow-wrap: anywhere; }}
    .mt-4  {{ margin-top: 4px; }}
    .mt-8  {{ margin-top: 8px; }}
    .mt-12 {{ margin-top: 12px; }}
    .mb-4  {{ margin-bottom: 4px; }}
    .mb-8  {{ margin-bottom: 8px; }}
    .mb-12 {{ margin-bottom: 12px; }}
    .flex  {{ display: flex; }}
    .flex-center {{ display: flex; align-items: center; }}
    .gap-8 {{ gap: 8px; }}
    .fw-600 {{ font-weight: 600; }}
    .note-box {{
      background: rgba(91,140,255,0.06); border: 1px solid var(--cobalt-dim);
      border-radius: 5px; padding: 10px 14px; font-size: 11.5px; color: var(--text-2);
    }}
    .warn-box {{
      background: rgba(227,169,75,0.07); border: 1px solid rgba(227,169,75,0.3);
      border-radius: 5px; padding: 10px 14px; font-size: 11.5px; color: var(--escalated);
    }}
    .method-section {{ margin-bottom: 28px; }}
    .method-section h3 {{ font-size: 14px; margin-bottom: 8px; color: var(--text); }}
    .method-section p, .method-section li {{ font-size: 12.5px; color: var(--text-2); line-height: 1.65; }}
    .method-section ul {{ padding-left: 20px; }}
    .method-section li {{ margin-bottom: 4px; }}
    pre.json-view {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 6px; padding: 12px; overflow-x: auto;
      font-size: 11px; line-height: 1.6; color: var(--text-2);
      white-space: pre-wrap; word-break: break-word; max-height: 340px; overflow-y: auto;
      overflow-wrap: anywhere;
    }}
    .copy-btn {{
      font-size: 10px; padding: 3px 8px; border-radius: 3px;
      background: var(--surface); border: 1px solid var(--border);
      color: var(--text-2); cursor: pointer; transition: color 0.15s;
    }}
    .copy-btn:hover {{ color: var(--cobalt); }}
    #toast {{
      position: fixed; bottom: 20px; right: 20px;
      background: var(--surface); border: 1px solid var(--border);
      color: var(--text); padding: 8px 14px; border-radius: 6px;
      font-size: 12px; opacity: 0; transition: opacity 0.2s;
      pointer-events: none; z-index: 999;
    }}
    #toast.show {{ opacity: 1; }}
  </style>
</head>
<body>
<div id="app">

  <!-- ── Top Bar ──────────────────────────────────────────────────────── -->
  <header id="topbar">
    <div class="wordmark">
      <div class="wordmark-logo">Re</div>
      <div>
        <div class="wordmark-name">ReAgent</div>
        <div class="wordmark-tagline">doesn't trust 'done.' It verifies it.</div>
      </div>
    </div>
    <div class="topbar-divider"></div>
    <span class="provenance-badge" id="provenance-badge">OFFLINE · DETERMINISTIC</span>
    <nav class="topbar-nav" role="navigation" aria-label="Primary navigation">
      <button class="nav-btn active" data-tab="runs" onclick="switchTab('runs')">Runs</button>
      <button class="nav-btn" data-tab="benchmark" onclick="switchTab('benchmark')">Benchmark Lab</button>
      <button class="nav-btn" data-tab="method" onclick="switchTab('method')">Method &amp; Limits</button>
    </nav>
    <div class="topbar-actions">
      <button class="action-btn" onclick="exportHTML()" title="Export self-contained HTML report">⬇ Export</button>
    </div>
  </header>

  <!-- ── Main ─────────────────────────────────────────────────────────── -->
  <main id="main" role="main">

    <!-- ════════════════════════════════════════════════════════════════
         TAB: RUNS
    ════════════════════════════════════════════════════════════════════ -->
    <div id="tab-runs" class="tab-panel active">

      <div id="inspector-layout">

        <!-- ── Left: Run Selector ─────────────────────────────────── -->
        <div>
          <div class="section-hdr mb-8">
            <h2>Recorded Runs</h2>
          </div>
          <ul id="run-selector-list" role="listbox" aria-label="Select a run to inspect"></ul>
        </div>

        <!-- ── Right: Run Inspector Panels ───────────────────────── -->
        <div class="gap-12" id="run-inspector-panels">

          <!-- Run Summary -->
          <div class="card" id="card-summary">
            <div class="card-header">
              <h3 id="run-title">Select a run</h3>
              <span id="run-status-pill"></span>
            </div>
            <div class="card-body">
              <div id="run-kv-rows"></div>
              <div class="note-box mt-8" id="provenance-note"></div>
            </div>
          </div>

          <!-- Claimed vs Observed -->
          <div class="card">
            <div class="card-header">
              <h3>Claimed vs. Observed</h3>
              <span class="pill cobalt" title="Tool exit status is not sufficient evidence of goal achievement">Why this matters</span>
            </div>
            <div class="card-body">
              <div class="cvo-grid" id="cvo-container"></div>
            </div>
          </div>

          <!-- Frozen Success Contract -->
          <div class="card">
            <div class="card-header">
              <h3>Frozen Success Contract</h3>
              <span id="contract-provenance-badge"></span>
            </div>
            <div class="card-body" id="contract-body"></div>
          </div>

          <!-- Predicate / Evidence Table -->
          <div class="card">
            <div class="card-header">
              <h3>Predicate Evidence Table</h3>
            </div>
            <div style="overflow-x:auto">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Predicate</th>
                    <th>Target</th>
                    <th>Expected</th>
                    <th>Evidence Source</th>
                    <th>Discrepancy</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody id="predicate-table-body"></tbody>
              </table>
            </div>
          </div>

          <!-- Execution Timeline + Step Inspector -->
          <div class="grid-2">
            <div class="card">
              <div class="card-header"><h3>Execution Timeline</h3></div>
              <div class="card-body" style="padding:10px 12px">
                <ul id="timeline-list" role="listbox" aria-label="Timeline events — click to inspect"></ul>
              </div>
            </div>
            <div class="card">
              <div class="card-header">
                <h3>Step Inspector</h3>
                <span class="text-3 text-sm" id="step-title">No step selected</span>
              </div>
              <div class="card-body">
                <div id="step-inspector">
                  <div class="inspector-empty">Select a timeline event to inspect its details.</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Receipt Viewer -->
          <div class="card">
            <div class="card-header">
              <h3>Structured Receipt</h3>
              <div class="flex gap-8">
                <button class="copy-btn" onclick="copyReceipt()">Copy JSON</button>
                <button class="copy-btn" onclick="toggleRawReceipt()">Toggle Raw</button>
              </div>
            </div>
            <div class="card-body" id="receipt-body"></div>
          </div>

        </div><!-- /run-inspector-panels -->
      </div><!-- /inspector-layout -->
    </div><!-- /tab-runs -->

    <!-- ════════════════════════════════════════════════════════════════
         TAB: BENCHMARK LAB
    ════════════════════════════════════════════════════════════════════ -->
    <div id="tab-benchmark" class="tab-panel">

      <div class="note-box mb-12">
        <strong>Benchmark scope:</strong> 5 seeded deterministic scenarios evaluated against the offline stand-in (<code>DevelopmentLLMProvider</code>).
        Baseline: an <strong>illustrative naive baseline</strong> executing fixed retries without diagnosis, fallback tool selection, or independent postcondition verification.
        Raw counts shown with exact denominators — no decorative percentages.
        <strong>Provenance: OFFLINE · DETERMINISTIC.</strong> No live API calls performed.
      </div>

      <!-- Summary Bar -->
      <div class="summary-bar" id="bench-summary-bar"></div>

      <!-- Head-to-head comparison table -->
      <div class="section">
        <div class="section-hdr">
          <h2>Head-to-Head: 5 Scenarios</h2>
        </div>
        <div style="overflow-x:auto">
          <table class="data-table">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Fault Category</th>
                <th>Expected Outcome</th>
                <th>ReAgent Outcome</th>
                <th>ReAgent Recovery</th>
                <th>Baseline Outcome</th>
                <th>Baseline Behavior</th>
                <th>Inspect</th>
              </tr>
            </thead>
            <tbody id="bench-table-body"></tbody>
          </table>
        </div>
      </div>

      <!-- Detailed scenario pairs -->
      <div class="section-hdr mt-12">
        <h2>Scenario Detail</h2>
      </div>
      <div id="scenario-pairs"></div>

    </div><!-- /tab-benchmark -->

    <!-- ════════════════════════════════════════════════════════════════
         TAB: METHOD & LIMITS
    ════════════════════════════════════════════════════════════════════ -->
    <div id="tab-method" class="tab-panel">
      <div style="max-width:760px">

        <div class="method-section">
          <h3>What ReAgent Verifies</h3>
          <p>ReAgent independently observes the outcome of a tool call and checks it against a
          <em>frozen success contract</em> — a set of typed postcondition predicates captured
          before execution starts. A tool returning exit code 0 or <code>success: true</code>
          is treated as a claim, not evidence.</p>
        </div>

        <div class="method-section">
          <h3>Why Tool Exit Status Is Not Sufficient</h3>
          <p>A tool can return success while the user's goal remains unmet — for example, if
          a required field is absent from the output, or if the tool performed a different action
          than intended. Without independent observation, these failures are invisible. ReAgent's
          deterministic verifier checks actual output keys, values, and file state against the
          frozen contract after every execution.</p>
        </div>

        <div class="method-section">
          <h3>Frozen Success Contracts</h3>
          <p>Before a tool is invoked, a <code>SuccessContract</code> is frozen from typed
          <code>Postcondition</code> predicates. The contract cannot be relaxed by the model
          during recovery. Supported predicates include:
          <code>KEY_EXISTS</code>, <code>VALUE_EQUALS</code>, <code>STATUS_SUCCESS</code>,
          <code>FILE_EXISTS</code>, <code>FILE_CONTAINS</code>, <code>JSON_VALID</code>.
          Predicate evaluation is deterministic Python code — no model involvement.</p>
        </div>

        <div class="method-section">
          <h3>Observation and Deterministic Verification</h3>
          <p>Evidence is collected from the actual tool output by <code>EvidenceCollector</code>.
          Each postcondition is evaluated by <code>DeterministicVerifier</code> against the
          collected evidence. The model has zero influence over the pass/fail verdict.</p>
        </div>

        <div class="method-section">
          <h3>Bounded Recovery</h3>
          <p>When verification fails, a failure is attributed to a category
          (<code>TRANSIENT_EXTERNAL</code>, <code>MODEL_OUTPUT</code>, <code>ENVIRONMENT_FAILURE</code>, <code>AUTHORIZATION</code>, etc.).
          A policy-approved recovery strategy is selected from an allowlist.
          The number of recovery attempts is bounded by <code>max_recovery_attempts</code>
          set before execution. If recovery is exhausted, the pipeline escalates —
          it never loops silently.</p>
        </div>

        <div class="method-section">
          <h3>Outcome Definitions</h3>
          <ul>
            <li><strong style="color:var(--verified)">VERIFIED</strong> — All required postconditions passed. Deterministic verifier confirmed goal state.</li>
            <li><strong style="color:var(--escalated)">ESCALATED</strong> — Recovery was attempted but exhausted or blocked by policy gate. The pipeline surfaced the failure explicitly rather than claiming success. Escalation is a clean, bounded terminal state; it is not counted as verification success or as an unhandled failure.</li>
            <li><strong style="color:var(--failed)">FAILED</strong> — Execution failed and no recovery was configured or applicable.</li>
            <li><strong style="color:var(--failed)">FALSE SUCCESS</strong> — Baseline-specific: the illustrative naive baseline trusted tool exit 0 without postcondition verification, incorrectly reporting success when the goal was not achieved.</li>
          </ul>
        </div>

        <div class="method-section">
          <h3>Benchmark Scenarios</h3>
          <ul>
            <li><strong>Scenario 1: Transient Tool Failure:</strong> Tool raises a 503-style error on the first call. ReAgent attributes TRANSIENT_EXTERNAL and applies WAIT_AND_RETRY. Expected terminal outcome: VERIFIED.</li>
            <li><strong>Scenario 2: Invalid Model Output:</strong> First call produces malformed JSON. ReAgent diagnoses MODEL_OUTPUT and retries with repaired input. Expected terminal outcome: VERIFIED.</li>
            <li><strong>Scenario 3: Incorrect Action Result (False-Success Detection):</strong> Tool returns success but a required key is absent. ReAgent verifies, detects discrepancy, and escalates cleanly. Baseline trusts exit 0 — FALSE SUCCESS. Expected terminal outcome: ESCALATED.</li>
            <li><strong>Scenario 4: Environment / Tool Dependency Failure:</strong> Requested tool is unregistered. ReAgent diagnoses ENVIRONMENT_FAILURE, selects a registered fallback tool. <em>Note: this scenario uses registry-level fallback execution rather than execute_with_contract(); contract and receipt are marked NOT APPLICABLE.</em> Expected terminal outcome: VERIFIED.</li>
            <li><strong>Scenario 5: Authentication Failure (Escalation):</strong> Unresolvable auth error (simulated 403). ReAgent escalates after policy gate blocks unsafe retry. Baseline burns retry budget blindly. Expected terminal outcome: ESCALATED.</li>
          </ul>
        </div>

        <div class="method-section">
          <h3>Baseline Description</h3>
          <p>The baseline is an <strong>illustrative naive baseline</strong>: it re-invokes the same
          tool with the same arguments up to a fixed number of times without root-cause diagnosis,
          modified input, fallback tool selection, or postcondition verification. It is used solely
          to demonstrate the failure modes that motivate ReAgent, not as a claim of superiority over
          any production agent framework.</p>
        </div>

        <div class="method-section">
          <div class="warn-box">
            <strong>Important limitations:</strong>
            <ul style="margin-top:6px; padding-left:18px">
              <li>All 5 scenarios are seeded and deterministic. Results are not from a random sample or statistical population.</li>
              <li>The <code>DevelopmentLLMProvider</code> is used for offline runs — no live model inference was performed.</li>
              <li>No live Nebius API calls were made in producing this benchmark.</li>
              <li>Scenario 4 uses a registry-level fallback, not a full pipeline cycle — contract and receipt pipeline is NOT APPLICABLE for this scenario.</li>
              <li>ESCALATED is a correct, bounded failure — it is not verified success. Do not interpret it as equivalent to VERIFIED.</li>
              <li>5 scenarios is an illustrative sample size (n=5). Raw counts and exact denominators are shown; no percentages are displayed.</li>
              <li>Trial credits and account balance do not guarantee that no payment card charge can occur for future API usage.</li>
            </ul>
          </div>
        </div>

        <div class="method-section">
          <h3>What This Implementation Does Not Prove</h3>
          <ul>
            <li>That the live Nebius Nemotron model integration functions end-to-end at scale (no live runs in this benchmark).</li>
            <li>That ReAgent is production-ready or independently validated across arbitrary real-world tasks.</li>
            <li>That results will generalize beyond these specific seeded scenarios.</li>
            <li>That ESCALATED outcomes are optimal — they represent cases where recovery could not succeed, handled safely.</li>
          </ul>
        </div>

      </div>
    </div><!-- /tab-method -->

  </main>
</div>

<div id="toast"></div>

<script>
// ── Data ─────────────────────────────────────────────────────────────────
const REPORT = {report_json_safe};

// ── State ─────────────────────────────────────────────────────────────────
let selectedRunIdx = 0;
let selectedStepIdx = null;
let receiptRawMode = false;

// ── Utilities ─────────────────────────────────────────────────────────────
function esc(s) {{
  return String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}
function statusPill(s, icon, label) {{
  return `<span class="pill ${{esc(s)}}" title="${{esc(label)}}">${{esc(icon)}} ${{esc(label)}}</span>`;
}}
function toast(msg) {{
  const el = document.getElementById('toast');
  el.textContent = msg; el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 1800);
}}

// ── Tab Switching ─────────────────────────────────────────────────────────
function switchTab(tab) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  document.querySelector(`[data-tab="${{tab}}"]`).classList.add('active');
}}

// ── Run Selector ──────────────────────────────────────────────────────────
function buildRunSelector() {{
  const list = document.getElementById('run-selector-list');
  const reagentRuns = REPORT.runs.filter(r => r.strategy === 'reagent_diagnosis_guided');
  list.innerHTML = '';
  reagentRuns.forEach((run, i) => {{
    const li = document.createElement('li');
    li.tabIndex = 0;
    li.setAttribute('role', 'option');
    li.setAttribute('aria-selected', i === selectedRunIdx ? 'true' : 'false');
    li.innerHTML = `
      <div class="run-item-name">${{esc(run.scenario_name)}}</div>
      <div class="run-item-meta">
        <span>${{esc(run.fault_type)}}</span>
        ${{statusPill(run.status_meta.css, run.status_meta.icon, run.status_meta.label)}}
        <span>${{run.duration_ms}}ms</span>
      </div>`;
    li.addEventListener('click', () => selectRun(i));
    li.addEventListener('keydown', e => {{ if (e.key === 'Enter' || e.key === ' ') selectRun(i); }});
    list.appendChild(li);
  }});
}}

function selectRun(i) {{
  selectedRunIdx = i;
  selectedStepIdx = null;
  const items = document.querySelectorAll('#run-selector-list li');
  items.forEach((li, j) => {{
    li.classList.toggle('active', j === i);
    li.setAttribute('aria-selected', j === i ? 'true' : 'false');
  }});
  renderRunInspector();
}}

// ── Run Inspector ─────────────────────────────────────────────────────────
function renderRunInspector() {{
  const reagentRuns = REPORT.runs.filter(r => r.strategy === 'reagent_diagnosis_guided');
  const run = reagentRuns[selectedRunIdx];
  if (!run) return;

  // Summary
  document.getElementById('run-title').textContent = run.scenario_name;
  document.getElementById('run-status-pill').innerHTML =
    statusPill(run.status_meta.css, run.status_meta.icon, run.status_meta.label);
  document.getElementById('provenance-note').textContent = REPORT.provenance_note;

  let kvHtml = '';
  kvHtml += kv('Task Objective', run.scenario_name);
  kvHtml += kv('Scenario ID', run.scenario_id, true);
  kvHtml += kv('Fault Category', run.fault_type, true);
  kvHtml += kv('Strategy', run.strategy_label);
  kvHtml += kv('Expected Outcome', run.expected_terminal_outcome || 'Not recorded', true);
  kvHtml += kv('Attempts Executed', run.attempts_executed);
  kvHtml += kv('Duration', run.duration_ms + ' ms');
  kvHtml += kv('Provenance', REPORT.provenance);
  kvHtml += kv('Diagnosed Failure', run.diagnosed_failure_type, true);
  kvHtml += kv('Selected Recovery', run.selected_recovery, true);
  document.getElementById('run-kv-rows').innerHTML = kvHtml;

  // Claimed vs Observed
  renderCVO(run);

  // Contract
  renderContract(run);

  // Predicate Table
  renderPredicateTable(run);

  // Timeline
  renderTimeline(run);

  // Receipt
  renderReceipt(run);
}}

function kv(key, val, mono=false) {{
  return `<div class="kv-row">
    <span class="kv-key">${{esc(key)}}</span>
    <span class="kv-val ${{mono ? 'mono' : ''}}">${{esc(val)}}</span>
  </div>`;
}}

// ── Claimed vs Observed ───────────────────────────────────────────────────
function renderCVO(run) {{
  const co = run.claimed_outcome;
  const oo = run.observed_outcome;
  const mismatch = oo && oo.verifier_verdict !== 'PASS';
  const mismatchClass = mismatch ? 'mismatch' : '';

  let claimedHtml = '<em class="text-3">Not recorded</em>';
  if (co) {{
    const success = co.tool_reported_success;
    claimedHtml = `
      <div class="cvo-verdict ${{success ? 'pass' : 'fail'}}">${{success ? 'SUCCESS' : 'FAILURE'}}</div>
      ${{kv('Tool', co.tool_name || 'Not recorded')}}
      ${{co.output_keys_present && co.output_keys_present.length > 0
        ? kv('Output Keys', co.output_keys_present.join(', '), true)
        : '<div class="kv-row"><span class="kv-key">Output Keys</span><span class="kv-val text-3">None / not recorded</span></div>'
      }}
      ${{co.error ? `<div class="kv-row"><span class="kv-key">Error</span><span class="kv-val text-3">${{esc(co.error)}}</span></div>` : ''}}
    `;
  }}

  let observedHtml = '<em class="text-3">Not recorded</em>';
  if (oo) {{
    const verdict = oo.verifier_verdict || 'UNKNOWN';
    const verdictClass = verdict === 'PASS' ? 'pass' : verdict === 'SKIPPED' ? 'skip' : 'fail';
    observedHtml = `
      <div class="cvo-verdict ${{verdictClass}}">${{esc(verdict)}}</div>
      <div class="kv-row">
        <span class="kv-key">Evidence Source</span>
        <span class="kv-val mono">${{esc(oo.evidence_source || 'not recorded')}}</span>
      </div>
      ${{oo.evidence_summary ? kv('Evidence Summary', oo.evidence_summary) : ''}}
      ${{(oo.discrepancies || []).map(d =>
        `<div class="cvo-discrepancy">${{esc(d)}}</div>`
      ).join('')}}
    `;
  }}

  document.getElementById('cvo-container').innerHTML = `
    <div class="cvo-panel cvo-claimed">
      <div class="cvo-header">← CLAIMED — What tool reported</div>
      <div class="cvo-body">${{claimedHtml}}</div>
    </div>
    <div class="cvo-panel cvo-observed ${{mismatchClass}}">
      <div class="cvo-header">${{mismatch ? '⚡ OBSERVED — Discrepancy detected' : '→ OBSERVED — Independently verified'}}</div>
      <div class="cvo-body">${{observedHtml}}</div>
    </div>
  `;
}}

// ── Contract Panel ────────────────────────────────────────────────────────
function renderContract(run) {{
  if (run.contract_applicable === false) {{
    document.getElementById('contract-body').innerHTML =
      '<div class="warn-box"><strong>NOT APPLICABLE:</strong> this scenario does not use the contract pipeline; timeline is fixture-derived.</div>';
    document.getElementById('contract-provenance-badge').innerHTML =
      '<span class="pill neutral">contract: not applicable</span>';
    return;
  }}

  const fc = run.frozen_contract_summary;
  if (!fc) {{
    document.getElementById('contract-body').innerHTML = '<em class="text-3">Not recorded</em>';
    document.getElementById('contract-provenance-badge').innerHTML = '';
    return;
  }}
  document.getElementById('contract-provenance-badge').innerHTML =
    `<span class="pill neutral">provenance: ${{esc(fc.provenance || 'not recorded')}}</span>`;

  let html = '';
  html += kv('Contract ID', fc.contract_id || 'Not recorded', true);
  html += kv('Frozen At', fc.frozen_at || 'Not recorded');
  html += kv('Provenance', fc.provenance || 'Not recorded');
  html += kv('Hash Recorded', fc.hash_recorded ? 'Yes' : 'No — hash not computed in current implementation');

  if (fc.predicates && fc.predicates.length) {{
    html += `<div class="card-section-label mt-12">Predicates (${{fc.predicates.length}})</div>`;
    fc.predicates.forEach((p, i) => {{
      html += `<div class="kv-row">
        <span class="kv-key text-3">#${{i+1}}</span>
        <span class="kv-val">
          <code class="font-mono" style="color:var(--cobalt)">${{esc(p.predicate)}}</code>
          ${{p.target ? `<span class="text-3"> → ${{esc(p.target)}}</span>` : ''}}
          ${{p.expected !== null && p.expected !== undefined ? `<span class="text-3"> = ${{esc(p.expected)}}</span>` : ''}}
          ${{!p.required ? '<span class="pill neutral" style="font-size:9px;margin-left:4px">optional</span>' : ''}}
        </span>
      </div>`;
    }});
  }}

  document.getElementById('contract-body').innerHTML = html;
}}

// ── Predicate Table ───────────────────────────────────────────────────────
function renderPredicateTable(run) {{
  const tbody = document.getElementById('predicate-table-body');

  if (run.contract_applicable === false) {{
    tbody.innerHTML = '<tr><td colspan="7" class="text-3" style="padding:16px;text-align:center">NOT APPLICABLE: this scenario does not use the contract pipeline; timeline is fixture-derived.</td></tr>';
    return;
  }}

  const fc = run.frozen_contract_summary;
  const oo = run.observed_outcome;

  if (!fc || !fc.predicates || !fc.predicates.length) {{
    tbody.innerHTML = '<tr><td colspan="7" class="text-3" style="padding:16px;text-align:center">No predicate data recorded for this run.</td></tr>';
    return;
  }}

  const discrepancies = (oo && oo.discrepancies) || run.discrepancies || [];
  const evidenceSource = (oo && oo.evidence_source) || 'deterministic_verifier';

  tbody.innerHTML = fc.predicates.map((p, i) => {{
    const disc = discrepancies.find(d => p.target && d.toLowerCase().includes(p.target.toLowerCase()));
    const passed = !disc && run.verified;
    const rowClass = passed ? 'passed' : 'failed';

    return `<tr class="${{rowClass}}">
      <td class="text-3">${{i+1}}</td>
      <td><code class="font-mono" style="color:var(--cobalt)">${{esc(p.predicate)}}</code></td>
      <td><code class="font-mono text-2">${{esc(p.target || '—')}}</code></td>
      <td class="text-2">${{esc(p.expected !== null && p.expected !== undefined ? p.expected : '—')}}</td>
      <td class="text-3 font-mono">${{esc(evidenceSource)}}</td>
      <td class="text-3">${{esc(disc || (passed ? '—' : 'See discrepancies'))}}</td>
      <td class="pred-result">${{passed ? '✓ PASS' : '✗ FAIL'}}</td>
    </tr>`;
  }}).join('');
}}

// ── Timeline ──────────────────────────────────────────────────────────────
function renderTimeline(run) {{
  const events = run.timeline_events && run.timeline_events.length > 0
    ? run.timeline_events
    : run.timeline_phases.map((ph, i) => ({{event_id: 'ph-'+i, phase: ph, occurred_at: null, detail: {{}}}}));

  const ul = document.getElementById('timeline-list');
  ul.innerHTML = events.map((ev, i) => `
    <li role="option" tabindex="0" data-idx="${{i}}" aria-selected="false"
        onclick="selectStep(${{i}})" onkeydown="if(event.key==='Enter'||event.key===' ')selectStep(${{i}})">
      <div class="tl-seq">${{i+1}}</div>
      <div>
        <div class="tl-phase">${{esc(ev.phase)}}</div>
        ${{ev.occurred_at ? `<div class="tl-time">${{esc(ev.occurred_at)}}</div>` : ''}}
      </div>
    </li>
  `).join('');
}}

function selectStep(i) {{
  selectedStepIdx = i;
  const reagentRuns = REPORT.runs.filter(r => r.strategy === 'reagent_diagnosis_guided');
  const run = reagentRuns[selectedRunIdx];
  const events = run.timeline_events && run.timeline_events.length > 0
    ? run.timeline_events
    : run.timeline_phases.map((ph, j) => ({{event_id: 'ph-'+j, phase: ph, occurred_at: null, detail: {{}}}}));

  document.querySelectorAll('#timeline-list li').forEach((li, j) => {{
    li.classList.toggle('selected', j === i);
    li.setAttribute('aria-selected', j === i ? 'true' : 'false');
  }});

  const ev = events[i];
  document.getElementById('step-title').textContent = ev.phase;

  const PHASE_DESCRIPTIONS = {{
    TASK_RECEIVED:       'Pipeline received the task objective and initialized the observability timeline.',
    CONTEXT_SELECTED:    'Context selector identified relevant historical context for this task.',
    MODEL_DECISION:      'Provider produced a structured decision (tool + arguments + proposed postconditions).',
    DECISION_VALIDATED:  'Schema validation: tool exists in allowlist, arguments conform to typed schema.',
    CONTRACT_FROZEN:     'SuccessContract frozen before tool execution. Cannot be relaxed by model.',
    POLICY_ALLOWED:      'PolicyGuard approved tool invocation: no violations detected.',
    TOOL_EXECUTED:       'Allowlisted tool was invoked with validated parameters.',
    EVIDENCE_COLLECTED:  'EvidenceCollector gathered facts from actual tool output.',
    VERIFICATION_FAILED: 'DeterministicVerifier found at least one postcondition unmet.',
    FAILURE_ATTRIBUTED:  'Failure categorized by deterministic rules (not model inference).',
    RECOVERY_SELECTED:   'Policy-approved recovery strategy selected based on failure category.',
    RECOVERY_ALLOWED:    'PolicyGuard approved recovery action.',
    RECOVERY_EXECUTED:   'Recovery tool invocation attempted.',
    REVERIFICATION:      'Verifier re-checked postconditions after recovery.',
    VERIFIED_RECEIPT:    'Immutable VerifiedReceipt emitted. Final verdict recorded.',
    TOOL_UNREGISTERED:   'Tool was not found in allowlisted registry; triggered fallback attribution.',
    FALLBACK_SELECTED:   'Allowlisted fallback tool selected to perform task.',
    RECOVERY_VERIFIED:   'Fallback tool completed task successfully.',
  }};

  const desc = PHASE_DESCRIPTIONS[ev.phase] || '';
  let kvHtml = '';
  kvHtml += kv('Phase', ev.phase, true);
  kvHtml += kv('Occurred At', ev.occurred_at || 'Not recorded');
  if (desc) kvHtml += kv('Description', desc);
  if (ev.detail && Object.keys(ev.detail).length) {{
    Object.entries(ev.detail).forEach(([k, v]) => {{
      const display = Array.isArray(v) ? v.join(', ') : String(v);
      kvHtml += kv(k, display, typeof v === 'string' && v.length < 80);
    }});
  }} else {{
    kvHtml += '<div class="kv-row"><span class="kv-key">Detail</span><span class="kv-val text-3">No additional detail recorded for this phase.</span></div>';
  }}

  document.getElementById('step-inspector').innerHTML = kvHtml;
}}

// ── Receipt ───────────────────────────────────────────────────────────────
function renderReceipt(run) {{
  if (run.contract_applicable === false) {{
    document.getElementById('receipt-body').innerHTML =
      '<div class="warn-box"><strong>NOT APPLICABLE:</strong> this scenario does not use the contract pipeline; timeline is fixture-derived.</div>';
    return;
  }}

  const r = run.safe_receipt;
  if (!r) {{
    document.getElementById('receipt-body').innerHTML = '<em class="text-3">Not recorded</em>';
    return;
  }}
  receiptRawMode = false;
  showReceiptFormatted(r, run);
}}

function showReceiptFormatted(r, run) {{
  const run_ = run || REPORT.runs.filter(x => x.strategy === 'reagent_diagnosis_guided')[selectedRunIdx];
  let html = '';
  html += kv('Receipt ID', r.receipt_id || 'Not recorded', true);
  html += kv('Task ID', r.task_id || 'Not recorded', true);
  html += kv('Timestamp', r.timestamp || 'Not recorded');
  html += kv('Tool', r.tool || 'Not recorded');
  html += kv('Final Status', r.final_status || 'Not recorded');
  html += kv('Verified', r.verified ? 'true' : 'false', true);
  html += kv('Recovery Count', String(r.recovery_count ?? 'Not recorded'));
  html += kv('Provenance', r.provenance || 'Not recorded');
  html += kv('Evidence Summary', r.evidence_summary || 'Not recorded');
  if (r.discrepancies && r.discrepancies.length) {{
    html += `<div class="card-section-label mt-12">Discrepancies</div>`;
    r.discrepancies.forEach(d => {{
      html += `<div class="cvo-discrepancy">${{esc(d)}}</div>`;
    }});
  }}
  document.getElementById('receipt-body').innerHTML = html;
}}

function toggleRawReceipt() {{
  const reagentRuns = REPORT.runs.filter(r => r.strategy === 'reagent_diagnosis_guided');
  const run = reagentRuns[selectedRunIdx];
  if (!run || run.contract_applicable === false) return;
  const r = run.safe_receipt;
  if (!r) return;
  receiptRawMode = !receiptRawMode;
  if (receiptRawMode) {{
    document.getElementById('receipt-body').innerHTML =
      `<pre class="json-view">${{esc(JSON.stringify(r, null, 2))}}</pre>`;
  }} else {{
    showReceiptFormatted(r, run);
  }}
}}

function copyReceipt() {{
  const reagentRuns = REPORT.runs.filter(r => r.strategy === 'reagent_diagnosis_guided');
  const run = reagentRuns[selectedRunIdx];
  if (!run || run.contract_applicable === false) {{ toast('Receipt not applicable for this run'); return; }}
  const r = run.safe_receipt;
  if (!r) {{ toast('No receipt to copy'); return; }}
  navigator.clipboard.writeText(JSON.stringify(r, null, 2))
    .then(() => toast('Receipt JSON copied'))
    .catch(() => toast('Copy failed — use browser context menu'));
}}

// ── Benchmark Lab ─────────────────────────────────────────────────────────
function buildBenchmarkLab() {{
  const sumBar = document.getElementById('bench-summary-bar');
  const r = REPORT;
  sumBar.innerHTML = `
    <div class="summary-stat">
      <div class="summary-stat-val c">${{r.scenarios_evaluated}}</div>
      <div class="summary-stat-label">Scenarios Evaluated</div>
      <div class="summary-stat-sub">Seeded · Deterministic</div>
    </div>
    <div class="summary-stat">
      <div class="summary-stat-val v">${{r.reagent_completed_and_verified}} / ${{r.expected_verifiable_count}}</div>
      <div class="summary-stat-label">Completed &amp; Verified</div>
      <div class="summary-stat-sub">Expected-verifiable</div>
    </div>
    <div class="summary-stat">
      <div class="summary-stat-val e">${{r.reagent_correctly_escalated}} / ${{r.expected_escalate_count}}</div>
      <div class="summary-stat-label">Correctly Escalated</div>
      <div class="summary-stat-sub">Expected-escalate</div>
    </div>
    <div class="summary-stat">
      <div class="summary-stat-val v">${{r.reagent_false_success_count}} / ${{r.scenarios_evaluated}}</div>
      <div class="summary-stat-label">ReAgent False Successes</div>
      <div class="summary-stat-sub">Verified = 0</div>
    </div>
    <div class="summary-stat">
      <div class="summary-stat-val f">${{r.baseline_false_success_count}} / ${{r.scenarios_evaluated}}</div>
      <div class="summary-stat-label">Baseline False Successes</div>
      <div class="summary-stat-sub">Illustrative naive baseline</div>
    </div>
  `;

  const reagentRuns = REPORT.runs.filter(r => r.strategy === 'reagent_diagnosis_guided');
  const baselineRuns = REPORT.runs.filter(r => r.strategy === 'naive_fixed_retry');

  const tbody = document.getElementById('bench-table-body');
  tbody.innerHTML = reagentRuns.map((rr, i) => {{
    const br = baselineRuns[i];
    const rSm = rr.status_meta;
    const bSm = br ? br.status_meta : {{css:'unknown',icon:'?',label:'?'}};
    return `<tr>
      <td><strong>${{esc(rr.scenario_name)}}</strong></td>
      <td><span class="pill cobalt">${{esc(rr.fault_type)}}</span></td>
      <td><span class="pill neutral">${{esc(rr.expected_terminal_outcome)}}</span></td>
      <td>${{statusPill(rSm.css, rSm.icon, rSm.label)}}</td>
      <td><code class="font-mono">${{esc(rr.selected_recovery)}}</code></td>
      <td>${{br ? statusPill(bSm.css, bSm.icon, bSm.label) : '—'}}</td>
      <td><code class="font-mono">${{br ? esc(br.selected_recovery) : '\u2014'}}</code></td>
      <td><button class="action-btn" onclick="inspectRun(${{i}})">Inspect →</button></td>
    </tr>`;
  }}).join('');

  const pairsEl = document.getElementById('scenario-pairs');
  pairsEl.innerHTML = reagentRuns.map((rr, i) => {{
    const br = baselineRuns[i];
    return `
    <div class="scenario-pair">
      <div class="scenario-pair-hdr">
        <div>
          <strong>${{esc(rr.scenario_name)}}</strong>
          <span class="pill cobalt" style="margin-left:8px">${{esc(rr.fault_type)}}</span>
          <span class="pill neutral" style="margin-left:4px">expected: ${{esc(rr.expected_terminal_outcome)}}</span>
        </div>
        <button class="action-btn" onclick="inspectRun(${{i}})">Inspect Run →</button>
      </div>
      <div class="scenario-pair-body">
        <div class="system-col reagent-col">
          <div class="system-col-header">ReAgent</div>
          ${{kv('Outcome', rr.status_meta.label)}}
          ${{kv('Strategy', rr.strategy_label)}}
          ${{kv('Recovery', rr.selected_recovery)}}
          ${{kv('Attempts', rr.attempts_executed)}}
          ${{kv('Duration', rr.duration_ms + ' ms')}}
          ${{rr.contract_applicable === false ? '<div class="warn-box mt-4">NOT APPLICABLE: this scenario does not use the contract pipeline; timeline is fixture-derived.</div>' : ''}}
          ${{rr.discrepancies && rr.discrepancies.length
            ? rr.discrepancies.map(d => `<div class="cvo-discrepancy mt-4">${{esc(d)}}</div>`).join('')
            : (rr.contract_applicable !== false ? '<div class="kv-row text-3" style="font-size:11px">No discrepancies recorded</div>' : '')
          }}
        </div>
        ${{br ? `
        <div class="system-col baseline-col">
          <div class="system-col-header">Illustrative Naive Baseline</div>
          ${{kv('Outcome', br.status_meta.label)}}
          ${{kv('Strategy', 'Illustrative Naive Baseline')}}
          ${{kv('Behavior', br.selected_recovery)}}
          ${{kv('Attempts', br.attempts_executed)}}
          ${{kv('Duration', br.duration_ms + ' ms')}}
          ${{br.discrepancies && br.discrepancies.length
            ? br.discrepancies.map(d => `<div class="cvo-discrepancy mt-4">${{esc(d)}}</div>`).join('')
            : ''
          }}
        </div>` : ''}}
      </div>
    </div>`;
  }}).join('');
}}

function inspectRun(i) {{
  selectRun(i);
  switchTab('runs');
}}

// ── Export ─────────────────────────────────────────────────────────────────
function exportHTML() {{
  try {{
    const blob = new Blob([document.documentElement.outerHTML], {{type: 'text/html;charset=utf-8'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'reagent_dashboard.html';
    a.click();
    URL.revokeObjectURL(a.href);
    toast('Dashboard exported');
  }} catch(e) {{
    toast('Export failed: ' + e.message);
  }}
}}

// ── Init ──────────────────────────────────────────────────────────────────
(function init() {{
  buildRunSelector();
  selectRun(0);
  buildBenchmarkLab();
}})();
</script>
</body>
</html>"""


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler. Runs benchmark on every request for accurate offline demo."""

    def log_message(self, format, *args):
        """Suppress per-request logs; only show server-level messages."""
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                report = run_benchmark_scenarios()
                html_str = generate_dashboard_html(report)
                body = html_str.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                err = f"Dashboard error: {type(exc).__name__}".encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)

        elif self.path == "/api/benchmark":
            try:
                report = run_benchmark_scenarios()
                body = report.model_dump_json().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                err = f'{{"error": "{type(exc).__name__}"}}'.encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)

        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8080):
    server = HTTPServer(("127.0.0.1", port), DashboardHTTPHandler)
    print(f"\n[DASHBOARD] ReAgent Judge UI -> http://127.0.0.1:{port}/")
    print("[DASHBOARD] Provenance: OFFLINE · DETERMINISTIC -- no live API calls.")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[DASHBOARD] Stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ReAgent Judge UI Server (M6.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve UI (default 8080)")
    parser.add_argument("--export-html", type=str, default=None,
                        help="Export self-contained HTML dashboard to specified path")
    args = parser.parse_args()

    if args.export_html:
        rep = run_benchmark_scenarios()
        html_content = generate_dashboard_html(rep)
        out_path = os.path.abspath(args.export_html)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[EXPORT] Static dashboard written to: {out_path}")
        print("[EXPORT] Provenance: OFFLINE · DETERMINISTIC -- all data from deterministic benchmark.")
    else:
        start_server(args.port)
