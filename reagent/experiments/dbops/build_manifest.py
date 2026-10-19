"""
build_submission_manifest.py — Generates submission_manifest.json with exact SHA-256 hashes.
"""

import hashlib
import json
import time
from pathlib import Path

FILES_TO_INDEX = [
    "reports/dbops/submission/README.md",
    "reports/dbops/submission/problem_statement.md",
    "reports/dbops/submission/architecture.md",
    "reports/dbops/submission/evaluation.md",
    "reports/dbops/submission/demo_script.md",
    "reports/dbops/submission/limitations.md",
    "reports/dbops/recovery_demo_artifact.json",
    "reports/dbops/comparative_evaluation_results.json",
    "reports/dbops/protocol_config_snapshot.json",
    "reagent/experiments/dbops/judge_demo.py",
    "reagent/experiments/dbops/recovery_demo.py",
    "reagent/experiments/dbops/comparative_eval.py",
]

def build_manifest(output_path: Path = Path("reports/dbops/submission/submission_manifest.json")) -> dict:
    manifest = {
        "submission_title": "ReAgent: AI Agent Reliability & Verification Layer",
        "product_tagline": "ReAgent doesn't trust 'done.' It verifies it.",
        "source_git_revision": "e1292b57abcccda57c0070fdbbe6a80bae1b8834",
        "manifest_version": "1.3.0-dbops-preregistered",
        "generated_at_utc": time.time(),
        "provenance": "SYNTHETIC_DETERMINISTIC_EVIDENCE",
        "artifacts": {}
    }

    for rel in FILES_TO_INDEX:
        p = Path(rel)
        if p.exists():
            content = p.read_bytes()
            h = hashlib.sha256(content).hexdigest()
            manifest["artifacts"][rel] = {
                "size_bytes": p.stat().st_size,
                "sha256": h,
                "status": "VERIFIED"
            }
        else:
            manifest["artifacts"][rel] = {
                "status": "MISSING"
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest

if __name__ == "__main__":
    m = build_manifest()
    print(f"Manifest written with {len(m['artifacts'])} indexed artifacts.")
