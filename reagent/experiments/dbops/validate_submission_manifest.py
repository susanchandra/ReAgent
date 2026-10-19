"""
validate_submission_manifest.py — Cryptographic and Schema Validator for Submission Manifest.

Parses reports/dbops/submission/submission_manifest.json, validates each declared artifact's
existence, verifies file size, and recomputes the SHA-256 hash against the actual disk bytes.
Exits 0 on complete integrity, exits nonzero on any mismatch, missing file, or schema error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


def compute_file_sha256(file_path: Path) -> str:
    """Computes hexadecimal SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_manifest(
    manifest_path: Path = Path("reports/dbops/submission/submission_manifest.json"),
    repo_root: Path = Path("."),
    quiet: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates submission manifest.
    Returns (success_bool, details_dict).
    """
    details: Dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "total_declared": 0,
        "verified_count": 0,
        "missing_count": 0,
        "mismatched_count": 0,
        "errors": [],
        "artifacts": {},
    }

    if not manifest_path.exists():
        err = f"Manifest not found at: {manifest_path}"
        details["errors"].append(err)
        if not quiet:
            print(f"[MANIFEST-VALIDATOR] ERROR: {err}")
        return False, details

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        err = f"Failed to parse manifest JSON: {exc}"
        details["errors"].append(err)
        if not quiet:
            print(f"[MANIFEST-VALIDATOR] ERROR: {err}")
        return False, details

    # Validate top-level schema
    required_keys = ["submission_title", "source_git_revision", "manifest_version", "artifacts"]
    for rk in required_keys:
        if rk not in data:
            err = f"Schema error: missing required root key '{rk}'"
            details["errors"].append(err)
            if not quiet:
                print(f"[MANIFEST-VALIDATOR] ERROR: {err}")
            return False, details

    artifacts = data.get("artifacts", {})
    if not isinstance(artifacts, dict) or len(artifacts) == 0:
        err = "Schema error: 'artifacts' must be a non-empty dictionary"
        details["errors"].append(err)
        if not quiet:
            print(f"[MANIFEST-VALIDATOR] ERROR: {err}")
        return False, details

    details["total_declared"] = len(artifacts)

    for rel_path, meta in artifacts.items():
        artifact_path = repo_root / rel_path
        artifact_record: Dict[str, Any] = {
            "declared_sha256": meta.get("sha256"),
            "declared_size": meta.get("size_bytes"),
            "status": "UNVERIFIED",
        }

        if not artifact_path.exists():
            artifact_record["status"] = "MISSING"
            details["missing_count"] += 1
            details["errors"].append(f"Missing artifact on disk: {rel_path}")
            details["artifacts"][rel_path] = artifact_record
            continue

        disk_size = artifact_path.stat().st_size
        actual_sha = compute_file_sha256(artifact_path)
        artifact_record["actual_size"] = disk_size
        artifact_record["actual_sha256"] = actual_sha

        expected_sha = meta.get("sha256")
        if expected_sha != actual_sha:
            artifact_record["status"] = "SHA_MISMATCH"
            details["mismatched_count"] += 1
            details["errors"].append(
                f"SHA-256 mismatch for {rel_path}: declared {expected_sha} vs actual {actual_sha}"
            )
            details["artifacts"][rel_path] = artifact_record
            continue

        expected_size = meta.get("size_bytes")
        if expected_size is not None and expected_size != disk_size:
            artifact_record["status"] = "SIZE_MISMATCH"
            details["mismatched_count"] += 1
            details["errors"].append(
                f"Size mismatch for {rel_path}: declared {expected_size} vs actual {disk_size}"
            )
            details["artifacts"][rel_path] = artifact_record
            continue

        artifact_record["status"] = "VERIFIED"
        details["verified_count"] += 1
        details["artifacts"][rel_path] = artifact_record

    success = (details["missing_count"] == 0 and details["mismatched_count"] == 0 and len(details["errors"]) == 0)

    if not quiet:
        print("=" * 78)
        print("          ReAgent: Cryptographic Submission Manifest Validator")
        print("=" * 78)
        print(f" Manifest Path      : {manifest_path}")
        print(f" Manifest Version   : {data.get('manifest_version')}")
        print(f" Git Revision       : {data.get('source_git_revision')}")
        print(f" Total Artifacts    : {details['total_declared']}")
        print(f" Verified Matches   : {details['verified_count']}")
        print(f" Missing Files      : {details['missing_count']}")
        print(f" Hash Mismatches    : {details['mismatched_count']}")
        print("-" * 78)

        for rel, rec in details["artifacts"].items():
            status_tag = f"[{rec['status']}]".ljust(16)
            print(f" {status_tag} {rel}")

        print("=" * 78)
        if success:
            print(" [VERDICT] PASS — All artifacts exist and match cryptographic SHA-256 hashes.")
        else:
            print(" [VERDICT] FAIL — Manifest validation failed:")
            for err in details["errors"]:
                print(f"   * {err}")
        print("=" * 78)

    return success, details


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ReAgent submission manifest integrity.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/dbops/submission/submission_manifest.json"),
        help="Path to submission_manifest.json",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress detailed console output; exit code only.",
    )
    args = parser.parse_args()

    success, _ = validate_manifest(manifest_path=args.manifest, quiet=args.quiet)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
