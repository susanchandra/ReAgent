"""
test_submission_manifest_validator.py — Unit tests for the cryptographic submission manifest validator.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reagent.experiments.dbops.validate_submission_manifest import validate_manifest, compute_file_sha256


class TestSubmissionManifestValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reagent_manifest_val_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_real_manifest_passes_cleanly(self):
        """The actual repository submission_manifest.json passes validation with 12 verified files."""
        success, details = validate_manifest(
            manifest_path=Path("reports/dbops/submission/submission_manifest.json"),
            repo_root=Path("."),
            quiet=True,
        )
        self.assertTrue(success, f"Manifest validation failed with errors: {details['errors']}")
        self.assertEqual(details["missing_count"], 0)
        self.assertEqual(details["mismatched_count"], 0)
        self.assertGreaterEqual(details["verified_count"], 12)

    def test_02_missing_artifact_fails_closed(self):
        """A manifest declaring a missing file fails closed with non-zero status."""
        manifest_data = {
            "submission_title": "Test Manifest",
            "source_git_revision": "test_rev",
            "manifest_version": "1.0.0",
            "artifacts": {
                "non_existent_file.txt": {
                    "sha256": "0" * 64,
                    "size_bytes": 100,
                    "status": "VERIFIED",
                }
            },
        }
        test_manifest = self.temp_dir / "test_manifest.json"
        test_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

        success, details = validate_manifest(
            manifest_path=test_manifest,
            repo_root=self.temp_dir,
            quiet=True,
        )
        self.assertFalse(success)
        self.assertEqual(details["missing_count"], 1)
        self.assertIn("non_existent_file.txt", details["errors"][0])

    def test_03_corrupted_hash_fails_closed(self):
        """A file whose disk content does not match the manifest SHA-256 fails closed."""
        real_file = self.temp_dir / "real_file.txt"
        real_file.write_text("Legitimate content", encoding="utf-8")

        manifest_data = {
            "submission_title": "Test Manifest",
            "source_git_revision": "test_rev",
            "manifest_version": "1.0.0",
            "artifacts": {
                "real_file.txt": {
                    "sha256": "bad_hash_" + "0" * 55,
                    "size_bytes": real_file.stat().st_size,
                    "status": "VERIFIED",
                }
            },
        }
        test_manifest = self.temp_dir / "test_manifest.json"
        test_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

        success, details = validate_manifest(
            manifest_path=test_manifest,
            repo_root=self.temp_dir,
            quiet=True,
        )
        self.assertFalse(success)
        self.assertEqual(details["mismatched_count"], 1)
        self.assertIn("SHA-256 mismatch", details["errors"][0])

    def test_04_malformed_schema_fails_closed(self):
        """Manifest missing required keys fails closed."""
        test_manifest = self.temp_dir / "malformed.json"
        test_manifest.write_text(json.dumps({"some_key": "val"}), encoding="utf-8")

        success, details = validate_manifest(
            manifest_path=test_manifest,
            repo_root=self.temp_dir,
            quiet=True,
        )
        self.assertFalse(success)
        self.assertIn("Schema error", details["errors"][0])

    def test_05_manifest_artifacts_non_mutating(self):
        """Verifies that reading and validating manifest does not mutate artifact bytes or hashes."""
        manifest_path = Path("reports/dbops/submission/submission_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        before_hashes = {}
        for rel_path in manifest_data["artifacts"]:
            p = Path(rel_path)
            self.assertTrue(p.exists(), f"Artifact {rel_path} must exist")
            before_hashes[rel_path] = compute_file_sha256(p)

        success, details = validate_manifest(manifest_path=manifest_path, repo_root=Path("."), quiet=True)
        self.assertTrue(success)

        for rel_path, before_hash in before_hashes.items():
            after_hash = compute_file_sha256(Path(rel_path))
            self.assertEqual(before_hash, after_hash, f"Artifact {rel_path} was mutated during validation!")


if __name__ == "__main__":
    unittest.main()
