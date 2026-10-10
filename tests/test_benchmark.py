"""
Unit tests for ReAgent Benchmark & Comparison Suite (reagent/benchmark.py).
Validates that benchmark scenarios execute deterministically offline and produce
consistent comparative run traces.
"""

import os
import unittest
import tempfile
import shutil

_MODULE_TEMP_DIR = None
_MODULE_PREV_LEDGER = None


def setUpModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    _MODULE_TEMP_DIR = tempfile.mkdtemp(prefix="reagent_bench_test_")
    _MODULE_PREV_LEDGER = os.environ.get("REAGENT_LEDGER_PATH")
    os.environ["REAGENT_LEDGER_PATH"] = os.path.join(_MODULE_TEMP_DIR, "ledger.json")


def tearDownModule():
    global _MODULE_TEMP_DIR, _MODULE_PREV_LEDGER
    if _MODULE_PREV_LEDGER is not None:
        os.environ["REAGENT_LEDGER_PATH"] = _MODULE_PREV_LEDGER
    else:
        os.environ.pop("REAGENT_LEDGER_PATH", None)
    if _MODULE_TEMP_DIR and os.path.exists(_MODULE_TEMP_DIR):
        shutil.rmtree(_MODULE_TEMP_DIR, ignore_errors=True)


from reagent.benchmark import run_benchmark_scenarios, BenchmarkComparisonReport


class TestReAgentBenchmark(unittest.TestCase):
    """Verifies the 5 benchmark comparison scenarios."""

    def test_benchmark_suite_execution(self):
        report = run_benchmark_scenarios()
        self.assertIsInstance(report, BenchmarkComparisonReport)
        self.assertEqual(report.scenarios_evaluated, 5)

        # ReAgent must achieve verification on recoverable tasks and escalate on unrecoverable
        self.assertEqual(report.reagent_verified_count, 3)
        self.assertEqual(report.reagent_escalated_count, 2)
        self.assertEqual(report.reagent_false_success_count, 0)

        # Baseline naive retry fails recoverable tasks and reports false success on unverified return status
        self.assertEqual(report.baseline_verified_count, 0)
        self.assertEqual(report.baseline_false_success_count, 1)

        # 10 traces total (5 ReAgent + 5 Baseline)
        self.assertEqual(len(report.runs), 10)


if __name__ == "__main__":
    unittest.main()
