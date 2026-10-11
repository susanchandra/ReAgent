"""
oracle.py — Independent verification oracle for the ambiguity benchmark.

STRICT INDEPENDENCE RULES:
1. Does NOT import DeterministicVerifier or ReAgent pipeline classes.
2. Directly inspects the authoritative simulator commit ledger.
3. Decides ground truth ground-state outcomes:
   - exactly_once_committed: True if exactly 1 commit recorded
   - duplicate_committed: True if >1 commit recorded
   - lost_effect: True if 0 commits recorded when 1 was desired
"""

from typing import Dict, Any
from reagent.experiments.ambiguity.simulator import StatefulSimulator


class AmbiguityOracle:
    """
    Independent ground-truth evaluator inspecting authoritative simulator state.
    """
    @staticmethod
    def evaluate(
        simulator: StatefulSimulator,
        op_name: str,
        expected_commits: int = 1,
    ) -> Dict[str, Any]:
        commit_count = simulator.get_commit_count(op_name)
        duplicate = commit_count > expected_commits
        lost = commit_count < expected_commits
        exactly_once = commit_count == expected_commits

        return {
            "authoritative_commit_count": commit_count,
            "exactly_once_committed": exactly_once,
            "duplicate_committed": duplicate,
            "lost_effect": lost,
        }
