"""
ReAgent Evidence Collector
Extracts structured, immutable empirical facts from action execution results and environment states.
"""

from typing import Dict, Any, Optional
import os

from reagent.models import ActionResult, Evidence


class EvidenceCollector:
    """
    Extracts ground-truth evidence from tool execution results and environment observations.
    """

    def collect_from_action(
        self,
        action_result: ActionResult,
        additional_facts: Optional[Dict[str, Any]] = None,
    ) -> Evidence:
        """Constructs an Evidence instance capturing the outcome of an action."""
        facts: Dict[str, Any] = {
            "tool_name": action_result.tool_name,
            "success": action_result.success,
            "duration_ms": action_result.duration_ms,
        }

        if action_result.output is not None:
            facts["output"] = action_result.output

        if action_result.error is not None:
            facts["error"] = action_result.error

        if additional_facts:
            facts.update(additional_facts)

        return Evidence(
            source=f"tool_execution:{action_result.tool_name}",
            facts=facts,
        )

    def collect_file_evidence(self, file_path: str) -> Evidence:
        """Captures filesystem state as ground-truth evidence."""
        exists = os.path.exists(file_path)
        facts = {
            "path": file_path,
            "exists": exists,
            "size_bytes": os.path.getsize(file_path) if exists else 0,
        }
        return Evidence(
            source=f"filesystem:{file_path}",
            facts=facts,
        )
