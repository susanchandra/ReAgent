"""
ReAgent Observability Event Timeline (Milestone 3)

Provides an ordered, append-only event log for a single task run.
Events are emitted at each phase of the frozen loop.
The log is never mutated — only appended.
"""

import logging
from typing import List, Dict, Any, Optional

from reagent.models import ObservabilityEvent

logger = logging.getLogger(__name__)

# Canonical phase names (in chronological order)
PHASE_TASK_RECEIVED         = "TASK_RECEIVED"
PHASE_CONTEXT_SELECTED      = "CONTEXT_SELECTED"
PHASE_MODEL_DECISION        = "MODEL_DECISION"
PHASE_DECISION_VALIDATED    = "DECISION_VALIDATED"
PHASE_CONTRACT_FROZEN       = "CONTRACT_FROZEN"
PHASE_POLICY_ALLOWED        = "POLICY_ALLOWED"
PHASE_TOOL_EXECUTED         = "TOOL_EXECUTED"
PHASE_EVIDENCE_COLLECTED    = "EVIDENCE_COLLECTED"
PHASE_VERIFICATION_FAILED   = "VERIFICATION_FAILED"
PHASE_FAILURE_ATTRIBUTED    = "FAILURE_ATTRIBUTED"
PHASE_RECOVERY_SELECTED     = "RECOVERY_SELECTED"
PHASE_RECOVERY_ALLOWED      = "RECOVERY_ALLOWED"
PHASE_RECOVERY_EXECUTED     = "RECOVERY_EXECUTED"
PHASE_REVERIFICATION        = "REVERIFICATION"
PHASE_VERIFIED_RECEIPT      = "VERIFIED_RECEIPT"


class ObservabilityTimeline:
    """
    Append-only ordered event log for a single task execution.
    Each event records which pipeline phase fired and key detail fields.
    Events are never modified after appending.
    """

    def __init__(self, task_id: str):
        self._task_id = task_id
        self._events: List[ObservabilityEvent] = []

    def emit(self, phase: str, detail: Optional[Dict[str, Any]] = None) -> ObservabilityEvent:
        """Append a new event and return it."""
        event = ObservabilityEvent(
            task_id=self._task_id,
            phase=phase,
            detail=detail or {},
        )
        self._events.append(event)
        logger.debug("[REAGENT] %s | task=%s | %s", phase, self._task_id, detail or {})
        return event

    @property
    def events(self) -> List[ObservabilityEvent]:
        """Read-only view of emitted events."""
        return list(self._events)

    def phases(self) -> List[str]:
        """Return ordered list of phase names that have fired."""
        return [e.phase for e in self._events]

    def has_phase(self, phase: str) -> bool:
        return any(e.phase == phase for e in self._events)

    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "event_count": len(self._events),
            "phases": self.phases(),
        }
