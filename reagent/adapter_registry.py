"""
reagent/adapter_registry.py — Phase 6B Extensible Adapter Registry

Eliminates hardcoded switch statements by providing a clean, pluggable adapter registry
for task execution backends.

Each adapter encapsulates:
  - Invoking the underlying simulator / store
  - Head-to-head comparison (naive vs. ReAgent)
  - Independent observation of authoritative state
  - Returning a typed DemoArtifact with verifiable evidence

Design:
  - Zero live network or model calls.
  - New tools / domain simulators can register their own adapters without modifying core routing.
  - Strict safety: NEVER falls back to an unrelated adapter (zero silent substitution).
  - Preserves exact scenario semantics (A, B, C, D, E) for all registered task families.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Dict, Any, Optional, Union, List

from reagent.experiments.dbops.recovery_demo import DemoArtifact
from reagent.models import StructuredDecision


# Signature: (scenario_id, timestamp_utc, run_id, use_model_contract, decision, **kwargs) -> DemoArtifact
AdapterCallable = Callable[..., DemoArtifact]


class AdapterRegistry:
    """Registry of task execution adapters for head-to-head demo runs."""

    def __init__(self) -> None:
        self._adapters: Dict[str, Any] = {}

    def register(self, task_template_key: str, adapter: Any) -> None:
        """
        Register an adapter for a task capability key.
        Accepts either a callable function or an object implementing execute().
        """
        self._adapters[task_template_key] = adapter

    def get_adapter(self, task_template_key: str) -> Optional[Any]:
        """Retrieve an adapter if registered."""
        return self._adapters.get(task_template_key)

    def is_registered(self, task_template_key: str) -> bool:
        """Check if a capability key has a registered execution adapter."""
        return task_template_key in self._adapters

    def list_adapters(self) -> List[str]:
        """List all registered capability keys."""
        return list(self._adapters.keys())

    def dispatch(
        self,
        task_template_key: str,
        scenario_id: str,
        timestamp_utc: Optional[float] = None,
        run_id: Optional[str] = None,
        use_model_contract: bool = False,
        decision: Optional[StructuredDecision] = None,
        **kwargs: Any,
    ) -> DemoArtifact:
        """
        Dispatch execution to the registered adapter for task_template_key.
        Strict invariant: Never silently substitutes an unrelated adapter.
        Fails closed with KeyError if no adapter is registered.
        """
        adapter = self._adapters.get(task_template_key)
        if adapter is None:
            raise KeyError(
                f"No execution adapter registered for task capability {task_template_key!r}. "
                f"Available adapters: {list(self._adapters.keys())}. "
                "ReAgent strictly forbids executing an unregistered capability or substituting an unrelated operation."
            )

        # Support both object with .execute() and direct callable
        if hasattr(adapter, "execute") and callable(adapter.execute):
            return adapter.execute(
                scenario_id=scenario_id,
                timestamp_utc=timestamp_utc,
                run_id=run_id,
                use_model_contract=use_model_contract,
                decision=decision,
                **kwargs,
            )
        elif callable(adapter):
            return adapter(
                scenario_id=scenario_id,
                timestamp_utc=timestamp_utc,
                run_id=run_id,
                use_model_contract=use_model_contract,
                decision=decision,
                **kwargs,
            )
        else:
            raise TypeError(
                f"Registered adapter for {task_template_key!r} is neither callable nor has an execute() method."
            )


# Module-level singleton
ADAPTER_REGISTRY = AdapterRegistry()
