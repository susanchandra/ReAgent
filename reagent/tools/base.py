"""
ReAgent Base Tool Definition
All executable tools must inherit from BaseTool, declare typed schemas,
and return structured ActionResult objects.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time

from reagent.models import ActionResult


class BaseTool(ABC):
    """Abstract base class for all allowlisted tools in ReAgent."""

    name: str = "base_tool"
    description: str = "Base tool description"
    parameters_schema: Dict[str, Any] = {}
    # Declares whether this tool changes external/persistent state.
    # "state-changing" tools MUST have at least one independent postcondition predicate.
    # "read-only" tools may use STATUS_SUCCESS only.
    effect_class: str = "state-changing"


    def get_schema(self) -> Dict[str, Any]:
        """Return standardized tool definition schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    def validate_parameters(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Validate input parameters against schema.
        Returns error string if invalid, or None if valid.
        """
        required = self.parameters_schema.get("required", [])
        for field in required:
            if field not in params:
                return f"Missing required parameter: '{field}'"
        return None

    def run(self, params: Dict[str, Any]) -> ActionResult:
        """
        Safe execution harness with parameter validation and timing.
        """
        val_error = self.validate_parameters(params)
        if val_error:
            return ActionResult(
                tool_name=self.name,
                success=False,
                error=f"ParameterValidationError: {val_error}",
                duration_ms=0,
            )

        t0 = time.perf_counter()
        try:
            output = self.execute(params)
            duration = int((time.perf_counter() - t0) * 1000)
            return ActionResult(
                tool_name=self.name,
                success=True,
                output=output,
                duration_ms=duration,
            )
        except Exception as e:
            duration = int((time.perf_counter() - t0) * 1000)
            return ActionResult(
                tool_name=self.name,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                duration_ms=duration,
            )

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Any:
        """Concrete tool execution logic."""
        pass
