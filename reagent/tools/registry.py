"""
ReAgent Tool Registry
Maintains the allowlist of approved tools and enforces strict access boundaries.
"""

from typing import Dict, List, Optional, Any
from reagent.tools.base import BaseTool
from reagent.models import ActionResult


class ToolRegistry:
    """
    Allowlist registry for ReAgent tools.
    Guarantees that no unapproved or arbitrary execution occurs.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register an allowlisted tool instance."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Look up tool by exact allowlisted name."""
        return self._tools.get(name)

    def is_allowlisted(self, name: str) -> bool:
        """Check if tool name is registered in the allowlist."""
        return name in self._tools

    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return schemas for all registered allowlisted tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    def list_tool_names(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ActionResult:
        """
        Execute tool through allowlist gate.
        Rejects any unregistered tool with a security violation.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return ActionResult(
                tool_name=tool_name,
                success=False,
                error=f"SecurityViolation: Tool '{tool_name}' is not in the approved allowlist.",
                duration_ms=0,
            )
        return tool.run(params)
