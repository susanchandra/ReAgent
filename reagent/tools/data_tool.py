"""
ReAgent Allowlisted Data Transform Tool
Extracts, transforms, and filters structured fields safely.
"""

from typing import Dict, Any, List
from reagent.tools.base import BaseTool


class DataTransformTool(BaseTool):
    name = "data_transform"
    description = "Extracts or filters key-value pairs from structured records safely."
    parameters_schema = {
        "type": "object",
        "properties": {
            "source_data": {"type": "object", "description": "Source dictionary"},
            "fields_to_extract": {"type": "array", "items": {"type": "string"}, "description": "Keys to extract"},
        },
        "required": ["source_data", "fields_to_extract"],
    }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        source = params["source_data"]
        fields = params["fields_to_extract"]

        if not isinstance(source, dict):
            raise TypeError(f"source_data must be a dict, got {type(source).__name__}")

        extracted = {k: source.get(k) for k in fields if k in source}
        return {
            "extracted_data": extracted,
            "extracted_count": len(extracted),
        }
