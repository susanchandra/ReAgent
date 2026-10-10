"""
ReAgent Allowlisted JSON Validator Tool
Validates JSON structures, syntax, and schema conformance.
"""

from typing import Dict, Any, List
import json
from reagent.tools.base import BaseTool


class JsonValidatorTool(BaseTool):
    name = "json_validator"
    description = "Validates that a JSON string is syntactically valid and contains required keys."
    effect_class = "read-only"
    parameters_schema = {

        "type": "object",
        "properties": {
            "json_payload": {"type": "string", "description": "Raw JSON string or serialized dict"},
            "required_keys": {"type": "array", "items": {"type": "string"}, "description": "Required top-level keys"},
        },
        "required": ["json_payload"],
    }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        payload_raw = params["json_payload"]
        required_keys = params.get("required_keys", [])

        if isinstance(payload_raw, dict):
            parsed = payload_raw
        elif isinstance(payload_raw, str):
            parsed = json.loads(payload_raw)
        else:
            raise TypeError(f"Expected string or dict, got {type(payload_raw).__name__}")

        missing = [k for k in required_keys if k not in parsed]
        return {
            "is_valid": len(missing) == 0,
            "parsed": parsed,
            "missing_keys": missing,
        }
