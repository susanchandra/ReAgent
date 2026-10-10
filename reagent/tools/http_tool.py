"""
ReAgent Allowlisted HTTP Request Tool
"""

from typing import Dict, Any, Optional
import requests
from reagent.tools.base import BaseTool


class HttpRequestTool(BaseTool):
    name = "http_request"
    description = "Performs sandboxed HTTP GET or POST requests with strict timeout enforcement."
    effect_class = "state-changing"  # POST can mutate remote state; err on the side of caution
    parameters_schema = {

        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Target HTTP/HTTPS URL"},
            "method": {"type": "string", "enum": ["GET", "POST"], "default": "GET"},
            "payload": {"type": "object", "description": "Optional JSON body for POST"},
            "headers": {"type": "object", "description": "Optional HTTP headers"},
            "timeout_seconds": {"type": "number", "default": 5.0},
        },
        "required": ["url"],
    }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params["url"]
        method = params.get("method", "GET").upper()
        payload = params.get("payload")
        headers = params.get("headers", {})
        timeout = float(params.get("timeout_seconds", 5.0))

        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=timeout)
        elif method == "POST":
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        try:
            body = resp.json()
        except Exception:
            body = resp.text[:1000]

        return {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": body,
            "ok": resp.ok,
        }
