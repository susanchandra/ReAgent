"""
ReAgent Sandboxed File Tool
Enables safe read/write operations strictly restricted to an allowlisted workspace directory.
Prevents directory traversal and access to unauthorized paths.
"""

import os
from typing import Dict, Any
from reagent.tools.base import BaseTool


MAX_READ_BYTES = 1 * 1024 * 1024  # 1 MB hard cap on file read operations


class SandboxedFileTool(BaseTool):
    name = "sandboxed_file"
    description = "Reads or writes files strictly within the authorized sandbox directory."
    effect_class = "state-changing"
    parameters_schema = {

        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "write", "exists"], "description": "Operation type"},
            "relative_path": {"type": "string", "description": "Path relative to sandbox root"},
            "content": {"type": "string", "description": "Content to write (for write action)"},
        },
        "required": ["action", "relative_path"],
    }

    def __init__(self, sandbox_root: str = "D:/Jarvis/workspace/sandbox"):
        self.sandbox_root = os.path.realpath(os.path.abspath(sandbox_root))
        os.makedirs(self.sandbox_root, exist_ok=True)

    def _resolve_safe_path(self, rel_path: str) -> str:
        if not rel_path or not str(rel_path).strip():
            raise PermissionError("SecurityViolation: empty path rejected.")
        # Reject foreign drive letters or drive specifications
        drive, _ = os.path.splitdrive(str(rel_path))
        if drive:
            raise PermissionError(f"SecurityViolation: Foreign drive path '{rel_path}' escapes sandbox boundary.")
        # Strip leading slashes to prevent root escaping
        clean_rel = rel_path.lstrip("/\\")
        target_path = os.path.realpath(os.path.abspath(os.path.join(self.sandbox_root, clean_rel)))
        # Ensure target is strictly inside sandbox_root (directory-boundary safe, junction/symlink dereferenced)
        if not (target_path == self.sandbox_root or target_path.startswith(self.sandbox_root + os.sep)):
            raise PermissionError(f"SecurityViolation: Path '{rel_path}' escapes sandbox boundary.")
        return target_path

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params["action"]
        rel_path = params["relative_path"]
        safe_path = self._resolve_safe_path(rel_path)

        if action == "exists":
            return {"exists": os.path.exists(safe_path), "path": rel_path}

        elif action == "read":
            if not os.path.exists(safe_path):
                raise FileNotFoundError(f"File not found in sandbox: '{rel_path}'")
            file_size = os.path.getsize(safe_path)
            if file_size > MAX_READ_BYTES:
                raise ValueError(
                    f"FileExceedsSizeLimit: '{rel_path}' is {file_size} bytes, "
                    f"exceeding cap of {MAX_READ_BYTES} bytes."
                )
            with open(safe_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"content": content, "size_bytes": len(content), "path": rel_path}

        elif action == "write":
            content = params.get("content", "")
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"written_bytes": len(content), "status": "success", "path": rel_path}

        else:
            raise ValueError(f"Unknown action: '{action}'")
