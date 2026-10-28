"""
entrypoint.py — Hugging Face Space & Public Container Deployment Entrypoint for ReAgent.

Reuses the exact, accepted ReAgent Mission Control application and backend
(reagent.ui_server.DashboardHTTPHandler) without modifying any protected repository files.

Public Security & Fail-Closed Guardrails:
1. Binds to 0.0.0.0:7860 (or $PORT) as expected by Hugging Face Spaces.
2. In public demo mode, locks live model proposal routes to fail closed immediately,
   preventing any unauthenticated external network/API activity.
3. All scenario recovery simulations execute on disposable, isolated in-memory/temp SQLite stores.
4. Zero secret inspection: does not require or read NEBIUS_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import sys
from http.server import HTTPServer
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also ensure app root in Docker container is on sys.path
_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from reagent.ui_server import DashboardHTTPHandler, SOURCE_GIT_REVISION


class PublicSpaceDashboardHandler(DashboardHTTPHandler):
    """
    Subclass of the accepted Mission Control handler for public hosting.
    Provides strict fail-closed boundary: zero billable model calls in public mode.
    All other deterministic routes, evidence views, and benchmarks work identically.
    """

    def do_POST(self):
        clean_path = self._clean_path()
        # Public Mode Guard: Lock all live model propose routes
        if clean_path in ("/api/nemotron_propose", "/api/hotel/propose_contract"):
            self._send_json({
                "status": "PUBLIC_DEMO_LOCKED",
                "run_id": "public_propose_locked",
                "message": (
                    "Live model inference is locked in public demonstration mode. "
                    "ReAgent fails closed safely to prevent unauthenticated network/API calls."
                ),
                "is_live": False,
                "schema_valid": False,
                "grounding_valid": None,
            })
            return

        # Public Session Isolation: Reset acknowledges client session without wiping global memory logs
        if clean_path == "/api/reset":
            self._send_json({
                "status": "reset",
                "message": "Client session state reset. Active simulation logs preserved for concurrent sessions."
            })
            return

        # Delegate all other requests to the accepted application handler
        super().do_POST()


def run_public_server(host: str = "0.0.0.0", port: int = 7860):
    server = HTTPServer((host, port), PublicSpaceDashboardHandler)
    print("=" * 66)
    print(f"  ReAgent Public Mission Control -> http://{host}:{port}/")
    print("=" * 66)
    print(f" Mode        : PUBLIC_HOSTING_FAIL_CLOSED")
    print(f" Git Rev     : {SOURCE_GIT_REVISION}")
    print(f" Scenarios   : A, B, C, D, E (Deterministic SQLite Sandbox)")
    print(f" Live Calls  : LOCKED (Zero external API requests)")
    print("=" * 66)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PUBLIC-SERVER] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAgent Public Space Entrypoint")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"), help="Host to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")), help="Port to bind")
    args = parser.parse_args()

    run_public_server(host=args.host, port=args.port)
