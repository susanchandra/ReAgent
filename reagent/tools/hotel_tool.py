"""
hotel_tool.py — Allowlisted Hotel Booking HTTP Tool & Independent Status Probe.

Enables ReAgent to interact with the Hotel Booking Service simulator over HTTP:
- Action Execution: HotelBookingHttpTool dispatches reservation mutations.
- State Observation: make_hotel_booking_probe_callback queries independent status.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable
import requests

from reagent.tools.base import BaseTool
from reagent.models import ActionResult


class HotelBookingHttpTool(BaseTool):
    name = "hotel_booking_http"
    description = (
        "Submits a room reservation request to the Hotel Booking Service over HTTP. "
        "State-changing tool requiring independent postcondition verification."
    )
    effect_class = "state-changing"
    supports_idempotency_key = True
    supports_fencing = True
    parameters_schema = {
        "type": "object",
        "properties": {
            "service_url": {"type": "string", "description": "Base URL of Hotel Booking Service"},
            "booking_id": {"type": "string", "description": "Unique booking reference ID"},
            "guest_name": {"type": "string", "description": "Guest full name"},
            "room_number": {"type": "string", "description": "Target room or suite number"},
            "amount": {"type": "number", "description": "Reservation total amount"},
            "fault_mode": {"type": "string", "description": "Optional fault mode for simulation"},
            "timeout_seconds": {"type": "number", "default": 2.0},
        },
        "required": ["service_url", "booking_id", "guest_name", "room_number", "amount"],
    }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        service_url = params["service_url"].rstrip("/")
        timeout = float(params.get("timeout_seconds", 2.0))
        booking_id = params["booking_id"]
        guest_name = params["guest_name"]
        room_number = params["room_number"]
        amount = float(params["amount"])
        fault_mode = params.get("fault_mode", "NO_FAULT")

        url = f"{service_url}/api/v1/bookings"
        payload = {
            "booking_id": booking_id,
            "guest_name": guest_name,
            "room_number": room_number,
            "amount": amount,
            "fault_mode": fault_mode,
        }
        if "operation_id" in params:
            payload["operation_id"] = params["operation_id"]
        if "fencing_token" in params:
            payload["fencing_token"] = params["fencing_token"]

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout:
            raise TimeoutError(f"HTTPConnectionTimeout: Reservation request for '{booking_id}' timed out after {timeout}s")
        except requests.exceptions.ConnectionError as exc:
            # Remote end closed connection or dropped packet (e.g. COMMIT_THEN_TIMEOUT)
            raise ConnectionError(f"HTTPConnectionDropped: Service connection dropped during reservation for '{booking_id}': {exc}")

        if not resp.ok:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:500]
            raise RuntimeError(f"HTTPError_{resp.status_code}: {err_body}")

        return resp.json()


class StatusUnavailableError(RuntimeError):
    """Raised when the independent observation probe cannot reach the status service."""
    pass


def make_hotel_booking_probe_callback(
    service_url: str,
    booking_id: str,
    observer_fault: Optional[str] = None,
    timeout_seconds: float = 3.0,
) -> Callable[[str, Dict[str, Any], int], Dict[str, Any]]:
    """
    Constructs an independent status probe callback for ReAgent's recovery loop.
    Queries GET /api/v1/bookings/{booking_id} over a fresh HTTP request.
    """

    def probe_callback(tool_name: str, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        url = f"{service_url.rstrip('/')}/api/v1/bookings/{booking_id}"
        headers = {}
        if observer_fault:
            headers["X-Fault-Mode"] = observer_fault

        try:
            resp = requests.get(url, headers=headers, timeout=timeout_seconds)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise StatusUnavailableError(f"ObserverUnavailable: Status probe connection failed: {exc}")

        if resp.status_code == 503 or observer_fault == "STATUS_UNAVAILABLE":
            raise StatusUnavailableError("ObserverUnavailable: Hotel booking status query returned 503 Service Unavailable")

        if resp.status_code == 404:
            # Authoritative absence: record does not exist
            return {
                "status": "CONFIRMED_ABSENT",
                "booking_id": booking_id,
                "data": None,
                "attempt": attempt,
            }

        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "PARTIAL" or not data.get("audit_logged", True):
                return {
                    "status": "DISCREPANCY",
                    "booking_id": booking_id,
                    "data": data,
                    "discrepancy": data.get("discrepancy", "State mismatch in database"),
                    "attempt": attempt,
                }

            # Confirmed present in database
            return {
                "status": "PRESENT",
                "booking_id": booking_id,
                "data": data,
                "attempt": attempt,
            }

        raise RuntimeError(f"UnexpectedProbeResponse: HTTP {resp.status_code}: {resp.text[:300]}")

    return probe_callback
