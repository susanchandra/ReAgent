"""
hotel_booking_service.py — Standalone Hotel Booking HTTP Service Simulator.

Provides an independent HTTP service with its own isolated SQLite persistence
layer, validating request schemas, tracking reservation and audit logs, and
providing deterministic fault injection for reliability testing:

Fault Modes:
- NO_FAULT: Normal atomic commit to reservations and audit_log, returns 201.
- COMMIT_THEN_TIMEOUT: Commits to SQLite, then abruptly drops connection/times out.
- FAIL_BEFORE_COMMIT: Fails with 500 before any database write.
- PARTIAL_COMMIT: Commits reservation but fails audit log, creating state discrepancy.
- STATUS_UNAVAILABLE: Observer endpoint returns 503 Service Unavailable.

All operations occur across a real local HTTP boundary on an ephemeral port.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class HotelBookingDB:
    """
    Independent SQLite persistence layer for Hotel Booking Simulator.
    Completely isolated from ReAgent's internal state.
    """

    def __init__(self, db_path: Optional[str] = None):
        import tempfile
        if db_path is None:
            self._tmp_file = tempfile.NamedTemporaryFile(suffix="_hotel_sim.db", delete=False)
            self._tmp_file.close()
            self.db_path = self._tmp_file.name
            self._is_temp = True
        else:
            self.db_path = db_path
            self._is_temp = False
        self._lock = threading.Lock()
        self._init_schema()

    def close(self) -> None:
        if getattr(self, "_is_temp", False) and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS reservations (
                        booking_id TEXT PRIMARY KEY,
                        guest_name TEXT NOT NULL,
                        room_number TEXT NOT NULL,
                        amount REAL NOT NULL,
                        status TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        committed_at REAL NOT NULL,
                        operation_id TEXT UNIQUE
                    );

                    CREATE TABLE IF NOT EXISTS reservation_audit_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        booking_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        logged_at REAL NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    def create_reservation(
        self,
        booking_id: str,
        guest_name: str,
        room_number: str,
        amount: float,
        status: str = "CONFIRMED",
        operation_id: Optional[str] = None,
        skip_audit_log: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            conn = self._get_connection()
            try:
                now = time.time()
                op_id = operation_id or f"op_{int(now * 1000)}"
                conn.execute(
                    """
                    INSERT INTO reservations (
                        booking_id, guest_name, room_number, amount, status, created_at, committed_at, operation_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (booking_id, guest_name, room_number, amount, status, now, now, op_id),
                )
                if not skip_audit_log:
                    conn.execute(
                        """
                        INSERT INTO reservation_audit_log (booking_id, action, logged_at)
                        VALUES (?, ?, ?)
                        """,
                        (booking_id, f"RESERVATION_CREATED:{room_number}", now),
                    )
                conn.commit()
                return {
                    "booking_id": booking_id,
                    "guest_name": guest_name,
                    "room_number": room_number,
                    "amount": amount,
                    "status": status,
                    "committed_at": now,
                    "operation_id": op_id,
                }
            finally:
                conn.close()

    def get_reservation(self, booking_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM reservations WHERE booking_id = ?",
                    (booking_id,),
                ).fetchone()
                if not row:
                    return None

                audit_rows = conn.execute(
                    "SELECT COUNT(*) FROM reservation_audit_log WHERE booking_id = ?",
                    (booking_id,),
                ).fetchone()[0]

                return {
                    "booking_id": row["booking_id"],
                    "guest_name": row["guest_name"],
                    "room_number": row["room_number"],
                    "amount": float(row["amount"]),
                    "status": row["status"],
                    "created_at": float(row["created_at"]),
                    "committed_at": float(row["committed_at"]),
                    "operation_id": row["operation_id"],
                    "audit_logged": (audit_rows > 0),
                    "audit_count": audit_rows,
                }
            finally:
                conn.close()

    def count_reservations(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                return conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0]
            finally:
                conn.close()

    def count_audit_logs(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                return conn.execute("SELECT COUNT(*) FROM reservation_audit_log").fetchone()[0]
            finally:
                conn.close()

    def reset(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM reservation_audit_log")
                conn.execute("DELETE FROM reservations")
                conn.commit()
            finally:
                conn.close()


class HotelSimulatorHTTPHandler(BaseHTTPRequestHandler):
    """
    HTTP Handler implementing validated endpoints and deterministic fault injection.
    """

    server: "HotelBookingHTTPServer"  # Type hint for server reference

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default noisy console logs

    def _send_json(self, data: Any, status: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        query_str = self.path.split("?")[1] if "?" in self.path else ""
        from urllib.parse import parse_qs
        qs = parse_qs(query_str)

        # Health endpoint
        if path in ("/api/v1/health", "/health"):
            db = self.server.db
            self._send_json({
                "status": "ok",
                "service": "HotelBookingSimulator",
                "reservations": db.count_reservations(),
                "audit_logs": db.count_audit_logs(),
            })
            return

        # Query booking status: GET /api/v1/bookings/<booking_id>
        if path.startswith("/api/v1/bookings/"):
            booking_id = path[len("/api/v1/bookings/") :]
            
            # Check fault injection for observer
            fault = qs.get("fault_mode", [None])[0] or self.headers.get("X-Fault-Mode") or self.server.global_fault
            if fault in ("STATUS_UNAVAILABLE", "OBSERVER_UNAVAILABLE"):
                self._send_json(
                    {
                        "error": "ObserverServiceUnavailable: Hotel reservation status system unreachable",
                        "status": "UNAVAILABLE",
                        "fault": "STATUS_UNAVAILABLE",
                    },
                    status=503,
                )
                return

            res = self.server.db.get_reservation(booking_id)
            if res is None:
                self._send_json(
                    {
                        "status": "ABSENT",
                        "booking_id": booking_id,
                        "message": "Reservation not found in database",
                    },
                    status=404,
                )
                return

            if not res["audit_logged"] or res["status"] == "PARTIAL":
                # Observable discrepancy: Partial state
                self._send_json(
                    {
                        "status": "PARTIAL",
                        "booking_id": res["booking_id"],
                        "guest_name": res["guest_name"],
                        "room_number": res["room_number"],
                        "amount": res["amount"],
                        "audit_logged": False,
                        "discrepancy": "Reservation record exists but audit log is missing",
                    },
                    status=200,
                )
                return

            # Confirmed complete reservation
            self._send_json(
                {
                    "status": "PRESENT",
                    "reservation_status": res["status"],
                    "booking_id": res["booking_id"],
                    "guest_name": res["guest_name"],
                    "room_number": res["room_number"],
                    "amount": res["amount"],
                    "committed_at": res["committed_at"],
                    "audit_logged": True,
                },
                status=200,
            )
            return

        self._send_json({"error": "Endpoint not found"}, status=404)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        # Reset endpoint: POST /api/v1/reset
        if path in ("/api/v1/reset", "/reset"):
            self.server.db.reset()
            self._send_json({"status": "ok", "message": "Simulator database reset"})
            return

        # Create booking: POST /api/v1/bookings
        if path in ("/api/v1/bookings", "/bookings"):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json({"error": "Empty request body"}, status=400)
                return

            raw_body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json({"error": "Malformed JSON payload"}, status=400)
                return

            # Parameter validation
            required = ["booking_id", "guest_name", "room_number", "amount"]
            for field in required:
                if field not in payload or payload[field] is None:
                    self._send_json({"error": f"Missing required parameter '{field}'"}, status=400)
                    return

            try:
                amount = float(payload["amount"])
                if amount <= 0:
                    raise ValueError()
            except (ValueError, TypeError):
                self._send_json({"error": "Parameter 'amount' must be a positive number"}, status=400)
                return

            booking_id = str(payload["booking_id"]).strip()
            guest_name = str(payload["guest_name"]).strip()
            room_number = str(payload["room_number"]).strip()
            if not booking_id or not guest_name or not room_number:
                self._send_json({"error": "Required fields cannot be empty strings"}, status=400)
                return

            fault_mode = payload.get("fault_mode") or self.headers.get("X-Fault-Mode") or self.server.global_fault

            # ── FAULT: FAIL_BEFORE_COMMIT ─────────────────────────────────────
            if fault_mode == "FAIL_BEFORE_COMMIT":
                if not hasattr(self.server, "attempt_counts"):
                    self.server.attempt_counts = {}
                attempts = self.server.attempt_counts.get(booking_id, 0) + 1
                self.server.attempt_counts[booking_id] = attempts
                if attempts == 1:
                    # Transient failure on attempt 1: fails before any write occurs to the database
                    self._send_json(
                        {
                            "error": "GatewayError: Transient upstream failure before reservation commit",
                            "status": "FAILED",
                            "fault": "FAIL_BEFORE_COMMIT",
                            "attempt": attempts,
                        },
                        status=500,
                    )
                    return
                # On retry (attempt >= 2), the transient failure clears: proceed to write below!

            # ── FAULT: PARTIAL_COMMIT ─────────────────────────────────────────
            if fault_mode == "PARTIAL_COMMIT":
                # Commits to reservations table, but intentionally omits audit log
                try:
                    res = self.server.db.create_reservation(
                        booking_id=booking_id,
                        guest_name=guest_name,
                        room_number=room_number,
                        amount=amount,
                        status="PARTIAL",
                        skip_audit_log=True,
                    )
                except sqlite3.IntegrityError:
                    self._send_json({"error": "DuplicateReservation: Booking ID already exists"}, status=409)
                    return

                self._send_json(
                    {
                        "error": "DatabaseDiscrepancy: Transaction completed partially; audit logging failed",
                        "status": "PARTIAL",
                        "booking_id": booking_id,
                    },
                    status=500,
                )
                return

            # ── FAULT: COMMIT_THEN_TIMEOUT ────────────────────────────────────
            if fault_mode == "COMMIT_THEN_TIMEOUT":
                # Commits cleanly to SQLite disk
                try:
                    self.server.db.create_reservation(
                        booking_id=booking_id,
                        guest_name=guest_name,
                        room_number=room_number,
                        amount=amount,
                        status="CONFIRMED",
                    )
                except sqlite3.IntegrityError:
                    self._send_json({"error": "DuplicateReservation: Booking ID already exists"}, status=409)
                    return

                # Drop connection abruptly without sending HTTP response headers
                try:
                    self.close_connection = True
                    if hasattr(self, "wfile") and self.wfile:
                        self.wfile.flush()
                    if hasattr(self, "connection") and self.connection:
                        self.connection.close()
                except Exception:
                    pass
                return

            # ── NORMAL SUCCESS (NO_FAULT) ─────────────────────────────────────
            try:
                res = self.server.db.create_reservation(
                    booking_id=booking_id,
                    guest_name=guest_name,
                    room_number=room_number,
                    amount=amount,
                    status="CONFIRMED",
                )
            except sqlite3.IntegrityError:
                self._send_json(
                    {"error": f"DuplicateReservation: Booking ID '{booking_id}' already exists"},
                    status=409,
                )
                return

            self._send_json(
                {
                    "status": "CONFIRMED",
                    "booking_id": res["booking_id"],
                    "guest_name": res["guest_name"],
                    "room_number": res["room_number"],
                    "amount": res["amount"],
                    "committed_at": res["committed_at"],
                    "operation_id": res["operation_id"],
                },
                status=201,
            )
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


class HotelBookingHTTPServer(HTTPServer):
    """Subclass holding db instance and global fault mode."""
    db: HotelBookingDB
    global_fault: Optional[str]


class HotelBookingServiceSimulator:
    """
    Manager for the Hotel Booking Service simulator process/thread.
    Binds to 127.0.0.1 on an ephemeral port (0) to eliminate any port conflicts.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db = HotelBookingDB(db_path)
        self.server: Optional[HotelBookingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.port: int = 0
        self.base_url: str = ""
        self._is_running = False

    def start(self) -> str:
        """Start simulator HTTP server on an ephemeral port."""
        if self._is_running:
            return self.base_url

        self.server = HotelBookingHTTPServer(("127.0.0.1", 0), HotelSimulatorHTTPHandler)
        self.server.db = self.db
        self.server.global_fault = None
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self._is_running = True
        time.sleep(0.05)  # Brief warm-up
        return self.base_url

    def stop(self) -> None:
        """Stop simulator HTTP server."""
        if not self._is_running or not self.server:
            return
        self.server.shutdown()
        self.server.server_close()
        self._is_running = False
        self.db.close()

    def reset_database(self) -> None:
        """Clear all records in the simulator database."""
        self.db.reset()
        if self.server and hasattr(self.server, "attempt_counts"):
            self.server.attempt_counts.clear()

    def set_global_fault(self, fault: Optional[str]) -> None:
        """Set a server-level fault mode."""
        if self.server:
            self.server.global_fault = fault
