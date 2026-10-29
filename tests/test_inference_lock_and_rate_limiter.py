"""
tests/test_inference_lock_and_rate_limiter.py

Regression tests for:
1. is_inference_locked() — only REAGENT_PUBLIC_MODE=1 or REAGENT_DISABLE_LIVE_INFERENCE=1
   lock inference; RENDER=true alone does NOT.
2. DemoRateLimiter — per-IP limit, global limit, cooldown, window expiry.
3. Render deployment unblocked — POST /api/nemotron_propose proceeds past the
   PUBLIC_DEMO_LOCKED guard when RENDER=true (key-absent → NOT_CONFIGURED, not locked).
4. Rate-limited response returns RATE_LIMITED status (not PUBLIC_DEMO_LOCKED).
"""

import json
import os
import threading
import time
import unittest
import urllib.request
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

from reagent.ui_server import (
    DashboardHTTPHandler,
    DemoRateLimiter,
    is_inference_locked,
    is_public_mode,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. is_inference_locked() logic
# ─────────────────────────────────────────────────────────────────────────────

class TestIsInferenceLocked(unittest.TestCase):
    """is_inference_locked() must only be True for explicit lock env vars."""

    def _locked(self, env: dict) -> bool:
        with patch.dict(os.environ, env, clear=True):
            return is_inference_locked()

    def test_01_no_env_vars_unlocked(self):
        """No special env vars → inference is NOT locked."""
        self.assertFalse(self._locked({}))

    def test_02_render_alone_does_not_lock(self):
        """RENDER=true alone must NOT lock inference (Render supports live inference)."""
        self.assertFalse(self._locked({"RENDER": "true"}))

    def test_03_reagent_public_mode_locks(self):
        """REAGENT_PUBLIC_MODE=1 must lock inference (HF Space / test guard)."""
        self.assertTrue(self._locked({"REAGENT_PUBLIC_MODE": "1"}))

    def test_04_disable_live_inference_locks(self):
        """REAGENT_DISABLE_LIVE_INFERENCE=1 must lock inference."""
        self.assertTrue(self._locked({"REAGENT_DISABLE_LIVE_INFERENCE": "1"}))

    def test_05_render_plus_disable_locks(self):
        """RENDER=true + REAGENT_DISABLE_LIVE_INFERENCE=1 → locked."""
        self.assertTrue(self._locked({"RENDER": "true", "REAGENT_DISABLE_LIVE_INFERENCE": "1"}))

    def test_06_is_public_mode_still_true_on_render(self):
        """is_public_mode() (for UI branding) still returns True on Render."""
        with patch.dict(os.environ, {"RENDER": "true"}, clear=True):
            self.assertTrue(is_public_mode())

    def test_07_is_public_mode_true_with_public_mode_flag(self):
        with patch.dict(os.environ, {"REAGENT_PUBLIC_MODE": "1"}, clear=True):
            self.assertTrue(is_public_mode())

    def test_08_public_mode_flag_locks_both(self):
        """REAGENT_PUBLIC_MODE=1 → both is_public_mode() and is_inference_locked() True."""
        with patch.dict(os.environ, {"REAGENT_PUBLIC_MODE": "1"}, clear=True):
            self.assertTrue(is_public_mode())
            self.assertTrue(is_inference_locked())


# ─────────────────────────────────────────────────────────────────────────────
# 2. DemoRateLimiter behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestDemoRateLimiter(unittest.TestCase):
    """DemoRateLimiter must enforce per-IP and global limits correctly."""

    def _fresh(self) -> DemoRateLimiter:
        return DemoRateLimiter()

    def test_01_first_request_admitted(self):
        rl = self._fresh()
        ok, reason = rl.check("1.2.3.4")
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_02_cooldown_blocks_immediate_repeat(self):
        rl = self._fresh()
        rl.check("1.2.3.4")
        ok, reason = rl.check("1.2.3.4")
        self.assertFalse(ok)
        self.assertIn("cooldown", reason.lower())

    def test_03_different_ips_independent(self):
        rl = self._fresh()
        ok1, _ = rl.check("1.2.3.4")
        ok2, _ = rl.check("5.6.7.8")
        self.assertTrue(ok1)
        self.assertTrue(ok2)

    def test_04_per_ip_limit_enforced(self):
        rl = DemoRateLimiter()
        # Bypass cooldown by directly manipulating timestamps
        ip = "10.0.0.1"
        # Admit PER_IP_MAX requests by spacing them across fake times
        admitted = 0
        for i in range(rl._PER_IP_MAX):
            # Fake last time to skip cooldown
            with rl._lock:
                rl._ip_last[ip] = 0.0  # reset cooldown
            ok, _ = rl.check(ip)
            if ok:
                admitted += 1
        # Reset cooldown one more time
        with rl._lock:
            rl._ip_last[ip] = 0.0
        ok_over, reason = rl.check(ip)
        self.assertEqual(admitted, rl._PER_IP_MAX)
        self.assertFalse(ok_over)
        self.assertIn("per", reason.lower())

    def test_05_global_limit_enforced(self):
        rl = DemoRateLimiter()
        # Fill up global log directly
        now = time.time()
        with rl._lock:
            rl._global_log = [now] * rl._GLOBAL_MAX
        ok, reason = rl.check("new.ip.0.1")
        self.assertFalse(ok)
        self.assertIn("global", reason.lower())

    def test_06_thread_safety(self):
        """Concurrent check() calls must not raise or corrupt state."""
        rl = self._fresh()
        results = []

        def worker(ip):
            ok, _ = rl.check(ip)
            results.append(ok)

        threads = [threading.Thread(target=worker, args=(f"192.168.0.{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All 20 different IPs should be admitted (global limit is 12 — some may be rejected)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 20)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Render environment: inference unblocked, key-absent → NOT_CONFIGURED
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderInferenceUnblocked(unittest.TestCase):
    """On Render without REAGENT_PUBLIC_MODE, the endpoint must NOT return PUBLIC_DEMO_LOCKED."""

    @classmethod
    def setUpClass(cls):
        # Simulate Render environment WITHOUT the inference lock flag
        cls._env_patcher = patch.dict(
            os.environ,
            {"RENDER": "true"},
            clear=False,
        )
        cls._env_patcher.start()
        # Ensure REAGENT_PUBLIC_MODE and REAGENT_DISABLE_LIVE_INFERENCE are absent
        os.environ.pop("REAGENT_PUBLIC_MODE", None)
        os.environ.pop("REAGENT_DISABLE_LIVE_INFERENCE", None)
        os.environ.pop("NEBIUS_API_KEY", None)
        # Reset the global rate limiter singleton so prior tests don't bleed in
        import reagent.ui_server as _ui
        _ui._demo_rate_limiter = DemoRateLimiter()
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls._env_patcher.stop()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_01_render_without_lock_does_not_return_public_demo_locked(self):
        """On Render (no lock flag, no key), endpoint must return NOT_CONFIGURED, not PUBLIC_DEMO_LOCKED."""
        req = urllib.request.Request(self._url("/api/nemotron_propose"), method="POST", data=b"{}")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        self.assertNotEqual(data["status"], "PUBLIC_DEMO_LOCKED",
                            "RENDER env alone must not trigger PUBLIC_DEMO_LOCKED")
        self.assertEqual(data["status"], "NOT_CONFIGURED",
                         "Without NEBIUS_API_KEY, status must be NOT_CONFIGURED")
        self.assertFalse(data["is_live"])

    def test_02_hotel_route_render_without_lock_not_demo_locked(self):
        """Same check for /api/hotel/propose_contract: must NOT return PUBLIC_DEMO_LOCKED."""
        req = urllib.request.Request(self._url("/api/hotel/propose_contract"), method="POST", data=b"{}")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        self.assertNotEqual(data["status"], "PUBLIC_DEMO_LOCKED",
                            "RENDER env alone must not trigger PUBLIC_DEMO_LOCKED on hotel route")
        # Acceptable statuses: NOT_CONFIGURED (no key) or RATE_LIMITED (cooldown from test_01)
        self.assertIn(data["status"], ("NOT_CONFIGURED", "RATE_LIMITED"),
                      f"Expected NOT_CONFIGURED or RATE_LIMITED, got: {data['status']}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Render + REAGENT_PUBLIC_MODE=1 still locks (explicit offline guard)
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderWithPublicModeStillLocked(unittest.TestCase):
    """When REAGENT_PUBLIC_MODE=1 is set (even on Render), inference must remain locked."""

    @classmethod
    def setUpClass(cls):
        cls._env_patcher = patch.dict(
            os.environ,
            {"RENDER": "true", "REAGENT_PUBLIC_MODE": "1"},
            clear=False,
        )
        cls._env_patcher.start()
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls._env_patcher.stop()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_01_public_mode_still_locks_on_render(self):
        """RENDER=true + REAGENT_PUBLIC_MODE=1 without ?live=1 → PUBLIC_DEMO_LOCKED."""
        req = urllib.request.Request(self._url("/api/nemotron_propose"), method="POST", data=b"{}")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        self.assertEqual(data["status"], "PUBLIC_DEMO_LOCKED")
        self.assertFalse(data["is_live"])
        self.assertIn("locked in public demonstration mode", data["message"])

    def test_02_public_mode_with_live_flag_bypasses_lock(self):
        """RENDER=true + REAGENT_PUBLIC_MODE=1 WITH ?live=1 → bypasses PUBLIC_DEMO_LOCKED."""
        import reagent.ui_server as _ui
        from reagent.ui_server import DemoRateLimiter
        _ui._demo_rate_limiter = DemoRateLimiter()
        req = urllib.request.Request(self._url("/api/nemotron_propose?live=1"), method="POST", data=b"{}")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        self.assertNotEqual(data["status"], "PUBLIC_DEMO_LOCKED",
                            "Explicit ?live=1 opt-in must bypass PUBLIC_DEMO_LOCKED")
        # If NEBIUS_API_KEY is present (.env loaded), CONTRACT_VALIDATED is returned; otherwise NOT_CONFIGURED
        self.assertIn(data["status"], ("CONTRACT_VALIDATED", "NOT_CONFIGURED", "GROUNDING_FAILED", "SCHEMA_INVALID", "RATE_LIMITED"))


if __name__ == "__main__":
    unittest.main()
