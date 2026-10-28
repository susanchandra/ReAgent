"""
test_network_binding.py — Strengthened tests for Network Binding & Public Inference Lock.

Verifies:
1. Real launch_demo.py argument parser honors PORT=10000 and HOST settings.
2. Actual start_server host resolution and bind behavior without opening public listener (mocked HTTPServer).
3. Both proposal endpoints (/api/nemotron_propose and /api/hotel/propose_contract) are locked
   in public mode BEFORE any external model/API call (mock provider verified uninvoked).
4. Public mode (RENDER=true or REAGENT_PUBLIC_MODE=1) binds to 0.0.0.0; local defaults remain 127.0.0.1.
5. start_server signature remains 100% backward compatible.
"""

import json
import os
import sys
import unittest
import urllib.request
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

from reagent.ui_server import (
    DashboardHTTPHandler,
    resolve_bind_host,
    start_server,
)


class TestNetworkBinding(unittest.TestCase):
    """Verifies host resolution, port handling, and backward compatibility."""

    def test_01_local_default_is_loopback(self):
        """Without env vars or arguments, host must default to 127.0.0.1 for local safety."""
        with patch.dict(os.environ, {}, clear=True):
            host = resolve_bind_host()
            self.assertEqual(host, "127.0.0.1")

    def test_02_explicit_argument_takes_precedence(self):
        """Explicit host parameter overrides environment variables."""
        with patch.dict(os.environ, {"HOST": "127.0.0.1"}, clear=True):
            self.assertEqual(resolve_bind_host("0.0.0.0"), "0.0.0.0")
            self.assertEqual(resolve_bind_host("10.0.0.1"), "10.0.0.1")

    def test_03_render_environment_resolves_to_all_interfaces(self):
        """Render cloud environment (RENDER=true) must resolve to 0.0.0.0."""
        with patch.dict(os.environ, {"RENDER": "true"}, clear=True):
            host = resolve_bind_host()
            self.assertEqual(host, "0.0.0.0")

    def test_04_public_mode_resolves_to_all_interfaces(self):
        """REAGENT_PUBLIC_MODE=1 must resolve to 0.0.0.0."""
        with patch.dict(os.environ, {"REAGENT_PUBLIC_MODE": "1"}, clear=True):
            host = resolve_bind_host()
            self.assertEqual(host, "0.0.0.0")

    def test_05_host_env_var_respected(self):
        """HOST environment variable is respected when set."""
        with patch.dict(os.environ, {"HOST": "0.0.0.0"}, clear=True):
            self.assertEqual(resolve_bind_host(), "0.0.0.0")

    def test_06_real_launch_demo_parser_honors_port_env(self):
        """Test the real launch_demo.py entrypoint and prove PORT=10000 is honored."""
        import launch_demo
        with patch.dict(os.environ, {"PORT": "10000"}, clear=True):
            with patch.object(sys, "argv", ["launch_demo.py"]):
                with patch("launch_demo.start_mission_control") as mock_start:
                    mock_start.return_value = 0
                    with patch("launch_demo.check_prerequisites", return_value=True):
                        ret = launch_demo.main()
                        self.assertEqual(ret, 0)
                        mock_start.assert_called_once_with(port=10000, host="127.0.0.1", open_browser=False)

    def test_07_real_launch_demo_parser_honors_render_host(self):
        """Under RENDER=true, launch_demo.py defaults host to 0.0.0.0 and reads PORT."""
        import launch_demo
        with patch.dict(os.environ, {"PORT": "10000", "RENDER": "true"}, clear=True):
            with patch.object(sys, "argv", ["launch_demo.py"]):
                with patch("launch_demo.start_mission_control") as mock_start:
                    mock_start.return_value = 0
                    with patch("launch_demo.check_prerequisites", return_value=True):
                        ret = launch_demo.main()
                        self.assertEqual(ret, 0)
                        mock_start.assert_called_once_with(port=10000, host="0.0.0.0", open_browser=False)

    def test_08_actual_start_server_binding_behavior_mocked(self):
        """Test actual start_server host resolution/binding without opening public listener."""
        # 1. Under Render environment, binds to 0.0.0.0
        with patch("reagent.ui_server.HTTPServer") as mock_http:
            mock_inst = MagicMock()
            mock_inst.serve_forever.side_effect = KeyboardInterrupt  # Clean stop
            mock_http.return_value = mock_inst
            with patch.dict(os.environ, {"RENDER": "true"}, clear=True):
                start_server(port=10000)
                mock_http.assert_called_once_with(("0.0.0.0", 10000), DashboardHTTPHandler)
                mock_inst.serve_forever.assert_called_once()

        # 2. Under local default, binds to 127.0.0.1
        with patch("reagent.ui_server.HTTPServer") as mock_http:
            mock_inst = MagicMock()
            mock_inst.serve_forever.side_effect = KeyboardInterrupt
            mock_http.return_value = mock_inst
            with patch.dict(os.environ, {}, clear=True):
                start_server(port=8080)
                mock_http.assert_called_once_with(("127.0.0.1", 8080), DashboardHTTPHandler)
                mock_inst.serve_forever.assert_called_once()

    def test_09_start_server_backward_compatible_signature(self):
        """start_server must accept port as first positional argument without error."""
        import inspect
        sig = inspect.signature(start_server)
        params = list(sig.parameters.keys())
        self.assertEqual(params[0], "port", "First parameter must be 'port' for backward compatibility")
        self.assertEqual(params[1], "host", "Second parameter must be 'host'")


class TestPublicInferenceLockBothRoutes(unittest.TestCase):
    """Verifies that public deployment locks BOTH model propose endpoints before any API call."""

    @classmethod
    def setUpClass(cls):
        # Start test server with REAGENT_PUBLIC_MODE=1 on safe ephemeral loopback port
        cls._env_patcher = patch.dict(os.environ, {"REAGENT_PUBLIC_MODE": "1"})
        cls._env_patcher.start()
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
        cls.port = cls.server.server_port
        import threading
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls._env_patcher.stop()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_01_public_lock_rejects_nemotron_propose_before_model_call(self):
        """POST /api/nemotron_propose must return PUBLIC_DEMO_LOCKED before any provider instantiation."""
        with patch("reagent.ui_server.NebiusNemotronProvider") as mock_provider:
            req = urllib.request.Request(self._url("/api/nemotron_propose"), method="POST", data=b"{}")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "PUBLIC_DEMO_LOCKED")
                self.assertFalse(data["is_live"])
                self.assertIn("locked in public demonstration mode", data["message"])
                # Prove provider was never invoked
                mock_provider.assert_not_called()

    def test_02_public_lock_rejects_hotel_propose_contract_before_model_call(self):
        """POST /api/hotel/propose_contract must return PUBLIC_DEMO_LOCKED before any provider instantiation."""
        with patch("reagent.ui_server.NebiusNemotronProvider") as mock_provider:
            req = urllib.request.Request(self._url("/api/hotel/propose_contract"), method="POST", data=b"{}")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "PUBLIC_DEMO_LOCKED")
                self.assertFalse(data["is_live"])
                self.assertIn("locked in public demonstration mode", data["message"])
                # Prove provider was never invoked
                mock_provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
