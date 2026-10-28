"""
tests/test_mobile_and_honest_status.py
Phase 2 & 3 Regression Tests:

1. Mobile navigation CSS — hamburger button, backdrop, drawer toggle classes.
2. No horizontal overflow markers at target viewport widths (CSS-level inspection).
3. Correct public-locked state labels and honest offline/reference labeling.
4. No API-key configuration disclosure in public mode (HTML + /api/model_preflight).
5. Both proposal endpoints locked before provider instantiation in public mode.
6. Desktop layout and deterministic offline workflow remain intact.
"""

import json
import os
import re
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from unittest.mock import patch

from reagent.ui_server import (
    DashboardHTTPHandler,
    generate_dashboard_html,
    is_public_mode,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _BaseServer(unittest.TestCase):
    """Base class that spins up an in-process HTTPServer."""

    _env_overrides: dict = {}

    @classmethod
    def setUpClass(cls):
        cls._env_patcher = patch.dict(os.environ, cls._env_overrides)
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

    def _get(self, path: str):
        with urllib.request.urlopen(self._url(path)) as r:
            return r.status, r.read().decode("utf-8")

    def _post(self, path: str, body: bytes = b"{}"):
        req = urllib.request.Request(self._url(path), data=body, method="POST")
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mobile Layout — CSS & HTML Structure
# ─────────────────────────────────────────────────────────────────────────────

class TestMobileLayoutCSSAndDOM(unittest.TestCase):
    """Verify mobile CSS and DOM elements are present in generated HTML."""

    def setUp(self):
        self.html = generate_dashboard_html()

    # ── Hamburger toggle button ──────────────────────────────────────────────

    def test_01_hamburger_button_id_present(self):
        """HTML must contain the mobile hamburger toggle button by ID."""
        self.assertIn("btnMobileMenuToggle", self.html,
                      "Mobile hamburger button #btnMobileMenuToggle must be in HTML")

    def test_02_hamburger_button_calls_toggle_function(self):
        """Hamburger button must call toggleMobileMenu()."""
        self.assertIn("toggleMobileMenu()", self.html,
                      "Hamburger button onclick must call toggleMobileMenu()")

    def test_03_mobile_menu_toggle_css_class_present(self):
        """CSS must declare .mobile-menu-toggle class."""
        self.assertIn(".mobile-menu-toggle", self.html,
                      ".mobile-menu-toggle CSS class must exist")

    # ── Backdrop overlay ─────────────────────────────────────────────────────

    def test_04_backdrop_element_present(self):
        """HTML must contain the mobile sidebar backdrop element."""
        self.assertIn("mobile-sidebar-backdrop", self.html,
                      "mobile-sidebar-backdrop element must be in HTML")
        self.assertIn("sidebarBackdrop", self.html,
                      "#sidebarBackdrop ID must be in HTML")

    def test_05_backdrop_onclick_closes_drawer(self):
        """Backdrop onclick must call toggleMobileMenu(false)."""
        self.assertIn("toggleMobileMenu(false)", self.html,
                      "Backdrop onclick must call toggleMobileMenu(false)")

    # ── Sidebar close button ─────────────────────────────────────────────────

    def test_06_sidebar_close_button_present(self):
        """Sidebar must contain an in-drawer close button for mobile."""
        self.assertIn("btnMobileSidebarClose", self.html,
                      "In-drawer close button #btnMobileSidebarClose must be in HTML")

    # ── JavaScript functions ─────────────────────────────────────────────────

    def test_07_toggle_mobile_menu_function_defined(self):
        """toggleMobileMenu JS function must be declared."""
        self.assertIn("function toggleMobileMenu", self.html,
                      "toggleMobileMenu() JS function must be defined")

    def test_08_navigate_to_closes_drawer(self):
        """navigateTo() must call toggleMobileMenu(false) to auto-close drawer."""
        # Find the navigateTo function body
        start = self.html.find("function navigateTo(viewId")
        end = self.html.find("\nfunction ", start + 1)
        nav_body = self.html[start:end]
        self.assertIn("toggleMobileMenu(false)", nav_body,
                      "navigateTo() must close the mobile drawer via toggleMobileMenu(false)")

    def test_09_escape_key_closes_drawer(self):
        """Escape key listener must call toggleMobileMenu(false)."""
        self.assertIn("e.key === 'Escape'", self.html,
                      "Escape key listener must be present to close mobile drawer")

    # ── CSS media queries for mobile ─────────────────────────────────────────

    def test_10_mobile_main_wrap_removes_margin(self):
        """On mobile, .app-main-wrap must have margin-left: 0."""
        self.assertIn("margin-left: 0 !important", self.html,
                      ".app-main-wrap must remove margin-left on mobile")

    def test_11_mobile_sidebar_off_canvas_transform(self):
        """Sidebar must use translateX(-100%) to hide off-canvas on mobile."""
        self.assertIn("translateX(-100%)", self.html,
                      "Sidebar must be hidden off-canvas via translateX(-100%) on mobile")

    def test_12_how_it_works_grid_responsive_class(self):
        """The 5-column how-it-works grid must carry a responsive CSS class."""
        self.assertIn("how-it-works-grid", self.html,
                      "How-it-works grid must use .how-it-works-grid class for mobile breakpoints")

    def test_13_workbench_main_grid_id_responsive(self):
        """#wbMainGrid must be referenced in mobile CSS rules."""
        self.assertIn("#wbMainGrid", self.html,
                      "#wbMainGrid must be referenced in mobile CSS for single-column layout")

    def test_14_table_responsive_wrapper_exists(self):
        """Tables must be wrapped in .table-responsive for mobile scroll."""
        self.assertIn("table-responsive", self.html,
                      ".table-responsive wrapper must exist around overflow tables")

    # ── Desktop layout preserved ──────────────────────────────────────────────

    def test_15_desktop_sidebar_width_preserved(self):
        """Desktop sidebar must still be 240px wide."""
        self.assertIn("width: 240px", self.html,
                      "Desktop sidebar width must remain 240px")

    def test_16_desktop_main_wrap_margin_preserved(self):
        """Desktop .app-main-wrap must retain margin-left: 240px."""
        self.assertIn("margin-left: 240px", self.html,
                      "Desktop main wrap must retain margin-left: 240px")

    def test_17_app_root_overflow_x_hidden(self):
        """app-root must have overflow-x: hidden to prevent horizontal bleed."""
        self.assertIn("overflow-x: hidden", self.html,
                      ".app-root must have overflow-x: hidden")

    def test_18_mobile_content_padding_reduced(self):
        """On mobile, app-content padding must be reduced to ~14px sides."""
        self.assertIn("padding: 16px 14px 32px 14px !important", self.html,
                      "Mobile .app-content padding must be reduced for narrow screens")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Honest Nemotron Status (Local mode — no credentials set)
# ─────────────────────────────────────────────────────────────────────────────

class TestHonestNemotronStatusLocal(unittest.TestCase):
    """Verify honest Nemotron status labels in local (non-public) mode."""

    def setUp(self):
        # Ensure we're not in public mode
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        # Remove any public mode env vars
        os.environ.pop("RENDER", None)
        os.environ.pop("REAGENT_PUBLIC_MODE", None)
        self.addCleanup(patcher.stop)
        self.html = generate_dashboard_html()

    def test_01_nemotron_card_title_present(self):
        """Settings view must contain the Nemotron card title."""
        self.assertIn("Real Nemotron (Nebius) Mode", self.html,
                      "Local mode must show 'Real Nemotron (Nebius) Mode' card title")

    def test_02_advisory_only_label_present(self):
        """Advisory Only label must always be present."""
        self.assertIn("ADVISORY ONLY", self.html,
                      "ADVISORY ONLY label must be in HTML")

    def test_03_offline_reference_label_in_js(self):
        """previewOfflineReferenceProposal must show 'Simulated Example' label."""
        self.assertIn("Simulated Example", self.html,
                      "Offline reference must be labeled as Simulated Example")

    def test_04_illustrative_tokens_label_in_js(self):
        """Token counts in offline reference must be labeled as illustrative fixture data."""
        self.assertIn("Simulated Fixture", self.html,
                      "Token counts in offline reference must be labeled as Simulated Fixture")

    def test_05_public_demo_locked_string_absent_from_card_title(self):
        """In local mode, public demo locked label must NOT appear in card title."""
        self.assertNotIn("Inference Locked", self.html,
                         "Local mode must not show 'Inference Locked' in card title")

    def test_06_run_real_nemotron_button_present_locally(self):
        """In local non-public mode, the Run Real Nemotron button must be present."""
        self.assertIn("Run Real Nemotron Contract Proposal", self.html,
                      "Local mode must include the live Nemotron proposal button")

    def test_07_public_demo_locked_status_handling_in_js(self):
        """runNemotronPropose JS must handle PUBLIC_DEMO_LOCKED status honestly."""
        self.assertIn("PUBLIC_DEMO_LOCKED", self.html,
                      "JS must have branch to handle PUBLIC_DEMO_LOCKED response")
        self.assertIn("Live inference is disabled on this public demo", self.html,
                      "JS must display honest message for PUBLIC_DEMO_LOCKED status")

    def test_08_not_configured_handling_in_js(self):
        """runNemotronPropose must handle NOT_CONFIGURED with proper label."""
        self.assertIn("NOT_CONFIGURED", self.html,
                      "JS must handle NOT_CONFIGURED status from proposal endpoint")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Public Mode — Honest Status & Zero Secret Disclosure
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicModeHonestStatus(unittest.TestCase):
    """Verify public mode produces honest status without secret disclosure."""

    def setUp(self):
        with patch.dict(os.environ, {"REAGENT_PUBLIC_MODE": "1"}):
            self.html = generate_dashboard_html()

    def test_01_public_mode_card_title_honest(self):
        """Public mode must NOT claim 'Real Nemotron — Ready'."""
        self.assertNotIn("Real Nemotron (Nebius) Mode — Ready", self.html,
                         "Public mode must not say live inference is Ready")

    def test_02_public_mode_inference_locked_label_present(self):
        """Public mode must show inference is locked."""
        self.assertIn("Inference Locked", self.html,
                      "Public mode card title must include 'Inference Locked'")

    def test_03_no_api_key_set_disclosure(self):
        """Public mode must NOT disclose ✓ SET or ✗ NOT SET for NEBIUS_API_KEY."""
        # Either the key-set or key-not-set label would disclose server credential state
        self.assertNotIn("&#x2713; SET", self.html,
                         "Public mode must not disclose ✓ SET (credential present)")
        self.assertNotIn("&#x2717; NOT SET", self.html,
                         "Public mode must not disclose ✗ NOT SET (credential absent)")

    def test_04_no_local_path_disclosure(self):
        """Public mode must NOT show local Windows paths in setup instructions."""
        self.assertNotIn("D:\\\\ReAgent", self.html,
                         "Public mode must not expose local filesystem paths")

    def test_05_no_local_powershell_instructions(self):
        """Public mode must NOT show local PowerShell setup instructions."""
        self.assertNotIn("$env:NEBIUS_API_KEY", self.html,
                         "Public mode must not show local PowerShell credential setup")

    def test_06_public_demo_status_pill_accurate(self):
        """Sidebar status pill must reflect public/locked state."""
        self.assertIn("Public Demo", self.html,
                      "Sidebar status pill must show Public Demo in public mode")

    def test_07_live_inference_button_hidden_or_absent_in_public(self):
        """In public mode, the live Nemotron inference button must not be rendered as an active element.
        The JS function may reference the ID, but the actual HTML <button id="btnNemotronPropose"> must not appear.
        """
        # The actual HTML button element must be hidden: it should be replaced with an HTML comment
        self.assertIn("<!-- Live inference button hidden on public demo -->", self.html,
                      "Public mode must replace live inference button with HTML comment")
        # The rendered <button id="btnNemotronPropose"> must not appear
        self.assertNotIn('id="btnNemotronPropose"', self.html,
                         "Public mode must not render <button id=btnNemotronPropose>")

    def test_08_offline_reference_button_label_honest(self):
        """In public mode, reference button must be labeled as Simulated Example."""
        self.assertIn("Simulated Example", self.html,
                      "Public mode reference button must be labeled as Simulated Example")

    def test_09_security_lock_notice_present(self):
        """Public mode must display an explicit security lock notice."""
        self.assertIn("Public Demonstration Security Lock", self.html,
                      "Public mode must show Public Demonstration Security Lock notice")

    def test_10_no_api_key_even_if_set(self):
        """Even with NEBIUS_API_KEY set, public mode must not disclose credential status."""
        with patch.dict(os.environ, {"REAGENT_PUBLIC_MODE": "1",
                                     "NEBIUS_API_KEY": "test-dummy-key"}):
            html = generate_dashboard_html()
        # Must NOT show ✓ SET
        self.assertNotIn("&#x2713; SET", html,
                         "Public mode must not show ✓ SET even when key IS present")
        # Must NOT show the key value
        self.assertNotIn("test-dummy-key", html,
                         "Public mode must never embed key value in HTML")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Public /api/model_preflight — No Credential Disclosure
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicPreflightLock(_BaseServer):
    """Verify /api/model_preflight does not expose credentials in public mode."""
    _env_overrides = {"REAGENT_PUBLIC_MODE": "1"}

    def test_01_preflight_returns_200(self):
        """GET /api/model_preflight must return HTTP 200 in public mode."""
        status, _ = self._get("/api/model_preflight")
        self.assertEqual(status, 200)

    def test_02_preflight_api_key_present_false_in_public_mode(self):
        """In public mode, api_key_present must always be False."""
        _, body = self._get("/api/model_preflight")
        data = json.loads(body)
        self.assertFalse(data["api_key_present"],
                         "api_key_present must be False in public mode to prevent credential disclosure")

    def test_03_preflight_public_locked_flag_present(self):
        """In public mode, preflight must include public_locked: true."""
        _, body = self._get("/api/model_preflight")
        data = json.loads(body)
        self.assertTrue(data.get("public_locked"),
                        "public_locked must be True in public mode response")

    def test_04_preflight_network_calls_false_in_public_mode(self):
        """/api/model_preflight must report network_calls_allowed: false."""
        _, body = self._get("/api/model_preflight")
        data = json.loads(body)
        self.assertFalse(data["network_calls_allowed"])

    def test_05_preflight_no_key_in_body_even_when_set(self):
        """Even when a key is set in env, its value must never appear in preflight body."""
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-secret-key-abc123"}):
            _, body = self._get("/api/model_preflight")
        self.assertNotIn("test-secret-key-abc123", body,
                         "Actual key value must never appear in preflight response")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Public Proposal Endpoints — Fail-Closed Before Provider Init
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicProposalEndpointsLocked(_BaseServer):
    """Both proposal endpoints must return PUBLIC_DEMO_LOCKED before any provider call."""
    _env_overrides = {"REAGENT_PUBLIC_MODE": "1"}

    def test_01_nemotron_propose_locked(self):
        """POST /api/nemotron_propose must return PUBLIC_DEMO_LOCKED in public mode."""
        with patch("reagent.ui_server.NebiusNemotronProvider") as mock_provider:
            status, body = self._post("/api/nemotron_propose")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "PUBLIC_DEMO_LOCKED",
                         "nemotron_propose must return PUBLIC_DEMO_LOCKED")
        self.assertFalse(data["is_live"])
        mock_provider.assert_not_called()

    def test_02_hotel_propose_contract_locked(self):
        """POST /api/hotel/propose_contract must return PUBLIC_DEMO_LOCKED in public mode."""
        with patch("reagent.ui_server.NebiusNemotronProvider") as mock_provider:
            status, body = self._post("/api/hotel/propose_contract")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "PUBLIC_DEMO_LOCKED",
                         "hotel/propose_contract must return PUBLIC_DEMO_LOCKED")
        self.assertFalse(data["is_live"])
        mock_provider.assert_not_called()

    def test_03_locked_message_honest(self):
        """PUBLIC_DEMO_LOCKED response message must accurately describe the lock."""
        _, body = self._post("/api/nemotron_propose")
        data = json.loads(body)
        self.assertIn("locked in public demonstration mode", data["message"],
                      "Lock message must accurately describe public demo lock")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Local Mode — Desktop Layout & Offline Workflow Intact
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalModeDesktopAndOffline(_BaseServer):
    """Desktop layout and deterministic offline workflow must remain intact locally."""
    _env_overrides = {}

    def test_01_index_200_and_core_elements(self):
        """GET / returns 200 and all core UI elements."""
        status, html = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("ReAgent — Verify outcomes. Recover safely.", html)
        self.assertIn("UNKNOWN is not ABSENT", html)
        self.assertIn("480-Trial Comparative Benchmark", html)
        self.assertIn("Run Recovery Simulation", html)

    def test_02_desktop_sidebar_width_in_html(self):
        """Desktop sidebar must keep 240px width in local mode HTML."""
        _, html = self._get("/")
        self.assertIn("width: 240px", html)

    def test_03_local_preflight_honest(self):
        """Local preflight api_key_present is a bool (True/False), never a string."""
        _, body = self._get("/api/model_preflight")
        data = json.loads(body)
        self.assertIsInstance(data["api_key_present"], bool)
        self.assertFalse(data["network_calls_allowed"])

    def test_04_offline_recovery_demo_works(self):
        """POST /api/run_recovery_demo returns deterministic offline result."""
        status, body = self._post("/api/run_recovery_demo")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn("naive_retry_arm", data)
        self.assertIn("reagent_arm", data)
        self.assertTrue(data["naive_retry_arm"]["duplicate_occurred"])
        self.assertFalse(data["reagent_arm"]["duplicate_occurred"])

    def test_05_mobile_hamburger_in_local_html(self):
        """Local HTML must include mobile hamburger toggle for responsive layout."""
        _, html = self._get("/")
        self.assertIn("btnMobileMenuToggle", html)
        self.assertIn("toggleMobileMenu", html)

    def test_06_nemotron_card_title_present_locally(self):
        """Local HTML must contain Real Nemotron card title."""
        _, html = self._get("/")
        self.assertIn("Real Nemotron (Nebius) Mode", html)

    def test_07_local_not_public_mode(self):
        """is_public_mode() must return False in local test environment."""
        # Local test server has no RENDER or REAGENT_PUBLIC_MODE set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RENDER", None)
            os.environ.pop("REAGENT_PUBLIC_MODE", None)
            self.assertFalse(is_public_mode())

    def test_08_default_theme_is_dark(self):
        """HTML element must declare data-theme='dark' by default."""
        _, html = self._get("/")
        self.assertIn('<html lang="en" data-theme="dark">', html)
        self.assertNotIn('<html lang="en" data-theme="light">', html)
        self.assertIn('title="Switch to light theme"', html)


if __name__ == "__main__":
    unittest.main()
