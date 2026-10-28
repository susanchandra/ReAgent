import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path

from reagent.ui_server import DashboardHTTPHandler, generate_dashboard_html


class TestIntroVideoIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_video_file_exists_on_disk(self):
        video_path = Path(__file__).resolve().parent.parent / "assets" / "video" / "reagent-intro.mp4"
        self.assertTrue(video_path.is_file(), f"Video file not found at {video_path}")
        self.assertGreater(video_path.stat().st_size, 10_000_000)

    def test_video_full_get_stream(self):
        req = urllib.request.Request(self._url("/assets/video/reagent-intro.mp4"))
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "video/mp4")
            self.assertEqual(resp.headers.get("Accept-Ranges"), "bytes")
            content_length = int(resp.headers.get("Content-Length", 0))
            self.assertGreater(content_length, 10_000_000)
            # Read first 1KB to confirm streaming works
            chunk = resp.read(1024)
            self.assertEqual(len(chunk), 1024)

    def test_video_range_request_partial_content(self):
        req = urllib.request.Request(self._url("/assets/video/reagent-intro.mp4"))
        req.add_header("Range", "bytes=0-1023")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 206)
            self.assertEqual(resp.headers.get("Content-Type"), "video/mp4")
            self.assertTrue(resp.headers.get("Content-Range", "").startswith("bytes 0-1023/"))
            self.assertEqual(resp.headers.get("Content-Length"), "1024")
            data = resp.read()
            self.assertEqual(len(data), 1024)

    def test_video_head_request(self):
        req = urllib.request.Request(self._url("/assets/video/reagent-intro.mp4"), method="HEAD")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "video/mp4")
            self.assertEqual(resp.headers.get("Accept-Ranges"), "bytes")
            body = resp.read()
            self.assertEqual(len(body), 0)

    def test_html_contains_intro_video_elements_and_lifecycle(self):
        html = generate_dashboard_html()
        self.assertIn('id="heroIntroVideo"', html)
        self.assertIn('src="/assets/video/reagent-intro.mp4"', html)
        self.assertIn('poster="data:image/jpeg;base64,', html)
        self.assertIn('muted playsinline preload="metadata"', html)
        self.assertIn('id="heroStaticLogo"', html)
        self.assertIn('id="heroReplayBtn"', html)
        self.assertIn("window._reagentIntro", html)
        self.assertIn("_introShowStatic", html)
        self.assertIn("_introStartPlayback", html)
        self.assertIn("heroReplayIntro", html)
        self.assertIn("prefers-reduced-motion", html)

    def test_hero_fullwidth_layout_and_lockup_structure(self):
        html = generate_dashboard_html()
        # Full-width background hero stage exists, old card is removed
        self.assertIn('class="hero-stage"', html)
        self.assertIn('class="hero-video-bg"', html)
        self.assertIn('class="hero-scrim"', html)
        self.assertNotIn('hero-brand-panel', html)
        self.assertNotIn('id="heroBrandPanel"', html)

        # Left 50-55% content overlay
        self.assertIn('class="hero-left-content"', html)
        self.assertIn("When agents fail,", html)
        self.assertIn("ReAgent tells", html)
        self.assertIn("UNKNOWN is not ABSENT", html)

        # Unified lockup: symbol on left, wordmark immediately to right
        self.assertIn('class="hero-post-lockup"', html)
        self.assertIn('class="hero-lockup-inner"', html)
        self.assertIn('class="hero-lockup-symbol"', html)
        self.assertIn('class="hero-lockup-symbol-img"', html)
        self.assertIn('class="hero-lockup-brand"', html)
        self.assertIn('class="hero-lockup-wordmark-img"', html)

        # Confirm symbol appears before wordmark in lockup HTML
        idx_symbol = html.find('class="hero-lockup-symbol"')
        idx_brand = html.find('class="hero-lockup-brand"')
        self.assertNotEqual(idx_symbol, -1)
        self.assertNotEqual(idx_brand, -1)
        self.assertLess(idx_symbol, idx_brand, "Symbol must be on left (before) wordmark on right")

        # Five cards remain below the hero stage
        idx_hero_stage = html.find('id="heroStage"')
        idx_cards = html.find("The Intelligence Behind Every Decision")
        self.assertNotEqual(idx_hero_stage, -1)
        self.assertNotEqual(idx_cards, -1)
        self.assertLess(idx_hero_stage, idx_cards, "Workflow cards must be below the hero stage")


if __name__ == "__main__":
    unittest.main()
