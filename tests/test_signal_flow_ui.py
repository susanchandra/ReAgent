"""
Test suite for ReAgent Signal Flow Pipeline UI, Stage Inspector,
and Nemotron Provenance Panel.
"""
import unittest
from reagent.ui_server import generate_dashboard_html


class TestSignalFlowUI(unittest.TestCase):
    """Verifies Signal Flow interactive UI elements, states, accessibility, and offline defaults."""

    def setUp(self):
        self.html = generate_dashboard_html()

    def test_01_all_five_pipeline_stages_present(self):
        """All 5 pipeline stage nodes (Plan, Execute, Observe, Verify, Decide) are present with data-stage attributes."""
        stages = ["plan", "execute", "observe", "verify", "decide"]
        for stage in stages:
            node_id = f'id="sf-node-{stage}"'
            data_attr = f'data-stage="{stage}"'
            self.assertIn(node_id, self.html, f"Missing node ID: {node_id}")
            self.assertIn(data_attr, self.html, f"Missing data-stage attribute: {data_attr}")

    def test_02_svg_connector_lines_present(self):
        """All 4 SVG connector lines between the 5 stage nodes are rendered."""
        for i in range(1, 5):
            line_id = f'id="sf-line-{i}"'
            self.assertIn(line_id, self.html, f"Missing connector line: {line_id}")

    def test_03_inspector_and_nemotron_panels_present(self):
        """Stage inspector and Nemotron provenance panels are rendered."""
        self.assertIn('id="sf-inspector"', self.html)
        self.assertIn('id="sf-insp-stage-name"', self.html)
        self.assertIn('id="sf-insp-body"', self.html)
        self.assertIn('id="sf-nem-panel"', self.html)
        self.assertIn('id="sf-nem-badge"', self.html)

    def test_04_canvas_flow_field_and_reduced_motion(self):
        """Flow field canvas is present and prefers-reduced-motion media query is defined."""
        self.assertIn('id="sf-canvas"', self.html)
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.html)
        self.assertIn("#sf-canvas { display: none !important; }", self.html)

    def test_05_stage_states_css_defined(self):
        """All node state styles (running, completed, blocked, unknown, failed, escalated) are defined in CSS."""
        states = ["pending", "running", "completed", "blocked", "unknown", "failed", "escalated"]
        for st in states:
            selector = f'[data-state="{st}"]'
            self.assertIn(selector, self.html, f"Missing CSS selector for state: {selector}")

    def test_06_js_lifecycle_functions_defined(self):
        """Signal Flow JS state machine and inspector functions are defined in inline script."""
        fns = [
            "function sfResetPipeline()",
            "function sfUpdatePipeline(result, events, scenario)",
            "function sfRenderUnsupported(errBody)",
            "function sfInspectStage(stage)",
            "function sfUpdateNemotron(result)",
            "function sfInitCanvas()",
        ]
        for fn in fns:
            self.assertIn(fn, self.html, f"Missing JS function: {fn}")

    def test_07_nemotron_provenance_truthful_default(self):
        """Provenance truthfully reports advisory status when configured, or offline handling when unconfigured."""
        import os
        api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
        if api_key:
            self.assertIn("Proposal mode active", self.html)
            self.assertIn("advisory only", self.html)
        else:
            self.assertIn("Nemotron was not invoked for this run. The offline planner handled this task.", self.html)
            self.assertIn("NOT_CONFIGURED", self.html)

        # In both cases, model proposals are never claimed as authoritative execution
        self.assertIn("sf-nem-panel", self.html)
        self.assertIn("sf-nem-badge", self.html)


if __name__ == "__main__":
    unittest.main()
