"""Focused source contracts for the opt-in accessibility adapter."""
from pathlib import Path
import unittest


HOST = Path(__file__).resolve().parents[1] / "host"
SERVICE = HOST / "src/ai/ouroboros/android/OuroborosAccessibilityService.java"
CONFIG = HOST / "res/xml/accessibility_service_config.xml"


class AccessibilityAdapterTest(unittest.TestCase):
    def test_service_exposes_bounded_snapshot_and_stable_snapshot_addresses(self):
        source = SERVICE.read_text()
        for marker in ("max_windows", "max_nodes", "max_depth", "truncation_reason", "node_address", "stale_or_missing_node_address"):
            self.assertIn(marker, source)
        self.assertIn('"bounded_interactive_window_tree"', source)
        self.assertIn("root.recycle()", source)
        self.assertIn("child.recycle()", source)

    def test_generic_node_actions_and_gesture_have_typed_failure_paths(self):
        source = SERVICE.read_text()
        for marker in ("ACTION_CLICK", "ACTION_SET_TEXT", "ACTION_SCROLL_FORWARD", "dispatchGesture",
                       "gesture_result_timeout", "accessibility_not_enabled"):
            self.assertIn(marker, source)

    def test_service_requests_interactive_windows_and_view_ids(self):
        config = CONFIG.read_text()
        self.assertIn("flagRetrieveInteractiveWindows", config)
        self.assertIn("flagReportViewIds", config)
        self.assertIn('android:canRetrieveWindowContent="true"', config)


if __name__ == "__main__":
    unittest.main()
