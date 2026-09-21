import unittest

from minicut_agent.agent import ToolRegistry
from minicut_agent.ui import MiniCutWindow


class ToolManifestCoverageTests(unittest.TestCase):
    def test_every_manifest_tool_has_ui_executor(self):
        registry = ToolRegistry(object())
        missing = [
            spec["name"]
            for spec in registry.manifest()["tools"]
            if not hasattr(MiniCutWindow, f"tool_{spec['name']}")
        ]
        self.assertEqual(missing, [])

    def test_major_gemini_controls_are_exposed(self):
        names = {x["name"] for x in ToolRegistry(object()).manifest()["tools"]}
        expected = {
            "open_video", "open_project", "choose_subtitle",
            "seek", "play", "pause", "set_playback_rate", "step_frame",
            "add_cut", "remove_cut", "clear_cuts", "divide_equal", "divide_interval",
            "start_film_cut", "cancel_film_cut", "apply_film_cut", "preview_film_cut",
            "set_export_mode", "save_project", "export_all", "undo",
        }
        self.assertTrue(expected.issubset(names))


if __name__ == "__main__":
    unittest.main()
