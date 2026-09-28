import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minicut_agent.camera_boundary import resolve_camera_boundary
from minicut_agent.camera_cut_window import CameraAwareMiniCutWindow
from minicut_agent.core import ProjectModel
from minicut_agent.manual_commands import extract_manual_timestamps, looks_like_manual_cut


class CameraBoundaryResolverTests(unittest.TestCase):
    def test_prefers_nearest_camera_change_then_locks_exact_master_pts(self):
        points = [
            {"time_ms": 10_042, "exact_time": "251/25"},
            {"time_ms": 10_083, "exact_time": "10083/1000"},
        ]
        with patch(
            "minicut_agent.camera_boundary.detect_visual_boundaries",
            return_value=[10_050, 11_200],
        ), patch(
            "minicut_agent.camera_boundary.probe_frame_points",
            return_value=points,
        ):
            result = resolve_camera_boundary(
                Path("movie.mp4"),
                "ffmpeg",
                "ffprobe",
                10_000,
                duration_ms=60_000,
            )

        self.assertTrue(result["camera_change_found"])
        self.assertFalse(result["fallback_to_nearest_frame"])
        self.assertEqual(result["raw_scene_boundary_ms"], 10_050)
        self.assertEqual(result["time_ms"], 10_042)
        self.assertEqual(result["exact_time"], "251/25")
        self.assertEqual(result["shift_ms"], 42)
        self.assertEqual(result["search_radius_ms"], 2_000)

    def test_expands_search_to_four_seconds_when_first_window_has_no_cut(self):
        with patch(
            "minicut_agent.camera_boundary.detect_visual_boundaries",
            side_effect=[[], [13_000]],
        ) as detect, patch(
            "minicut_agent.camera_boundary.probe_frame_points",
            return_value=[{"time_ms": 13_000, "exact_time": "13"}],
        ):
            result = resolve_camera_boundary(
                Path("movie.mp4"),
                "ffmpeg",
                "ffprobe",
                10_000,
                duration_ms=60_000,
            )

        self.assertEqual(detect.call_count, 2)
        self.assertTrue(result["camera_change_found"])
        self.assertEqual(result["search_radius_ms"], 4_000)
        self.assertEqual(result["time_ms"], 13_000)

    def test_falls_back_to_nearest_real_frame_when_no_camera_change_exists(self):
        points = [
            {"time_ms": 9_958, "exact_time": "4979/500"},
            {"time_ms": 10_000, "exact_time": "10"},
            {"time_ms": 10_042, "exact_time": "5021/500"},
        ]
        with patch(
            "minicut_agent.camera_boundary.detect_visual_boundaries",
            return_value=[],
        ), patch(
            "minicut_agent.camera_boundary.probe_frame_points",
            return_value=points,
        ):
            result = resolve_camera_boundary(
                Path("movie.mp4"),
                "ffmpeg",
                "ffprobe",
                10_000,
                duration_ms=60_000,
            )

        self.assertFalse(result["camera_change_found"])
        self.assertTrue(result["fallback_to_nearest_frame"])
        self.assertEqual(result["time_ms"], 10_000)
        self.assertEqual(result["exact_time"], "10")


class CameraAwareToolTests(unittest.TestCase):
    def test_manual_cut_keeps_requested_target_but_uses_camera_boundary_as_actual(self):
        class ExportModeStub:
            def __init__(self):
                self.value = "fast"

            def findData(self, value):
                return 0 if value == "smartcut" else -1

            def setCurrentIndex(self, _index):
                self.value = "smartcut"

        class Host:
            _require_tool_project_ready = CameraAwareMiniCutWindow._require_tool_project_ready
            _require_tool_media_ready = CameraAwareMiniCutWindow._require_tool_media_ready
            tool_add_cut = CameraAwareMiniCutWindow.tool_add_cut

            def _refresh(self):
                pass

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")

            host = Host()
            host.model = ProjectModel()
            host.model.source = source
            host.model.duration_ms = 120_000
            host.analyze_worker = None
            host.export_mode = ExportModeStub()

            resolved = {
                "time_ms": 61_208,
                "exact_time": "15302/250",
                "camera_change_found": True,
                "fallback_to_nearest_frame": False,
                "shift_ms": -29,
                "search_radius_ms": 2_000,
                "scene_threshold": 0.27,
                "reason": "camera boundary",
            }
            with patch(
                "minicut_agent.camera_cut_window.find_tool",
                side_effect=lambda name: name,
            ), patch(
                "minicut_agent.camera_cut_window.resolve_camera_boundary",
                return_value=resolved,
            ):
                result = host.tool_add_cut(61_237)

        self.assertTrue(result["ok"])
        self.assertTrue(result["requested_timestamp_preserved"])
        self.assertFalse(result["exact_timestamp_preserved"])
        self.assertTrue(result["camera_change_found"])
        self.assertEqual(host.model.cuts[0].requested_ms, 61_237)
        self.assertEqual(host.model.cuts[0].actual_ms, 61_208)
        self.assertEqual(host.model.cuts[0].exact_time, "15302/250")
        self.assertEqual(result["frame_authority"], "local-camera-boundary")
        self.assertEqual(host.export_mode.value, "smartcut")


class SamawaCutListParserTests(unittest.TestCase):
    def test_pasted_samawa_style_list_extracts_all_six_cut_targets(self):
        text = """Samawa 2025
===========
Titik Potong 1 : 00:14:55.500
Titik Potong 2 : 00:31:14.750
Titik Potong 3 : 00:49:21.500
Titik Potong 4 : 01:02:13.000
Titik Potong 5 : 01:17:35.500
Titik Potong 6 : 01:30:57.000
"""
        items = extract_manual_timestamps(text)

        self.assertTrue(looks_like_manual_cut(text))
        self.assertEqual(
            [item.time_ms for item in items],
            [
                895_500,
                1_874_750,
                2_961_500,
                3_733_000,
                4_655_500,
                5_457_000,
            ],
        )


if __name__ == "__main__":
    unittest.main()
