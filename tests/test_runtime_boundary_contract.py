from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from minicut_agent.core import ProjectModel
from minicut_agent.gemini import GeminiUsage
from minicut_agent.gemini_boundary import BoundaryAwareGeminiClient
from minicut_agent.runtime_window import RuntimeMiniCutWindow


class RuntimeFilmCutVerificationTests(unittest.TestCase):
    def _host(self, source: Path):
        host = object.__new__(RuntimeMiniCutWindow)
        host.model = ProjectModel()
        host.model.source = source
        host.model.duration_ms = 60_000
        return host

    def test_ai_exact_pts_is_accepted_only_when_local_boundary_matches(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")
            host = self._host(source)
            result = {
                "selected_time_ms": 12_804,
                "selected_time": "00:00:12.804",
                "selected_time_exact": "307307/24000",
                "frame_verified": True,
                "needs_review": False,
            }
            resolved = {
                "time_ms": 12_804,
                "exact_time": "307307/24000",
                "camera_boundary_verified": True,
                "pts_verified": True,
                "pts": 307307,
                "time_base": "1/24000",
                "raw_scene_boundary_exact": "307307/24000",
            }
            with patch(
                "minicut_agent.runtime_window.find_tool",
                side_effect=lambda name: name,
            ), patch(
                "minicut_agent.runtime_window.resolve_camera_boundary",
                return_value=resolved,
            ):
                checked = host._verify_film_cut_results([result])[0]

            self.assertTrue(checked["camera_boundary_verified"])
            self.assertTrue(checked["pts_verified"])
            self.assertEqual(checked["selected_time_ms"], 12_804)
            self.assertEqual(checked["selected_time_exact"], "307307/24000")
            self.assertEqual(checked["selected_source_pts"], 307307)

    def test_ai_nearby_boundary_is_reviewed_and_never_silently_snapped(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")
            host = self._host(source)
            gemini_exact = Fraction(308308, 24000)
            result = {
                "selected_time_ms": 12_846,
                "selected_time": "00:00:12.846",
                "selected_time_exact": str(gemini_exact),
                "frame_verified": True,
                "needs_review": False,
            }
            resolved = {
                "time_ms": 12_804,
                "exact_time": "307307/24000",
                "camera_boundary_verified": True,
                "pts_verified": True,
                "pts": 307307,
                "time_base": "1/24000",
                "raw_scene_boundary_exact": "307307/24000",
            }
            with patch(
                "minicut_agent.runtime_window.find_tool",
                side_effect=lambda name: name,
            ), patch(
                "minicut_agent.runtime_window.resolve_camera_boundary",
                return_value=resolved,
            ):
                checked = host._verify_film_cut_results([result])[0]

            self.assertFalse(checked["camera_boundary_verified"])
            self.assertTrue(checked["needs_review"])
            self.assertFalse(checked["fallback_to_nearest_frame"])
            # Critical contract: Gemini's selection stays untouched.
            self.assertEqual(checked["selected_time_ms"], 12_846)
            self.assertEqual(Fraction(checked["selected_time_exact"]), gemini_exact)
            self.assertEqual(checked["verified_candidate_time_ms"], 12_804)
            self.assertEqual(checked["verified_candidate_exact"], "307307/24000")


class HighFpsGeminiRefinementTests(unittest.TestCase):
    class FakeClient(BoundaryAwareGeminiClient):
        def __init__(self):
            self.usage = GeminiUsage()
            self.calls: list[tuple[str, list[dict]]] = []

        def _choose_exact_frame_pass(
            self,
            ffmpeg,
            source,
            target_ms,
            hint_ms,
            frame_points,
            subtitles,
            phase,
        ):
            self.calls.append((phase, list(frame_points)))
            center = min(
                range(len(frame_points)),
                key=lambda i: abs(int(frame_points[i]["time_ms"]) - 2500),
            )
            return {
                "selected_frame_index": center + 1,
                "confidence": 0.95,
                "needs_review": False,
                "cut_intent": "scene_transition",
                "reason": "fixture",
            }

    def test_120fps_fine_pass_keeps_time_context_not_fixed_31_frames(self):
        client = self.FakeClient()
        points = []
        for n in range(0, 601):
            exact = Fraction(n, 120)
            points.append({
                "time_ms": int(exact * 1000 + Fraction(1, 2)),
                "exact_time": str(exact),
                "seek_time": str(float(exact)),
            })

        result = client.choose_exact_master_frame(
            "ffmpeg",
            Path("movie.mp4"),
            2500,
            2500,
            points,
            None,
        )

        self.assertEqual([phase for phase, _ in client.calls], ["coarse", "refine", "fine"])
        fine_points = client.calls[-1][1]
        # At 120 fps, ±250 ms is roughly 61 real frames. The old fixed 31-frame
        # window covered only ~258 ms total and could exclude the true boundary.
        self.assertGreaterEqual(len(fine_points), 55)
        fine_span = int(fine_points[-1]["time_ms"]) - int(fine_points[0]["time_ms"])
        self.assertGreaterEqual(fine_span, 450)
        self.assertLessEqual(len(fine_points), 65)
        self.assertEqual(result["selected_time_ms"], 2500)
        self.assertGreaterEqual(result["exact_frame_fine_span_ms"], 450)


if __name__ == "__main__":
    unittest.main()
