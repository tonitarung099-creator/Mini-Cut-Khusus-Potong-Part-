import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minicut_agent.candidates import (
    LocalCandidate, proximity_shortlist, rank_candidates, target_times
)
from minicut_agent.frame_resolver import resolve_semantic_frame
from minicut_agent.gemini import _bounded_offset, _needs_deep_check
from minicut_agent.subtitles import SubtitleTrack


SRT = """1
00:14:40,000 --> 00:14:48,000
Dialog sebelum pergantian scene.

2
00:15:10,000 --> 00:15:18,000
Dialog scene berikutnya.

3
00:29:50,000 --> 00:30:05,000
Dialog yang melewati target 30 menit.
"""


class SubtitleTests(unittest.TestCase):
    def test_parse_and_safe_gap(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.srt"
            path.write_text(SRT, encoding="utf-8")
            track = SubtitleTrack.load(path)
            self.assertEqual(len(track.cues), 3)
            self.assertFalse(track.is_safe_cut(14 * 60_000 + 45_000))
            self.assertTrue(track.is_safe_cut(14 * 60_000 + 58_000))
            gaps = track.gap_boundaries(14 * 60_000, 16 * 60_000)
            self.assertTrue(any(14 * 60_000 + 48_000 < x < 15 * 60_000 + 10_000 for x in gaps))

            synced = track.between_text(14 * 60_000 + 35_000, 14 * 60_000 + 55_000)
            self.assertIn("Dialog sebelum pergantian scene.", synced)
            self.assertNotIn("Dialog scene berikutnya.", synced)

    def test_target_times_avoids_short_tail(self):
        duration = 62 * 60_000
        self.assertEqual(target_times(duration, 15 * 60_000), [
            15 * 60_000, 30 * 60_000, 45 * 60_000
        ])

    def test_target_grid_stays_absolute_15_30_45_60(self):
        duration = 80 * 60_000
        self.assertEqual(target_times(duration, 15 * 60_000), [
            15 * 60_000,
            30 * 60_000,
            45 * 60_000,
            60 * 60_000,
            75 * 60_000,
        ])


class CandidateTests(unittest.TestCase):
    def test_visual_safe_candidate_wins(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.srt"
            path.write_text(SRT, encoding="utf-8")
            track = SubtitleTrack.load(path)
            target = 15 * 60_000
            visual = [target - 2_000, target + 25_000]
            silence = [target + 25_300]
            ranked = rank_candidates(target, 120_000, visual, silence, track, top_n=3)
            self.assertTrue(ranked)
            self.assertTrue(ranked[0].subtitle_safe)
            self.assertLessEqual(abs(ranked[0].time_ms - (target + 25_000)), 500)


class SemanticCutTests(unittest.TestCase):
    def test_nearest_valid_candidate_precedes_farther_stronger_candidate(self):
        target = 15 * 60_000
        near = LocalCandidate(
            target_ms=target,
            time_ms=target + 27_000,
            distance_ms=27_000,
            visual=True,
            silence=False,
            subtitle_gap=True,
            subtitle_safe=True,
            score=0.72,
        )
        far = LocalCandidate(
            target_ms=target,
            time_ms=target + 78_000,
            distance_ms=78_000,
            visual=True,
            silence=True,
            subtitle_gap=True,
            subtitle_safe=True,
            score=0.98,
        )
        shortlist = proximity_shortlist([far, near], target, max_n=4)
        self.assertEqual(shortlist[0].time_ms, target + 27_000)
        self.assertEqual(shortlist[1].time_ms, target + 78_000)

    def test_visual_change_outranks_plain_dialogue_gap(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.srt"
            path.write_text(SRT, encoding="utf-8")
            track = SubtitleTrack.load(path)
            target = 15 * 60_000
            ranked = rank_candidates(
                target,
                120_000,
                visual_points=[target + 25_000],
                silence_points=[],
                subtitles=track,
                top_n=3,
            )
            self.assertTrue(ranked)
            self.assertTrue(ranked[0].visual)
            self.assertLessEqual(abs(ranked[0].time_ms - (target + 25_000)), 700)

    def test_frame_resolver_uses_real_pts_and_prefers_before_new_content(self):
        frames = [9990, 10030, 10070, 10110, 10150]
        with patch("minicut_agent.frame_resolver.probe_frame_timestamps", return_value=frames):
            result = resolve_semantic_frame(
                Path("movie.mp4"),
                "ffprobe",
                preferred_ms=10100,
                zone_start_ms=10000,
                zone_end_ms=10160,
                subtitles=None,
                prefer_before_ms=10120,
            )
        self.assertTrue(result["frame_verified"])
        self.assertEqual(result["time_ms"], 10110)

    def test_semantic_offset_is_clamped_to_deep_window(self):
        self.assertEqual(_bounded_offset(999999, 0), 1500)
        self.assertEqual(_bounded_offset(-999999, 0), -1500)

    def test_low_confidence_requests_deep_check(self):
        self.assertTrue(_needs_deep_check({
            "confidence": 0.60,
            "scene_change": True,
            "dialog_safe": True,
            "needs_review": False,
        }))

    def test_safe_high_confidence_skips_deep_check(self):
        self.assertFalse(_needs_deep_check({
            "confidence": 0.91,
            "scene_change": True,
            "dialog_safe": True,
            "action_safe": True,
            "needs_deep_check": False,
            "needs_review": False,
        }))


if __name__ == "__main__":
    unittest.main()
