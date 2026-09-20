import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minicut_agent.candidates import rank_candidates, target_times
from minicut_agent.frame_resolver import resolve_semantic_frame
from minicut_agent.gemini import _bounded_offset
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

    def test_target_times_avoids_short_tail(self):
        duration = 62 * 60_000
        self.assertEqual(target_times(duration, 15 * 60_000), [
            15 * 60_000, 30 * 60_000, 45 * 60_000
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

    def test_semantic_offset_is_clamped_to_clip(self):
        self.assertEqual(_bounded_offset(999999, 0), 1200)
        self.assertEqual(_bounded_offset(-999999, 0), -1200)


if __name__ == "__main__":
    unittest.main()
