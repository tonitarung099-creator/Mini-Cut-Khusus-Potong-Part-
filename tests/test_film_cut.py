import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minicut_agent.candidates import (
    LocalCandidate, proximity_shortlist, rank_candidates, target_times
)
from minicut_agent.core import ProjectModel, _clear_export_parts, _export_part_files
from minicut_agent.frame_resolver import (
    probe_frame_points, resolve_requested_frame, resolve_semantic_frame
)
from minicut_agent.workers import FilmCutWorker
from minicut_agent.gemini import (
    GeminiClient, _bounded_offset, _evenly_sample_times, _needs_deep_check
)
from minicut_agent.manual_commands import extract_manual_timestamps, looks_like_manual_cut
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


class ManualCommandTests(unittest.TestCase):
    def test_dot_timestamps_are_minutes_seconds(self):
        items = extract_manual_timestamps(
            "Potong di menit 15.32, 31.12, 45.10, 59.34"
        )
        self.assertTrue(looks_like_manual_cut("Potong di menit 15.32"))
        self.assertEqual(
            [x.time_ms for x in items],
            [
                15 * 60_000 + 32_000,
                31 * 60_000 + 12_000,
                45 * 60_000 + 10_000,
                59 * 60_000 + 34_000,
            ],
        )

    def test_hh_mm_ss_and_natural_indonesian(self):
        items = extract_manual_timestamps(
            "cut 01:15:32 dan 20 menit 7 detik"
        )
        self.assertEqual(items[0].time_ms, (3600 + 15 * 60 + 32) * 1000)
        self.assertEqual(items[1].time_ms, (20 * 60 + 7) * 1000)

    def test_comma_separated_timestamps_without_spaces(self):
        items = extract_manual_timestamps(
            "Potong 15.32,31.12,45.10,59.34"
        )
        self.assertEqual(
            [x.time_ms for x in items],
            [
                15 * 60_000 + 32_000,
                31 * 60_000 + 12_000,
                45 * 60_000 + 10_000,
                59 * 60_000 + 34_000,
            ],
        )

    def test_requested_frame_uses_nearest_real_pts_only(self):
        with patch(
            "minicut_agent.frame_resolver.probe_frame_timestamps",
            return_value=[931_960, 932_000, 932_040],
        ):
            result = resolve_requested_frame(
                Path("movie.mp4"),
                "ffprobe",
                requested_ms=932_013,
            )
        self.assertTrue(result["frame_verified"])
        self.assertEqual(result["time_ms"], 932_000)
        self.assertEqual(result["frame_delta_ms"], -13)


    def test_frame_cut_does_not_snap_to_keyframe(self):
        model = ProjectModel()
        model.duration_ms = 2_000_000
        model.keyframes = [900_000, 940_000]
        cut = model.add_frame_cut(932_000, 932_013)
        self.assertEqual(cut.requested_ms, 932_000)
        self.assertEqual(cut.actual_ms, 932_013)


class GeminiExactFrameAuthorityTests(unittest.TestCase):
    def test_exact_frame_result_is_one_of_master_pts_and_has_zero_local_delta(self):
        client = GeminiClient("dummy-key", "dummy-model")
        frames = [
            {
                "time_ms": 10_000 + (i * 40),
                "exact_time": f"{250 + i}/25",
            }
            for i in range(80)
        ]

        with patch.object(
            client,
            "_choose_exact_frame_pass",
            side_effect=[
                {"selected_frame_index": 7, "confidence": 0.8},
                {
                    "selected_frame_index": 16,
                    "confidence": 0.95,
                    "needs_review": False,
                    "reason": "frame transisi final",
                },
            ],
        ):
            result = client.choose_exact_master_frame(
                "ffmpeg",
                Path("movie.mp4"),
                target_ms=10_000,
                boundary_hint_ms=11_500,
                frame_points=frames,
                subtitles=None,
            )

        self.assertIn(
            result["selected_time_ms"],
            [item["time_ms"] for item in frames],
        )
        self.assertIn(
            result["selected_time_exact"],
            [item["exact_time"] for item in frames],
        )
        self.assertEqual(result["frame_delta_ms"], 0)
        self.assertEqual(result["frame_authority"], "gemini")
        self.assertEqual(result["frame_selection"], "gemini-exact-master-pts")

    def test_probe_frame_points_preserves_rational_pts(self):
        payload = {
            "streams": [{"time_base": "1/24000"}],
            "format": {"start_time": "0.000000"},
            "frames": [
                {
                    "best_effort_timestamp": 1001,
                    "best_effort_timestamp_time": "0.041708",
                }
            ],
        }
        fake = type("Result", (), {
            "returncode": 0,
            "stdout": __import__("json").dumps(payload),
            "stderr": "",
        })()
        with patch(
            "minicut_agent.frame_resolver.run_text",
            return_value=fake,
        ):
            points = probe_frame_points(
                Path("movie.mp4"),
                "ffprobe",
                0,
                100,
            )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["time_ms"], 42)
        self.assertEqual(points[0]["exact_time"], "1001/24000")

    def test_probe_frame_points_handles_nonzero_container_start(self):
        clock_payload = {
            "streams": [{"time_base": "1/1000"}],
            "format": {"start_time": "5.000000"},
        }
        frame_payload = {
            "frames": [{
                "best_effort_timestamp": 5042,
                "best_effort_timestamp_time": "5.042000",
            }],
        }
        results = [
            type("Result", (), {
                "returncode": 0,
                "stdout": __import__("json").dumps(clock_payload),
                "stderr": "",
            })(),
            type("Result", (), {
                "returncode": 0,
                "stdout": __import__("json").dumps(frame_payload),
                "stderr": "",
            })(),
        ]
        with patch(
            "minicut_agent.frame_resolver.run_text",
            side_effect=results,
        ):
            points = probe_frame_points(
                Path("movie.ts"),
                "ffprobe",
                0,
                100,
            )

        self.assertEqual(points[0]["time_ms"], 42)
        self.assertEqual(points[0]["exact_time"], "21/500")

    def test_even_sampling_keeps_real_values_only(self):
        frames = [1_000 + i * 41 for i in range(100)]
        sampled = _evenly_sample_times(frames, 13)
        self.assertLessEqual(len(sampled), 13)
        self.assertEqual(sampled[0], frames[0])
        self.assertEqual(sampled[-1], frames[-1])
        self.assertTrue(all(value in frames for value in sampled))


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


class ExportCleanupTests(unittest.TestCase):
    def test_previous_parts_are_removed_without_glob_name_bug(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            base = "film[versi-1]"
            stale1 = out / f"{base}_Part-01.mp4"
            stale2 = out / f"{base}_Part-10.mp4"
            keep = out / "film-lain_Part-01.mp4"
            user_file = out / f"{base}_Part-catatan.mp4"
            stale1.write_bytes(b"old-1")
            stale2.write_bytes(b"old-2")
            keep.write_bytes(b"keep")
            user_file.write_bytes(b"user-data")

            self.assertEqual(
                [p.name for p in _export_part_files(out, base, ".mp4")],
                [stale1.name, stale2.name],
            )
            _clear_export_parts(out, base, ".mp4")

            self.assertFalse(stale1.exists())
            self.assertFalse(stale2.exists())
            self.assertTrue(keep.exists())
            self.assertTrue(user_file.exists())


class FilmCutCacheTests(unittest.TestCase):
    def test_cache_without_exact_pts_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "movie.mp4"
            srt = root / "movie.srt"
            source.write_bytes(b"video")
            srt.write_text("subtitle", encoding="utf-8")

            worker = FilmCutWorker(
                "ffmpeg",
                "ffprobe",
                source,
                120_000,
                srt,
                "dummy-key",
                "dummy-model",
            )
            worker._save_cache([{
                "target_ms": 60_000,
                "selected_time_ms": 61_000,
            }])
            self.assertEqual(worker._load_cache(), {})

    def test_cache_is_invalidated_when_srt_contents_change_at_same_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "movie.mp4"
            srt = root / "movie.srt"
            source.write_bytes(b"video-a")
            srt.write_text("subtitle-a", encoding="utf-8")

            worker = FilmCutWorker(
                "ffmpeg",
                "ffprobe",
                source,
                120_000,
                srt,
                "dummy-key",
                "dummy-model",
            )
            worker._save_cache([{
                "target_ms": 60_000,
                "selected_time_ms": 61_000,
                "selected_time_exact": "61",
            }])
            self.assertIn(60_000, worker._load_cache())

            srt.write_text("subtitle-b-different", encoding="utf-8")
            self.assertEqual(worker._load_cache(), {})

    def test_cache_is_invalidated_when_video_contents_change_at_same_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "movie.mp4"
            srt = root / "movie.srt"
            source.write_bytes(b"video-a")
            srt.write_text("subtitle-a", encoding="utf-8")

            worker = FilmCutWorker(
                "ffmpeg",
                "ffprobe",
                source,
                120_000,
                srt,
                "dummy-key",
                "dummy-model",
            )
            worker._save_cache([{
                "target_ms": 60_000,
                "selected_time_ms": 61_000,
                "selected_time_exact": "61",
            }])
            self.assertIn(60_000, worker._load_cache())

            source.write_bytes(b"video-b-longer")
            self.assertEqual(worker._load_cache(), {})


if __name__ == "__main__":
    unittest.main()
