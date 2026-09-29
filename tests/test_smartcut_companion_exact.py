from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from unittest.mock import patch

import smartcut_runner


class _FakeStream:
    time_base = Fraction(1, 24)


class _FakeSource:
    def __init__(self):
        self.video_stream = _FakeStream()
        self.start_time = Fraction(10, 1)
        # First displayed frame is absolute PTS 10s. There are 200 frames.
        self.video_frame_times_pts = list(range(240, 440))
        self.closed = False

    def close(self):
        self.closed = True


class SmartCutExactFramePlanTests(unittest.TestCase):
    def test_exact_boundary_maps_to_master_frame_index_without_tolerance(self):
        source = _FakeSource()
        self.assertEqual(
            smartcut_runner._exact_frame_index(source, "97/24"),
            97,
        )
        with self.assertRaisesRegex(RuntimeError, "grid time_base"):
            smartcut_runner._exact_frame_index(source, "1/100")

    def test_part_before_boundary_expects_k_frames_and_builds_one_frame_retry(self):
        source = _FakeSource()
        plan = smartcut_runner._build_exact_keep_plan(source, "start,97/24")
        self.assertEqual(plan.expected_frames, 97)
        # Retry end is frame K+1, relative to the same zero-based timeline.
        self.assertEqual(plan.retry_keep, "start,49/12")

    def test_part_after_boundary_starts_at_k_and_keeps_remaining_frames(self):
        source = _FakeSource()
        plan = smartcut_runner._build_exact_keep_plan(source, "97/24,end")
        self.assertEqual(plan.expected_frames, 200 - 97)
        self.assertIsNone(plan.retry_keep)

    def test_exact_fraction_syntax_activates_hardening_on_staging_output(self):
        final_name = [
            "movie.mp4",
            "movie_Part-01.mp4",
            "--keep",
            "start,97/24",
        ]
        staging_name = [
            "movie.mp4",
            ".movie_Part-01.minicut-stage-abc.mp4",
            "--keep",
            "start,97/24",
        ]
        rounded_fallback = [
            "movie.mp4",
            ".movie_Part-01.minicut-stage-abc.mp4",
            "--keep",
            "start,4.041667",
        ]
        self.assertTrue(
            smartcut_runner._looks_like_minicut_exact_keep(final_name)
        )
        self.assertTrue(
            smartcut_runner._looks_like_minicut_exact_keep(staging_name)
        )
        self.assertFalse(
            smartcut_runner._looks_like_minicut_exact_keep(rounded_fallback)
        )


class SmartCutExactFrameRetryTests(unittest.TestCase):
    def test_main_retries_only_one_missing_final_frame_on_staging_output(self):
        source = _FakeSource()
        argv = [
            "MiniCut SmartCut.exe",
            "movie.mp4",
            ".movie_Part-01.minicut-stage-abc.mp4",
            "--keep",
            "start,97/24",
            "--log-level",
            "warning",
        ]
        runs: list[list[str]] = []

        def record_run(args):
            runs.append(list(args))

        with patch.object(sys, "argv", argv), patch(
            "smartcut_runner.MediaContainer",
            return_value=source,
        ), patch(
            "smartcut_runner._run_upstream",
            side_effect=record_run,
        ), patch(
            "smartcut_runner._video_frame_count",
            side_effect=[96, 97],
        ):
            smartcut_runner.main()

        self.assertEqual(len(runs), 2)
        self.assertEqual(
            runs[0][runs[0].index("--keep") + 1],
            "start,97/24",
        )
        self.assertEqual(
            runs[1][runs[1].index("--keep") + 1],
            "start,49/12",
        )

    def test_main_rejects_mismatch_larger_than_one_frame(self):
        source = _FakeSource()
        argv = [
            "MiniCut SmartCut.exe",
            "movie.mp4",
            ".movie_Part-01.minicut-stage-abc.mp4",
            "--keep",
            "start,97/24",
        ]
        with patch.object(sys, "argv", argv), patch(
            "smartcut_runner.MediaContainer",
            return_value=source,
        ), patch(
            "smartcut_runner._run_upstream",
        ), patch(
            "smartcut_runner._video_frame_count",
            return_value=95,
        ):
            with self.assertRaisesRegex(RuntimeError, "jumlah frame"):
                smartcut_runner.main()


if __name__ == "__main__":
    unittest.main()
