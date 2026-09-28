import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from minicut_agent.frame_resolver import (
    probe_frame_timestamps,
    probe_keyframes_relative,
)


class FrameResolverStartTimeTests(unittest.TestCase):
    def test_frame_timestamps_are_relative_to_nonzero_container_start(self):
        discovery = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({
                "format": {"start_time": "1.500"},
                "frames": [],
            }),
            stderr="",
        )
        corrected_frames = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({
                "frames": [
                    {"best_effort_timestamp_time": "2.500"},
                    {"best_effort_timestamp_time": "2.540"},
                    {"best_effort_timestamp_time": "2.600"},
                ]
            }),
            stderr="",
        )

        with patch(
            "minicut_agent.frame_resolver.run_text",
            side_effect=[discovery, corrected_frames],
        ) as runner:
            result = probe_frame_timestamps(
                Path("movie.ts"),
                "ffprobe",
                1000,
                1100,
            )

        self.assertEqual(result, [1000, 1040, 1100])
        self.assertEqual(runner.call_count, 2)
        frame_cmd = runner.call_args_list[1].args[0]
        interval = frame_cmd[frame_cmd.index("-read_intervals") + 1]
        self.assertEqual(interval, "2.5%2.6")

    def test_keyframes_are_relative_to_nonzero_container_start(self):
        clock = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"format": {"start_time": "1.500"}}),
            stderr="",
        )
        keyframes = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="1.500000\n3.500000\n5.500000\n",
            stderr="",
        )

        with patch(
            "minicut_agent.frame_resolver.run_text",
            side_effect=[clock, keyframes],
        ):
            result = probe_keyframes_relative(Path("movie.ts"), "ffprobe")

        self.assertEqual(result, [0, 2000, 4000])


if __name__ == "__main__":
    unittest.main()
