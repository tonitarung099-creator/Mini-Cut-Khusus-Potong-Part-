import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minicut_agent.core import export_segments, load_project_file


class ProjectPortabilityTests(unittest.TestCase):
    def test_relative_source_wins_over_stale_existing_absolute_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project_dir = root / "portable"
            project_dir.mkdir()
            relative_video = project_dir / "movie.mp4"
            relative_video.write_bytes(b"portable-video")
            stale = root / "old-location.mp4"
            stale.write_bytes(b"old-video")

            project = project_dir / "movie.minicut.json"
            project.write_text(json.dumps({
                "app": "MiniCut Studio",
                "version": 2,
                "source_absolute": str(stale),
                "source_relative": "movie.mp4",
                "cuts": [],
            }), encoding="utf-8")

            _data, source = load_project_file(project)
            self.assertEqual(source, relative_video.resolve())


class FastExportNamingTests(unittest.TestCase):
    def test_fast_export_starts_part_number_at_one(self):
        class DummyProc:
            def __init__(self):
                self.stdout = io.StringIO("out_time_ms=1000000\n")

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as td, patch(
            "minicut_agent.core.subprocess.Popen",
            return_value=DummyProc(),
        ) as popen:
            export_segments(
                "ffmpeg",
                Path(td) / "movie.mp4",
                Path(td) / "parts",
                "movie",
                [1_000],
                2_000,
            )

        cmd = popen.call_args.args[0]
        index = cmd.index("-segment_start_number")
        self.assertEqual(cmd[index + 1], "1")


class FastExportZeroCutTests(unittest.TestCase):
    def test_zero_cuts_exports_exactly_one_direct_part(self):
        class DummyProc:
            def __init__(self, cmd):
                self.stdout = io.StringIO("out_time_ms=1000000\n")
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"video")

            def wait(self, timeout=None):
                return 0

        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return DummyProc(cmd)

        with tempfile.TemporaryDirectory() as td, patch(
            "minicut_agent.core.subprocess.Popen",
            side_effect=fake_popen,
        ):
            count, size = export_segments(
                "ffmpeg",
                Path(td) / "movie.mp4",
                Path(td) / "parts",
                "movie",
                [],
                2_000,
            )

        self.assertEqual(count, 1)
        self.assertGreater(size, 0)
        self.assertNotIn("-f", captured["cmd"])
        self.assertTrue(str(captured["cmd"][-1]).endswith("movie_Part-01.mp4"))

    def test_success_code_without_expected_files_is_rejected(self):
        class DummyProc:
            def __init__(self):
                self.stdout = io.StringIO("out_time_ms=1000000\n")

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as td, patch(
            "minicut_agent.core.subprocess.Popen",
            return_value=DummyProc(),
        ):
            with self.assertRaisesRegex(RuntimeError, "hasil part tidak lengkap"):
                export_segments(
                    "ffmpeg",
                    Path(td) / "movie.mp4",
                    Path(td) / "parts",
                    "movie",
                    [1_000],
                    2_000,
                )


if __name__ == "__main__":
    unittest.main()
