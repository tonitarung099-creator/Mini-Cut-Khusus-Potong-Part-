import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minicut_agent.core import (
    CutPoint, ProjectModel, export_segments, export_segments_smartcut,
    load_project_file, resolve_project_subtitle
)


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
            def __init__(self, cmd):
                self.stdout = io.StringIO("out_time_ms=1000000\n")
                out_dir = Path(cmd[-1]).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "movie_Part-01.mp4").write_bytes(b"part-1")
                (out_dir / "movie_Part-02.mp4").write_bytes(b"part-2")

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
            count, _size = export_segments(
                "ffmpeg",
                Path(td) / "movie.mp4",
                Path(td) / "parts",
                "movie",
                [1_000],
                2_000,
            )

        self.assertEqual(count, 2)
        cmd = captured["cmd"]
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


class SmartCutExactPtsTests(unittest.TestCase):
    def test_smartcut_receives_exact_rational_pts_without_ms_rounding(self):
        captured = []

        class DummyProc:
            def __init__(self, cmd):
                captured.append(cmd)
                Path(cmd[2]).write_bytes(b"part")
                self.returncode = 0

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as td, patch(
            "minicut_agent.core.subprocess.Popen",
            side_effect=lambda cmd, **kwargs: DummyProc(cmd),
        ):
            count, _size = export_segments_smartcut(
                "smartcut",
                Path(td) / "movie.mp4",
                Path(td) / "parts",
                "movie",
                [42],
                1000,
                cut_exact_times=["1001/24000"],
            )

        self.assertEqual(count, 2)
        keeps = [
            cmd[cmd.index("--keep") + 1]
            for cmd in captured
        ]
        self.assertEqual(
            keeps,
            ["start,1001/24000", "1001/24000,end"],
        )


class SrtExportIntegrationTests(unittest.TestCase):
    def test_fast_export_installs_matching_srt_parts_with_zero_based_timing(self):
        class SuccessProc:
            def __init__(self, cmd):
                self.stdout = io.StringIO("out_time_ms=10000000\n")
                out_dir = Path(cmd[-1]).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "movie_Part-01.mp4").write_bytes(b"part-1")
                (out_dir / "movie_Part-02.mp4").write_bytes(b"part-2")

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            srt = root / "movie.srt"
            srt.write_text(
                "1\n"
                "00:00:04,500 --> 00:00:06,500\n"
                "lintas batas\n\n"
                "2\n"
                "00:00:07,000 --> 00:00:08,000\n"
                "part dua\n",
                encoding="utf-8",
            )
            out = root / "movie_Parts"

            with patch(
                "minicut_agent.core.subprocess.Popen",
                side_effect=lambda cmd, **kwargs: SuccessProc(cmd),
            ):
                count, _size = export_segments(
                    "ffmpeg",
                    root / "movie.mp4",
                    out,
                    "movie",
                    [5_000],
                    10_000,
                    srt_path=srt,
                )

            self.assertEqual(count, 2)
            self.assertTrue((out / "movie_Part-01.mp4").is_file())
            self.assertTrue((out / "movie_Part-02.mp4").is_file())
            part1 = (out / "movie_Part-01.srt").read_text(encoding="utf-8")
            part2 = (out / "movie_Part-02.srt").read_text(encoding="utf-8")
            self.assertIn(
                "00:00:04,500 --> 00:00:05,000\nlintas batas",
                part1,
            )
            self.assertIn(
                "00:00:00,000 --> 00:00:01,500\nlintas batas",
                part2,
            )
            self.assertIn(
                "00:00:02,000 --> 00:00:03,000\npart dua",
                part2,
            )


    def test_smartcut_export_installs_matching_srt_parts(self):
        class SmartCutProc:
            def __init__(self, cmd):
                Path(cmd[2]).write_bytes(b"smartcut-part")
                self.returncode = 0

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            srt = root / "movie.srt"
            srt.write_text(
                "1\n"
                "00:00:04,500 --> 00:00:06,500\n"
                "lintas smartcut\n",
                encoding="utf-8",
            )
            out = root / "movie_Parts"

            with patch(
                "minicut_agent.core.subprocess.Popen",
                side_effect=lambda cmd, **kwargs: SmartCutProc(cmd),
            ):
                count, _size = export_segments_smartcut(
                    "smartcut",
                    root / "movie.mp4",
                    out,
                    "movie",
                    [5_000],
                    10_000,
                    cut_exact_times=["5"],
                    srt_path=srt,
                )

            self.assertEqual(count, 2)
            self.assertTrue((out / "movie_Part-01.mp4").is_file())
            self.assertTrue((out / "movie_Part-02.mp4").is_file())
            self.assertIn(
                "00:00:00,000 --> 00:00:01,500\nlintas smartcut",
                (out / "movie_Part-02.srt").read_text(encoding="utf-8"),
            )


    def test_smartcut_srt_boundary_uses_exact_pts_rounded_to_millisecond(self):
        class SmartCutProc:
            def __init__(self, cmd):
                Path(cmd[2]).write_bytes(b"smartcut-part")
                self.returncode = 0

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            srt = root / "movie.srt"
            srt.write_text(
                "1\n"
                "00:00:00,040 --> 00:00:00,050\n"
                "frame boundary\n",
                encoding="utf-8",
            )
            out = root / "movie_Parts"

            with patch(
                "minicut_agent.core.subprocess.Popen",
                side_effect=lambda cmd, **kwargs: SmartCutProc(cmd),
            ):
                export_segments_smartcut(
                    "smartcut",
                    root / "movie.mp4",
                    out,
                    "movie",
                    [42],
                    100,
                    cut_exact_times=["41/1000"],
                    srt_path=srt,
                )

            part2 = (out / "movie_Part-02.srt").read_text(encoding="utf-8")
            self.assertIn(
                "00:00:00,000 --> 00:00:00,009\nframe boundary",
                part2,
            )


class ProjectAtomicSaveTests(unittest.TestCase):
    def test_project_save_replaces_temp_and_leaves_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "movie.mp4"
            source.write_bytes(b"video")
            project = root / "movie.minicut.json"

            model = ProjectModel()
            model.source = source.resolve()
            model.duration_ms = 10_000
            model.dirty = True
            model.save(project)

            self.assertTrue(project.is_file())
            self.assertFalse((root / "movie.minicut.json.tmp").exists())
            data = json.loads(project.read_text(encoding="utf-8"))
            self.assertEqual(data["app"], "MiniCut Studio")
            self.assertFalse(model.dirty)


class SubtitleProjectPersistenceTests(unittest.TestCase):
    def test_project_saves_and_resolves_relative_subtitle_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project_dir = root / "portable"
            project_dir.mkdir()
            source = project_dir / "movie.mp4"
            subtitle = project_dir / "subtitle-custom.srt"
            project = project_dir / "movie.minicut.json"
            source.write_bytes(b"video")
            subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHalo\n",
                encoding="utf-8",
            )

            model = ProjectModel()
            model.source = source.resolve()
            model.duration_ms = 2_000
            model.save(project, subtitle_path=subtitle)

            data, _source = load_project_file(project)
            resolved, auto_disabled, present = resolve_project_subtitle(
                data,
                project,
            )

            self.assertTrue(present)
            self.assertFalse(auto_disabled)
            self.assertEqual(resolved, subtitle.resolve())
            self.assertEqual(data["subtitle_relative"], "subtitle-custom.srt")

    def test_project_persists_explicit_video_only_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "movie.mp4"
            project = root / "movie.minicut.json"
            source.write_bytes(b"video")

            model = ProjectModel()
            model.source = source.resolve()
            model.duration_ms = 2_000
            model.save(project, subtitle_auto_disabled=True)

            data, _source = load_project_file(project)
            resolved, auto_disabled, present = resolve_project_subtitle(
                data,
                project,
            )

            self.assertTrue(present)
            self.assertTrue(auto_disabled)
            self.assertIsNone(resolved)

    def test_legacy_project_without_subtitle_fields_remains_auto_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project = root / "legacy.minicut.json"
            project.write_text(
                json.dumps({
                    "app": "MiniCut Studio",
                    "version": 2,
                    "cuts": [],
                }),
                encoding="utf-8",
            )

            data, _source = load_project_file(project)
            resolved, auto_disabled, present = resolve_project_subtitle(
                data,
                project,
            )

            self.assertFalse(present)
            self.assertFalse(auto_disabled)
            self.assertIsNone(resolved)


class ExactPtsProjectPersistenceTests(unittest.TestCase):
    def test_project_save_keeps_exact_pts_fraction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "movie.mp4"
            source.write_bytes(b"video")
            project = root / "movie.minicut.json"

            model = ProjectModel()
            model.source = source.resolve()
            model.duration_ms = 10_000
            model.cuts = [CutPoint(42, 42, "1001/24000")]
            model.save(project)

            data, _source = load_project_file(project)
            self.assertEqual(
                data["cuts"][0]["exact_time"],
                "1001/24000",
            )


class ProjectValidationTests(unittest.TestCase):
    def test_broken_json_is_rejected_with_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "broken.minicut.json"
            project.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rusak atau tidak dapat dibaca"):
                load_project_file(project)

    def test_non_list_cuts_are_rejected_before_video_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "broken.minicut.json"
            project.write_text(json.dumps({
                "app": "MiniCut Studio",
                "cuts": {"actual_ms": 1000},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Daftar cut"):
                load_project_file(project)

    def test_null_cuts_are_normalized_to_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "old.minicut.json"
            project.write_text(json.dumps({
                "app": "MiniCut Studio",
                "cuts": None,
            }), encoding="utf-8")
            data, _source = load_project_file(project)
            self.assertEqual(data["cuts"], [])


class ExportStagingSafetyTests(unittest.TestCase):
    def test_failed_fast_export_preserves_previous_successful_parts(self):
        class FailingProc:
            def __init__(self):
                self.stdout = io.StringIO("ffmpeg error\n")

            def wait(self, timeout=None):
                return 1

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "movie_Parts"
            out.mkdir()
            previous = out / "movie_Part-01.mp4"
            previous.write_bytes(b"previous-good-export")

            with patch(
                "minicut_agent.core.subprocess.Popen",
                return_value=FailingProc(),
            ):
                with self.assertRaisesRegex(RuntimeError, "FFmpeg berhenti"):
                    export_segments(
                        "ffmpeg",
                        root / "movie.mp4",
                        out,
                        "movie",
                        [],
                        2_000,
                    )

            self.assertEqual(
                previous.read_bytes(),
                b"previous-good-export",
            )

    def test_successful_export_replaces_previous_parts_after_validation(self):
        class SuccessProc:
            def __init__(self, cmd):
                self.stdout = io.StringIO("out_time_ms=2000000\n")
                Path(cmd[-1]).write_bytes(b"new-export")

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "movie_Parts"
            out.mkdir()
            previous = out / "movie_Part-01.mp4"
            stale = out / "movie_Part-02.mp4"
            previous.write_bytes(b"old-one")
            stale.write_bytes(b"old-two")

            def fake_popen(cmd, **kwargs):
                return SuccessProc(cmd)

            with patch(
                "minicut_agent.core.subprocess.Popen",
                side_effect=fake_popen,
            ):
                count, _size = export_segments(
                    "ffmpeg",
                    root / "movie.mp4",
                    out,
                    "movie",
                    [],
                    2_000,
                )

            self.assertEqual(count, 1)
            self.assertEqual(previous.read_bytes(), b"new-export")
            self.assertFalse(stale.exists())


    def test_reexport_removes_uppercase_extension_parts(self):
        class SuccessProc:
            def __init__(self, cmd):
                self.stdout = io.StringIO("out_time_ms=2000000\n")
                Path(cmd[-1]).write_bytes(b"new-video")

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "movie_Parts"
            out.mkdir()
            stale_upper = out / "movie_Part-01.MP4"
            stale_upper.write_bytes(b"old-uppercase")

            with patch(
                "minicut_agent.core.subprocess.Popen",
                side_effect=lambda cmd, **kwargs: SuccessProc(cmd),
            ):
                count, _size = export_segments(
                    "ffmpeg",
                    root / "movie.mp4",
                    out,
                    "movie",
                    [],
                    2_000,
                )

            self.assertEqual(count, 1)
            self.assertFalse(stale_upper.exists())
            self.assertTrue((out / "movie_Part-01.mp4").is_file())

    def test_reexport_removes_stale_parts_from_previous_video_extension(self):
        class SuccessProc:
            def __init__(self, cmd):
                self.stdout = io.StringIO("out_time_ms=2000000\n")
                Path(cmd[-1]).write_bytes(b"new-mp4")

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "movie_Parts"
            out.mkdir()
            stale_mkv = out / "movie_Part-01.mkv"
            stale_mkv.write_bytes(b"old-mkv")
            keep_user_file = out / "movie_Part-catatan.mkv"
            keep_user_file.write_bytes(b"keep-me")

            with patch(
                "minicut_agent.core.subprocess.Popen",
                side_effect=lambda cmd, **kwargs: SuccessProc(cmd),
            ):
                count, _size = export_segments(
                    "ffmpeg",
                    root / "movie.mp4",
                    out,
                    "movie",
                    [],
                    2_000,
                )

            self.assertEqual(count, 1)
            self.assertFalse(stale_mkv.exists())
            self.assertTrue((out / "movie_Part-01.mp4").is_file())
            self.assertTrue(keep_user_file.exists())

    def test_reexport_without_srt_removes_old_generated_srt(self):
        class SuccessProc:
            def __init__(self, cmd):
                self.stdout = io.StringIO("out_time_ms=2000000\n")
                Path(cmd[-1]).write_bytes(b"new-export")

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "movie_Parts"
            out.mkdir()
            (out / "movie_Part-01.mp4").write_bytes(b"old-video")
            stale_srt_1 = out / "movie_Part-01.srt"
            stale_srt_2 = out / "movie_Part-02.srt"
            stale_srt_1.write_text("old subtitle 1", encoding="utf-8")
            stale_srt_2.write_text("old subtitle 2", encoding="utf-8")

            with patch(
                "minicut_agent.core.subprocess.Popen",
                side_effect=lambda cmd, **kwargs: SuccessProc(cmd),
            ):
                count, _size = export_segments(
                    "ffmpeg",
                    root / "movie.mp4",
                    out,
                    "movie",
                    [],
                    2_000,
                )

            self.assertEqual(count, 1)
            self.assertFalse(stale_srt_1.exists())
            self.assertFalse(stale_srt_2.exists())

    def test_invalid_srt_preserves_previous_video_and_subtitle(self):
        class SuccessProc:
            def __init__(self, cmd):
                self.stdout = io.StringIO("out_time_ms=2000000\n")
                Path(cmd[-1]).write_bytes(b"new-video")

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "movie_Parts"
            out.mkdir()
            previous_video = out / "movie_Part-01.mp4"
            previous_srt = out / "movie_Part-01.srt"
            previous_video.write_bytes(b"previous-video")
            previous_srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nold\n",
                encoding="utf-8",
            )
            broken_srt = root / "broken.srt"
            broken_srt.write_text("not a valid subtitle", encoding="utf-8")

            with patch(
                "minicut_agent.core.subprocess.Popen",
                side_effect=lambda cmd, **kwargs: SuccessProc(cmd),
            ):
                with self.assertRaisesRegex(ValueError, "SRT tidak berisi"):
                    export_segments(
                        "ffmpeg",
                        root / "movie.mp4",
                        out,
                        "movie",
                        [],
                        2_000,
                        srt_path=broken_srt,
                    )

            self.assertEqual(previous_video.read_bytes(), b"previous-video")
            self.assertIn("old", previous_srt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
