from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from minicut_agent.camera_boundary_strict import resolve_camera_boundary
from minicut_agent.core import export_segments_smartcut, fraction_seconds_to_ms


ROOT = Path(__file__).resolve().parents[1]


def _tool(env_name: str, basename: str) -> str | None:
    configured = os.environ.get(env_name)
    if configured and Path(configured).is_file():
        return configured
    candidates = [
        ROOT / "ffmpeg-runtime" / f"{basename}.exe",
        ROOT / "ffmpeg-runtime" / basename,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return shutil.which(basename)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-8000:])


def _make_video(
    ffmpeg: str,
    path: Path,
    *,
    rate: str,
    frames: int,
    geq: str,
) -> None:
    # The bundled LGPL FFmpeg deliberately excludes libx264. Use the native
    # MPEG-4 encoder so this is a real test of the exact runtime shipped in ZIP.
    _run([
        ffmpeg,
        "-v", "error",
        "-y",
        "-f", "lavfi",
        "-i", f"nullsrc=s=64x64:r={rate},{geq}",
        "-frames:v", str(frames),
        "-c:v", "mpeg4",
        "-q:v", "1",
        "-g", "240",
        "-bf", "0",
        "-pix_fmt", "yuv420p",
        str(path),
    ])


def _gray_frames(ffmpeg: str, path: Path) -> tuple[int, list[bytes]]:
    proc = subprocess.run(
        [
            ffmpeg,
            "-v", "error",
            "-i", str(path),
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
    frame_size = 64 * 64
    if len(proc.stdout) % frame_size:
        raise AssertionError("Ukuran raw frame hasil ekspor tidak utuh.")
    frames = [
        proc.stdout[offset:offset + frame_size]
        for offset in range(0, len(proc.stdout), frame_size)
    ]
    return len(frames), frames


class RealFFmpegCameraBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = _tool("MINICUT_FFMPEG", "ffmpeg")
        cls.ffprobe = _tool("MINICUT_FFPROBE", "ffprobe")
        if not cls.ffmpeg or not cls.ffprobe:
            raise unittest.SkipTest("FFmpeg/ffprobe nyata tidak tersedia.")

    def test_rounding_regression_keeps_first_new_shot_exact_pts(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "rounding-23976.mp4"
            _make_video(
                self.ffmpeg,
                source,
                rate="24000/1001",
                frames=432,
                geq="geq=lum='if(lt(N,307),16,235)':cb=128:cr=128",
            )
            result = resolve_camera_boundary(
                source,
                self.ffmpeg,
                self.ffprobe,
                13_000,
                duration_ms=18_000,
                require_camera_change=True,
            )

            expected = Fraction(307307, 24000)
            self.assertTrue(result["camera_boundary_verified"], result)
            self.assertTrue(result["pts_verified"], result)
            self.assertFalse(result["fallback_to_nearest_frame"], result)
            self.assertEqual(Fraction(str(result["exact_time"])), expected)
            self.assertEqual(int(result["time_ms"]), 12_804)

    def test_subtle_luma_change_is_found_without_nearest_frame_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "subtle-luma.mp4"
            _make_video(
                self.ffmpeg,
                source,
                rate="25",
                frames=400,
                geq="geq=lum='if(lt(N,300),16,36)':cb=128:cr=128",
            )
            result = resolve_camera_boundary(
                source,
                self.ffmpeg,
                self.ffprobe,
                12_123,
                duration_ms=16_000,
                require_camera_change=True,
            )
            self.assertTrue(result["camera_boundary_verified"], result)
            self.assertEqual(Fraction(str(result["exact_time"])), Fraction(12, 1))
            self.assertFalse(result["fallback_to_nearest_frame"])

    def test_chroma_only_change_is_found(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "chroma.mp4"
            _make_video(
                self.ffmpeg,
                source,
                rate="25",
                frames=400,
                geq="geq=lum=80:cb='if(lt(N,300),80,180)':cr=128",
            )
            result = resolve_camera_boundary(
                source,
                self.ffmpeg,
                self.ffprobe,
                12_123,
                duration_ms=16_000,
                require_camera_change=True,
            )
            self.assertTrue(result["camera_boundary_verified"], result)
            self.assertEqual(Fraction(str(result["exact_time"])), Fraction(12, 1))

    def test_single_frame_flash_is_review_not_verified_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "flash.mp4"
            _make_video(
                self.ffmpeg,
                source,
                rate="25",
                frames=400,
                geq=(
                    "geq=lum='if(eq(N,300),235,16)':cb=128:cr=128"
                ),
            )
            result = resolve_camera_boundary(
                source,
                self.ffmpeg,
                self.ffprobe,
                12_123,
                duration_ms=16_000,
                require_camera_change=True,
            )
            self.assertFalse(result["camera_boundary_verified"], result)
            self.assertTrue(result["needs_review"], result)
            self.assertFalse(result["fallback_to_nearest_frame"], result)

    def test_rounding_boundary_smartcut_export_has_correct_edge_frames(self):
        smartcut = os.environ.get("MINICUT_SMARTCUT_EXE")
        if not smartcut or not Path(smartcut).is_file():
            raise unittest.SkipTest(
                "Companion SmartCut belum tersedia; test ini dijalankan lagi setelah build companion."
            )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "rounding-23976.mp4"
            out_dir = root / "parts"
            _make_video(
                self.ffmpeg,
                source,
                rate="24000/1001",
                frames=432,
                geq="geq=lum='if(lt(N,307),16,235)':cb=128:cr=128",
            )
            result = resolve_camera_boundary(
                source,
                self.ffmpeg,
                self.ffprobe,
                13_000,
                duration_ms=18_000,
                require_camera_change=True,
            )
            self.assertTrue(result["camera_boundary_verified"], result)
            exact = Fraction(str(result["exact_time"]))
            expected = Fraction(307307, 24000)
            self.assertEqual(exact, expected)

            count, _size = export_segments_smartcut(
                smartcut,
                source,
                out_dir,
                "rounding",
                [fraction_seconds_to_ms(exact)],
                18_018,
                cut_exact_times=[str(exact)],
            )
            self.assertEqual(count, 2)
            part1 = out_dir / "rounding_Part-01.mp4"
            part2 = out_dir / "rounding_Part-02.mp4"
            self.assertTrue(part1.is_file())
            self.assertTrue(part2.is_file())

            count1, frames1 = _gray_frames(self.ffmpeg, part1)
            count2, frames2 = _gray_frames(self.ffmpeg, part2)
            self.assertEqual(count1, 307)
            self.assertEqual(count2, 125)
            self.assertEqual(count1 + count2, 432)

            last_old_mean = sum(frames1[-1]) / len(frames1[-1])
            first_new_mean = sum(frames2[0]) / len(frames2[0])
            self.assertLess(last_old_mean, 80.0)
            self.assertGreater(first_new_mean, 170.0)


if __name__ == "__main__":
    unittest.main()
