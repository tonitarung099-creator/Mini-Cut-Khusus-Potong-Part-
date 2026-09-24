from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from .subtitles import write_srt_parts

SUPPORTED_VIDEO = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mts"}

def bundle_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]

def find_tool(name: str) -> str | None:
    exe = name + ".exe" if os.name == "nt" else name
    for base in (bundle_dir(), bundle_dir() / "_internal"):
        p = base / exe
        if p.is_file():
            return str(p)
    return shutil.which(exe) or shutil.which(name)

def creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

def clock_text(ms: int) -> str:
    ms = max(0, int(ms))
    sec, milli = divmod(ms, 1000)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"

def short_clock(ms: int) -> str:
    return clock_text(ms).split(".")[0]

def parse_time_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower().replace(",", ".")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*ms", text):
        return int(float(text[:-2].strip()))

    # Gemini/tools may send a natural-language time instead of an integer even
    # though time_ms is preferred. Accept arbitrary hour/minute/second values,
    # compact forms, and the same common aliases as the local chat parser.
    number = r"\d+(?:\.\d+)?"
    hour_unit = r"(?:jam|hours?|hrs?|hr|h|j)(?![A-Za-z])"
    minute_unit = r"(?:menit|minutes?|mins?|min|mnt|m)(?![A-Za-z])"
    second_unit = r"(?:detik|dtik|dtk|seconds?|secs?|sec|s|d)(?![A-Za-z])"
    joiner = r"(?:(?:lebih|lewat|plus|dan)|\+)?"
    natural = re.fullmatch(
        rf"\s*(?:(?P<hours>{number})\s*{hour_unit})?"
        rf"\s*{joiner}\s*"
        rf"(?:(?P<minutes>{number})\s*{minute_unit})?"
        rf"\s*{joiner}\s*"
        rf"(?:(?P<seconds>{number})\s*{second_unit})?\s*",
        text,
        flags=re.I,
    )
    if natural and any(
        natural.group(name) is not None
        for name in ("hours", "minutes", "seconds")
    ):
        hours = float(natural.group("hours") or 0)
        minutes = float(natural.group("minutes") or 0)
        seconds = float(natural.group("seconds") or 0)
        return round((hours * 3600 + minutes * 60 + seconds) * 1000)

    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return int((h * 3600 + m * 60 + s) * 1000)
        if len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return int((m * 60 + s) * 1000)
        # Keep legacy behavior: a bare numeric string means seconds.
        return int(float(text) * 1000)
    except ValueError as exc:
        raise ValueError(f"Format waktu tidak dikenali: {value}") from exc

def run_text(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags(),
        check=False,
    )

def probe_media(source: Path, ffprobe: str) -> dict[str, Any]:
    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(source),
    ]
    result = run_text(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe gagal membaca video.")
    data = json.loads(result.stdout or "{}")
    duration = 0.0
    try:
        duration = float(data.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    fps = 0.0
    width = height = 0
    codec = ""
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        codec = str(stream.get("codec_name") or "")
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
        try:
            n, d = rate.split("/", 1)
            fps = float(n) / max(float(d), 1e-9)
        except Exception:
            fps = 0.0
        if duration <= 0:
            try:
                duration = float(stream.get("duration") or 0)
            except (TypeError, ValueError):
                pass
        break
    return {
        "duration_ms": max(0, int(duration * 1000)),
        "fps": fps,
        "width": width,
        "height": height,
        "codec": codec,
    }

def probe_keyframes(source: Path, ffprobe: str) -> list[int]:
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-skip_frame", "nokey", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(source),
    ]
    result = run_text(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Keyframe tidak dapat dibaca.")
    points: list[int] = []
    for line in result.stdout.splitlines():
        raw = line.strip().split(",", 1)[0]
        if not raw:
            continue
        try:
            points.append(int(float(raw) * 1000))
        except ValueError:
            pass
    points = sorted(set(points))
    if not points:
        raise RuntimeError("Tidak ada keyframe yang ditemukan.")
    return points

@dataclass
class CutPoint:
    requested_ms: int
    actual_ms: int
    exact_time: str | None = None

class ProjectModel:
    def __init__(self):
        self.source: Path | None = None
        self.project_path: Path | None = None
        self.duration_ms = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.codec = ""
        self.keyframes: list[int] = []
        self.cuts: list[CutPoint] = []
        self.playhead_ms = 0
        self.dirty = False

    def reset(self, source: Path, metadata: dict[str, Any]) -> None:
        self.source = source.resolve()
        self.project_path = None
        self.duration_ms = int(metadata.get("duration_ms") or 0)
        self.fps = float(metadata.get("fps") or 0)
        self.width = int(metadata.get("width") or 0)
        self.height = int(metadata.get("height") or 0)
        self.codec = str(metadata.get("codec") or "")
        self.keyframes = []
        self.cuts = []
        self.playhead_ms = 0
        self.dirty = False

    def clamp(self, ms: int) -> int:
        return max(0, min(int(ms), max(0, self.duration_ms)))

    def snap_keyframe(self, requested_ms: int) -> int:
        requested_ms = self.clamp(requested_ms)
        if not self.keyframes:
            return requested_ms
        pos = bisect.bisect_left(self.keyframes, requested_ms)
        choices: list[int] = []
        if pos < len(self.keyframes):
            choices.append(self.keyframes[pos])
        if pos > 0:
            choices.append(self.keyframes[pos - 1])
        return min(choices, key=lambda x: abs(x - requested_ms)) if choices else requested_ms

    def _normalize(self) -> None:
        unique: dict[int, CutPoint] = {}
        for cut in self.cuts:
            if 0 < cut.actual_ms < self.duration_ms:
                unique[cut.actual_ms] = cut
        self.cuts = sorted(unique.values(), key=lambda x: x.actual_ms)

    def add_cut(self, requested_ms: int) -> CutPoint:
        requested = self.clamp(requested_ms)
        if requested <= 0 or requested >= self.duration_ms:
            raise ValueError("Cut harus berada di antara awal dan akhir video.")
        actual = self.snap_keyframe(requested)
        if actual <= 0 or actual >= self.duration_ms:
            raise ValueError("Tidak ada keyframe aman di posisi tersebut.")
        if any(abs(c.actual_ms - actual) < 2 for c in self.cuts):
            raise ValueError("Cut pada keyframe tersebut sudah ada.")
        cut = CutPoint(requested, actual)
        self.cuts.append(cut)
        self._normalize()
        self.dirty = True
        return cut

    def add_frame_cut(
        self,
        requested_ms: int,
        actual_ms: int,
        exact_time: str | None = None,
    ) -> CutPoint:
        """Add a cut at an already verified real frame PTS.

        This is used for SmartCut/manual frame-accurate cuts and intentionally
        does not snap to a keyframe.
        """
        requested = self.clamp(requested_ms)
        actual = self.clamp(actual_ms)
        if requested <= 0 or requested >= self.duration_ms:
            raise ValueError("Cut manual harus berada di antara awal dan akhir video.")
        if actual <= 0 or actual >= self.duration_ms:
            raise ValueError("Frame cut berada di luar durasi video.")
        if any(abs(c.actual_ms - actual) < 2 for c in self.cuts):
            raise ValueError("Cut pada frame tersebut sudah ada.")
        cut = CutPoint(
            requested,
            actual,
            str(exact_time).strip() if exact_time else None,
        )
        self.cuts.append(cut)
        self._normalize()
        self.dirty = True
        return cut

    def remove_cut(self, index: int) -> CutPoint:
        if index < 0 or index >= len(self.cuts):
            raise IndexError("Index cut tidak valid.")
        self.dirty = True
        return self.cuts.pop(index)

    def clear_cuts(self) -> None:
        self.cuts.clear()
        self.dirty = True

    def divide_equal(self, parts: int) -> None:
        if parts < 2:
            raise ValueError("Jumlah part minimal 2.")
        if self.duration_ms <= 0:
            raise ValueError("Belum ada video.")
        self.cuts = []
        for i in range(1, parts):
            requested = round(self.duration_ms * i / parts)
            actual = self.snap_keyframe(requested)
            if 0 < actual < self.duration_ms:
                self.cuts.append(CutPoint(requested, actual))
        self._normalize()
        self.dirty = True

    def divide_interval(self, interval_ms: int) -> None:
        if interval_ms <= 0:
            raise ValueError("Interval harus lebih dari 0.")
        self.cuts = []
        t = interval_ms
        while t < self.duration_ms:
            actual = self.snap_keyframe(t)
            if 0 < actual < self.duration_ms:
                self.cuts.append(CutPoint(t, actual))
            t += interval_ms
        self._normalize()
        self.dirty = True

    def part_ranges(self) -> list[tuple[int, int]]:
        marks = [0] + [c.actual_ms for c in self.cuts] + [self.duration_ms]
        return [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]

    def has_non_keyframe_cuts(self) -> bool:
        """True when at least one current cut is not a probed keyframe."""
        if not self.cuts:
            return False
        keyframes = {int(value) for value in self.keyframes}
        return any(
            int(cut.actual_ms) not in keyframes
            for cut in self.cuts
        )

    def state(self) -> dict[str, Any]:
        return {
            "source": str(self.source) if self.source else None,
            "duration_ms": self.duration_ms,
            "duration": clock_text(self.duration_ms),
            "playhead_ms": self.playhead_ms,
            "playhead": clock_text(self.playhead_ms),
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}" if self.width and self.height else "",
            "codec": self.codec,
            "keyframes_ready": bool(self.keyframes),
            "cuts": [asdict(c) for c in self.cuts],
            "parts": len(self.cuts) + 1 if self.source else 0,
            "project_path": str(self.project_path) if self.project_path else None,
            "dirty": self.dirty,
        }

    def to_project_dict(
        self,
        project_path: Path | None = None,
        subtitle_path: Path | None = None,
        subtitle_auto_disabled: bool = False,
    ) -> dict[str, Any]:
        source_abs = str(self.source) if self.source else ""
        source_rel = ""
        if self.source and project_path:
            try:
                source_rel = os.path.relpath(self.source, project_path.parent)
            except ValueError:
                source_rel = ""

        subtitle_abs = ""
        subtitle_rel = ""
        if subtitle_path is not None:
            resolved_subtitle = Path(subtitle_path).resolve()
            subtitle_abs = str(resolved_subtitle)
            if project_path:
                try:
                    subtitle_rel = os.path.relpath(
                        resolved_subtitle,
                        project_path.parent,
                    )
                except ValueError:
                    subtitle_rel = ""

        return {
            "app": "MiniCut Studio",
            "version": 2,
            "source_absolute": source_abs,
            "source_relative": source_rel,
            "subtitle_absolute": subtitle_abs,
            "subtitle_relative": subtitle_rel,
            "subtitle_auto_disabled": bool(subtitle_auto_disabled),
            "metadata": {
                "duration_ms": self.duration_ms,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "codec": self.codec,
            },
            "cuts": [asdict(c) for c in self.cuts],
            "parts": len(self.cuts) + 1,
        }

    def save(
        self,
        path: Path,
        subtitle_path: Path | None = None,
        subtitle_auto_disabled: bool = False,
    ) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = json.dumps(
            self.to_project_dict(
                path,
                subtitle_path=subtitle_path,
                subtitle_auto_disabled=subtitle_auto_disabled,
            ),
            ensure_ascii=False,
            indent=2,
        )
        tmp.write_text(payload, encoding="utf-8")
        try:
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        self.project_path = path
        self.dirty = False

def resolve_project_subtitle(
    data: dict[str, Any],
    project_path: Path,
) -> tuple[Path | None, bool, bool]:
    """Resolve subtitle stored in a project.

    Returns (subtitle_path, auto_disabled, setting_present). Relative path wins
    so project folders stay portable after being moved.
    """
    setting_present = bool(
        data.get("subtitle_absolute")
        or data.get("subtitle_relative")
        or data.get("subtitle_auto_disabled", False)
    )
    auto_disabled = bool(data.get("subtitle_auto_disabled", False))
    if auto_disabled:
        return None, True, setting_present

    candidates: list[Path] = []
    if data.get("subtitle_relative"):
        candidates.append(
            (project_path.parent / str(data["subtitle_relative"])).resolve()
        )
    if data.get("subtitle_absolute"):
        candidates.append(Path(str(data["subtitle_absolute"])))

    subtitle = next((path for path in candidates if path.is_file()), None)
    return subtitle, False, setting_present


def load_project_file(path: Path) -> tuple[dict[str, Any], Path | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("File proyek MiniCut rusak atau tidak dapat dibaca.") from exc
    if not isinstance(data, dict) or data.get("app") != "MiniCut Studio":
        raise ValueError("File JSON bukan proyek MiniCut Studio.")

    cuts = data.get("cuts", [])
    if cuts is None:
        cuts = []
        data["cuts"] = cuts
    if not isinstance(cuts, list):
        raise ValueError("Daftar cut pada proyek MiniCut rusak.")
    for index, item in enumerate(cuts, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Data cut ke-{index} pada proyek MiniCut rusak.")
        try:
            requested = int(item.get("requested_ms", item.get("actual_ms", 0)))
            actual = int(item.get("actual_ms", requested))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Timestamp cut ke-{index} pada proyek MiniCut tidak valid."
            ) from exc
        item["requested_ms"] = requested
        item["actual_ms"] = actual
        exact_time = item.get("exact_time")
        item["exact_time"] = (
            str(exact_time).strip()
            if exact_time not in (None, "")
            else None
        )

    candidates = []
    # Relative source keeps a project folder portable after it is moved/copied.
    if data.get("source_relative"):
        candidates.append((path.parent / data["source_relative"]).resolve())
    if data.get("source_absolute"):
        candidates.append(Path(data["source_absolute"]))
    source = next((p for p in candidates if p.is_file()), None)
    return data, source

def _export_part_files(output_dir: Path, base_name: str, ext: str) -> list[Path]:
    """List only numeric part files generated by MiniCut for this source."""
    if not output_dir.is_dir():
        return []
    pattern = re.compile(
        rf"^{re.escape(base_name)}_Part-(?P<number>\d+){re.escape(ext)}$",
        flags=re.IGNORECASE,
    )
    matched: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            matched.append((int(match.group("number")), path))
    return [path for _number, path in sorted(matched, key=lambda item: item[0])]


def _clear_export_parts(output_dir: Path, base_name: str, ext: str) -> None:
    for path in _export_part_files(output_dir, base_name, ext):
        path.unlink(missing_ok=True)


def _generated_video_part_files(output_dir: Path, base_name: str) -> list[Path]:
    """List numeric MiniCut part files across supported video extensions.

    Extension matching is case-insensitive because Windows users may open
    sources named .MP4/.MKV and MiniCut preserves the source suffix on export.
    """
    if not output_dir.is_dir():
        return []
    pattern = re.compile(
        rf"^{re.escape(base_name)}_Part-(?P<number>\d+)(?P<ext>\.[^.]+)$"
    )
    matched: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        if match.group("ext").lower() not in SUPPORTED_VIDEO:
            continue
        matched.append((int(match.group("number")), path))
    return [
        path
        for _number, path in sorted(
            matched,
            key=lambda item: (item[0], item[1].suffix.lower(), item[1].name),
        )
    ]


def _part_ranges_from_cuts(
    cut_times_ms: list[int],
    duration_ms: int,
) -> list[tuple[int, int]]:
    duration_ms = max(0, int(duration_ms))
    marks = [0]
    marks.extend(
        sorted({
            int(value)
            for value in cut_times_ms
            if 0 < int(value) < duration_ms
        })
    )
    marks.append(duration_ms)
    return [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]


def _commit_staged_export_parts(
    staging_dir: Path,
    output_dir: Path,
    base_name: str,
    ext: str,
    include_subtitles: bool = False,
) -> list[Path]:
    """Replace video + optional SRT companions as one transaction."""
    staged_video = _export_part_files(staging_dir, base_name, ext)
    if not staged_video:
        raise RuntimeError("Tidak ada hasil ekspor baru untuk dipindahkan.")

    staged_srt = (
        _export_part_files(staging_dir, base_name, ".srt")
        if include_subtitles
        else []
    )
    if include_subtitles:
        if len(staged_srt) != len(staged_video):
            raise RuntimeError(
                "Hasil subtitle part tidak lengkap dan ekspor tidak dipasang."
            )
        video_numbers = [
            int(re.search(r"_Part-(\d+)", path.stem).group(1))
            for path in staged_video
        ]
        subtitle_numbers = [
            int(re.search(r"_Part-(\d+)", path.stem).group(1))
            for path in staged_srt
        ]
        if subtitle_numbers != video_numbers:
            raise RuntimeError(
                "Nomor part subtitle tidak cocok dengan part video."
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Selalu perlakukan SRT hasil MiniCut lama sebagai bagian dari transaksi.
    # Jika ekspor terbaru tidak memakai SRT, companion lama harus dihapus agar
    # tidak terlihat seolah masih sinkron dengan video baru.
    previous = _generated_video_part_files(output_dir, base_name)
    previous += _export_part_files(output_dir, base_name, ".srt")

    staged_all = list(staged_video) + list(staged_srt)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.backup-",
        dir=str(output_dir.parent),
    ) as backup_raw:
        backup_dir = Path(backup_raw)
        backed_up: list[tuple[Path, Path]] = []
        installed: list[Path] = []
        try:
            for old in previous:
                backup = backup_dir / old.name
                old.replace(backup)
                backed_up.append((backup, old))

            for new_part in staged_all:
                destination = output_dir / new_part.name
                new_part.replace(destination)
                installed.append(destination)
        except Exception:
            for destination in installed:
                destination.unlink(missing_ok=True)
            for backup, original in backed_up:
                if backup.exists():
                    backup.replace(original)
            raise

    return _export_part_files(output_dir, base_name, ext)


def export_segments(
    ffmpeg: str,
    source: Path,
    output_dir: Path,
    base_name: str,
    cut_times_ms: list[int],
    duration_ms: int,
    progress: Callable[[int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    srt_path: Path | None = None,
) -> tuple[int, int]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    ext = source.suffix or ".mp4"
    valid_cuts = sorted({
        int(value)
        for value in cut_times_ms
        if 0 < int(value) < int(duration_ms)
    })
    stage_ctx = tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.stage-",
        dir=str(output_dir.parent),
    )
    staging_dir = Path(stage_ctx.name)
    try:
        pattern = staging_dir / f"{base_name}_Part-%02d{ext}"
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-nostats", "-progress", "pipe:1",
            "-i", str(source), "-map", "0", "-c", "copy", "-map_metadata", "0",
        ]
        if valid_cuts:
            cmd += [
                "-segment_times", ",".join(
                    f"{v / 1000:.3f}" for v in valid_cuts
                ),
                "-reset_timestamps", "1",
                "-segment_start_number", "1",
                "-f", "segment",
                str(pattern),
            ]
        else:
            # A timeline without cuts is exactly one part.
            cmd += [str(staging_dir / f"{base_name}_Part-01{ext}")]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags(),
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            if cancelled and cancelled():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                raise InterruptedError("Ekspor dibatalkan.")
            line = raw.strip()
            if line.startswith("out_time_ms="):
                try:
                    out_ms = int(line.split("=", 1)[1]) // 1000
                    pct = int(
                        min(100, out_ms / max(1, duration_ms) * 100)
                    )
                    if progress:
                        progress(pct, clock_text(out_ms))
                except ValueError:
                    pass
            elif (
                line
                and log
                and not line.startswith(
                    ("frame=", "fps=", "stream_", "progress=")
                )
            ):
                log(line)

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"FFmpeg berhenti dengan kode {rc}.")

        files = [
            p for p in _export_part_files(staging_dir, base_name, ext)
            if p.stat().st_size > 0
        ]
        expected = len(valid_cuts) + 1
        if len(files) != expected:
            raise RuntimeError(
                f"FFmpeg selesai tetapi hasil part tidak lengkap "
                f"({len(files)}/{expected} file valid)."
            )

        if srt_path is not None:
            ranges = _part_ranges_from_cuts(valid_cuts, duration_ms)
            subtitle_files = write_srt_parts(
                srt_path,
                staging_dir,
                base_name,
                ranges,
            )
            if len(subtitle_files) != expected:
                raise RuntimeError(
                    f"Subtitle selesai tetapi hasil part tidak lengkap "
                    f"({len(subtitle_files)}/{expected} file)."
                )
            if log:
                log(f"SRT: {len(subtitle_files)} part siap dan timestamp di-reset ke 00:00.")

        committed = _commit_staged_export_parts(
            staging_dir,
            output_dir,
            base_name,
            ext,
            include_subtitles=srt_path is not None,
        )
        return len(committed), sum(p.stat().st_size for p in committed)
    finally:
        stage_ctx.cleanup()

def export_segments_smartcut(
    smartcut_exe: str,
    source: Path,
    output_dir: Path,
    base_name: str,
    cut_times_ms: list[int],
    duration_ms: int,
    cut_exact_times: list[str | None] | None = None,
    progress: Callable[[int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    srt_path: Path | None = None,
) -> tuple[int, int]:
    """Frame-accurate export using the SmartCut companion.

    New parts are written to a staging directory first. Existing successful
    exports remain untouched until every new part has completed successfully.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    ext = source.suffix or ".mp4"

    exact_values = list(cut_exact_times or [])
    cut_specs: dict[int, str | None] = {}
    for index, value in enumerate(cut_times_ms):
        ms = int(value)
        if not (0 < ms < int(duration_ms)):
            continue
        exact = (
            str(exact_values[index]).strip()
            if index < len(exact_values)
            and exact_values[index] not in (None, "")
            else None
        )
        if exact is not None:
            try:
                exact_fraction = Fraction(exact)
            except Exception as exc:
                raise RuntimeError(
                    f"PTS exact cut tidak valid pada {clock_text(ms)}: {exact}"
                ) from exc
            delta_ms = abs(float(exact_fraction) * 1000 - ms)
            if delta_ms > 2.0:
                raise RuntimeError(
                    "PTS exact cut tidak cocok dengan timestamp timeline "
                    f"({clock_text(ms)} vs {exact})."
                )
            # Normalize rational text but never round it to milliseconds.
            exact = str(exact_fraction)
        cut_specs[ms] = exact

    marks: list[tuple[int, str | None]] = (
        [(0, "start")]
        + sorted(cut_specs.items(), key=lambda item: item[0])
        + [(int(duration_ms), "end")]
    )
    ranges = [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]

    stage_ctx = tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.stage-",
        dir=str(output_dir.parent),
    )
    staging_dir = Path(stage_ctx.name)
    created: list[Path] = []
    try:
        for idx, ((start_ms, start_exact), (end_ms, end_exact)) in enumerate(ranges, 1):
            if cancelled and cancelled():
                raise InterruptedError("Ekspor dibatalkan.")

            out = staging_dir / f"{base_name}_Part-{idx:02d}{ext}"
            start_arg = (
                "start"
                if start_ms <= 0
                else (start_exact or f"{start_ms / 1000:.6f}")
            )
            end_arg = (
                "end"
                if end_ms >= duration_ms
                else (end_exact or f"{end_ms / 1000:.6f}")
            )
            keep = f"{start_arg},{end_arg}"
            if progress:
                progress(
                    int((idx - 1) / max(1, len(ranges)) * 100),
                    f"SmartCut Part-{idx:02d} · "
                    f"{clock_text(start_ms)} → {clock_text(end_ms)}",
                )
            if log:
                log(f"SmartCut Part-{idx:02d}: {keep}")

            with tempfile.TemporaryFile(
                mode="w+t",
                encoding="utf-8",
            ) as smartcut_log:
                proc = subprocess.Popen(
                    [
                        smartcut_exe,
                        str(source),
                        str(out),
                        "--keep",
                        keep,
                        "--log-level",
                        "warning",
                    ],
                    stdout=smartcut_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creation_flags(),
                )
                while proc.poll() is None:
                    if cancelled and cancelled():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=5)
                        raise InterruptedError("Ekspor dibatalkan.")
                    time.sleep(0.12)
                smartcut_log.seek(0)
                lines = [
                    x.strip()
                    for x in smartcut_log.readlines()
                    if x.strip()
                ]
                if log:
                    for clean in lines[-20:]:
                        log(clean)

            if proc.returncode != 0:
                tail = "\n".join(lines[-20:])
                raise RuntimeError(
                    f"SmartCut gagal pada Part-{idx:02d} "
                    f"(kode {proc.returncode})."
                    + (f"\n{tail}" if tail else "")
                )
            if not out.is_file() or out.stat().st_size <= 0:
                raise RuntimeError(
                    f"SmartCut tidak menghasilkan Part-{idx:02d}."
                )

            created.append(out)
            if progress:
                progress(
                    int(idx / max(1, len(ranges)) * 100),
                    f"SmartCut Part-{idx:02d} selesai",
                )

        if srt_path is not None:
            def subtitle_boundary_ms(mark: tuple[int, str | None]) -> int:
                ms, exact = mark
                if exact in (None, "start", "end"):
                    return int(ms)
                try:
                    # SRT hanya punya resolusi milidetik. Gunakan milidetik
                    # terdekat dari PTS exact agar boundary subtitle mengikuti
                    # SmartCut sedekat mungkin, tanpa float rounding.
                    return int(round(Fraction(str(exact)) * 1000))
                except Exception:
                    return int(ms)

            subtitle_ranges = [
                (
                    subtitle_boundary_ms(start),
                    subtitle_boundary_ms(end),
                )
                for start, end in ranges
            ]
            subtitle_files = write_srt_parts(
                srt_path,
                staging_dir,
                base_name,
                subtitle_ranges,
            )
            if len(subtitle_files) != len(ranges):
                raise RuntimeError(
                    f"Subtitle selesai tetapi hasil part tidak lengkap "
                    f"({len(subtitle_files)}/{len(ranges)} file)."
                )
            if log:
                log(f"SRT: {len(subtitle_files)} part siap dan timestamp di-reset ke 00:00.")

        committed = _commit_staged_export_parts(
            staging_dir,
            output_dir,
            base_name,
            ext,
            include_subtitles=srt_path is not None,
        )
        return len(committed), sum(p.stat().st_size for p in committed)
    finally:
        stage_ctx.cleanup()

