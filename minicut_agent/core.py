from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

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
    if text.endswith("ms"):
        return int(float(text[:-2].strip()))
    if text.endswith("s") and ":" not in text:
        return int(float(text[:-1]) * 1000)
    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return int((h * 3600 + m * 60 + s) * 1000)
        if len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return int((m * 60 + s) * 1000)
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

    def to_project_dict(self, project_path: Path | None = None) -> dict[str, Any]:
        source_abs = str(self.source) if self.source else ""
        source_rel = ""
        if self.source and project_path:
            try:
                source_rel = os.path.relpath(self.source, project_path.parent)
            except ValueError:
                source_rel = ""
        return {
            "app": "MiniCut Studio",
            "version": 2,
            "source_absolute": source_abs,
            "source_relative": source_rel,
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

    def save(self, path: Path) -> None:
        path = path.resolve()
        path.write_text(json.dumps(self.to_project_dict(path), ensure_ascii=False, indent=2), encoding="utf-8")
        self.project_path = path
        self.dirty = False

def load_project_file(path: Path) -> tuple[dict[str, Any], Path | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("app") != "MiniCut Studio":
        raise ValueError("File JSON bukan proyek MiniCut Studio.")
    candidates = []
    if data.get("source_absolute"):
        candidates.append(Path(data["source_absolute"]))
    if data.get("source_relative"):
        candidates.append((path.parent / data["source_relative"]).resolve())
    source = next((p for p in candidates if p.is_file()), None)
    return data, source

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
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = source.suffix or ".mp4"
    pattern = output_dir / f"{base_name}_Part-%02d{ext}"
    cmd = [ffmpeg, "-y", "-hide_banner", "-nostats", "-progress", "pipe:1", "-i", str(source),
           "-map", "0", "-c", "copy", "-map_metadata", "0"]
    if cut_times_ms:
        cmd += ["-segment_times", ",".join(f"{v / 1000:.3f}" for v in cut_times_ms)]
    cmd += ["-reset_timestamps", "1", "-f", "segment", str(pattern)]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", creationflags=creation_flags()
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        if cancelled and cancelled():
            proc.terminate()
            proc.wait(timeout=10)
            raise InterruptedError("Ekspor dibatalkan.")
        line = raw.strip()
        if line.startswith("out_time_ms="):
            try:
                out_ms = int(line.split("=", 1)[1]) // 1000
                pct = int(min(100, out_ms / max(1, duration_ms) * 100))
                if progress:
                    progress(pct, clock_text(out_ms))
            except ValueError:
                pass
        elif line and log and not line.startswith(("frame=", "fps=", "stream_", "progress=")):
            log(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"FFmpeg berhenti dengan kode {rc}.")
    files = sorted(output_dir.glob(f"{base_name}_Part-*{ext}"))
    return len(files) or len(cut_times_ms) + 1, sum(p.stat().st_size for p in files if p.is_file())



def preview_proxy_path(source: Path) -> Path:
    """Stable local cache path for a preview-only proxy.

    The cache key changes when source path, size, or modification time changes.
    The proxy is never used for AI decisions or final export.
    """
    source = source.resolve()
    stat = source.stat()
    identity = f"preview-v1|{source}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", errors="replace")
    token = hashlib.sha1(identity).hexdigest()[:20]
    base_env = os.environ.get("LOCALAPPDATA")
    if base_env:
        root = Path(base_env) / "MiniCut Studio Agent" / "preview-proxy"
    else:
        root = Path.home() / ".minicut" / "preview-proxy"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"proxy-{token}.mp4"


def build_preview_proxy(
    ffmpeg: str,
    source: Path,
    output: Path,
    duration_ms: int,
    source_height: int = 0,
    fps: float = 25.0,
    progress: Callable[[int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Build a lightweight H.264/AAC preview proxy with dense keyframes."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 64 * 1024:
        return output

    tmp = output.with_name(output.stem + ".part.mp4")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    gop = max(12, min(120, int(round(fps if fps > 0 else 25.0))))
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-nostats", "-progress", "pipe:1",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
    ]
    if int(source_height or 0) > 720:
        cmd += ["-vf", "scale=-2:720"]
    else:
        cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-sc_threshold", "0",
        "-c:a", "aac",
        "-b:a", "96k",
        "-ac", "2",
        "-movflags", "+faststart",
        str(tmp),
    ]

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
    last_pct = -1
    lines: list[str] = []
    for raw in proc.stdout:
        if cancelled and cancelled():
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise InterruptedError("Pembuatan proxy dibatalkan.")

        line = raw.strip()
        if not line:
            continue
        if line.startswith("out_time_ms="):
            try:
                out_ms = int(line.split("=", 1)[1]) // 1000
                pct = int(min(99, out_ms / max(1, int(duration_ms)) * 100))
                if pct != last_pct and progress:
                    progress(pct, clock_text(out_ms))
                last_pct = pct
            except ValueError:
                pass
        elif not line.startswith(("frame=", "fps=", "stream_", "progress=", "bitrate=", "speed=", "total_size=", "dup_frames=", "drop_frames=")):
            lines.append(line)
            if log and len(lines) <= 8:
                log(line)

    rc = proc.wait()
    if rc != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        tail = "\n".join(lines[-12:])
        raise RuntimeError(
            "Gagal membuat proxy preview."
            + (f"\n{tail}" if tail else f" FFmpeg code {rc}.")
        )

    tmp.replace(output)
    if progress:
        progress(100, clock_text(duration_ms))
    return output


def export_segments_smartcut(
    smartcut_exe: str,
    source: Path,
    output_dir: Path,
    base_name: str,
    cut_times_ms: list[int],
    duration_ms: int,
    progress: Callable[[int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    """Frame-accurate export using the SmartCut companion.

    SmartCut minimally re-encodes around non-keyframe cut points and copies
    the rest of the media whenever its codec/container permits it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = source.suffix or ".mp4"
    marks = [0] + sorted(
        set(int(v) for v in cut_times_ms if 0 < int(v) < int(duration_ms))
    ) + [int(duration_ms)]
    ranges = [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
    created: list[Path] = []

    for idx, (start_ms, end_ms) in enumerate(ranges, 1):
        if cancelled and cancelled():
            raise InterruptedError("Ekspor dibatalkan.")

        out = output_dir / f"{base_name}_Part-{idx:02d}{ext}"
        start_arg = "start" if start_ms <= 0 else f"{start_ms / 1000:.6f}"
        end_arg = "end" if end_ms >= duration_ms else f"{end_ms / 1000:.6f}"
        keep = f"{start_arg},{end_arg}"
        if progress:
            progress(
                int((idx - 1) / max(1, len(ranges)) * 100),
                f"SmartCut Part-{idx:02d} · {clock_text(start_ms)} → {clock_text(end_ms)}",
            )
        if log:
            log(f"SmartCut Part-{idx:02d}: {keep}")

        if out.exists():
            out.unlink()
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as smartcut_log:
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
                    raise InterruptedError("Ekspor dibatalkan.")
                time.sleep(0.12)
            smartcut_log.seek(0)
            lines = [x.strip() for x in smartcut_log.readlines() if x.strip()]
            if log:
                for clean in lines[-20:]:
                    log(clean)

        if proc.returncode != 0:
            tail = "\n".join(lines[-20:])
            raise RuntimeError(
                f"SmartCut gagal pada Part-{idx:02d} (kode {proc.returncode})."
                + (f"\n{tail}" if tail else "")
            )
        if not out.is_file() or out.stat().st_size <= 0:
            raise RuntimeError(f"SmartCut tidak menghasilkan Part-{idx:02d}.")
        created.append(out)
        if progress:
            progress(
                int(idx / max(1, len(ranges)) * 100),
                f"SmartCut Part-{idx:02d} selesai",
            )

    return len(created), sum(p.stat().st_size for p in created)
