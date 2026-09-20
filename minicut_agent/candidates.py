from __future__ import annotations

import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core import creation_flags
from .subtitles import SubtitleTrack, format_ms

_PTS_RE = re.compile(r"pts_time:(?P<t>\d+(?:\.\d+)?)")
_SILENCE_RE = re.compile(r"silence_(?:start|end):\s*(?P<t>\d+(?:\.\d+)?)")

@dataclass
class LocalCandidate:
    target_ms: int
    time_ms: int
    distance_ms: int
    visual: bool = False
    silence: bool = False
    subtitle_gap: bool = False
    dialogue_edge: bool = False
    subtitle_safe: bool = True
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = format_ms(self.target_ms)
        data["time"] = format_ms(self.time_ms)
        return data

def target_times(duration_ms: int, interval_ms: int = 15 * 60_000, min_tail_ms: int = 5 * 60_000) -> list[int]:
    if interval_ms <= 0:
        raise ValueError("Interval harus lebih dari 0.")
    targets = []
    t = interval_ms
    while t < duration_ms:
        if duration_ms - t < min_tail_ms:
            break
        targets.append(t)
        t += interval_ms
    return targets

def _run_ffmpeg_lines(cmd: list[str]) -> list[str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags(),
        check=False,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-20:])
        raise RuntimeError(tail or f"FFmpeg gagal dengan kode {proc.returncode}.")
    return proc.stdout.splitlines()

def detect_visual_boundaries(
    ffmpeg: str,
    source: Path,
    start_ms: int,
    end_ms: int,
    scene_threshold: float = 0.27,
) -> list[int]:
    start_s = max(0, start_ms) / 1000
    duration_s = max(0.2, end_ms - start_ms) / 1000
    filt = f"select='gt(scene,{scene_threshold})',showinfo"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-ss", f"{start_s:.3f}", "-i", str(source),
        "-t", f"{duration_s:.3f}",
        "-vf", filt, "-an", "-f", "null", "-"
    ]
    lines = _run_ffmpeg_lines(cmd)
    result = []
    for line in lines:
        m = _PTS_RE.search(line)
        if m:
            result.append(start_ms + int(float(m.group("t")) * 1000))
    return _dedupe(result, 350)

def detect_silence_boundaries(
    ffmpeg: str,
    source: Path,
    start_ms: int,
    end_ms: int,
    noise_db: int = -34,
    min_silence_s: float = 0.30,
) -> list[int]:
    start_s = max(0, start_ms) / 1000
    duration_s = max(0.2, end_ms - start_ms) / 1000
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-ss", f"{start_s:.3f}", "-i", str(source),
        "-t", f"{duration_s:.3f}", "-vn",
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f", "null", "-"
    ]
    lines = _run_ffmpeg_lines(cmd)
    result = []
    for line in lines:
        m = _SILENCE_RE.search(line)
        if m:
            result.append(start_ms + int(float(m.group("t")) * 1000))
    return _dedupe(result, 500)

def _dedupe(points: list[int], tolerance_ms: int) -> list[int]:
    out: list[int] = []
    for p in sorted(set(int(v) for v in points)):
        if not out or abs(p - out[-1]) > tolerance_ms:
            out.append(p)
        else:
            out[-1] = (out[-1] + p) // 2
    return out

def _nearest_distance(point: int, values: list[int]) -> int | None:
    if not values:
        return None
    return min(abs(point - v) for v in values)

def rank_candidates(
    target_ms: int,
    window_ms: int,
    visual_points: list[int],
    silence_points: list[int],
    subtitles: SubtitleTrack | None,
    top_n: int = 3,
) -> list[LocalCandidate]:
    pool = list(visual_points)
    pool.extend(silence_points)
    if subtitles:
        # SRT dipakai sebagai penjaga dialog, bukan sumber kandidat utama.
        # Ini sengaja mengikuti perilaku versi lama yang lebih kuat memilih
        # perpindahan scene/lokasi/waktu secara visual.
        pool.extend(subtitles.gap_boundaries(target_ms - window_ms, target_ms + window_ms))
    if not pool:
        pool = [target_ms]

    pool = _dedupe([p for p in pool if abs(p - target_ms) <= window_ms], 650)
    result: list[LocalCandidate] = []
    for point in pool:
        dist = abs(point - target_ms)
        closeness = max(0.0, 1.0 - dist / max(1, window_ms))
        visual_dist = _nearest_distance(point, visual_points)
        silence_dist = _nearest_distance(point, silence_points)
        visual = visual_dist is not None and visual_dist <= 650
        silence = silence_dist is not None and silence_dist <= 900
        subtitle_safe = subtitles.is_safe_cut(point, 300) if subtitles else True
        subtitle_gap = False
        if subtitles:
            gap_dist = _nearest_distance(
                point,
                subtitles.gap_boundaries(target_ms - window_ms, target_ms + window_ms),
            )
            subtitle_gap = gap_dist is not None and gap_dist <= 700

        # Restore the pre-SmartCut selection behavior:
        # major visual scene changes dominate; SRT mainly vetoes unsafe dialogue cuts.
        score = closeness * 0.32
        score += 0.43 if visual else 0.0
        score += 0.12 if silence else 0.0
        score += 0.18 if subtitle_gap else 0.0
        score += 0.08 if subtitle_safe else -0.55
        if not visual:
            score -= 0.08
        score = round(max(-1.0, min(1.0, score)), 4)

        result.append(LocalCandidate(
            target_ms=target_ms,
            time_ms=point,
            distance_ms=dist,
            visual=visual,
            silence=silence,
            subtitle_gap=subtitle_gap,
            dialogue_edge=False,
            subtitle_safe=subtitle_safe,
            score=score,
        ))

    result.sort(key=lambda c: (c.score, -c.distance_ms), reverse=True)
    safe = [c for c in result if c.subtitle_safe]
    return (safe or result)[:max(1, top_n)]

def find_candidates_for_target(
    ffmpeg: str,
    source: Path,
    target_ms: int,
    window_ms: int,
    subtitles: SubtitleTrack | None,
    top_n: int = 3,
) -> list[LocalCandidate]:
    start_ms = max(0, target_ms - window_ms)
    end_ms = target_ms + window_ms
    visual = detect_visual_boundaries(ffmpeg, source, start_ms, end_ms)
    silence = detect_silence_boundaries(ffmpeg, source, start_ms, end_ms)
    return rank_candidates(target_ms, window_ms, visual, silence, subtitles, top_n=top_n)
