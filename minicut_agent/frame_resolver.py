from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import run_text
from .subtitles import SubtitleTrack, format_ms


def probe_frame_timestamps(
    source: Path,
    ffprobe: str,
    start_ms: int,
    end_ms: int,
) -> list[int]:
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms + 1, int(end_ms))
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-read_intervals", f"{start_ms / 1000:.6f}%{end_ms / 1000:.6f}",
        "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json",
        str(source),
    ]
    result = run_text(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Frame timestamp tidak dapat dibaca.")
    data = json.loads(result.stdout or "{}")
    frames: list[int] = []
    for frame in data.get("frames", []):
        raw = frame.get("best_effort_timestamp_time")
        if raw in (None, "N/A"):
            continue
        try:
            ms = int(round(float(raw) * 1000))
        except (TypeError, ValueError):
            continue
        if start_ms - 1000 <= ms <= end_ms + 1000:
            frames.append(ms)
    return sorted(set(frames))


def resolve_semantic_frame(
    source: Path,
    ffprobe: str,
    preferred_ms: int,
    zone_start_ms: int,
    zone_end_ms: int,
    subtitles: SubtitleTrack | None = None,
    prefer_before_ms: int | None = None,
) -> dict[str, Any]:
    """Resolve an AI semantic boundary to a real source-frame timestamp.

    The AI may describe a narrative boundary between frames. This function
    never invents milliseconds: it chooses an actual video frame PTS.
    """
    preferred_ms = max(0, int(preferred_ms))
    zone_start_ms = max(0, int(zone_start_ms))
    zone_end_ms = max(zone_start_ms, int(zone_end_ms))
    if zone_end_ms - zone_start_ms < 80:
        zone_start_ms = max(0, preferred_ms - 1200)
        zone_end_ms = preferred_ms + 1200

    probe_start = max(0, zone_start_ms - 1200)
    probe_end = zone_end_ms + 1200
    frames = probe_frame_timestamps(source, ffprobe, probe_start, probe_end)
    if not frames:
        return {
            "time_ms": preferred_ms,
            "time": format_ms(preferred_ms),
            "frame_verified": False,
            "reason": "Frame PTS tidak tersedia; memakai waktu semantic AI.",
        }

    allowed = [f for f in frames if zone_start_ms <= f <= zone_end_ms]
    if not allowed:
        allowed = frames

    if prefer_before_ms is not None:
        before = [f for f in allowed if f < int(prefer_before_ms)]
        if before:
            allowed = before

    if subtitles:
        subtitle_safe = [f for f in allowed if subtitles.is_safe_cut(f, padding_ms=80)]
        if subtitle_safe:
            allowed = subtitle_safe

    chosen = min(
        allowed,
        key=lambda f: (
            abs(f - preferred_ms),
            0 if f <= preferred_ms else 1,
        ),
    )
    return {
        "time_ms": chosen,
        "time": format_ms(chosen),
        "frame_verified": True,
        "frame_delta_ms": chosen - preferred_ms,
        "zone_start_ms": zone_start_ms,
        "zone_end_ms": zone_end_ms,
        "reason": "Dikunci ke PTS frame nyata yang paling dekat dengan batas naratif.",
    }
