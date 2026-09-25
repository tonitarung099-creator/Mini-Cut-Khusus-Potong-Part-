from __future__ import annotations

import json
from decimal import Decimal, localcontext
from fractions import Fraction
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
            ms = int(round(Fraction(str(raw)) * 1000))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if start_ms - 1000 <= ms <= end_ms + 1000:
            frames.append(ms)
    return sorted(set(frames))



def probe_frame_points(
    source: Path,
    ffprobe: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    """Return real master frames with UI milliseconds and exact relative PTS."""
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms + 1, int(end_ms))

    # Read the source clock first. Some TS/MTS and remuxed files begin at a
    # non-zero container timestamp while MiniCut's timeline still begins at 0.
    clock_cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_streams",
        "-show_format",
        "-show_entries", "stream=time_base:format=start_time",
        "-of", "json",
        str(source),
    ]
    clock_result = run_text(clock_cmd)
    if clock_result.returncode != 0:
        raise RuntimeError(
            clock_result.stderr.strip() or "Clock video tidak dapat dibaca."
        )
    clock_data = json.loads(clock_result.stdout or "{}")
    streams = clock_data.get("streams") or []
    if not streams:
        raise RuntimeError("Time base video tidak tersedia.")
    try:
        time_base = Fraction(str(streams[0]["time_base"]))
    except Exception as exc:
        raise RuntimeError("Time base video tidak valid.") from exc
    try:
        start_time = Fraction(
            str((clock_data.get("format") or {}).get("start_time") or "0")
        )
    except Exception:
        start_time = Fraction(0)

    absolute_start = start_time + Fraction(start_ms, 1000)
    absolute_end = start_time + Fraction(end_ms, 1000)
    with localcontext() as ctx:
        ctx.prec = 30
        abs_start_text = format(
            Decimal(absolute_start.numerator)
            / Decimal(absolute_start.denominator),
            "f",
        )
        abs_end_text = format(
            Decimal(absolute_end.numerator)
            / Decimal(absolute_end.denominator),
            "f",
        )

    frame_cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-read_intervals", f"{abs_start_text}%{abs_end_text}",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp,best_effort_timestamp_time",
        "-of", "json",
        str(source),
    ]
    frame_result = run_text(frame_cmd)
    if frame_result.returncode != 0:
        raise RuntimeError(
            frame_result.stderr.strip() or "Frame PTS exact tidak dapat dibaca."
        )
    data = json.loads(frame_result.stdout or "{}")

    points: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    for frame in data.get("frames", []):
        raw_pts = frame.get("best_effort_timestamp")
        if raw_pts in (None, "N/A"):
            continue
        try:
            relative = Fraction(int(raw_pts)) * time_base - start_time
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if relative < 0:
            continue

        exact_time = (
            str(relative.numerator)
            if relative.denominator == 1
            else f"{relative.numerator}/{relative.denominator}"
        )
        if exact_time in seen_exact:
            continue

        # Jangan lewat float: PTS exact dan boundary SRT harus membulatkan
        # dari Fraction yang sama agar tidak berbeda 1 ms di titik tertentu.
        ms = int(round(relative * 1000))
        if start_ms <= ms <= end_ms:
            # Seek one microsecond before this PTS for the JPEG shown to Gemini.
            # The exact rational PTS itself is still preserved for SmartCut.
            seek_time = max(Fraction(0), relative - Fraction(1, 1_000_000))
            with localcontext() as ctx:
                ctx.prec = 30
                seek_decimal = (
                    Decimal(seek_time.numerator)
                    / Decimal(seek_time.denominator)
                )
                seek_text = format(seek_decimal, "f")

            points.append({
                "time_ms": ms,
                "exact_time": exact_time,
                "seek_time": seek_text,
                "pts": int(raw_pts),
                "time_base": str(time_base),
            })
            seen_exact.add(exact_time)

    points.sort(key=lambda item: Fraction(str(item["exact_time"])))
    return points


def resolve_requested_frame(
    source: Path,
    ffprobe: str,
    requested_ms: int,
    radius_ms: int = 1500,
) -> dict[str, Any]:
    """Snap an explicit user timestamp to the nearest REAL master frame PTS.

    Unlike semantic scene resolution, this never searches for a better scene and
    never uses subtitle safety. The user's requested time is authoritative.
    """
    requested_ms = max(0, int(requested_ms))
    radius_ms = max(250, int(radius_ms))
    frames = probe_frame_timestamps(
        source,
        ffprobe,
        max(0, requested_ms - radius_ms),
        requested_ms + radius_ms,
    )
    if not frames and radius_ms < 4000:
        frames = probe_frame_timestamps(
            source,
            ffprobe,
            max(0, requested_ms - 4000),
            requested_ms + 4000,
        )
    if not frames:
        raise RuntimeError(
            "PTS frame nyata tidak ditemukan di sekitar timestamp manual."
        )
    chosen = min(
        frames,
        key=lambda f: (
            abs(f - requested_ms),
            0 if f <= requested_ms else 1,
        ),
    )
    return {
        "requested_ms": requested_ms,
        "requested_time": format_ms(requested_ms),
        "time_ms": chosen,
        "time": format_ms(chosen),
        "frame_verified": True,
        "frame_delta_ms": chosen - requested_ms,
        "reason": "Timestamp manual dikunci ke PTS frame master nyata terdekat.",
    }

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
