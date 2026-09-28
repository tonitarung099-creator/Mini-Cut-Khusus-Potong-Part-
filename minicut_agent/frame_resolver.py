from __future__ import annotations

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from .core import fraction_seconds_to_ms, run_text
from .subtitles import SubtitleTrack, format_ms


def _fraction_text(value: Fraction) -> str:
    with localcontext() as ctx:
        ctx.prec = 30
        return format(
            Decimal(value.numerator) / Decimal(value.denominator),
            "f",
        )


def _probe_frame_timestamp_payload(
    source: Path,
    ffprobe: str,
    interval_start: Fraction,
    interval_end: Fraction,
) -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-read_intervals",
        f"{_fraction_text(interval_start)}%{_fraction_text(interval_end)}",
        "-show_frames",
        "-show_format",
        "-show_entries", "frame=best_effort_timestamp_time:format=start_time",
        "-of", "json",
        str(source),
    ]
    result = run_text(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Frame timestamp tidak dapat dibaca."
        )
    return json.loads(result.stdout or "{}")


def probe_frame_timestamps(
    source: Path,
    ffprobe: str,
    start_ms: int,
    end_ms: int,
) -> list[int]:
    """Return real frame timestamps on MiniCut's zero-based UI timeline.

    This helper intentionally keeps the lightweight timestamp-only ffprobe path
    used by frame stepping and semantic candidate selection. For normal files
    whose container starts at zero it needs one probe, preserving the legacy
    behavior and exact decimal rounding. If ffprobe reports a non-zero
    ``format.start_time`` (common in TS/MTS/remuxed media), it repeats the read
    using an absolute container interval and subtracts that start offset from
    every returned frame timestamp.
    """
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms + 1, int(end_ms))
    relative_start = Fraction(start_ms, 1000)
    relative_end = Fraction(end_ms, 1000)

    data = _probe_frame_timestamp_payload(
        source,
        ffprobe,
        relative_start,
        relative_end,
    )
    try:
        start_time = Fraction(
            str((data.get("format") or {}).get("start_time") or "0")
        )
    except Exception:
        start_time = Fraction(0)

    # -read_intervals is interpreted on the source/container clock. Once we
    # discover a non-zero start time, repeat the small read window at the
    # correct absolute position. The first read is only clock discovery in that
    # case; its frames are deliberately ignored.
    if start_time != 0:
        data = _probe_frame_timestamp_payload(
            source,
            ffprobe,
            start_time + relative_start,
            start_time + relative_end,
        )

    frames: list[int] = []
    for frame in data.get("frames", []):
        raw = frame.get("best_effort_timestamp_time")
        if raw in (None, "N/A"):
            continue
        try:
            relative = Fraction(str(raw)) - start_time
            ms = fraction_seconds_to_ms(relative)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if start_ms - 1000 <= ms <= end_ms + 1000:
            frames.append(ms)
    return sorted(set(frames))


def _probe_container_start_time(source: Path, ffprobe: str) -> Fraction:
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=start_time",
        "-of", "json",
        str(source),
    ]
    result = run_text(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Clock media tidak dapat dibaca."
        )
    try:
        return Fraction(
            str((json.loads(result.stdout or "{}").get("format") or {}).get("start_time") or "0")
        )
    except Exception:
        return Fraction(0)


def probe_keyframes_relative(source: Path, ffprobe: str) -> list[int]:
    """Return keyframes relative to MiniCut's zero-based media timeline."""
    start_time = _probe_container_start_time(source, ffprobe)
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0",
        str(source),
    ]
    result = run_text(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Keyframe tidak dapat dibaca.")

    points: list[int] = []
    for line in result.stdout.splitlines():
        raw = line.strip().split(",", 1)[0]
        if not raw or raw == "N/A":
            continue
        try:
            relative = Fraction(raw) - start_time
        except (ValueError, ZeroDivisionError):
            continue
        if relative < 0:
            continue
        points.append(fraction_seconds_to_ms(relative))

    points = sorted(set(points))
    if not points:
        raise RuntimeError("Tidak ada keyframe yang ditemukan.")
    return points


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
    abs_start_text = _fraction_text(absolute_start)
    abs_end_text = _fraction_text(absolute_end)

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
        ms = fraction_seconds_to_ms(relative)
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
