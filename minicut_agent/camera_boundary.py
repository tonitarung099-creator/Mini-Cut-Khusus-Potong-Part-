from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

from .candidates import _run_ffmpeg_lines, detect_visual_boundaries
from .core import fraction_seconds_to_ms
from .frame_resolver import probe_frame_points
from .subtitles import format_ms


DEFAULT_SCENE_THRESHOLD = 0.27
DEFAULT_SEARCH_RADII_MS = (2_000, 4_000)
CAMERA_SCAN_PREROLL_MS = 10_000
CAMERA_DEDUPE_TOLERANCE_MS = 1

_SHOWINFO_TIME_BASE_RE = re.compile(
    r"config in time_base:\s*(?P<num>-?\d+)\/(?P<den>\d+)"
)
_SHOWINFO_PTS_RE = re.compile(r"\bpts:\s*(?P<pts>-?\d+)\b")
_METADATA_FRAME_RE = re.compile(
    r"\bframe:(?P<n>\d+)\s+pts:(?P<pts>-?\d+)\s+pts_time:"
)
_SIGNAL_RE = re.compile(
    r"lavfi\.signalstats\.(?P<key>YAVG|UAVG|VAVG)=(?P<value>-?\d+(?:\.\d+)?)"
)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _bounded_window(
    center_ms: int,
    radius_ms: int,
    duration_ms: int | None,
) -> tuple[int, int]:
    start_ms = max(0, int(center_ms) - max(1, int(radius_ms)))
    end_ms = int(center_ms) + max(1, int(radius_ms))
    if duration_ms is not None and int(duration_ms) > 0:
        end_ms = min(end_ms, int(duration_ms))
    return start_ms, max(start_ms + 1, end_ms)


def _nearest_scene_boundary(requested_ms: int, values: list[int]) -> int:
    return min(
        (int(value) for value in values),
        key=lambda value: (
            abs(value - int(requested_ms)),
            0 if value <= int(requested_ms) else 1,
            value,
        ),
    )


def _nearest_frame_point(
    reference_ms: int,
    points: list[dict[str, Any]],
    *,
    prefer_after_on_tie: bool,
) -> dict[str, Any]:
    if not points:
        raise RuntimeError("PTS frame master tidak ditemukan di sekitar titik potong.")

    eligible = list(points)
    if prefer_after_on_tie:
        after = [
            item for item in eligible
            if int(item["time_ms"]) >= int(reference_ms)
        ]
        if after:
            return min(
                after,
                key=lambda item: (
                    int(item["time_ms"]),
                    str(item.get("exact_time") or ""),
                ),
            )

    return min(
        eligible,
        key=lambda item: (
            abs(int(item["time_ms"]) - int(reference_ms)),
            0 if int(item["time_ms"]) <= int(reference_ms) else 1,
            int(item["time_ms"]),
        ),
    )


def _probe_exact_point(
    source: Path,
    ffprobe: str,
    reference_ms: int,
    duration_ms: int | None,
    *,
    prefer_after_on_tie: bool,
) -> dict[str, Any]:
    for radius_ms in (350, 1_200, 4_000):
        start_ms, end_ms = _bounded_window(
            reference_ms,
            radius_ms,
            duration_ms,
        )
        points = probe_frame_points(source, ffprobe, start_ms, end_ms)
        if points:
            return _nearest_frame_point(
                reference_ms,
                points,
                prefer_after_on_tie=prefer_after_on_tie,
            )
    raise RuntimeError("PTS frame master tidak ditemukan di sekitar titik potong.")


def _parse_showinfo_events(
    lines: list[str],
    *,
    scan_start_ms: int,
    start_ms: int,
    end_ms: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Parse FFmpeg integer PTS without passing through pts_time/milliseconds."""
    time_base: Fraction | None = None
    result: list[dict[str, Any]] = []
    scan_offset = Fraction(int(scan_start_ms), 1000)

    for line in lines:
        config = _SHOWINFO_TIME_BASE_RE.search(line)
        if config:
            try:
                time_base = Fraction(
                    int(config.group("num")),
                    int(config.group("den")),
                )
            except (TypeError, ValueError, ZeroDivisionError):
                time_base = None
            continue

        match = _SHOWINFO_PTS_RE.search(line)
        if match is None or time_base is None:
            continue
        try:
            pts = int(match.group("pts"))
            exact = scan_offset + pts * time_base
            time_ms = fraction_seconds_to_ms(exact)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if int(start_ms) <= time_ms <= int(end_ms):
            result.append({
                "exact_time": exact,
                "time_ms": time_ms,
                "detector": "scene",
                "threshold": float(threshold),
            })
    return result


def _detect_scene_events_exact(
    ffmpeg: str,
    source: Path,
    start_ms: int,
    end_ms: int,
    threshold: float,
    *,
    seek_preroll_ms: int,
) -> list[dict[str, Any]]:
    preroll_ms = max(0, int(seek_preroll_ms))
    scan_start_ms = max(0, int(start_ms) - preroll_ms)
    duration_s = max(0.2, int(end_ms) - scan_start_ms) / 1000
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-ss", f"{scan_start_ms / 1000:.3f}", "-i", str(source),
        "-t", f"{duration_s:.3f}",
        "-vf", f"select='gt(scene,{float(threshold):.6f})',showinfo",
        "-an", "-f", "null", "-",
    ]
    lines = _run_ffmpeg_lines(cmd)
    return _parse_showinfo_events(
        lines,
        scan_start_ms=scan_start_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        threshold=threshold,
    )


def _parse_signal_frames(
    lines: list[str],
    *,
    scan_start_ms: int,
) -> list[dict[str, Any]]:
    time_base: Fraction | None = None
    current: dict[str, Any] | None = None
    parsed: list[dict[str, Any]] = []
    scan_offset = Fraction(int(scan_start_ms), 1000)

    def finish() -> None:
        nonlocal current
        if not current or time_base is None:
            current = None
            return
        if not {"YAVG", "UAVG", "VAVG"}.issubset(current):
            current = None
            return
        pts = int(current["pts"])
        exact = scan_offset + pts * time_base
        parsed.append({
            "exact_time": exact,
            "time_ms": fraction_seconds_to_ms(exact),
            "y": float(current["YAVG"]),
            "u": float(current["UAVG"]),
            "v": float(current["VAVG"]),
        })
        current = None

    for line in lines:
        config = _SHOWINFO_TIME_BASE_RE.search(line)
        if config:
            try:
                time_base = Fraction(
                    int(config.group("num")),
                    int(config.group("den")),
                )
            except (TypeError, ValueError, ZeroDivisionError):
                time_base = None
            continue

        header = _METADATA_FRAME_RE.search(line)
        if header:
            finish()
            current = {"pts": int(header.group("pts"))}
            continue
        signal = _SIGNAL_RE.search(line)
        if signal and current is not None:
            current[signal.group("key")] = float(signal.group("value"))
    finish()
    return parsed


def _detect_signal_frames(
    ffmpeg: str,
    source: Path,
    start_ms: int,
    end_ms: int,
    *,
    context_ms: int = 500,
) -> list[dict[str, Any]]:
    scan_start_ms = max(0, int(start_ms) - max(0, int(context_ms)))
    scan_end_ms = int(end_ms) + max(0, int(context_ms))
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-ss", f"{scan_start_ms / 1000:.3f}", "-i", str(source),
        "-t", f"{max(0.2, scan_end_ms - scan_start_ms) / 1000:.3f}",
        "-vf", "signalstats,metadata=print,showinfo",
        "-an", "-f", "null", "-",
    ]
    return _parse_signal_frames(
        _run_ffmpeg_lines(cmd),
        scan_start_ms=scan_start_ms,
    )


def _signature_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    # Normalize studio/full-range component deltas into a compact 0..~3 score.
    return (
        abs(float(a["y"]) - float(b["y"])) / 219.0
        + abs(float(a["u"]) - float(b["u"])) / 224.0
        + abs(float(a["v"]) - float(b["v"])) / 224.0
    )


def _median_signature(frames: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "y": median(float(item["y"]) for item in frames),
        "u": median(float(item["u"]) for item in frames),
        "v": median(float(item["v"]) for item in frames),
    }


def _signal_change_events(
    frames: list[dict[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    min_distance: float = 0.055,
) -> list[dict[str, Any]]:
    """Detect persistent luma/chroma changes and reject one-frame flashes.

    We compare short medians on both sides. A one-frame A->B->A flash collapses
    back to A in the post median and therefore becomes ambiguous instead of a
    verified camera boundary. Very short montage shots are not called invalid;
    they are left to the scene detector and the ambiguity rule below.
    """
    events: list[dict[str, Any]] = []
    if len(frames) < 4:
        return events

    for index in range(1, len(frames)):
        point = frames[index]
        point_ms = int(point["time_ms"])
        if not (int(start_ms) <= point_ms <= int(end_ms)):
            continue
        before = frames[max(0, index - 3):index]
        after = frames[index:min(len(frames), index + 3)]
        if len(before) < 2 or len(after) < 2:
            continue
        before_sig = _median_signature(before)
        after_sig = _median_signature(after)
        distance = _signature_distance(before_sig, after_sig)
        if distance < float(min_distance):
            continue
        events.append({
            "exact_time": Fraction(point["exact_time"]),
            "time_ms": point_ms,
            "detector": "signal",
            "signal_distance": float(distance),
        })
    return events


def _merge_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[Fraction, dict[str, Any]] = {}
    for event in events:
        exact = Fraction(event["exact_time"])
        item = merged.setdefault(exact, {
            "exact_time": exact,
            "time_ms": fraction_seconds_to_ms(exact),
            "detectors": set(),
            "thresholds": [],
            "signal_distance": 0.0,
        })
        detector = str(event.get("detector") or "")
        if detector:
            item["detectors"].add(detector)
        if event.get("threshold") is not None:
            item["thresholds"].append(float(event["threshold"]))
        item["signal_distance"] = max(
            float(item.get("signal_distance") or 0.0),
            float(event.get("signal_distance") or 0.0),
        )
    result = list(merged.values())
    result.sort(key=lambda item: Fraction(item["exact_time"]))
    return result


def _mark_ambiguous_rapid_reversals(
    events: list[dict[str, Any]],
    *,
    window_ms: int = 120,
) -> None:
    for index, event in enumerate(events):
        exact = Fraction(event["exact_time"])
        ambiguous = False
        for other_index, other in enumerate(events):
            if index == other_index:
                continue
            delta_ms = abs(float(Fraction(other["exact_time"]) - exact) * 1000.0)
            if delta_ms <= float(window_ms):
                ambiguous = True
                break
        event["ambiguous_rapid_change"] = ambiguous


def detect_camera_boundary_events(
    ffmpeg: str,
    source: Path,
    start_ms: int,
    end_ms: int,
    *,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    seek_preroll_ms: int = CAMERA_SCAN_PREROLL_MS,
) -> list[dict[str, Any]]:
    """Return exact rational camera-boundary candidates for a small window."""
    thresholds: list[float] = []
    for value in (
        float(scene_threshold),
        float(scene_threshold) * 0.65,
        float(scene_threshold) * 0.40,
        0.06,
    ):
        normalized = max(0.035, min(0.95, float(value)))
        if all(abs(normalized - existing) > 1e-9 for existing in thresholds):
            thresholds.append(normalized)

    raw_events: list[dict[str, Any]] = []
    for threshold in thresholds:
        raw_events.extend(_detect_scene_events_exact(
            ffmpeg,
            source,
            start_ms,
            end_ms,
            threshold,
            seek_preroll_ms=seek_preroll_ms,
        ))

    signal_frames = _detect_signal_frames(
        ffmpeg,
        source,
        start_ms,
        end_ms,
    )
    raw_events.extend(_signal_change_events(
        signal_frames,
        start_ms=start_ms,
        end_ms=end_ms,
    ))

    events = _merge_events(raw_events)
    _mark_ambiguous_rapid_reversals(events)
    for event in events:
        event["camera_boundary_verified"] = not bool(
            event.get("ambiguous_rapid_change")
        )
    return events


def _nearest_exact_event(
    requested_ms: int,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    requested = Fraction(int(requested_ms), 1000)
    return min(
        events,
        key=lambda item: (
            abs(Fraction(item["exact_time"]) - requested),
            0 if Fraction(item["exact_time"]) <= requested else 1,
            Fraction(item["exact_time"]),
        ),
    )


def _probe_matching_exact_point(
    source: Path,
    ffprobe: str,
    exact_boundary: Fraction,
    duration_ms: int | None,
) -> dict[str, Any] | None:
    reference_ms = fraction_seconds_to_ms(exact_boundary)
    for radius_ms in (350, 1_200, 4_000):
        start_ms, end_ms = _bounded_window(reference_ms, radius_ms, duration_ms)
        for point in probe_frame_points(source, ffprobe, start_ms, end_ms):
            try:
                if Fraction(str(point["exact_time"])) == exact_boundary:
                    return point
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    return None


def _unresolved_result(
    requested_ms: int,
    *,
    search_radius_ms: int,
    scene_threshold: float,
    reason: str,
    raw_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "requested_ms": int(requested_ms),
        "requested_time": format_ms(int(requested_ms)),
        "time_ms": int(requested_ms),
        "time": format_ms(int(requested_ms)),
        "exact_time": None,
        "pts": None,
        "time_base": None,
        "camera_change_found": bool(raw_event),
        "fallback_to_nearest_frame": False,
        "raw_scene_boundary_ms": (
            int(raw_event["time_ms"]) if raw_event else None
        ),
        "raw_scene_boundary": (
            format_ms(int(raw_event["time_ms"])) if raw_event else ""
        ),
        "raw_scene_boundary_exact": (
            _fraction_text(Fraction(raw_event["exact_time"])) if raw_event else None
        ),
        "shift_ms": 0,
        "search_radius_ms": int(search_radius_ms),
        "scene_threshold": float(scene_threshold),
        "frame_verified": False,
        "pts_verified": False,
        "camera_boundary_verified": False,
        "needs_review": True,
        "frame_side": "unresolved-camera-boundary",
        "reason": str(reason),
    }


def resolve_camera_boundary(
    source: Path,
    ffmpeg: str,
    ffprobe: str,
    requested_ms: int,
    *,
    duration_ms: int | None = None,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    search_radii_ms: tuple[int, ...] = DEFAULT_SEARCH_RADII_MS,
    require_camera_change: bool = False,
) -> dict[str, Any]:
    """Resolve a target to the exact first master frame of a camera change.

    Exact detection keeps integer FFmpeg PTS + rational time_base until it can
    be matched to the identical master-frame PTS from ffprobe. Milliseconds are
    display/SRT values only. When ``require_camera_change`` is true, a missing or
    ambiguous boundary is returned as REVIEW and is never silently replaced by
    the nearest ordinary frame.
    """
    requested_ms = max(0, int(requested_ms))
    if duration_ms is not None and int(duration_ms) > 0:
        requested_ms = min(requested_ms, int(duration_ms))

    radii = tuple(
        sorted({max(250, int(value)) for value in search_radii_ms})
    ) or DEFAULT_SEARCH_RADII_MS
    last_ambiguous: dict[str, Any] | None = None

    for radius_ms in radii:
        start_ms, end_ms = _bounded_window(
            requested_ms,
            radius_ms,
            duration_ms,
        )
        events = detect_camera_boundary_events(
            ffmpeg,
            source,
            start_ms,
            end_ms,
            scene_threshold=float(scene_threshold),
            seek_preroll_ms=CAMERA_SCAN_PREROLL_MS,
        )
        verified_events = [
            item for item in events
            if bool(item.get("camera_boundary_verified"))
        ]
        if not verified_events:
            if events:
                last_ambiguous = _nearest_exact_event(requested_ms, events)
            continue

        event = _nearest_exact_event(requested_ms, verified_events)
        exact_boundary = Fraction(event["exact_time"])
        point = _probe_matching_exact_point(
            source,
            ffprobe,
            exact_boundary,
            duration_ms,
        )
        if point is None:
            last_ambiguous = event
            continue

        final_ms = int(point["time_ms"])
        return {
            "requested_ms": requested_ms,
            "requested_time": format_ms(requested_ms),
            "time_ms": final_ms,
            "time": format_ms(final_ms),
            "exact_time": str(point["exact_time"]),
            "pts": point.get("pts"),
            "time_base": point.get("time_base"),
            "camera_change_found": True,
            "fallback_to_nearest_frame": False,
            "raw_scene_boundary_ms": int(event["time_ms"]),
            "raw_scene_boundary": format_ms(int(event["time_ms"])),
            "raw_scene_boundary_exact": _fraction_text(exact_boundary),
            "shift_ms": final_ms - requested_ms,
            "search_radius_ms": int(radius_ms),
            "scene_threshold": float(scene_threshold),
            "detectors": sorted(str(x) for x in event.get("detectors", set())),
            "detector_thresholds": sorted(set(event.get("thresholds") or [])),
            "signal_distance": float(event.get("signal_distance") or 0.0),
            "frame_verified": True,
            "pts_verified": True,
            "camera_boundary_verified": True,
            "needs_review": False,
            "frame_side": "first-frame-new-shot",
            "reason": (
                "Pergantian kamera diverifikasi dan dipetakan ke PTS rasional "
                "frame master yang sama tanpa pembulatan milidetik."
            ),
        }

    if require_camera_change:
        reason = (
            "Kandidat perubahan visual terlalu singkat/ambigu (misalnya flash atau "
            "montage sangat cepat); boundary belum boleh dianggap terverifikasi."
            if last_ambiguous is not None
            else "Pergantian kamera tidak terbukti dalam radius pencarian; perlu review."
        )
        return _unresolved_result(
            requested_ms,
            search_radius_ms=int(radii[-1]),
            scene_threshold=float(scene_threshold),
            reason=reason,
            raw_event=last_ambiguous,
        )

    # Legacy ordinary-timestamp mode: snapping to a real frame remains available
    # for callers that did not request a mandatory camera transition. It is never
    # labeled camera-boundary verified.
    point = _probe_exact_point(
        source,
        ffprobe,
        requested_ms,
        duration_ms,
        prefer_after_on_tie=False,
    )
    final_ms = int(point["time_ms"])
    return {
        "requested_ms": requested_ms,
        "requested_time": format_ms(requested_ms),
        "time_ms": final_ms,
        "time": format_ms(final_ms),
        "exact_time": str(point["exact_time"]),
        "pts": point.get("pts"),
        "time_base": point.get("time_base"),
        "camera_change_found": False,
        "fallback_to_nearest_frame": True,
        "raw_scene_boundary_ms": None,
        "raw_scene_boundary": "",
        "raw_scene_boundary_exact": None,
        "shift_ms": final_ms - requested_ms,
        "search_radius_ms": int(radii[-1]),
        "scene_threshold": float(scene_threshold),
        "frame_verified": True,
        "pts_verified": True,
        "camera_boundary_verified": False,
        "needs_review": True,
        "frame_side": "nearest-master-frame",
        "reason": (
            "Pergantian kamera tidak ditemukan; mode timestamp biasa memakai PTS "
            "frame master terdekat, tetapi hasil ini bukan boundary kamera terverifikasi."
        ),
    }
