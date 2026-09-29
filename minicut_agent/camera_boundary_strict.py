from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

from . import camera_boundary as _base_boundary
from .camera_boundary import (
    _detect_signal_frames,
    _unresolved_result,
    resolve_camera_boundary as _resolve_camera_boundary,
)
from .core import fraction_seconds_to_ms


def _median_signature(frames: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "y": median(float(item["y"]) for item in frames),
        "u": median(float(item["u"]) for item in frames),
        "v": median(float(item["v"]) for item in frames),
    }


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return (
        abs(float(a["y"]) - float(b["y"])) / 219.0
        + abs(float(a["u"]) - float(b["u"])) / 224.0
        + abs(float(a["v"]) - float(b["v"])) / 224.0
    )


def _persistent_signal_change_events(
    frames: list[dict[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    min_distance: float = 0.055,
) -> list[dict[str, Any]]:
    """Emit the first real frame whose luma/chroma state changed persistently."""
    events: list[dict[str, Any]] = []
    if len(frames) < 5:
        return events

    last_emitted_index = -10_000
    for index in range(2, len(frames) - 1):
        if index - last_emitted_index <= 2:
            continue

        point = frames[index]
        point_ms = int(point["time_ms"])
        if not (int(start_ms) <= point_ms <= int(end_ms)):
            continue

        before = frames[max(0, index - 3):index]
        future = frames[index:min(len(frames), index + 3)]
        if len(before) < 2 or len(future) < 2:
            continue

        before_sig = _median_signature(before)
        future_sig = _median_signature(future)
        current_sig = {
            "y": float(point["y"]),
            "u": float(point["u"]),
            "v": float(point["v"]),
        }
        entry_jump = _distance(before_sig, current_sig)
        persistent_jump = _distance(before_sig, future_sig)
        current_to_future = _distance(current_sig, future_sig)

        if entry_jump < float(min_distance):
            continue
        if persistent_jump < float(min_distance):
            continue
        if current_to_future > max(0.030, float(min_distance) * 0.75):
            continue

        events.append({
            "exact_time": Fraction(point["exact_time"]),
            "time_ms": point_ms,
            "detector": "signal",
            "signal_distance": float(entry_jump),
        })
        last_emitted_index = index
    return events


def _strict_mark_ambiguous_rapid_reversals(
    events: list[dict[str, Any]],
    *,
    window_ms: int = 120,
) -> None:
    """Separate duplicate detector hits from genuinely rapid/ambiguous cuts."""
    for event in events:
        event["ambiguous_rapid_change"] = False

    if len(events) < 2:
        return

    index = 0
    while index < len(events):
        cluster = [events[index]]
        end = index + 1
        while end < len(events):
            previous = Fraction(cluster[-1]["exact_time"])
            current = Fraction(events[end]["exact_time"])
            delta_ms = float(current - previous) * 1000.0
            if delta_ms > float(window_ms):
                break
            cluster.append(events[end])
            end += 1

        if len(cluster) > 1:
            signal_events = [
                item for item in cluster
                if "signal" in set(item.get("detectors") or set())
            ]
            if len(signal_events) == 1:
                signal_event = signal_events[0]
                for item in cluster:
                    item["ambiguous_rapid_change"] = item is not signal_event
                    if item is not signal_event:
                        item["duplicate_of_signal_event"] = True
            else:
                for item in cluster:
                    item["ambiguous_rapid_change"] = True
        index = end


def _probe_master_point_with_seek_tolerance(
    source: Path,
    ffprobe: str,
    exact_boundary: Fraction,
    duration_ms: int | None,
) -> dict[str, Any] | None:
    """Map detector PTS to the identical master frame without millisecond rounding.

    Input seeking with a decimal -ss can shift filter PTS by a fraction of one
    stream time-base tick (for example 31.25 microseconds). We therefore compare
    rational values directly and accept only a tiny delta: at most 1 ms and at
    most one quarter of the local master-frame cadence. A one-frame-late event
    (about 40 ms at 25 fps / 41.7 ms at 23.976 fps) can never pass this gate.
    The returned timestamp is always the original ffprobe master PTS.
    """
    reference_ms = fraction_seconds_to_ms(exact_boundary)
    for radius_ms in (350, 1_200, 4_000):
        start_ms, end_ms = _base_boundary._bounded_window(
            reference_ms,
            radius_ms,
            duration_ms,
        )
        points = _base_boundary.probe_frame_points(
            source,
            ffprobe,
            start_ms,
            end_ms,
        )
        if not points:
            continue

        parsed: list[tuple[Fraction, dict[str, Any]]] = []
        for point in points:
            try:
                parsed.append((Fraction(str(point["exact_time"])), point))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
        if not parsed:
            continue

        parsed.sort(key=lambda item: item[0])
        nearest_exact, nearest_point = min(
            parsed,
            key=lambda item: abs(item[0] - exact_boundary),
        )
        delta = abs(nearest_exact - exact_boundary)
        if delta == 0:
            return nearest_point

        cadences = [
            parsed[i + 1][0] - parsed[i][0]
            for i in range(len(parsed) - 1)
            if parsed[i + 1][0] > parsed[i][0]
        ]
        cadence_limit = min(cadences) / 4 if cadences else Fraction(1, 1000)
        tolerance = min(Fraction(1, 1000), cadence_limit)
        if delta <= tolerance:
            matched = dict(nearest_point)
            matched["detector_exact_time"] = str(exact_boundary)
            matched["detector_master_delta"] = str(delta)
            return matched
    return None


# detect_camera_boundary_events / resolve_camera_boundary look these globals up
# dynamically. Exact detection stays in camera_boundary.py; strict runtime only
# replaces policies that need persistence, duplicate handling and micro-offset
# reconciliation back to the original master PTS.
_base_boundary._signal_change_events = _persistent_signal_change_events
_base_boundary._mark_ambiguous_rapid_reversals = _strict_mark_ambiguous_rapid_reversals
_base_boundary._probe_matching_exact_point = _probe_master_point_with_seek_tolerance


def _looks_like_transient_flash(
    ffmpeg: str,
    source: Path,
    exact_time: Fraction,
) -> bool:
    """Reject A->B->A one-frame flashes even when scene emits one edge only."""
    center_ms = fraction_seconds_to_ms(exact_time)
    frames = _detect_signal_frames(
        ffmpeg,
        source,
        max(0, center_ms - 300),
        center_ms + 300,
        context_ms=120,
    )
    if len(frames) < 5:
        return False

    index = min(
        range(len(frames)),
        key=lambda i: abs(Fraction(frames[i]["exact_time"]) - exact_time),
    )
    candidate = frames[index]
    if abs(float(Fraction(candidate["exact_time"]) - exact_time)) > 0.08:
        return False

    before = frames[max(0, index - 3):index]
    after = frames[index + 1:min(len(frames), index + 4)]
    if len(before) < 2 or len(after) < 2:
        return False

    before_sig = _median_signature(before)
    after_sig = _median_signature(after)
    candidate_sig = {
        "y": float(candidate["y"]),
        "u": float(candidate["u"]),
        "v": float(candidate["v"]),
    }
    entry_jump = _distance(before_sig, candidate_sig)
    persistent_jump = _distance(before_sig, after_sig)
    return entry_jump >= 0.055 and persistent_jump < 0.025


def resolve_camera_boundary(
    source: Path,
    ffmpeg: str,
    ffprobe: str,
    requested_ms: int,
    *,
    duration_ms: int | None = None,
    scene_threshold: float = 0.27,
    search_radii_ms: tuple[int, ...] = (2_000, 4_000),
    require_camera_change: bool = False,
) -> dict[str, Any]:
    result = _resolve_camera_boundary(
        source,
        ffmpeg,
        ffprobe,
        requested_ms,
        duration_ms=duration_ms,
        scene_threshold=scene_threshold,
        search_radii_ms=search_radii_ms,
        require_camera_change=require_camera_change,
    )
    if not bool(result.get("camera_boundary_verified")):
        return result

    try:
        exact = Fraction(str(result["exact_time"]))
        transient = _looks_like_transient_flash(ffmpeg, source, exact)
    except Exception:
        transient = False
    if not transient:
        result["persistence_verified"] = True
        return result

    if require_camera_change:
        review = _unresolved_result(
            int(requested_ms),
            search_radius_ms=int(result.get("search_radius_ms") or search_radii_ms[-1]),
            scene_threshold=float(scene_threshold),
            reason=(
                "Perubahan visual hanya bertahan sangat singkat lalu kembali ke "
                "gambar sebelumnya; kandidat dianggap flash/insert ambigu dan perlu review."
            ),
            raw_event={
                "time_ms": int(result["time_ms"]),
                "exact_time": exact,
            },
        )
        review["transient_flash_detected"] = True
        review["persistence_verified"] = False
        return review

    result["camera_boundary_verified"] = False
    result["needs_review"] = True
    result["transient_flash_detected"] = True
    result["persistence_verified"] = False
    result["reason"] = (
        "Frame nyata ditemukan, tetapi perubahan visual hanya sesaat dan tidak "
        "diverifikasi sebagai boundary kamera yang persisten."
    )
    return result
