from __future__ import annotations

from pathlib import Path
from typing import Any

from .candidates import detect_visual_boundaries
from .frame_resolver import probe_frame_points
from .subtitles import format_ms


DEFAULT_SCENE_THRESHOLD = 0.27
DEFAULT_SEARCH_RADII_MS = (2_000, 4_000)


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

    def key(item: dict[str, Any]):
        value = int(item["time_ms"])
        if prefer_after_on_tie:
            side = 0 if value >= int(reference_ms) else 1
        else:
            side = 0 if value <= int(reference_ms) else 1
        return (abs(value - int(reference_ms)), side, value)

    return min(points, key=key)


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


def resolve_camera_boundary(
    source: Path,
    ffmpeg: str,
    ffprobe: str,
    requested_ms: int,
    *,
    duration_ms: int | None = None,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    search_radii_ms: tuple[int, ...] = DEFAULT_SEARCH_RADII_MS,
) -> dict[str, Any]:
    """Resolve a manual target time to the nearest real camera/shot change.

    The requested time remains the user's authoritative *target*. MiniCut first
    searches for a visual scene boundary around that target, then maps the
    detected boundary to an exact rational PTS from the source master. If no
    visual boundary is found, the fallback is the nearest real master frame so
    SmartCut still receives a frame-valid boundary instead of an invented time.
    """
    requested_ms = max(0, int(requested_ms))
    if duration_ms is not None and int(duration_ms) > 0:
        requested_ms = min(requested_ms, int(duration_ms))

    radii = tuple(
        sorted({max(250, int(value)) for value in search_radii_ms})
    ) or DEFAULT_SEARCH_RADII_MS

    for radius_ms in radii:
        start_ms, end_ms = _bounded_window(
            requested_ms,
            radius_ms,
            duration_ms,
        )
        visual_points = detect_visual_boundaries(
            ffmpeg,
            source,
            start_ms,
            end_ms,
            scene_threshold=float(scene_threshold),
        )
        if not visual_points:
            continue

        boundary_ms = _nearest_scene_boundary(requested_ms, visual_points)
        point = _probe_exact_point(
            source,
            ffprobe,
            boundary_ms,
            duration_ms,
            prefer_after_on_tie=True,
        )
        final_ms = int(point["time_ms"])
        return {
            "requested_ms": requested_ms,
            "requested_time": format_ms(requested_ms),
            "time_ms": final_ms,
            "time": format_ms(final_ms),
            "exact_time": str(point["exact_time"]),
            "camera_change_found": True,
            "fallback_to_nearest_frame": False,
            "raw_scene_boundary_ms": int(boundary_ms),
            "raw_scene_boundary": format_ms(int(boundary_ms)),
            "shift_ms": final_ms - requested_ms,
            "search_radius_ms": int(radius_ms),
            "scene_threshold": float(scene_threshold),
            "frame_verified": True,
            "reason": "Dikunci ke PTS frame master pada pergantian kamera terdekat.",
        }

    # No visual cut was detected even after expansion. Keep the operation
    # frame-accurate by snapping only to the nearest *real* source frame.
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
        "camera_change_found": False,
        "fallback_to_nearest_frame": True,
        "raw_scene_boundary_ms": None,
        "raw_scene_boundary": "",
        "shift_ms": final_ms - requested_ms,
        "search_radius_ms": int(radii[-1]),
        "scene_threshold": float(scene_threshold),
        "frame_verified": True,
        "reason": (
            "Pergantian kamera tidak ditemukan dalam radius pencarian; "
            "dipakai PTS frame master terdekat."
        ),
    }
