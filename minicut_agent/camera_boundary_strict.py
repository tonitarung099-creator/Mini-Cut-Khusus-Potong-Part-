from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

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


def _looks_like_transient_flash(
    ffmpeg: str,
    source: Path,
    exact_time: Fraction,
) -> bool:
    """True when the candidate frame changes sharply but immediately returns.

    FFmpeg's scene score can emit only the entry edge of an A->B->A one-frame
    flash. Compare the candidate itself with a stable median before and after it:
    a large pre->candidate jump followed by a near-identical pre/post state is a
    transient flash (or an ultra-short ambiguous insert), not an auto-verifiable
    part boundary.
    """
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
