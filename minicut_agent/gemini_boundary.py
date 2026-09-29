from __future__ import annotations

from typing import Any

from .gemini import (
    EXACT_COARSE_MAX_FRAMES,
    GeminiClient,
    _evenly_sample_times,
    _frame_index,
)
from .subtitles import SubtitleTrack, format_ms


MID_CONTEXT_RADIUS_MS = 800
MID_MAX_FRAMES = 31
FINAL_CONTEXT_RADIUS_MS = 250
FINAL_MAX_FRAMES = 65


class BoundaryAwareGeminiClient(GeminiClient):
    """Gemini exact-frame chooser with time-based refinement windows.

    The original fine pass always used 31 consecutive frames. At 120 fps that
    covers only about 258 ms, so a coarse choice could exclude the true first
    frame of the new shot. This implementation refines by *duration* first,
    then presents every real master frame in a small final time window. Gemini
    still chooses the final frame; local code does not silently move it.
    """

    @staticmethod
    def _normalise_points(frame_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points = [
            {
                "time_ms": max(0, int(item["time_ms"])),
                "exact_time": str(item["exact_time"]),
                "seek_time": str(
                    item.get("seek_time")
                    or max(0, int(item["time_ms"])) / 1000
                ),
            }
            for item in frame_points
            if item.get("exact_time") not in (None, "")
        ]
        points.sort(key=lambda item: (item["time_ms"], item["exact_time"]))
        return points

    @staticmethod
    def _points_near(
        points: list[dict[str, Any]],
        center_ms: int,
        radius_ms: int,
    ) -> list[dict[str, Any]]:
        result = [
            item for item in points
            if abs(int(item["time_ms"]) - int(center_ms)) <= int(radius_ms)
        ]
        if result:
            return result
        nearest = min(
            points,
            key=lambda item: abs(int(item["time_ms"]) - int(center_ms)),
        )
        return [nearest]

    def choose_exact_master_frame(
        self,
        ffmpeg: str,
        source,
        target_ms: int,
        boundary_hint_ms: int,
        frame_points: list[dict[str, Any]],
        subtitles: SubtitleTrack | None,
    ) -> dict[str, Any]:
        points = self._normalise_points(frame_points)
        if not points:
            raise ValueError("Tidak ada frame master untuk dipilih Gemini.")

        coarse_points = _evenly_sample_times(points, EXACT_COARSE_MAX_FRAMES)
        coarse = self._choose_exact_frame_pass(
            ffmpeg,
            source,
            target_ms,
            boundary_hint_ms,
            coarse_points,
            subtitles,
            phase="coarse",
        )
        coarse_index = _frame_index(
            coarse.get("selected_frame_index"), len(coarse_points)
        )
        coarse_point = coarse_points[coarse_index - 1]
        coarse_ms = int(coarse_point["time_ms"])

        mid_pool = self._points_near(
            points,
            coarse_ms,
            MID_CONTEXT_RADIUS_MS,
        )
        mid_points = _evenly_sample_times(mid_pool, MID_MAX_FRAMES)
        mid = self._choose_exact_frame_pass(
            ffmpeg,
            source,
            target_ms,
            coarse_ms,
            mid_points,
            subtitles,
            phase="refine",
        )
        mid_index = _frame_index(mid.get("selected_frame_index"), len(mid_points))
        mid_point = mid_points[mid_index - 1]
        mid_ms = int(mid_point["time_ms"])

        final_points = self._points_near(
            points,
            mid_ms,
            FINAL_CONTEXT_RADIUS_MS,
        )
        if len(final_points) > FINAL_MAX_FRAMES:
            final_points = _evenly_sample_times(final_points, FINAL_MAX_FRAMES)

        fine = self._choose_exact_frame_pass(
            ffmpeg,
            source,
            target_ms,
            mid_ms,
            final_points,
            subtitles,
            phase="fine",
        )
        fine_index = _frame_index(
            fine.get("selected_frame_index"), len(final_points)
        )
        selected = final_points[fine_index - 1]
        selected_ms = int(selected["time_ms"])
        selected_exact = str(selected["exact_time"])

        span_ms = (
            int(final_points[-1]["time_ms"]) - int(final_points[0]["time_ms"])
            if len(final_points) > 1 else 0
        )
        return {
            "decision": "CUT_FOUND",
            "selected_time_ms": selected_ms,
            "selected_time": format_ms(selected_ms),
            "selected_time_exact": selected_exact,
            "selected_frame_index": fine_index,
            "confidence": float(fine.get("confidence") or 0.0),
            "needs_review": bool(fine.get("needs_review", False)),
            "cut_intent": str(fine.get("cut_intent") or "scene_transition"),
            "reason": str(
                fine.get("reason")
                or "Gemini memilih frame master final dari PTS yang tersedia."
            ),
            "frame_verified": True,
            "frame_delta_ms": 0,
            "frame_authority": "gemini",
            "frame_selection": "gemini-exact-master-pts-time-refined",
            "coarse_selected_ms": coarse_ms,
            "coarse_selected_time": format_ms(coarse_ms),
            "coarse_selected_time_exact": str(coarse_point["exact_time"]),
            "refine_selected_ms": mid_ms,
            "refine_selected_time": format_ms(mid_ms),
            "refine_selected_time_exact": str(mid_point["exact_time"]),
            "exact_frame_pool_count": len(points),
            "exact_frame_mid_count": len(mid_points),
            "exact_frame_fine_count": len(final_points),
            "exact_frame_fine_span_ms": span_ms,
            "usage": self.usage.__dict__.copy(),
        }
