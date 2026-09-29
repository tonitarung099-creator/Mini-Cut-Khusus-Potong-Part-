from __future__ import annotations

from fractions import Fraction
from typing import Any

from . import workers as workers_module
from .camera_boundary_strict import resolve_camera_boundary
from .core import clock_text, find_tool
from .final_window import FinalMiniCutWindow
from .gemini_boundary import BoundaryAwareGeminiClient
from .ui import MiniCutWindow as BaseMiniCutWindow


class RuntimeMiniCutWindow(FinalMiniCutWindow):
    """Final runtime: strict boundary detection plus no-silent-snap AI checks."""

    def __init__(self):
        # FilmCutWorker resolves this module-level binding when it starts.
        # Replace only the client class; API-key rotation and worker lifecycle
        # remain unchanged.
        workers_module.GeminiClient = BoundaryAwareGeminiClient
        super().__init__()

    def _gemini_chat_ready(self, result: dict):
        actions = result.get("actions") or []
        camera_action = any(
            isinstance(item, dict)
            and str(item.get("tool") or "").strip()
            in {"manual_frame_cut", "frame_cut", "add_cut"}
            for item in actions
        )
        if not camera_action:
            return super()._gemini_chat_ready(result)

        safe = dict(result)
        safe["reply"] = (
            "MiniCut memakai timestamp sebagai target lalu memverifikasi pergantian "
            "kamera terhadap PTS frame master. Cut hanya dipasang bila frame pertama "
            "shot baru terbukti; bila tidak, titik ditandai REVIEW dan tidak difallback "
            "ke frame terdekat."
        )
        # Skip the older optimistic CameraAware wording while preserving the
        # base usage/history/action machinery. Actions still resolve through
        # this runtime's strict tool_add_cut implementation.
        return BaseMiniCutWindow._gemini_chat_ready(self, safe)

    def _verify_film_cut_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Verify Gemini's exact PTS; never replace it with a nearby boundary."""
        ffmpeg = find_tool("ffmpeg")
        ffprobe = find_tool("ffprobe")
        validated: list[dict[str, Any]] = []

        for raw in results:
            item = dict(raw)
            selected_ms = int(item.get("selected_time_ms") or 0)
            if selected_ms <= 0:
                validated.append(item)
                continue

            selected_exact: Fraction | None = None
            try:
                if item.get("selected_time_exact") not in (None, ""):
                    selected_exact = Fraction(str(item["selected_time_exact"]))
            except (TypeError, ValueError, ZeroDivisionError):
                selected_exact = None

            item["gemini_selected_time_ms"] = selected_ms
            item["gemini_selected_time"] = str(item.get("selected_time") or "")
            item["gemini_selected_time_exact"] = (
                str(selected_exact) if selected_exact is not None else None
            )

            if not ffmpeg or not ffprobe or not self.model.source or selected_exact is None:
                item["camera_boundary_verified"] = False
                item["pts_verified"] = bool(item.get("frame_verified"))
                item["needs_review"] = True
                item["fallback_to_nearest_frame"] = False
                item["camera_verification_needs_review"] = True
                item["camera_verification_reason"] = (
                    "PTS pilihan Gemini tidak dapat diverifikasi terhadap boundary kamera."
                )
                validated.append(item)
                continue

            resolved = resolve_camera_boundary(
                self.model.source,
                ffmpeg,
                ffprobe,
                selected_ms,
                duration_ms=self.model.duration_ms,
                search_radii_ms=(250, 600),
                require_camera_change=True,
            )
            resolved_exact: Fraction | None = None
            try:
                if resolved.get("exact_time") not in (None, ""):
                    resolved_exact = Fraction(str(resolved["exact_time"]))
            except (TypeError, ValueError, ZeroDivisionError):
                resolved_exact = None

            exact_match = bool(
                resolved.get("camera_boundary_verified")
                and resolved_exact is not None
                and resolved_exact == selected_exact
            )
            if exact_match:
                item["camera_boundary_verified"] = True
                item["pts_verified"] = True
                item["fallback_to_nearest_frame"] = False
                item["camera_verification_needs_review"] = False
                item["selected_source_pts"] = resolved.get("pts")
                item["selected_time_base"] = resolved.get("time_base")
                item["raw_scene_boundary_exact"] = resolved.get(
                    "raw_scene_boundary_exact"
                )
                item["persistence_verified"] = bool(
                    resolved.get("persistence_verified", True)
                )
                item["camera_verification_reason"] = (
                    "PTS frame pilihan Gemini sama persis dengan PTS boundary kamera "
                    "persisten yang diverifikasi lokal."
                )
            else:
                item["camera_boundary_verified"] = False
                item["pts_verified"] = bool(item.get("frame_verified"))
                item["needs_review"] = True
                item["fallback_to_nearest_frame"] = False
                item["camera_verification_needs_review"] = True
                item["verified_candidate_time_ms"] = (
                    int(resolved["time_ms"])
                    if resolved.get("camera_boundary_verified") else None
                )
                item["verified_candidate_time"] = (
                    clock_text(int(resolved["time_ms"]))
                    if resolved.get("camera_boundary_verified") else ""
                )
                item["verified_candidate_exact"] = (
                    str(resolved_exact) if resolved_exact is not None else None
                )
                item["camera_verification_reason"] = (
                    "Frame pilihan Gemini belum sama dengan boundary kamera yang "
                    "terverifikasi. MiniCut tidak menggeser keputusan Gemini secara "
                    "diam-diam; titik harus direview atau dianalisis ulang."
                )
            validated.append(item)

        return validated

    def _film_cut_done(self, results: list):
        validated = self._verify_film_cut_results([dict(item) for item in results])
        # FinalMiniCutWindow -> VerifiedMiniCutWindow normally snaps Gemini's
        # selected frame to a nearby locally detected boundary. We deliberately
        # bypass that override: equality is required instead.
        return BaseMiniCutWindow._film_cut_done(self, validated)

    def _camera_export_error(self) -> str | None:
        # A missing mandatory target should block *any* export mode; otherwise a
        # user could switch to Fast Copy and unknowingly export a timeline from
        # which the requested unresolved boundary simply disappeared.
        if self._pending_camera_reviews:
            return (
                f"Ada {len(self._pending_camera_reviews)} target potong berstatus REVIEW. "
                "Boundary kamera belum terbukti, jadi ekspor ditahan agar target yang "
                "belum selesai tidak diam-diam hilang dari hasil."
            )
        return super()._camera_export_error()


MiniCutWindow = RuntimeMiniCutWindow
