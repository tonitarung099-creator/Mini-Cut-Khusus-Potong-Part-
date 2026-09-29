from __future__ import annotations

from fractions import Fraction
from typing import Any

from . import workers as workers_module
from .camera_boundary import resolve_camera_boundary
from .core import clock_text, find_tool
from .gemini_boundary import BoundaryAwareGeminiClient
from .manual_commands import extract_manual_timestamps, looks_like_manual_cut
from .ui import MiniCutWindow as BaseMiniCutWindow
from .verified_window import VerifiedMiniCutWindow


class RuntimeMiniCutWindow(VerifiedMiniCutWindow):
    """Final MiniCut runtime with one camera-boundary contract on every path."""

    def __init__(self):
        # FilmCutWorker resolves GeminiClient from this module-level binding at
        # run time. Replace it before the UI can start any worker so high-fps
        # exact-frame refinement uses duration-aware windows.
        workers_module.GeminiClient = BoundaryAwareGeminiClient
        super().__init__()

    # ---------- pasted timestamp lists / chat ----------
    def _send_gemini_chat(self):
        text = self.gemini_chat_input.toPlainText().strip()
        if not text:
            return

        timestamps = extract_manual_timestamps(text)
        local_manual_cut = looks_like_manual_cut(text) and bool(timestamps)
        if not local_manual_cut or self._manual_cut_workers_busy():
            return super()._send_gemini_chat()

        self._append_local_user_message(text)
        if not self.model.source:
            self._append_gemini_chat(
                "system",
                "Buka video terlebih dahulu sebelum memproses daftar titik potong.",
            )
            return

        steps = [
            {"tool": "add_cut", "args": {"time_ms": item.time_ms}}
            for item in timestamps
        ]
        results = self._apply_gemini_chat_steps(steps)
        if not results:
            return

        lines: list[str] = []
        installed = 0
        duplicates = 0
        reviews = 0
        for index, item in enumerate(results, 1):
            result = dict(item.get("result") or {})
            if not result.get("ok"):
                lines.append(f"Titik {index}: gagal · {result.get('error') or 'error'}")
                continue

            requested = str(result.get("requested_time") or "")
            if bool(result.get("needs_review")):
                reviews += 1
                mode = "REVIEW · boundary kamera belum terbukti · cut tidak dipasang"
                final_text = "—"
            elif bool(result.get("duplicate_boundary")):
                duplicates += 1
                mode = "boundary terverifikasi yang sama sudah ada · dilewati"
                final_text = str(result.get("final_time") or "")
            elif bool(result.get("camera_boundary_verified")):
                installed += 1
                mode = "TERVERIFIKASI · frame pertama shot baru"
                final_text = str(result.get("final_time") or "")
            else:
                reviews += 1
                mode = "REVIEW · status boundary belum terverifikasi"
                final_text = str(result.get("final_time") or "—")

            lines.append(
                f"Titik {index}: {requested} → {final_text} · {mode}"
                + (
                    f" · geser {int(result.get('shift_ms') or 0):+d} ms"
                    if final_text != "—" else ""
                )
            )

        self._append_gemini_chat(
            "system",
            "Titik potong diproses lokal dengan verifikasi boundary kamera:\n"
            + "\n".join(lines),
        )
        if hasattr(self, "gemini_chat_status"):
            self.gemini_chat_status.setText(
                f"{installed} dipasang · {duplicates} duplikat · {reviews} REVIEW."
            )

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
        # Skip CameraAwareMiniCutWindow's older optimistic wording while keeping
        # the base Gemini usage/history/action machinery unchanged.
        return BaseMiniCutWindow._gemini_chat_ready(self, safe)

    # ---------- AI Film Cut ----------
    def _verify_film_cut_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ffmpeg = find_tool("ffmpeg")
        ffprobe = find_tool("ffprobe")
        validated: list[dict[str, Any]] = []

        for raw in results:
            item = dict(raw)
            selected_ms = int(item.get("selected_time_ms") or 0)
            if selected_ms <= 0:
                validated.append(item)
                continue

            selected_exact_raw = item.get("selected_time_exact")
            selected_exact: Fraction | None = None
            try:
                if selected_exact_raw not in (None, ""):
                    selected_exact = Fraction(str(selected_exact_raw))
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
                # This search may discover a nearby candidate for diagnosis, but
                # the selected Gemini PTS is never silently replaced by it.
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
                item["camera_verification_reason"] = (
                    "PTS frame pilihan Gemini sama persis dengan PTS boundary kamera "
                    "yang diverifikasi lokal."
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
        # Bypass VerifiedMiniCutWindow._film_cut_done because that older layer
        # snapped Gemini's choice to a nearby local boundary. We now validate
        # equality instead, preserving Gemini's semantic decision.
        return BaseMiniCutWindow._film_cut_done(self, validated)

    # ---------- export guard ----------
    def _camera_export_error(self) -> str | None:
        if self._pending_camera_reviews:
            return (
                f"Ada {len(self._pending_camera_reviews)} target potong berstatus REVIEW. "
                "Boundary kamera belum terbukti, jadi ekspor ditahan agar target yang "
                "belum selesai tidak diam-diam hilang dari hasil."
            )
        return super()._camera_export_error()


MiniCutWindow = RuntimeMiniCutWindow
