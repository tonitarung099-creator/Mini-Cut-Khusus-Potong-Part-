from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any

from PySide6.QtWidgets import QMessageBox

from . import APP_TITLE
from .camera_boundary import resolve_camera_boundary
from .core import CutPoint, clock_text, find_tool, parse_time_ms
from .hardened_window import HardenedMiniCutWindow


_CUT_META_FIELDS = (
    "provenance",
    "source_pts",
    "source_time_base",
    "raw_scene_boundary_exact",
    "pts_verified",
    "camera_boundary_verified",
    "needs_review",
    "fallback_to_nearest_frame",
    "requires_camera_boundary",
    "verification_reason",
    "verification_status",
)


def cut_verification_metadata(cut: CutPoint) -> dict[str, Any]:
    return {
        field: copy.deepcopy(getattr(cut, field))
        for field in _CUT_META_FIELDS
        if hasattr(cut, field)
    }


def apply_cut_verification_metadata(cut: CutPoint, data: dict[str, Any]) -> None:
    for field in _CUT_META_FIELDS:
        if field in data:
            setattr(cut, field, copy.deepcopy(data[field]))


def _legacy_verification_metadata(cut: CutPoint) -> dict[str, Any]:
    """Old projects never proved camera-boundary identity after save/load."""
    requires_camera = bool(getattr(cut, "exact_time", None))
    return {
        "provenance": "legacy-project",
        "pts_verified": False,
        "camera_boundary_verified": False,
        "needs_review": requires_camera,
        "fallback_to_nearest_frame": False,
        "requires_camera_boundary": requires_camera,
        "verification_reason": (
            "Proyek dibuat sebelum metadata verifikasi boundary disimpan; "
            "titik SmartCut ini perlu diverifikasi ulang."
            if requires_camera
            else "Cut keyframe lama tidak mewajibkan boundary kamera."
        ),
        "verification_status": (
            "legacy-unverified" if requires_camera else "not-required"
        ),
    }


class VerifiedMiniCutWindow(HardenedMiniCutWindow):
    """Runtime layer that makes camera-boundary verification durable and strict."""

    def __init__(self):
        self._pending_camera_reviews: list[dict[str, Any]] = []
        super().__init__()
        self._install_project_serializer()

    # ---------- project persistence ----------
    def _install_project_serializer(self) -> None:
        original = self.model.to_project_dict

        def verified_to_project_dict(
            model_self,
            project_path: Path | None = None,
            subtitle_path: Path | None = None,
            subtitle_auto_disabled: bool = False,
        ) -> dict[str, Any]:
            payload = original(
                project_path,
                subtitle_path=subtitle_path,
                subtitle_auto_disabled=subtitle_auto_disabled,
            )
            payload["version"] = max(3, int(payload.get("version") or 0))
            for item, cut in zip(payload.get("cuts", []), model_self.cuts):
                item.update(cut_verification_metadata(cut))
            payload["pending_camera_reviews"] = copy.deepcopy(
                self._pending_camera_reviews
            )
            payload["camera_boundary_schema"] = 1
            return payload

        self.model.to_project_dict = MethodType(
            verified_to_project_dict,
            self.model,
        )

    def _analysis_ready(self, metadata: dict, keyframes: list):
        project_data = None
        if self.pending_load is not None:
            project_data = self.pending_load[1]

        stored_meta: dict[tuple[int, int, str | None], dict[str, Any]] = {}
        pending: list[dict[str, Any]] = []
        if isinstance(project_data, dict):
            for item in project_data.get("cuts", []) or []:
                if not isinstance(item, dict):
                    continue
                try:
                    requested = int(item.get("requested_ms", item.get("actual_ms", 0)))
                    actual = int(item.get("actual_ms", requested))
                except (TypeError, ValueError):
                    continue
                exact_raw = item.get("exact_time")
                exact = (
                    str(exact_raw).strip()
                    if exact_raw not in (None, "")
                    else None
                )
                stored_meta[(requested, actual, exact)] = {
                    key: copy.deepcopy(item[key])
                    for key in _CUT_META_FIELDS
                    if key in item
                }
            raw_pending = project_data.get("pending_camera_reviews") or []
            if isinstance(raw_pending, list):
                pending = [
                    copy.deepcopy(item)
                    for item in raw_pending
                    if isinstance(item, dict)
                ]

        super()._analysis_ready(metadata, keyframes)

        for cut in self.model.cuts:
            key = (
                int(cut.requested_ms),
                int(cut.actual_ms),
                str(cut.exact_time).strip() if cut.exact_time else None,
            )
            meta = stored_meta.get(key)
            if meta:
                apply_cut_verification_metadata(cut, meta)
            else:
                apply_cut_verification_metadata(
                    cut,
                    _legacy_verification_metadata(cut),
                )
        self._pending_camera_reviews = pending
        self.bridge_state = self._state_with_subtitle()
        self._refresh()

    def _snapshot(self) -> dict:
        snapshot = super()._snapshot()
        snapshot["camera_cut_metadata"] = [
            cut_verification_metadata(cut) for cut in self.model.cuts
        ]
        snapshot["pending_camera_reviews"] = copy.deepcopy(
            self._pending_camera_reviews
        )
        return snapshot

    def _restore_snapshot(self, snapshot: dict) -> None:
        metadata = list(snapshot.get("camera_cut_metadata") or [])
        pending = copy.deepcopy(snapshot.get("pending_camera_reviews") or [])
        super()._restore_snapshot(snapshot)
        for index, cut in enumerate(self.model.cuts):
            if index < len(metadata) and isinstance(metadata[index], dict):
                apply_cut_verification_metadata(cut, metadata[index])
            elif cut.exact_time:
                apply_cut_verification_metadata(
                    cut,
                    _legacy_verification_metadata(cut),
                )
        self._pending_camera_reviews = [
            item for item in pending if isinstance(item, dict)
        ]

    def _state_with_subtitle(self) -> dict:
        state = super()._state_with_subtitle()
        cuts = state.get("cuts") or []
        for item, cut in zip(cuts, self.model.cuts):
            item.update(cut_verification_metadata(cut))
        state["pending_camera_reviews"] = copy.deepcopy(
            self._pending_camera_reviews
        )
        state["camera_review_count"] = len(self._pending_camera_reviews)
        return state

    # ---------- verification helpers ----------
    @staticmethod
    def _verified_metadata(resolved: dict[str, Any], provenance: str) -> dict[str, Any]:
        return {
            "provenance": str(provenance),
            "source_pts": resolved.get("pts"),
            "source_time_base": resolved.get("time_base"),
            "raw_scene_boundary_exact": resolved.get("raw_scene_boundary_exact"),
            "pts_verified": bool(resolved.get("pts_verified")),
            "camera_boundary_verified": bool(
                resolved.get("camera_boundary_verified")
            ),
            "needs_review": bool(resolved.get("needs_review")),
            "fallback_to_nearest_frame": bool(
                resolved.get("fallback_to_nearest_frame")
            ),
            "requires_camera_boundary": True,
            "verification_reason": str(resolved.get("reason") or ""),
            "verification_status": (
                "verified"
                if resolved.get("camera_boundary_verified")
                else "needs-review"
            ),
        }

    def _remember_pending_review(
        self,
        requested_ms: int,
        resolved: dict[str, Any],
        *,
        provenance: str,
    ) -> None:
        record = {
            "requested_ms": int(requested_ms),
            "requested_time": clock_text(int(requested_ms)),
            "provenance": str(provenance),
            "needs_review": True,
            "camera_boundary_verified": False,
            "fallback_to_nearest_frame": False,
            "raw_scene_boundary_ms": resolved.get("raw_scene_boundary_ms"),
            "raw_scene_boundary_exact": resolved.get("raw_scene_boundary_exact"),
            "reason": str(resolved.get("reason") or "Boundary belum terverifikasi."),
        }
        self._pending_camera_reviews = [
            item for item in self._pending_camera_reviews
            if int(item.get("requested_ms") or -1) != int(requested_ms)
        ]
        self._pending_camera_reviews.append(record)
        self._pending_camera_reviews.sort(
            key=lambda item: int(item.get("requested_ms") or 0)
        )
        self.model.dirty = True

    def _clear_pending_near(self, requested_ms: int) -> None:
        self._pending_camera_reviews = [
            item for item in self._pending_camera_reviews
            if abs(int(item.get("requested_ms") or 0) - int(requested_ms)) > 2
        ]

    # ---------- manual/chat/list cuts ----------
    def tool_add_cut(self, time_ms):
        self._ensure_timeline_mutation_idle()
        self._require_tool_media_ready()
        if not self.model.source:
            raise ValueError("Belum ada video.")

        requested_ms = self.model.clamp(parse_time_ms(time_ms))
        if requested_ms <= 0 or requested_ms >= self.model.duration_ms:
            raise ValueError("Cut harus berada di antara awal dan akhir video.")

        ffmpeg = find_tool("ffmpeg")
        ffprobe = find_tool("ffprobe")
        if not ffmpeg or not ffprobe:
            raise RuntimeError(
                "FFmpeg/ffprobe tidak ditemukan. Paket MiniCut lengkap diperlukan "
                "untuk memverifikasi boundary kamera."
            )

        resolved = resolve_camera_boundary(
            self.model.source,
            ffmpeg,
            ffprobe,
            requested_ms,
            duration_ms=self.model.duration_ms,
            require_camera_change=True,
        )
        if not bool(resolved.get("camera_boundary_verified")):
            self._remember_pending_review(
                requested_ms,
                resolved,
                provenance="manual-camera-target",
            )
            self._refresh()
            return {
                "ok": True,
                "applied": False,
                "camera_boundary_resolved": False,
                "camera_change_found": bool(resolved.get("camera_change_found")),
                "camera_boundary_verified": False,
                "pts_verified": False,
                "needs_review": True,
                "fallback_to_nearest_frame": False,
                "requested_ms": requested_ms,
                "requested_time": clock_text(requested_ms),
                "final_ms": None,
                "final_time": "",
                "shift_ms": 0,
                "reason": str(resolved.get("reason") or "Boundary perlu review."),
                "parts": len(self.model.cuts) + 1,
                "frame_authority": "review-required",
                "export_mode": str(self.export_mode.currentData() or "smartcut"),
            }

        final_ms = int(resolved["time_ms"])
        exact_time = str(resolved["exact_time"])
        existing = next(
            (
                item for item in self.model.cuts
                if (
                    str(getattr(item, "exact_time", "") or "") == exact_time
                    or abs(int(item.actual_ms) - final_ms) < 2
                )
            ),
            None,
        )
        duplicate = existing is not None
        if existing is None:
            cut = self.model.add_frame_cut(
                requested_ms,
                final_ms,
                exact_time=exact_time,
            )
        else:
            cut = existing
            if not cut.exact_time:
                cut.exact_time = exact_time
                self.model.dirty = True

        apply_cut_verification_metadata(
            cut,
            self._verified_metadata(resolved, "local-camera-boundary"),
        )
        self._clear_pending_near(requested_ms)
        self.model.dirty = True

        smart_index = self.export_mode.findData("smartcut")
        if smart_index >= 0:
            self.export_mode.setCurrentIndex(smart_index)
        self._refresh()

        payload = self._camera_cut_payload(
            cut=cut,
            resolved=resolved,
            requested_ms=requested_ms,
            final_ms=final_ms,
            exact_time=exact_time,
            duplicate_boundary=duplicate,
            exact_time_enriched=bool(duplicate and cut.exact_time == exact_time),
        )
        payload.update({
            "applied": not duplicate,
            "pts_verified": True,
            "camera_boundary_verified": True,
            "needs_review": False,
            "fallback_to_nearest_frame": False,
            "source_pts": resolved.get("pts"),
            "source_time_base": resolved.get("time_base"),
            "raw_scene_boundary_exact": resolved.get("raw_scene_boundary_exact"),
        })
        return payload

    def tool_clear_cuts(self):
        result = super().tool_clear_cuts()
        self._pending_camera_reviews = []
        return result

    def tool_divide_equal(self, parts):
        result = super().tool_divide_equal(parts)
        self._pending_camera_reviews = []
        return result

    def tool_divide_interval(self, interval_ms):
        result = super().tool_divide_interval(interval_ms)
        self._pending_camera_reviews = []
        return result

    # ---------- AI Film Cut ----------
    def _film_cut_done(self, results: list):
        ffmpeg = find_tool("ffmpeg")
        ffprobe = find_tool("ffprobe")
        validated: list[dict[str, Any]] = []

        for raw in results:
            item = dict(raw)
            selected_ms = int(item.get("selected_time_ms") or 0)
            if selected_ms <= 0:
                validated.append(item)
                continue

            item["gemini_selected_time_ms"] = selected_ms
            item["gemini_selected_time"] = str(item.get("selected_time") or "")
            item["gemini_selected_time_exact"] = item.get("selected_time_exact")

            if not ffmpeg or not ffprobe or not self.model.source:
                item["camera_boundary_verified"] = False
                item["pts_verified"] = bool(item.get("frame_verified"))
                item["needs_review"] = True
                item["fallback_to_nearest_frame"] = False
                item["camera_verification_reason"] = (
                    "FFmpeg/ffprobe tidak tersedia untuk verifikasi boundary kamera."
                )
                validated.append(item)
                continue

            resolved = resolve_camera_boundary(
                self.model.source,
                ffmpeg,
                ffprobe,
                selected_ms,
                duration_ms=self.model.duration_ms,
                search_radii_ms=(600, 1_500, 2_500),
                require_camera_change=True,
            )
            if bool(resolved.get("camera_boundary_verified")):
                item["selected_time_ms"] = int(resolved["time_ms"])
                item["selected_time"] = clock_text(int(resolved["time_ms"]))
                item["selected_time_exact"] = str(resolved["exact_time"])
                item["selected_source_pts"] = resolved.get("pts")
                item["selected_time_base"] = resolved.get("time_base")
                item["camera_boundary_verified"] = True
                item["pts_verified"] = True
                item["fallback_to_nearest_frame"] = False
                item["camera_verification_needs_review"] = False
                item["camera_verification_reason"] = str(
                    resolved.get("reason") or ""
                )
                item["raw_scene_boundary_exact"] = resolved.get(
                    "raw_scene_boundary_exact"
                )
                item["camera_shift_from_gemini_ms"] = (
                    int(resolved["time_ms"]) - selected_ms
                )
            else:
                item["camera_boundary_verified"] = False
                item["pts_verified"] = bool(item.get("frame_verified"))
                item["needs_review"] = True
                item["fallback_to_nearest_frame"] = False
                item["camera_verification_needs_review"] = True
                item["camera_verification_reason"] = str(
                    resolved.get("reason") or "Boundary kamera belum terverifikasi."
                )
            validated.append(item)

        return super()._film_cut_done(validated)

    def _apply_film_cut(self, *, record_undo: bool = True) -> dict:
        unresolved = [
            item for item in self.film_cut_results
            if int(item.get("selected_time_ms") or 0) > 0
            and not bool(item.get("camera_boundary_verified"))
        ]
        if unresolved:
            QMessageBox.warning(
                self,
                APP_TITLE,
                f"Ada {len(unresolved)} titik AI yang belum terverifikasi sebagai "
                "pergantian kamera. Titik tersebut tidak akan diterapkan. "
                "Review atau analisis ulang terlebih dahulu.",
            )
            return {
                "ok": False,
                "needs_review": True,
                "unverified_camera_boundaries": len(unresolved),
                "error": "Boundary kamera AI belum semuanya terverifikasi.",
            }

        result = dict(super()._apply_film_cut(record_undo=record_undo) or {})
        if not result.get("ok"):
            return result

        by_ms = {
            int(item.get("selected_time_ms") or 0): item
            for item in self.film_cut_results
            if int(item.get("selected_time_ms") or 0) > 0
        }
        for cut in self.model.cuts:
            item = by_ms.get(int(cut.actual_ms))
            if not item:
                continue
            apply_cut_verification_metadata(cut, {
                "provenance": "gemini-context+local-camera-verification",
                "source_pts": item.get("selected_source_pts"),
                "source_time_base": item.get("selected_time_base"),
                "raw_scene_boundary_exact": item.get("raw_scene_boundary_exact"),
                "pts_verified": True,
                "camera_boundary_verified": True,
                "needs_review": False,
                "fallback_to_nearest_frame": False,
                "requires_camera_boundary": True,
                "verification_reason": str(
                    item.get("camera_verification_reason") or ""
                ),
                "verification_status": "verified",
            })
        self._pending_camera_reviews = []
        self.model.dirty = True
        self._refresh()
        return result

    # ---------- export guard ----------
    def _camera_export_error(self) -> str | None:
        mode = str(self.export_mode.currentData() or "smartcut").strip().lower()
        if mode != "smartcut":
            return None
        if self._pending_camera_reviews:
            return (
                f"Ada {len(self._pending_camera_reviews)} target potong yang masih "
                "berstatus REVIEW karena boundary kamera belum terbukti. SmartCut "
                "tidak dimulai agar hasil tidak diklaim frame-boundary akurat."
            )
        unresolved = [
            cut for cut in self.model.cuts
            if bool(getattr(cut, "requires_camera_boundary", False))
            and not bool(getattr(cut, "camera_boundary_verified", False))
        ]
        if unresolved:
            return (
                f"Ada {len(unresolved)} cut SmartCut yang status boundary kameranya "
                "belum terverifikasi (termasuk proyek lama). Verifikasi ulang titik "
                "tersebut sebelum ekspor."
            )
        return None

    def tool_export_all(self):
        error = self._camera_export_error()
        if error:
            raise RuntimeError(error)
        return super().tool_export_all()


# Public runtime alias.
MiniCutWindow = VerifiedMiniCutWindow
