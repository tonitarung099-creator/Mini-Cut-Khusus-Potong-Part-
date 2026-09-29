from __future__ import annotations

# VerifiedMiniCutWindow defines all manual/chat/AI/save-load/export behavior.
# Swap its resolver dependency to the stricter persistence validator before the
# runtime class is instantiated, so every camera-required path shares the same
# A->B->A flash protection without duplicating the window implementation.
from . import verified_window as _verified_module
from .camera_boundary_strict import resolve_camera_boundary as _strict_resolver

_verified_module.resolve_camera_boundary = _strict_resolver

from .verified_window import VerifiedMiniCutWindow  # noqa: E402


class FinalMiniCutWindow(VerifiedMiniCutWindow):
    def _camera_aware_manifest(self):
        manifest = super()._camera_aware_manifest()
        for spec in manifest.get("tools", []):
            if spec.get("name") == "add_cut":
                spec["description"] = (
                    "Tambah batas part dengan time_ms sebagai target. MiniCut "
                    "mencari pergantian kamera, memverifikasi persistensi visual, "
                    "dan mengunci exact rational PTS frame pertama shot baru. "
                    "Jika boundary tidak terbukti, target ditandai REVIEW dan "
                    "tidak diganti otomatis dengan frame terdekat."
                )
        return manifest

    def _append_gemini_chat(self, role: str, text: str):
        # The older camera-aware summary called every no-camera result a
        # "nearest-frame fallback". Runtime v0.3 never does that for mandatory
        # camera cuts; make the visible chat status truthful.
        if role == "system" and self._pending_camera_reviews:
            text = str(text).replace(
                "frame terdekat (fallback)",
                "REVIEW · boundary kamera belum terbukti",
            )
        return super()._append_gemini_chat(role, text)

    def _send_gemini_chat(self):
        before = len(self._pending_camera_reviews)
        result = super()._send_gemini_chat()
        after = len(self._pending_camera_reviews)
        if after > before and hasattr(self, "gemini_chat_status"):
            self.gemini_chat_status.setText(
                f"{after} target masih REVIEW · belum dipasang ke timeline."
            )
        return result


MiniCutWindow = FinalMiniCutWindow
