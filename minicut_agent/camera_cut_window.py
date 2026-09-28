from __future__ import annotations

from dataclasses import asdict

from . import workers as workers_module
from .camera_boundary import resolve_camera_boundary
from .core import clock_text, find_tool, parse_time_ms
from .frame_resolver import probe_keyframes_relative
from .manual_commands import extract_manual_timestamps, looks_like_manual_cut
from .ui import MiniCutWindow as BaseMiniCutWindow


class CameraAwareMiniCutWindow(BaseMiniCutWindow):
    """MiniCut UI with camera-boundary-aware manual/chat cuts.

    AI Film Cut keeps its existing Gemini exact-frame path. Only manual cuts,
    playhead cuts, and chat actions that resolve to ``add_cut`` use this layer.
    """

    def __init__(self):
        # AnalyzeWorker historically imported core.probe_keyframes directly.
        # The active camera-aware app uses the relative-clock resolver so
        # TS/MTS/remuxed media with non-zero container start_time do not shift
        # Fast Copy/keyframe snapping away from the UI timeline.
        workers_module.probe_keyframes = probe_keyframes_relative
        super().__init__()
        # The base registry is created before this subclass gets control. Keep
        # the same registry/host, but make its manifest truthful for Gemini and
        # external bridge clients now that add_cut treats time_ms as a target.
        self._base_registry_manifest = self.registry.manifest
        self.registry.manifest = self._camera_aware_manifest
        self.bridge.manifest_provider = self.registry.manifest

    def _camera_aware_manifest(self):
        manifest = self._base_registry_manifest()
        for spec in manifest.get("tools", []):
            if spec.get("name") == "add_cut":
                spec["description"] = (
                    "Tambah batas part dengan time_ms sebagai target waktu. "
                    "MiniCut mencari pergantian kamera terdekat ±2 detik, "
                    "memperluas sampai ±4 detik bila perlu, lalu mengunci "
                    "ke PTS frame master pertama pada shot baru untuk SmartCut."
                )
        return manifest

    def _snapshot(self) -> dict:
        snapshot = super()._snapshot()
        snapshot["export_mode"] = (
            self.export_mode.currentData()
            if hasattr(self, "export_mode")
            else None
        )
        return snapshot

    def _restore_snapshot(self, snapshot: dict) -> None:
        super()._restore_snapshot(snapshot)
        mode = str(snapshot.get("export_mode") or "").strip().lower()
        if mode and hasattr(self, "export_mode"):
            index = self.export_mode.findData(mode)
            if index >= 0 and self.export_mode.currentIndex() != index:
                self.export_mode.setCurrentIndex(index)
                self._refresh()

    def _camera_cut_payload(
        self,
        *,
        cut,
        resolved: dict,
        requested_ms: int,
        final_ms: int,
        exact_time: str,
        duplicate_boundary: bool = False,
        exact_time_enriched: bool = False,
    ) -> dict:
        return {
            "ok": True,
            "cut": asdict(cut),
            "parts": len(self.model.cuts) + 1,
            # Compatibility field: now false when the requested target moves
            # to the real camera boundary. requested_ms itself is preserved.
            "exact_timestamp_preserved": final_ms == requested_ms,
            "requested_timestamp_preserved": True,
            "camera_boundary_resolved": True,
            "camera_change_found": bool(resolved["camera_change_found"]),
            "fallback_to_nearest_frame": bool(
                resolved["fallback_to_nearest_frame"]
            ),
            "duplicate_boundary": bool(duplicate_boundary),
            "skipped_duplicate": bool(duplicate_boundary),
            "exact_time_enriched": bool(exact_time_enriched),
            "requested_ms": requested_ms,
            "requested_time": clock_text(requested_ms),
            "final_ms": final_ms,
            "final_time": clock_text(final_ms),
            "shift_ms": int(resolved["shift_ms"]),
            "exact_time": exact_time,
            "raw_scene_boundary_ms": resolved.get("raw_scene_boundary_ms"),
            "raw_scene_boundary": str(resolved.get("raw_scene_boundary") or ""),
            "frame_side": str(resolved.get("frame_side") or ""),
            "search_radius_ms": int(resolved["search_radius_ms"]),
            "scene_threshold": float(resolved["scene_threshold"]),
            "reason": (
                "Boundary kamera tersebut sudah ada di timeline; titik duplikat dilewati."
                if duplicate_boundary
                else str(resolved["reason"])
            ),
            "frame_authority": (
                "local-camera-boundary"
                if resolved["camera_change_found"]
                else "local-nearest-master-frame"
            ),
            "export_mode": "smartcut",
        }

    def tool_start_film_cut(self):
        """Start AI Film Cut without depending on QThread scheduler timing.

        QThread.start() schedules the worker asynchronously. Immediately asking
        isRunning() can briefly return False on a busy machine even though the
        worker launch was accepted. The base tool treated that transient state
        as a failed action, which could make an agent/bridge caller report an
        error while analysis started moments later.
        """
        self._require_tool_media_ready()
        worker = getattr(self, "film_cut_worker", None)
        if worker is not None:
            if worker.isRunning():
                return {
                    "ok": True,
                    "started": False,
                    "already_running": True,
                    "stop_plan": True,
                }
            return {
                "ok": False,
                "started": False,
                "error": "AI Film Cut sebelumnya sedang memfinalisasi hasil.",
                "stop_plan": True,
            }

        self._start_film_cut()
        worker = getattr(self, "film_cut_worker", None)
        if worker is None:
            return {
                "ok": False,
                "started": False,
                "error": "AI Film Cut belum dapat dimulai.",
            }

        return {
            "ok": True,
            "started": True,
            "running": bool(worker.isRunning()),
            "stop_plan": True,
        }

    def tool_cancel_film_cut(self):
        """Cancel even during the short QThread start/finalization race window."""
        worker = getattr(self, "film_cut_worker", None)
        requested = worker is not None
        if requested:
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()
            if hasattr(self, "film_cancel_btn"):
                self.film_cancel_btn.setEnabled(False)
            if hasattr(self, "film_status_label"):
                self.film_status_label.setText(
                    "Membatalkan setelah langkah aktif selesai…"
                )
        return {"ok": True, "cancel_requested": requested}

    def tool_cancel_export(self):
        """Cancel export even before QThread.isRunning() flips to True."""
        worker = getattr(self, "export_worker", None)
        requested = worker is not None
        if requested:
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()
            if hasattr(self, "status"):
                self.status.setText(
                    "Membatalkan ekspor setelah proses aktif selesai…"
                )
        return {"ok": True, "cancel_requested": requested}

    def tool_add_cut(self, time_ms):
        self._require_tool_media_ready()
        if not self.model.source:
            raise ValueError("Belum ada video.")
        if getattr(self, "film_cut_worker", None) is not None:
            raise RuntimeError(
                "Tunggu AI Film Cut selesai dan hasilnya difinalisasi sebelum menambah cut manual."
            )
        if getattr(self, "export_worker", None) is not None:
            raise RuntimeError(
                "Tunggu ekspor selesai dan hasilnya difinalisasi sebelum menambah cut manual."
            )

        requested_ms = self.model.clamp(parse_time_ms(time_ms))
        if requested_ms <= 0 or requested_ms >= self.model.duration_ms:
            raise ValueError("Cut harus berada di antara awal dan akhir video.")

        ffmpeg = find_tool("ffmpeg")
        ffprobe = find_tool("ffprobe")
        if not ffmpeg or not ffprobe:
            raise RuntimeError(
                "FFmpeg/ffprobe tidak ditemukan. Paket MiniCut lengkap diperlukan "
                "untuk mencari pergantian kamera secara frame-accurate."
            )

        resolved = resolve_camera_boundary(
            self.model.source,
            ffmpeg,
            ffprobe,
            requested_ms,
            duration_ms=self.model.duration_ms,
        )
        final_ms = int(resolved["time_ms"])
        exact_time = str(resolved["exact_time"])

        # Two nearby targets can legitimately resolve to the same camera
        # boundary. Treating the second one as an exception used to make
        # AgentPlanner roll the whole multi-cut transaction back, so a list of
        # six otherwise-valid points could result in zero cuts. A repeated
        # boundary is idempotent: keep the existing cut and continue the batch.
        existing = next(
            (
                item
                for item in self.model.cuts
                if abs(int(item.actual_ms) - final_ms) < 2
            ),
            None,
        )
        exact_time_enriched = False
        if existing is not None:
            if not getattr(existing, "exact_time", None):
                existing.exact_time = exact_time
                self.model.dirty = True
                exact_time_enriched = True
            smart_index = self.export_mode.findData("smartcut")
            if smart_index >= 0:
                self.export_mode.setCurrentIndex(smart_index)
            self._refresh()
            return self._camera_cut_payload(
                cut=existing,
                resolved=resolved,
                requested_ms=requested_ms,
                final_ms=final_ms,
                exact_time=exact_time,
                duplicate_boundary=True,
                exact_time_enriched=exact_time_enriched,
            )

        cut = self.model.add_frame_cut(
            requested_ms,
            final_ms,
            exact_time=exact_time,
        )

        smart_index = self.export_mode.findData("smartcut")
        if smart_index >= 0:
            self.export_mode.setCurrentIndex(smart_index)

        self._refresh()
        return self._camera_cut_payload(
            cut=cut,
            resolved=resolved,
            requested_ms=requested_ms,
            final_ms=final_ms,
            exact_time=exact_time,
        )

    def _manual_cut_workers_busy(self) -> bool:
        # film/export workers remain non-None during their final callback window;
        # treat that state as busy too so a late mutation cannot race finalization.
        if getattr(self, "film_cut_worker", None) is not None:
            return True
        if getattr(self, "export_worker", None) is not None:
            return True

        workers = (
            getattr(self, "gemini_batch_worker", None),
            getattr(self, "gemini_test_worker", None),
            getattr(self, "gemini_chat_worker", None),
        )
        return any(
            worker is not None
            and callable(getattr(worker, "isRunning", None))
            and worker.isRunning()
            for worker in workers
        )

    def _trim_model_history_from(self, size: int) -> None:
        """Keep local UI commands visible without replaying them to Gemini."""
        if len(self._gemini_chat_history_data) > int(size):
            del self._gemini_chat_history_data[int(size):]

    def _append_local_user_message(self, text: str) -> None:
        """Show a locally handled command without adding it to model history."""
        history_size = len(self._gemini_chat_history_data)
        self._append_gemini_chat("user", text)
        self._trim_model_history_from(history_size)
        self.gemini_chat_input.clear()

    def _send_gemini_chat(self):
        """Handle explicit cuts/cancellation locally, leaving other chat to Gemini."""
        text = self.gemini_chat_input.toPlainText().strip()
        if not text:
            return

        low = " ".join(text.lower().split())
        cancel_words = ("batalkan", "batal", "cancel", "stop", "hentikan")
        wants_cancel = any(word in low for word in cancel_words)
        film_worker = getattr(self, "film_cut_worker", None)
        export_worker = getattr(self, "export_worker", None)

        # A worker object exists from scheduling until its final callback. Do
        # not rely on isRunning(): immediately after QThread.start() it can be
        # False even though the job is about to run.
        if film_worker is not None:
            if wants_cancel and any(
                hint in low
                for hint in ("film cut", "ai film", "analisis film", "analisa film")
            ):
                self._append_local_user_message(text)
                result = self.tool_cancel_film_cut()
                self._append_gemini_chat(
                    "system",
                    "Permintaan pembatalan AI Film Cut dikirim."
                    if result.get("cancel_requested")
                    else "AI Film Cut tidak sedang berjalan.",
                )
                return
            from PySide6.QtWidgets import QMessageBox
            from . import APP_TITLE
            QMessageBox.information(
                self,
                APP_TITLE,
                "AI Film Cut sedang berjalan atau memfinalisasi hasil. "
                "Kamu tetap bisa mengetik 'batalkan AI Film Cut'.",
            )
            return

        if export_worker is not None:
            if wants_cancel and any(
                hint in low for hint in ("ekspor", "export", "render")
            ):
                self._append_local_user_message(text)
                result = self.tool_cancel_export()
                self._append_gemini_chat(
                    "system",
                    "Permintaan pembatalan ekspor dikirim."
                    if result.get("cancel_requested")
                    else "Ekspor tidak sedang berjalan.",
                )
                return
            from PySide6.QtWidgets import QMessageBox
            from . import APP_TITLE
            QMessageBox.information(
                self,
                APP_TITLE,
                "Ekspor sedang berjalan atau memfinalisasi hasil. "
                "Kamu tetap bisa mengetik 'batalkan ekspor'.",
            )
            return

        timestamps = extract_manual_timestamps(text)
        local_manual_cut = looks_like_manual_cut(text) and bool(timestamps)
        busy = self._manual_cut_workers_busy()
        if not local_manual_cut or busy:
            # Base chat appends failed setup attempts to the same hidden history
            # used for future Gemini calls. If no Gemini worker is launched, any
            # new history entry must not be replayed to the model on retry.
            history_size = len(self._gemini_chat_history_data)
            result = super()._send_gemini_chat()
            if (
                len(self._gemini_chat_history_data) > history_size
                and getattr(self, "gemini_chat_worker", None) is None
            ):
                self._trim_model_history_from(history_size)
            return result

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
        for index, item in enumerate(results, 1):
            result = dict(item.get("result") or {})
            if not result.get("ok"):
                continue
            if result.get("duplicate_boundary"):
                mode = "boundary sama sudah ada · dilewati"
            else:
                mode = (
                    "ganti kamera · frame pertama shot baru"
                    if result.get("camera_change_found")
                    else "frame terdekat (fallback)"
                )
            lines.append(
                f"Titik {index}: {result.get('requested_time')} → "
                f"{result.get('final_time')} · {mode} · "
                f"geser {int(result.get('shift_ms') or 0):+d} ms"
            )

        if lines:
            unique_count = sum(
                1
                for item in results
                if not (item.get("result") or {}).get("duplicate_boundary")
            )
            duplicate_count = len(lines) - unique_count
            summary = f"{unique_count} boundary unik dipasang"
            if duplicate_count:
                summary += f" · {duplicate_count} duplikat dilewati"
            self._append_gemini_chat(
                "system",
                "Titik potong diproses lokal tanpa memakai API Gemini:\n"
                + "\n".join(lines),
            )
            if hasattr(self, "gemini_chat_status"):
                self.gemini_chat_status.setText(summary + ".")

    def _gemini_chat_ready(self, result: dict):
        # GeminiClient versi dasar masih mendeskripsikan manual_frame_cut sebagai
        # timestamp exact. Pada window aktif, nilai itu adalah TARGET. Jangan
        # meneruskan balasan yang bisa mengklaim timestamp akan dipakai mentah.
        actions = result.get("actions") or []
        camera_cut_action = any(
            isinstance(item, dict)
            and str(item.get("tool") or "").strip()
            in {"manual_frame_cut", "frame_cut", "add_cut"}
            for item in actions
        )
        if camera_cut_action:
            result = dict(result)
            extra = ""
            if len(actions) > 1:
                extra = " Aksi lain pada perintah yang sama juga akan diproses sesuai urutan."
            result["reply"] = (
                "MiniCut akan memakai timestamp yang diminta sebagai target, "
                "mencari pergantian kamera terdekat, lalu mengunci potongan ke "
                "frame master pertama pada shot baru."
                + extra
            )
        return super()._gemini_chat_ready(result)

    def _gemini_chat_failed(self, message: str):
        """Do not carry a terminally failed command into the next model turn."""
        super()._gemini_chat_failed(message)
        if (
            getattr(self, "gemini_chat_worker", None) is None
            and getattr(self, "_gemini_chat_retry_payload", None) is None
            and self._gemini_chat_history_data
            and self._gemini_chat_history_data[-1].get("role") == "user"
        ):
            self._gemini_chat_history_data.pop()


# Short alias so callers can import this module as the active MiniCut window.
MiniCutWindow = CameraAwareMiniCutWindow
