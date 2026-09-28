from __future__ import annotations

import queue
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from . import APP_TITLE
from .agent import MUTATING_TOOLS
from .camera_cut_window import CameraAwareMiniCutWindow as BaseCameraAwareMiniCutWindow


class HardenedMiniCutWindow(BaseCameraAwareMiniCutWindow):
    """Final runtime safety layer for the active MiniCut window.

    QThread.start() is asynchronous, so a worker object can exist briefly while
    ``isRunning()`` still reports False. Treat worker ownership as authoritative
    until its UI callback clears the attribute. This prevents project switching,
    duplicate commands, timeline mutation, or application shutdown from racing
    a worker that is scheduled or finalizing.
    """

    _WORKER_LABELS = (
        ("Analisis video", "analyze_worker"),
        ("AI Agent", "agent_worker"),
        ("Tes Gemini", "gemini_test_worker"),
        ("Cek Semua API", "gemini_batch_worker"),
        ("Gemini Chat", "gemini_chat_worker"),
        ("AI Film Cut", "film_cut_worker"),
        ("Ekspor", "export_worker"),
    )

    def _present_worker_labels(self, names: set[str] | None = None) -> list[str]:
        result: list[str] = []
        for label, attr in self._WORKER_LABELS:
            if names is not None and attr not in names:
                continue
            if getattr(self, attr, None) is not None:
                result.append(label)
        return result

    def _project_switch_blockers(self) -> list[str]:
        # Even workers that do not directly edit the timeline share status/UI
        # callbacks with the current project. Do not let callbacks from project A
        # arrive after project B has already replaced the active state.
        return self._present_worker_labels()

    def _timeline_mutation_blockers(self) -> list[str]:
        return self._present_worker_labels({
            "analyze_worker",
            "agent_worker",
            "gemini_chat_worker",
            "film_cut_worker",
            "export_worker",
        })

    def _gemini_key_mutation_blockers(self) -> list[str]:
        # These workers retain a concrete key ID until their completion callback
        # records usage/status. Editing or deleting that record mid-flight can
        # make the callback fail with KeyError and leave UI state half-finalized.
        return self._present_worker_labels({
            "gemini_test_worker",
            "gemini_batch_worker",
            "gemini_chat_worker",
            "film_cut_worker",
        })

    def _ensure_timeline_mutation_idle(self) -> None:
        blockers = self._timeline_mutation_blockers()
        if blockers:
            raise RuntimeError(
                "Timeline sedang dikunci oleh proses aktif/finalisasi: "
                + " · ".join(blockers)
                + ". Tunggu proses tersebut selesai sebelum mengubah cut."
            )

    def _begin_load(
        self,
        source: Path,
        project_data: dict | None = None,
        project_path: Path | None = None,
    ) -> bool:
        blockers = self._project_switch_blockers()
        if blockers:
            QMessageBox.information(
                self,
                APP_TITLE,
                "Tunggu proses berikut selesai sebelum membuka video/proyek lain:\n"
                + " · ".join(blockers),
            )
            return False
        return super()._begin_load(source, project_data, project_path)

    def _require_tool_project_ready(self) -> None:
        # Base UI only checked AnalyzeWorker.isRunning(). A scheduled worker can
        # still report False for a short time after start(), and a finished worker
        # can remain attached until its queued callback finalizes the load.
        if getattr(self, "analyze_worker", None) is not None:
            raise RuntimeError(
                "Video/proyek baru sedang dianalisis atau memfinalisasi load. "
                "Tunggu sampai proses load selesai."
            )
        return super()._require_tool_project_ready()

    def _manual_cut_workers_busy(self) -> bool:
        # Any owned worker is busy, including the scheduler/finalization windows.
        return bool(self._present_worker_labels())

    def _send_gemini_chat(self):
        text = self.gemini_chat_input.toPlainText().strip()
        if not text:
            return

        # Parent camera window must keep handling local cancellation while Film
        # Cut/export owns the app. Do not block those deterministic commands.
        if (
            getattr(self, "film_cut_worker", None) is not None
            or getattr(self, "export_worker", None) is not None
        ):
            return super()._send_gemini_chat()

        blockers = self._present_worker_labels({
            "analyze_worker",
            "agent_worker",
            "gemini_test_worker",
            "gemini_batch_worker",
            "gemini_chat_worker",
        })
        if blockers:
            QMessageBox.information(
                self,
                APP_TITLE,
                "Tunggu proses berikut selesai sebelum mengirim perintah baru:\n"
                + " · ".join(blockers),
            )
            return
        return super()._send_gemini_chat()

    def _start_film_cut(self):
        blockers = self._present_worker_labels({
            "analyze_worker",
            "agent_worker",
            "gemini_test_worker",
            "gemini_batch_worker",
            "gemini_chat_worker",
            "export_worker",
        })
        if blockers:
            QMessageBox.information(
                self,
                APP_TITLE,
                "AI Film Cut belum dapat dimulai. Tunggu proses berikut selesai:\n"
                + " · ".join(blockers),
            )
            return
        return super()._start_film_cut()

    def _edit_gemini_key(self):
        blockers = self._gemini_key_mutation_blockers()
        if blockers:
            QMessageBox.information(
                self,
                APP_TITLE,
                "API key belum boleh diedit karena masih dipakai proses berikut:\n"
                + " · ".join(blockers),
            )
            return
        return super()._edit_gemini_key()

    def _remove_gemini_key(self):
        blockers = self._gemini_key_mutation_blockers()
        if blockers:
            QMessageBox.information(
                self,
                APP_TITLE,
                "API key belum boleh dihapus karena masih dipakai proses berikut:\n"
                + " · ".join(blockers),
            )
            return
        return super()._remove_gemini_key()

    def _fast_copy_safety_error(self) -> str | None:
        mode = str(self.export_mode.currentData() or "smartcut").strip().lower()
        if mode != "fast" or not self.model.cuts:
            return None
        if self.model.has_non_keyframe_cuts():
            return (
                "Fast Copy hanya aman untuk batas part yang berada tepat pada keyframe. "
                "Timeline saat ini memiliki cut frame-accurate/non-keyframe, sehingga "
                "FFmpeg dapat menggeser hasil ke keyframe lain. Pilih SmartCut agar "
                "hasil ekspor tetap sama persis dengan titik cut di timeline."
            )
        return None

    def tool_export_all(self):
        blockers = self._present_worker_labels({
            "analyze_worker",
            "agent_worker",
            "gemini_chat_worker",
            "film_cut_worker",
        })
        if blockers:
            raise RuntimeError(
                "Ekspor belum dapat dimulai karena proses berikut masih aktif/finalisasi: "
                + " · ".join(blockers)
            )
        error = self._fast_copy_safety_error()
        if error:
            raise RuntimeError(error)
        return super().tool_export_all()

    def tool_add_cut(self, time_ms):
        self._ensure_timeline_mutation_idle()
        return super().tool_add_cut(time_ms)

    def tool_remove_cut(self, index):
        self._ensure_timeline_mutation_idle()
        return super().tool_remove_cut(index)

    def tool_clear_cuts(self):
        self._ensure_timeline_mutation_idle()
        return super().tool_clear_cuts()

    def tool_divide_equal(self, parts):
        self._ensure_timeline_mutation_idle()
        return super().tool_divide_equal(parts)

    def tool_divide_interval(self, interval_ms):
        self._ensure_timeline_mutation_idle()
        return super().tool_divide_interval(interval_ms)

    def tool_apply_film_cut(self):
        self._ensure_timeline_mutation_idle()
        return super().tool_apply_film_cut()

    def tool_undo(self):
        self._ensure_timeline_mutation_idle()
        return super().tool_undo()

    def _choose_srt(self) -> bool:
        blockers = self._present_worker_labels({
            "analyze_worker",
            "gemini_chat_worker",
            "film_cut_worker",
            "export_worker",
        })
        if blockers:
            QMessageBox.information(
                self,
                APP_TITLE,
                "Tunggu proses berikut selesai sebelum mengganti SRT:\n"
                + " · ".join(blockers),
            )
            return False
        return super()._choose_srt()

    def _clear_srt(self):
        blockers = self._present_worker_labels({
            "analyze_worker",
            "gemini_chat_worker",
            "film_cut_worker",
            "export_worker",
        })
        if blockers:
            QMessageBox.information(
                self,
                APP_TITLE,
                "Tunggu proses berikut selesai sebelum melepas SRT:\n"
                + " · ".join(blockers),
            )
            return
        return super()._clear_srt()

    def _drain_bridge(self):
        """Execute bridge calls without replaying requests that already timed out.

        The bridge handler and GUI thread share ``call.lock``. After a tool
        returns, publishing its result and setting the completion event happen in
        the same critical section. Therefore timeout cannot win in the tiny gap
        between "mutation committed" and "caller told it completed".
        """
        self.bridge_state = self._state_with_subtitle()
        for _ in range(20):
            try:
                call = self.bridge_queue.get_nowait()
            except queue.Empty:
                break

            snapshot = None
            try:
                with call.lock:
                    if call.cancelled.is_set():
                        call.result.setdefault("ok", False)
                        call.result.setdefault("cancelled", True)
                        call.result.setdefault(
                            "error",
                            "Request bridge sudah dibatalkan sebelum dieksekusi.",
                        )
                        call.event.set()
                        continue
                    call.started.set()

                if call.tool in MUTATING_TOOLS:
                    snapshot = self._snapshot()

                result = self.registry.execute(call.tool, call.args)

                timed_out = False
                with call.lock:
                    timed_out = call.cancelled.is_set()
                    if not timed_out:
                        if (
                            call.tool in MUTATING_TOOLS
                            and isinstance(result, dict)
                            and result.get("ok", True)
                            and snapshot is not None
                        ):
                            self.undo_stack.append(snapshot)
                        call.result.update(result)
                        call.result.setdefault("ok", True)
                        # Result publication and completion are atomic relative
                        # to the HTTP timeout path in LocalBridge.
                        call.event.set()

                if timed_out:
                    # Timeout already released the HTTP caller. Undo any local
                    # state mutation instead of committing a late ghost action.
                    if snapshot is not None:
                        self._restore_snapshot(snapshot)
                    if (
                        call.tool == "export_all"
                        and getattr(self, "export_worker", None) is not None
                    ):
                        self.tool_cancel_export()
                    with call.lock:
                        call.result.clear()
                        call.result.update({
                            "ok": False,
                            "cancelled": True,
                            "error": (
                                "Request bridge timeout saat sedang diproses; "
                                "perubahan timeline dibatalkan."
                            ),
                        })
                        call.event.set()
                else:
                    # Refresh is deliberately outside call.lock. The caller can
                    # safely receive the completed tool result even if repainting
                    # UI takes a little longer.
                    self._refresh()
            except Exception as exc:
                cancelled = call.cancelled.is_set()
                if cancelled and snapshot is not None:
                    try:
                        self._restore_snapshot(snapshot)
                    except Exception:
                        pass
                with call.lock:
                    if cancelled:
                        call.result.clear()
                        call.result.update({
                            "ok": False,
                            "cancelled": True,
                            "error": str(exc),
                        })
                    else:
                        call.result.update({"ok": False, "error": str(exc)})
                    call.event.set()

    def closeEvent(self, event):
        # Network/analyze workers are deliberately not force-terminated. The
        # safest behavior is to keep the window alive until their queued callback
        # clears the worker reference.
        blocking = self._present_worker_labels({
            "analyze_worker",
            "agent_worker",
            "gemini_test_worker",
            "gemini_chat_worker",
        })
        if blocking:
            QMessageBox.information(
                self,
                APP_TITLE,
                "Tunggu proses berikut selesai sebelum keluar:\n"
                + " · ".join(blocking),
            )
            event.ignore()
            return

        batch = getattr(self, "gemini_batch_worker", None)
        if batch is not None:
            cancel = getattr(batch, "cancel", None)
            if callable(cancel):
                cancel()
            if hasattr(self, "gemini_batch_status_label"):
                self.gemini_batch_status_label.setText(
                    "Membatalkan cek semua API sebelum aplikasi ditutup…"
                )
            QMessageBox.information(
                self,
                APP_TITLE,
                "Cek Semua API sedang dihentikan. Tutup aplikasi lagi setelah status dibatalkan.",
            )
            event.ignore()
            return

        film = getattr(self, "film_cut_worker", None)
        if film is not None:
            answer = QMessageBox.question(
                self,
                APP_TITLE,
                "AI Film Cut masih berjalan atau memfinalisasi hasil. Batalkan proses?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.tool_cancel_film_cut()
            event.ignore()
            return

        export = getattr(self, "export_worker", None)
        if export is not None:
            answer = QMessageBox.question(
                self,
                APP_TITLE,
                "Ekspor masih berjalan atau memfinalisasi hasil. Batalkan ekspor?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.tool_cancel_export()
            event.ignore()
            return

        return super().closeEvent(event)


# Active runtime aliases used by the desktop entry point and future imports.
CameraAwareMiniCutWindow = HardenedMiniCutWindow
MiniCutWindow = HardenedMiniCutWindow
