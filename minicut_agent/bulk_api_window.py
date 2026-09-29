from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
)

from . import APP_TITLE
from .gemini_key_import import add_unnamed_gemini_key, import_gemini_api_keys_text
from .gemini_keys import MAX_GEMINI_KEYS
from .runtime_window import RuntimeMiniCutWindow as BaseRuntimeMiniCutWindow


class BulkApiRuntimeMiniCutWindow(BaseRuntimeMiniCutWindow):
    """Runtime window with simple unnamed API entry and bulk TXT import."""

    def _gemini_keys_tab(self):
        page = super()._gemini_keys_tab()

        # Keep the existing compact 3-button row, but replace the old manual
        # name/project editor with the workflow the user actually needs.
        self.gemini_add_key_btn.setText("+ Tambah 1 API")
        self.gemini_edit_key_btn.setText("Import TXT")
        self.gemini_edit_key_btn.setToolTip(
            "Import banyak Gemini API key dari TXT. Format paling mudah: satu key per baris."
        )
        try:
            self.gemini_edit_key_btn.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.gemini_edit_key_btn.clicked.connect(self._import_gemini_keys_txt)

        if hasattr(self, "gemini_manager_note"):
            self.gemini_manager_note.setText(
                "Import TXT = masukkan banyak API sekaligus tanpa nama/project; "
                "cukup satu API key per baris. Baris kosong dan duplikat dilewati otomatis. "
                "" + self.gemini_manager_note.text()
            )
        return page

    def _gemini_key_dialog(self, key_id: str | None = None):
        existing = None
        if key_id:
            existing = next(
                (item for item in self.gemini_keys.summaries() if item.id == key_id),
                None,
            )
            if not existing:
                QMessageBox.warning(self, APP_TITLE, "API key tidak ditemukan.")
                return

        dialog = QDialog(self)
        dialog.setWindowTitle("Ganti Gemini API" if existing else "Tambah Gemini API")
        dialog.resize(520, 145)
        form = QFormLayout(dialog)

        key_edit = QLineEdit()
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setPlaceholderText(
            "Tempel Gemini API key baru" if existing else "Tempel Gemini API key"
        )
        form.addRow("API key", key_edit)

        note = QLabel(
            "Tidak perlu memberi nama API atau project. MiniCut memberi nomor otomatis "
            "dan menyimpan key terenkripsi untuk user Windows ini."
        )
        note.setWordWrap(True)
        form.addRow(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            new_key = key_edit.text().strip()
            if existing:
                if not new_key:
                    return
                # Preserve legacy metadata internally for backward compatibility;
                # the UI no longer asks the user to maintain it.
                self.gemini_keys.update(
                    existing.id,
                    existing.name,
                    existing.project,
                    new_key,
                )
            else:
                add_unnamed_gemini_key(self.gemini_keys, new_key)
            self._refresh_gemini_key_views()
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))

    @staticmethod
    def _read_api_txt(path: Path) -> str:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def _import_gemini_keys_txt(self):
        if self.gemini_keys.count() >= MAX_GEMINI_KEYS:
            QMessageBox.information(
                self,
                APP_TITLE,
                f"Batas {MAX_GEMINI_KEYS} API key sudah tercapai. Hapus API lama jika ingin mengimpor lagi.",
            )
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import banyak Gemini API key",
            "",
            "Text (*.txt);;Semua file (*)",
        )
        if not selected:
            return

        try:
            text = self._read_api_txt(Path(selected))
            result = import_gemini_api_keys_text(self.gemini_keys, text)
        except Exception as exc:
            QMessageBox.critical(
                self,
                APP_TITLE,
                "Import API gagal:\n" + str(exc),
            )
            return

        self._refresh_gemini_key_views()
        summary = (
            f"Import selesai.\n\n"
            f"API baru: {result.added}\n"
            f"Duplikat dilewati: {result.duplicates}\n"
            f"Baris tidak valid: {result.invalid}\n"
            f"Lewat batas {MAX_GEMINI_KEYS}: {result.capacity_skipped}\n"
            f"Total tersimpan sekarang: {self.gemini_keys.count()} / {MAX_GEMINI_KEYS}"
        )
        if hasattr(self, "gemini_batch_status_label"):
            self.gemini_batch_status_label.setText(
                f"Import TXT · masuk {result.added} · duplikat {result.duplicates} · "
                f"tidak valid {result.invalid} · total {self.gemini_keys.count()}"
            )
        QMessageBox.information(self, APP_TITLE, summary)


MiniCutWindow = BulkApiRuntimeMiniCutWindow
