from __future__ import annotations

import bisect
import json
import queue
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPalette, QPen, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGridLayout, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget
)

from . import APP_TITLE
from .agent import AgentPlanner, MUTATING_TOOLS, ToolRegistry
from .bridge import BridgeCall, LocalBridge
from .core import (
    CutPoint, ProjectModel, SUPPORTED_VIDEO, clock_text, find_tool,
    load_project_file, parse_time_ms
)
from .gemini import DEFAULT_MODEL
from .frame_resolver import probe_frame_timestamps
from .gemini_keys import GeminiKeyStore, MAX_GEMINI_KEYS
from .manual_commands import extract_manual_timestamps, looks_like_manual_cut
from .preview_player import PreviewPlayer
from .workers import (
    AgentWorker, AnalyzeWorker, ExportWorker, FilmCutWorker,
    GeminiBatchTestWorker, GeminiChatWorker, GeminiTestWorker,
    ManualFrameCutWorker
)


class TimelineSlider(QSlider):
    def __init__(self):
        super().__init__(Qt.Orientation.Horizontal)
        self.marks: list[int] = []

    def set_marks(self, marks: list[int]):
        self.marks = list(marks)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.marks or self.maximum() <= 0:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#9b7bff"), 2))
        width = max(1, self.width() - 12)
        for mark in self.marks:
            x = 6 + int(width * mark / self.maximum())
            painter.drawLine(x, 3, x, self.height() - 3)


class MiniCutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 780)
        self.setAcceptDrops(True)

        self.model = ProjectModel()
        self.registry = ToolRegistry(self)
        self.planner = AgentPlanner(self.registry)
        self.undo_stack: list[dict] = []
        self.pending_plan: dict | None = None
        self.pending_load: tuple[Path, dict | None, Path | None] | None = None
        self.analyze_worker: AnalyzeWorker | None = None
        self.export_worker: ExportWorker | None = None
        self.agent_worker: AgentWorker | None = None
        self.gemini_test_worker: GeminiTestWorker | None = None
        self.gemini_batch_worker: GeminiBatchTestWorker | None = None
        self.gemini_chat_worker: GeminiChatWorker | None = None
        self.manual_cut_worker: ManualFrameCutWorker | None = None
        self.film_cut_worker: FilmCutWorker | None = None
        self._player_media_path: Path | None = None
        self._frame_pts_cache: list[int] = []
        self._frame_pts_cache_start = 0
        self._frame_pts_cache_end = 0
        self._scrub_pending_ms: int | None = None
        self._scrub_resume_after = False
        self.film_cut_results: list[dict] = []
        self.srt_path: Path | None = None
        self.gemini_keys = GeminiKeyStore()
        self._gemini_test_key_id: str | None = None
        self._gemini_chat_key_id: str | None = None
        self._gemini_chat_model: str = DEFAULT_MODEL
        self._gemini_chat_history_data: list[dict[str, str]] = []
        self._gemini_chat_pending_manual: list[dict] = []
        self._gemini_chat_deferred_actions: list[dict] = []
        self._gemini_chat_request_had_manual = False
        self._film_active_key_id: str | None = None
        self._film_active_model: str = DEFAULT_MODEL
        self._film_usage_seen_requests = 0
        self._film_usage_seen_prompt_tokens = 0

        self.bridge_queue: "queue.Queue[BridgeCall]" = queue.Queue()
        self.bridge_state: dict = self.model.state()
        self.bridge = LocalBridge(
            self.bridge_queue,
            state_provider=lambda: dict(self.bridge_state),
            manifest_provider=self.registry.manifest,
        )

        self.player = PreviewPlayer(self)

        self._build_ui()
        self._apply_modern_theme()
        self._connect_player()
        self._install_shortcuts()

        try:
            self.bridge.start()
            self.bridge_label.setText("Bridge aktif · 127.0.0.1:8765")
        except OSError as exc:
            self.bridge_label.setText("Bridge gagal: " + str(exc))

        self.bridge_timer = QTimer(self)
        self.bridge_timer.timeout.connect(self._drain_bridge)
        self.bridge_timer.start(120)

        self.quota_timer = QTimer(self)
        self.quota_timer.timeout.connect(self._refresh_active_quota_display)
        self.quota_timer.start(1000)

        self._refresh()

    # ---------- UI ----------
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("Root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Layout kembali seperti versi awal: toolbar atas, preview kiri, panel tab kanan.
        topbar = QWidget()
        topbar.setObjectName("TopBar")
        toolbar = QHBoxLayout(topbar)
        toolbar.setContentsMargins(12, 7, 12, 7)
        toolbar.setSpacing(6)

        brand = QLabel("MINI CUT")
        brand.setObjectName("Brand")
        toolbar.addWidget(brand)
        toolbar.addSpacing(6)

        self.open_video_btn = QPushButton("＋ Buka Video")
        self.open_project_btn = QPushButton("Buka Proyek")
        self.save_btn = QPushButton("Simpan")
        self.export_btn = QPushButton("Ekspor Semua Part")
        self.export_btn.setObjectName("PrimaryButton")
        self.export_mode = QComboBox()
        self.export_mode.setMinimumWidth(165)
        self.export_mode.addItem("SmartCut · Frame Accurate", "smartcut")
        self.export_mode.addItem("Fast Copy · Keyframe", "fast")

        toolbar.addWidget(self.open_video_btn)
        toolbar.addWidget(self.open_project_btn)
        toolbar.addWidget(self.save_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(self.export_mode)
        toolbar.addWidget(self.export_btn)
        outer.addWidget(topbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("WorkspaceSplitter")
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        # Preview besar di kiri, sama seperti layout awal.
        preview = QWidget()
        preview.setObjectName("PreviewPanel")
        pv = QVBoxLayout(preview)
        pv.setContentsMargins(10, 10, 8, 8)
        pv.setSpacing(6)

        preview_header = QHBoxLayout()
        preview_title = QLabel("PREVIEW")
        preview_title.setObjectName("SectionTitle")
        self.review_badge = QLabel("MASTER DIRECT")
        self.review_badge.setObjectName("Badge")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.review_badge)
        pv.addLayout(preview_header)

        self.video = self.player.widget
        self.video.setMinimumSize(500, 280)
        pv.addWidget(self.video, 1)

        self.timeline = TimelineSlider()
        self.timeline.setObjectName("ViewerTimeline")
        self.timeline.setRange(0, 0)
        pv.addWidget(self.timeline)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.back_btn = QPushButton("◀ 1 Frame")
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setObjectName("PlayButton")
        self.forward_btn = QPushButton("1 Frame ▶")

        self.speed_combo = QComboBox()
        for label, rate in (
            ("0.5x", 0.5), ("1x", 1.0), ("1.5x", 1.5),
            ("2x", 2.0), ("3x", 3.0), ("4x", 4.0)
        ):
            self.speed_combo.addItem(label, rate)
        self.speed_combo.setCurrentIndex(1)

        self.player_backend_label = QLabel("Player: menyiapkan mpv…")
        self.player_backend_label.setObjectName("MutedLabel")
        self.position_label = QLabel("00:00:00.000 / 00:00:00.000")
        self.position_label.setObjectName("Timecode")

        controls.addWidget(self.back_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.forward_btn)
        controls.addSpacing(6)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.speed_combo)
        controls.addWidget(self.player_backend_label)
        controls.addStretch(1)
        controls.addWidget(self.position_label)
        pv.addLayout(controls)
        splitter.addWidget(preview)

        # Semua alat kembali berada di panel kanan seperti versi awal.
        self.tabs = QTabWidget()
        self.tabs.setObjectName("InspectorTabs")
        self.tabs.setMinimumWidth(410)
        tab_specs = [
            (self._parts_tab(), "Timeline", "Timeline Part"),
            (self._agent_tab(), "Agent", "AI Agent"),
            (self._film_cut_tab(), "Film Cut", "AI Film Cut otomatis"),
            (self._gemini_chat_tab(), "Chat", "Gemini Chat / cut manual"),
            (self._gemini_keys_tab(), "API", "Gemini API Manager"),
            (self._log_tab(), "Log", "Log aplikasi"),
        ]
        for page, title, tip in tab_specs:
            index = self.tabs.addTab(page, title)
            self.tabs.setTabToolTip(index, tip)
        splitter.addWidget(self.tabs)
        splitter.setSizes([800, 480])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        footer = QWidget()
        footer.setObjectName("Footer")
        bottom = QHBoxLayout(footer)
        bottom.setContentsMargins(10, 5, 10, 5)
        self.status = QLabel("Buka video untuk mulai.")
        self.status.setObjectName("StatusLabel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(240)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.progress)
        outer.addWidget(footer)

        self.setCentralWidget(root)

        self.open_video_btn.clicked.connect(self._choose_video)
        self.open_project_btn.clicked.connect(self._choose_project)
        self.save_btn.clicked.connect(lambda: self.tool_save_project())
        self.export_btn.clicked.connect(lambda: self.tool_export_all())
        self.play_btn.clicked.connect(self._toggle_play)
        self.back_btn.clicked.connect(lambda: self._step_frame(-1))
        self.forward_btn.clicked.connect(lambda: self._step_frame(1))
        self.speed_combo.currentIndexChanged.connect(self._playback_rate_changed)

        # Direct-master scrubbing. Tidak ada pembuatan proxy.
        self.scrub_timer = QTimer(self)
        self.scrub_timer.setInterval(45)
        self.scrub_timer.timeout.connect(self._flush_scrub)
        self.timeline.sliderPressed.connect(self._scrub_started)
        self.timeline.sliderMoved.connect(self._queue_scrub)
        self.timeline.sliderReleased.connect(self._scrub_finished)

    def _media_panel(self):
        panel = QWidget()
        panel.setObjectName("MediaPanel")
        panel.setMinimumWidth(190)
        panel.setMaximumWidth(285)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("PROJECT MEDIA")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.media_open_btn = QPushButton("＋ Import Video")
        self.media_open_btn.setObjectName("PrimaryButton")
        layout.addWidget(self.media_open_btn)

        card = QWidget()
        card.setObjectName("MediaCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        self.media_name_label = QLabel("No media loaded")
        self.media_name_label.setObjectName("MediaName")
        self.media_name_label.setWordWrap(True)
        self.media_meta_label = QLabel("Drag & drop video di sini")
        self.media_meta_label.setObjectName("MutedLabel")
        self.media_meta_label.setWordWrap(True)
        card_layout.addWidget(self.media_name_label)
        card_layout.addWidget(self.media_meta_label)
        layout.addWidget(card)

        review_title = QLabel("REVIEW")
        review_title.setObjectName("SectionTitle")
        layout.addWidget(review_title)
        review_note = QLabel(
            "Preview memakai master asli langsung. Backend utama mpv/libmpv dengan hardware decode; "
            "Qt Multimedia hanya fallback bila mpv tidak tersedia."
        )
        review_note.setObjectName("MutedLabel")
        review_note.setWordWrap(True)
        layout.addWidget(review_note)
        layout.addStretch(1)
        return panel

    def _apply_modern_theme(self):
        self.setMinimumSize(1120, 680)

        # Palette global menjaga teks tetap terang pada widget/popup native Windows
        # yang kadang tidak sepenuhnya mengikuti QSS (terutama QComboBox popup).
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#0f1117"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8eaf0"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#14171e"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#181c25"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#e8ebf2"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#222733"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f3f5fa"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#7c5cff"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#20242e"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#7f8798"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#686f7d"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#686f7d"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#686f7d"))
        self.setPalette(palette)
        QApplication.instance().setPalette(palette)

        self.setStyleSheet("""
            QMainWindow, QWidget#Root {
                background: #0f1117;
                color: #e8eaf0;
                font-family: "Segoe UI";
                font-size: 9pt;
            }
            QWidget, QLabel, QCheckBox, QRadioButton, QGroupBox {
                color: #e7e9ef;
            }
            QWidget#TopBar {
                background: #12151c;
                border-bottom: 1px solid #272b36;
            }
            QLabel#Brand {
                color: #ffffff;
                font-size: 12pt;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#BrandSub, QLabel#MutedLabel {
                color: #858b9b;
            }
            QLabel#SectionTitle {
                color: #aeb4c3;
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#Badge {
                color: #bcaeff;
                background: #241d43;
                border: 1px solid #4c3d83;
                border-radius: 9px;
                padding: 3px 8px;
                font-size: 7.5pt;
                font-weight: 700;
            }
            QLabel#Timecode {
                color: #d9dce6;
                font-family: "Consolas";
                font-size: 8.5pt;
                font-weight: 600;
            }
            QWidget#QuotaCard {
                background: #191d27;
                border: 1px solid #353b4b;
                border-radius: 9px;
            }
            QLabel#QuotaValue {
                color: #f2f3f8;
                font-family: "Consolas";
                font-size: 9pt;
                font-weight: 700;
            }
            QLabel#SessionUsage {
                color: #bcaeff;
                font-family: "Consolas";
                font-weight: 700;
            }
            QLabel#MediaName {
                color: #ffffff;
                font-weight: 700;
            }
            QWidget#MediaPanel, QWidget#PreviewPanel, QWidget#TimelinePanel {
                background: #151820;
            }
            QWidget#MediaCard {
                background: #1b1f29;
                border: 1px solid #2a2f3b;
                border-radius: 10px;
            }
            QWidget#VideoSurface, QWidget#MpvSurface, QVideoWidget#QtVideoSurface {
                background: #050608;
                border: 1px solid #2b303c;
                border-radius: 8px;
            }
            QTabWidget#InspectorTabs::pane {
                border: 0px;
                background: #151820;
            }
            QTabBar::tab {
                background: #151820;
                color: #aab0bf;
                padding: 7px 7px;
                font-size: 8.5pt;
                border: 0px;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected {
                color: #ffffff;
                border-bottom: 2px solid #8b6cff;
            }
            QPushButton {
                background: #222733;
                color: #e7e9ef;
                border: 1px solid #303645;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 8.5pt;
            }
            QPushButton:hover {
                background: #2a3040;
                border-color: #454d61;
            }
            QPushButton:pressed {
                background: #1b1f29;
            }
            QPushButton:disabled {
                color: #5f6573;
                background: #191c24;
                border-color: #242833;
            }
            QPushButton#PrimaryButton {
                background: #7c5cff;
                color: white;
                border: 1px solid #9178ff;
                font-weight: 700;
            }
            QPushButton#PrimaryButton:hover {
                background: #8b6cff;
            }
            QPushButton#PlayButton {
                min-width: 38px;
                font-size: 9pt;
                background: #f1f3f8;
                color: #101219;
                border: 0px;
            }
            QComboBox, QLineEdit, QSpinBox, QPlainTextEdit {
                background: #1b1f29;
                color: #e7e9ef;
                border: 1px solid #303645;
                border-radius: 6px;
                padding: 4px 6px;
                font-size: 8.5pt;
                selection-background-color: #7c5cff;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: #1b1f29;
                color: #f3f5fa;
                border: 1px solid #3a4152;
                selection-background-color: #7c5cff;
                selection-color: #ffffff;
                outline: 0;
            }
            QComboBox QAbstractItemView::item {
                color: #f3f5fa;
                background: #1b1f29;
                min-height: 23px;
                padding: 3px 6px;
            }
            QComboBox QAbstractItemView::item:selected {
                color: #ffffff;
                background: #7c5cff;
            }
            QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QPlainTextEdit:disabled {
                color: #686f7d;
                background: #171a22;
                border-color: #242a35;
            }
            QDialog, QMessageBox {
                background: #11151d;
                color: #e8ebf2;
            }
            QMenu {
                background: #1b1f29;
                color: #f3f5fa;
                border: 1px solid #3a4152;
            }
            QMenu::item:selected {
                background: #7c5cff;
                color: #ffffff;
            }
            QListView, QTreeView {
                background: #14171e;
                color: #e7e9ef;
                selection-background-color: #39305f;
                selection-color: #ffffff;
            }
            QTableWidget {
                background: #14171e;
                font-size: 8.5pt;
                alternate-background-color: #181c25;
                color: #e8ebf2;
                border: 1px solid #2a2f3b;
                border-radius: 7px;
                gridline-color: #252a35;
                selection-background-color: #39305f;
            }
            QHeaderView::section {
                background: #1b1f29;
                color: #9da4b5;
                border: 0px;
                border-right: 1px solid #292e39;
                border-bottom: 1px solid #292e39;
                padding: 4px;
                font-size: 8pt;
                font-weight: 700;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #2b303b;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #8b6cff;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 13px;
                margin: -5px 0;
                background: #ffffff;
                border: 2px solid #8b6cff;
                border-radius: 7px;
            }
            QProgressBar {
                background: #1b1f29;
                border: 1px solid #2d3340;
                border-radius: 5px;
                text-align: center;
                color: #dfe2ea;
                min-height: 13px;
                font-size: 8pt;
            }
            QProgressBar::chunk {
                background: #7c5cff;
                border-radius: 4px;
            }
            QWidget#Footer {
                background: #11141a;
                border-top: 1px solid #252a34;
            }
            QLabel#StatusLabel {
                color: #9ca3b3;
                font-size: 8pt;
            }
            QSplitter::handle {
                background: #242934;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
            QSplitter::handle:vertical {
                height: 2px;
            }
            QScrollBar:vertical {
                background: #12161e;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3a4150;
                min-height: 28px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #50596b;
            }
            QScrollBar:add-line:vertical, QScrollBar:sub-line:vertical,
            QScrollBar:add-page:vertical, QScrollBar:sub-page:vertical {
                background: transparent;
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #12161e;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #3a4150;
                min-width: 28px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #50596b;
            }
            QScrollBar:add-line:horizontal, QScrollBar:sub-line:horizontal,
            QScrollBar:add-page:horizontal, QScrollBar:sub-page:horizontal {
                background: transparent;
                width: 0px;
            }
            QToolTip {
                background: #20242e;
                color: white;
                border: 1px solid #3b4252;
                padding: 5px;
            }
        """)

    def _install_shortcuts(self):
        self.shortcut_play = QShortcut(QKeySequence("Space"), self)
        self.shortcut_play.activated.connect(self._toggle_play)
        self.shortcut_prev = QShortcut(QKeySequence("Left"), self)
        self.shortcut_prev.activated.connect(lambda: self._step_frame(-1))
        self.shortcut_next = QShortcut(QKeySequence("Right"), self)
        self.shortcut_next.activated.connect(lambda: self._step_frame(1))
        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(self._choose_video)
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(lambda: self.tool_save_project())

    def _scrub_started(self):
        self._scrub_resume_after = self.player.is_playing()
        if self._scrub_resume_after:
            self.player.pause()
        self._scrub_pending_ms = int(self.timeline.value())
        if not self.scrub_timer.isActive():
            self.scrub_timer.start()

    def _queue_scrub(self, value: int):
        self._scrub_pending_ms = int(value)
        self.position_label.setText(
            f"{clock_text(int(value))} / {clock_text(self.model.duration_ms)}"
        )

    def _flush_scrub(self):
        if self._scrub_pending_ms is None:
            return
        ms = self.model.clamp(int(self._scrub_pending_ms))
        self._scrub_pending_ms = None
        # Selama drag gunakan keyframe seek yang jauh lebih ringan pada master asli.
        self.player.set_position(ms, exact=False)
        self.model.playhead_ms = ms

    def _scrub_finished(self):
        final_ms = self.model.clamp(int(self.timeline.value()))
        self._scrub_pending_ms = None
        self.scrub_timer.stop()
        # Setelah mouse dilepas, kunci ke waktu exact.
        self.player.set_position(final_ms, exact=True)
        self.model.playhead_ms = final_ms
        if self._scrub_resume_after:
            self.player.play()
        self._scrub_resume_after = False

    def _parts_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(7)
        header = QHBoxLayout()
        timeline_title = QLabel("TIMELINE PARTS")
        timeline_title.setObjectName("SectionTitle")
        self.info_label = QLabel("Belum ada video.")
        self.info_label.setObjectName("MutedLabel")
        self.info_label.setWordWrap(True)
        header.addWidget(timeline_title)
        header.addSpacing(12)
        header.addWidget(self.info_label, 1)
        layout.addLayout(header)

        self.parts_table = QTableWidget(0, 4)
        self.parts_table.setHorizontalHeaderLabels(["Part", "Mulai", "Selesai", "Durasi"])
        self.parts_table.horizontalHeader().setStretchLastSection(True)
        self.parts_table.verticalHeader().setVisible(False)
        self.parts_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.parts_table.setAlternatingRowColors(True)
        self.parts_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.parts_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.parts_table.setMinimumHeight(120)
        layout.addWidget(self.parts_table, 1)

        row1 = QGridLayout()
        row1.setHorizontalSpacing(6)
        row1.setVerticalSpacing(6)
        self.add_cut_btn = QPushButton("Tambah Cut di Playhead")
        self.remove_cut_btn = QPushButton("Hapus Cut Terpilih")
        self.clear_cut_btn = QPushButton("Hapus Semua Cut")
        row1.addWidget(self.add_cut_btn, 0, 0)
        row1.addWidget(self.remove_cut_btn, 0, 1)
        row1.addWidget(self.clear_cut_btn, 1, 0, 1, 2)
        row1.setColumnStretch(0, 1)
        row1.setColumnStretch(1, 1)
        layout.addLayout(row1)

        row2 = QGridLayout()
        row2.setHorizontalSpacing(6)
        row2.setVerticalSpacing(6)
        self.equal_parts = QSpinBox()
        self.equal_parts.setRange(2, 999)
        self.equal_parts.setValue(4)
        self.divide_equal_btn = QPushButton("Bagi Sama")
        self.interval_edit = QLineEdit("00:10:00")
        self.divide_interval_btn = QPushButton("Bagi Tiap Interval")
        row2.addWidget(QLabel("Part:"), 0, 0)
        row2.addWidget(self.equal_parts, 0, 1)
        row2.addWidget(self.divide_equal_btn, 0, 2)
        row2.addWidget(QLabel("Interval:"), 1, 0)
        row2.addWidget(self.interval_edit, 1, 1)
        row2.addWidget(self.divide_interval_btn, 1, 2)
        row2.setColumnStretch(1, 1)
        layout.addLayout(row2)

        self.add_cut_btn.clicked.connect(lambda: self._manual_mutation("add_cut", {"time_ms": self.model.playhead_ms}))
        self.remove_cut_btn.clicked.connect(self._remove_selected)
        self.clear_cut_btn.clicked.connect(lambda: self._manual_mutation("clear_cuts", {}))
        self.divide_equal_btn.clicked.connect(
            lambda: self._manual_mutation("divide_equal", {"parts": self.equal_parts.value()})
        )
        self.divide_interval_btn.clicked.connect(self._divide_interval_clicked)
        return w

    def _agent_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        self.agent_state = QLabel()
        self.agent_state.setWordWrap(True)
        layout.addWidget(self.agent_state)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.agent_mode = QComboBox()
        self.agent_mode.addItems(["Lokal · tanpa API", "OpenAI-compatible API"])
        self.endpoint_edit = QLineEdit("http://127.0.0.1:1234/v1")
        self.model_edit = QLineEdit("local-model")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Mode", self.agent_mode)
        form.addRow("Endpoint", self.endpoint_edit)
        form.addRow("Model", self.model_edit)
        form.addRow("API key", self.api_key_edit)
        layout.addLayout(form)

        self.agent_input = QPlainTextEdit()
        self.agent_input.setPlaceholderText(
            "Contoh: bagi jadi 8 part\n"
            "tiap 10 menit\n"
            "tambah cut di 00:12:30 dan 00:25:00"
        )
        self.agent_input.setMaximumHeight(105)
        layout.addWidget(self.agent_input)

        buttons = QHBoxLayout()
        self.plan_btn = QPushButton("Buat Rencana")
        self.apply_btn = QPushButton("Terapkan")
        self.undo_btn = QPushButton("Undo Agent")
        self.apply_btn.setEnabled(False)
        buttons.addWidget(self.plan_btn)
        buttons.addWidget(self.apply_btn)
        buttons.addWidget(self.undo_btn)
        layout.addLayout(buttons)

        self.plan_preview = QPlainTextEdit()
        self.plan_preview.setReadOnly(True)
        layout.addWidget(self.plan_preview, 1)

        self.bridge_label = QLabel("Bridge belum aktif")
        layout.addWidget(self.bridge_label)

        self.agent_mode.currentIndexChanged.connect(self._agent_mode_changed)
        self.plan_btn.clicked.connect(self._make_plan)
        self.apply_btn.clicked.connect(self._apply_plan)
        self.undo_btn.clicked.connect(lambda: self.tool_undo())
        self._agent_mode_changed()
        return w


    def _gemini_chat_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        intro = QLabel(
            "Gemini di tab ini adalah command agent MiniCut: ia memahami bahasa natural, reasoning "
            "secara internal, lalu dapat memakai tool timeline/playback/proyek yang tersedia. "
            "Cut waktu tertentu tetap dikunci ke PTS frame master nyata agar frame-accurate."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.gemini_chat_api_label = QLabel("API aktif: — · Model: —")
        self.gemini_chat_api_label.setObjectName("QuotaValue")
        self.gemini_chat_api_label.setWordWrap(True)
        layout.addWidget(self.gemini_chat_api_label)

        self.gemini_chat_status = QLabel(
            "Contoh: cut 1 jam lebih 2 menit · bagi jadi 8 part · lompat ke 00:42:10 lalu pause"
        )
        self.gemini_chat_status.setObjectName("MutedLabel")
        self.gemini_chat_status.setWordWrap(True)
        layout.addWidget(self.gemini_chat_status)

        self.gemini_chat_history = QPlainTextEdit()
        self.gemini_chat_history.setReadOnly(True)
        self.gemini_chat_history.setPlaceholderText(
            "Percakapan dengan Gemini akan muncul di sini."
        )
        layout.addWidget(self.gemini_chat_history, 1)

        self.gemini_chat_input = QPlainTextEdit()
        self.gemini_chat_input.setMaximumHeight(88)
        self.gemini_chat_input.setPlaceholderText(
            "Tulis perintah seperti manusia… (Ctrl+Enter untuk kirim)\n"
            "Contoh: cut 1 jam lebih 2 menit, atau bagi jadi 8 part lalu lompat ke 30 menit"
        )
        layout.addWidget(self.gemini_chat_input)

        buttons = QHBoxLayout()
        self.gemini_chat_send_btn = QPushButton("Kirim")
        self.gemini_chat_send_btn.setObjectName("PrimaryButton")
        self.gemini_chat_clear_btn = QPushButton("Bersihkan Chat")
        self.gemini_chat_undo_btn = QPushButton("Undo Cut")
        buttons.addWidget(self.gemini_chat_send_btn)
        buttons.addWidget(self.gemini_chat_clear_btn)
        buttons.addWidget(self.gemini_chat_undo_btn)
        layout.addLayout(buttons)

        note = QLabel(
            "Gemini boleh merencanakan beberapa aksi sekaligus, tetapi hanya tool MiniCut yang "
            "diizinkan yang dapat dijalankan. Cut waktu spesifik memakai frame-lock lokal; "
            "ekspor SmartCut mempertahankan hasil frame-accurate."
        )
        note.setObjectName("MutedLabel")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.gemini_chat_send_btn.clicked.connect(self._send_gemini_chat)
        self.gemini_chat_clear_btn.clicked.connect(self._clear_gemini_chat)
        self.gemini_chat_undo_btn.clicked.connect(lambda: self.tool_undo())
        self.gemini_chat_send_shortcut = QShortcut(
            QKeySequence("Ctrl+Return"), self.gemini_chat_input
        )
        self.gemini_chat_send_shortcut.activated.connect(self._send_gemini_chat)
        self.gemini_chat_send_shortcut2 = QShortcut(
            QKeySequence("Ctrl+Enter"), self.gemini_chat_input
        )
        self.gemini_chat_send_shortcut2.activated.connect(self._send_gemini_chat)
        return w


    def _film_cut_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        intro = QLabel(
            "Target tetap pada grid 15, 30, 45, 60 menit, dst. Contact sheet + SRT dipakai untuk "
            "memahami konteks, lalu MiniCut mengambil kandidat TERDEKAT dari target dan membandingkan "
            "hingga 4 kandidat dengan video pendek sekitar 20 detik + SRT. Kandidat lebih jauh hanya "
            "boleh dipilih bila semua kandidat yang lebih dekat memang bukan boundary scene yang valid."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        self.gemini_key_combo = QComboBox()
        self.gemini_manage_btn = QPushButton("Kelola API")
        key_layout.addWidget(self.gemini_key_combo, 1)
        key_layout.addWidget(self.gemini_manage_btn)
        self.gemini_model_combo = QComboBox()
        self.gemini_model_combo.addItems([
            DEFAULT_MODEL,
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
        ])
        form.addRow("API aktif", key_row)
        form.addRow("Model", self.gemini_model_combo)

        srt_row = QWidget()
        srt_layout = QHBoxLayout(srt_row)
        srt_layout.setContentsMargins(0, 0, 0, 0)
        self.srt_edit = QLineEdit()
        self.srt_edit.setReadOnly(True)
        self.srt_edit.setPlaceholderText("Belum ada SRT")
        self.srt_btn = QPushButton("Pilih SRT")
        srt_layout.addWidget(self.srt_edit, 1)
        srt_layout.addWidget(self.srt_btn)
        form.addRow("Subtitle", srt_row)

        self.film_interval = QSpinBox()
        self.film_interval.setRange(5, 60)
        self.film_interval.setValue(15)
        self.film_interval.setSuffix(" menit")
        self.film_window = QSpinBox()
        self.film_window.setRange(1, 5)
        self.film_window.setValue(2)
        self.film_window.setSuffix(" menit")
        self.film_cache = QCheckBox("Simpan hasil parsial agar bisa dilanjutkan")
        self.film_cache.setChecked(True)
        self.film_deep_check = QCheckBox("Deep Check otomatis jika storyboard masih ragu")
        self.film_deep_check.setChecked(True)
        form.addRow("Grid target", self.film_interval)
        form.addRow("Window awal ±", self.film_window)
        grid_note = QLabel("Contoh 15 menit → target tetap 15:00, 30:00, 45:00, 60:00…")
        grid_note.setObjectName("MutedLabel")
        grid_note.setWordWrap(True)
        form.addRow("Logika", grid_note)
        form.addRow("Verifikasi", self.film_deep_check)
        form.addRow("Resume", self.film_cache)
        layout.addLayout(form)

        quota_card = QWidget()
        quota_card.setObjectName("QuotaCard")
        quota_layout = QVBoxLayout(quota_card)
        quota_layout.setContentsMargins(10, 8, 10, 8)
        quota_layout.setSpacing(3)
        quota_title = QLabel("KUOTA GEMINI AKTIF · PERKIRAAN LOKAL MINICUT")
        quota_title.setObjectName("SectionTitle")
        self.film_quota_label = QLabel("RPM sisa —  ·  TPM sisa —  ·  RPD sisa —")
        self.film_quota_label.setObjectName("QuotaValue")
        self.film_quota_label.setWordWrap(True)
        self.film_quota_reset_label = QLabel(
            "Pilih API key aktif untuk melihat sisa kuota yang tercatat MiniCut."
        )
        self.film_quota_reset_label.setObjectName("MutedLabel")
        self.film_quota_reset_label.setWordWrap(True)
        quota_layout.addWidget(quota_title)
        quota_layout.addWidget(self.film_quota_label)
        quota_layout.addWidget(self.film_quota_reset_label)
        layout.addWidget(quota_card)

        actions = QGridLayout()
        actions.setHorizontalSpacing(6)
        actions.setVerticalSpacing(6)
        self.gemini_test_btn = QPushButton("Tes API")
        self.film_analyze_btn = QPushButton("Analisis Film")
        self.film_analyze_btn.setObjectName("PrimaryButton")
        self.film_cancel_btn = QPushButton("Batalkan")
        self.film_apply_btn = QPushButton("Terapkan Semua Cut")
        self.film_cancel_btn.setEnabled(False)
        self.film_apply_btn.setEnabled(False)
        actions.addWidget(self.gemini_test_btn, 0, 0)
        actions.addWidget(self.film_analyze_btn, 0, 1)
        actions.addWidget(self.film_cancel_btn, 1, 0)
        actions.addWidget(self.film_apply_btn, 1, 1)
        actions.setColumnStretch(0, 1)
        actions.setColumnStretch(1, 1)
        layout.addLayout(actions)

        self.film_status_label = QLabel("Siap. Buka video, pilih SRT, lalu pilih Gemini API aktif.")
        self.film_status_label.setWordWrap(True)
        self.film_usage_label = QLabel(
            "SESI SAAT INI · Request 0 · Input 0 · Output 0 · Total 0 token"
        )
        self.film_usage_label.setObjectName("SessionUsage")
        self.film_usage_label.setWordWrap(True)
        layout.addWidget(self.film_status_label)
        layout.addWidget(self.film_usage_label)

        self.film_table = QTableWidget(0, 5)
        self.film_table.setHorizontalHeaderLabels(
            ["Target", "Cut final", "Conf.", "Status", "Alasan"]
        )
        self.film_table.verticalHeader().setVisible(False)
        self.film_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.film_table.setAlternatingRowColors(True)
        self.film_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.film_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.film_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.film_table.setWordWrap(False)
        film_header = self.film_table.horizontalHeader()
        film_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        film_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        film_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        film_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        film_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.film_table, 1)

        review_actions = QHBoxLayout()
        self.film_preview_btn = QPushButton("Preview Terpilih")
        self.film_preview_btn.setEnabled(False)
        review_hint = QLabel("Double-click baris atau pilih lalu Preview. NO CUT membuka target grid.")
        review_hint.setObjectName("MutedLabel")
        review_hint.setWordWrap(True)
        review_actions.addWidget(self.film_preview_btn)
        review_actions.addWidget(review_hint, 1)
        layout.addLayout(review_actions)

        self.srt_btn.clicked.connect(self._choose_srt)
        self.gemini_manage_btn.clicked.connect(self._open_gemini_manager)
        self.gemini_key_combo.currentIndexChanged.connect(self._film_key_changed)
        self.gemini_model_combo.currentIndexChanged.connect(self._refresh_gemini_key_views)
        self.gemini_test_btn.clicked.connect(self._test_gemini)
        self.film_analyze_btn.clicked.connect(self._start_film_cut)
        self.film_cancel_btn.clicked.connect(self._cancel_film_cut)
        self.film_apply_btn.clicked.connect(lambda: self._apply_film_cut(record_undo=True))
        self.film_preview_btn.clicked.connect(self._preview_selected_film_cut)
        self.film_table.itemSelectionChanged.connect(self._film_cut_selection_changed)
        self.film_table.cellDoubleClicked.connect(self._preview_film_cut_row)
        self._refresh_gemini_key_views()
        return w


    def _gemini_keys_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        intro = QLabel(
            "Simpan hingga 100 Gemini API key. Key disimpan terenkripsi dengan Windows DPAPI. "
            "Persentase RPM/TPM/RPD adalah pemakaian yang dicatat MiniCut, bukan dashboard Google penuh."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.gemini_key_count_label = QLabel()
        layout.addWidget(self.gemini_key_count_label)

        manager_quota_card = QWidget()
        manager_quota_card.setObjectName("QuotaCard")
        manager_quota_layout = QVBoxLayout(manager_quota_card)
        manager_quota_layout.setContentsMargins(10, 8, 10, 8)
        manager_quota_layout.setSpacing(3)
        manager_quota_title = QLabel("SISA KUOTA API AKTIF")
        manager_quota_title.setObjectName("SectionTitle")
        self.manager_quota_label = QLabel("RPM sisa —  ·  TPM sisa —  ·  RPD sisa —")
        self.manager_quota_label.setObjectName("QuotaValue")
        self.manager_quota_label.setWordWrap(True)
        self.manager_quota_reset_label = QLabel("Belum ada API key aktif.")
        self.manager_quota_reset_label.setObjectName("MutedLabel")
        self.manager_quota_reset_label.setWordWrap(True)
        manager_quota_layout.addWidget(manager_quota_title)
        manager_quota_layout.addWidget(self.manager_quota_label)
        manager_quota_layout.addWidget(self.manager_quota_reset_label)
        layout.addWidget(manager_quota_card)

        self.gemini_keys_table = QTableWidget(0, 6)
        self.gemini_keys_table.setHorizontalHeaderLabels([
            "Aktif", "API", "Status", "RPM", "TPM", "RPD"
        ])
        self.gemini_keys_table.verticalHeader().setVisible(False)
        self.gemini_keys_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.gemini_keys_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.gemini_keys_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.gemini_keys_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.gemini_keys_table.setWordWrap(False)
        api_header = self.gemini_keys_table.horizontalHeader()
        api_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        api_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        api_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        api_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        api_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        api_header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.gemini_keys_table, 1)

        self.gemini_add_key_btn = QPushButton("+ Tambah API")
        self.gemini_edit_key_btn = QPushButton("Edit")
        self.gemini_remove_key_btn = QPushButton("Hapus")
        self.gemini_activate_key_btn = QPushButton("Jadikan Aktif")
        self.gemini_test_selected_btn = QPushButton("Tes Terpilih")
        self.gemini_refresh_all_btn = QPushButton("Lihat Semua Status")
        self.gemini_test_all_btn = QPushButton("Cek Semua API Online")
        self.gemini_use_ready_btn = QPushButton("Pakai API SIAP")
        self.gemini_cancel_all_btn = QPushButton("Batalkan Cek")
        self.gemini_cancel_all_btn.setEnabled(False)

        key_actions = QGridLayout()
        key_actions.setHorizontalSpacing(6)
        key_actions.addWidget(self.gemini_add_key_btn, 0, 0)
        key_actions.addWidget(self.gemini_edit_key_btn, 0, 1)
        key_actions.addWidget(self.gemini_remove_key_btn, 0, 2)
        key_actions.setColumnStretch(0, 1)
        key_actions.setColumnStretch(1, 1)
        key_actions.setColumnStretch(2, 1)
        layout.addLayout(key_actions)

        status_actions = QGridLayout()
        status_actions.setHorizontalSpacing(6)
        status_actions.setVerticalSpacing(6)
        status_actions.addWidget(self.gemini_activate_key_btn, 0, 0)
        status_actions.addWidget(self.gemini_test_selected_btn, 0, 1)
        status_actions.addWidget(self.gemini_refresh_all_btn, 1, 0, 1, 2)
        status_actions.setColumnStretch(0, 1)
        status_actions.setColumnStretch(1, 1)
        layout.addLayout(status_actions)

        batch_actions = QGridLayout()
        batch_actions.setHorizontalSpacing(6)
        batch_actions.setVerticalSpacing(6)
        batch_actions.addWidget(self.gemini_test_all_btn, 0, 0, 1, 2)
        batch_actions.addWidget(self.gemini_use_ready_btn, 1, 0)
        batch_actions.addWidget(self.gemini_cancel_all_btn, 1, 1)
        batch_actions.setColumnStretch(0, 1)
        batch_actions.setColumnStretch(1, 1)
        layout.addLayout(batch_actions)

        self.gemini_batch_status_label = QLabel("Belum ada pengecekan semua API.")
        self.gemini_batch_status_label.setObjectName("MutedLabel")
        self.gemini_batch_status_label.setWordWrap(True)
        layout.addWidget(self.gemini_batch_status_label)

        self.gemini_manager_note = QLabel(
            "Lihat Semua Status = hanya membaca catatan lokal dan tidak memakai request. "
            "Cek Semua API Online = mengirim 1 request tes per key secara bertahap dan dapat memakai kuota. "
            "MiniCut tidak merotasi key otomatis untuk melewati kuota; pilih key SIAP dengan satu klik. "
            "Pada limit sementara, key aktif akan retry/backoff terlebih dahulu."
        )
        self.gemini_manager_note.setWordWrap(True)
        layout.addWidget(self.gemini_manager_note)

        self.gemini_add_key_btn.clicked.connect(self._add_gemini_key)
        self.gemini_edit_key_btn.clicked.connect(self._edit_gemini_key)
        self.gemini_remove_key_btn.clicked.connect(self._remove_gemini_key)
        self.gemini_activate_key_btn.clicked.connect(self._activate_selected_gemini_key)
        self.gemini_test_selected_btn.clicked.connect(self._test_selected_gemini_key)
        self.gemini_refresh_all_btn.clicked.connect(self._refresh_all_gemini_status)
        self.gemini_test_all_btn.clicked.connect(self._test_all_gemini_keys)
        self.gemini_use_ready_btn.clicked.connect(self._use_ready_gemini_key)
        self.gemini_cancel_all_btn.clicked.connect(self._cancel_all_gemini_tests)
        self._refresh_gemini_key_views()
        return w

    def _log_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        return w

    def _connect_player(self):
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackChanged.connect(self._playback_changed)
        self.player.backendChanged.connect(self._player_backend_changed)
        self.player.errorOccurred.connect(lambda text: self._log("Player: " + str(text)))
        self._player_backend_changed(self.player.backend_name)

    # ---------- loading ----------
    def _choose_video(self) -> bool:
        path, _ = QFileDialog.getOpenFileName(
            self, "Buka video", "", "Video (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts *.mts);;Semua file (*)"
        )
        if not path:
            return False
        return self._begin_load(Path(path))

    def _choose_project(self) -> bool:
        path, _ = QFileDialog.getOpenFileName(self, "Buka proyek MiniCut", "", "MiniCut JSON (*.json)")
        if not path:
            return False
        try:
            data, source = load_project_file(Path(path))
            if not source:
                chosen, _ = QFileDialog.getOpenFileName(self, "Cari video sumber", "", "Video (*.*)")
                if not chosen:
                    return False
                source = Path(chosen)
            return self._begin_load(source, data, Path(path))
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))
            return False

    def _begin_load(self, source: Path, project_data: dict | None = None, project_path: Path | None = None) -> bool:
        if self.model.dirty:
            answer = QMessageBox.question(
                self,
                APP_TITLE,
                "Perubahan cut belum disimpan. Tetap buka video/proyek lain?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False

        if (
            (self.manual_cut_worker and self.manual_cut_worker.isRunning())
            or (self.gemini_chat_worker and self.gemini_chat_worker.isRunning())
            or (self.film_cut_worker and self.film_cut_worker.isRunning())
            or (self.export_worker and self.export_worker.isRunning())
            or (self.analyze_worker and self.analyze_worker.isRunning())
        ):
            QMessageBox.information(
                self,
                APP_TITLE,
                "Tunggu proses edit/AI yang sedang berjalan sebelum membuka video atau proyek lain.",
            )
            return False
        self._frame_pts_cache = []
        self._frame_pts_cache_start = 0
        self._frame_pts_cache_end = 0
        if hasattr(self, "player_backend_label"):
            self.player_backend_label.setText("Player: " + self.player.backend_name)
        ffprobe = find_tool("ffprobe")
        if not ffprobe:
            QMessageBox.critical(self, APP_TITLE, "ffprobe tidak ditemukan. Pastikan FFmpeg tersedia.")
            return False
        if not source.is_file():
            QMessageBox.warning(self, APP_TITLE, "File video tidak ditemukan.")
            return False
        self.pending_load = (source.resolve(), project_data, project_path)
        self.status.setText("Menganalisis video dan keyframe…")
        self.progress.setRange(0, 0)
        self.analyze_worker = AnalyzeWorker(ffprobe, source.resolve())
        self.analyze_worker.ready.connect(self._analysis_ready)
        self.analyze_worker.failed.connect(self._analysis_failed)
        self.analyze_worker.start()
        return True

    def _analysis_ready(self, metadata: dict, keyframes: list):
        assert self.pending_load is not None
        source, data, project_path = self.pending_load
        self.model.reset(source, metadata)
        self.model.keyframes = keyframes
        project_cut_warning = ""
        if data:
            cuts = []
            for item in data.get("cuts", []):
                req = int(item.get("requested_ms", item.get("actual_ms", 0)))
                actual = int(item.get("actual_ms", req))
                cuts.append(CutPoint(req, actual))
            loaded_count = len(cuts)
            self.model.cuts = cuts
            self.model._normalize()
            self.model.project_path = project_path
            normalized_count = len(self.model.cuts)
            changed = normalized_count != loaded_count
            self.model.dirty = changed
            if changed:
                project_cut_warning = (
                    f"{loaded_count - normalized_count} cut proyek tidak lagi valid "
                    "untuk durasi/video sumber saat ini dan tidak dimuat. "
                    "File proyek asli belum diubah."
                )
        self._switch_player_media(source, resume=False)
        self.timeline.setRange(0, max(0, self.model.duration_ms))
        self.undo_stack.clear()

        # Subtitle dan hasil AI selalu terkait video tertentu. Jangan membawa
        # SRT/hasil Film Cut dari media sebelumnya ke video yang baru dibuka.
        self.srt_path = None
        if hasattr(self, "srt_edit"):
            self.srt_edit.clear()
            self.srt_edit.setPlaceholderText("Belum ada SRT")
        if hasattr(self, "film_status_label"):
            self.film_status_label.setText(
                "Video baru dibuka. Pilih SRT yang sesuai sebelum Analisis Film."
            )

        self.film_cut_results = []
        if hasattr(self, 'film_table'):
            self.film_table.setRowCount(0)
            self.film_apply_btn.setEnabled(False)
            if hasattr(self, "film_preview_btn"):
                self.film_preview_btn.setEnabled(False)
        self.pending_load = None
        self.analyze_worker = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText(f"Siap · {source.name} · {len(keyframes)} keyframe")
        self._log(f"Video dibuka: {source}")
        if project_cut_warning:
            self._log("Peringatan proyek: " + project_cut_warning)
            QMessageBox.warning(self, APP_TITLE, project_cut_warning)
        self._refresh()
        self._player_backend_changed(self.player.backend_name)


    # ---------- master-direct playback ----------
    def _current_playback_rate(self) -> float:
        if not hasattr(self, "speed_combo"):
            return 1.0
        try:
            return float(self.speed_combo.currentData() or 1.0)
        except Exception:
            return 1.0

    def _switch_player_media(self, path: Path, resume: bool | None = None):
        path = Path(path).resolve()
        if resume is None:
            resume = self.player.is_playing()
        self._player_media_path = path
        self.player.load(
            path,
            position_ms=int(self.model.playhead_ms),
            autoplay=bool(resume),
        )
        self.player.set_rate(self._current_playback_rate())

    def _player_backend_changed(self, name: str):
        if hasattr(self, "player_backend_label"):
            self.player_backend_label.setText("Player: " + str(name))
        if hasattr(self, "review_badge"):
            self.review_badge.setText(
                "MPV · MASTER DIRECT"
                if self.player.using_mpv
                else "QT FALLBACK · MASTER"
            )
        self._log("Playback backend: " + str(name))

    def _playback_rate_changed(self, *_):
        rate = self._current_playback_rate()

        # At very high review speed audio time-stretching is expensive.
        # Video still comes from the original master; no proxy is created.
        fast_visual = rate >= 3.0
        self.player.set_muted(fast_visual)
        self.player.set_rate(rate)
        if hasattr(self, "review_badge"):
            base = "MPV · MASTER DIRECT" if self.player.using_mpv else "QT FALLBACK · MASTER"
            self.review_badge.setText(
                base + (" · AUDIO OFF" if fast_visual else "")
            )
        self._log(
            f"Playback speed: {rate:g}x · master direct"
            + (" · audio preview off" if fast_visual else "")
        )

    def _load_frame_pts_window(self, center_ms: int, radius_ms: int = 2500) -> list[int]:
        if not self.model.source:
            return []
        center_ms = self.model.clamp(center_ms)
        if (
            self._frame_pts_cache
            and self._frame_pts_cache_start + 300 <= center_ms <= self._frame_pts_cache_end - 300
        ):
            return self._frame_pts_cache
        ffprobe = find_tool("ffprobe")
        if not ffprobe:
            return []
        start_ms = max(0, center_ms - radius_ms)
        end_ms = min(self.model.duration_ms, center_ms + radius_ms)
        points = probe_frame_timestamps(
            self.model.source,
            ffprobe,
            start_ms,
            max(start_ms + 100, end_ms),
        )
        self._frame_pts_cache = points
        self._frame_pts_cache_start = start_ms
        self._frame_pts_cache_end = end_ms
        return points

    def _analysis_failed(self, message: str):
        self.pending_load = None
        self.analyze_worker = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Analisis gagal.")
        QMessageBox.critical(self, APP_TITLE, "Analisis video gagal:\n" + message)

    # ---------- player ----------
    def _position_changed(self, ms: int):
        if self.timeline.isSliderDown():
            # Saat user sedang drag, jangan biarkan update player/keyframe
            # menimpa timecode di bawah kursor.
            return
        self.model.playhead_ms = int(ms)
        self.timeline.setValue(int(ms))
        self.position_label.setText(
            f"{clock_text(ms)} / {clock_text(self.model.duration_ms)}"
        )
        self.bridge_state = self.model.state()

    def _duration_changed(self, ms: int):
        if self.model.source and self.model.duration_ms <= 0 and ms > 0:
            self.model.duration_ms = int(ms)
            self.timeline.setRange(0, int(ms))
            self._refresh()

    def _playback_changed(self, playing: bool):
        self.play_btn.setText("⏸ Pause" if bool(playing) else "▶ Play")

    def _toggle_play(self):
        if not self.model.source:
            return
        if self.player.is_playing():
            self.player.pause()
        else:
            self.player.play()

    def _step_frame(self, direction: int):
        if not self.model.source:
            return
        self.player.pause()
        current = int(self.model.playhead_ms)
        try:
            points = self._load_frame_pts_window(current)
            if direction > 0:
                pos = bisect.bisect_right(points, current + 1)
                target = points[pos] if pos < len(points) else None
            else:
                pos = bisect.bisect_left(points, current - 1) - 1
                target = points[pos] if pos >= 0 else None

            if target is None:
                # Refresh with a wider window when stepping across cache edge.
                self._frame_pts_cache = []
                points = self._load_frame_pts_window(current, radius_ms=10_000)
                if direction > 0:
                    pos = bisect.bisect_right(points, current + 1)
                    target = points[pos] if pos < len(points) else None
                else:
                    pos = bisect.bisect_left(points, current - 1) - 1
                    target = points[pos] if pos >= 0 else None

            if target is not None:
                self.tool_seek(int(target))
                return
        except Exception as exc:
            self._log("Frame step PTS fallback: " + str(exc))

        fps = self.model.fps if self.model.fps > 0 else 25.0
        step = max(1, round(1000 / fps))
        self.tool_seek(current + direction * step)

    # ---------- manual operations ----------
    def _snapshot(self) -> dict:
        return {
            "cuts": [
                CutPoint(c.requested_ms, c.actual_ms)
                for c in self.model.cuts
            ],
            "dirty": bool(self.model.dirty),
        }

    def _restore_snapshot(self, snapshot: dict) -> None:
        cuts = snapshot.get("cuts") or []
        self.model.cuts = [
            CutPoint(c.requested_ms, c.actual_ms)
            for c in cuts
        ]
        self.model._normalize()
        self.model.dirty = bool(snapshot.get("dirty", False))
        self._refresh()

    def tool_transaction_snapshot(self) -> dict:
        return self._snapshot()

    def tool_transaction_restore(self, snapshot: dict) -> None:
        self._restore_snapshot(snapshot)

    def _manual_mutation(self, tool: str, args: dict):
        try:
            before = self._snapshot()
            self.registry.execute(tool, args)
            self.undo_stack.append(before)
            self._refresh()
        except Exception as exc:
            QMessageBox.warning(self, APP_TITLE, str(exc))

    def _remove_selected(self):
        row = self.parts_table.currentRow()
        if row < 0:
            return
        # Selecting Part-N removes the cut before the next part when possible.
        cut_index = min(row, len(self.model.cuts) - 1)
        if cut_index >= 0:
            self._manual_mutation("remove_cut", {"index": cut_index})

    def _divide_interval_clicked(self):
        try:
            interval = parse_time_ms(self.interval_edit.text())
            self._manual_mutation("divide_interval", {"interval_ms": interval})
        except Exception as exc:
            QMessageBox.warning(self, APP_TITLE, str(exc))

    # ---------- agent ----------
    def _agent_mode_changed(self):
        enabled = self.agent_mode.currentIndex() == 1
        for widget in (self.endpoint_edit, self.model_edit, self.api_key_edit):
            widget.setEnabled(enabled)

    def _make_plan(self):
        text = self.agent_input.toPlainText().strip()
        if not text:
            return
        self.pending_plan = None
        self.apply_btn.setEnabled(False)
        if self.agent_mode.currentIndex() == 0:
            try:
                self._set_plan(self.planner.local_plan(text))
            except Exception as exc:
                QMessageBox.warning(self, APP_TITLE, str(exc))
            return
        self.plan_btn.setEnabled(False)
        self.plan_preview.setPlainText("Menghubungi model…")
        self.agent_worker = AgentWorker(
            self.planner, self.endpoint_edit.text(), self.model_edit.text(),
            self.api_key_edit.text(), text, self.model.state()
        )
        self.agent_worker.ready.connect(self._remote_plan_ready)
        self.agent_worker.failed.connect(self._remote_plan_failed)
        self.agent_worker.start()

    def _remote_plan_ready(self, plan: dict):
        self.plan_btn.setEnabled(True)
        self.agent_worker = None
        self._set_plan(plan)

    def _remote_plan_failed(self, message: str):
        self.plan_btn.setEnabled(True)
        self.agent_worker = None
        self.plan_preview.setPlainText("Rencana gagal: " + message)

    def _set_plan(self, plan: dict):
        self.pending_plan = self.planner.validate(plan)
        self.plan_preview.setPlainText(json.dumps(self.pending_plan, ensure_ascii=False, indent=2))
        self.apply_btn.setEnabled(True)

    def _apply_plan(self):
        if not self.pending_plan:
            return
        before = self._snapshot()
        try:
            results = self.planner.apply(self.pending_plan)
            executed_steps = [item["step"] for item in results]
            if any(step["tool"] in MUTATING_TOOLS for step in executed_steps):
                self.undo_stack.append(before)
            self.plan_preview.setPlainText(json.dumps(
                {"plan": self.pending_plan, "results": results}, ensure_ascii=False, indent=2
            ))
            self._log("Agent menerapkan: " + self.pending_plan.get("summary", ""))
            self.pending_plan = None
            self.apply_btn.setEnabled(False)
            self._refresh()
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, "Agent berhenti karena error:\n" + str(exc))



    # ---------- Gemini API manager ----------
    def _current_gemini_model(self) -> str:
        if hasattr(self, "gemini_model_combo"):
            return self.gemini_model_combo.currentText().strip() or DEFAULT_MODEL
        return DEFAULT_MODEL

    @staticmethod
    def _quota_countdown_text(seconds: int) -> str:
        seconds = max(0, int(seconds or 0))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _refresh_active_quota_display(self):
        active_id = self.gemini_keys.active_id()
        model = self._current_gemini_model()
        if not active_id:
            quota = "RPM sisa —  ·  TPM sisa —  ·  RPD sisa —"
            detail = "Belum ada API key aktif."
            chat_api_text = "API aktif: — · Model: " + model
        else:
            try:
                snap = self.gemini_keys.snapshot(active_id, model)
                summary = self.gemini_keys.active_summary()
                quota = (
                    f"RPM sisa {snap['rpm_remaining']}/{snap['rpm_limit']}  ·  "
                    f"TPM sisa {snap['tpm_remaining']:,}/{snap['tpm_limit']:,}  ·  "
                    f"RPD sisa {snap['rpd_remaining']}/{snap['rpd_limit']}"
                )
                chat_api_text = (
                    f"API aktif: {summary.name if summary else 'Gemini API'} · Model: {model}"
                )
                detail = (
                    f"{summary.name if summary else 'Gemini API'} · {model} · "
                    f"RPM/TPM pulih otomatis setelah jendela 60 detik · "
                    f"Reset RPD dalam {self._quota_countdown_text(snap['rpd_reset_seconds'])} "
                    f"(Pacific). Angka sisa = catatan lokal MiniCut, bukan dashboard live Google."
                )
            except Exception as exc:
                quota = "RPM sisa ?  ·  TPM sisa ?  ·  RPD sisa ?"
                detail = "Tidak dapat membaca catatan kuota lokal: " + str(exc)
                chat_api_text = "API aktif: ? · Model: " + model

        if hasattr(self, "gemini_chat_api_label"):
            self.gemini_chat_api_label.setText(chat_api_text)
        if hasattr(self, "film_quota_label"):
            self.film_quota_label.setText(quota)
            self.film_quota_reset_label.setText(detail)
        if hasattr(self, "manager_quota_label"):
            self.manager_quota_label.setText(quota)
            self.manager_quota_reset_label.setText(detail)

    def _refresh_gemini_key_views(self):
        summaries = self.gemini_keys.summaries()
        active_id = self.gemini_keys.active_id()
        model = self._current_gemini_model()

        if hasattr(self, "gemini_key_combo"):
            self.gemini_key_combo.blockSignals(True)
            self.gemini_key_combo.clear()
            if summaries:
                active_index = 0
                for i, item in enumerate(summaries):
                    label = item.name
                    if item.project:
                        label += f" · {item.project}"
                    self.gemini_key_combo.addItem(label, item.id)
                    if item.id == active_id:
                        active_index = i
                self.gemini_key_combo.setCurrentIndex(active_index)
            else:
                self.gemini_key_combo.addItem("Belum ada API key", None)
            self.gemini_key_combo.blockSignals(False)

        if hasattr(self, "gemini_key_count_label"):
            self.gemini_key_count_label.setText(
                f"Tersimpan: {len(summaries)} / {MAX_GEMINI_KEYS} API key · model status: {model}"
            )

        if hasattr(self, "gemini_keys_table"):
            self.gemini_keys_table.setRowCount(len(summaries))
            for row, item in enumerate(summaries):
                snap = self.gemini_keys.snapshot(item.id, model)
                status = snap.get("status", "unknown")
                status_text = {
                    "ready": "🟢 SIAP",
                    "limited": "🔴 LIMIT",
                    "error": "🟠 ERROR",
                    "unknown": "⚪ BELUM",
                }.get(status, status.upper())
                api_name = item.name + (f" · {item.project}" if item.project else "")
                values = [
                    "●" if item.id == active_id else "",
                    api_name,
                    status_text,
                    f"{snap['rpm_remaining']}/{snap['rpm_limit']}",
                    f"{snap['tpm_remaining']:,}/{snap['tpm_limit']:,}",
                    f"{snap['rpd_remaining']}/{snap['rpd_limit']}",
                ]
                for col, value in enumerate(values):
                    cell = QTableWidgetItem(str(value))
                    if col == 0:
                        cell.setData(Qt.ItemDataRole.UserRole, item.id)
                    if col == 1:
                        cell.setToolTip(
                            f"Nama: {item.name}\n"
                            f"Project: {item.project or '-'}\n"
                            f"Key: {item.masked_key}"
                        )
                    elif col == 2:
                        cell.setToolTip(str(snap.get("last_error") or status_text))
                    self.gemini_keys_table.setItem(row, col, cell)

        self._refresh_active_quota_display()

    def _selected_gemini_key_id(self) -> str | None:
        if not hasattr(self, "gemini_keys_table"):
            return None
        row = self.gemini_keys_table.currentRow()
        if row < 0:
            return None
        item = self.gemini_keys_table.item(row, 0)
        if not item:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _film_key_changed(self, index: int):
        if not hasattr(self, "gemini_key_combo"):
            return
        key_id = self.gemini_key_combo.itemData(index)
        if key_id:
            try:
                self.gemini_keys.set_active(str(key_id))
            except Exception as exc:
                QMessageBox.warning(self, APP_TITLE, str(exc))
        self._refresh_gemini_key_views()

    def _open_gemini_manager(self):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "API":
                self.tabs.setCurrentIndex(i)
                break

    def _gemini_key_dialog(self, key_id: str | None = None):
        existing = None
        if key_id:
            existing = next((x for x in self.gemini_keys.summaries() if x.id == key_id), None)
            if not existing:
                QMessageBox.warning(self, APP_TITLE, "API key tidak ditemukan.")
                return

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Gemini API" if existing else "Tambah Gemini API")
        dialog.resize(520, 190)
        form = QFormLayout(dialog)

        name_edit = QLineEdit(existing.name if existing else "")
        project_edit = QLineEdit(existing.project if existing else "")
        key_edit = QLineEdit()
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setPlaceholderText(
            "Kosongkan jika tidak ingin mengganti key" if existing else "Tempel Gemini API key"
        )

        form.addRow("Nama", name_edit)
        form.addRow("Project", project_edit)
        form.addRow("API key", key_edit)

        note = QLabel(
            "API key disimpan terenkripsi untuk user Windows ini dan tidak dimasukkan ke proyek GitHub."
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
            if existing:
                self.gemini_keys.update(
                    existing.id,
                    name_edit.text(),
                    project_edit.text(),
                    key_edit.text() or None,
                )
            else:
                self.gemini_keys.add(
                    name_edit.text(),
                    project_edit.text(),
                    key_edit.text(),
                )
            self._refresh_gemini_key_views()
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))

    def _add_gemini_key(self):
        if self.gemini_keys.count() >= MAX_GEMINI_KEYS:
            QMessageBox.information(
                self, APP_TITLE, f"Batas {MAX_GEMINI_KEYS} API key sudah tercapai."
            )
            return
        self._gemini_key_dialog()

    def _edit_gemini_key(self):
        key_id = self._selected_gemini_key_id()
        if not key_id:
            QMessageBox.information(self, APP_TITLE, "Pilih satu API key di tabel.")
            return
        self._gemini_key_dialog(key_id)

    def _remove_gemini_key(self):
        key_id = self._selected_gemini_key_id()
        if not key_id:
            QMessageBox.information(self, APP_TITLE, "Pilih satu API key di tabel.")
            return
        summary = next((x for x in self.gemini_keys.summaries() if x.id == key_id), None)
        name = summary.name if summary else "API ini"
        answer = QMessageBox.question(
            self, APP_TITLE, f"Hapus {name} dari penyimpanan MiniCut?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.gemini_keys.remove(key_id)
            self._refresh_gemini_key_views()
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))

    def _activate_selected_gemini_key(self):
        key_id = self._selected_gemini_key_id()
        if not key_id:
            QMessageBox.information(self, APP_TITLE, "Pilih satu API key di tabel.")
            return
        try:
            self.gemini_keys.set_active(key_id)
            self._refresh_gemini_key_views()
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))

    def _begin_gemini_test(self, key_id: str):
        if self.gemini_chat_worker and self.gemini_chat_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Gemini Chat sedang berjalan.")
            return
        if self.film_cut_worker and self.film_cut_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "AI Film Cut sedang berjalan.")
            return
        if self.gemini_batch_worker and self.gemini_batch_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Cek semua API sedang berjalan.")
            return
        if self.gemini_test_worker and self.gemini_test_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Tes API sedang berjalan.")
            return
        try:
            key = self.gemini_keys.get_secret(key_id)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))
            return

        model = self._current_gemini_model()
        self._gemini_test_key_id = key_id
        self.gemini_test_btn.setEnabled(False)
        if hasattr(self, "gemini_test_selected_btn"):
            self.gemini_test_selected_btn.setEnabled(False)
        self.film_status_label.setText("Menguji Gemini API…")
        self.gemini_test_worker = GeminiTestWorker(key, model)
        self.gemini_test_worker.ready.connect(self._gemini_test_ready)
        self.gemini_test_worker.failed.connect(self._gemini_test_failed)
        self.gemini_test_worker.start()

    def _test_selected_gemini_key(self):
        key_id = self._selected_gemini_key_id()
        if not key_id:
            QMessageBox.information(self, APP_TITLE, "Pilih satu API key di tabel.")
            return
        self._begin_gemini_test(key_id)

    def _refresh_all_gemini_status(self):
        self._refresh_gemini_key_views()
        summaries = self.gemini_keys.summaries()
        model = self._current_gemini_model()
        ready = 0
        limited = 0
        error = 0
        unknown = 0
        for item in summaries:
            status = str(self.gemini_keys.snapshot(item.id, model).get("status") or "unknown")
            if status == "ready":
                ready += 1
            elif status == "limited":
                limited += 1
            elif status == "error":
                error += 1
            else:
                unknown += 1
        if hasattr(self, "gemini_batch_status_label"):
            self.gemini_batch_status_label.setText(
                f"Status lokal · SIAP {ready} · LIMIT {limited} · "
                f"ERROR {error} · BELUM DICEK {unknown} · total {len(summaries)}"
            )

    def _use_ready_gemini_key(self):
        model = self._current_gemini_model()
        candidates = []
        for item in self.gemini_keys.summaries():
            snap = self.gemini_keys.snapshot(item.id, model)
            if (
                snap.get("status") == "ready"
                and int(snap.get("rpm_remaining") or 0) > 0
                and int(snap.get("rpd_remaining") or 0) > 0
            ):
                candidates.append((item, snap))

        if not candidates:
            QMessageBox.information(
                self,
                APP_TITLE,
                "Belum ada API berstatus SIAP dengan sisa RPM/RPD lokal. "
                "Gunakan 'Cek Semua API Online' atau tunggu jendela RPM pulih.",
            )
            return

        # Manual one-click selection: prioritaskan sisa harian lalu sisa token menit.
        candidates.sort(
            key=lambda x: (
                int(x[1].get("rpd_remaining") or 0),
                int(x[1].get("tpm_remaining") or 0),
                int(x[1].get("rpm_remaining") or 0),
            ),
            reverse=True,
        )
        item, snap = candidates[0]
        self.gemini_keys.set_active(item.id)
        self._refresh_gemini_key_views()
        self.gemini_batch_status_label.setText(
            f"API aktif: {item.name} · RPD sisa lokal "
            f"{snap['rpd_remaining']}/{snap['rpd_limit']} · "
            f"RPM sisa {snap['rpm_remaining']}/{snap['rpm_limit']}."
        )

    def _set_batch_test_controls(self, running: bool):
        if hasattr(self, "gemini_test_all_btn"):
            self.gemini_test_all_btn.setEnabled(not running)
            self.gemini_cancel_all_btn.setEnabled(running)
            self.gemini_test_selected_btn.setEnabled(not running)
            self.gemini_refresh_all_btn.setEnabled(not running)
            self.gemini_use_ready_btn.setEnabled(not running)

    def _test_all_gemini_keys(self):
        if self.gemini_chat_worker and self.gemini_chat_worker.isRunning():
            QMessageBox.information(
                self, APP_TITLE, "Tunggu Gemini Chat selesai sebelum Cek Semua API."
            )
            return
        if self.gemini_test_worker and self.gemini_test_worker.isRunning():
            QMessageBox.information(
                self, APP_TITLE, "Tunggu Tes API selesai sebelum Cek Semua API."
            )
            return
        if self.film_cut_worker and self.film_cut_worker.isRunning():
            QMessageBox.information(
                self, APP_TITLE, "Tunggu AI Film Cut selesai sebelum Cek Semua API."
            )
            return
        if self.gemini_batch_worker and self.gemini_batch_worker.isRunning():
            return
        summaries = self.gemini_keys.summaries()
        if not summaries:
            QMessageBox.information(self, APP_TITLE, "Belum ada Gemini API key.")
            return

        keys: list[tuple[str, str, str]] = []
        unreadable = 0
        for item in summaries:
            try:
                keys.append((item.id, item.name, self.gemini_keys.get_secret(item.id)))
            except Exception as exc:
                unreadable += 1
                self.gemini_keys.mark_error(
                    item.id, self._current_gemini_model(), str(exc)
                )

        if not keys:
            self._refresh_gemini_key_views()
            QMessageBox.warning(self, APP_TITLE, "Tidak ada API key yang dapat dibaca.")
            return

        model = self._current_gemini_model()
        self.gemini_batch_status_label.setText(
            f"Memulai cek online {len(keys)} API untuk {model}. "
            "Tes dilakukan bertahap agar tidak membanjiri rate limit."
            + (f" · {unreadable} key tidak dapat dibaca." if unreadable else "")
        )
        self._set_batch_test_controls(True)
        self.gemini_batch_worker = GeminiBatchTestWorker(keys, model)
        self.gemini_batch_worker.progress_changed.connect(self._batch_test_progress)
        self.gemini_batch_worker.key_ready.connect(self._batch_test_ready)
        self.gemini_batch_worker.key_failed.connect(self._batch_test_failed)
        self.gemini_batch_worker.done.connect(self._batch_test_done)
        self.gemini_batch_worker.cancelled.connect(self._batch_test_cancelled)
        self.gemini_batch_worker.start()

    def _batch_test_progress(self, index: int, total: int, name: str):
        self.gemini_batch_status_label.setText(
            f"Cek semua API {index}/{total} · {name} · model {self._current_gemini_model()}"
        )

    def _batch_test_ready(self, key_id: str, result: dict):
        usage = result.get("usage") or {}
        model = str(result.get("model") or self._current_gemini_model())
        self.gemini_keys.record_usage(
            key_id,
            model,
            requests=int(usage.get("requests") or 0),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            status="ready",
            checked=True,
        )
        self._refresh_gemini_key_views()

    def _batch_test_failed(self, key_id: str, message: str):
        self.gemini_keys.mark_error(
            key_id, self._current_gemini_model(), message
        )
        self._refresh_gemini_key_views()

    def _batch_test_done(self, ok: int, failed: int):
        self.gemini_batch_worker = None
        self._set_batch_test_controls(False)
        self._refresh_gemini_key_views()
        self.gemini_batch_status_label.setText(
            f"Cek semua selesai · SIAP {ok} · gagal/limit {failed}. "
            "Klik 'Pakai API SIAP' untuk memilih salah satu yang siap."
        )

    def _cancel_all_gemini_tests(self):
        if self.gemini_batch_worker and self.gemini_batch_worker.isRunning():
            self.gemini_batch_worker.cancel()
            self.gemini_batch_status_label.setText(
                "Membatalkan cek semua API setelah tes aktif selesai…"
            )
            self.gemini_cancel_all_btn.setEnabled(False)

    def _batch_test_cancelled(self):
        self.gemini_batch_worker = None
        self._set_batch_test_controls(False)
        self._refresh_gemini_key_views()
        self.gemini_batch_status_label.setText("Cek semua API dibatalkan.")

    # ---------- Gemini Chat / manual frame cuts ----------
    def _append_gemini_chat(self, role: str, text: str):
        clean = str(text or "").strip()
        if not clean:
            return
        label = {
            "user": "Kamu",
            "assistant": "Gemini",
            "system": "MiniCut",
        }.get(role, role.title())
        if hasattr(self, "gemini_chat_history"):
            if self.gemini_chat_history.toPlainText().strip():
                self.gemini_chat_history.appendPlainText("")
            self.gemini_chat_history.appendPlainText(f"{label}: {clean}")
            bar = self.gemini_chat_history.verticalScrollBar()
            bar.setValue(bar.maximum())
        if role in {"user", "assistant"}:
            self._gemini_chat_history_data.append({
                "role": role,
                "text": clean,
            })
            self._gemini_chat_history_data = self._gemini_chat_history_data[-16:]

    def _clear_gemini_chat(self):
        if self._gemini_chat_busy():
            QMessageBox.information(
                self, APP_TITLE, "Tunggu proses Gemini/frame-lock selesai sebelum membersihkan chat."
            )
            return
        self._gemini_chat_history_data = []
        self._gemini_chat_pending_manual = []
        self._gemini_chat_deferred_actions = []
        if hasattr(self, "gemini_chat_history"):
            self.gemini_chat_history.clear()
        if hasattr(self, "gemini_chat_status"):
            self.gemini_chat_status.setText(
                "Chat dibersihkan. Timestamp manual tetap tidak mengubah cut yang sudah ada."
            )

    def _gemini_chat_busy(self) -> bool:
        return bool(
            (self.gemini_chat_worker and self.gemini_chat_worker.isRunning())
            or (self.manual_cut_worker and self.manual_cut_worker.isRunning())
        )

    def _refresh_gemini_chat_controls(self):
        if not hasattr(self, "gemini_chat_send_btn"):
            return
        busy = self._gemini_chat_busy()
        self.gemini_chat_send_btn.setEnabled(not busy)
        self.gemini_chat_clear_btn.setEnabled(not busy)
        self.gemini_chat_undo_btn.setEnabled(
            not (self.manual_cut_worker and self.manual_cut_worker.isRunning())
        )

    def _send_gemini_chat(self):
        text = self.gemini_chat_input.toPlainText().strip()
        if not text:
            return
        low = " ".join(text.lower().split())
        cancel_words = ("batalkan", "batal", "cancel", "stop", "hentikan")

        # Cancellation must remain available while a long-running process owns
        # Gemini/the export worker. These commands are deterministic local
        # controls, so they do not need another API request just to stop work.
        if self.film_cut_worker and self.film_cut_worker.isRunning():
            if any(word in low for word in cancel_words) and any(
                hint in low for hint in ("film cut", "ai film", "analisis film", "analisa film")
            ):
                self._append_gemini_chat("user", text)
                self.gemini_chat_input.clear()
                result = self.tool_cancel_film_cut()
                self._append_gemini_chat(
                    "system",
                    "Permintaan pembatalan AI Film Cut dikirim."
                    if result.get("cancel_requested")
                    else "AI Film Cut tidak sedang berjalan.",
                )
                return
            QMessageBox.information(
                self, APP_TITLE,
                "AI Film Cut sedang berjalan. Kamu tetap bisa mengetik 'batalkan AI Film Cut'."
            )
            return

        if self.export_worker and self.export_worker.isRunning():
            if any(word in low for word in cancel_words) and any(
                hint in low for hint in ("ekspor", "export", "render")
            ):
                self._append_gemini_chat("user", text)
                self.gemini_chat_input.clear()
                result = self.tool_cancel_export()
                self._append_gemini_chat(
                    "system",
                    "Permintaan pembatalan ekspor dikirim."
                    if result.get("cancel_requested")
                    else "Ekspor tidak sedang berjalan.",
                )
                return
            QMessageBox.information(
                self, APP_TITLE,
                "Ekspor sedang berjalan. Kamu tetap bisa mengetik 'batalkan ekspor'."
            )
            return

        if self.gemini_batch_worker and self.gemini_batch_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Cek Semua API sedang berjalan.")
            return
        if self.gemini_test_worker and self.gemini_test_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Tes API sedang berjalan.")
            return
        if self.gemini_chat_worker and self.gemini_chat_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Gemini Chat sedang menjawab.")
            return
        if self.manual_cut_worker and self.manual_cut_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Sedang mengunci timestamp ke frame master.")
            return

        timestamps = extract_manual_timestamps(text)
        manual_cut = bool(timestamps) and looks_like_manual_cut(text)
        locked = [
            {"raw": item.raw, "time_ms": item.time_ms, "time": item.time}
            for item in timestamps
        ]

        if manual_cut and not self.model.source:
            QMessageBox.warning(self, APP_TITLE, "Buka video terlebih dahulu sebelum memberi perintah cut.")
            return

        self._append_gemini_chat("user", text)
        self.gemini_chat_input.clear()
        self._gemini_chat_pending_manual = []
        self._gemini_chat_request_had_manual = manual_cut

        if manual_cut:
            locked_text = ", ".join(item["time"] for item in locked)
            self.gemini_chat_status.setText(
                "Timestamp terkunci: " + locked_text + " · frame-lock lokal dimulai…"
            )
            self._start_manual_frame_cuts(locked)

        key_id = self.gemini_keys.active_id()
        if not key_id:
            if manual_cut:
                self._append_gemini_chat(
                    "system",
                    "Belum ada API Gemini aktif. Cut manual tetap diproses lokal tanpa menunggu AI.",
                )
                self._refresh_gemini_chat_controls()
                return
            QMessageBox.warning(
                self, APP_TITLE, "Tambahkan Gemini API key terlebih dahulu."
            )
            self._open_gemini_manager()
            return

        try:
            secret = self.gemini_keys.get_secret(key_id)
        except Exception as exc:
            if manual_cut:
                self._append_gemini_chat(
                    "system",
                    "API Gemini tidak dapat dibaca, tetapi frame-lock manual tetap berjalan lokal.",
                )
                self._refresh_gemini_chat_controls()
                return
            QMessageBox.critical(self, APP_TITLE, str(exc))
            return

        self._gemini_chat_key_id = key_id
        self._gemini_chat_model = self._current_gemini_model()
        history_for_api = self._gemini_chat_history_data[:-1]
        self.gemini_chat_worker = GeminiChatWorker(
            secret,
            self._gemini_chat_model,
            text,
            self.model.state(),
            locked if manual_cut else [],
            history_for_api,
            self.registry.manifest(),
        )
        self.gemini_chat_worker.ready.connect(self._gemini_chat_ready)
        self.gemini_chat_worker.failed.connect(self._gemini_chat_failed)
        self.gemini_chat_worker.start()
        self._refresh_gemini_chat_controls()

    def _apply_gemini_chat_steps(self, steps: list[dict]):
        if not steps:
            return
        before = self._snapshot()
        try:
            plan = self.planner.validate({
                "version": 1,
                "summary": "Gemini Chat",
                "steps": steps,
            })
            results = self.planner.apply(plan)
            executed_steps = [item["step"] for item in results]
            if any(step["tool"] in MUTATING_TOOLS for step in executed_steps):
                self.undo_stack.append(before)
            names = ", ".join(step["tool"] for step in executed_steps)
            suffix = ""
            if len(executed_steps) < len(plan["steps"]):
                suffix = " · aksi berikutnya menunggu proses aktif selesai"
            self._append_gemini_chat(
                "system",
                "Aksi MiniCut diproses: " + names + suffix,
            )
            self._log("Gemini Chat menjalankan: " + names)
            self._refresh()
            return results
        except Exception as exc:
            self._append_gemini_chat(
                "system",
                "Aksi Gemini dihentikan karena error: " + str(exc),
            )
            return None

    def _run_deferred_gemini_chat_actions(self):
        steps = list(self._gemini_chat_deferred_actions)
        self._gemini_chat_deferred_actions = []
        if steps:
            self._apply_gemini_chat_steps(steps)

    def _execute_gemini_chat_actions(self, result: dict):
        raw_actions = result.get("actions") or []
        if not isinstance(raw_actions, list) or not raw_actions:
            return

        manual_items: list[dict] = []
        regular_steps: list[dict] = []
        manual_aliases = {"manual_frame_cut", "frame_cut", "add_cut"}

        for action in raw_actions[:12]:
            if not isinstance(action, dict):
                continue
            tool = str(action.get("tool") or "").strip()
            args = action.get("args")
            if not isinstance(args, dict):
                args = {}

            if tool in manual_aliases:
                # Explicit timestamps parsed locally are already running through
                # frame-lock. Ignore Gemini's duplicate interpretation.
                if self._gemini_chat_request_had_manual:
                    continue
                value = args.get("time_ms")
                try:
                    if isinstance(value, str) and value.strip().isdigit():
                        time_ms = int(value.strip())
                    else:
                        time_ms = parse_time_ms(value)
                except Exception:
                    self._append_gemini_chat(
                        "system",
                        f"Gemini memberi waktu cut yang tidak valid: {value!r}",
                    )
                    continue
                manual_items.append({
                    "raw": str(args.get("raw") or clock_text(time_ms)),
                    "time_ms": int(time_ms),
                    "time": clock_text(time_ms),
                })
                continue

            regular_steps.append({"tool": tool, "args": dict(args)})

        # If a frame-lock is part of the same request, actions after it are
        # deferred until the exact frame has been resolved and applied.
        frame_lock_active = bool(
            self.manual_cut_worker and self.manual_cut_worker.isRunning()
        )
        if manual_items:
            if not self.model.source:
                self._append_gemini_chat(
                    "system",
                    "Gemini meminta cut, tetapi belum ada video yang dibuka.",
                )
            elif not frame_lock_active:
                self._start_manual_frame_cuts(manual_items)
                frame_lock_active = bool(
                    self.manual_cut_worker and self.manual_cut_worker.isRunning()
                )

        if regular_steps:
            if frame_lock_active:
                self._gemini_chat_deferred_actions.extend(regular_steps)
                self._append_gemini_chat(
                    "system",
                    "Aksi lanjutan menunggu frame-lock selesai.",
                )
            else:
                self._apply_gemini_chat_steps(regular_steps)

    def _gemini_chat_ready(self, result: dict):
        self.gemini_chat_worker = None
        usage = result.get("usage") or {}
        if self._gemini_chat_key_id:
            self.gemini_keys.record_usage(
                self._gemini_chat_key_id,
                self._gemini_chat_model,
                requests=int(usage.get("requests") or 0),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                status="ready",
                checked=True,
            )
        reply = str(result.get("reply") or "").strip()
        if reply:
            self._append_gemini_chat("assistant", reply)
        self._execute_gemini_chat_actions(result)
        self._refresh_gemini_key_views()

        if not (self.manual_cut_worker and self.manual_cut_worker.isRunning()):
            self.gemini_chat_status.setText(
                f"Gemini siap · {self._gemini_chat_model}"
            )
        self._gemini_chat_request_had_manual = False
        self._refresh_gemini_chat_controls()

    def _gemini_chat_failed(self, message: str):
        self.gemini_chat_worker = None
        if self._gemini_chat_key_id:
            self.gemini_keys.mark_error(
                self._gemini_chat_key_id,
                self._gemini_chat_model,
                message,
            )
        self._gemini_chat_pending_manual = []
        self._refresh_gemini_key_views()

        if self._gemini_chat_request_had_manual:
            self._append_gemini_chat(
                "system",
                "Gemini sedang tidak tersedia/limit. Cut manual tidak dibatalkan karena "
                "frame-lock berjalan lokal dan timestamp tetap dikunci.",
            )
            if not (self.manual_cut_worker and self.manual_cut_worker.isRunning()):
                self.gemini_chat_status.setText("Gemini gagal · cut manual tetap lokal.")
        else:
            self.gemini_chat_status.setText("Gemini Chat gagal.")
            self._append_gemini_chat("system", "Gemini gagal: " + str(message))
        self._gemini_chat_request_had_manual = False
        self._refresh_gemini_chat_controls()

    def _start_manual_frame_cuts(self, locked: list[dict]):
        if not self.model.source:
            self._append_gemini_chat("system", "Buka video terlebih dahulu.")
            return
        ffprobe = find_tool("ffprobe")
        if not ffprobe:
            self._append_gemini_chat(
                "system",
                "ffprobe tidak ditemukan. Frame master belum bisa dikunci.",
            )
            return

        valid: list[dict] = []
        outside: list[str] = []
        for item in locked:
            ms = int(item.get("time_ms") or 0)
            if 0 < ms < self.model.duration_ms:
                valid.append(dict(item))
            else:
                outside.append(str(item.get("raw") or clock_text(ms)))
        if outside:
            self._append_gemini_chat(
                "system",
                "Timestamp di luar durasi video: " + ", ".join(outside),
            )
        if not valid:
            return

        self.gemini_chat_status.setText(
            f"Mengunci {len(valid)} timestamp ke frame master nyata…"
        )
        self.manual_cut_worker = ManualFrameCutWorker(
            self.model.source,
            ffprobe,
            valid,
        )
        self.manual_cut_worker.progress_changed.connect(self._manual_cut_progress)
        self.manual_cut_worker.ready.connect(self._manual_cut_ready)
        self.manual_cut_worker.failed.connect(self._manual_cut_failed)
        self.manual_cut_worker.start()
        self._refresh_gemini_chat_controls()

    def _manual_cut_progress(self, index: int, total: int, raw: str):
        self.gemini_chat_status.setText(
            f"Frame lock {index}/{total} · {raw}"
        )

    def _manual_cut_ready(self, resolved: list):
        self.manual_cut_worker = None
        before = self._snapshot()
        added = []
        skipped = []
        try:
            for item in resolved:
                actual_ms = int(item["time_ms"])
                if any(abs(c.actual_ms - actual_ms) < 2 for c in self.model.cuts):
                    skipped.append(item)
                    continue
                cut = self.model.add_frame_cut(
                    int(item["requested_ms"]),
                    actual_ms,
                )
                added.append((cut, item))
        except Exception as exc:
            self._restore_snapshot(before)
            self._append_gemini_chat(
                "system",
                "Cut manual dibatalkan karena frame lock gagal: " + str(exc),
            )
            if self._gemini_chat_deferred_actions:
                self._append_gemini_chat(
                    "system",
                    "Aksi lanjutan dibatalkan karena frame-lock tidak berhasil.",
                )
                self._gemini_chat_deferred_actions = []
            return

        if added:
            self.undo_stack.append(before)
            self._refresh()
            first_actual = int(added[0][0].actual_ms)
            self.player.set_position(first_actual)
            self.model.playhead_ms = first_actual

        lines = []
        for cut, item in added:
            delta = int(item.get("frame_delta_ms") or 0)
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"{clock_text(cut.requested_ms)} → {clock_text(cut.actual_ms)} "
                f"({sign}{delta} ms)"
            )
        if skipped:
            lines.append(f"{len(skipped)} titik dilewati karena frame cut sudah ada.")

        if lines:
            self._append_gemini_chat(
                "system",
                "Cut frame-accurate diterapkan:\n" + "\n".join(lines),
            )
        self.gemini_chat_status.setText(
            f"Selesai · {len(added)} cut dikunci ke PTS frame master."
        )
        self._refresh_gemini_chat_controls()
        self._run_deferred_gemini_chat_actions()

    def _manual_cut_failed(self, message: str):
        self.manual_cut_worker = None
        self.gemini_chat_status.setText("Frame lock manual gagal.")
        self._append_gemini_chat(
            "system",
            "Gagal mengunci timestamp ke frame master: " + str(message),
        )
        if self._gemini_chat_deferred_actions:
            self._append_gemini_chat(
                "system",
                "Aksi lanjutan dibatalkan karena bergantung pada frame-lock yang gagal.",
            )
            self._gemini_chat_deferred_actions = []
        self._refresh_gemini_chat_controls()

    # ---------- AI Film Cut / Gemini ----------
    def _choose_srt(self) -> bool:
        path, _ = QFileDialog.getOpenFileName(self, "Pilih subtitle SRT", "", "Subtitle (*.srt)")
        if not path:
            return False
        new_path = Path(path).resolve()
        changed = self.srt_path != new_path
        self.srt_path = new_path
        self.srt_edit.setText(str(self.srt_path))
        if changed:
            self.film_cut_results = []
            if hasattr(self, "film_table"):
                self.film_table.setRowCount(0)
            if hasattr(self, "film_apply_btn"):
                self.film_apply_btn.setEnabled(False)
            if hasattr(self, "film_preview_btn"):
                self.film_preview_btn.setEnabled(False)
        self.film_status_label.setText(
            "SRT siap. MiniCut akan menggunakannya untuk verifikasi dialog."
            + (" Hasil AI lama dibersihkan." if changed else "")
        )
        return True

    def _test_gemini(self):
        key_id = self.gemini_keys.active_id()
        if not key_id:
            QMessageBox.warning(self, APP_TITLE, "Tambahkan Gemini API key terlebih dahulu.")
            self._open_gemini_manager()
            return
        self._begin_gemini_test(key_id)

    def _gemini_test_ready(self, result: dict):
        self.gemini_test_btn.setEnabled(True)
        if hasattr(self, "gemini_test_selected_btn"):
            self.gemini_test_selected_btn.setEnabled(True)
        self.gemini_test_worker = None
        usage = result.get("usage") or {}
        model = str(result.get("model") or self._current_gemini_model())
        key_id = self._gemini_test_key_id
        if key_id:
            self.gemini_keys.record_usage(
                key_id,
                model,
                requests=int(usage.get("requests") or 0),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                status="ready",
                checked=True,
            )
        self._gemini_test_key_id = None
        self.film_usage_label.setText(
            f"TES API · Request {int(usage.get('requests') or 0)} · "
            f"Input {int(usage.get('prompt_tokens') or 0):,} · "
            f"Output {int(usage.get('output_tokens') or 0):,} · "
            f"Total {int(usage.get('total_tokens') or 0):,} token"
        )
        self.film_status_label.setText("Gemini terhubung · " + model)
        self._refresh_gemini_key_views()

    def _gemini_test_failed(self, message: str):
        self.gemini_test_btn.setEnabled(True)
        if hasattr(self, "gemini_test_selected_btn"):
            self.gemini_test_selected_btn.setEnabled(True)
        self.gemini_test_worker = None
        key_id = self._gemini_test_key_id
        if key_id:
            self.gemini_keys.mark_error(key_id, self._current_gemini_model(), message)
        self._gemini_test_key_id = None
        self.film_status_label.setText("Tes Gemini gagal.")
        self._refresh_gemini_key_views()
        QMessageBox.critical(self, APP_TITLE, "Gemini API gagal:\n" + message)

    def _start_film_cut(self):
        if self.gemini_chat_worker and self.gemini_chat_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Tunggu Gemini Chat selesai.")
            return
        if self.gemini_test_worker and self.gemini_test_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Tunggu Tes API selesai.")
            return
        if self.manual_cut_worker and self.manual_cut_worker.isRunning():
            QMessageBox.information(self, APP_TITLE, "Tunggu frame-lock manual selesai.")
            return
        if self.gemini_batch_worker and self.gemini_batch_worker.isRunning():
            QMessageBox.information(
                self, APP_TITLE, "Cek Semua API sedang berjalan. Tunggu atau batalkan dulu."
            )
            return
        if not self.model.source:
            QMessageBox.warning(self, APP_TITLE, "Buka video terlebih dahulu.")
            return
        if not self.srt_path or not self.srt_path.is_file():
            QMessageBox.warning(self, APP_TITLE, "Pilih file SRT yang sesuai dengan film.")
            return
        key_id = self.gemini_keys.active_id()
        if not key_id:
            QMessageBox.warning(self, APP_TITLE, "Tambahkan dan pilih Gemini API key terlebih dahulu.")
            self._open_gemini_manager()
            return
        try:
            key = self.gemini_keys.get_secret(key_id)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))
            return
        ffmpeg = find_tool("ffmpeg")
        ffprobe = find_tool("ffprobe")
        if not ffmpeg or not ffprobe:
            QMessageBox.critical(self, APP_TITLE, "FFmpeg/ffprobe tidak ditemukan.")
            return
        if self.film_cut_worker and self.film_cut_worker.isRunning():
            return

        self.film_cut_results = []
        self._film_active_key_id = key_id
        self._film_active_model = self._current_gemini_model()
        self._film_usage_seen_requests = 0
        self._film_usage_seen_prompt_tokens = 0
        self.film_table.setRowCount(0)
        self.film_apply_btn.setEnabled(False)
        self.film_preview_btn.setEnabled(False)
        self.film_analyze_btn.setEnabled(False)
        self.film_cancel_btn.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.film_cut_worker = FilmCutWorker(
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            source=self.model.source,
            duration_ms=self.model.duration_ms,
            srt_path=self.srt_path,
            api_key=key,
            model=self._film_active_model,
            interval_ms=self.film_interval.value() * 60_000,
            window_ms=self.film_window.value() * 60_000,
            top_n=6,
            use_cache=self.film_cache.isChecked(),
            allow_deep_check=self.film_deep_check.isChecked(),
        )
        self.film_cut_worker.progress_changed.connect(self._film_cut_progress)
        self.film_cut_worker.target_result.connect(self._film_cut_target_result)
        self.film_cut_worker.usage_changed.connect(self._film_cut_usage)
        self.film_cut_worker.done.connect(self._film_cut_done)
        self.film_cut_worker.failed.connect(self._film_cut_failed)
        self.film_cut_worker.cancelled.connect(self._film_cut_cancelled)
        self.film_status_label.setText(
            "Grid tetap 15/30/45/60… · contact sheet 1 frame/2 dtk + SRT · "
            "nearest-valid → video kandidat + SRT → exact frame."
        )
        self._log(
            "AI Film Cut: grid absolut → contact sheet+SRT → shortlist kandidat terdekat → "
            "video pendek+SRT → nearest-valid → frame PTS nyata."
        )
        self.film_cut_worker.start()

    def _film_cut_progress(self, index: int, total: int, stage: str):
        pct = int(((index - 1) / max(1, total)) * 100)
        self.progress.setValue(pct)
        self.status.setText(f"AI Film Cut {index}/{total} · {stage}")
        self.film_status_label.setText(f"Target {index}/{total} · {stage}")

    def _film_cut_target_result(self, result: dict):
        target_ms = int(result.get("target_ms") or 0)
        existing = next(
            (i for i, x in enumerate(self.film_cut_results)
             if int(x.get("target_ms") or 0) == target_ms),
            None,
        )
        if existing is None:
            self.film_cut_results.append(dict(result))
        else:
            self.film_cut_results[existing] = dict(result)
        self.film_cut_results.sort(key=lambda x: int(x.get("target_ms") or 0))

        # Update row bila target yang sama dikirim ulang (resume/retry), jangan duplikat.
        row = -1
        for i in range(self.film_table.rowCount()):
            item = self.film_table.item(i, 0)
            if item and int(item.data(Qt.ItemDataRole.UserRole) or -1) == target_ms:
                row = i
                break
        if row < 0:
            row = self.film_table.rowCount()
            self.film_table.insertRow(row)

        confidence = float(result.get("confidence") or 0)
        decision = str(result.get("decision") or result.get("broad_decision") or "").upper()
        selected_ms = int(result.get("selected_time_ms") or 0)
        no_cut = selected_ms <= 0 or decision in {
            "NO_CUT", "NO_CUT_MAX_EXPAND", "SKIPPED_AFTER_PREVIOUS_CUT"
        }
        review = bool(result.get("needs_review")) or (not no_cut and confidence < 0.55)

        if no_cut:
            semantic_time = "—"
            frame_time = "—"
            intent = "scene_continues"
            if decision == "SKIPPED_AFTER_PREVIOUS_CUT":
                status_text = "SKIP · GRID"
            else:
                status_text = "NO CUT"
        else:
            semantic_time = str(
                result.get("semantic_preferred_time")
                or result.get("candidate_time")
                or clock_text(int(result.get("semantic_preferred_ms") or 0))
            )
            frame_time = str(
                result.get("selected_time")
                or clock_text(selected_ms)
            )
            intent = str(result.get("cut_intent") or "scene_transition")
            frame_ok = bool(result.get("frame_verified"))
            if review:
                status_text = "REVIEW"
            elif result.get("cached"):
                status_text = "CACHE"
            elif result.get("deep_check_used"):
                status_text = "DEEP · OK"
            else:
                status_text = "SCENE · OK"
            if not frame_ok:
                status_text += " · NO LOCK"

        target_text = str(result.get("target") or clock_text(target_ms))
        reason = str(result.get("reason") or "")
        values = [
            target_text,
            frame_time,
            f"{confidence:.0%}",
            status_text,
            reason,
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, target_ms)
            if col == 1:
                item.setData(Qt.ItemDataRole.UserRole, selected_ms)
                item.setToolTip(
                    f"Batas AI: {semantic_time}\nIntent: {intent}\n"
                    f"Mode: {result.get('analysis_mode') or '-'}\n"
                    "Double-click untuk lompat ke frame ini."
                )
            elif col == 3:
                item.setToolTip(
                    f"Decision: {decision or '-'}\n"
                    f"Deep Check: {'ya' if result.get('deep_check_used') else 'tidak'}"
                )
            elif col == 4:
                item.setToolTip(reason)
            self.film_table.setItem(row, col, item)

        preview_marks = [
            int(x.get("selected_time_ms") or 0)
            for x in self.film_cut_results
            if int(x.get("selected_time_ms") or 0) > 0
        ]
        self.timeline.set_marks(preview_marks)
        if self.film_table.currentRow() < 0:
            self.film_table.selectRow(row)
        self._film_cut_selection_changed()

    def _film_cut_selection_changed(self):
        has_selection = self.film_table.currentRow() >= 0
        self.film_preview_btn.setEnabled(bool(has_selection and self.model.source))

    def _preview_selected_film_cut(self):
        row = self.film_table.currentRow()
        if row >= 0:
            self._preview_film_cut_row(row, 1)

    def _preview_film_cut_row(self, row: int, _column: int):
        cut_item = self.film_table.item(int(row), 1)
        target_item = self.film_table.item(int(row), 0)
        if not cut_item or not self.model.source:
            return

        preview_ms = int(cut_item.data(Qt.ItemDataRole.UserRole) or 0)
        preview_kind = "cut"
        if preview_ms <= 0 and target_item:
            preview_ms = int(target_item.data(Qt.ItemDataRole.UserRole) or 0)
            preview_kind = "target grid · NO CUT"
        if preview_ms <= 0:
            return

        self.player.pause()
        self.tool_seek(preview_ms)
        self.status.setText(
            f"Preview {preview_kind} · {clock_text(preview_ms)} · "
            "pilih hasil lain untuk membandingkan."
        )

    def _film_cut_usage(self, usage: dict):
        requests = int(usage.get("requests") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        delta_requests = max(0, requests - self._film_usage_seen_requests)
        delta_prompt_tokens = max(0, prompt_tokens - self._film_usage_seen_prompt_tokens)
        if self._film_active_key_id and (delta_requests or delta_prompt_tokens):
            self.gemini_keys.record_usage(
                self._film_active_key_id,
                self._film_active_model,
                requests=delta_requests,
                prompt_tokens=delta_prompt_tokens,
                status="ready",
            )
        self._film_usage_seen_requests = requests
        self._film_usage_seen_prompt_tokens = prompt_tokens
        output_tokens = int(usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        self.film_usage_label.setText(
            f"SESI SAAT INI · Request {requests} · "
            f"Input {prompt_tokens:,} · Output {output_tokens:,} · "
            f"Total {total_tokens:,} token"
        )
        self._refresh_gemini_key_views()

    def _film_cut_done(self, results: list):
        self.film_cut_worker = None
        self.film_analyze_btn.setEnabled(True)
        self.film_cancel_btn.setEnabled(False)
        self.progress.setValue(100)
        self.film_cut_results = sorted(
            [dict(x) for x in results], key=lambda x: int(x.get("target_ms") or 0)
        )
        review_count = sum(
            1 for x in self.film_cut_results
            if int(x.get("selected_time_ms") or 0) > 0
            and (
                bool(x.get("needs_review"))
                or float(x.get("confidence") or 0) < 0.55
            )
        )
        valid_cut_count = sum(
            1 for x in self.film_cut_results
            if int(x.get("selected_time_ms") or 0) > 0
        )
        self.film_apply_btn.setEnabled(valid_cut_count > 0)
        if review_count:
            self.film_status_label.setText(
                f"Selesai · {len(results)} titik · {review_count} perlu review sebelum diterapkan."
            )
        else:
            self.film_status_label.setText(
                f"Selesai · {valid_cut_count} cut natural dari {len(results)} target grid siap diterapkan."
            )
        self.status.setText("AI Film Cut selesai · belum diterapkan ke timeline.")
        if self._film_active_key_id:
            self.gemini_keys.record_usage(
                self._film_active_key_id,
                self._film_active_model,
                status="ready",
                checked=True,
            )
        self._refresh_gemini_key_views()
        self._log(f"AI Film Cut selesai: {len(results)} titik.")

    def _film_cut_failed(self, message: str):
        self.film_cut_worker = None
        self.film_analyze_btn.setEnabled(True)
        self.film_cancel_btn.setEnabled(False)
        self.status.setText("AI Film Cut gagal.")
        self.film_status_label.setText("Analisis berhenti. Hasil yang sudah selesai disimpan di cache.")
        lower = message.lower()
        limited_failure = (
            "429" in lower or "quota" in lower or "rate limit" in lower
        )
        if self._film_active_key_id and (
            "gemini" in lower or limited_failure
        ):
            self.gemini_keys.mark_error(
                self._film_active_key_id,
                self._film_active_model,
                message,
            )
        if limited_failure and hasattr(self, "gemini_batch_status_label"):
            self.gemini_batch_status_label.setText(
                "API aktif terkena LIMIT setelah retry/backoff. "
                "Buka tab Gemini API → Cek Semua API Online → Pakai API SIAP, "
                "lalu jalankan Analisis Film lagi. Cache hasil sebelumnya tetap ada."
            )
        self._refresh_gemini_key_views()
        QMessageBox.critical(self, APP_TITLE, "AI Film Cut gagal:\n" + message)

    def _film_cut_cancelled(self):
        self.film_cut_worker = None
        self.film_analyze_btn.setEnabled(True)
        self.film_cancel_btn.setEnabled(False)
        self.status.setText("AI Film Cut dibatalkan.")
        self.film_status_label.setText("Dibatalkan. Hasil sebelumnya tetap tersimpan di cache.")

    def _cancel_film_cut(self):
        if self.film_cut_worker and self.film_cut_worker.isRunning():
            self.film_cut_worker.cancel()
            self.film_cancel_btn.setEnabled(False)
            self.film_status_label.setText("Membatalkan setelah langkah aktif selesai…")

    def _apply_film_cut(self, *, record_undo: bool = True) -> dict:
        if not self.film_cut_results:
            return {"ok": False, "error": "Belum ada hasil AI Film Cut untuk diterapkan."}
        review_count = sum(
            1 for x in self.film_cut_results
            if int(x.get("selected_time_ms") or 0) > 0
            and (
                bool(x.get("needs_review"))
                or float(x.get("confidence") or 0) < 0.55
            )
        )
        if review_count:
            answer = QMessageBox.question(
                self,
                APP_TITLE,
                f"Ada {review_count} titik bertanda REVIEW. Tetap terapkan semua?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return {"ok": False, "cancelled": True}
        if self.model.cuts:
            answer = QMessageBox.question(
                self,
                APP_TITLE,
                "Timeline sudah mempunyai cut. Ganti dengan hasil AI Film Cut?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return {"ok": False, "cancelled": True}

        before = self._snapshot()
        exact = []
        for result in self.film_cut_results:
            t = int(result.get("selected_time_ms") or 0)
            if 0 < t < self.model.duration_ms:
                exact.append(CutPoint(t, t))
        exact.sort(key=lambda x: x.actual_ms)
        if not exact:
            QMessageBox.warning(self, APP_TITLE, "Tidak ada titik potong valid untuk diterapkan.")
            return {"ok": False, "error": "Tidak ada titik potong AI yang valid."}
        if record_undo:
            self.undo_stack.append(before)
        self.model.cuts = exact
        self.model._normalize()
        self.model.dirty = True
        self._refresh()
        self.film_apply_btn.setEnabled(False)
        self.film_status_label.setText(
            f"{len(exact)} titik AI diterapkan ke timeline. Timestamp dipertahankan presisi."
        )
        self._log(f"AI Film Cut diterapkan: {len(exact)} cut presisi.")
        return {"ok": True, "applied_cuts": len(exact)}

    # ---------- tool API ----------
    def _require_tool_media_ready(self) -> None:
        if self.analyze_worker and self.analyze_worker.isRunning():
            raise RuntimeError(
                "Video/proyek baru sedang dianalisis. Tunggu sampai proses load selesai."
            )
        if not self.model.source:
            raise ValueError("Belum ada video.")
        if not self.model.source.is_file():
            raise FileNotFoundError(
                "Video sumber sudah tidak ditemukan di lokasi semula. "
                "Buka kembali video/proyek dan pilih file sumber yang benar."
            )

    def tool_get_state(self):
        return {"ok": True, "state": self.model.state()}

    def tool_open_video(self):
        """Open the normal video picker from Gemini chat."""
        before = str(self.model.source) if self.model.source else None
        started = bool(self._choose_video())
        return {
            "ok": started,
            "cancelled": not started,
            "dialog": "open_video",
            "previous_source": before,
            "loading": started,
            "stop_plan": started,
        }

    def tool_open_project(self):
        """Open the normal MiniCut project picker from Gemini chat."""
        before = str(self.model.project_path) if self.model.project_path else None
        started = bool(self._choose_project())
        return {
            "ok": started,
            "cancelled": not started,
            "dialog": "open_project",
            "previous_project": before,
            "loading": started,
            "stop_plan": started,
        }

    def tool_choose_subtitle(self):
        """Open the SRT picker used by AI Film Cut."""
        selected = bool(self._choose_srt())
        return {
            "ok": selected,
            "cancelled": not selected,
            "subtitle": str(self.srt_path) if self.srt_path else None,
        }

    def tool_seek(self, time_ms):
        self._require_tool_media_ready()
        ms = self.model.clamp(parse_time_ms(time_ms))
        self.player.set_position(ms)
        self.model.playhead_ms = ms
        return {"ok": True, "playhead_ms": ms}

    def tool_play(self):
        self._require_tool_media_ready()
        self.player.play()
        return {"ok": True}

    def tool_pause(self):
        self._require_tool_media_ready()
        self.player.pause()
        return {"ok": True}

    def tool_set_playback_rate(self, rate):
        rate = float(rate)
        if not 0.25 <= rate <= 4.0:
            raise ValueError("Kecepatan playback harus antara 0.25x sampai 4x.")
        index = self.speed_combo.findData(rate)
        if index < 0:
            self.speed_combo.addItem(f"{rate:g}x", rate)
            index = self.speed_combo.findData(rate)
        self.speed_combo.setCurrentIndex(index)
        # currentIndexChanged normally applies the rate; call explicitly as a
        # deterministic fallback for headless/tool-driven use.
        self._playback_rate_changed()
        return {"ok": True, "rate": self._current_playback_rate()}

    def tool_step_frame(self, direction):
        self._require_tool_media_ready()
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError("direction harus -1 (mundur) atau 1 (maju).")
        before = int(self.model.playhead_ms)
        self._step_frame(direction)
        return {
            "ok": True,
            "direction": direction,
            "before_ms": before,
            "playhead_ms": int(self.model.playhead_ms),
        }

    def tool_add_cut(self, time_ms):
        self._require_tool_media_ready()
        cut = self.model.add_cut(parse_time_ms(time_ms))
        self._refresh()
        return {"ok": True, "cut": asdict(cut), "parts": len(self.model.cuts) + 1}

    def tool_remove_cut(self, index):
        self._require_tool_media_ready()
        cut = self.model.remove_cut(int(index))
        self._refresh()
        return {"ok": True, "removed": asdict(cut), "parts": len(self.model.cuts) + 1}

    def tool_clear_cuts(self):
        self._require_tool_media_ready()
        self.model.clear_cuts()
        self._refresh()
        return {"ok": True, "parts": 1}

    def tool_divide_equal(self, parts):
        self._require_tool_media_ready()
        self.model.divide_equal(int(parts))
        self._refresh()
        return {"ok": True, "parts": len(self.model.cuts) + 1}

    def tool_divide_interval(self, interval_ms):
        self._require_tool_media_ready()
        self.model.divide_interval(parse_time_ms(interval_ms) if isinstance(interval_ms, str) else int(interval_ms))
        self._refresh()
        return {"ok": True, "parts": len(self.model.cuts) + 1}

    def tool_start_film_cut(self):
        self._require_tool_media_ready()
        if self.film_cut_worker and self.film_cut_worker.isRunning():
            return {
                "ok": True,
                "started": False,
                "already_running": True,
                "stop_plan": True,
            }
        self._start_film_cut()
        running = bool(self.film_cut_worker and self.film_cut_worker.isRunning())
        return {
            "ok": running,
            "started": running,
            "stop_plan": running,
        }

    def tool_cancel_film_cut(self):
        running = bool(self.film_cut_worker and self.film_cut_worker.isRunning())
        self._cancel_film_cut()
        return {"ok": True, "cancel_requested": running}

    def tool_apply_film_cut(self):
        self._require_tool_media_ready()
        before = len(self.model.cuts)
        result = dict(self._apply_film_cut(record_undo=False) or {})
        after = len(self.model.cuts)
        result.update({
            "cuts_before": before,
            "cuts_after": after,
            "parts": len(self.model.cuts) + 1 if self.model.source else 0,
        })
        result.setdefault("ok", False)
        if result.get("ok"):
            result["stop_plan"] = True
        return result

    def tool_preview_film_cut(self, row):
        self._require_tool_media_ready()
        row = int(row)
        if row < 1 or row > self.film_table.rowCount():
            raise ValueError("Baris hasil AI Film Cut di luar rentang.")
        self._preview_film_cut_row(row - 1, 1)
        return {"ok": True, "row": row, "playhead_ms": int(self.model.playhead_ms)}

    def tool_set_export_mode(self, mode):
        mode = str(mode or "").strip().lower()
        if mode not in {"smartcut", "fast"}:
            raise ValueError("Mode ekspor harus 'smartcut' atau 'fast'.")
        index = self.export_mode.findData(mode)
        if index < 0:
            raise RuntimeError("Mode ekspor tidak tersedia di UI.")
        self.export_mode.setCurrentIndex(index)
        return {"ok": True, "mode": str(self.export_mode.currentData())}

    def tool_save_project(self):
        self._require_tool_media_ready()
        if not self.model.source:
            raise ValueError("Belum ada proyek.")
        path = self.model.project_path
        if not path:
            suggested = self.model.source.with_suffix(".minicut.json")
            selected, _ = QFileDialog.getSaveFileName(self, "Simpan proyek", str(suggested), "MiniCut JSON (*.json)")
            if not selected:
                return {"ok": False, "cancelled": True}
            path = Path(selected)
        self.model.save(path)
        self.status.setText("Proyek tersimpan · " + str(path))
        self._refresh()
        return {"ok": True, "path": str(path)}

    def tool_export_all(self):
        self._require_tool_media_ready()
        if not self.model.source:
            raise ValueError("Belum ada video.")
        if self.manual_cut_worker and self.manual_cut_worker.isRunning():
            raise RuntimeError("Tunggu frame-lock manual selesai sebelum ekspor.")
        if self.film_cut_worker and self.film_cut_worker.isRunning():
            raise RuntimeError("Tunggu AI Film Cut selesai sebelum ekspor.")
        if self.export_worker and self.export_worker.isRunning():
            raise RuntimeError("Ekspor sedang berjalan.")
        ffmpeg = find_tool("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg tidak ditemukan. Pastikan FFmpeg tersedia.")
        mode = str(self.export_mode.currentData() or "smartcut")
        if mode == "fast" and self.model.cuts:
            keyframes = set(int(x) for x in self.model.keyframes)
            has_exact_nonkey = any(
                int(cut.actual_ms) not in keyframes for cut in self.model.cuts
            )
            if has_exact_nonkey:
                answer = QMessageBox.question(
                    self,
                    APP_TITLE,
                    "Timeline memiliki cut frame-accurate yang bukan keyframe. "
                    "Fast Copy dapat menggeser titik potong. Tetap gunakan Fast Copy?",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return {"ok": False, "cancelled": True}
        smartcut_exe = None
        if mode == "smartcut":
            smartcut_exe = find_tool("MiniCut SmartCut") or find_tool("smartcut")
            if not smartcut_exe:
                raise RuntimeError(
                    "MiniCut SmartCut tidak ditemukan. Gunakan paket aplikasi lengkap "
                    "atau pilih Fast Copy."
                )
        parent = QFileDialog.getExistingDirectory(self, "Pilih folder hasil ekspor")
        if not parent:
            return {"ok": False, "cancelled": True}
        out_dir = Path(parent) / (self.model.source.stem + "_Parts")
        self.export_worker = ExportWorker(
            ffmpeg,
            self.model.source,
            out_dir,
            self.model.source.stem,
            [c.actual_ms for c in self.model.cuts],
            self.model.duration_ms,
            mode=mode,
            smartcut_exe=smartcut_exe,
        )
        self.export_worker.progress_changed.connect(self._export_progress)
        self.export_worker.log_line.connect(self._log)
        self.export_worker.done.connect(self._export_done)
        self.export_worker.failed.connect(self._export_failed)
        self.export_worker.cancelled.connect(self._export_cancelled)
        self.progress.setValue(0)
        self.status.setText(
            "SmartCut frame-accurate…" if mode == "smartcut" else "Fast Copy…"
        )
        self._log(
            "Mode ekspor: SmartCut frame-accurate"
            if mode == "smartcut" else "Mode ekspor: Fast Copy keyframe"
        )
        self.export_worker.start()
        return {
            "ok": True,
            "started": True,
            "mode": mode,
            "output_dir": str(out_dir),
            "stop_plan": True,
        }

    def tool_cancel_export(self):
        running = bool(self.export_worker and self.export_worker.isRunning())
        if running:
            self.export_worker.cancel()
            self.status.setText("Membatalkan ekspor setelah proses aktif selesai…")
        return {"ok": True, "cancel_requested": running}

    def tool_undo(self):
        self._require_tool_media_ready()
        if not self.undo_stack:
            return {"ok": False, "error": "Belum ada perubahan yang bisa di-undo."}
        snapshot = self.undo_stack.pop()
        self._restore_snapshot(snapshot)
        return {
            "ok": True,
            "parts": len(self.model.cuts) + 1,
            "dirty": bool(self.model.dirty),
        }

    def ui_layout_issues(self) -> list[str]:
        """Return obvious overlap/clipping problems at the current window size.

        This is intentionally conservative and checks interactive controls only.
        It is used by the packaged Windows smoke test so UI regressions fail CI.
        """
        issues: list[str] = []
        original_tab = self.tabs.currentIndex() if hasattr(self, "tabs") else -1

        def rect_in_window(widget):
            top_left = widget.mapTo(self, widget.rect().topLeft())
            return widget.rect().translated(top_left)

        def check_visible_controls(scope_name: str, root: QWidget):
            control_types = (QPushButton, QComboBox, QLineEdit, QSpinBox)
            controls = [
                w for w in root.findChildren(QWidget)
                if isinstance(w, control_types)
                and w.isVisible()
                and w.width() > 0
                and w.height() > 0
            ]
            for widget in controls:
                if isinstance(widget, QPushButton):
                    need = widget.sizeHint().width()
                    if widget.width() + 4 < need:
                        issues.append(
                            f"{scope_name}: tombol '{widget.text()}' terlalu sempit "
                            f"({widget.width()} < {need})"
                        )
            for i, first in enumerate(controls):
                r1 = rect_in_window(first)
                for second in controls[i + 1:]:
                    # Parent/child controls are not peers and may legitimately
                    # share global area; only compare independent controls.
                    if first.isAncestorOf(second) or second.isAncestorOf(first):
                        continue
                    r2 = rect_in_window(second)
                    inter = r1.intersected(r2)
                    if inter.width() > 3 and inter.height() > 3:
                        issues.append(
                            f"{scope_name}: kontrol tumpang tindih "
                            f"'{getattr(first, 'text', lambda: first.objectName())()}' dan "
                            f"'{getattr(second, 'text', lambda: second.objectName())()}'"
                        )

        QApplication.processEvents()
        check_visible_controls("workspace", self.centralWidget())
        if hasattr(self, "tabs"):
            for index in range(self.tabs.count()):
                self.tabs.setCurrentIndex(index)
                QApplication.processEvents()
                check_visible_controls(
                    f"tab {self.tabs.tabText(index)}",
                    self.tabs.widget(index),
                )
            if original_tab >= 0:
                self.tabs.setCurrentIndex(original_tab)
                QApplication.processEvents()
        return sorted(set(issues))

    # ---------- export ----------
    def _export_progress(self, pct: int, text: str):
        self.progress.setValue(pct)
        self.status.setText(f"Ekspor {pct}% · {text}")

    def _export_done(self, out_dir: str, count: int, size: int, elapsed: float):
        self.progress.setValue(100)
        self.status.setText(f"Selesai · {count} part · {elapsed:.1f}s")
        self._log(f"Ekspor selesai: {count} file, {size / 1024 / 1024:.1f} MiB → {out_dir}")
        self.export_worker = None
        QMessageBox.information(self, APP_TITLE, f"Ekspor selesai.\n{count} part\n{out_dir}")

    def _export_failed(self, message: str):
        self.status.setText("Ekspor gagal.")
        self.export_worker = None
        QMessageBox.critical(self, APP_TITLE, "Ekspor gagal:\n" + message)

    def _export_cancelled(self):
        self.status.setText("Ekspor dibatalkan.")
        self.export_worker = None

    # ---------- bridge ----------
    def _drain_bridge(self):
        self.bridge_state = self.model.state()
        for _ in range(20):
            try:
                call = self.bridge_queue.get_nowait()
            except queue.Empty:
                break
            try:
                before = self._snapshot()
                result = self.registry.execute(call.tool, call.args)
                if (
                    call.tool in MUTATING_TOOLS
                    and isinstance(result, dict)
                    and result.get("ok", True)
                ):
                    self.undo_stack.append(before)
                call.result.update(result)
                call.result.setdefault("ok", True)
                self._refresh()
            except Exception as exc:
                call.result.update({"ok": False, "error": str(exc)})
            finally:
                call.event.set()

    # ---------- refresh/log ----------
    def _refresh(self):
        state = self.model.state()
        self.bridge_state = state
        self.timeline.set_marks([c.actual_ms for c in self.model.cuts])
        self.position_label.setText(
            f"{clock_text(self.model.playhead_ms)} / {clock_text(self.model.duration_ms)}"
        )
        if not self.model.source:
            self.info_label.setText("Belum ada video.")
            self.agent_state.setText("Belum ada timeline untuk dikontrol agent.")
            if hasattr(self, "media_name_label"):
                self.media_name_label.setText("No media loaded")
                self.media_meta_label.setText("Drag & drop video di sini")
            self.parts_table.setRowCount(0)
            return
        self.info_label.setText(
            f"{self.model.source.name} · {clock_text(self.model.duration_ms)} · "
            f"{self.model.fps:.3f} fps · {len(self.model.cuts) + 1} part"
        )
        if hasattr(self, "media_name_label"):
            self.media_name_label.setText(self.model.source.name)
            self.media_meta_label.setText(
                f"{clock_text(self.model.duration_ms)}\n"
                f"{self.model.fps:.3f} fps · {len(self.model.cuts) + 1} part\n"
                f"Master tetap dipakai untuk export."
            )
        self.agent_state.setText(
            f"{self.model.source.name} · {len(self.model.cuts) + 1} part · "
            f"playhead {clock_text(self.model.playhead_ms)} · "
            f"keyframe {'siap' if self.model.keyframes else 'belum'}"
        )
        ranges = self.model.part_ranges()
        self.parts_table.setRowCount(len(ranges))
        for row, (start, end) in enumerate(ranges):
            values = [f"Part-{row + 1:02d}", clock_text(start), clock_text(end), clock_text(end - start)]
            for col, value in enumerate(values):
                self.parts_table.setItem(row, col, QTableWidgetItem(value))

    def _log(self, text: str):
        self.log.appendPlainText(text)

    # ---------- drag/drop/close ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() == ".json":
            try:
                data, source = load_project_file(path)
                if not source:
                    chosen, _ = QFileDialog.getOpenFileName(
                        self,
                        "Video sumber proyek tidak ditemukan. Cari video sumber",
                        str(path.parent),
                        "Video (*.*)",
                    )
                    if not chosen:
                        return
                    source = Path(chosen)
                self._begin_load(source, data, path)
            except Exception as exc:
                QMessageBox.warning(self, APP_TITLE, str(exc))
        elif path.suffix.lower() in SUPPORTED_VIDEO:
            self._begin_load(path)

    def closeEvent(self, event):
        # Jangan hancurkan QThread yang masih aktif. Network worker tidak dipaksa
        # terminate karena itu lebih berisiko daripada menunggu request selesai.
        blocking = []
        for name, worker in (
            ("Analisis video", self.analyze_worker),
            ("AI Agent", self.agent_worker),
            ("Tes Gemini", self.gemini_test_worker),
            ("Gemini Chat", self.gemini_chat_worker),
            ("Frame-lock manual", self.manual_cut_worker),
        ):
            if worker and worker.isRunning():
                blocking.append(name)
        if blocking:
            QMessageBox.information(
                self,
                APP_TITLE,
                "Tunggu proses berikut selesai sebelum keluar:\n" + " · ".join(blocking),
            )
            event.ignore()
            return

        if self.gemini_batch_worker and self.gemini_batch_worker.isRunning():
            self.gemini_batch_worker.cancel()
            self.gemini_batch_worker.wait(5000)
            if self.gemini_batch_worker.isRunning():
                QMessageBox.information(
                    self,
                    APP_TITLE,
                    "Cek Semua API sedang dihentikan. Coba tutup lagi beberapa saat.",
                )
                event.ignore()
                return

        if self.film_cut_worker and self.film_cut_worker.isRunning():
            answer = QMessageBox.question(
                self, APP_TITLE, "AI Film Cut masih berjalan. Batalkan proses?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.film_cut_worker.cancel()
            self.film_cut_worker.wait(5000)
            if self.film_cut_worker.isRunning():
                QMessageBox.information(
                    self,
                    APP_TITLE,
                    "AI Film Cut sedang menunggu request aktif selesai. "
                    "Coba tutup lagi beberapa saat.",
                )
                event.ignore()
                return

        if self.export_worker and self.export_worker.isRunning():
            answer = QMessageBox.question(
                self, APP_TITLE, "Ekspor masih berjalan. Batalkan ekspor?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.export_worker.cancel()
            self.export_worker.wait(5000)
            if self.export_worker.isRunning():
                QMessageBox.information(
                    self,
                    APP_TITLE,
                    "Ekspor sedang dihentikan. Coba tutup lagi beberapa saat.",
                )
                event.ignore()
                return

        if self.model.dirty:
            answer = QMessageBox.question(
                self, APP_TITLE, "Perubahan cut belum disimpan. Tetap keluar?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        self.bridge.stop()
        self.player.shutdown()
        event.accept()
