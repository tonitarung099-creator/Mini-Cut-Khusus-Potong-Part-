from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt, Signal, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QStackedLayout, QWidget


def _runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _prepare_windows_dll_search() -> None:
    if os.name != "nt":
        return
    root = _runtime_dir()
    os.environ["PATH"] = str(root) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(str(root))
    except (AttributeError, FileNotFoundError, OSError):
        pass


class PreviewPlayer(QObject):
    """Master-direct preview player.

    mpv/libmpv is the preferred backend because it is designed for embedding,
    hardware decoding and accurate seeking. Qt Multimedia stays as a fallback
    so the application can still open if libmpv is unavailable on a machine.
    """

    positionChanged = Signal(int)
    durationChanged = Signal(int)
    playbackChanged = Signal(bool)
    errorOccurred = Signal(str)
    backendChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.widget = QWidget(parent)
        self.widget.setObjectName("VideoSurface")

        self._stack = QStackedLayout(self.widget)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._mpv_surface = QWidget(self.widget)
        self._mpv_surface.setObjectName("MpvSurface")
        self._mpv_surface.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self._mpv_surface.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)

        self._qt_surface = QVideoWidget(self.widget)
        self._qt_surface.setObjectName("QtVideoSurface")

        self._stack.addWidget(self._mpv_surface)
        self._stack.addWidget(self._qt_surface)

        self._qt_audio = QAudioOutput(self)
        self._qt = QMediaPlayer(self)
        self._qt.setAudioOutput(self._qt_audio)
        self._qt.setVideoOutput(self._qt_surface)
        self._qt.positionChanged.connect(self._qt_position)
        self._qt.durationChanged.connect(self._qt_duration)
        self._qt.playbackStateChanged.connect(self._qt_state)
        self._qt.errorOccurred.connect(
            lambda _error, text: self.errorOccurred.emit("Qt fallback: " + str(text))
        )

        self._mpv = None
        self._backend = "qt"
        self._source: Path | None = None
        self._rate = 1.0
        self._muted = False
        self._last_position = 0
        self._last_duration = 0
        self._last_playing = False
        self._pending_position = 0
        self._pending_autoplay = False

        self._stack.setCurrentWidget(self._qt_surface)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(33)
        self._poll_timer.timeout.connect(self._poll_mpv)
        self._poll_timer.start()

        QTimer.singleShot(0, self._try_init_mpv)

    @property
    def backend_name(self) -> str:
        return "mpv · Master Direct" if self._backend == "mpv" else "Qt · Master Direct"

    @property
    def using_mpv(self) -> bool:
        return self._backend == "mpv" and self._mpv is not None

    def _try_init_mpv(self) -> None:
        try:
            _prepare_windows_dll_search()
            import mpv  # type: ignore

            wid = str(int(self._mpv_surface.winId()))
            self._mpv = mpv.MPV(
                wid=wid,
                vo="gpu-next",
                hwdec="auto-safe",
                keep_open="yes",
                idle="yes",
                osc=False,
                input_default_bindings=False,
                input_vo_keyboard=False,
                audio_pitch_correction="yes",
                video_sync="audio",
                hr_seek="yes",
                hr_seek_framedrop="yes",
                cache="yes",
                demuxer_readahead_secs="3",
                demuxer_max_bytes="256MiB",
                demuxer_max_back_bytes="64MiB",
                log_handler=self._mpv_log,
            )
            self._backend = "mpv"
            self._stack.setCurrentWidget(self._mpv_surface)
            self.backendChanged.emit(self.backend_name)

            if self._source:
                source = self._source
                pos = self._pending_position
                autoplay = self._pending_autoplay
                self._qt.stop()
                self._load_mpv(source, pos, autoplay)
        except Exception as exc:
            self._mpv = None
            self._backend = "qt"
            self._stack.setCurrentWidget(self._qt_surface)
            self.backendChanged.emit(self.backend_name)
            self.errorOccurred.emit(
                "libmpv tidak tersedia; memakai Qt fallback. " + str(exc)
            )

    def _mpv_log(self, level, component, message) -> None:
        if str(level).lower() in {"fatal", "error"}:
            self.errorOccurred.emit(f"mpv/{component}: {str(message).strip()}")

    def load(
        self,
        path: Path,
        position_ms: int = 0,
        autoplay: bool = False,
    ) -> None:
        self._source = Path(path).resolve()
        self._pending_position = max(0, int(position_ms))
        self._pending_autoplay = bool(autoplay)
        if self.using_mpv:
            self._load_mpv(self._source, self._pending_position, autoplay)
        else:
            self._load_qt(self._source, self._pending_position, autoplay)

    def _load_mpv(self, path: Path, position_ms: int, autoplay: bool) -> None:
        assert self._mpv is not None
        try:
            self._mpv.command("loadfile", str(path), "replace")
            self._mpv.pause = not autoplay
            self._mpv.speed = float(self._rate)
            self._mpv.mute = bool(self._muted)
            if position_ms > 0:
                QTimer.singleShot(
                    80, lambda ms=position_ms: self.set_position(ms)
                )
        except Exception as exc:
            self.errorOccurred.emit("mpv load: " + str(exc))

    def _load_qt(self, path: Path, position_ms: int, autoplay: bool) -> None:
        self._stack.setCurrentWidget(self._qt_surface)
        self._qt.setSource(QUrl.fromLocalFile(str(path)))
        self._qt.setPlaybackRate(self._rate)
        self._qt_audio.setMuted(self._muted)

        def apply_state():
            self._qt.setPosition(position_ms)
            self._qt.setPlaybackRate(self._rate)
            if autoplay:
                self._qt.play()
            else:
                self._qt.pause()

        QTimer.singleShot(180, apply_state)

    def play(self) -> None:
        if self.using_mpv:
            try:
                self._mpv.pause = False
            except Exception as exc:
                self.errorOccurred.emit(str(exc))
        else:
            self._qt.play()

    def pause(self) -> None:
        if self.using_mpv:
            try:
                self._mpv.pause = True
            except Exception as exc:
                self.errorOccurred.emit(str(exc))
        else:
            self._qt.pause()

    def stop(self) -> None:
        if self.using_mpv:
            try:
                self._mpv.command("stop")
            except Exception:
                pass
        self._qt.stop()

    def shutdown(self) -> None:
        self.stop()
        if self._mpv is not None:
            try:
                self._mpv.terminate()
            except Exception:
                pass
            self._mpv = None

    def is_playing(self) -> bool:
        if self.using_mpv:
            try:
                return bool(self._source) and not bool(self._mpv.pause)
            except Exception:
                return False
        return self._qt.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def set_position(self, ms: int, exact: bool = True) -> None:
        ms = max(0, int(ms))
        self._last_position = ms
        if self.using_mpv:
            try:
                mode = "absolute+exact" if exact else "absolute+keyframes"
                self._mpv.command("seek", f"{ms / 1000:.6f}", mode)
                self.positionChanged.emit(ms)
            except Exception as exc:
                self.errorOccurred.emit("mpv seek: " + str(exc))
        else:
            self._qt.setPosition(ms)

    def position(self) -> int:
        if self.using_mpv:
            return int(self._last_position)
        return int(self._qt.position())

    def set_rate(self, rate: float) -> None:
        self._rate = max(0.05, float(rate))
        if self.using_mpv:
            try:
                self._mpv.speed = self._rate
            except Exception as exc:
                self.errorOccurred.emit("mpv speed: " + str(exc))
        else:
            self._qt.setPlaybackRate(self._rate)

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if self.using_mpv:
            try:
                self._mpv.mute = self._muted
            except Exception:
                pass
        self._qt_audio.setMuted(self._muted)

    def _poll_mpv(self) -> None:
        if not self.using_mpv:
            return
        try:
            pos = self._mpv.time_pos
            duration = self._mpv.duration
            paused = bool(self._mpv.pause)
        except Exception:
            return

        if pos is not None:
            ms = max(0, int(float(pos) * 1000))
            if abs(ms - self._last_position) >= 8:
                self._last_position = ms
                self.positionChanged.emit(ms)

        if duration is not None:
            dur_ms = max(0, int(float(duration) * 1000))
            if abs(dur_ms - self._last_duration) >= 10:
                self._last_duration = dur_ms
                self.durationChanged.emit(dur_ms)

        playing = bool(self._source) and not paused
        if playing != self._last_playing:
            self._last_playing = playing
            self.playbackChanged.emit(playing)

    def _qt_position(self, ms: int) -> None:
        if self._backend != "qt":
            return
        self._last_position = int(ms)
        self.positionChanged.emit(int(ms))

    def _qt_duration(self, ms: int) -> None:
        if self._backend != "qt":
            return
        self._last_duration = int(ms)
        self.durationChanged.emit(int(ms))

    def _qt_state(self, state) -> None:
        if self._backend != "qt":
            return
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._last_playing = playing
        self.playbackChanged.emit(playing)
