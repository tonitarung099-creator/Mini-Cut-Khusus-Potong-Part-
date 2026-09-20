from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .candidates import find_candidates_for_target, target_times
from .core import build_preview_proxy, export_segments, export_segments_smartcut, probe_keyframes, probe_media
from .gemini import GeminiClient
from .frame_resolver import resolve_semantic_frame
from .subtitles import SubtitleTrack, format_ms

class AnalyzeWorker(QThread):
    ready = Signal(dict, list)
    failed = Signal(str)

    def __init__(self, ffprobe: str, source: Path):
        super().__init__()
        self.ffprobe = ffprobe
        self.source = source

    def run(self):
        try:
            metadata = probe_media(self.source, self.ffprobe)
            keyframes = probe_keyframes(self.source, self.ffprobe)
            self.ready.emit(metadata, keyframes)
        except Exception as exc:
            self.failed.emit(str(exc))


class ProxyWorker(QThread):
    progress_changed = Signal(int, str)
    log_line = Signal(str)
    ready = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        ffmpeg: str,
        source: Path,
        output: Path,
        duration_ms: int,
        source_height: int,
        fps: float,
    ):
        super().__init__()
        self.ffmpeg = ffmpeg
        self.source = source
        self.output = output
        self.duration_ms = int(duration_ms)
        self.source_height = int(source_height or 0)
        self.fps = float(fps or 0.0)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            path = build_preview_proxy(
                self.ffmpeg,
                self.source,
                self.output,
                self.duration_ms,
                source_height=self.source_height,
                fps=self.fps,
                progress=lambda p, t: self.progress_changed.emit(p, t),
                log=lambda s: self.log_line.emit(s),
                cancelled=lambda: self._cancel,
            )
            self.ready.emit(str(path))
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportWorker(QThread):
    progress_changed = Signal(int, str)
    log_line = Signal(str)
    done = Signal(str, int, int, float)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, ffmpeg: str, source: Path, output_dir: Path, base_name: str, cuts: list[int], duration_ms: int, mode: str = "fast", smartcut_exe: str | None = None):
        super().__init__()
        self.ffmpeg = ffmpeg
        self.source = source
        self.output_dir = output_dir
        self.base_name = base_name
        self.cuts = list(cuts)
        self.duration_ms = duration_ms
        self.mode = mode
        self.smartcut_exe = smartcut_exe
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        started = time.time()
        try:
            if self.mode == "smartcut":
                if not self.smartcut_exe:
                    raise RuntimeError("MiniCut SmartCut tidak ditemukan.")
                count, size = export_segments_smartcut(
                    self.smartcut_exe,
                    self.source,
                    self.output_dir,
                    self.base_name,
                    self.cuts,
                    self.duration_ms,
                    progress=lambda p, t: self.progress_changed.emit(p, t),
                    log=lambda s: self.log_line.emit(s),
                    cancelled=lambda: self._cancel,
                )
            else:
                count, size = export_segments(
                    self.ffmpeg,
                    self.source,
                    self.output_dir,
                    self.base_name,
                    self.cuts,
                    self.duration_ms,
                    progress=lambda p, t: self.progress_changed.emit(p, t),
                    log=lambda s: self.log_line.emit(s),
                    cancelled=lambda: self._cancel,
                )
            self.done.emit(str(self.output_dir), count, size, time.time() - started)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

class AgentWorker(QThread):
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, planner, endpoint: str, model: str, api_key: str, text: str, state: dict):
        super().__init__()
        self.planner = planner
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.text = text
        self.state = state

    def run(self):
        try:
            self.ready.emit(self.planner.remote_plan(self.endpoint, self.model, self.api_key, self.text, self.state))
        except Exception as exc:
            self.failed.emit(str(exc))

class GeminiTestWorker(QThread):
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.model = model

    def run(self):
        try:
            self.ready.emit(GeminiClient(self.api_key, self.model).test_connection())
        except Exception as exc:
            self.failed.emit(str(exc))

class FilmCutWorker(QThread):
    progress_changed = Signal(int, int, str)
    target_result = Signal(dict)
    usage_changed = Signal(dict)
    done = Signal(list)
    failed = Signal(str)
    cancelled = Signal()

    CACHE_VERSION = 4

    def __init__(
        self,
        ffmpeg: str,
        ffprobe: str,
        source: Path,
        duration_ms: int,
        srt_path: Path,
        api_key: str,
        model: str,
        interval_ms: int = 15 * 60_000,
        window_ms: int = 2 * 60_000,
        top_n: int = 3,
        use_cache: bool = True,
    ):
        super().__init__()
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.source = source
        self.duration_ms = int(duration_ms)
        self.srt_path = srt_path
        self.api_key = api_key
        self.model = model
        self.interval_ms = int(interval_ms)
        self.window_ms = int(window_ms)
        self.top_n = int(top_n)
        self.use_cache = bool(use_cache)
        self._cancel = False

    @property
    def cache_path(self) -> Path:
        return self.source.with_suffix(self.source.suffix + ".minicut-ai-cache.json")

    def cancel(self):
        self._cancel = True

    def _load_cache(self) -> dict[int, dict]:
        if not self.use_cache or not self.cache_path.is_file():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            valid = (
                data.get("version") == self.CACHE_VERSION
                and data.get("source") == str(self.source.resolve())
                and data.get("srt") == str(self.srt_path.resolve())
                and int(data.get("interval_ms") or 0) == self.interval_ms
                and int(data.get("window_ms") or 0) == self.window_ms
                and data.get("model") == self.model
            )
            if not valid:
                return {}
            return {int(x["target_ms"]): x for x in data.get("results", []) if "target_ms" in x}
        except Exception:
            return {}

    def _save_cache(self, results: list[dict]):
        if not self.use_cache:
            return
        payload = {
            "version": self.CACHE_VERSION,
            "source": str(self.source.resolve()),
            "srt": str(self.srt_path.resolve()),
            "interval_ms": self.interval_ms,
            "window_ms": self.window_ms,
            "model": self.model,
            "results": results,
        }
        try:
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def run(self):
        try:
            subtitles = SubtitleTrack.load(self.srt_path)
            client = GeminiClient(self.api_key, self.model)
            targets = target_times(self.duration_ms, self.interval_ms)
            cached = self._load_cache()
            results: list[dict] = []

            for index, target_ms in enumerate(targets, 1):
                if self._cancel:
                    self.cancelled.emit()
                    return

                self.progress_changed.emit(index, len(targets), "Mencari kandidat lokal")
                if target_ms in cached:
                    result = dict(cached[target_ms])
                    result["cached"] = True
                    results.append(result)
                    self.target_result.emit(result)
                    continue

                local = find_candidates_for_target(
                    self.ffmpeg,
                    self.source,
                    target_ms,
                    self.window_ms,
                    subtitles,
                    top_n=self.top_n,
                )
                if self._cancel:
                    self.cancelled.emit()
                    return

                self.progress_changed.emit(index, len(targets), "Gemini memilih perpindahan scene")
                verdict = client.verify_candidates(
                    self.ffmpeg,
                    self.source,
                    target_ms,
                    local,
                    subtitles,
                )
                if self._cancel:
                    self.cancelled.emit()
                    return

                self.progress_changed.emit(index, len(targets), "Mengunci ke frame nyata")
                resolved = resolve_semantic_frame(
                    self.source,
                    self.ffprobe,
                    preferred_ms=int(verdict.get("preferred_time_ms") or verdict.get("candidate_time_ms") or target_ms),
                    zone_start_ms=int(verdict.get("boundary_start_ms") or verdict.get("candidate_time_ms") or target_ms),
                    zone_end_ms=int(verdict.get("boundary_end_ms") or verdict.get("candidate_time_ms") or target_ms),
                    subtitles=subtitles,
                    prefer_before_ms=verdict.get("new_content_starts_ms"),
                )
                verdict["semantic_preferred_ms"] = int(verdict.get("preferred_time_ms") or 0)
                verdict["semantic_preferred_time"] = (
                    verdict.get("candidate_time")
                    if not verdict.get("preferred_time_ms")
                    else format_ms(int(verdict["preferred_time_ms"]))
                )
                verdict["selected_time_ms"] = int(resolved["time_ms"])
                verdict["selected_time"] = str(resolved["time"])
                verdict["frame_verified"] = bool(resolved.get("frame_verified"))
                verdict["frame_delta_ms"] = int(resolved.get("frame_delta_ms") or 0)
                verdict["frame_resolution_reason"] = str(resolved.get("reason") or "")
                verdict["local_candidates"] = [c.to_dict() for c in local]
                verdict["cached"] = False
                results.append(verdict)
                self._save_cache(results)
                self.target_result.emit(verdict)
                self.usage_changed.emit(client.usage.__dict__.copy())

            self.done.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
