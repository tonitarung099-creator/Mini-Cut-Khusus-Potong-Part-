from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .candidates import target_times
from .core import export_segments, export_segments_smartcut, probe_keyframes, probe_media
from .gemini import GeminiClient
from .gemini_keys import model_limits
from .frame_resolver import probe_frame_points
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


class ExportWorker(QThread):
    progress_changed = Signal(int, str)
    log_line = Signal(str)
    done = Signal(str, int, int, float)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        ffmpeg: str,
        source: Path,
        output_dir: Path,
        base_name: str,
        cuts: list[int],
        duration_ms: int,
        mode: str = "fast",
        smartcut_exe: str | None = None,
        exact_cuts: list[str | None] | None = None,
        srt_path: Path | None = None,
    ):
        super().__init__()
        self.ffmpeg = ffmpeg
        self.source = source
        self.output_dir = output_dir
        self.base_name = base_name
        self.cuts = list(cuts)
        self.duration_ms = duration_ms
        self.mode = mode
        self.smartcut_exe = smartcut_exe
        self.exact_cuts = list(exact_cuts or [])
        self.srt_path = Path(srt_path).resolve() if srt_path is not None else None
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
                    cut_exact_times=self.exact_cuts,
                    progress=lambda p, t: self.progress_changed.emit(p, t),
                    log=lambda s: self.log_line.emit(s),
                    cancelled=lambda: self._cancel,
                    srt_path=self.srt_path,
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
                    srt_path=self.srt_path,
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

class GeminiChatWorker(QThread):
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        api_key: str,
        model: str,
        text: str,
        state: dict,
        locked_timestamps: list[dict],
        history: list[dict[str, str]],
        tool_manifest: dict,
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.text = text
        self.state = dict(state)
        self.locked_timestamps = list(locked_timestamps)
        self.history = list(history)
        self.tool_manifest = dict(tool_manifest)

    def run(self):
        try:
            client = GeminiClient(self.api_key, self.model)
            result = client.chat_command(
                self.text,
                self.state,
                locked_timestamps=self.locked_timestamps,
                history=self.history,
                tool_manifest=self.tool_manifest,
            )
            self.ready.emit(result)
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

class GeminiBatchTestWorker(QThread):
    progress_changed = Signal(int, int, str)
    key_ready = Signal(str, dict)
    key_failed = Signal(str, str)
    done = Signal(int, int)
    cancelled = Signal()

    def __init__(self, keys: list[tuple[str, str, str]], model: str):
        super().__init__()
        self.keys = list(keys)
        self.model = model
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok = 0
        failed = 0
        total = len(self.keys)
        rpm_limit, _tpm, _rpd = model_limits(self.model)
        # Konservatif jika beberapa key ternyata milik project Google yang sama.
        delay_s = max(1.0, 60.0 / max(1, rpm_limit) + 0.35)

        for index, (key_id, name, secret) in enumerate(self.keys, 1):
            if self._cancel:
                self.cancelled.emit()
                return
            self.progress_changed.emit(index, total, name)
            try:
                result = GeminiClient(secret, self.model).test_connection()
                self.key_ready.emit(key_id, result)
                ok += 1
            except Exception as exc:
                self.key_failed.emit(key_id, str(exc))
                failed += 1

            if index < total:
                waited = 0.0
                while waited < delay_s:
                    if self._cancel:
                        self.cancelled.emit()
                        return
                    step = min(0.25, delay_s - waited)
                    time.sleep(step)
                    waited += step

        self.done.emit(ok, failed)


class FilmCutWorker(QThread):
    progress_changed = Signal(int, int, str)
    target_result = Signal(dict)
    usage_changed = Signal(dict)
    done = Signal(list)
    failed = Signal(str)
    cancelled = Signal()

    CACHE_VERSION = 12

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
        top_n: int = 4,
        use_cache: bool = True,
        allow_deep_check: bool = True,
        expansion_step_ms: int = 3 * 60_000,
        max_expand_ms: int = 15 * 60_000,
        refine_window_ms: int = 20_000,
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
        self.allow_deep_check = bool(allow_deep_check)
        self.expansion_step_ms = max(60_000, int(expansion_step_ms))
        self.max_expand_ms = max(0, int(max_expand_ms))
        self.refine_window_ms = max(8_000, int(refine_window_ms))
        self._cancel = False

    @property
    def cache_path(self) -> Path:
        return self.source.with_suffix(self.source.suffix + ".minicut-ai-cache.json")

    @staticmethod
    def _file_signature(path: Path) -> dict | None:
        try:
            stat = path.stat()
            return {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        except OSError:
            return None

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
                and data.get("source_signature") == self._file_signature(self.source)
                and data.get("srt_signature") == self._file_signature(self.srt_path)
                and int(data.get("interval_ms") or 0) == self.interval_ms
                and int(data.get("window_ms") or 0) == self.window_ms
                and int(data.get("top_n") or 0) == self.top_n
                and bool(data.get("allow_deep_check")) == self.allow_deep_check
                and int(data.get("expansion_step_ms") or 0) == self.expansion_step_ms
                and int(data.get("max_expand_ms") or 0) == self.max_expand_ms
                and int(data.get("refine_window_ms") or 0) == self.refine_window_ms
                and data.get("model") == self.model
            )
            if not valid:
                return {}
            restored: dict[int, dict] = {}
            cacheable_no_cut = {
                "NO_CUT",
                "NO_CUT_MAX_EXPAND",
                "SKIPPED_AFTER_PREVIOUS_CUT",
            }
            for item in data.get("results", []):
                if not isinstance(item, dict) or "target_ms" not in item:
                    continue
                target = int(item["target_ms"])
                selected = int(item.get("selected_time_ms") or 0)
                decision = str(item.get("decision") or "").upper()
                if selected > 0:
                    if not str(item.get("selected_time_exact") or "").strip():
                        continue
                    restored[target] = item
                elif decision in cacheable_no_cut:
                    restored[target] = item
            return restored
        except Exception:
            return {}

    def _save_cache(self, results: list[dict]):
        if not self.use_cache:
            return
        payload = {
            "version": self.CACHE_VERSION,
            "source": str(self.source.resolve()),
            "srt": str(self.srt_path.resolve()),
            "source_signature": self._file_signature(self.source),
            "srt_signature": self._file_signature(self.srt_path),
            "interval_ms": self.interval_ms,
            "window_ms": self.window_ms,
            "top_n": self.top_n,
            "allow_deep_check": self.allow_deep_check,
            "expansion_step_ms": self.expansion_step_ms,
            "max_expand_ms": self.max_expand_ms,
            "refine_window_ms": self.refine_window_ms,
            "model": self.model,
            "results": results,
        }
        try:
            self.cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _cancelled(self) -> bool:
        if self._cancel:
            self.cancelled.emit()
            return True
        return False

    def run(self):
        try:
            subtitles = SubtitleTrack.load(self.srt_path)
            client = GeminiClient(self.api_key, self.model)
            cached = self._load_cache()
            results: list[dict] = []

            # Grid target SELALU absolut: 15, 30, 45, 60 menit, dst.
            # Boundary natural boleh bergeser dari target, tetapi tidak menggeser
            # target berikutnya. Contoh: cut target 15 ditemukan di 18:00,
            # target selanjutnya tetap 30:00.
            targets = target_times(self.duration_ms, self.interval_ms)
            estimated_total = max(1, len(targets))
            previous_cut_ms = 0

            for part_index, target_ms in enumerate(targets, 1):
                if self._cancelled():
                    return

                # Jika boundary target sebelumnya sampai melewati target nominal ini,
                # jangan membuat cut mundur/duplikat. Tetap catat bahwa grid ini dilewati.
                if target_ms <= previous_cut_ms:
                    skipped = {
                        "target_ms": target_ms,
                        "target": format_ms(target_ms),
                        "selected_time_ms": 0,
                        "selected_time": "",
                        "decision": "SKIPPED_AFTER_PREVIOUS_CUT",
                        "needs_review": False,
                        "confidence": 1.0,
                        "reason": (
                            "Target nominal sudah terlewati oleh boundary natural "
                            "target sebelumnya; tidak membuat cut tambahan."
                        ),
                        "analysis_mode": "fixed-grid-skip",
                        "cached": False,
                        "usage": client.usage.__dict__.copy(),
                    }
                    results.append(skipped)
                    self._save_cache(results)
                    self.target_result.emit(skipped)
                    continue

                if target_ms in cached:
                    result = dict(cached[target_ms])
                    result["cached"] = True
                    results.append(result)
                    self.target_result.emit(result)
                    selected_cached = int(result.get("selected_time_ms") or 0)
                    if selected_cached > previous_cut_ms:
                        previous_cut_ms = selected_cached
                    continue

                initial_start = max(previous_cut_ms, target_ms - self.window_ms)
                initial_end = min(self.duration_ms, target_ms + self.window_ms)

                # Jangan biarkan ekspansi target sekarang mengambil wilayah target
                # nominal berikutnya. Target berikutnya harus tetap dianalisis sendiri.
                next_target_ms = target_ms + self.interval_ms
                search_ceiling = min(
                    self.duration_ms,
                    max(initial_end, next_target_ms - 30_000),
                )

                pass_start = initial_start
                pass_end = min(initial_end, search_ceiling)
                expanded_ms = 0
                pass_no = 1
                previous_summary = ""
                broad: dict = {}
                found_cut = False

                while True:
                    if self._cancelled():
                        return

                    stage = (
                        f"Target {format_ms(target_ms)} · memahami "
                        f"{format_ms(pass_start)}–{format_ms(pass_end)} "
                        f"· contact sheet + SRT"
                    )
                    self.progress_changed.emit(part_index, estimated_total, stage)

                    broad = client.analyze_scene_window(
                        self.ffmpeg,
                        self.source,
                        target_ms,
                        pass_start,
                        pass_end,
                        subtitles,
                        previous_summary=previous_summary,
                        pass_label=(
                            "awal ± target"
                            if pass_no == 1
                            else f"perluasan +{expanded_ms // 60_000} menit"
                        ),
                    )
                    self.usage_changed.emit(client.usage.__dict__.copy())
                    if self._cancelled():
                        return

                    decision = str(broad.get("decision") or "").upper()
                    if decision == "CUT_FOUND":
                        found_cut = True
                        break

                    previous_summary = str(
                        broad.get("continuity_summary")
                        or broad.get("reason")
                        or previous_summary
                    )

                    reached_limit = (
                        pass_end >= search_ceiling
                        or expanded_ms >= self.max_expand_ms
                    )
                    if reached_limit:
                        no_cut_result = {
                            "target_ms": target_ms,
                            "target": format_ms(target_ms),
                            "selected_time_ms": 0,
                            "selected_time": "",
                            "decision": "NO_CUT",
                            "needs_review": False,
                            "confidence": float(broad.get("confidence") or 0.0),
                            "reason": (
                                str(broad.get("reason") or "")
                                + " · Tidak ada boundary natural sebelum area target berikutnya."
                            ).strip(" ·"),
                            "continuity_summary": previous_summary,
                            "expanded_ms": expanded_ms,
                            "analysis_mode": "contact-sheet+srt-expand-no-cut",
                            "cached": False,
                            "usage": client.usage.__dict__.copy(),
                        }
                        results.append(no_cut_result)
                        self._save_cache(results)
                        self.target_result.emit(no_cut_result)
                        self.usage_changed.emit(client.usage.__dict__.copy())
                        break

                    old_end = pass_end
                    new_end = min(
                        search_ceiling,
                        old_end + self.expansion_step_ms,
                    )
                    # Hanya bagian tambahan yang dikirim, dengan overlap 30 detik
                    # untuk menjaga kontinuitas visual + subtitle.
                    pass_start = max(previous_cut_ms, old_end - 30_000)
                    pass_end = new_end
                    expanded_ms += max(0, new_end - old_end)
                    pass_no += 1

                if not found_cut:
                    # Grid selanjutnya tetap 30/45/60 dst.; NO_CUT tidak menggeser target.
                    continue

                hint_ms = int(broad.get("boundary_hint_ms") or target_ms)

                # FINAL AUTHORITY = GEMINI.
                # Local code only enumerates REAL master-frame PTS around Gemini's
                # rough boundary hint. It does not rank, snap, or move the cut.
                self.progress_changed.emit(
                    part_index,
                    estimated_total,
                    "Gemini memilih frame master final · tanpa snap lokal",
                )
                radius_ms = 2_500
                frame_start = max(
                    previous_cut_ms + 1,
                    hint_ms - radius_ms,
                )
                frame_end = min(
                    self.duration_ms,
                    hint_ms + radius_ms,
                )
                master_frames = probe_frame_points(
                    self.source,
                    self.ffprobe,
                    frame_start,
                    frame_end,
                )
                if not master_frames:
                    raise RuntimeError(
                        "Frame master tidak ditemukan di sekitar boundary Gemini."
                    )

                exact = client.choose_exact_master_frame(
                    self.ffmpeg,
                    self.source,
                    target_ms,
                    hint_ms,
                    master_frames,
                    subtitles,
                )
                self.usage_changed.emit(client.usage.__dict__.copy())
                if self._cancelled():
                    return

                selected_ms = int(exact["selected_time_ms"])
                if selected_ms <= previous_cut_ms:
                    raise RuntimeError(
                        "Frame final Gemini tidak berada setelah cut sebelumnya."
                    )
                if selected_ms >= self.duration_ms:
                    raise RuntimeError(
                        "Frame final Gemini berada di luar durasi video."
                    )

                verdict = dict(broad)
                verdict.update(exact)
                verdict["decision"] = "CUT_FOUND"
                verdict["broad_decision"] = str(
                    broad.get("decision") or "CUT_FOUND"
                ).upper()
                verdict["expanded_ms"] = expanded_ms
                verdict["boundary_hint_ms"] = hint_ms
                verdict["boundary_hint"] = format_ms(hint_ms)
                verdict["semantic_preferred_ms"] = selected_ms
                verdict["semantic_preferred_time"] = format_ms(selected_ms)
                verdict["selected_time_ms"] = selected_ms
                verdict["selected_time"] = format_ms(selected_ms)
                verdict["frame_verified"] = True
                verdict["frame_delta_ms"] = 0
                verdict["frame_resolution_reason"] = (
                    "Gemini memilih langsung PTS frame master final; "
                    "MiniCut tidak melakukan snap atau penyesuaian lokal."
                )
                verdict["local_candidates"] = []
                verdict["analysis_mode"] = (
                    "fixed-grid+contact-sheet+srt+gemini-exact-master-frame"
                )
                verdict["cached"] = False
                verdict["usage"] = client.usage.__dict__.copy()

                results.append(verdict)
                self._save_cache(results)
                self.target_result.emit(verdict)
                self.usage_changed.emit(client.usage.__dict__.copy())
                previous_cut_ms = selected_ms

            self.done.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
