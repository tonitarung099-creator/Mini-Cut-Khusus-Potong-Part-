from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .candidates import find_candidates_for_target, proximity_shortlist, target_times
from .core import build_preview_proxy, export_segments, export_segments_smartcut, probe_keyframes, probe_media
from .gemini import GeminiClient
from .gemini_keys import model_limits
from .frame_resolver import resolve_requested_frame, resolve_semantic_frame
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

class ManualFrameCutWorker(QThread):
    progress_changed = Signal(int, int, str)
    ready = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        source: Path,
        ffprobe: str,
        requested_times: list[dict],
    ):
        super().__init__()
        self.source = source
        self.ffprobe = ffprobe
        self.requested_times = list(requested_times)

    def run(self):
        try:
            results = []
            total = len(self.requested_times)
            for index, item in enumerate(self.requested_times, 1):
                requested_ms = int(item["time_ms"])
                self.progress_changed.emit(
                    index,
                    total,
                    str(item.get("raw") or format_ms(requested_ms)),
                )
                resolved = resolve_requested_frame(
                    self.source,
                    self.ffprobe,
                    requested_ms,
                )
                resolved["raw"] = str(item.get("raw") or "")
                results.append(resolved)
            self.ready.emit(results)
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
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.text = text
        self.state = dict(state)
        self.locked_timestamps = list(locked_timestamps)
        self.history = list(history)

    def run(self):
        try:
            client = GeminiClient(self.api_key, self.model)
            result = client.chat_command(
                self.text,
                self.state,
                locked_timestamps=self.locked_timestamps,
                history=self.history,
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

    CACHE_VERSION = 8

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
                and int(data.get("top_n") or 0) == self.top_n
                and bool(data.get("allow_deep_check")) == self.allow_deep_check
                and int(data.get("expansion_step_ms") or 0) == self.expansion_step_ms
                and int(data.get("max_expand_ms") or 0) == self.max_expand_ms
                and int(data.get("refine_window_ms") or 0) == self.refine_window_ms
                and data.get("model") == self.model
            )
            if not valid:
                return {}
            return {
                int(x["target_ms"]): x
                for x in data.get("results", [])
                if "target_ms" in x and int(x.get("selected_time_ms") or 0) > 0
            }
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

                # Untuk window AWAL, jangan biarkan hint contact-sheet mengunci
                # refinement ke area yang terlalu jauh (contoh 16:18) dan menghapus
                # kandidat lebih dekat (contoh 15:27). Refinement kembali ke target
                # nominal ±window awal, lalu shortlist nearest-first.
                if expanded_ms == 0:
                    refine_center_ms = target_ms
                    refine_window_ms = self.window_ms
                    reference_ms = target_ms
                    refine_label = (
                        f"Target {format_ms(target_ms)} · bandingkan kandidat terdekat "
                        f"dalam ±{self.window_ms // 60_000} menit"
                    )
                else:
                    # Jika window awal memang tidak punya boundary valid dan pencarian
                    # sudah diperluas, refinement cukup di sekitar hint perluasan.
                    refine_center_ms = hint_ms
                    refine_window_ms = self.refine_window_ms
                    reference_ms = target_ms
                    refine_label = (
                        f"Target {format_ms(target_ms)} · refinement hasil perluasan "
                        f"sekitar {format_ms(hint_ms)}"
                    )

                self.progress_changed.emit(
                    part_index, estimated_total, refine_label
                )

                local_pool = find_candidates_for_target(
                    self.ffmpeg,
                    self.source,
                    refine_center_ms,
                    refine_window_ms,
                    subtitles,
                    top_n=200,
                )
                local = proximity_shortlist(
                    local_pool,
                    reference_ms=reference_ms,
                    max_n=min(4, max(1, self.top_n)),
                )
                if self._cancelled():
                    return

                verdict = client.verify_candidates(
                    self.ffmpeg,
                    self.source,
                    target_ms,
                    local,
                    subtitles,
                    allow_deep_check=self.allow_deep_check,
                    force_deep_check=True,
                )
                self.usage_changed.emit(client.usage.__dict__.copy())
                if self._cancelled():
                    return

                if str(verdict.get("decision") or "").upper() == "NO_VALID_CANDIDATE":
                    no_cut_result = {
                        "target_ms": target_ms,
                        "target": format_ms(target_ms),
                        "selected_time_ms": 0,
                        "selected_time": "",
                        "decision": "NO_CUT",
                        "needs_review": False,
                        "confidence": float(verdict.get("confidence") or 0.0),
                        "reason": str(
                            verdict.get("reason")
                            or "Tidak ada kandidat terdekat yang valid sebagai boundary scene."
                        ),
                        "expanded_ms": expanded_ms,
                        "analysis_mode": (
                            "fixed-grid+nearest-first+short-video+srt+no-valid-candidate"
                        ),
                        "local_candidates": [x.to_dict() for x in local],
                        "cached": False,
                        "usage": client.usage.__dict__.copy(),
                    }
                    results.append(no_cut_result)
                    self._save_cache(results)
                    self.target_result.emit(no_cut_result)
                    self.usage_changed.emit(client.usage.__dict__.copy())
                    continue

                self.progress_changed.emit(
                    part_index, estimated_total, "Mengunci ke frame PTS master"
                )
                resolved = resolve_semantic_frame(
                    self.source,
                    self.ffprobe,
                    preferred_ms=int(
                        verdict.get("preferred_time_ms")
                        or verdict.get("candidate_time_ms")
                        or hint_ms
                    ),
                    zone_start_ms=int(
                        verdict.get("boundary_start_ms")
                        or verdict.get("candidate_time_ms")
                        or hint_ms
                    ),
                    zone_end_ms=int(
                        verdict.get("boundary_end_ms")
                        or verdict.get("candidate_time_ms")
                        or hint_ms
                    ),
                    subtitles=subtitles,
                    prefer_before_ms=verdict.get("new_content_starts_ms"),
                )

                selected_ms = int(resolved["time_ms"])
                if selected_ms <= previous_cut_ms:
                    raise RuntimeError(
                        "Boundary AI tidak berada setelah cut sebelumnya."
                    )

                # Boundary target ini juga tidak boleh masuk terlalu jauh ke area
                # target nominal berikutnya.
                if selected_ms >= search_ceiling:
                    no_cut_result = {
                        "target_ms": target_ms,
                        "target": format_ms(target_ms),
                        "selected_time_ms": 0,
                        "selected_time": "",
                        "decision": "NO_CUT",
                        "needs_review": True,
                        "confidence": float(verdict.get("confidence") or 0.0),
                        "reason": (
                            "Boundary hasil refinement sudah masuk area target berikutnya; "
                            "cut tidak diterapkan pada target ini."
                        ),
                        "expanded_ms": expanded_ms,
                        "analysis_mode": "fixed-grid-boundary-guard",
                        "cached": False,
                        "usage": client.usage.__dict__.copy(),
                    }
                    results.append(no_cut_result)
                    self._save_cache(results)
                    self.target_result.emit(no_cut_result)
                    continue

                verdict["target_ms"] = target_ms
                verdict["target"] = format_ms(target_ms)
                verdict["broad_window_start_ms"] = int(broad.get("window_start_ms") or 0)
                verdict["broad_window_end_ms"] = int(broad.get("window_end_ms") or 0)
                verdict["broad_decision"] = str(broad.get("decision") or "")
                verdict["broad_reason"] = str(broad.get("reason") or "")
                verdict["continuity_summary"] = str(
                    broad.get("continuity_summary") or ""
                )
                verdict["expanded_ms"] = expanded_ms
                verdict["boundary_hint_ms"] = hint_ms
                verdict["boundary_hint"] = format_ms(hint_ms)
                verdict["semantic_preferred_ms"] = int(
                    verdict.get("preferred_time_ms") or 0
                )
                verdict["semantic_preferred_time"] = format_ms(
                    int(
                        verdict.get("preferred_time_ms")
                        or verdict.get("candidate_time_ms")
                        or hint_ms
                    )
                )
                verdict["selected_time_ms"] = selected_ms
                verdict["selected_time"] = str(resolved["time"])
                verdict["frame_verified"] = bool(resolved.get("frame_verified"))
                verdict["frame_delta_ms"] = int(
                    resolved.get("frame_delta_ms") or 0
                )
                verdict["frame_resolution_reason"] = str(
                    resolved.get("reason") or ""
                )
                verdict["local_candidates"] = [x.to_dict() for x in local]
                verdict["analysis_mode"] = (
                    "fixed-grid+contact-sheet+srt-expand+refine"
                    + ("+deep-video" if verdict.get("deep_check_used") else "")
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
