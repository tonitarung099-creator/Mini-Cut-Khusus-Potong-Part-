from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidates import LocalCandidate
from .core import creation_flags
from .subtitles import SubtitleTrack, format_ms

DEFAULT_MODEL = "gemini-3.5-flash-lite"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Broad scene understanding: contact sheets + SRT, never a multi-minute video.
CONTACT_SEGMENT_MS = 30_000
CONTACT_FRAME_STEP_MS = 2_000
CONTACT_COLS = 5
CONTACT_ROWS = 3
CONTACT_CELL_WIDTH = 168

# Precise refinement after a real scene boundary has been found.
STORYBOARD_OFFSETS_MS = (-5000, -1200, 1200, 5000)
FRAME_WIDTH = 416
DEEP_CHECK_CONFIDENCE = 0.72
DEEP_CHECK_RADIUS_MS = 10_000
DEEP_CHECK_WIDTH = 360
DEEP_CHECK_FPS = 8
SEMANTIC_ZONE_LIMIT_MS = 1500

# Final cut authority: Gemini chooses from REAL master-frame PTS values.
# Local code may enumerate/display frames, but it never moves Gemini's choice.
EXACT_FRAME_RADIUS_MS = 2500
EXACT_COARSE_MAX_FRAMES = 13
EXACT_FINE_MAX_FRAMES = 31
EXACT_FRAME_WIDTH = 384


@dataclass
class GeminiUsage:
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key.strip()
        self.model = model.strip() or DEFAULT_MODEL
        self.usage = GeminiUsage()
        if not self.api_key:
            raise ValueError("Gemini API key belum diisi.")

    def _post(
        self,
        payload: dict[str, Any],
        timeout: int = 90,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        url = f"{API_ROOT}/models/{self.model}:generateContent"
        body_bytes = json.dumps(payload).encode("utf-8")

        last_error: Exception | None = None
        max_attempts = max(1, min(int(max_attempts), 3))
        for attempt in range(max_attempts):
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                try:
                    detail = json.loads(raw).get("error", {}).get("message") or raw
                except Exception:
                    detail = raw
                last_error = RuntimeError(f"Gemini API {exc.code}: {detail}")

                # Retry/backoff hanya pada limit sementara / service unavailable.
                # Tidak berpindah API key secara otomatis.
                if exc.code not in (429, 503) or attempt >= max_attempts - 1:
                    raise last_error from exc

                retry_after = 0.0
                try:
                    retry_after = float(exc.headers.get("Retry-After") or 0)
                except Exception:
                    retry_after = 0.0
                fallback = (4.0, 10.0)[min(attempt, 1)]
                time.sleep(max(fallback, min(60.0, retry_after)))
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Tidak dapat terhubung ke Gemini: {exc.reason}"
                ) from exc
        else:
            raise last_error or RuntimeError("Gemini request gagal setelah retry.")

        self.usage.requests += 1
        meta = data.get("usageMetadata") or {}
        self.usage.prompt_tokens += int(meta.get("promptTokenCount") or 0)
        self.usage.output_tokens += int(meta.get("candidatesTokenCount") or 0)
        self.usage.total_tokens += int(meta.get("totalTokenCount") or 0)
        return data

    def test_connection(self) -> dict[str, Any]:
        payload = {
            "contents": [{"parts": [{"text": "Return only JSON: {\"ok\":true}"}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 32,
                "responseMimeType": "application/json",
            },
        }
        data = self._post(payload, timeout=30)
        text = _response_text(data)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"raw": text}
        return {
            "ok": True,
            "model": self.model,
            "response": parsed,
            "usage": self.usage.__dict__.copy(),
        }

    def chat_command(
        self,
        user_text: str,
        state: dict[str, Any],
        locked_timestamps: list[dict[str, Any]] | None = None,
        history: list[dict[str, str]] | None = None,
        tool_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reasoning command agent for the MiniCut Chat tab.

        Gemini is the authority for chat-driven cuts. Explicit timestamps are
        returned by Gemini and applied unchanged by MiniCut; there is no local
        frame-lock/snap step after Gemini chooses the cut time.
        """
        locked = list(locked_timestamps or [])
        history = list(history or [])[-12:]
        manifest = dict(tool_manifest or {})
        project = {
            "source": state.get("source"),
            "duration_ms": state.get("duration_ms"),
            "duration": state.get("duration"),
            "playhead_ms": state.get("playhead_ms"),
            "playhead": state.get("playhead"),
            "fps": state.get("fps"),
            "parts": state.get("parts"),
            "cuts": state.get("cuts"),
        }
        prompt = f"""
Kamu adalah Gemini Command Agent di aplikasi MiniCut Studio.
Berkomunikasilah natural dalam Bahasa Indonesia, singkat, jelas, dan fokus ke maksud pengguna.

Kamu TIDAK terbatas pada perintah cut manual. Pahami perintah seperti manusia lalu pilih aksi
MiniCut yang tepat dari tool yang tersedia. Kamu boleh menyusun beberapa aksi berurutan untuk
perintah majemuk.

KONDISI PROYEK SAAT INI:
{json.dumps(project, ensure_ascii=False)}

TOOL MINICUT YANG TERSEDIA:
{json.dumps(manifest, ensure_ascii=False)}

TIMESTAMP REFERENSI DARI UI (jika ada):
{json.dumps(locked, ensure_ascii=False)}

TOOL KHUSUS CHAT:
- manual_frame_cut(time_ms:int, raw?:str)
  Gunakan untuk cut tepat pada waktu yang diminta pengguna. Nilai time_ms dari Gemini adalah FINAL:
  MiniCut akan menyimpannya apa adanya tanpa snap keyframe, tanpa frame-lock lokal, dan tanpa
  menggeser ke timestamp lain.

ATURAN REASONING DAN EKSEKUSI:
- Pikirkan kebutuhan pengguna secara internal sebelum memilih aksi. Jangan tampilkan chain-of-thought.
- Jika pengguna hanya bertanya/berdiskusi, jawab natural dan actions harus [].
- Jika pengguna meminta tindakan, isi actions hanya dengan tool nyata yang tersedia.
- Jangan keluarkan get_state sebagai action; state proyek sudah diberikan di prompt ini.
- Untuk cut pada waktu tertentu, gunakan manual_frame_cut, BUKAN add_cut.
- Bedakan TITIK WAKTU dengan INTERVAL/DURASI. Contoh:
  "potong di 1 jam 12 detik" = satu manual_frame_cut pada 01:00:12.
  "potong tiap 1 jam 12 detik" = divide_interval dengan interval_ms 3.612.000.
  "setiap part 1 jam 12 detik" = divide_interval, BUKAN satu cut di 01:00:12.
- Jika ada timestamp eksplisit pada pesan pengguna, Gemini sendiri harus mengubahnya ke time_ms
  integer yang tepat lalu mengeluarkan manual_frame_cut. Jangan mengubah nilainya untuk mencari
  titik yang "lebih pas".
- Pahami waktu natural Indonesia dan format ringkas/typo umum. Contoh:
  "1 jam lebih 2 menit" = 3.720.000 ms.
  "1 jam 2 menit 30 detik" = 3.750.000 ms.
  "1jam 12dtik" = 3.612.000 ms.
  "1h12s" = 3.612.000 ms.
  "62mnt 5dtk" = 3.725.000 ms.
  "90 detik" = 90.000 ms.
- Waktu dan interval bersifat ARBITRER. Jangan pernah membatasi, membulatkan, atau mengubah
  permintaan pengguna menjadi kelipatan 5 detik, 10 detik, 1 menit, atau preset lain.
- Jika pengguna meminta cut di 1 jam 12 detik, artinya tepat 01:00:12.000 pada timeline,
  bukan 5 detik, bukan 01:00:00, dan bukan durasi preset. MiniCut tidak akan menggesernya lagi.
- time_ms dari manual_frame_cut harus integer MILIDETIK, bukan detik.
- Selama tool tersedia, gunakan kemampuan MiniCut yang relevan: timeline, playback, seek, frame-step,
  cut, pembagian part, AI Film Cut, pilihan mode ekspor, simpan, ekspor, dan undo.
- Boleh memahami variasi bahasa seperti potong/cut/pecah/split, pergi/lompat/seek, putar/play,
  jeda/pause, hapus, bagi part, simpan, ekspor, batalkan/cancel proses, dan perintah majemuk.
- Jika perintah pengguna valid tetapi format bahasanya tidak persis sama dengan contoh, pahami maksudnya
  secara semantik; jangan menolak hanya karena frasa atau singkatannya berbeda.
- Jangan mengarang tool, path file, index cut, atau fakta yang tidak ada di state.
- save_project dan export_all hanya boleh dipakai jika pengguna memintanya secara eksplisit.
- clear_cuts/remove_cut hanya boleh dipakai jika pengguna jelas meminta penghapusan.
- Jangan mengklaim tindakan sudah berhasil; balasan menjelaskan apa yang akan dilakukan.
  Aplikasi akan memberi hasil eksekusi setelah JSON ini diproses.
- Jangan meminta API key di chat; gunakan API aktif aplikasi.
- Maksimal 12 actions agar satu pesan tidak membuat loop tak terkendali.

Kembalikan HANYA JSON valid:
{{
  "reply": "jawaban natural untuk pengguna",
  "intent": "act" atau "chat",
  "actions": [
    {{"tool": "nama_tool", "args": {{}}}}
  ]
}}
""".strip()

        contents: list[dict[str, Any]] = []
        for item in history:
            role = "model" if item.get("role") == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": str(item.get("text") or "")[:5000]}],
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt + "\n\nPESAN PENGGUNA:\n" + str(user_text)}],
        })
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1200,
                "responseMimeType": "application/json",
            },
        }
        data = self._post(payload, timeout=45, max_attempts=1)
        raw = _response_text(data)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini Chat tidak mengembalikan JSON valid.") from exc

        reply = str(result.get("reply") or "").strip()
        intent = str(result.get("intent") or "chat").strip().lower()
        if intent not in {"act", "chat"}:
            intent = "act" if result.get("actions") else "chat"

        raw_actions = result.get("actions")
        actions: list[dict[str, Any]] = []
        if isinstance(raw_actions, list):
            for item in raw_actions[:12]:
                if not isinstance(item, dict):
                    continue
                tool = str(item.get("tool") or "").strip()
                if not tool:
                    continue
                args = item.get("args")
                if not isinstance(args, dict):
                    args = {}
                actions.append({"tool": tool, "args": args})

        result = {
            "reply": reply,
            "intent": intent,
            "actions": actions,
            "usage": self.usage.__dict__.copy(),
            "model": self.model,
        }
        return result

    def analyze_scene_window(
        self,
        ffmpeg: str,
        source: Path,
        target_ms: int,
        window_start_ms: int,
        window_end_ms: int,
        subtitles: SubtitleTrack | None,
        previous_summary: str = "",
        pass_label: str = "awal",
    ) -> dict[str, Any]:
        """Pahami satu rentang film sebagai rangkaian scene, bukan kandidat terpisah."""
        start_ms = max(0, int(window_start_ms))
        end_ms = max(start_ms + 1, int(window_end_ms))
        segments: list[dict[str, Any]] = []
        parts: list[dict[str, Any]] = [{
            "text": _scene_window_prompt(
                target_ms=target_ms,
                window_start_ms=start_ms,
                window_end_ms=end_ms,
                previous_summary=previous_summary,
                pass_label=pass_label,
            )
        }]

        seg_start = start_ms
        seg_index = 1
        while seg_start < end_ms:
            seg_end = min(end_ms, seg_start + CONTACT_SEGMENT_MS)
            frame_times = list(range(seg_start, seg_end, CONTACT_FRAME_STEP_MS))
            if not frame_times:
                frame_times = [seg_start]
            frame_times = frame_times[: CONTACT_COLS * CONTACT_ROWS]

            srt_text = (
                subtitles.between_text(seg_start, seg_end, max_chars=3400)
                if subtitles else ""
            )
            mapping = " · ".join(
                f"F{i + 1}={format_ms(t)}" for i, t in enumerate(frame_times)
            )
            parts.append({
                "text": (
                    f"SEGMENT {seg_index} | {format_ms(seg_start)}–{format_ms(seg_end)}\n"
                    f"Urutan grid: kiri→kanan, baris atas lalu bawah. {mapping}\n"
                    "SRT pada rentang yang SAMA:\n"
                    + (srt_text or "(tidak ada subtitle pada segmen ini)")
                )
            })
            sheet = extract_contact_sheet_jpeg(
                ffmpeg,
                source,
                seg_start,
                seg_end,
                frame_step_ms=CONTACT_FRAME_STEP_MS,
                cell_width=CONTACT_CELL_WIDTH,
                cols=CONTACT_COLS,
                rows=CONTACT_ROWS,
            )
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(sheet).decode("ascii"),
                },
                "media_resolution": {"level": "MEDIA_RESOLUTION_LOW"},
            })
            segments.append({
                "index": seg_index,
                "start_ms": seg_start,
                "end_ms": seg_end,
                "frame_times": frame_times,
            })
            seg_start = seg_end
            seg_index += 1

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.05,
                "maxOutputTokens": 950,
                "responseMimeType": "application/json",
            },
        }
        data = self._post(payload, timeout=120)
        text = _response_text(data)
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini tidak mengembalikan JSON valid untuk pemahaman scene.") from exc

        decision = str(result.get("decision") or "").strip().upper()
        allowed = {"CUT_FOUND", "NO_CUT", "EXPAND_FORWARD", "EXPAND_BACKWARD"}
        if decision not in allowed:
            raise RuntimeError("Gemini tidak mengembalikan decision scene yang valid.")
        result["decision"] = decision
        result["window_start_ms"] = start_ms
        result["window_end_ms"] = end_ms
        result["window_start"] = format_ms(start_ms)
        result["window_end"] = format_ms(end_ms)
        result["analysis_mode"] = "contact-sheet+srt"
        result["contact_segments"] = len(segments)

        if decision == "CUT_FOUND":
            try:
                seg_no = int(result.get("segment_index") or 0)
                frame_no = int(result.get("frame_index") or 0)
            except Exception as exc:
                raise RuntimeError("CUT_FOUND tidak memiliki segment/frame yang valid.") from exc
            if seg_no < 1 or seg_no > len(segments):
                raise RuntimeError("segment_index di luar rentang contact sheet.")
            segment = segments[seg_no - 1]
            frame_times = segment["frame_times"]
            if frame_no < 1 or frame_no > len(frame_times):
                raise RuntimeError("frame_index di luar contact sheet.")
            hint_ms = int(frame_times[frame_no - 1])
            result["boundary_hint_ms"] = hint_ms
            result["boundary_hint"] = format_ms(hint_ms)

        result["usage"] = self.usage.__dict__.copy()
        return result

    def choose_exact_master_frame(
        self,
        ffmpeg: str,
        source: Path,
        target_ms: int,
        boundary_hint_ms: int,
        frame_points: list[dict[str, Any]],
        subtitles: SubtitleTrack | None,
    ) -> dict[str, Any]:
        """Let Gemini choose the final cut from real master-frame PTS values.

        Local code only enumerates source frames and preserves their exact
        rational PTS. Gemini chooses the final frame; that exact PTS is returned
        unchanged for SmartCut export.
        """
        points = sorted(
            [
                {
                    "time_ms": max(0, int(item["time_ms"])),
                    "exact_time": str(item["exact_time"]),
                    "seek_time": str(
                        item.get("seek_time")
                        or max(0, int(item["time_ms"])) / 1000
                    ),
                }
                for item in frame_points
                if item.get("exact_time") not in (None, "")
            ],
            key=lambda item: item["time_ms"],
        )
        if not points:
            raise ValueError("Tidak ada frame master untuk dipilih Gemini.")

        coarse_points = _evenly_sample_times(points, EXACT_COARSE_MAX_FRAMES)
        coarse = self._choose_exact_frame_pass(
            ffmpeg,
            source,
            target_ms,
            boundary_hint_ms,
            coarse_points,
            subtitles,
            phase="coarse",
        )
        coarse_index = _frame_index(
            coarse.get("selected_frame_index"), len(coarse_points)
        )
        coarse_point = coarse_points[coarse_index - 1]
        coarse_ms = int(coarse_point["time_ms"])

        center = min(
            range(len(points)),
            key=lambda i: (
                abs(int(points[i]["time_ms"]) - coarse_ms),
                i,
            ),
        )
        half = EXACT_FINE_MAX_FRAMES // 2
        fine_start = max(0, center - half)
        fine_end = min(len(points), fine_start + EXACT_FINE_MAX_FRAMES)
        fine_start = max(0, fine_end - EXACT_FINE_MAX_FRAMES)
        fine_points = points[fine_start:fine_end]

        fine = self._choose_exact_frame_pass(
            ffmpeg,
            source,
            target_ms,
            coarse_ms,
            fine_points,
            subtitles,
            phase="fine",
        )
        fine_index = _frame_index(
            fine.get("selected_frame_index"), len(fine_points)
        )
        selected = fine_points[fine_index - 1]
        selected_ms = int(selected["time_ms"])
        selected_exact = str(selected["exact_time"])

        return {
            "decision": "CUT_FOUND",
            "selected_time_ms": selected_ms,
            "selected_time": format_ms(selected_ms),
            "selected_time_exact": selected_exact,
            "selected_frame_index": fine_index,
            "confidence": float(fine.get("confidence") or 0.0),
            "needs_review": bool(fine.get("needs_review", False)),
            "cut_intent": str(fine.get("cut_intent") or "scene_transition"),
            "reason": str(
                fine.get("reason")
                or "Gemini memilih frame master final dari PTS yang tersedia."
            ),
            "frame_verified": True,
            "frame_delta_ms": 0,
            "frame_authority": "gemini",
            "frame_selection": "gemini-exact-master-pts",
            "coarse_selected_ms": coarse_ms,
            "coarse_selected_time": format_ms(coarse_ms),
            "coarse_selected_time_exact": str(coarse_point["exact_time"]),
            "exact_frame_pool_count": len(points),
            "exact_frame_fine_count": len(fine_points),
            "usage": self.usage.__dict__.copy(),
        }

    def _choose_exact_frame_pass(
        self,
        ffmpeg: str,
        source: Path,
        target_ms: int,
        hint_ms: int,
        frame_points: list[dict[str, Any]],
        subtitles: SubtitleTrack | None,
        phase: str,
    ) -> dict[str, Any]:
        if not frame_points:
            raise ValueError("Daftar frame Gemini kosong.")

        srt_text = (
            subtitles.nearby_text(
                hint_ms,
                radius_ms=5_000,
                max_chars=2600,
            )
            if subtitles else ""
        )
        prompt = f"""
Pilih FRAME MASTER FINAL untuk titik potong Part film.

TARGET PART: {format_ms(target_ms)}
PETUNJUK BOUNDARY: {format_ms(hint_ms)}
TAHAP: {phase}

Semua frame yang diberikan adalah frame nyata dari video master. Setiap frame membawa
MASTER_PTS_EXACT yang berasal langsung dari time-base video. Anda HARUS memilih tepat
satu frame yang tersedia.

ATURAN:
1. Gemini adalah penentu akhir titik potong. Pilih frame pertama yang paling tepat untuk
   memulai scene/konteks baru, sehingga Part sebelumnya berakhir tepat sebelum frame itu.
2. Jangan memilih berdasarkan kedekatan waktu saja. Utamakan perpindahan scene/konteks natural.
3. Jangan memotong di tengah dialog, aksi-reaksi, gerakan penting, atau kontinuitas yang sama.
4. Jangan mengarang timestamp dan jangan meminta aplikasi menggeser pilihan.
5. selected_frame_index HARUS menunjuk salah satu frame yang benar-benar diberikan.
6. Pada tahap fine, pilihan ini FINAL. MASTER_PTS_EXACT dari frame itu akan langsung
   diteruskan ke SmartCut tanpa pembulatan milidetik atau snap lokal.

SRT sekitar boundary:
{srt_text or "(tidak ada subtitle)"}

Kembalikan HANYA JSON valid:
{{
  "selected_frame_index": 1,
  "confidence": 0.0,
  "needs_review": false,
  "cut_intent": "scene_transition",
  "reason": "alasan singkat dalam Bahasa Indonesia"
}}
""".strip()

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for i, point in enumerate(frame_points, 1):
            time_ms = int(point["time_ms"])
            exact_time = str(point["exact_time"])
            parts.append({
                "text": (
                    f"FRAME {i} | MASTER_PTS_EXACT={exact_time} | "
                    f"DISPLAY_MS={time_ms} | {format_ms(time_ms)}"
                )
            })
            jpg = extract_frame_jpeg(
                ffmpeg,
                source,
                time_ms,
                EXACT_FRAME_WIDTH,
                seek_time=str(point.get("seek_time") or ""),
            )
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(jpg).decode("ascii"),
                },
                "media_resolution": {"level": "MEDIA_RESOLUTION_LOW"},
            })

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 500,
                "responseMimeType": "application/json",
            },
        }
        data = self._post(payload, timeout=120)
        raw = _response_text(data)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Gemini tidak mengembalikan JSON valid untuk pemilihan frame exact."
            ) from exc

        _frame_index(result.get("selected_frame_index"), len(frame_points))
        return result

    def verify_candidates(
        self,
        ffmpeg: str,
        source: Path,
        target_ms: int,
        candidates: list[LocalCandidate],
        subtitles: SubtitleTrack | None,
        frame_width: int = FRAME_WIDTH,
        allow_deep_check: bool = True,
        force_deep_check: bool = False,
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("Tidak ada kandidat untuk diverifikasi Gemini.")

        # Kandidat SELALU diurutkan berdasarkan jarak dari target nominal.
        # Kandidat yang lebih jauh hanya boleh menang bila kandidat yang lebih dekat
        # dinilai tidak valid sebagai boundary scene.
        ordered = sorted(
            candidates,
            key=lambda x: (
                abs(int(x.time_ms) - int(target_ms)),
                -float(x.score),
                int(x.time_ms),
            ),
        )[:4]

        if allow_deep_check:
            result = self._deep_check_candidates(
                ffmpeg=ffmpeg,
                source=source,
                target_ms=target_ms,
                candidates=ordered,
                candidate_indexes=list(range(1, len(ordered) + 1)),
                subtitles=subtitles,
            )
            decision = str(result.get("decision") or "").upper()
            if decision == "NO_VALID_CANDIDATE":
                result["selected_candidate_index"] = None
                result["candidate_time_ms"] = 0
                result["candidate_time"] = ""
                result["preferred_time_ms"] = 0
                result["target_ms"] = target_ms
                result["target"] = format_ms(target_ms)
                result["analysis_mode"] = "nearest-first+short-video+srt"
                result["deep_check_used"] = True
                result["candidate_count"] = len(ordered)
                result["usage"] = self.usage.__dict__.copy()
                return result

            selected_index = _candidate_index(
                result.get("selected_candidate_index"), len(ordered)
            )
            result["deep_check_used"] = True
        else:
            # Fallback hemat: storyboard gambar saja. Tetap gunakan nearest-first.
            prompt = _storyboard_prompt(target_ms, ordered, subtitles)
            parts: list[dict[str, Any]] = [{"text": prompt}]
            labels = tuple(
                f"{offset / 1000:+g}s" for offset in STORYBOARD_OFFSETS_MS
            )
            for i, candidate in enumerate(ordered, 1):
                parts.append({
                    "text": (
                        f"KANDIDAT {i} · pusat {format_ms(candidate.time_ms)} · "
                        f"jarak target {candidate.distance_ms/1000:.1f}s"
                    )
                })
                for label, offset_ms in zip(labels, STORYBOARD_OFFSETS_MS):
                    frame_ms = max(0, candidate.time_ms + offset_ms)
                    jpg = extract_frame_jpeg(ffmpeg, source, frame_ms, frame_width)
                    parts.append({
                        "text": f"Kandidat {i} · {label} · {format_ms(frame_ms)}"
                    })
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(jpg).decode("ascii"),
                        },
                        "media_resolution": {"level": "MEDIA_RESOLUTION_LOW"},
                    })

            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.05,
                    "maxOutputTokens": 850,
                    "responseMimeType": "application/json",
                },
            }
            data = self._post(payload)
            text = _response_text(data)
            try:
                result = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Gemini tidak mengembalikan JSON valid pada refinement storyboard."
                ) from exc

            decision = str(result.get("decision") or "CUT_FOUND").upper()
            if decision == "NO_VALID_CANDIDATE":
                result["selected_candidate_index"] = None
                result["candidate_time_ms"] = 0
                result["candidate_time"] = ""
                result["preferred_time_ms"] = 0
                result["target_ms"] = target_ms
                result["target"] = format_ms(target_ms)
                result["analysis_mode"] = "nearest-first+storyboard+srt"
                result["deep_check_used"] = False
                result["candidate_count"] = len(ordered)
                result["usage"] = self.usage.__dict__.copy()
                return result
            selected_index = _candidate_index(
                result.get("selected_candidate_index"), len(ordered)
            )
            result["deep_check_used"] = False

        selected = ordered[selected_index - 1]
        preferred_off = _bounded_offset(result.get("preferred_offset_ms"), 0)
        preferred_ms = max(0, selected.time_ms + preferred_off)

        result["decision"] = "CUT_FOUND"
        result["selected_candidate_index"] = selected_index
        result["candidate_time_ms"] = selected.time_ms
        result["candidate_time"] = format_ms(selected.time_ms)
        result["candidate_distance_ms"] = abs(selected.time_ms - target_ms)
        result["preferred_time_ms"] = preferred_ms
        result["boundary_start_ms"] = max(0, preferred_ms - 700)
        result["boundary_end_ms"] = preferred_ms + 700
        result["new_content_starts_ms"] = None
        result["target_ms"] = target_ms
        result["target"] = format_ms(target_ms)
        result["analysis_mode"] = (
            "nearest-first+short-video+srt"
            if result["deep_check_used"]
            else "nearest-first+storyboard+srt"
        )
        result["frames_per_candidate"] = len(STORYBOARD_OFFSETS_MS)
        result["candidate_count"] = len(ordered)
        result["cut_intent"] = str(result.get("cut_intent") or "scene_transition")
        result["usage"] = self.usage.__dict__.copy()
        return result

    def _deep_check_candidates(
        self,
        ffmpeg: str,
        source: Path,
        target_ms: int,
        candidates: list[LocalCandidate],
        candidate_indexes: list[int],
        subtitles: SubtitleTrack | None,
    ) -> dict[str, Any]:
        # Kandidat dibandingkan dengan video pendek TANPA AUDIO + SRT sinkron.
        # Urutan kandidat sengaja nearest-first.
        parts: list[dict[str, Any]] = [{
            "text": _deep_check_prompt(
                target_ms, candidates, candidate_indexes, subtitles
            )
        }]
        for idx in candidate_indexes:
            candidate = candidates[idx - 1]
            clip = extract_silent_context_clip_mp4(
                ffmpeg,
                source,
                candidate.time_ms,
                radius_ms=DEEP_CHECK_RADIUS_MS,
            )
            srt_text = (
                subtitles.nearby_text(
                    candidate.time_ms,
                    radius_ms=DEEP_CHECK_RADIUS_MS + 2_000,
                    max_chars=2600,
                )
                if subtitles else ""
            )
            parts.append({
                "text": (
                    f"KANDIDAT {idx} · {format_ms(candidate.time_ms)} · "
                    f"jarak target {candidate.distance_ms/1000:.1f}s · "
                    f"VIDEO ±{DEEP_CHECK_RADIUS_MS/1000:.0f}s TANPA AUDIO\n"
                    "SRT pada rentang kandidat yang sama:\n"
                    + (srt_text or "(tidak ada subtitle)")
                )
            })
            parts.append({
                "inline_data": {
                    "mime_type": "video/mp4",
                    "data": base64.b64encode(clip).decode("ascii"),
                },
                "media_resolution": {"level": "MEDIA_RESOLUTION_LOW"},
            })

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.02,
                "maxOutputTokens": 850,
                "responseMimeType": "application/json",
            },
        }
        data = self._post(payload, timeout=120)
        text = _response_text(data)
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Gemini tidak mengembalikan JSON valid pada pemeriksaan video kandidat."
            ) from exc

        decision = str(result.get("decision") or "CUT_FOUND").upper()
        if decision == "NO_VALID_CANDIDATE":
            result["decision"] = decision
            result["needs_review"] = bool(result.get("needs_review", False))
            result["deep_check_compared"] = candidate_indexes
            return result
        if decision != "CUT_FOUND":
            raise RuntimeError("Decision video kandidat tidak valid.")

        selected = _candidate_index(
            result.get("selected_candidate_index"), len(candidates)
        )
        if selected not in candidate_indexes:
            raise RuntimeError(
                "Pemeriksaan video memilih kandidat di luar daftar."
            )

        result["decision"] = "CUT_FOUND"
        result["selected_candidate_index"] = selected
        result["needs_review"] = bool(result.get("needs_review", False))
        result["deep_check_compared"] = candidate_indexes
        return result

def _frame_index(value: Any, count: int) -> int:
    try:
        index = int(value)
    except Exception as exc:
        raise RuntimeError(
            "Gemini tidak mengembalikan selected_frame_index."
        ) from exc
    if index < 1 or index > count:
        raise RuntimeError("Gemini memilih frame di luar daftar master.")
    return index


def _evenly_sample_times(values: list[Any], max_items: int) -> list[Any]:
    values = list(values)
    max_items = max(1, int(max_items))
    if len(values) <= max_items:
        return values
    if max_items == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    indexes = sorted({
        round(i * last / (max_items - 1))
        for i in range(max_items)
    })
    return [values[i] for i in indexes]


def _candidate_index(value: Any, count: int) -> int:
    try:
        index = int(value)
    except Exception as exc:
        raise RuntimeError("Gemini tidak mengembalikan selected_candidate_index.") from exc
    if index < 1 or index > count:
        raise RuntimeError("Gemini memilih kandidat di luar daftar.")
    return index


def _optional_candidate_index(value: Any, count: int, selected: int) -> int | None:
    try:
        index = int(value)
    except Exception:
        return None
    if index < 1 or index > count or index == selected:
        return None
    return index


def _best_alternate_index(candidates: list[LocalCandidate], selected_index: int) -> int | None:
    choices = [
        (i + 1, candidate)
        for i, candidate in enumerate(candidates)
        if i + 1 != selected_index
    ]
    if not choices:
        return None
    choices.sort(key=lambda item: (item[1].score, -item[1].distance_ms), reverse=True)
    return choices[0][0]


def _needs_deep_check(result: dict[str, Any]) -> bool:
    try:
        confidence = float(result.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    return bool(
        result.get("needs_deep_check")
        or result.get("needs_video_check")
        or result.get("needs_review")
        or not result.get("scene_change", False)
        or not result.get("dialog_safe", False)
        or confidence < DEEP_CHECK_CONFIDENCE
    )


def _bounded_offset(value: Any, default: int | None) -> int:
    if value is None:
        if default is None:
            raise ValueError("offset kosong")
        value = default
    try:
        offset = int(round(float(value)))
    except Exception:
        if default is None:
            raise
        offset = default
    return max(-SEMANTIC_ZONE_LIMIT_MS, min(SEMANTIC_ZONE_LIMIT_MS, offset))


def extract_contact_sheet_jpeg(
    ffmpeg: str,
    source: Path,
    start_ms: int,
    end_ms: int,
    frame_step_ms: int = CONTACT_FRAME_STEP_MS,
    cell_width: int = CONTACT_CELL_WIDTH,
    cols: int = CONTACT_COLS,
    rows: int = CONTACT_ROWS,
) -> bytes:
    """Buat satu JPEG contact sheet dari rentang waktu pendek secara lokal."""
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms + 1, int(end_ms))
    duration_s = max(0.2, (end_ms - start_ms) / 1000)
    step_s = max(0.5, int(frame_step_ms) / 1000)
    cells = max(1, int(cols) * int(rows))
    vf = (
        f"fps=1/{step_s:g},"
        f"scale={max(96, int(cell_width))}:-2:flags=fast_bilinear,"
        f"tile={max(1, int(cols))}x{max(1, int(rows))}:nb_frames={cells}:padding=2:margin=2"
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_ms / 1000:.3f}",
        "-i", str(source),
        "-t", f"{duration_s:.3f}",
        "-vf", vf,
        "-frames:v", "1",
        "-q:v", "7",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags(),
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(err.strip() or "Gagal membuat contact sheet Gemini.")
    return proc.stdout


def extract_frame_jpeg(
    ffmpeg: str,
    source: Path,
    time_ms: int,
    width: int = FRAME_WIDTH,
    seek_time: str | None = None,
) -> bytes:
    seek_arg = (
        str(seek_time).strip()
        if seek_time not in (None, "")
        else f"{max(0, time_ms) / 1000:.3f}"
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", seek_arg,
        "-i", str(source),
        "-frames:v", "1",
        "-vf", f"scale={max(256, int(width))}:-2",
        "-q:v", "6",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags(),
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(err.strip() or "Gagal mengambil frame untuk Gemini.")
    return proc.stdout



def extract_silent_context_clip_mp4(
    ffmpeg: str,
    source: Path,
    center_ms: int,
    radius_ms: int = DEEP_CHECK_RADIUS_MS,
    width: int = DEEP_CHECK_WIDTH,
    fps: int = DEEP_CHECK_FPS,
) -> bytes:
    radius_ms = max(3000, min(int(radius_ms), 12_000))
    start_ms = max(0, int(center_ms) - radius_ms)
    duration_ms = radius_ms * 2
    with tempfile.TemporaryDirectory(prefix="minicut-deep-") as td:
        out = Path(td) / "deep-check.mp4"
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{start_ms / 1000:.3f}",
            "-i", str(source),
            "-t", f"{duration_ms / 1000:.3f}",
            "-map", "0:v:0",
            "-an",
            "-vf", f"scale={max(256, int(width))}:-2,fps={max(4, int(fps))}",
            "-c:v", "mpeg4",
            "-q:v", "10",
            "-movflags", "+faststart",
            str(out),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags(),
            check=False,
        )
        if proc.returncode != 0 or not out.is_file():
            err = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(err.strip() or "Gagal membuat video Deep Check.")
        data = out.read_bytes()
        if not data:
            raise RuntimeError("Video Deep Check kosong.")
        return data


def _response_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        raise RuntimeError(
            "Gemini tidak menghasilkan kandidat respons: "
            + json.dumps(feedback, ensure_ascii=False)
        )
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(p.get("text") or "") for p in parts).strip()
    if not text:
        raise RuntimeError("Respons Gemini kosong.")
    marker = chr(96) * 3
    if text.startswith(marker):
        text = text.strip(chr(96)).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _scene_window_prompt(
    target_ms: int,
    window_start_ms: int,
    window_end_ms: int,
    previous_summary: str = "",
    pass_label: str = "awal",
) -> str:
    prior = previous_summary.strip() or "(belum ada ringkasan sebelumnya)"
    return f"""
Anda memahami alur film untuk menentukan batas PART yang natural.

PATOKAN PART: {format_ms(target_ms)}
RENTANG YANG SEDANG DILIHAT: {format_ms(window_start_ms)}–{format_ms(window_end_ms)}
PASS: {pass_label}

RINGKASAN KONTINUITAS SEBELUMNYA:
{prior}

Anda menerima beberapa SEGMENT 30 detik. Setiap segment berisi:
- satu contact sheet visual berurutan;
- daftar timestamp frame F1, F2, dst.;
- SRT dari rentang waktu yang SAMA.

Pahami VISUAL + ISI SUBTITLE sebagai satu alur. Subtitle bukan hanya penjaga kalimat:
gunakan isi percakapan untuk memahami apakah topik, aksi, sebab-akibat, pertanyaan-jawaban,
dan kejadian masih berlanjut.

ATURAN UTAMA:
1. Target sekitar 15 menit hanya referensi. Jangan membuat cut hanya agar dekat target.
2. Jika lokasi, waktu, percakapan, aksi, atau kejadian masih satu rangkaian, JANGAN POTONG.
3. Cut kamera, close-up, angle baru, silence, atau akhir satu kalimat BUKAN alasan cukup.
4. Lokasi sama boleh dipotong hanya jika jelas ada time jump / kejadian baru / konteks baru.
5. Perubahan lokasi, waktu, siang↔malam, interior↔eksterior, establishing scene baru,
   atau percakapan yang jelas masuk konteks baru adalah bukti kuat batas scene.
6. Jangan membelah pertanyaan-jawaban, aksi-reaksi, sebab-akibat, gerakan penting,
   atau dialog yang masih menyambung.
7. Jika seluruh rentang masih satu scene, jangan dipaksa memilih titik.
8. Jika tidak ada cut natural, pilih EXPAND_FORWARD agar aplikasi mencari beberapa menit lagi.
9. CUT_FOUND hanya jika contact sheet + SRT benar-benar menunjukkan boundary scene.
10. Jangan mengarang timestamp. Untuk CUT_FOUND pilih SEGMENT dan FRAME yang tersedia
    paling dekat dengan awal scene baru / akhir scene lama.
11. continuity_summary harus ringkas tetapi cukup untuk pass berikutnya:
    lokasi, waktu, siapa/apa yang sedang terjadi, topik dialog, dan apakah adegan masih berlanjut.

Kembalikan HANYA JSON valid:
{{
  "decision": "CUT_FOUND",
  "segment_index": 1,
  "frame_index": 1,
  "confidence": 0.0,
  "same_scene": false,
  "same_location": false,
  "same_time_context": false,
  "dialogue_continues": false,
  "needs_video_check": false,
  "continuity_summary": "ringkasan keadaan terakhir untuk pass berikutnya",
  "reason": "alasan singkat dalam Bahasa Indonesia"
}}

decision hanya salah satu:
CUT_FOUND | NO_CUT | EXPAND_FORWARD | EXPAND_BACKWARD

Jika tidak ada boundary natural, gunakan EXPAND_FORWARD atau NO_CUT.
""".strip()


def _storyboard_prompt(
    target_ms: int,
    candidates: list[LocalCandidate],
    subtitles: SubtitleTrack | None,
) -> str:
    rows = []
    for i, candidate in enumerate(candidates, 1):
        rows.append(
            f"{i}. {format_ms(candidate.time_ms)} | "
            f"jarak target={candidate.distance_ms/1000:.1f}s | "
            f"visual_change={candidate.visual} | silence={candidate.silence} | "
            f"subtitle_gap={candidate.subtitle_gap} | subtitle_safe={candidate.subtitle_safe} | "
            f"local_score={candidate.score:.3f}"
        )
        if subtitles:
            excerpt = subtitles.nearby_text(
                candidate.time_ms, radius_ms=14_000, max_chars=1500
            )
            rows.append("SRT sekitar kandidat:\n" + (excerpt or "(tidak ada subtitle)"))

    return f"""
Anda menentukan batas Part film di sekitar patokan {format_ms(target_ms)}.

LOGIKA WAJIB:
1. Patokan sekitar 15 menit hanya REFERENSI, bukan waktu potong wajib.
2. Visual adalah sumber utama untuk memahami pergantian scene.
3. SRT hanya alat verifikasi dialog dan kontinuitas percakapan.
4. Bedakan SHOT dan SCENE. Angle, close-up, wide, shot-reverse-shot, atau cut kamera
   dalam percakapan/kejadian yang sama BUKAN scene baru.
5. Cari batas saat scene/adegan lama benar-benar selesai dan scene berikutnya mulai natural:
   pindah lokasi, pindah waktu, siang↔malam, interior↔eksterior, establishing scene baru,
   atau perubahan kejadian/konteks yang nyata.
6. JANGAN memilih titik yang membelah dialog, pertanyaan-jawaban, aksi, reaksi,
   sebab-akibat, gerakan penting, atau transisi yang masih satu rangkaian.
7. Keutuhan adegan dan alur LEBIH PENTING daripada kedekatan dengan patokan.
8. Kandidat diurutkan dari yang paling dekat target. Kandidat lebih jauh hanya boleh dipilih
   jika semua kandidat yang lebih dekat tidak valid sebagai boundary scene.
9. Setelah menemukan kandidat yang sudah cukup valid, berhenti; jangan mencari boundary
   yang lebih dramatis tetapi lebih jauh.
10. Jangan mengarang timestamp baru. Pilih kandidat yang tersedia.
11. Jika tidak ada kandidat yang valid, kembalikan NO_VALID_CANDIDATE.

Anda menerima storyboard refinement 4 frame per kandidat:
-5s, -1.2s, +1.2s, +5s.
Tidak ada video/audio pada tahap ini kecuali Deep Check memang diperlukan.

Kandidat:
{chr(10).join(rows)}

Kembalikan HANYA JSON valid:
{{
  "decision": "CUT_FOUND",
  "selected_candidate_index": 1,
  "confidence": 0.0,
  "scene_change": true,
  "dialog_safe": true,
  "action_safe": true,
  "needs_review": false,
  "cut_intent": "scene_transition",
  "reason": "alasan singkat dalam Bahasa Indonesia"
}}

Atau bila tidak ada kandidat valid:
{{
  "decision": "NO_VALID_CANDIDATE",
  "confidence": 0.0,
  "needs_review": false,
  "reason": "semua kandidat belum merupakan boundary scene yang layak"
}}
""".strip()


def _deep_check_prompt(
    target_ms: int,
    candidates: list[LocalCandidate],
    candidate_indexes: list[int],
    subtitles: SubtitleTrack | None,
) -> str:
    rows = []
    for idx in candidate_indexes:
        candidate = candidates[idx - 1]
        distance_ms = abs(candidate.time_ms - target_ms)
        tier = (
            "±45 dtk"
            if distance_ms <= 45_000
            else "±90 dtk"
            if distance_ms <= 90_000
            else "±120 dtk / lebih"
        )
        rows.append(
            f"KANDIDAT {idx}: {format_ms(candidate.time_ms)} | "
            f"jarak={distance_ms/1000:.1f}s | prioritas={tier} | "
            f"visual_change={candidate.visual} | silence={candidate.silence} | "
            f"subtitle_gap={candidate.subtitle_gap} | subtitle_safe={candidate.subtitle_safe}"
        )

    return f"""
PEMERIKSAAN FINAL titik potong film untuk target {format_ms(target_ms)}.

Anda menerima beberapa VIDEO PENDEK kandidat, masing-masing sekitar 20 detik total,
TANPA AUDIO, ditambah SRT sinkron pada rentang kandidat yang sama.

Kandidat SUDAH DIURUTKAN dari yang PALING DEKAT target ke yang lebih jauh:
{chr(10).join(rows)}

ATURAN WAJIB — NEAREST VALID WINS:
1. Nilai kandidat mulai dari nomor 1, lalu 2, lalu 3, lalu 4.
2. Jika kandidat yang lebih dekat SUDAH merupakan boundary scene yang cukup natural,
   PILIH kandidat itu dan BERHENTI. Jangan memilih kandidat lebih jauh hanya karena
   transisinya lebih dramatis atau lebih mudah terlihat.
3. Kandidat lebih jauh hanya boleh dipilih jika SEMUA kandidat yang lebih dekat TIDAK VALID.
4. Alasan tidak valid harus konkret: masih satu scene, hanya cut kamera/angle,
   dialog/topik masih menyambung, pertanyaan-jawaban belum selesai, aksi/reaksi belum selesai,
   atau belum ada perubahan konteks yang cukup.
5. Target 15/30/45/60 adalah referensi kuat. Setelah boundary cukup valid ditemukan,
   kedekatan ke target lebih penting daripada mencari boundary yang "paling kuat".
6. Gunakan VISUAL + ARTI SRT bersama-sama. Subtitle bukan cuma penjaga jeda.
7. Lokasi sama + waktu sama + kejadian/dialog masih satu rangkaian = JANGAN POTONG.
8. Lokasi sama tetap boleh menjadi scene baru jika ada time jump, kejadian baru,
   atau konteks dialog baru yang jelas.
9. Silence, akhir kalimat, atau pergantian shot saja tidak cukup.
10. Jika TIDAK ADA satu pun kandidat yang valid, pilih NO_VALID_CANDIDATE.
11. Jika boundary valid berada sedikit sebelum/sesudah pusat kandidat,
    preferred_offset_ms boleh ±1500 ms.

Contoh prinsip:
- 15:27 valid dan 16:18 juga valid → pilih 15:27.
- 15:27 hanya shot change dalam percakapan yang sama, 16:18 scene baru nyata → pilih 16:18.

Kembalikan HANYA JSON valid:
{{
  "decision": "CUT_FOUND",
  "selected_candidate_index": 1,
  "preferred_offset_ms": 0,
  "confidence": 0.0,
  "scene_change": true,
  "dialog_safe": true,
  "action_safe": true,
  "needs_review": false,
  "cut_intent": "scene_transition",
  "closer_candidates_rejected": [
    {{"index": 1, "reason": "isi hanya bila kandidat ini ditolak"}}
  ],
  "reason": "alasan singkat dalam Bahasa Indonesia"
}}

Jika tidak ada kandidat valid:
{{
  "decision": "NO_VALID_CANDIDATE",
  "confidence": 0.0,
  "needs_review": false,
  "reason": "semua kandidat masih satu scene / belum aman"
}}
""".strip()
