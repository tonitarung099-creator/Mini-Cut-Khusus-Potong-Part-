from __future__ import annotations

import base64
import json
import subprocess
import tempfile
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

# Stage 1 is token-conscious: a wider storyboard + SRT, never the ±2 minute video.
# Stage 2 sends short SILENT clips only when Stage 1 is uncertain.
STORYBOARD_OFFSETS_MS = (-8000, -4000, -1000, 1000, 4000, 8000)
FRAME_WIDTH = 384
DEEP_CHECK_CONFIDENCE = 0.72
DEEP_CHECK_RADIUS_MS = 5000
DEEP_CHECK_WIDTH = 360
DEEP_CHECK_FPS = 8
SEMANTIC_ZONE_LIMIT_MS = 1500


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

    def _post(self, payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
        url = f"{API_ROOT}/models/{self.model}:generateContent"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("error", {}).get("message") or body
            except Exception:
                detail = body
            raise RuntimeError(f"Gemini API {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Tidak dapat terhubung ke Gemini: {exc.reason}") from exc

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

    def verify_candidates(
        self,
        ffmpeg: str,
        source: Path,
        target_ms: int,
        candidates: list[LocalCandidate],
        subtitles: SubtitleTrack | None,
        frame_width: int = FRAME_WIDTH,
        allow_deep_check: bool = True,
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("Tidak ada kandidat untuk diverifikasi Gemini.")

        # TAHAP 1 — storyboard lebar + SRT. Gemini membandingkan semua kandidat
        # dalam satu keputusan, tetapi tidak menerima video ±2 menit.
        prompt = _storyboard_prompt(target_ms, candidates, subtitles)
        parts: list[dict[str, Any]] = [{"text": prompt}]
        labels = ("-8s", "-4s", "-1s", "+1s", "+4s", "+8s")

        for i, candidate in enumerate(candidates, 1):
            parts.append({
                "text": (
                    f"KANDIDAT {i} · pusat {format_ms(candidate.time_ms)} · "
                    "urutan storyboard sebelum → sesudah"
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
                "temperature": 0.1,
                "maxOutputTokens": 850,
                "responseMimeType": "application/json",
            },
        }
        data = self._post(payload)
        text = _response_text(data)
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini tidak mengembalikan JSON valid pada storyboard.") from exc

        selected_index = _candidate_index(result.get("selected_candidate_index"), len(candidates))
        alternate_index = _optional_candidate_index(
            result.get("alternate_candidate_index"), len(candidates), selected_index
        )

        deep_needed = allow_deep_check and _needs_deep_check(result)
        if deep_needed:
            if alternate_index is None:
                alternate_index = _best_alternate_index(candidates, selected_index)
            compare_indexes = [selected_index]
            if alternate_index and alternate_index != selected_index:
                compare_indexes.append(alternate_index)

            deep = self._deep_check_candidates(
                ffmpeg=ffmpeg,
                source=source,
                target_ms=target_ms,
                candidates=candidates,
                candidate_indexes=compare_indexes,
                subtitles=subtitles,
            )
            selected_index = _candidate_index(
                deep.get("selected_candidate_index"), len(candidates)
            )
            # Keep the deep decision fields but retain the original storyboard
            # diagnostics for audit.
            result["storyboard_selected_candidate_index"] = result.get(
                "selected_candidate_index"
            )
            result["storyboard_confidence"] = result.get("confidence")
            result["storyboard_reason"] = result.get("reason")
            result.update(deep)
            result["deep_check_used"] = True
        else:
            result["deep_check_used"] = False

        selected = candidates[selected_index - 1]
        preferred_off = _bounded_offset(result.get("preferred_offset_ms"), 0)
        preferred_ms = max(0, selected.time_ms + preferred_off)

        result["selected_candidate_index"] = selected_index
        result["candidate_time_ms"] = selected.time_ms
        result["candidate_time"] = format_ms(selected.time_ms)
        result["preferred_time_ms"] = preferred_ms
        result["boundary_start_ms"] = max(0, preferred_ms - 700)
        result["boundary_end_ms"] = preferred_ms + 700
        result["new_content_starts_ms"] = None
        result["target_ms"] = target_ms
        result["target"] = format_ms(target_ms)
        result["analysis_mode"] = (
            "storyboard+srt+deep-silent-video"
            if result["deep_check_used"]
            else "storyboard+srt"
        )
        result["frames_per_candidate"] = len(STORYBOARD_OFFSETS_MS)
        result["candidate_count"] = len(candidates)
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
        # TAHAP 2 — hanya kandidat ambigu yang mendapat cuplikan pendek.
        # Cuplikan sengaja TANPA AUDIO; SRT tetap menjadi verifikasi dialog.
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
            parts.append({
                "text": (
                    f"KANDIDAT {idx} · video visual pendek ±"
                    f"{DEEP_CHECK_RADIUS_MS/1000:.0f}s · TANPA AUDIO"
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
                "temperature": 0.05,
                "maxOutputTokens": 700,
                "responseMimeType": "application/json",
            },
        }
        data = self._post(payload)
        text = _response_text(data)
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini tidak mengembalikan JSON valid pada Deep Check.") from exc

        selected = _candidate_index(result.get("selected_candidate_index"), len(candidates))
        if selected not in candidate_indexes:
            raise RuntimeError("Deep Check memilih kandidat di luar kandidat yang dibandingkan.")
        result["selected_candidate_index"] = selected
        result["needs_review"] = bool(result.get("needs_review", False))
        result["deep_check_compared"] = candidate_indexes
        return result

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


def extract_frame_jpeg(
    ffmpeg: str,
    source: Path,
    time_ms: int,
    width: int = FRAME_WIDTH,
) -> bytes:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{max(0, time_ms) / 1000:.3f}",
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
    radius_ms = max(3000, min(int(radius_ms), 7000))
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
8. Bandingkan SEMUA kandidat, jangan hanya mengikuti local_score.
9. Jangan mengarang timestamp baru. Pilih kandidat yang tersedia.
10. Jika storyboard tidak cukup untuk memastikan gerakan/transisi, minta Deep Check.

Anda menerima storyboard 6 frame per kandidat:
-8s, -4s, -1s, +1s, +4s, +8s.
Tidak ada video/audio pada tahap ini.

Kandidat:
{chr(10).join(rows)}

Kembalikan HANYA JSON valid:
{{
  "selected_candidate_index": 1,
  "alternate_candidate_index": 2,
  "confidence": 0.0,
  "scene_change": true,
  "dialog_safe": true,
  "action_safe": true,
  "needs_deep_check": false,
  "needs_review": false,
  "cut_intent": "scene_transition",
  "reason": "alasan singkat dalam Bahasa Indonesia"
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
        rows.append(
            f"KANDIDAT {idx}: {format_ms(candidate.time_ms)} | "
            f"jarak={candidate.distance_ms/1000:.1f}s | "
            f"visual_change={candidate.visual} | subtitle_safe={candidate.subtitle_safe}"
        )
        if subtitles:
            excerpt = subtitles.nearby_text(
                candidate.time_ms, radius_ms=14_000, max_chars=1800
            )
            rows.append("SRT:\n" + (excerpt or "(tidak ada subtitle)"))

    return f"""
DEEP CHECK titik potong film sekitar patokan {format_ms(target_ms)}.

Anda hanya membandingkan kandidat berikut:
{chr(10).join(rows)}

Anda menerima VIDEO VISUAL PENDEK sekitar ±5 detik untuk setiap kandidat.
VIDEO TIDAK MEMILIKI AUDIO. Jangan mengarang fakta audio.
Gunakan SRT untuk dialog.

Tentukan kandidat yang paling sesuai aturan:
- scene lama selesai secara natural;
- scene berikutnya masuk wajar;
- bukan sekadar pergantian shot/kamera;
- tidak membelah dialog, pertanyaan-jawaban, aksi, reaksi, sebab-akibat;
- perubahan lokasi/waktu/konteks besar adalah bukti kuat;
- keutuhan adegan/alur lebih penting daripada tepat 15 menit.

Jika gerak visual menunjukkan batas ideal sedikit sebelum/sesudah pusat kandidat,
preferred_offset_ms boleh diisi dalam ±1500 ms. Jika pusat sudah tepat, gunakan 0.

Kembalikan HANYA JSON valid:
{{
  "selected_candidate_index": {candidate_indexes[0]},
  "preferred_offset_ms": 0,
  "confidence": 0.0,
  "scene_change": true,
  "dialog_safe": true,
  "action_safe": true,
  "needs_review": false,
  "cut_intent": "scene_transition",
  "reason": "alasan Deep Check singkat dalam Bahasa Indonesia"
}}
""".strip()
