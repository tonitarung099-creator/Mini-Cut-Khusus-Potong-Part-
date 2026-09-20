# Mini Cut Khusus Potong Part

Aplikasi desktop Windows untuk membagi film/video menjadi beberapa part dengan bantuan AI Gemini, tetap hemat API dan frame-accurate.

## Fitur utama

- **Scene Boundary Analyzer** — mengikuti logika pemotongan berbasis keutuhan adegan: patokan 15 menit hanya referensi, shot biasa bukan scene baru, dan keutuhan alur lebih penting daripada tepat waktu.
- **Analisis adaptif Gemini** — MiniCut mencari **hingga 6 kandidat secara lokal**, lalu Gemini membandingkan **6-frame storyboard per kandidat + SRT**.
- **Deep Check hemat** — video visual pendek ±5 detik hanya dikirim untuk 1–2 kandidat ketika storyboard masih ragu. Video ±2 menit tidak dikirim dan Deep Check tidak mengirim audio.
- **SRT sebagai penjaga dialog** — membantu menghindari cut di tengah dialog/percakapan.
- **Exact-frame resolver** — keputusan AI dikunci ke PTS frame asli dari master.
- **SmartCut Frame Accurate** — default export; meminimalkan re-encode di sekitar titik potong.
- **Fast Copy** — opsi ekspor cepat berbasis keyframe.
- **Preview Proxy lokal** — proxy H.264 720p untuk playback/scrubbing lebih ringan.
- **Playback 0.5x–4x** dan tombol maju/mundur 1 frame berbasis PTS master.
- **Gemini API Manager** — hingga 100 API key, disimpan lokal menggunakan Windows DPAPI.
- **Cache/resume AI Film Cut**.
- **MCP companion + local bridge** untuk integrasi agent eksternal.

## Cara kerja AI Film Cut

```text
Target sekitar 15 menit
        ↓
MiniCut lokal scan ± window
        ↓
hingga 6 kandidat scene
        ↓
Gemini: storyboard 6 frame + SRT
        ↓
cukup yakin? ── tidak ──→ Deep Check video visual pendek kandidat terbaik
        ↓ ya                         ↓
        └──────── pilih batas scene natural
                         ↓
              MiniCut kunci ke PTS frame master
                         ↓
                   SmartCut export
```

Target 15 menit adalah patokan, bukan batas wajib. Perpindahan scene yang natural lebih diprioritaskan.

## Struktur

- `main.py` — entry point aplikasi.
- `minicut_agent/ui.py` — UI PySide6.
- `minicut_agent/core.py` — proyek, FFmpeg, proxy, export.
- `minicut_agent/candidates.py` — pencarian/ranking kandidat lokal.
- `minicut_agent/gemini.py` — Gemini verifier visual-first.
- `minicut_agent/frame_resolver.py` — penguncian ke PTS frame asli.
- `minicut_agent/gemini_keys.py` — manager API key lokal.
- `minicut_agent/subtitles.py` — SRT.
- `minicut_agent/workers.py` — background workers.
- `mcp_server.py` — companion MCP.
- `smartcut_runner.py` — companion SmartCut.
- `tests/` — unit tests.
- `.github/workflows/build-windows.yml` — build Windows otomatis.

## Kebutuhan lokal

- Windows
- **FFmpeg + ffprobe tersedia di PATH**
- Untuk menjalankan dari source: Python 3.11

```powershell
pip install -r requirements-dev.txt
python main.py
```

## Build Windows

GitHub Actions otomatis menjalankan compile check, unit test, PyInstaller build, SmartCut smoke test, aplikasi smoke test, lalu mengunggah ZIP Windows sebagai artifact.

Build manual:

```bat
build_windows.bat
```

## Keamanan API key

Gemini API key tidak dimasukkan ke source code atau cache proyek. Penyimpanan lokal menggunakan Windows DPAPI dan terikat ke user Windows.

## Third-party

SmartCut digunakan sebagai engine frame-accurate cutting. Lihat `THIRD_PARTY_SMARTCUT.txt` untuk notice lisensi MIT.
