# Mini Cut Khusus Potong Part

Aplikasi desktop Windows untuk membagi film/video menjadi beberapa part dengan bantuan AI Gemini, tetap hemat API dan frame-accurate.

## Fitur utama

- **AI Film Cut visual-first** — mengutamakan perpindahan scene/konteks visual besar seperti pindah lokasi, siang ↔ malam, interior ↔ eksterior, atau adegan baru.
- **Hemat Gemini API** — MiniCut mencari kandidat secara lokal. Gemini hanya menerima **3 frame kecil per kandidat + SRT + metadata lokal**, bukan video penuh dan bukan audio.
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
kandidat visual / silence / SRT
        ↓
Gemini: 3 frame + SRT
        ↓
pilih batas scene paling natural
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
