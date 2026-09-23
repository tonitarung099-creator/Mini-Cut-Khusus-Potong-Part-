# Mini Cut Khusus Potong Part

Aplikasi desktop Windows untuk membagi film/video menjadi beberapa part dengan bantuan AI Gemini, tetap hemat API dan frame-accurate.

## Fitur utama

- **Gemini Chat Command Agent** — chat tidak lagi terbatas pada format cut manual. Gemini memahami bahasa natural lalu memakai tool MiniCut untuk timeline/playback/proyek. Contoh `cut 1 jam lebih 2 menit`, `bagi jadi 8 part`, atau perintah majemuk. Untuk cut waktu spesifik, timestamp yang dikeluarkan Gemini disimpan **apa adanya** tanpa snap/frame-lock lokal.
- **Scene Boundary Analyzer** — grid target tetap absolut (mis. 15, 30, 45, 60 menit). Boundary natural boleh bergeser, tetapi tidak menggeser target berikutnya.
- **Contact sheet + SRT sinkron** — untuk window awal ±2 menit, MiniCut membuat contact sheet lokal sekitar 1 frame/3 detik dan mengirim SRT pada blok waktu yang sama agar Gemini memahami visual + isi percakapan bersama-sama.
- **NO CUT / Expand** — bila visual dan dialog masih satu rangkaian, Gemini boleh memilih NO CUT. MiniCut lalu menambah pencarian sekitar +3 menit tanpa memaksa cut dan tanpa menggeser grid target berikutnya.
- **Refinement + Deep Check hemat** — setelah area boundary ditemukan, kandidat lokal diperiksa lebih rapat. Video visual pendek sekitar ±8 detik hanya dikirim jika boundary masih ambigu; audio tidak dikirim karena dialog dipahami dari SRT.
- **SRT sebagai penjaga dialog** — membantu menghindari cut di tengah dialog/percakapan.
- **Gemini exact-frame authority** — pada AI Film Cut, MiniCut hanya membaca daftar PTS frame master di sekitar boundary dan menampilkannya ke Gemini. Gemini memilih frame final; MiniCut tidak meranking, snap, atau menggeser pilihan itu lagi.
- **SmartCut Frame Accurate** — default export; meminimalkan re-encode di sekitar titik potong.
- **Portable FFmpeg + ffprobe** — build Windows membawa tool media sendiri; pengguna ZIP tidak perlu menginstal FFmpeg atau mengatur PATH.
- **Fast Copy** — opsi ekspor cepat berbasis keyframe.
- **Preview master-direct** — tidak membuat proxy. Backend utama mpv/libmpv dengan hardware decoding; Qt Multimedia menjadi fallback.
- **Scrubbing dua tahap** — saat drag memakai seek cepat, saat dilepas dikunci ke posisi exact pada master asli.
- **Playback 0.5x–4x** dan tombol maju/mundur 1 frame berbasis PTS master.
- **Gemini API Manager** — hingga 100 API key, disimpan lokal menggunakan Windows DPAPI.
- **Cache/resume AI Film Cut**.
- **MCP companion + local bridge** untuk integrasi agent eksternal.

## Cara kerja AI Film Cut

```text
Grid target absolut: 15 → 30 → 45 → 60 → ...
        ↓
Window awal ±2 menit di target aktif
        ↓
contact sheet 30 detik + SRT yang sinkron
        ↓
Gemini memahami lokasi + waktu + kejadian + isi dialog
        ↓
CUT_FOUND ? ── tidak ──→ NO CUT → tambah +3 menit → analisis bagian tambahan
        ↓ ya
frame PTS master sekitar boundary ditampilkan ke Gemini
        ↓
Gemini memilih frame master FINAL
        ↓
timestamp Gemini diteruskan tanpa snap lokal
        ↓
SmartCut export pada timestamp yang sama
```

Target 15 menit adalah patokan, bukan batas wajib. Perpindahan scene yang natural lebih diprioritaskan.

## Struktur

- `main.py` — entry point aplikasi.
- `minicut_agent/ui.py` — UI PySide6.
- `minicut_agent/core.py` — proyek, FFmpeg, dan export.
- `minicut_agent/preview_player.py` — playback master-direct mpv/libmpv + fallback Qt.
- `minicut_agent/candidates.py` — utilitas kandidat lama/pendukung.
- `minicut_agent/gemini.py` — pemahaman scene + pemilihan frame master final oleh Gemini.
- `minicut_agent/frame_resolver.py` — pembacaan PTS frame master dan utilitas kompatibilitas.
- `minicut_agent/gemini_keys.py` — manager API key lokal.
- `minicut_agent/subtitles.py` — SRT.
- `minicut_agent/workers.py` — background workers.
- `mcp_server.py` — companion MCP.
- `smartcut_runner.py` — companion SmartCut.
- `tests/` — unit tests.
- `.github/workflows/build-windows.yml` — build Windows otomatis.

## Kebutuhan lokal

- Windows
- Untuk menjalankan langsung dari source: **FFmpeg + ffprobe tersedia di PATH**
- Python 3.11

```powershell
pip install -r requirements-dev.txt
python main.py
```

## Build Windows

GitHub Actions otomatis mengambil runtime FFmpeg/libmpv yang dipin, menjalankan compile check, unit test, PyInstaller build, SmartCut smoke test, aplikasi smoke test, lalu mengunggah ZIP Windows portable sebagai artifact.

Build manual:

```bat
build_windows.bat
```

## Keamanan API key

Gemini API key tidak dimasukkan ke source code atau cache proyek. Penyimpanan lokal menggunakan Windows DPAPI dan terikat ke user Windows.

## Third-party

SmartCut digunakan sebagai engine frame-accurate cutting. Lihat `THIRD_PARTY_SMARTCUT.txt`.
Preview memakai python-mpv + libmpv runtime terpisah. Lihat `THIRD_PARTY_MPV.txt` untuk sumber dan notice lisensi.
Build portable juga menyertakan FFmpeg/ffprobe dari build Windows LGPL. Lihat `THIRD_PARTY_FFMPEG.txt`.
