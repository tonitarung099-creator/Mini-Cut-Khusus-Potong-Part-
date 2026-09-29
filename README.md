# Mini Cut Khusus Potong Part

Aplikasi desktop Windows untuk membagi film/video menjadi beberapa part dengan bantuan AI Gemini, tetap hemat API dan frame-accurate.

## Fitur utama

- **Manual Camera Boundary Cut** — timestamp manual diperlakukan sebagai target waktu. MiniCut mencari pergantian kamera/shot terdekat secara lokal pada radius awal ±2 detik, lalu memperluas sampai ±4 detik bila perlu. Boundary visual yang ditemukan dikunci ke PTS frame master rasional asli sebelum diteruskan ke SmartCut. Jika tidak ada pergantian kamera pada radius tersebut, MiniCut fallback ke frame master nyata terdekat dan memberi status fallback.
- **Daftar titik potong tanpa API** — format seperti `Titik Potong 1 : 00:14:55.500` dapat ditempel langsung ke Chat. Jika pesan berisi daftar timestamp eksplisit, MiniCut mem-parsing daftar tersebut secara lokal dan memproses semua titik ke camera boundary tanpa menghabiskan request Gemini.
- **Gemini Chat Command Agent** — memahami bahasa natural lalu memakai tool MiniCut untuk timeline/playback/proyek. Contoh `cut 1 jam lebih 2 menit`, `bagi jadi 8 part`, atau perintah majemuk. Perintah timestamp eksplisit yang dikenali parser lokal tidak perlu dikirim ke API.
- **Scene Boundary Analyzer** — grid target tetap absolut (mis. 15, 30, 45, 60 menit). Boundary natural boleh bergeser, tetapi tidak menggeser target berikutnya.
- **Contact sheet + SRT sinkron** — untuk window awal ±2 menit, MiniCut membuat contact sheet lokal sekitar 1 frame/2 detik dan mengirim SRT pada blok waktu yang sama agar Gemini memahami visual + isi percakapan bersama-sama.
- **NO CUT / Expand** — bila visual dan dialog masih satu rangkaian, Gemini boleh memilih NO CUT. MiniCut lalu menambah pencarian sekitar +3 menit tanpa memaksa cut dan tanpa menggeser grid target berikutnya.
- **Pemilihan frame final oleh Gemini untuk AI Film Cut** — setelah area boundary ditemukan, MiniCut membaca frame PTS nyata di sekitar boundary dan mengirim preview frame-frame itu ke Gemini dalam dua tahap (coarse lalu fine). Jalur AI Film Cut tetap terpisah dari manual camera-boundary cut.
- **SRT wajib untuk AI Film Cut dan SmartCut** — AI Film Cut memakai SRT sinkron untuk memahami dialog. Saat mode SmartCut dipakai, ekspor wajib menghasilkan pasangan video + SRT; MiniCut memakai SRT yang sudah dipilih, referensi SRT proyek, atau `NamaVideo.srt` yang terdeteksi otomatis, dan akan meminta SRT jika belum ada yang valid. SRT dipotong menjadi `NamaVideo_Part-01.srt`, `Part-02.srt`, dst.; timestamp tiap part di-reset mulai `00:00:00,000`, dan cue yang melewati boundary diklip ke kedua part tanpa mengubah teks. **Fast Copy mengekspor video saja dan tidak memerlukan atau membuat SRT.**
- **Gemini exact-frame authority untuk AI Film Cut** — pada AI Film Cut, MiniCut hanya membaca daftar PTS frame master di sekitar boundary dan menampilkannya ke Gemini. Gemini memilih frame final; PTS rasional asli disimpan sampai SmartCut sehingga tidak dibulatkan ke milidetik untuk render.
- **SmartCut Frame Accurate** — default export; meminimalkan re-encode di sekitar titik potong.
- **Portable FFmpeg + ffprobe** — build Windows membawa tool media sendiri; pengguna ZIP tidak perlu menginstal FFmpeg atau mengatur PATH.
- **Fast Copy** — opsi ekspor cepat berbasis keyframe, video-only tanpa SRT.
- **Preview master-direct** — tidak membuat proxy. Backend utama mpv/libmpv dengan hardware decoding; Qt Multimedia menjadi fallback.
- **Scrubbing dua tahap** — saat drag memakai seek cepat, saat dilepas dikunci ke posisi exact pada master asli.
- **Playback 0.5x–4x** dan tombol maju/mundur 1 frame berbasis PTS master.
- **Gemini API Manager** — hingga 100 API key, disimpan lokal menggunakan Windows DPAPI. Jika key aktif terkena limit, MiniCut dapat berpindah otomatis ke key lain yang masih dapat dipakai tanpa mencoba key yang sama berulang kali.
- **Import banyak API dari TXT** — tidak perlu memberi nama API atau project. Tekan `Import TXT`, isi file dengan satu API key per baris, lalu MiniCut menambahkan semuanya sekaligus. Baris kosong, komentar, dan key duplikat dilewati otomatis; batas maksimum tetap 100 key.
- **Cache/resume AI Film Cut** — hasil CUT, NO CUT, dan grid yang dilewati dapat dipulihkan saat failover/retry agar target selesai tidak dianalisis ulang.
- **MCP companion + local bridge** untuk integrasi agent eksternal.

## Import banyak Gemini API dari TXT

Di tab **API**, pilih **Import TXT**. Format paling sederhana:

```text
AIza...................................
AIza...................................
AIza...................................
```

Nama API tidak perlu ditulis. MiniCut memberi nomor otomatis seperti `API 001`, `API 002`, dan seterusnya. Format `.env` seperti `GEMINI_API_KEY=...` juga diterima. Import tidak otomatis mengetes API secara online, sehingga tidak menghabiskan request hanya karena file dimasukkan. Untuk pengecekan, gunakan tombol **Cek Semua API Online** secara manual.

## Cara kerja cut manual / daftar titik potong

```text
Timestamp target pengguna
        ↓
Cari pergantian kamera ±2 detik
        ↓ tidak ditemukan
Perluas pencarian sampai ±4 detik
        ↓
Pilih pergantian kamera terdekat dari target
        ↓
Kunci ke PTS frame master rasional asli
        ↓
SmartCut frame-accurate
```

Jika tidak ditemukan pergantian kamera pada radius maksimal, MiniCut tidak mengarang boundary visual. Aplikasi memakai frame master nyata terdekat dan menandainya sebagai fallback.

Contoh input Chat:

```text
Samawa 2025
===========
Titik Potong 1 : 00:14:55.500
Titik Potong 2 : 00:31:14.750
Titik Potong 3 : 00:49:21.500
Titik Potong 4 : 01:02:13.000
Titik Potong 5 : 01:17:35.500
Titik Potong 6 : 01:30:57.000
```

Keenam timestamp di atas adalah target. Hasil final dapat bergeser beberapa frame agar potongan terjadi tepat pada pergantian kamera.

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
PTS rasional asli diteruskan ke SmartCut
```

Target 15 menit adalah patokan, bukan batas wajib. Perpindahan scene yang natural lebih diprioritaskan.

## Struktur

- `main.py` — entry point aplikasi; memakai runtime dengan bulk API import.
- `minicut_agent/ui.py` — UI PySide6 dasar.
- `minicut_agent/bulk_api_window.py` — layer UI untuk tambah API tanpa nama dan import TXT massal.
- `minicut_agent/gemini_key_import.py` — parser TXT, deduplikasi, dan import API key massal.
- `minicut_agent/camera_cut_window.py` — layer UI aktif untuk manual/chat camera-boundary cut.
- `minicut_agent/camera_boundary.py` — deteksi pergantian kamera lokal dan penguncian ke PTS master.
- `minicut_agent/core.py` — proyek, FFmpeg, dan export.
- `minicut_agent/preview_player.py` — playback master-direct mpv/libmpv + fallback Qt.
- `minicut_agent/candidates.py` — scene detection lokal dan utilitas kandidat.
- `minicut_agent/gemini.py` — pemahaman scene + pemilihan frame master final oleh Gemini untuk AI Film Cut.
- `minicut_agent/frame_resolver.py` — pembacaan PTS frame master dan utilitas kompatibilitas.
- `minicut_agent/gemini_keys.py` — manager API key lokal.
- `minicut_agent/subtitles.py` — pembacaan SRT, sinkronisasi dialog, dan ekspor SRT per part.
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

Gemini API key tidak dimasukkan ke source code atau cache proyek. Penyimpanan lokal menggunakan Windows DPAPI dan terikat ke user Windows. Import TXT hanya membaca file lokal yang dipilih pengguna; isi key tidak ditulis ke log aplikasi.

## Third-party

SmartCut digunakan sebagai engine frame-accurate cutting. Lihat `THIRD_PARTY_SMARTCUT.txt`.
Preview memakai python-mpv + libmpv runtime terpisah. Lihat `THIRD_PARTY_MPV.txt` untuk sumber dan notice lisensi.
Build portable juga menyertakan FFmpeg/ffprobe dari build Windows LGPL. Lihat `THIRD_PARTY_FFMPEG.txt`.