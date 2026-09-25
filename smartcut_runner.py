"""MiniCut SmartCut companion.

MiniCut mengekspor subtitle sebagai file .srt eksternal per part. SmartCut
upstream secara default ikut menyalin subtitle stream internal dari container
sumber. Itu dapat membuat player menampilkan subtitle lama/embedded alih-alih
SRT hasil MiniCut. Companion ini menonaktifkan output subtitle internal sambil
tetap mempertahankan video dan audio SmartCut.
"""

from smartcut.media_container import MediaContainer


_original_media_container_init = MediaContainer.__init__


def _init_without_embedded_subtitles(self, *args, **kwargs):
    _original_media_container_init(self, *args, **kwargs)
    # smart_cut() menentukan stream subtitle output dari panjang list ini.
    # Kosongkan setelah source selesai dianalisis agar embedded subtitle tidak
    # ikut dimux. External .srt per part tetap dibuat oleh MiniCut sendiri.
    self.subtitle_tracks = []


MediaContainer.__init__ = _init_without_embedded_subtitles

from smartcut.__main__ import main  # noqa: E402


if __name__ == "__main__":
    main()
