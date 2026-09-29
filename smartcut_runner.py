"""MiniCut SmartCut companion.

MiniCut mengekspor subtitle sebagai file .srt eksternal per part. SmartCut
upstream secara default ikut menyalin subtitle stream internal dari container
sumber. Itu dapat membuat player menampilkan subtitle lama/embedded alih-alih
SRT hasil MiniCut.

Companion ini juga menjaga kontrak frame MiniCut untuk cut exact-PTS. SmartCut
1.7 pada beberapa codec/GOP dapat membuang frame terakhir segmen ketika end
time tepat berada pada PTS frame berikutnya. MiniCut memverifikasi jumlah frame
hasil terhadap indeks PTS master. Bila tepat satu frame akhir hilang, companion
mengulang segmen dengan end-time satu frame master sesudah boundary. Hasil retry
harus memiliki jumlah frame persis; selain itu ekspor gagal, tidak pernah snap
atau menerima hasil yang ambigu.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from smartcut.media_container import MediaContainer


_original_media_container_init = MediaContainer.__init__


def _init_without_embedded_subtitles(self, *args, **kwargs):
    _original_media_container_init(self, *args, **kwargs)
    # smart_cut() menentukan stream subtitle output dari panjang list ini.
    # Kosongkan setelah source selesai dianalisis agar embedded subtitle tidak
    # ikut dimux. External .srt per part tetap dibuat oleh MiniCut sendiri.
    self.subtitle_tracks = []


MediaContainer.__init__ = _init_without_embedded_subtitles

from smartcut.__main__ import main as _smartcut_main  # noqa: E402


_TERMINAL_START = {"s", "start"}
_TERMINAL_END = {"e", "end", "-0"}
_MINICUT_PART_RE = re.compile(r"_Part-\d+\.[^.]+$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class _ExactKeepPlan:
    expected_frames: int
    retry_keep: str | None


def _as_fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _is_canonical_exact_token(token: str) -> bool:
    """True for MiniCut's normalized Fraction text, not ms fallback decimals."""
    text = str(token).strip().lower()
    if text in _TERMINAL_START or text in _TERMINAL_END:
        return True
    # core.py emits exact PTS via str(Fraction): either integer or n/d.
    # Non-exact fallback timestamps are always fixed decimals (x.xxxxxx), so
    # requiring integer/fraction syntax lets the companion distinguish them.
    if not re.fullmatch(r"-?\d+(?:/\d+)?", text):
        return False
    try:
        Fraction(text)
    except (ValueError, ZeroDivisionError):
        return False
    return True


def _looks_like_minicut_exact_keep(argv: list[str]) -> bool:
    if "--frames" in argv or "--keep" not in argv or len(argv) < 2:
        return False
    try:
        keep_index = argv.index("--keep")
        keep = argv[keep_index + 1]
    except (ValueError, IndexError):
        return False
    values = [item.strip() for item in keep.split(",")]
    if len(values) != 2:
        return False
    # Restrict automatic hardening to MiniCut's own generated Part filenames.
    # This avoids changing generic standalone SmartCut CLI semantics.
    if not _MINICUT_PART_RE.search(Path(argv[1]).name):
        return False
    non_terminal = [
        value for value in values
        if value.lower() not in _TERMINAL_START | _TERMINAL_END
    ]
    return bool(non_terminal) and all(
        _is_canonical_exact_token(value) for value in values
    )


def _video_time_base(source: MediaContainer) -> Fraction:
    stream = getattr(source, "video_stream", None)
    if stream is None or getattr(stream, "time_base", None) is None:
        raise RuntimeError("Video SmartCut tidak memiliki time_base untuk verifikasi PTS.")
    return _as_fraction(stream.time_base)


def _relative_frame_time(source: MediaContainer, index: int) -> Fraction:
    pts_values = getattr(source, "video_frame_times_pts", None)
    if pts_values is None or index < 0 or index >= len(pts_values):
        raise IndexError("Index frame master di luar rentang video.")
    absolute = Fraction(int(pts_values[index])) * _video_time_base(source)
    return absolute - _as_fraction(getattr(source, "start_time", Fraction(0)))


def _exact_frame_index(source: MediaContainer, exact_time: str) -> int:
    target = Fraction(str(exact_time).strip())
    pts_values = getattr(source, "video_frame_times_pts", None)
    if pts_values is None:
        raise RuntimeError("Daftar PTS frame master SmartCut tidak tersedia.")
    time_base = _video_time_base(source)
    start_time = _as_fraction(getattr(source, "start_time", Fraction(0)))
    target_absolute = target + start_time

    # Compare in the integer PTS domain whenever possible. This avoids float
    # tolerance and guarantees MiniCut never silently maps to a nearby frame.
    pts_fraction = target_absolute / time_base
    if pts_fraction.denominator != 1:
        raise RuntimeError(
            f"Exact PTS {target} tidak berada pada grid time_base video {time_base}."
        )
    target_pts = pts_fraction.numerator
    for index, raw_pts in enumerate(pts_values):
        if int(raw_pts) == target_pts:
            return index
    raise RuntimeError(
        f"Exact PTS {target} tidak ditemukan pada frame master SmartCut."
    )


def _build_exact_keep_plan(source: MediaContainer, keep: str) -> _ExactKeepPlan:
    values = [item.strip() for item in keep.split(",")]
    if len(values) != 2:
        raise RuntimeError("MiniCut exact keep harus memiliki satu start dan satu end.")
    start_raw, end_raw = values
    total_frames = len(getattr(source, "video_frame_times_pts", []))
    if total_frames <= 0:
        raise RuntimeError("Video tidak memiliki frame master untuk diverifikasi.")

    if start_raw.lower() in _TERMINAL_START:
        start_index = 0
    else:
        start_index = _exact_frame_index(source, start_raw)

    retry_keep: str | None = None
    if end_raw.lower() in _TERMINAL_END:
        end_exclusive = total_frames
    else:
        end_exclusive = _exact_frame_index(source, end_raw)
        # SmartCut 1.7 sometimes drops the final desired frame when the end is
        # exactly this boundary. Retry may expose one additional source frame to
        # the muxer; output validation below requires the exact expected count.
        next_index = end_exclusive + 1
        if next_index < total_frames:
            retry_end = _relative_frame_time(source, next_index)
            retry_keep = f"{start_raw},{retry_end}"

    expected = end_exclusive - start_index
    if expected <= 0:
        raise RuntimeError(
            f"Rentang exact SmartCut tidak valid: start={start_raw}, end={end_raw}."
        )
    return _ExactKeepPlan(expected_frames=expected, retry_keep=retry_keep)


def _video_frame_count(path: Path) -> int:
    media = MediaContainer(str(path))
    try:
        return len(getattr(media, "video_frame_times_pts", []))
    finally:
        media.close()


def _run_upstream(argv: list[str]) -> None:
    previous = list(sys.argv)
    try:
        sys.argv = [previous[0], *argv]
        _smartcut_main()
    finally:
        sys.argv = previous


def _replace_keep(argv: list[str], keep: str) -> list[str]:
    rewritten = list(argv)
    index = rewritten.index("--keep")
    rewritten[index + 1] = keep
    return rewritten


def main() -> None:
    argv = list(sys.argv[1:])
    if not _looks_like_minicut_exact_keep(argv):
        _run_upstream(argv)
        return

    keep_index = argv.index("--keep")
    keep = argv[keep_index + 1]
    source = MediaContainer(argv[0])
    try:
        plan = _build_exact_keep_plan(source, keep)
    finally:
        source.close()

    _run_upstream(argv)
    output = Path(argv[1])
    actual = _video_frame_count(output)
    if actual == plan.expected_frames:
        return

    # Only compensate the precise SmartCut 1.7 failure we reproduced: exactly
    # one final frame missing. Any other mismatch is a hard failure.
    if actual == plan.expected_frames - 1 and plan.retry_keep:
        _run_upstream(_replace_keep(argv, plan.retry_keep))
        retry_actual = _video_frame_count(output)
        if retry_actual == plan.expected_frames:
            return
        raise RuntimeError(
            "SmartCut retry exact-frame tetap tidak cocok: "
            f"hasil {retry_actual}, seharusnya {plan.expected_frames}."
        )

    raise RuntimeError(
        "SmartCut menghasilkan jumlah frame yang tidak sesuai boundary exact: "
        f"hasil {actual}, seharusnya {plan.expected_frames}."
    )


if __name__ == "__main__":
    main()
