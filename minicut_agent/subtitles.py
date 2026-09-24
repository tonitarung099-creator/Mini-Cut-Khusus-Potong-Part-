from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TIME_RE = re.compile(
    r"(?<!\d)(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})"
    r"[,.](?P<ms>\d{1,3})(?!\d)"
)

def _time_ms(text: str) -> int:
    m = _TIME_RE.search(text.strip())
    if not m:
        raise ValueError(f"Timestamp SRT tidak valid: {text}")
    hours = int(m.group("h"))
    minutes = int(m.group("m"))
    seconds = int(m.group("s"))
    if not (0 <= minutes < 60 and 0 <= seconds < 60):
        raise ValueError(f"Timestamp SRT di luar rentang: {text}")
    milli = int(m.group("ms").ljust(3, "0"))
    return (
        hours * 3_600_000
        + minutes * 60_000
        + seconds * 1_000
        + milli
    )


def _is_time_line(text: str) -> bool:
    if "-->" not in text:
        return False
    left, right = text.split("-->", 1)
    return bool(_TIME_RE.search(left.strip()) and _TIME_RE.search(right.strip()))

@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str

def _read_srt_text(path: str | Path) -> str:
    """Decode common SRT encodings without silently changing characters."""
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Encoding SRT tidak didukung. Simpan sebagai UTF-8, UTF-16, "
                "atau Windows-1252 tanpa mengubah isi subtitle."
            ) from exc


class SubtitleTrack:
    def __init__(self, cues: list[SubtitleCue] | None = None):
        self.cues = sorted(cues or [], key=lambda c: c.start_ms)

    @classmethod
    def load(cls, path: str | Path) -> "SubtitleTrack":
        raw = _read_srt_text(path)
        lines = raw.splitlines()
        cues: list[SubtitleCue] = []
        fallback_index = 1
        i = 0

        while i < len(lines):
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break

            idx = fallback_index
            if _is_time_line(lines[i]):
                time_line = lines[i]
                i += 1
            elif i + 1 < len(lines) and _is_time_line(lines[i + 1]):
                try:
                    idx = int(lines[i].strip())
                except ValueError:
                    idx = fallback_index
                time_line = lines[i + 1]
                i += 2
            else:
                # Abaikan baris sampah/header tanpa membuang cue berikutnya.
                i += 1
                continue

            text_lines: list[str] = []
            while i < len(lines):
                current = lines[i]

                # Blank line normal adalah pemisah cue SRT.
                if not current.strip():
                    i += 1
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    break

                # Toleransi SRT tanpa blank line: timestamp berikutnya atau
                # "nomor cue + timestamp" menandai cue baru.
                if _is_time_line(current):
                    break
                if (
                    i + 1 < len(lines)
                    and current.strip().isdigit()
                    and _is_time_line(lines[i + 1])
                ):
                    break

                text_lines.append(current.rstrip())
                i += 1

            left, right = time_line.split("-->", 1)
            start_ms, end_ms = _time_ms(left), _time_ms(right)
            if end_ms <= start_ms:
                raise ValueError(
                    "Durasi cue SRT tidak valid: "
                    + time_line.strip()
                )

            text = "\n".join(text_lines).strip("\n")
            if not text:
                continue

            cues.append(SubtitleCue(idx, start_ms, end_ms, text))
            fallback_index += 1

        if not cues:
            raise ValueError("SRT tidak berisi cue subtitle yang dapat dibaca.")
        return cls(cues)

    def active_at(self, time_ms: int, padding_ms: int = 0) -> list[SubtitleCue]:
        return [
            c for c in self.cues
            if c.start_ms - padding_ms <= time_ms <= c.end_ms + padding_ms
        ]

    def is_safe_cut(self, time_ms: int, padding_ms: int = 250) -> bool:
        return not self.active_at(time_ms, padding_ms=padding_ms)

    def nearby(self, time_ms: int, radius_ms: int = 12_000) -> list[SubtitleCue]:
        lo, hi = time_ms - radius_ms, time_ms + radius_ms
        return [c for c in self.cues if c.end_ms >= lo and c.start_ms <= hi]

    def nearby_text(self, time_ms: int, radius_ms: int = 12_000, max_chars: int = 2400) -> str:
        rows = []
        for cue in self.nearby(time_ms, radius_ms):
            compact_text = " ".join(cue.text.splitlines())
            rows.append(
                f"{format_ms(cue.start_ms)} --> {format_ms(cue.end_ms)} | {compact_text}"
            )
        return "\n".join(rows)[:max_chars]

    def between(self, start_ms: int, end_ms: int) -> list[SubtitleCue]:
        """Subtitle yang bertumpang-tindih dengan rentang waktu tertentu."""
        lo, hi = int(start_ms), int(end_ms)
        return [
            cue for cue in self.cues
            if cue.end_ms >= lo and cue.start_ms < hi
        ]

    def between_text(
        self,
        start_ms: int,
        end_ms: int,
        max_chars: int = 3200,
    ) -> str:
        """SRT bertimestamp untuk dipasangkan dengan blok visual yang sama."""
        rows = []
        for cue in self.between(start_ms, end_ms):
            compact_text = " ".join(cue.text.splitlines())
            rows.append(
                f"{format_ms(cue.start_ms)} --> {format_ms(cue.end_ms)} | {compact_text}"
            )
        return "\n".join(rows)[:max_chars]


    def clipped_for_range(
        self,
        start_ms: int,
        end_ms: int,
    ) -> list[SubtitleCue]:
        """Klip cue ke satu part dan reset timestamp relatif ke awal part."""
        start_ms = int(start_ms)
        end_ms = int(end_ms)
        if end_ms <= start_ms:
            raise ValueError("Rentang subtitle part tidak valid.")

        result: list[SubtitleCue] = []
        for cue in self.cues:
            # Interval diperlakukan [start, end), sehingga cue yang tepat mulai
            # di boundary hanya masuk ke part berikutnya.
            if cue.end_ms <= start_ms or cue.start_ms >= end_ms:
                continue
            clipped_start = max(cue.start_ms, start_ms)
            clipped_end = min(cue.end_ms, end_ms)
            if clipped_end <= clipped_start:
                continue
            result.append(
                SubtitleCue(
                    index=len(result) + 1,
                    start_ms=clipped_start - start_ms,
                    end_ms=clipped_end - start_ms,
                    text=cue.text,
                )
            )
        return result

    def to_srt_range(self, start_ms: int, end_ms: int) -> str:
        """Render satu rentang sebagai SRT mandiri yang dimulai dari 00:00."""
        blocks: list[str] = []
        for cue in self.clipped_for_range(start_ms, end_ms):
            blocks.append(
                f"{cue.index}\n"
                f"{format_srt_ms(cue.start_ms)} --> {format_srt_ms(cue.end_ms)}\n"
                f"{cue.text}"
            )
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def dialogue_boundaries(self, start_ms: int, end_ms: int) -> list[int]:
        """Return local dialogue-edge hints for semantic cut discovery.

        A cut can be natural just before new dialogue starts or just after
        old dialogue ends, even when the picture changes later.
        """
        result: list[int] = []
        for cue in self.cues:
            if cue.end_ms < start_ms or cue.start_ms > end_ms:
                continue
            before_start = cue.start_ms - 420
            after_end = cue.end_ms + 420
            if start_ms <= before_start <= end_ms:
                result.append(before_start)
            if start_ms <= after_end <= end_ms:
                result.append(after_end)
        return sorted(set(result))

    def gap_boundaries(self, start_ms: int, end_ms: int, min_gap_ms: int = 450) -> list[int]:
        result: list[int] = []
        relevant = [
            c for c in self.cues
            if c.end_ms >= start_ms - 2_000 and c.start_ms <= end_ms + 2_000
        ]
        for a, b in zip(relevant, relevant[1:]):
            if b.start_ms - a.end_ms >= min_gap_ms:
                boundary = a.end_ms + (b.start_ms - a.end_ms) // 2
                if start_ms <= boundary <= end_ms:
                    result.append(boundary)
        return result

def format_ms(ms: int) -> str:
    ms = max(0, int(ms))
    total_s, milli = divmod(ms, 1000)
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


def format_srt_ms(ms: int) -> str:
    """Format timestamp standar SRT dengan pemisah milidetik koma."""
    return format_ms(ms).replace(".", ",")


def write_srt_parts(
    source_srt: str | Path,
    output_dir: str | Path,
    base_name: str,
    ranges: list[tuple[int, int]],
) -> list[Path]:
    """Buat satu file SRT untuk setiap rentang video.

    Teks subtitle dipertahankan. Cue yang melewati boundary dibelah secara
    lossless terhadap teksnya dan masing-masing sisi diklip ke durasi part.
    Timestamp setiap part selalu dimulai dari 00:00:00,000.
    """
    track = SubtitleTrack.load(source_srt)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for index, (start_ms, end_ms) in enumerate(ranges, 1):
        path = destination / f"{base_name}_Part-{index:02d}.srt"
        payload = track.to_srt_range(int(start_ms), int(end_ms))
        path.write_text(payload, encoding="utf-8", newline="\n")
        files.append(path)
    return files
