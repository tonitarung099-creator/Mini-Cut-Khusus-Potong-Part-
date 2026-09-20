from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TIME_RE = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})")

def _time_ms(text: str) -> int:
    m = _TIME_RE.search(text.strip())
    if not m:
        raise ValueError(f"Timestamp SRT tidak valid: {text}")
    return (
        int(m.group("h")) * 3_600_000
        + int(m.group("m")) * 60_000
        + int(m.group("s")) * 1_000
        + int(m.group("ms"))
    )

@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str

class SubtitleTrack:
    def __init__(self, cues: list[SubtitleCue] | None = None):
        self.cues = sorted(cues or [], key=lambda c: c.start_ms)

    @classmethod
    def load(cls, path: str | Path) -> "SubtitleTrack":
        raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
        blocks = re.split(r"\r?\n\s*\r?\n", raw.strip())
        cues: list[SubtitleCue] = []
        fallback_index = 1
        for block in blocks:
            lines = [line.rstrip() for line in block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            if "-->" in lines[0]:
                idx = fallback_index
                time_line = lines[0]
                text_lines = lines[1:]
            else:
                try:
                    idx = int(lines[0].strip())
                except ValueError:
                    idx = fallback_index
                if len(lines) < 3 or "-->" not in lines[1]:
                    continue
                time_line = lines[1]
                text_lines = lines[2:]
            left, right = time_line.split("-->", 1)
            start_ms, end_ms = _time_ms(left), _time_ms(right)
            if end_ms <= start_ms:
                continue
            text = " ".join(x.strip() for x in text_lines).strip()
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
            rows.append(
                f"{format_ms(cue.start_ms)} --> {format_ms(cue.end_ms)} | {cue.text}"
            )
        return "\n".join(rows)[:max_chars]


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
