from __future__ import annotations

import re
from dataclasses import dataclass

from .core import clock_text


@dataclass(frozen=True)
class ManualTimestamp:
    raw: str
    time_ms: int

    @property
    def time(self) -> str:
        return clock_text(self.time_ms)


_CUT_WORDS = ("potong", "cut", "pecah", "split", "titik potong")


def looks_like_manual_cut(text: str) -> bool:
    low = str(text or "").lower()
    return any(word in low for word in _CUT_WORDS)


def extract_manual_timestamps(text: str) -> list[ManualTimestamp]:
    """Extract explicit user timestamps without AI reinterpretation.

    Supported:
      15.32 / 15:32          -> 15m32s
      01:15:32              -> 1h15m32s
      15 menit 32 detik     -> 15m32s
      59.34.500              -> 59m34.500s
    """
    raw_text = str(text or "")
    found: list[tuple[int, ManualTimestamp]] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= x or a >= y) for x, y in occupied)

    # H:MM:SS(.mmm)
    for m in re.finditer(
        r"(?<!\d)(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?(?!\d)",
        raw_text,
    ):
        h, minute, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if minute > 59 or sec > 59:
            continue
        milli = int((m.group(4) or "0").ljust(3, "0"))
        ms = ((h * 3600 + minute * 60 + sec) * 1000) + milli
        found.append((m.start(), ManualTimestamp(m.group(0), ms)))
        occupied.append((m.start(), m.end()))

    # M:SS, M.SS, optionally milliseconds as M.SS.mmm / M:SS.mmm.
    for m in re.finditer(
        r"(?<![\d:])(\d{1,3})[.:](\d{2})(?:[.:](\d{1,3}))?(?!\d)",
        raw_text,
    ):
        if overlaps(m.start(), m.end()):
            continue
        minute, sec = int(m.group(1)), int(m.group(2))
        if sec > 59:
            continue
        milli = int((m.group(3) or "0").ljust(3, "0"))
        ms = ((minute * 60 + sec) * 1000) + milli
        found.append((m.start(), ManualTimestamp(m.group(0), ms)))
        occupied.append((m.start(), m.end()))

    # Natural Indonesian "15 menit 32 detik".
    for m in re.finditer(
        r"(?<!\d)(\d{1,3})\s*menit(?:\s*(\d{1,2})(?:[.,](\d{1,3}))?\s*detik)?",
        raw_text,
        flags=re.I,
    ):
        if overlaps(m.start(), m.end()):
            continue
        minute = int(m.group(1))
        sec = int(m.group(2) or 0)
        if sec > 59:
            continue
        milli = int((m.group(3) or "0").ljust(3, "0"))
        ms = ((minute * 60 + sec) * 1000) + milli
        found.append((m.start(), ManualTimestamp(m.group(0), ms)))
        occupied.append((m.start(), m.end()))

    found.sort(key=lambda item: item[0])
    result: list[ManualTimestamp] = []
    seen: set[int] = set()
    for _pos, item in found:
        if item.time_ms not in seen:
            result.append(item)
            seen.add(item.time_ms)
    return result
