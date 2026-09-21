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


_CUT_WORDS = ("potong", "cut", "pecah", "split", "titik potong", "belah")


def looks_like_manual_cut(text: str) -> bool:
    low = str(text or "").lower()
    return any(word in low for word in _CUT_WORDS)


def _number(text: str | None) -> float:
    return float(str(text or "0").replace(",", "."))


def extract_manual_timestamps(text: str) -> list[ManualTimestamp]:
    """Extract explicit/natural user timestamps without AI reinterpretation.

    Supported examples:
      15.32 / 15:32                  -> 15m32s
      01:15:32                      -> 1h15m32s
      15 menit 32 detik             -> 15m32s
      1 jam 2 menit                 -> 1h02m
      1 jam lebih 2 menit           -> 1h02m
      1 jam lewat 2 menit 3 detik   -> 1h02m03s
      59.34.500                     -> 59m34.500s

    The parser intentionally keeps explicit user time values deterministic.
    Gemini can still understand broader natural-language commands when no local
    timestamp is found.
    """
    raw_text = str(text or "")
    found: list[tuple[int, ManualTimestamp]] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= x or a >= y) for x, y in occupied)

    def add_match(start: int, end: int, raw: str, ms: int) -> None:
        if ms < 0 or overlaps(start, end):
            return
        found.append((start, ManualTimestamp(raw, int(ms))))
        occupied.append((start, end))

    # H:MM:SS(.mmm)
    for m in re.finditer(
        r"(?<!\d)(\d{1,3}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?(?!\d)",
        raw_text,
    ):
        h, minute, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if minute > 59 or sec > 59:
            continue
        milli = int((m.group(4) or "0").ljust(3, "0"))
        ms = ((h * 3600 + minute * 60 + sec) * 1000) + milli
        add_match(m.start(), m.end(), m.group(0), ms)

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
        add_match(m.start(), m.end(), m.group(0), ms)

    # Natural Indonesian with hours. Connectors such as "lebih", "lewat",
    # "plus", "+", or "dan" are accepted so phrases like
    # "1 jam lebih 2 menit" are read as 01:02:00.
    for m in re.finditer(
        r"(?<!\d)(\d{1,3}(?:[.,]\d+)?)\s*jam"
        r"(?:\s*(?:(?:lebih|lewat|plus|dan)\s*|\+\s*)?"
        r"(\d{1,3}(?:[.,]\d+)?)\s*menit)?"
        r"(?:\s*(?:(?:lebih|lewat|plus|dan)\s*|\+\s*)?"
        r"(\d{1,3}(?:[.,]\d+)?)\s*detik)?",
        raw_text,
        flags=re.I,
    ):
        if overlaps(m.start(), m.end()):
            continue
        hours = _number(m.group(1))
        minutes = _number(m.group(2))
        seconds = _number(m.group(3))
        ms = round((hours * 3600 + minutes * 60 + seconds) * 1000)
        add_match(m.start(), m.end(), m.group(0), ms)

    # Natural Indonesian "15 menit 32 detik".
    for m in re.finditer(
        r"(?<!\d)(\d{1,4}(?:[.,]\d+)?)\s*menit"
        r"(?:\s*(\d{1,2}(?:[.,]\d+)?)\s*detik)?",
        raw_text,
        flags=re.I,
    ):
        if overlaps(m.start(), m.end()):
            continue
        minute = _number(m.group(1))
        sec = _number(m.group(2))
        if sec >= 60:
            continue
        ms = round((minute * 60 + sec) * 1000)
        add_match(m.start(), m.end(), m.group(0), ms)

    # Natural Indonesian "90 detik" is useful for short clips and seek/cut
    # commands. It is parsed after larger units so it never duplicates them.
    for m in re.finditer(
        r"(?<!\d)(\d{1,6}(?:[.,]\d+)?)\s*detik\b",
        raw_text,
        flags=re.I,
    ):
        if overlaps(m.start(), m.end()):
            continue
        ms = round(_number(m.group(1)) * 1000)
        add_match(m.start(), m.end(), m.group(0), ms)

    found.sort(key=lambda item: item[0])
    result: list[ManualTimestamp] = []
    seen: set[int] = set()
    for _pos, item in found:
        if item.time_ms not in seen:
            result.append(item)
            seen.add(item.time_ms)
    return result
