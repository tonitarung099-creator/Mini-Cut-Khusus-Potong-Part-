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
_NON_ADD_CUT_HINTS = (
    "hapus", "remove", "delete", "undo", "urungkan", "batalkan",
    "cek cut", "lihat cut", "berapa cut", "apakah ada cut",
)


# Time-unit aliases accepted by the deterministic local parser. These are
# intentionally tolerant because chat commands are often typed compactly or
# with small typos (for example "1jam12dtik"). The negative lookahead keeps
# one-letter aliases from consuming the prefix of an unknown word.
_HOUR_UNIT = r"(?:jam|hours?|hrs?|hr|h|j)(?![A-Za-z])"
_MINUTE_UNIT = r"(?:menit|minutes?|mins?|min|mnt|m)(?![A-Za-z])"
_SECOND_UNIT = r"(?:detik|dtik|dtk|seconds?|secs?|sec|s|d)(?![A-Za-z])"
_JOINER = r"(?:(?:lebih|lewat|plus|dan)\s*|\+\s*)?"


def looks_like_manual_cut(text: str) -> bool:
    low = " ".join(str(text or "").lower().split())
    if any(hint in low for hint in _NON_ADD_CUT_HINTS):
        return False
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
      1jam 12dtik / 1h12s           -> 1h00m12s
      62mnt 5dtk                     -> 62m05s

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

    # Natural Indonesian/English with hours. Compact forms and common typing
    # variants are accepted, e.g. "1jam12dtik", "1h12s", or "1j 2m 3d".
    # If an hour-only match is immediately followed by another number that we
    # do not understand, skip the partial match and let Gemini reason about the
    # whole phrase instead of silently cutting at the wrong hour boundary.
    for m in re.finditer(
        rf"(?<!\d)(\d{{1,3}}(?:[.,]\d+)?)\s*{_HOUR_UNIT}"
        rf"(?:\s*{_JOINER}(\d{{1,3}}(?:[.,]\d+)?)\s*{_MINUTE_UNIT})?"
        rf"(?:\s*{_JOINER}(\d{{1,3}}(?:[.,]\d+)?)\s*{_SECOND_UNIT})?",
        raw_text,
        flags=re.I,
    ):
        if overlaps(m.start(), m.end()):
            continue
        tail = raw_text[m.end():]
        if m.group(2) is None and m.group(3) is None and re.match(r"\s*\d", tail):
            continue
        hours = _number(m.group(1))
        minutes = _number(m.group(2))
        seconds = _number(m.group(3))
        ms = round((hours * 3600 + minutes * 60 + seconds) * 1000)
        add_match(m.start(), m.end(), m.group(0), ms)

    # Minutes with an optional seconds component. Accept compact aliases too,
    # while refusing a suspicious partial match followed by an unknown number.
    for m in re.finditer(
        rf"(?<!\d)(\d{{1,4}}(?:[.,]\d+)?)\s*{_MINUTE_UNIT}"
        rf"(?:\s*{_JOINER}(\d{{1,3}}(?:[.,]\d+)?)\s*{_SECOND_UNIT})?",
        raw_text,
        flags=re.I,
    ):
        if overlaps(m.start(), m.end()):
            continue
        tail = raw_text[m.end():]
        if m.group(2) is None and re.match(r"\s*\d", tail):
            continue
        minute = _number(m.group(1))
        sec = _number(m.group(2))
        ms = round((minute * 60 + sec) * 1000)
        add_match(m.start(), m.end(), m.group(0), ms)

    # Seconds, including common abbreviations/typos such as dtk/dtik/sec/s.
    # Parsed after larger units so the same timestamp is never duplicated.
    for m in re.finditer(
        rf"(?<!\d)(\d{{1,6}}(?:[.,]\d+)?)\s*{_SECOND_UNIT}",
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
