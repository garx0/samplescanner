"""How a number, a time or a file name is written for a human."""

from __future__ import annotations

import os
import re


def fmt_size(num: float) -> str:
    """A number of bytes as a human reads it: "14.9 MB", "2.4 GB"."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def fmt_duration(seconds: float) -> str:
    """Human readable duration: "20.6 min" or "45.3 s"."""
    if seconds < 90:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def fmt_elapsed(seconds: float) -> str:
    """How long something took, the way a person says it out loud:
    7s, 3m12s, 1h5m47s — no leading zeros, no colons to count."""
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m{sec}s"
    if minutes:
        return f"{minutes}m{sec}s"
    return f"{sec}s"


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(seconds, 60.0)
    return f"{int(m)}:{s:06.3f}"


def exe_name(program: str) -> str:
    """
    The name a program has on the machine it is running on: ffmpeg.exe on
    Windows, plain ffmpeg on macOS and Linux, where a binary program normally
    carries no extension at all.

    It matters for exactly one thing, and that thing is looking for ffmpeg and
    ffprobe BESIDE the program — the documented way to install them. One in PATH
    is found whatever it is called, but a build unpacked next to the program is
    only found on the platform that puts the extension on, and on the others the
    program then says it cannot find what is standing right next to it.
    """
    return program + ".exe" if os.name == "nt" else program


_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(name: str, max_len: int = 90) -> str:
    """File name without extension, made safe for any OS."""
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = _ILLEGAL.sub("_", stem).strip(" .")
    stem = re.sub(r"\s+", " ", stem)
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip(" .")
    return stem or "audio"


def fmt_stamp(seconds: float) -> str:
    """
    Timestamp for a file name: 3m16s129 — the minutes are not padded, the
    seconds and the milliseconds are, so that the numbers of a list of names
    line up under each other.
    """
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    return f"{total_s // 60}m{total_s % 60:02d}s{ms:03d}"

