"""The --pitch argument: what the shifts are, and in what order they are tried.

Pure python — nothing here touches an audio sample, so the window can import it
and show the very same order the search will use."""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Pitch shifting
# --------------------------------------------------------------------------- #
# Only the classical case is handled: pitch and tempo change TOGETHER, the way
# it happens when somebody drags a sample in a tracker. That is a resampling by
# 2^(n/12), nothing more — no time-stretching, so the pitch and the tempo of a
# query cannot be changed independently of each other.
# PITCH_PRECISION moved to ssc.audio, where the resampling itself happens

# How wide a range is built when the ORDER pattern is worked out: the pattern
# has to be the same whatever the ends of the range are, so the ends are
# replaced by a width wide enough that no group can run out of values.
_PATTERN_WIDTH = 16.0

# A pitch range looks like -2:3 or -2:3:0.5. argparse reads any token that
# starts with "-" as an option unless it looks like a plain negative number, so
# the range form has to arrive in the "--pitch=..." shape, which bypasses that
# check. Plain shifts (-1, -2.5) are fine as they are.
_PITCH_RANGE_TOKEN = re.compile(r"^-[\d.]+:[^=]+$")


def normalise_pitch_args(argv: Sequence[str]) -> List[str]:
    """
    Rewrites a bare '-2:3:0.5' into '--pitch=-2:3:0.5', which is the form
    argparse accepts. A bare -p/--pitch that has not been given any value yet is
    dropped in the process.

    A range cannot be mixed with loose shifts in one command (argparse keeps
    only the last occurrence): write --pitch=-2:3 instead of --pitch -1 -2:3.
    """
    out: List[str] = []
    pending = False
    for tok in argv:
        if _PITCH_RANGE_TOKEN.match(tok):
            if pending and out and out[-1] in ("-p", "--pitch"):
                out.pop()
            out.append(f"--pitch={tok}")
            pending = False
            continue
        out.append(tok)
        pending = tok in ("-p", "--pitch")
    return out


def semitone_ratio(semitones: float) -> float:
    """Frequency multiplier for a shift of `semitones` semitones."""
    return 2.0 ** (semitones / 12.0)


def fmt_semitones(semitones: float) -> str:
    """
    +3, -2.5, +0.125, 0 — the sign is always shown, trailing zeros are not.
    The precision follows the value: whole and half steps stay short, and an
    eighth step is never rounded away into a different shift.
    """
    digits = 4
    for d in (0, 1, 2, 3, 4):
        if abs(round(semitones, d) - semitones) < 1e-9:
            digits = d
            break
    if digits == 0:
        s = f"{semitones:+.0f}"
    else:
        s = f"{semitones:+.{digits}f}".rstrip("0").rstrip(".")
    return "0" if s in ("+0", "-0") else s


def _pitch_range(start: float, end: float, step: float) -> List[float]:
    """
    start, start+step, ... up to and including end — the intent of python's
    range(start, end+step, step), but with float steps that python's range()
    cannot take, and without the drift that accumulating i*step would give.
    """
    if step == 0:
        raise ValueError("the step of a pitch range must not be 0")
    count = int(math.floor((end - start) / step + 1e-9)) + 1
    return [round(start + i * step, 6) for i in range(max(0, count))]


def _is_power_of_two(step: float) -> bool:
    """True for 1, 0.5, 0.25, 0.125 ... — the steps the finer order accepts."""
    if step <= 0:
        return False
    for d in range(41):
        y = step * (2.0 ** d)
        if abs(y - 1.0) < 1e-9:
            return True
    return False


def _fineness(v: float) -> int:
    """
    Which power-of-two level a shift sits on: 0 for whole semitones (1.0, 2.0,
    3.0), 1 for halves (0.5, 1.5), 2 for quarters (0.25, 0.75, 1.25), and so
    on. It is the smallest number of halvings that turns the value into a whole
    number, which is what lets the finer order be built without knowing the
    step.
    """
    x = abs(float(v))
    if x == 0.0:
        return 0
    for d in range(41):
        y = x * (2.0 ** d)
        if abs(y - round(y)) < 1e-9:
            return d
    return 41          # not a power-of-two fraction: keep it out of the way


def order_pitches(values: Sequence[float], order: str = "finer") -> List[float]:
    """
    "symm"  — by distance from zero, the upward shift first:
              0, +1, -1, +2, -2, +3, -3 ...
    "finer" — by how fine the shift is first, and only then by distance:
              0, +1, -1, +2, -2, +3, -3, then +0.5, -0.5, +1.5, -1.5 ...,
              then +0.25, -0.25, +0.75, -0.75 ...
            Whole semitones are tried before fractions, so a sample that was
            not transposed at all is answered first.
    """
    if order == "symm":
        return sorted(values, key=lambda v: (abs(v), -v))
    return sorted(values, key=lambda v: (_fineness(v), abs(v), -v))


def group_end(pitches: Sequence[float], at: int,
              order: str = "finer") -> int:
    """
    The index one past the last shift of the group that pitches[at] sits in -
    the very groups describe_pitch_order prints: one per level of fineness for
    "finer", the whole list for "symm".

    This is what turns "every shift that is left" into "every shift that is
    left in this group". The order the finer order sorts by puts the levels of
    fineness next to each other and in one piece each, so a run of equal
    fineness is a group, and with a step of 1 every shift is on level zero: one
    group covering all of them, which leaves the caller with what it had.

    Ordered, but not assumed to be: a list that was never through
    order_pitches would have the levels scattered, and the answer would then be
    the first run rather than the group. The search only ever hands over an
    ordered list, which is why this walks instead of counting.
    """
    if at < 0 or at >= len(pitches) or order != "finer":
        return len(pitches)
    level = _fineness(pitches[at])
    end = at + 1
    while end < len(pitches) and _fineness(pitches[end]) == level:
        end += 1
    return end


def describe_pitch_set(low: float, high: float,
                       step: float = 1.0) -> str:
    """
    The shifts of the range as a plain list, from the lowest to the highest:

        -4, -3.75, ..., +1.75, +2

    A short range is written out; a wide one only shows its first two and its
    last two values, because the whole list would be a line of nothing but
    numbers. Which of them is tried first is a different question, and
    describe_pitch_order answers it.
    """
    if high < low:
        low, high = high, low
    values = sorted(_pitch_range(low, high, step))
    if len(values) <= 6:
        return ", ".join(fmt_semitones(v) for v in values)
    return (", ".join(fmt_semitones(v) for v in values[:2]) + ", ..., "
            + ", ".join(fmt_semitones(v) for v in values[-2:]))


def count_pitches(low: float, high: float, step: float = 1.0) -> int:
    """
    How many shifts the range holds, and so how many times a query is compared
    with the whole collection — the number that decides how long the run takes,
    which is why the window shows it next to the shifts themselves.

    Counted the way parse_pitch_args counts them, duplicates dropped the same way,
    so the number on the screen is the number the search really makes.
    """
    if high < low:
        low, high = high, low
    return len({round(v, 6) for v in _pitch_range(low, high, step)})


def describe_pitch_order(step: float = 1.0, order: str = "finer") -> str:
    """
    The pattern the shifts are tried in, which the step decides on its own —
    the ends of the range have nothing to do with it:

        finer, step 0.125   0, ±1, ±2, ...; ±0.5, ±1.5, ...
                               ±0.25, ±0.75, ...; ±0.125, ±0.375, ...
        symm,  step 0.125   0, ±0.125, ±0.25, ±0.375, ...

    "finer" puts the whole semitones first and the fractions afterwards, one
    group per level of fineness, separated by "; ". "symm" is a single
    progression that climbs from zero in both directions.

    A value whose opposite is in the same group is written once, as "±v" — that
    is what keeps the line short. The pattern never ends, so every group ends
    with "...".
    """
    groups = _levels(-_PATTERN_WIDTH, _PATTERN_WIDTH, step, order)
    out = []
    for i, level in enumerate(groups):
        # the first group starts with zero, so it has room for two pairs of its
        # own; the ones after it are pairs from the first
        items, _ = _level_text(level, limit=4 if order == "symm"
                               else 3 if i == 0 else 2)
        out.append(", ".join(items) + ", ...")
    return "; ".join(out)


def _levels(low: float, high: float, step: float,
            order: str) -> List[List[float]]:
    """
    The shifts of the range, grouped the way they are tried: one group per level
    of fineness for "finer", a single progression for "symm".
    """
    values = order_pitches(_pitch_range(low, high, step), order)
    if order == "symm":
        return [values]
    groups: List[List[float]] = []
    for v in values:
        if not groups or _fineness(groups[-1][0]) != _fineness(v):
            groups.append([v])
        else:
            groups[-1].append(v)
    return groups


def _level_text(level: List[float], limit: int) -> Tuple[List[str], bool]:
    """
    The first `limit` values of one group, and whether there were more.

    A value whose opposite is in the same group is written once, as "±v", and
    its twin is left out — that is what halves the length of the line. Zero is
    never merged: it is one shift, not two.
    """
    items: List[str] = []
    done = set()
    for v in level:
        key = round(v, 9)
        if key in done:
            continue
        mirrored = abs(v) > 1e-9 and any(abs(u + v) < 1e-9 for u in level)
        if mirrored:
            done.add(round(-v, 9))
            items.append("±" + fmt_semitones(abs(v)).lstrip("+"))
        else:
            items.append(fmt_semitones(v))
    return items[:limit], len(items) > limit


def parse_pitch_args(values: Optional[Sequence[str]],
                     order: str = "finer") -> List[float]:
    """
    Turns --pitch into the list of shifts to sweep, in the order they are
    searched:

      --pitch 2                 one shift
      --pitch -1 0 1 2          several shifts
      --pitch=-2:3              from -2 to 3 in whole semitones
      --pitch=-2:3:0.5          ...in half semitones

    With the "finer" order a range must use a step that is a power of two
    (1, 0.5, 0.25, 0.125 ...), because the order itself is built out of those
    levels. Without --pitch the list is just [0.0], i.e. what the program did
    before.
    """
    if not values:
        return [0.0]
    out: List[float] = []
    for item in values:
        parts = item.split(":")
        try:
            nums = [float(p) for p in parts if p != ""]
        except ValueError:
            raise ValueError(
                f"Cannot read the pitch shift {item!r} — use semitones, "
                f"like -2, -1 0 1 2 or -2:3:0.5")
        if not nums or len(nums) > 3:
            raise ValueError(
                f"Cannot read the pitch shift {item!r} — expected n, "
                f"n:n or n:n:n")
        if len(nums) == 1:
            out.append(nums[0])
        elif len(nums) == 2:
            out.extend(_pitch_range(nums[0], nums[1], 1.0))
        else:
            if order != "symm" and not _is_power_of_two(nums[2]):
                raise ValueError(
                    f"The step {nums[2]:g} of {item!r} is not a power of two. "
                    f"With --pitchorder finer use 1, 0.5, 0.25, 0.125 ... "
                    f"(or --pitchorder symm for any step)")
            out.extend(_pitch_range(nums[0], nums[1], nums[2]))
    seen = set()
    uniq: List[float] = []
    for v in out:
        key = round(v, 6)
        if key not in seen:
            seen.add(key)
            uniq.append(float(key))
    return order_pitches(uniq, order)

