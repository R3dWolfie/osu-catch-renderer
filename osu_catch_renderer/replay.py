"""Decode an osu!catch .osr into per-frame catcher positions.

Unlike mania (which only needs the key bitmask), catch needs the catcher's
absolute x (0..512) every frame plus whether the dash key is held. osrparse
already parses the raw frames; we just read the catch-relevant fields.
"""
from __future__ import annotations

from pathlib import Path

from osrparse import Replay

from .models import CatchFrame, ReplayMeta

# osrparse seeds the last frame with this sentinel time_delta (RNG seed).
_SEED_DELTA = -12345


class ReplayParseError(RuntimeError):
    pass


def _dashing(ev) -> bool:
    """Whether the dash key is held this frame, tolerant to osrparse versions.

    Newer osrparse exposes ReplayEventCatch.dashing (bool). Older builds carry
    a `keys` int/flag — any nonzero key means a button (dash) is held.
    """
    d = getattr(ev, "dashing", None)
    if isinstance(d, bool):
        return d
    keys = getattr(ev, "keys", 0)
    try:
        return int(keys) != 0
    except (TypeError, ValueError):
        return bool(keys)


def parse_replay(path: Path) -> tuple[list[CatchFrame], ReplayMeta]:
    if not path.exists():
        raise ReplayParseError(f"replay not found: {path}")
    try:
        r = Replay.from_path(path)
    except Exception as e:  # noqa: BLE001 - osrparse raises bare exceptions
        raise ReplayParseError(f"osrparse failed: {e}") from e

    frames: list[CatchFrame] = []
    t = 0
    first = True
    for ev in r.replay_data or []:
        delta = int(getattr(ev, "time_delta", 0))
        if delta == _SEED_DELTA:
            continue  # RNG seed frame, not a real position
        if first:
            first = False
            # osu! sometimes prefixes a garbage placeholder frame with a huge
            # negative delta (e.g. -11144ms) that would shift the whole
            # timeline. A normal AudioLeadIn is only ~1-2s, so anything beyond
            # that is the placeholder — ignore its jump. Real frame times are
            # in MAP time (DT/HT do not compress them), so they then line up
            # 1:1 with beatmap object times.
            if delta < -5000:
                delta = 0
        t += delta
        x = getattr(ev, "x", None)
        if x is None:
            continue  # non-positional frame
        frames.append(CatchFrame(time_ms=max(t, 0), x=float(x), dashing=_dashing(ev)))
    frames.sort(key=lambda f: f.time_ms)

    total = r.count_300 + r.count_100 + r.count_50 + r.count_miss
    if total > 0:
        acc = (r.count_300 + r.count_100 / 2.0 + r.count_50 / 4.0) / total
    else:
        acc = 1.0
    meta = ReplayMeta(
        mode=int(r.mode.value if hasattr(r.mode, "value") else r.mode),
        beatmap_md5=str(getattr(r, "beatmap_hash", "") or ""),
        player_name=r.username,
        mods=int(r.mods),
        score=int(r.score),
        max_combo=int(r.max_combo),
        count_300=int(r.count_300),
        count_100=int(r.count_100),
        count_50=int(r.count_50),
        count_katu=int(r.count_katu),
        count_miss=int(r.count_miss),
        accuracy=round(acc * 100, 2),
        grade=_grade(acc, r),
        game_version=int(getattr(r, "game_version", 0) or 0),
    )
    return frames, meta


def catcher_x_at(frames: list[CatchFrame], t_ms: int) -> tuple[float, bool]:
    """Linearly interpolate catcher x (and dash state) at time t_ms.

    Uses binary search for O(log n) lookup; callers sweep forward so this is
    cheap. Returns (x_osu, dashing).
    """
    if not frames:
        return 256.0, False
    lo, hi = 0, len(frames) - 1
    if t_ms <= frames[0].time_ms:
        return frames[0].x, frames[0].dashing
    if t_ms >= frames[-1].time_ms:
        return frames[-1].x, frames[-1].dashing
    while lo < hi:
        mid = (lo + hi) // 2
        if frames[mid].time_ms < t_ms:
            lo = mid + 1
        else:
            hi = mid
    a, b = frames[lo - 1], frames[lo]
    span = b.time_ms - a.time_ms
    if span <= 0:
        return b.x, b.dashing
    f = (t_ms - a.time_ms) / span
    return a.x + (b.x - a.x) * f, b.dashing


def _grade(acc: float, r) -> str:
    miss = int(r.count_miss)
    if acc >= 1.0:
        return "SS"
    if acc > 0.98 and miss == 0:
        return "S"
    if acc > 0.94:
        return "A"
    if acc > 0.90:
        return "B"
    if acc > 0.85:
        return "C"
    return "D"
