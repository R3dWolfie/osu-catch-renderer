"""Decode an osu!catch .osr into per-frame catcher positions.

Unlike mania (which only needs the key bitmask), catch needs the catcher's
absolute x (0..512) every frame plus whether the dash key is held. osrparse
already parses the raw frames; we just read the catch-relevant fields.
"""
from __future__ import annotations

import lzma
import struct
from pathlib import Path

from osrparse import Replay

from osu_catch_renderer.beatmap.models import CatchFrame, ReplayMeta

# osrparse seeds the last frame with this sentinel time_delta (RNG seed).
_SEED_DELTA = -12345


class ReplayParseError(RuntimeError):
    pass


def _dashing(ev) -> bool:
    """Whether the dash key is held this frame, tolerant to osrparse versions.

    Newer osrparse exposes ReplayEventCatch.dashing (bool). Older builds carry
    a `keys` int/flag — any nonzero key means a button (dash) is held.

    NOTE: osrparse 7.x derives `dashing` with an EXACT `keys == 1` compare,
    but stable writes the buttons field as a ReplayButtonState BITMASK and
    real catch replays carry extra bits (observed: 17 = Left1|Smoke on a
    full dash-hold play, which osrparse reads as `dashing=False` for every
    frame — the "dash never pressed" overlay bug). parse_replay therefore
    prefers the RAW buttons stream (`_raw_button_states`, dash = Left1 bit,
    exactly stable's `buttonState & Left1` test); this helper is the
    fallback when the raw stream can't be aligned.
    """
    d = getattr(ev, "dashing", None)
    if isinstance(d, bool):
        return d
    keys = getattr(ev, "keys", 0)
    try:
        return int(keys) != 0
    except (TypeError, ValueError):
        return bool(keys)


def _raw_button_states(path: Path) -> list[int] | None:
    """The raw per-frame buttons ints, aligned 1:1 with osrparse's
    `replay_data` (same leading-sentinel skip rule; the RNG-seed tail frame is
    kept, as osrparse keeps it). None on any decode problem — callers fall
    back to osrparse's own (broken-for-bitmasks) dashing field."""
    try:
        data = Path(path).read_bytes()
        off = 0

        def _skip_string() -> None:
            nonlocal off
            tag = data[off]
            off += 1
            if tag == 0x00:
                return
            if tag != 0x0b:
                raise ValueError(f"bad string tag {tag}")
            length = shift = 0
            while True:
                byte = data[off]
                off += 1
                length |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            off += length

        off += 1 + 4                   # mode, game version
        _skip_string(); _skip_string(); _skip_string()
        off += 2 * 6 + 4 + 2 + 1 + 4   # counts, score, combo, perfect, mods
        _skip_string()                 # life bar
        off += 8                       # timestamp
        rlen = struct.unpack_from("<i", data, off)[0]
        off += 4
        raw = lzma.decompress(data[off:off + rlen],
                              format=lzma.FORMAT_AUTO).decode("ascii", "replace")
        out: list[int] = []
        groups = raw.rstrip(",").split(",")
        for i, group in enumerate(groups):
            if not group:
                continue
            fields = group.split("|")
            if len(fields) < 4:
                return None
            if int(fields[0]) == _SEED_DELTA and i == len(groups) - 1:
                continue               # RNG-seed tail — osrparse drops it too
            if (i < 2 and float(fields[1]) == 256.0
                    and float(fields[2]) == -500.0):
                continue               # the sentinels osrparse strips
            out.append(int(float(fields[3])))
        return out
    except Exception:  # noqa: BLE001 — fall back to osrparse's field
        return None


def _recover_leadin_offset(path: Path) -> int:
    """Recover the replay-clock lead-in that osrparse silently discards.

    osu!stable replays begin with up to two placeholder frames at the
    off-screen sentinel position (256, -500). osu!'s own LegacyScoreDecoder
    ACCUMULATES each frame's time delta into the running clock *before* it
    drops those placeholders (``lastTime += diff`` precedes the
    ``i < 2 && (256,-500)`` skip), so the second placeholder's delta carries
    the audio lead-in / intro-skip offset. osrparse instead ``continue``s past
    these frames (osrparse/replay.py: ``if i < 2 and float(x) == 256 and
    float(y) == -500: continue``), throwing their deltas away — the lead-in
    never reaches ``Replay.replay_data``. Accumulating that stream from 0 then
    starts the clock too early by the whole lead-in, so every object is
    sampled before the player reached it (mass over-miss on stable replays
    whose intro-skip is not cancelled by a <-5000 ms first frame).

    Reproduce osu!'s accumulation: read the raw replay-data blob and sum the
    deltas of exactly the leading placeholder frames osrparse strips. Returns 0
    when there are none (lazer replays carry no placeholder frames; a clean
    stable play carries a ~0 ms lead-in), so already-aligned replays are left
    byte-identical. Fail-soft: any decode problem returns 0 (the pre-fix
    behaviour) rather than raising.
    """
    try:
        data = Path(path).read_bytes()
        off = 0

        def _skip_string() -> None:
            nonlocal off
            tag = data[off]
            off += 1
            if tag == 0x00:
                return
            if tag != 0x0b:
                raise ValueError(f"bad string tag {tag}")
            length = shift = 0
            while True:
                byte = data[off]
                off += 1
                length |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            off += length

        off += 1                       # mode (byte)
        off += 4                       # game version (int32)
        _skip_string()                 # beatmap md5
        _skip_string()                 # player name
        _skip_string()                 # replay md5
        off += 2 * 6                   # 300/100/50/geki/katu/miss (6 shorts)
        off += 4                       # score (int32)
        off += 2                       # max combo (short)
        off += 1                       # perfect (byte)
        off += 4                       # mods (int32)
        _skip_string()                 # life-bar graph
        off += 8                       # timestamp (int64)
        rlen = struct.unpack_from("<i", data, off)[0]
        off += 4                       # replay-data length (int32)
        raw = lzma.decompress(data[off:off + rlen],
                              format=lzma.FORMAT_AUTO).decode("ascii", "replace")

        lead = 0
        for i, group in enumerate(raw.rstrip(",").split(",")):
            if not group:
                continue
            fields = group.split("|")
            delta = int(fields[0])
            if delta == _SEED_DELTA:   # RNG seed (never a leading frame) — stop
                break
            # osrparse strips only the first two frames, and only when they sit
            # at the (256, -500) sentinel; mirror that set exactly.
            if i < 2 and float(fields[1]) == 256.0 and float(fields[2]) == -500.0:
                lead += delta
                continue
            break                      # first real frame: nothing more to strip
        return lead
    except Exception:  # noqa: BLE001 - never let a header quirk break parsing
        return 0


def parse_replay(path: Path) -> tuple[list[CatchFrame], ReplayMeta]:
    if not path.exists():
        raise ReplayParseError(f"replay not found: {path}")
    try:
        r = Replay.from_path(path)
    except Exception as e:  # noqa: BLE001 - osrparse raises bare exceptions
        raise ReplayParseError(f"osrparse failed: {e}") from e

    frames: list[CatchFrame] = []
    # Seed the clock with the lead-in osrparse discarded (see
    # _recover_leadin_offset) so the catcher timeline matches osu! exactly; 0
    # for lazer / already-aligned stable replays. Real frame times are in MAP
    # time (DT/HT do not compress them), so they line up 1:1 with beatmap
    # object times.
    t = _recover_leadin_offset(path)
    # Raw buttons stream (dash = Left1 BIT, stable semantics) — osrparse 7.x
    # compares the whole bitmask == 1 and reports dashing=False whenever any
    # other bit rides along (observed 17 = Left1|Smoke). Aligned by index;
    # any mismatch falls back to osrparse's field for the whole replay.
    zs = _raw_button_states(path)
    evs = r.replay_data or []
    use_z = zs is not None and len(zs) == len(evs)
    for idx, ev in enumerate(evs):
        delta = int(getattr(ev, "time_delta", 0))
        if delta == _SEED_DELTA:
            continue  # RNG seed frame, not a real position
        t += delta
        x = getattr(ev, "x", None)
        if x is None:
            continue  # non-positional frame
        dash = ((zs[idx] & 1) != 0) if use_z else _dashing(ev)
        frames.append(CatchFrame(time_ms=max(t, 0), x=float(x), dashing=dash))
    frames.sort(key=lambda f: f.time_ms)

    # osu!catch accuracy: every caught object (fruit / large droplet /
    # tiny droplet) counts equally; the denominator includes missed tiny
    # droplets (count_katu) and missed fruit/large droplets (count_miss).
    # NOT the std 300/100-half/50-quarter weighting.
    total = (r.count_300 + r.count_100 + r.count_50
             + r.count_katu + r.count_miss)
    if total > 0:
        acc = (r.count_300 + r.count_100 + r.count_50) / total
    else:
        acc = 1.0

    # Detect a fail from the life-bar graph. In osu!catch a play ends the
    # instant HP hits 0 — but the renderer otherwise plays the *full* map,
    # freezing the catcher after the replay's last frame so every unreached
    # object reads as a miss (the "fruits dropping through a stationary
    # catcher" bug). Capturing the death time lets render_core stop at the
    # fail. NoFail replays never die, so they're exempt.
    death_ms: int | None = None
    death_from_lifebar = False
    NF = 0x1
    if not (int(r.mods) & NF):
        life_bar = getattr(r, "life_bar_graph", None) or []
        for e in life_bar:
            try:
                if float(e.life) <= 0.001:
                    death_ms = int(e.time)
                    death_from_lifebar = True
                    break
            except (TypeError, ValueError, AttributeError):
                continue
        # Frame-timing fallback ONLY when there is genuinely NO life-bar graph
        # (lazer). A life-bar graph that EXISTS and never hit 0 PROVES the player
        # survived — stable input legitimately stops before the last object on
        # trailing bananas / no-movement sections (convert maps especially), so
        # treating that as a death fabricates a FALSE FAIL on a clear.
        # (Bug 2026-08-16: Stark's S on "down" — full life the whole play, frames
        # stop ~2s before the ending doubles, so the fallback invented a death.)
        if death_ms is None and not life_bar:
            _t = 0
            for f in (getattr(r, "replay_data", None) or []):
                _dt = getattr(f, "time_delta", 0)
                if _dt is None or _dt < 0:   # -12345 end marker / seed frame
                    continue
                _t += _dt
            if _t > 0:
                death_ms = _t

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
        death_ms=death_ms,
        death_from_lifebar=death_from_lifebar,
        timestamp=getattr(r, "timestamp", None),
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
    # osu!catch rank is accuracy-ONLY, inclusive cutoffs (CatchScoreProcessor
    # RankFromScore): SS=100%, S>=98, A>=94, B>=90, C>=85, else D. There is NO
    # zero-miss requirement for S in catch (unlike std) — the old `miss == 0`
    # gate + strict `>` here mis-graded 1-miss ≥98% plays and exact boundaries.
    if acc >= 1.0:
        return "SS"
    if acc >= 0.98:
        return "S"
    if acc >= 0.94:
        return "A"
    if acc >= 0.90:
        return "B"
    if acc >= 0.85:
        return "C"
    return "D"
