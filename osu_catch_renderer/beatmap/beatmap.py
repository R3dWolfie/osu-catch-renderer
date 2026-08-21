"""Parse a .osu file and generate the osu!catch object stream.

Circle      -> one FRUIT at (x, time)
Slider      -> FRUIT at head, DROPLETs along the path, FRUIT at each repeat
               arrival and at the tail
Spinner     -> a BANANA shower spread across [time, end_time]

Slider timing uses the active (red) timing point's beat length and the active
(green) slider-velocity multiplier, exactly like the game, so droplet/repeat
times land where the player actually had to catch them.
"""
from __future__ import annotations

import math
from pathlib import Path

from osu_catch_renderer.beatmap.legacy_random import RNG_SEED, LegacyRandom
from osu_catch_renderer.beatmap.models import CatchBeatmap, CatchObject, HitSample, ObjType
from osu_catch_renderer.render.sliderpath import SliderPath

# HitObject type bitfield
_TYPE_CIRCLE = 1 << 0
_TYPE_SLIDER = 1 << 1
_TYPE_NEW_COMBO = 1 << 2
_TYPE_SPINNER = 1 << 3


class BeatmapParseError(RuntimeError):
    pass


def parse_beatmap(path: Path, *, mods: int = 0, lazer: bool = False,
                  position_offsets: bool = True) -> CatchBeatmap:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = _split_sections(text)

    diff = _kv(sections.get("Difficulty", ""))
    meta = _kv(sections.get("Metadata", ""))
    general = _kv(sections.get("General", ""))

    cs = _f(diff.get("CircleSize"), 5.0)
    ar = _f(diff.get("ApproachRate"), _f(diff.get("OverallDifficulty"), 9.0))
    od = _f(diff.get("OverallDifficulty"), 7.0)
    hp = _f(diff.get("HPDrainRate"), 5.0)
    slider_mult = _f(diff.get("SliderMultiplier"), 1.4)
    tick_rate = _f(diff.get("SliderTickRate"), 1.0)

    # --- mods: difficulty adjustment + playback rate ---
    ez = bool(mods & (1 << 1))
    hr = bool(mods & (1 << 4))   # catch HR: CS/AR up; NO x-mirror (catch HR does not flip)
    if ez:
        cs *= 0.5; ar *= 0.5; od *= 0.5; hp *= 0.5
    if hr:
        cs = min(10.0, cs * 1.3)
        ar = min(10.0, ar * 1.4)
        od = min(10.0, od * 1.4)
        hp = min(10.0, hp * 1.4)
    # NOTE: osu! stores catch replay frames in MAP time (their cumulative span
    # equals the map length, not map/rate), so DT/HT must NOT rescale object
    # times here — the replay and beatmap share the map-time axis. Rate only
    # affects real-world *playback* speed: we keep simulating on the map-time
    # axis, then compress the output video timeline + atempo the audio by `rate`
    # (done in render.py), which is what makes a DT play look/sound sped up.
    dt = bool(mods & (1 << 6)) or bool(mods & (1 << 9))  # DoubleTime or Nightcore
    ht = bool(mods & (1 << 8))                            # HalfTime
    rate = 1.5 if dt else (0.75 if ht else 1.0)

    timing = _parse_timing(sections.get("TimingPoints", ""))
    # osu!catch position offsets — CatchBeatmapProcessor.ApplyPositionOffsets
    # (both stable and lazer run this in beatmap conversion): banana x =
    # NextDouble()*512, tiny-droplet XOffset = clamp(Next(-20,20), field), and
    # under HardRock the per-CIRCLE applyHardRockOffset chain (the fix for the
    # "end-of-map combo mismatch": an HR stable replay plays OFFSET fruit
    # positions, so simulating the un-offset .osu x's put misses in the wrong
    # places — NoMyDarknesss/'clarity rmx' showed 4 phantom misses, max combo
    # 159 vs the header's 310). Bit-exact only when our nested generation
    # matches osu's counts (true on verified maps; a count drift desyncs the
    # stream and degrades ONLY these cosmetic/±20 offsets — the count
    # reconcile in scene.py still anchors judgements). `position_offsets=False`
    # (or R3D_CATCH_NO_POSOFFSETS=1) restores the pre-offset stream — the
    # certified-argon identity kill-switch.
    import os
    rng = None
    if position_offsets and not os.environ.get("R3D_CATCH_NO_POSOFFSETS"):
        rng = LegacyRandom(RNG_SEED)
    objects = _parse_hit_objects(
        sections.get("HitObjects", ""),
        timing=timing,
        slider_mult=slider_mult,
        tick_rate=tick_rate,
        hr=hr,
        lazer=lazer,
        rng=rng,
    )
    objects.sort(key=lambda o: (o.time_ms, 0 if o.kind is ObjType.FRUIT else 1))
    objects = _mark_hyperdash(objects, cs)

    return CatchBeatmap(
        objects=objects,
        cs=cs, ar=ar, od=od, hp=hp, rate=rate,
        audio_filename=general.get("AudioFilename"),
        background=_parse_background(sections.get("Events", "")),
        breaks=_parse_breaks(sections.get("Events", "")),
        title=meta.get("Title", ""),
        artist=meta.get("Artist", ""),
        version=meta.get("Version", ""),
        creator=meta.get("Creator", ""),
        combo_colors=_parse_combo_colors(sections.get("Colours", "")),
        timing=timing,
        sample_set_default=_SET_IDS.get(
            str(general.get("SampleSet", "Normal")).strip().lower(), 1),
    )


# [General] SampleSet name -> osu set id (1 normal / 2 soft / 3 drum)
_SET_IDS = {"normal": 1, "soft": 2, "drum": 3, "none": 1}


# --- timing -------------------------------------------------------------------

class _Timing:
    """Resolves beat length and SV multiplier at any time."""

    def __init__(self, points: list[tuple[float, float, bool]],
                 hs_points: list[tuple[float, int, int, int]] | None = None,
                 kiai_ranges: list[tuple[float, float]] | None = None):
        # (time, value, uninherited). value = beatLength for red lines,
        # negative-inverse SV for green lines.
        self.points = points
        # hitsound state per timing point (red AND green lines both carry it):
        # (time, sampleSet 0-3, sampleIndex, volume 0-100). Sorted by time.
        self.hs_points = hs_points or []
        # kiai regions [(start_ms, end_ms)] — legacy catcher-kiai sprite swap.
        self.kiai = kiai_ranges or []

    def sample_info(self, t: float) -> tuple[int, int, int]:
        """(sampleSet, sampleIndex, volume) active at map-time `t` — the last
        point with time <= t; before the first point, the first point's state
        (stable/mania convention). No points -> neutral (0, 0, 100)."""
        pts = self.hs_points
        if not pts:
            return 0, 0, 100
        cur = pts[0]
        for p in pts:
            if p[0] > t:
                break
            cur = p
        return cur[1], cur[2], cur[3]

    def beat_length(self, t: float) -> float:
        # osu! (stable + lazer ControlPointInfo.TimingPointAt) extends the FIRST
        # uninherited timing point backward to -inf: an object placed before the
        # first red line uses that first line's beatLength, NOT a 500ms default.
        # (Common: first object 1-2ms before the first timing point -> its slider
        # would otherwise get velocity ~half, doubling span duration and mangling
        # repeats/tail positions.) 500.0 is only correct when there are NO
        # uninherited points at all.
        bl = None
        for time, val, uninh in self.points:   # sorted by time
            if uninh and val > 0:
                if bl is None:
                    bl = val          # backward fallback = first uninherited line
                if time <= t:
                    bl = val          # last uninherited line at/before t
            if time > t:
                break
        return bl if bl is not None else 500.0

    def sv_mult(self, t: float) -> float:
        mult = 1.0
        for time, val, uninh in self.points:
            if time > t:
                break
            if not uninh and val < 0:
                mult = 100.0 / -val   # -100 => 1.0x, -50 => 2.0x
            elif uninh:
                mult = 1.0
        return mult


def _parse_timing(block: str) -> _Timing:
    pts: list[tuple[float, float, bool]] = []
    hs: list[tuple[float, int, int, int]] = []
    kpts: list[tuple[float, bool]] = []   # (time, kiai) for kiai ranges
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        time = _f(parts[0], 0.0)
        beat = _f(parts[1], 500.0)
        uninherited = True if len(parts) < 7 else parts[6].strip() == "1"
        pts.append((time, beat, uninherited))
        kpts.append((time, bool(_i(parts[7], 0) & 1) if len(parts) >= 8 else False))
        # hitsound state: time,beatLength,meter,sampleSet,sampleIndex,volume,…
        if len(parts) >= 6:
            hs.append((time, _i(parts[3], 0), _i(parts[4], 0),
                       max(0, min(100, _i(parts[5], 100)))))
    pts.sort(key=lambda p: p[0])
    hs.sort(key=lambda p: p[0])
    kpts.sort(key=lambda p: p[0])
    # Kiai regions: a TP with the kiai bit on runs until the next TP with it
    # off (or end of map). Only affects LEGACY skins that ship
    # fruit-catcher-kiai; the Argon default catcher ignores kiai (lazer
    # ArgonCatcher does too).
    kiai_ranges: list[tuple[float, float]] = []
    _kstart: float | None = None
    for _kt, _kon in kpts:
        if _kon and _kstart is None:
            _kstart = _kt
        elif not _kon and _kstart is not None:
            kiai_ranges.append((_kstart, _kt)); _kstart = None
    if _kstart is not None:
        kiai_ranges.append((_kstart, float("inf")))
    return _Timing(pts, hs, kiai_ranges)


def _i(s, default: int) -> int:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


# --- hit objects --------------------------------------------------------------

def _parse_hit_sample(s: str) -> tuple[int, int, int, int, str]:
    """The trailing hitSample field `normalSet:additionSet:index:volume:filename`
    (later fields optional). Returns (normal_set, addition_set, index, volume,
    filename); zeros mean 'inherit from the timing point'."""
    parts = (s or "").split(":")

    def _n(i: int) -> int:
        try:
            return int(parts[i])
        except (IndexError, ValueError):
            return 0

    fname = parts[4].strip() if len(parts) > 4 else ""
    return _n(0), _n(1), _n(2), _n(3), fname


# lazer Banana.default_banana_samples — BananaHitSampleInfo("Gameplay/
# metronomelow" / "Gameplay/catch-banana", volume 100): every caught banana
# plays THIS, never the spinner's own hitsounds. One shared instance.
_BANANA_SAMPLE = HitSample(volume=100, kind="banana")


def _parse_hit_objects(block: str, *, timing, slider_mult, tick_rate, hr, lazer=False, rng=None) -> list[CatchObject]:
    out: list[CatchObject] = []
    combo_index = -1  # incremented to 0 on the first new-combo
    started = False
    # applyHardRockOffset chain state (CatchBeatmapProcessor): the previous
    # object's (possibly already-offset) x and start time. Only top-level
    # CIRCLE fruits receive HR offsets; juice streams update the chain state
    # with stable's two known bugs, faithfully ported (see _slider last-pos).
    last_pos: float | None = None
    last_t = 0.0

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split(",")
        if len(f) < 4:
            continue
        x = float(f[0])
        time = int(float(f[2]))
        typ = int(f[3])

        is_new = bool(typ & _TYPE_NEW_COMBO) or not started
        started = True
        if is_new:
            combo_index += 1

        if typ & _TYPE_CIRCLE:
            xe = x
            if rng is not None and hr:
                xe, last_pos, last_t = _hard_rock_offset(x, float(time),
                                                         last_pos, last_t, rng)
                xe = _clamp(xe, 0.0, 512.0)   # CatchHitObject.EffectiveX clamp
            bits = _i(f[4], 0) if len(f) > 4 else 0
            ns, ads, idx, vol, fname = _parse_hit_sample(f[5] if len(f) > 5 else "")
            out.append(CatchObject(time, xe, ObjType.FRUIT, combo_index, is_new,
                                   sample=HitSample(bits, ns, ads, idx, vol,
                                                    fname, "hit")))
        elif typ & _TYPE_SLIDER:
            out.extend(_slider_objects(
                f, x, time, combo_index, is_new,
                timing=timing, slider_mult=slider_mult, tick_rate=tick_rate, hr=hr,
                lazer=lazer, rng=rng,
            ))
            if rng is not None and hr:
                # lazer `case JuiceStream:` — two stable bugs kept verbatim:
                # lastPosition = OriginalX + LAST CONTROL POINT x (the raw
                # curve point, NOT the computed path end), and lastStartTime =
                # the stream's START time (not its end).
                last_pos = _last_curve_x(f[5], x) if len(f) > 5 else x
                last_t = float(time)
        elif typ & _TYPE_SPINNER:
            end = int(float(f[5])) if len(f) > 5 else time + 1000
            out.extend(_banana_shower(time, end, combo_index, is_new, rng=rng))
    return out


def _last_curve_x(curve: str, x0: float) -> float:
    """The slider's last raw control point x (absolute osu px). lazer:
    `juiceStream.OriginalX + Path.ControlPoints[^1].Position.X` — control
    points are stored head-relative, so this is simply the last `px:py`
    token's x; a pointless curve falls back to the head x."""
    last = x0
    for p in curve.split("|")[1:]:
        if ":" in p:
            try:
                last = float(p.split(":", 1)[0])
            except ValueError:
                continue
    return last


def _hard_rock_offset(x: float, t: float, last_pos: float | None,
                      last_t: float, rng) -> tuple[float, float | None, float]:
    """osu!lazer CatchBeatmapProcessor.applyHardRockOffset, exact port
    (itself a faithful reproduction of stable's HR offset chain, including
    the int-truncated timeDiff). Returns (offset_x, last_pos', last_t')."""
    pos = x
    if last_pos is None:
        return pos, pos, t
    position_diff = pos - last_pos
    time_diff = int(t - last_t)          # stable bug: int truncation, kept
    if time_diff > 1000:
        return pos, pos, t
    if position_diff == 0.0:
        pos = _apply_random_offset(pos, time_diff / 4.0, rng)
        return pos, last_pos, last_t     # stable bug: chain state NOT advanced
    if abs(position_diff) < time_diff / 3.0:
        pos = _apply_offset(pos, position_diff)
    return pos, pos, t


def _apply_random_offset(position: float, max_offset: float, rng) -> float:
    """applyRandomOffset: random direction, magnitude min(20, Next(0, max)),
    clamped inside the playfield by flipping direction at the walls."""
    right = rng.next_bool()
    # LegacyRandom.Next(double, double) == (int)(lo + NextDouble()*(hi-lo)) —
    # the DOUBLE overload (maxOffset stays fractional inside the multiply).
    rand = min(20.0, float(rng.next_range(0.0, max(0.0, max_offset))))
    if right:
        return position + rand if position + rand <= 512.0 else position - rand
    return position - rand if position - rand >= 0.0 else position + rand


def _apply_offset(position: float, amount: float) -> float:
    """applyOffset: shift by `amount` only if the result stays in-field."""
    if amount > 0.0:
        if position + amount < 512.0:
            position += amount
    else:
        if position + amount > 0.0:
            position += amount
    return position


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# Degenerate-map guard: cap nested-object generation so a converted std
# "2B"/aspire map with billion-pixel sliders can't generate billions of droplet
# objects and OOM the render during setup (2026-07: one such map — beatmap
# 4679228, max slider 554e9 px — ballooned a render subprocess to ~63G before a
# single frame drew). Real sliders never approach these caps, so every
# legitimate map stays byte-identical; only a degenerate slider is bounded.
_MAX_TICKS_PER_SPAN = 10_000
_MAX_TINY_PER_GAP = 10_000
_MAX_SPANS = 10_000


def _slider_objects(f, x0, time, combo_index, is_new, *, timing, slider_mult, tick_rate, hr, lazer=False, rng=None):
    """Faithful osu!catch juice-stream nesting (lazer JuiceStream):
    Head/Repeat/Tail -> Fruit, Tick -> large Droplet, gaps filled with
    TinyDroplets. This is what makes the accuracy denominator real.

    Hitsounds ride along exactly as lazer's JuiceStream.CreateNestedHitObjects
    assigns Samples: head/repeat/tail fruits get GetNodeSamples(node) — the
    per-edge edgeSounds bits + edgeSets banks (falling back to the stream's own
    hitSound/hitSample) — large droplets get the stream's samples renamed
    "slidertick" (kind="tick"), and TINY droplets get none (silent).
    """
    # slider line: x,y,time,type,hitSound,curve,slides,length[,edgeSounds]
    # [,edgeSets][,hitSample]
    body_bits = _i(f[4], 0) if len(f) > 4 else 0
    hs_ns, hs_as, hs_idx, hs_vol, _hs_fname = _parse_hit_sample(
        f[10] if len(f) > 10 else "")
    edge_bits: list[int] = []
    if len(f) > 8 and f[8].strip():
        edge_bits = [_i(tok, body_bits) for tok in f[8].split("|")]
    edge_sets: list[tuple[int, int]] = []
    if len(f) > 9 and f[9].strip():
        for tok in f[9].split("|"):
            ab = tok.split(":")
            edge_sets.append((_i(ab[0], 0) if len(ab) > 0 else 0,
                              _i(ab[1], 0) if len(ab) > 1 else 0))

    def _node_sample(i: int) -> HitSample:
        """lazer GetNodeSamples(i): NodeSamples[i] when present, else the
        stream's own Samples; edgeSets banks of 0 fall through to the
        stream's hitSample banks (readCustomSampleBanks defaults)."""
        bits = edge_bits[i] if i < len(edge_bits) else body_bits
        ns, ads = edge_sets[i] if i < len(edge_sets) else (0, 0)
        return HitSample(bits, ns or hs_ns, ads or hs_as, hs_idx, hs_vol,
                         "", "hit")

    # large droplet = the stream's samples with the name swapped to
    # "slidertick" (bank/index/volume kept) — lazer dropletSamples.
    tick_sample = HitSample(body_bits, hs_ns, hs_as, hs_idx, hs_vol, "", "tick")

    head = [CatchObject(time, x0, ObjType.FRUIT, combo_index, is_new,
                        sample=_node_sample(0))]
    if len(f) < 8:
        return head

    y0 = _f(f[1], 0.0)   # head Y — needed for correct P/B slider-curve geometry
    curve = f[5]
    spans = int(f[6]) if f[6].isdigit() else 1
    spans = min(spans, _MAX_SPANS)   # degenerate-map guard
    pixel_length = _f(f[7], 0.0)
    length = pixel_length
    # Lazer replays: use the bit-exact lazer SliderPath (pixel-length truncated).
    # Stable replays: stable's catch positions differ (no truncation); the
    # full-polyline approximation matches stable closely.
    path = SliderPath(curve, x0, hr, length) if lazer else _SliderPath(x0, y0, curve, hr, length)

    beat = timing.beat_length(time)
    sv = timing.sv_mult(time)
    # velocity = base_scoring_distance(100) * SliderMultiplier * SV / beatLength
    velocity = 100.0 * slider_mult * sv / beat if beat > 0 else 0.0
    if velocity <= 0 or length <= 0:
        return head
    span_dur = length / velocity
    tick_distance = (100.0 * slider_mult * sv) / tick_rate if tick_rate > 0 else length
    tick_distance = max(0.0, min(tick_distance, length))
    min_from_end = velocity * 10.0

    # --- slider events in time order: head, ticks/repeats per span, tail ---
    # 4th field = hitsound node index (head 0, repeat s+1, tail spans; -1 tick)
    events: list[tuple[float, float, str, int]] = [(float(time), 0.0, "head", 0)]
    if tick_distance != 0:
        for s in range(spans):
            span_start = time + s * span_dur
            rev = (s % 2) == 1
            ticks: list[tuple[float, float, str, int]] = []
            d = tick_distance
            _nt = 0
            while d <= length - min_from_end and _nt < _MAX_TICKS_PER_SPAN:
                pp = d / length
                tp = (1.0 - pp) if rev else pp
                ticks.append((span_start + tp * span_dur, pp, "tick", -1))
                d += tick_distance
                _nt += 1
            ticks.sort(key=lambda e: e[0])
            events.extend(ticks)
            if s < spans - 1:
                rep_pp = 1.0 if (s % 2 == 0) else 0.0
                events.append((span_start + span_dur, rep_pp, "repeat", s + 1))
    total_dur = spans * span_dur
    tail_pp = 1.0 if (spans % 2 == 1) else 0.0
    # osu! LegacyLastTickOffset: on legacy (.osu) beatmaps the final scoring
    # point of a slider lands 36ms before the true end (clamped to >= the last
    # span's midpoint). This shortens the final gap, so fewer tiny droplets are
    # generated near the tail — matching how the game converts these maps.
    LEGACY_LAST_TICK_OFFSET = 36.0
    last_span_start = time + (spans - 1) * span_dur
    tail_time = max(last_span_start + span_dur / 2.0,
                    time + total_dur - LEGACY_LAST_TICK_OFFSET)
    events.append((tail_time, tail_pp, "tail", spans))

    # Emit nested objects in osu's exact order (head, [tinies], event, ...),
    # consuming the RNG per object as CatchBeatmapProcessor does: tiny droplet
    # -> Next(-20,20) XOffset; large droplet -> Next() (rotation, no offset);
    # fruit -> nothing (non-HardRock).
    out: list[CatchObject] = list(head)
    prev = events[0]
    for cur in events[1:]:
        # osu!catch generates tinies against INTEGER-truncated event times:
        # lazer computes sinceLastTick = (int)e.Time - (int)lastEvent.Value.Time
        # (JuiceStream.CreateNestedHitObjects). Using the raw float gap here
        # over-halves the interval near 100*2^k boundaries (float 200.1 keeps
        # halving to 3 tinies where int 200 stops at 1), over-generating tiny
        # droplets on some tick-rate/SV/slider patterns. Truncate to match the
        # game exactly. The tiny StartTime/PathProgress below still use the raw
        # float prev[0]/prev[1], exactly as lazer uses lastEvent.Value.Time.
        since = int(cur[0]) - int(prev[0])
        if since > 80:
            tb = since
            while tb > 100:
                tb /= 2.0
            t = tb
            _ntd = 0
            while t < since and _ntd < _MAX_TINY_PER_GAP:
                pp = prev[1] + (t / since) * (cur[1] - prev[1])
                ox = path.x_at(pp)
                if rng is not None:
                    ox += _clamp(rng.next_range(-20, 20), -ox, 512.0 - ox)
                out.append(CatchObject(int(prev[0] + t), ox, ObjType.TINY_DROPLET, combo_index, False))
                t += tb
                _ntd += 1
        if cur[2] == "tick":
            if rng is not None:
                rng.next()   # osu!stable retrieved a random droplet rotation
            out.append(CatchObject(int(cur[0]), path.x_at(cur[1]), ObjType.DROPLET, combo_index, False,
                                   sample=tick_sample))
        else:  # repeat / tail fruit (no RNG without HardRock)
            # The tail event time is the LegacyLastTick (end-36) so tiny-droplet
            # generation stops there; but the tail FRUIT is caught at the TRUE
            # slider end, so emit it at time+total_dur.
            ft = int(time + total_dur) if cur[2] == "tail" else int(cur[0])
            out.append(CatchObject(ft, path.x_at(cur[1]), ObjType.FRUIT, combo_index, False,
                                   sample=_node_sample(cur[3])))
        prev = cur
    return out


def _banana_shower(time, end, combo_index, is_new, rng=None):
    """Exact osu! BananaShower.createBananas: spacing = duration halved until
    <=100, a banana at each step. Each banana consumes the RNG (x + 3 Next),
    which MUST match so the RNG stays synced for later tiny droplets.
    """
    out: list[CatchObject] = []
    start = int(time)
    end_t = int(end)
    spacing = float(end - time)
    while spacing > 100:
        spacing /= 2.0
    if spacing <= 0:
        return out
    t = float(start)
    idx = 0
    while t <= end_t:
        if rng is not None:
            x = rng.next_double() * 512.0
            rng.next()  # banana type
            rng.next()  # banana rotation
            rng.next()  # banana colour
        else:
            x = (math.sin(idx * 2.399963) * 0.5 + 0.5) * 512.0
        out.append(CatchObject(int(t), x, ObjType.BANANA, combo_index, is_new and idx == 0,
                               sample=_BANANA_SAMPLE))
        t += spacing
        idx += 1
    return out


def _mark_hyperdash(objects: list[CatchObject], cs: float) -> list[CatchObject]:
    """Flag palpable objects that require a hyperdash to reach the next one.

    Exact port of osu!lazer's CatchBeatmapProcessor.initialiseHyperDash:
    palpable objects = Fruits + large Droplets (NO tiny droplets, NO bananas),
    ordered by start time. Walk consecutive pairs carrying lastDirection /
    lastExcess; flag `current` as a hyperdash to next.x when the catcher
    cannot cover the gap in time at base walk speed (distanceToHyper < 0).
    """
    from dataclasses import replace

    # Catcher constants (osu!lazer Catcher.cs). BASE_SPEED is osu px per ms;
    # hyperdash generation uses base speed, NOT dash speed.
    BASE_SIZE = 106.75
    BASE_SPEED = 1.0
    # CalculateCatchWidth = BASE_SIZE * |scale| * ALLOWED_CATCH_RANGE (0.8);
    # halfCatcherWidth = width / 2 / ALLOWED_CATCH_RANGE — the 0.8 cancels.
    scale = 1.0 - 0.7 * (cs - 5.0) / 5.0
    half_catcher_width = BASE_SIZE * abs(scale) / 2.0

    # GetPalpableObjects orders by StartTime (slider tails can be emitted
    # out of file order here, so the sort matters).
    palpable = sorted(
        (o for o in objects if o.kind in (ObjType.FRUIT, ObjType.DROPLET)),
        key=lambda o: o.time_ms,
    )

    hyper_ids: dict[int, float] = {}
    last_direction = 0
    last_excess = half_catcher_width
    for cur, nxt in zip(palpable, palpable[1:]):
        this_direction = 1 if nxt.x > cur.x else -1
        # The -1000/60/4 term is 1/4 of a 60fps frame of slack, per lazer.
        time_to_next = nxt.time_ms - cur.time_ms - 1000.0 / 60.0 / 4.0
        distance_to_next = abs(nxt.x - cur.x) - (
            last_excess if last_direction == this_direction else half_catcher_width
        )
        distance_to_hyper = time_to_next * BASE_SPEED - distance_to_next
        if distance_to_hyper < 0:
            hyper_ids[id(cur)] = nxt.x
            last_excess = half_catcher_width
        else:
            last_excess = max(0.0, min(distance_to_hyper, half_catcher_width))
        last_direction = this_direction

    return [replace(o, hyperdash=True, hyper_target_x=hyper_ids[id(o)])
            if id(o) in hyper_ids else o for o in objects]


class _SliderPath:
    """Samples a slider curve into a dense polyline and exposes x_at(frac).

    Supports L(inear), P(erfect circle, 3 points) and B(ezier). C(atmull) is
    legacy-rare; we approximate it as a polyline through the control points.
    """

    def __init__(self, x0: float, y0: float, curve: str, hr: bool, expected_length: float = 0.0):
        parts = curve.split("|")
        kind = parts[0]
        pts = [(x0, y0)]
        for p in parts[1:]:
            if ":" in p:
                px, py = p.split(":")
                fx = float(px)
                pts.append((fx, float(py)))
        self._poly = self._build(kind, pts)
        self._cum, self._total = self._arc_lengths(self._poly)
        # osu truncates (or extends) the curve to the slider's pixel length, so
        # frac maps over `length`, NOT the raw control-point arc. Without this a
        # P/B arc longer than its length puts the tail/droplets past the real end.
        self._exp = expected_length if expected_length > 0 else self._total

    def x_at(self, frac: float) -> float:
        frac = max(0.0, min(1.0, frac))
        target = min(frac * self._exp, self._total)
        # find segment by cumulative length
        lo, hi = 0, len(self._cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return self._poly[0][0]
        a, b = self._poly[lo - 1], self._poly[lo]
        seg = self._cum[lo] - self._cum[lo - 1]
        if seg <= 0:
            return b[0]
        f = (target - self._cum[lo - 1]) / seg
        return a[0] + (b[0] - a[0]) * f

    @staticmethod
    def _build(kind: str, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(pts) < 2:
            return pts or [(256.0, 0.0)]
        if kind == "L":
            return _densify_linear(pts)
        if kind == "P" and len(pts) == 3:
            arc = _perfect_circle(pts)
            return arc if arc else _densify_linear(pts)
        # B (bezier) and C/fallback: split bezier at repeated anchor points
        return _bezier_path(pts)

    @staticmethod
    def _arc_lengths(poly):
        cum = [0.0]
        for i in range(1, len(poly)):
            dx = poly[i][0] - poly[i - 1][0]
            dy = poly[i][1] - poly[i - 1][1]
            cum.append(cum[-1] + math.hypot(dx, dy))
        return cum, cum[-1] if cum else 0.0


def _densify_linear(pts, step=5.0):
    out = [pts[0]]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(dist / step))
        for k in range(1, n + 1):
            f = k / n
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return out


def _bezier_path(pts):
    """Piecewise bezier: osu splits segments at consecutive duplicate points."""
    out = [pts[0]]
    seg_start = 0
    for i in range(1, len(pts)):
        if pts[i] == pts[i - 1] or i == len(pts) - 1:
            seg = pts[seg_start:i + 1] if i == len(pts) - 1 else pts[seg_start:i]
            out.extend(_bezier_segment(seg)[1:])
            seg_start = i
    return out


def _bezier_segment(ctrl, samples=40):
    if len(ctrl) < 2:
        return ctrl
    out = []
    for s in range(samples + 1):
        t = s / samples
        out.append(_de_casteljau(ctrl, t))
    return out


def _de_casteljau(ctrl, t):
    pts = list(ctrl)
    while len(pts) > 1:
        pts = [
            (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
             pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t)
            for i in range(len(pts) - 1)
        ]
    return pts[0]


def _perfect_circle(pts, step_deg=3.0):
    (ax, ay), (bx, by), (cx, cy) = pts
    # circumcenter
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    a0 = math.atan2(ay - uy, ax - ux)
    a1 = math.atan2(cy - uy, cx - ux)
    amid = math.atan2(by - uy, bx - ux)
    # ensure we sweep through the middle point
    def norm(a):
        while a < a0:
            a += 2 * math.pi
        return a
    a1n, amidn = norm(a1), norm(amid)
    if amidn > a1n:  # wrong way; sweep negative
        a1n = a1 - 2 * math.pi if a1 > a0 else a1
        sweep = a1n - a0
    else:
        sweep = a1n - a0
    n = max(2, int(abs(math.degrees(sweep)) / step_deg))
    return [(ux + r * math.cos(a0 + sweep * k / n), uy + r * math.sin(a0 + sweep * k / n))
            for k in range(n + 1)]


# --- tiny helpers -------------------------------------------------------------

def _split_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("[") and line.rstrip().endswith("]"):
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur = line.strip()[1:-1]
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out


def _kv(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_combo_colors(block: str) -> list:
    """[Colours] Combo1..N -> RGB tuples in ascending combo index (osu!)."""
    combos = {}
    for k, v in _kv(block).items():
        kl = k.lower()
        if kl.startswith("combo") and kl[5:].isdigit():
            p = [t.strip() for t in v.split(",")]
            if len(p) >= 3:
                try:
                    combos[int(kl[5:])] = (int(p[0]), int(p[1]), int(p[2]))
                except ValueError:
                    continue
    return [combos[i] for i in sorted(combos)]


def _parse_breaks(events: str) -> list:
    """Break periods from [Events]: lines '2,start,end' or 'Break,start,end'."""
    out = []
    for line in events.splitlines():
        f = line.split(",")
        if len(f) >= 3 and f[0].strip() in ("2", "Break"):
            try:
                out.append((int(float(f[1])), int(float(f[2]))))
            except ValueError:
                continue
    return out


def _parse_background(events: str) -> str | None:
    for line in events.splitlines():
        f = line.split(",")
        if len(f) >= 3 and f[0].strip() in ("0", "Background"):
            return f[2].strip().strip('"')
    return None


def _f(s, default: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default
