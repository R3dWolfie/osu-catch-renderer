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

from .legacy_random import RNG_SEED, LegacyRandom
from .models import CatchBeatmap, CatchObject, ObjType
from .sliderpath import SliderPath

# HitObject type bitfield
_TYPE_CIRCLE = 1 << 0
_TYPE_SLIDER = 1 << 1
_TYPE_NEW_COMBO = 1 << 2
_TYPE_SPINNER = 1 << 3


class BeatmapParseError(RuntimeError):
    pass


def parse_beatmap(path: Path, *, mods: int = 0, lazer: bool = False) -> CatchBeatmap:
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
    hr = bool(mods & (1 << 4))   # also mirrors x in catch
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
    # osu!catch position offsets (banana x + tiny-droplet +-20 XOffset) need the
    # RNG consumed in osu's EXACT object order, which requires bit-exact
    # tiny/droplet/banana generation — any mismatch desyncs the RNG and makes
    # tiny offsets random-wrong (worse than no offset). Disabled until the
    # nested generation is verified bit-exact. Set RNG_SEED-seeded rng to enable.
    rng = None
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
    )


# --- timing -------------------------------------------------------------------

class _Timing:
    """Resolves beat length and SV multiplier at any time."""

    def __init__(self, points: list[tuple[float, float, bool]]):
        # (time, value, uninherited). value = beatLength for red lines,
        # negative-inverse SV for green lines.
        self.points = points

    def beat_length(self, t: float) -> float:
        bl = 500.0
        for time, val, uninh in self.points:
            if time > t:
                break
            if uninh and val > 0:
                bl = val
        return bl

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
    pts.sort(key=lambda p: p[0])
    return _Timing(pts)


# --- hit objects --------------------------------------------------------------

def _parse_hit_objects(block: str, *, timing, slider_mult, tick_rate, hr, lazer=False, rng=None) -> list[CatchObject]:
    out: list[CatchObject] = []
    combo_index = -1  # incremented to 0 on the first new-combo
    started = False

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
        if hr:
            x = 512.0 - x

        is_new = bool(typ & _TYPE_NEW_COMBO) or not started
        started = True
        if is_new:
            combo_index += 1

        if typ & _TYPE_CIRCLE:
            out.append(CatchObject(time, x, ObjType.FRUIT, combo_index, is_new))
        elif typ & _TYPE_SLIDER:
            out.extend(_slider_objects(
                f, x, time, combo_index, is_new,
                timing=timing, slider_mult=slider_mult, tick_rate=tick_rate, hr=hr,
                lazer=lazer, rng=rng,
            ))
        elif typ & _TYPE_SPINNER:
            end = int(float(f[5])) if len(f) > 5 else time + 1000
            out.extend(_banana_shower(time, end, combo_index, is_new, rng=rng))
    return out


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _slider_objects(f, x0, time, combo_index, is_new, *, timing, slider_mult, tick_rate, hr, lazer=False, rng=None):
    """Faithful osu!catch juice-stream nesting (lazer JuiceStream):
    Head/Repeat/Tail -> Fruit, Tick -> large Droplet, gaps filled with
    TinyDroplets. This is what makes the accuracy denominator real.
    """
    head = [CatchObject(time, x0, ObjType.FRUIT, combo_index, is_new)]
    if len(f) < 8:
        return head

    y0 = _f(f[1], 0.0)   # head Y — needed for correct P/B slider-curve geometry
    curve = f[5]
    spans = int(f[6]) if f[6].isdigit() else 1
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
    events: list[tuple[float, float, str]] = [(float(time), 0.0, "head")]
    if tick_distance != 0:
        for s in range(spans):
            span_start = time + s * span_dur
            rev = (s % 2) == 1
            ticks: list[tuple[float, float, str]] = []
            d = tick_distance
            while d <= length - min_from_end:
                pp = d / length
                tp = (1.0 - pp) if rev else pp
                ticks.append((span_start + tp * span_dur, pp, "tick"))
                d += tick_distance
            ticks.sort(key=lambda e: e[0])
            events.extend(ticks)
            if s < spans - 1:
                rep_pp = 1.0 if (s % 2 == 0) else 0.0
                events.append((span_start + span_dur, rep_pp, "repeat"))
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
    events.append((tail_time, tail_pp, "tail"))

    # Emit nested objects in osu's exact order (head, [tinies], event, ...),
    # consuming the RNG per object as CatchBeatmapProcessor does: tiny droplet
    # -> Next(-20,20) XOffset; large droplet -> Next() (rotation, no offset);
    # fruit -> nothing (non-HardRock).
    out: list[CatchObject] = list(head)
    prev = events[0]
    for cur in events[1:]:
        since = cur[0] - prev[0]
        if since > 80:
            tb = since
            while tb > 100:
                tb /= 2.0
            t = tb
            while t < since:
                pp = prev[1] + (t / since) * (cur[1] - prev[1])
                ox = path.x_at(pp)
                if rng is not None:
                    ox += _clamp(rng.next_range(-20, 20), -ox, 512.0 - ox)
                out.append(CatchObject(int(prev[0] + t), ox, ObjType.TINY_DROPLET, combo_index, False))
                t += tb
        if cur[2] == "tick":
            if rng is not None:
                rng.next()   # osu!stable retrieved a random droplet rotation
            out.append(CatchObject(int(cur[0]), path.x_at(cur[1]), ObjType.DROPLET, combo_index, False))
        else:  # repeat / tail fruit (no RNG without HardRock)
            # The tail event time is the LegacyLastTick (end-36) so tiny-droplet
            # generation stops there; but the tail FRUIT is caught at the TRUE
            # slider end, so emit it at time+total_dur.
            ft = int(time + total_dur) if cur[2] == "tail" else int(cur[0])
            out.append(CatchObject(ft, path.x_at(cur[1]), ObjType.FRUIT, combo_index, False))
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
        out.append(CatchObject(int(t), x, ObjType.BANANA, combo_index, is_new and idx == 0))
        t += spacing
        idx += 1
    return out


def _mark_hyperdash(objects: list[CatchObject], cs: float) -> list[CatchObject]:
    """Flag fruits that require a hyperdash to reach the next catchable object.

    Mirrors osu!catch's reachability test (catcher can/can't cover the gap in
    time at dash speed) using the lastExcess/lastDirection carry. DASH_SPEED is
    in osu px/ms; it's an approximation but flags the genuine hyper jumps.
    """
    from dataclasses import replace

    from .models import cs_to_catcher_half_width
    DASH_SPEED = 1.35   # osu px/ms, calibrated so only genuine big jumps flag
    half = cs_to_catcher_half_width(cs) / 0.8   # full catcher, not just catch range
    catchable = [o for o in objects if o.kind in (ObjType.FRUIT, ObjType.DROPLET)]
    hyper_ids: dict[int, float] = {}
    last_dir = 0
    last_excess = half
    for a, b in zip(catchable, catchable[1:]):
        dt = b.time_ms - a.time_ms
        if dt <= 0:
            continue
        this_dir = 1 if b.x > a.x else -1
        dist = abs(b.x - a.x) - (last_excess if last_dir == this_dir else half)
        reach = dt * DASH_SPEED - dist
        if reach < 0:
            hyper_ids[id(a)] = b.x
            last_excess = half
        else:
            last_excess = max(0.0, min(reach, half))
        last_dir = this_dir
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
                fx = 512.0 - float(px) if hr else float(px)
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
