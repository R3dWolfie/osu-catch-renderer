"""Simulate the catch playthrough and build per-frame draw lists.

The catcher path comes straight from the replay (we render what the player
did), so "caught" is decided geometrically: was the catcher within its
catch-range of the object's x at the object's time. That drives combo, a
score estimate, and an HP estimate. The HUD's *final* numbers come from the
replay's authoritative counts; the live values are our running simulation.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .assets import ARGON_CANVAS, ARGON_VARIANTS
from .models import (
    CatchBeatmap,
    CatchFrame,
    ObjType,
    RenderConfig,
    SceneState,
    Sprite,
    ar_to_preempt_ms,
    cs_to_catcher_half_width,
)
from .replay import catcher_x_at

PLAYFIELD = 512.0
FRUIT_TEX = {0: "fruit0", 1: "fruit1", 2: "fruit2", 3: "fruit3"}


def _hue(h: float):
    import colorsys
    return colorsys.hsv_to_rgb(h % 1.0, 0.85, 1.0)


def _obj_rand01(time_ms: float, x: float, series: int = 0) -> float:
    """Deterministic per-object random in [0,1) — our stand-in for lazer's
    StatelessRNG.NextSingle(RandomSeed, series), where an object's RandomSeed
    is its start time (DrawableCatchHitObject). Stable per object across
    frames, varies object-to-object; x joins the hash so two objects sharing
    a start time still differ. splitmix32-style finalizer for avalanche."""
    n = (int(round(time_ms)) * 2654435761
         + int(round(x * 127.0)) * 40503 + series * 69069) & 0xFFFFFFFF
    n ^= n >> 16
    n = (n * 0x7FEB352D) & 0xFFFFFFFF
    n ^= n >> 15
    n = (n * 0x846CA68B) & 0xFFFFFFFF
    n ^= n >> 16
    return n / 4294967296.0


@dataclass
class _Checkpoint:
    time: int
    combo: int
    score: int
    hp: float
    accuracy: float = 1.0
    max_combo: int = 0
    counts: tuple = (0, 0, 0, 0, 0)   # fruit, large-drop, tiny, miss-tiny, miss
    pp: float = 0.0


# osu!CATCH mod-score multipliers — from lazer's CatchScoreMultiplierCalculator
# (NOT the std/taiko table; catch values differ). EZ/NF 0.5, HD 1.06, HR 1.12,
# DT/NC 1.1 (rateAdjust 1+(1.5-1)/5), HT 0.3, FL 1.12, RX 0.1. Catch has NO
# Spun Out. Applied to the standardised 1,000,000 total.
_MOD_SCORE_MULT = {1 << 0: 0.5,   # NF
                   1 << 1: 0.5,   # EZ
                   1 << 3: 1.06,  # HD
                   1 << 4: 1.12,  # HR
                   1 << 6: 1.1,   # DT
                   1 << 7: 0.1,   # RX (Relax)
                   1 << 8: 0.3,   # HT
                   1 << 9: 1.1,   # NC
                   1 << 10: 1.12} # FL


def mods_score_multiplier(mods: int) -> float:
    mods = int(mods or 0)
    # NC (Nightcore) is stored as DT|NC (bits 64|512); the speed change is ONE
    # score multiplier, so drop the implied DT bit or it squares (1.23²=1.51,
    # pushing scores past the 1e6 max). Same for DC=HT|DC if ever present.
    if mods & (1 << 9):        # NC set → clear implied DT
        mods &= ~(1 << 6)
    m = 1.0
    for bit, mult in _MOD_SCORE_MULT.items():
        if mods & bit:
            m *= mult
    return m


class CatchSim:
    def __init__(self, beatmap: CatchBeatmap, frames: list[CatchFrame], cfg: RenderConfig,
                 skin=None, has_bg: bool = False, meta=None,
                 end_ms: int | None = None):
        self.bm = beatmap
        self.frames = frames
        self.cfg = cfg
        self.skin = skin
        self.meta = meta
        # skin.ini [CatchTheBeat] hyperdash cue colours (parsed by CatchSkin;
        # HyperDashFruit/AfterImage fall back to HyperDash, which defaults to
        # red). ONLY the skinned draw paths read these — the Argon elements
        # keep lazer's hard red (the Argon skin has no skin.ini to honour).
        _red = (1.0, 0.0, 0.0)
        self.hyper_rgb = (getattr(skin, "hyper_color", _red)
                          if skin is not None else _red)
        self.hyper_fruit_rgb = (getattr(skin, "hyper_fruit_color", _red)
                                if skin is not None else _red)
        self.hyper_after_rgb = (getattr(skin, "hyper_afterimage_color", _red)
                                if skin is not None else _red)
        # Overlay/versus mode forces the white Argon catcher so a per-player
        # colour tint reads cleanly (tinting a skin's already-coloured catcher
        # muddies it). Single renders leave this False = skin catcher as usual.
        self.force_argon_catcher = False
        # On a failed play, gameplay stops at the death time. Objects past
        # it were never reached, so they're excluded from BOTH the catch
        # simulation (otherwise the count-reconcile spreads the played
        # counts across unplayed objects → phantom misses) and the draw
        # loop. None = play the whole map.
        self._end_ms = end_ms
        self._objs = [o for o in beatmap.objects
                      if end_ms is None or o.time_ms <= end_ms]
        if not self._objs:   # death before any object — don't blank the sim
            self._objs = list(beatmap.objects)
        # lazer picks a fruit's VisualRepresentation from IndexInBeatmap % 4
        # (Fruit.GetVisualRepresentation). We approximate IndexInBeatmap with a
        # running index over the flattened palpable objects in beatmap order, so
        # successive fruits cycle Pear/Grape/Pineapple/Raspberry like the game.
        self._obj_index = {id(o): i for i, o in enumerate(beatmap.objects)}
        self.score_scale = 1.0
        self.acc_offset = 0.0   # shifts sim accuracy to the replay's real final
        self.final_counts = (0, 0, 0, 0, 0)  # (300, 100, 50, katu, miss)
        self.final_accuracy = 1.0
        self.has_bg = has_bg
        self.bg_dim = 0.30
        # approach window in the rate-adjusted (real) timeline the replay uses
        self.preempt = ar_to_preempt_ms(beatmap.ar) / beatmap.rate
        # Hidden (HD, mod bit 8): fruits fade out as they near the catcher.
        self.hidden = bool((getattr(meta, "mods", 0) or 0) & 8)
        self.half = cs_to_catcher_half_width(beatmap.cs)
        # R3D intro splash (show_logo): render.py sets logo_start_ms when the
        # flag is on; the splash fades out exactly as the first fruit begins
        # its approach (first object time - preempt), matching the std splash.
        self.logo_start_ms: float | None = None
        self.first_spawn_ms = (min((o.time_ms for o in self._objs), default=0)
                               - self.preempt)

        w, h = cfg.resolution
        self.screen_w, self.screen_h = w, h
        # osu!catch playfield mapping (CatchPlayfieldAdjustmentContainer): the
        # 4:3 base (1024x768) is scaled to fit the screen by height (for >=4:3
        # outputs), and the 512-unit playfield fills 0.8 of that base width,
        # centred. So objects live in a centred ~60%-width column, NOT the full
        # screen — mapping to full width is what made everything look zoomed.
        BASE_W, BASE_H, PF_ADJ = 1024.0, 768.0, 0.8
        pf_scale = h / BASE_H
        self.unit_px = (BASE_W * PF_ADJ / PLAYFIELD) * pf_scale   # = 1.6 * pf_scale
        self.x_off = (w - BASE_W * pf_scale) / 2.0 + (BASE_W * (1.0 - PF_ADJ) / 2.0) * pf_scale
        self.x_scale = self.unit_px   # back-comp alias
        self.plane_y = h * cfg.catcher_plane
        # object scale from CS (lazer CalculateScaleFromCircleSize): (1-0.7(cs-5)/5)/2
        self.obj_scale = (1.0 - 0.7 * (beatmap.cs - 5.0) / 5.0) / 2.0
        # fruit display diameter ~= 2 * OBJECT_RADIUS(64) * scale, in screen px
        self.fruit_screen = max(10.0, 128.0 * self.obj_scale * self.unit_px)
        # catcher visual width: BASE_SIZE 106.75 * (scale*2), in screen px
        self.catcher_w = 106.75 * self.obj_scale * 2.0 * self.unit_px

        self._caught: list[bool] = []
        self._checkpoints: list[_Checkpoint] = []
        self._catches: list = []   # (time, x, combo_index, hyper, combo)
        # Plate pile (lazer computePositionInStack): per caught fruit, a mutable
        # [catch_time, offset_x, offset_y (osu units, ride the catcher),
        #  combo_index, hyper, clear_time, anim] — clear_time/anim filled by the
        # post-sim group pass ("explode" if the combo's last object was hit,
        # else "drop").
        self._plate: list = []
        self._hyper_windows: list[tuple[int, int]] = []    # catcher glows red in these
        self._calibrate_offset()
        self._simulate()
        # PERF: state_at bisects this precomputed list instead of rebuilding
        # it every call (it is immutable after _simulate).
        self._cp_times = [c.time for c in self._checkpoints]
        # PERF: build_scene narrows the falling-object scan to the visible
        # time window by bisecting these start times. Only valid when the
        # object list is time-sorted (it is, in beatmap order — but verify;
        # an unsorted list falls back to the full scan, output-identical).
        self._obj_times = [o.time_ms for o in self._objs]
        self._objs_sorted = all(a <= b for a, b in
                                zip(self._obj_times, self._obj_times[1:]))

    # --- simulation -----------------------------------------------------------

    # lazer combo-portion log accumulation constants (CatchScoreProcessor)
    _COMBO_BASE = 4
    _COMBO_CAP = 200

    def _calibrate_offset(self) -> None:
        """A handful of catch replays carry a constant timeline shift vs the
        beatmap: the recorded catcher path is offset by a fixed lag, so every
        fruit lands while the plate is somewhere else (the "catcher wrong
        position" bug). Detect it by cross-correlating the catcher trajectory
        against the catchable fruit positions, then shift the frames to match
        ONLY when the evidence is overwhelming -- a sharp, high alignment peak
        well above the no-shift baseline. Normal/stable replays calibrate to
        ~0 and are left completely untouched, so this can never harm them.
        """
        import sys
        frames = self.frames
        if len(frames) < 50:
            return
        span_end = frames[-1].time_ms
        objs = [o for o in self._objs
                if o.kind is not ObjType.BANANA and o.time_ms <= span_end]
        if len(objs) > 2000:                       # cap cost on long maps
            objs = objs[:: (len(objs) // 2000) + 1]
        if len(objs) < 60:                         # too few to trust a peak
            return
        half = self.half

        def hit_rate(off: int) -> float:
            hit = 0
            for o in objs:
                cx, _ = catcher_x_at(frames, o.time_ms + off)
                if abs(cx - o.x) <= half:
                    hit += 1
            return hit / len(objs)

        base = hit_rate(0)
        best_off, best = 0, base
        for off in range(-4000, 1001, 50):         # coarse sweep
            r = hit_rate(off)
            if r > best:
                best, best_off = r, off
        if best_off:                               # refine around the peak
            for off in range(best_off - 50, best_off + 51, 10):
                r = hit_rate(off)
                if r > best:
                    best, best_off = r, off

        # Strict guard: only correct an unmistakable constant shift.
        if abs(best_off) >= 150 and best >= 0.85 and (best - base) >= 0.20:
            shift = -best_off
            self.frames = [CatchFrame(time_ms=f.time_ms + shift, x=f.x,
                                      dashing=f.dashing) for f in frames]
            print(f"[catch] replay timeline shift {shift:+d}ms applied "
                  f"(catcher alignment {base * 100:.0f}% -> {best * 100:.0f}%)",
                  file=sys.stderr, flush=True)

    def _simulate(self) -> None:
        import math
        log_cap = math.log(self._COMBO_CAP, self._COMBO_BASE)
        objs = self._objs

        # --- pass 1: geometric catch + signed margin (dist-half) per object ---
        margin: list[float] = []
        for obj in objs:
            cx, _ = catcher_x_at(self.frames, obj.time_ms)
            d = abs(cx - obj.x)
            self._caught.append(d <= self.half)
            margin.append(d - self.half)   # <=0 caught; smaller |margin| = more borderline

        # --- reconcile borderline calls to the replay's authoritative per-type
        # counts. Geometry gives the timing/positions; the osr gives the truth
        # (tiny-droplet RNG offsets can't be reproduced bit-exact, so a few
        # boundary objects land on the wrong side). Flip the least-confident
        # calls so each category's caught total matches the replay exactly.
        m = self.meta
        if m is not None:
            def _reconcile(kind, target_caught):
                if target_caught is None:
                    return
                idxs = [i for i, o in enumerate(objs) if o.kind is kind]
                if not idxs:
                    return
                diff = target_caught - sum(self._caught[i] for i in idxs)
                if diff > 0:   # need more caught: flip closest missed (smallest +margin)
                    for i in sorted((i for i in idxs if not self._caught[i]),
                                    key=lambda i: margin[i])[:diff]:
                        self._caught[i] = True
                elif diff < 0:  # need fewer: flip most-marginal caught (margin nearest 0)
                    for i in sorted((i for i in idxs if self._caught[i]),
                                    key=lambda i: margin[i])[diff:]:
                        self._caught[i] = False
            _reconcile(ObjType.FRUIT, m.count_300)
            _reconcile(ObjType.DROPLET, m.count_100)
            _reconcile(ObjType.TINY_DROPLET, m.count_50)

        # --- pass 2: lazer-standardised ScoreV3 from reconciled catches ------
        # Base numerics per catch judgement: Fruit=Great=300, Droplet=
        # LargeTick=30 (both AFFECT COMBO + accuracy), TinyDroplet=SmallTick=10
        # (accuracy only, NO combo). combo_portion += base·√combo — the
        # power-based curve of the current lazer ScoreProcessor, NOT the old
        # log-based ScoreV2 the renderer used to scale to meta.score. The
        # 500k combo / 500k accuracy split is then ×mod-multiplier.
        # NOTE: banana-shower bonus (LargeBonus +50 each) is not modelled —
        # there are no Banana objects in the sim, so score is a hair low on
        # banana maps. Fine for ranking; flagged for exactness.
        _BASE = {ObjType.FRUIT: 300.0, ObjType.DROPLET: 30.0,
                 ObjType.TINY_DROPLET: 10.0}
        _combo_kinds = (ObjType.FRUIT, ObjType.DROPLET)
        max_combo_portion = 0.0
        max_base_total = 0.0
        _pc = 0
        for obj in objs:
            max_base_total += _BASE.get(obj.kind, 0.0)
            if obj.kind in _combo_kinds:
                _pc += 1
                max_combo_portion += _BASE[obj.kind] * (_pc ** 0.5)
        _mod_mult = mods_score_multiplier(getattr(self.meta, "mods", 0) or 0)

        combo = max_combo = 0
        hp = 1.0
        c300 = c100 = c50 = ckatu = cmiss = ctiny_miss = 0
        combo_portion = 0.0
        cur_base = cur_max_base = 0.0
        pending_hyper: int | None = None
        pending_target: float | None = None
        import random as _random
        _pile_ci = -1                          # current combo group for the pile
        _pile_placed: list = []                # (offset_x, offset_y) already placed
        _pile_adj = 128.0 * self.obj_scale * (10.0 / 64.0)   # jitter radius, osu units
        for obj, caught in zip(objs, self._caught):
            cur_max_base += _BASE.get(obj.kind, 0.0)
            if obj.kind in (ObjType.FRUIT, ObjType.DROPLET):
                base = _BASE[obj.kind]
                if pending_hyper is not None:
                    # lazer ends the hyperdash the INSTANT the catcher reaches the
                    # target x (SetHyperDashState clears on arrival), NOT at the
                    # next object — otherwise the catcher sits solid-red and
                    # stationary for the whole gap (the "bright red glow" bug).
                    # Scan the replay path forward for arrival, capped at the next
                    # object's time.
                    end = obj.time_ms
                    if pending_target is not None:
                        th = pending_hyper
                        while th < obj.time_ms:
                            cxh, _ = catcher_x_at(self.frames, th)
                            if abs(cxh - pending_target) <= self.half:
                                end = th
                                break
                            th += 16
                    self._hyper_windows.append((pending_hyper, end))
                    pending_hyper = None
                    pending_target = None
                if caught:
                    combo += 1
                    max_combo = max(max_combo, combo)
                    combo_portion += base * (combo ** 0.5)
                    cur_base += base
                    hp = min(1.0, hp + 0.025)
                    if obj.kind is ObjType.FRUIT:
                        c300 += 1
                        self._catches.append((obj.time_ms, obj.x, obj.combo_index, obj.hyperdash, combo))
                        # lazer computePositionInStack: land at where it was caught
                        # (offset from plate centre), then jitter ONLY to de-overlap
                        # against fruit already on the plate this combo.
                        if obj.combo_index != _pile_ci:
                            _pile_ci = obj.combo_index
                            _pile_placed = []
                        cxh, _ = catcher_x_at(self.frames, obj.time_ms)
                        # Cluster toward the catcher centre (0.55) instead of the
                        # full caught-offset, so the pile stacks UP rather than
                        # smearing across the whole bar width (Red: "stack on top
                        # of each other", not a diagonal chain).
                        px = (obj.x - cxh) * 0.55
                        py = 0.0
                        chk = _pile_adj * _pile_adj
                        rng = _random.Random(int(obj.time_ms))
                        _g = 0
                        while _g < 64 and any((px - fx) ** 2 + (py - fy) ** 2 < chk
                                              for fx, fy in _pile_placed):
                            px += rng.uniform(-_pile_adj * 0.45, _pile_adj * 0.45)
                            py -= rng.uniform(3.0, 7.0)     # build the tower UPWARD
                            _g += 1
                        _pile_placed.append((px, py))
                        self._plate.append([obj.time_ms, px, py, obj.combo_index,
                                            obj.hyperdash, None, None])
                    else:
                        c100 += 1
                    if obj.hyperdash:
                        pending_hyper = obj.time_ms
                        pending_target = obj.hyper_target_x
                else:
                    combo = 0
                    hp = max(0.0, hp - 0.10)
                    if obj.kind is ObjType.FRUIT:
                        cmiss += 1
                    else:
                        ckatu += 1
            elif obj.kind is ObjType.TINY_DROPLET:
                if caught:
                    c50 += 1
                    cur_base += _BASE[ObjType.TINY_DROPLET]
                else:
                    ctiny_miss += 1
            elif obj.kind is ObjType.BANANA and caught and self.skin is not None:
                # BANANAS ARE CAUGHT TOO ("bananas ignored by the platter"):
                # pass 1 already marks them caught geometrically, which makes
                # them vanish at the plane — but they never touched the plate,
                # so a shower looked like it rained straight through. lazer's
                # Catcher explodes a caught banana off the plate IMMEDIATELY
                # (bananas never persist in the pile; no combo/hp change,
                # LargeBonus +50 not modelled — see note above). clear_time is
                # pre-set to the catch time so the group pass below leaves it
                # alone, and anim="banana" routes the burst to the banana
                # sprite. Skin-gated: the certified argon path stays
                # bit-identical (flagged as a known gap for the next argon
                # certification pass).
                cxb, _ = catcher_x_at(self.frames, obj.time_ms)
                self._plate.append([obj.time_ms, (obj.x - cxb) * 0.55, 0.0,
                                    obj.combo_index, obj.hyperdash,
                                    obj.time_ms, "banana"])
            # lazer standardised: 500k·acc·comboRatio + 500k·acc⁵·progress ×mult
            if max_combo_portion > 0 and max_base_total > 0 and cur_max_base > 0:
                sacc = cur_base / cur_max_base
                score = (500_000.0 * sacc * (combo_portion / max_combo_portion)
                         + 500_000.0 * (sacc ** 5) * (cur_max_base / max_base_total)
                         ) * _mod_mult
            else:
                score = 0.0
            caught_acc = c300 + c100 + c50
            total_acc = caught_acc + cmiss + ckatu + ctiny_miss
            acc = (caught_acc / total_acc) if total_acc else 1.0
            self._checkpoints.append(
                _Checkpoint(obj.time_ms, combo, int(round(score)), hp, acc, max_combo,
                            counts=(c300, c100, c50, ctiny_miss, cmiss + ckatu)))

        # Plate clear points: at each combo's LAST object (lazer LastInCombo) the
        # pile Explodes (that object was HIT) or Drops (missed) — Catcher.cs.
        group_last: dict = {}
        for o, c in zip(self._objs, self._caught):
            ci = o.combo_index
            if ci not in group_last or o.time_ms >= group_last[ci][0]:
                group_last[ci] = (o.time_ms, c)
        for rec in self._plate:
            if rec[6] is not None:      # banana: pre-cleared at catch time
                continue
            gt, gc = group_last.get(rec[3], (rec[0], True))
            rec[5] = max(gt, rec[0])                    # clear_time (>= catch time)
            rec[6] = "explode" if gc else "drop"

        # osu!catch legacy counts: 300=caught fruit, 100=caught large droplet,
        # 50=caught tiny droplet, katu=MISSED tiny, miss=missed fruit + large.
        self.final_counts = (c300, c100, c50, ctiny_miss, cmiss + ckatu)
        self.final_accuracy = acc if self._checkpoints else 1.0
        # score curve is already the absolute lazer-standardised ScoreV3 —
        # no post-hoc scale to the replay's mixed-format total.
        self.score_scale = 1.0
        # anchor accuracy to the replay's authoritative final (tiny-droplet
        # positions can't be reproduced bit-exact, so trust the replay's counts)
        m = self.meta
        if m is not None:
            caught = m.count_300 + m.count_100 + m.count_50
            total = caught + m.count_katu + m.count_miss
            if total > 0:
                self.real_accuracy = caught / total
                self.final_counts = (m.count_300, m.count_100, m.count_50,
                                     m.count_katu, m.count_miss)
                self.final_accuracy = self.real_accuracy
                # NOTE: no acc_offset — the geometric sim is now accurate (head-Y
                # slider fix + LegacyLastTickOffset), so the running accuracy comes
                # straight from the sim. final_counts/accuracy above are the
                # replay's authoritative numbers, used only for the results screen.

    def state_at(self, t_ms: int) -> _Checkpoint:
        if not self._checkpoints:
            return _Checkpoint(t_ms, 0, 0, 1.0)
        i = bisect_right(self._cp_times, t_ms) - 1
        if i < 0:
            return _Checkpoint(t_ms, 0, 0, 1.0)
        return self._checkpoints[i]

    # --- geometry -------------------------------------------------------------

    def _sx(self, osu_x: float) -> float:
        return self.x_off + osu_x * self.unit_px

    def _hyper_amount(self, t_ms) -> float:
        """Catcher red-tint strength 0..1 with lazer's 180ms OutQuint fade in at
        hyper start and fade out at hyper end (HYPER_DASH_TRANSITION_DURATION).
        0 outside a hyper window + its 180ms fade-out tail."""
        if not self.cfg.show_hyperdash:
            return 0.0
        D = 180.0
        amt = 0.0
        for a, b in self._hyper_windows:
            if a <= t_ms <= b:
                u = min((t_ms - a) / D, 1.0)
                amt = max(amt, 1.0 - (1.0 - u) ** 5)     # OutQuint fade-IN white->red
            elif b < t_ms <= b + D:
                u = (t_ms - b) / D
                amt = max(amt, (1.0 - u) ** 5)           # fade-OUT red->white
        return amt

    def _fruit_y(self, obj_time: int, t: int) -> float:
        f = (t - (obj_time - self.preempt)) / self.preempt
        return -self.fruit_screen + (self.plane_y + self.fruit_screen) * f

    # --- frame ----------------------------------------------------------------

    def build_scene(self, t_ms: int) -> SceneState:
        s = SceneState(time_ms=t_ms)
        # break check, shared by the bg dim + the letterbox below (PERF hoist)
        in_break = (any(a <= t_ms <= b for a, b in self.bm.breaks)
                    if self.bm.breaks else False)
        # dimmed beatmap background (drawn first, behind everything)
        if self.has_bg:
            # preset bg dim per phase: % dim (higher=darker) -> brightness mult
            first_t = self.bm.objects[0].time_ms if self.bm.objects else 0
            dim_pct = (self.cfg.bg_dim_breaks if in_break
                       else self.cfg.bg_dim_intro if t_ms < first_t
                       else self.cfg.bg_dim_game)
            d = max(0.0, 1.0 - dim_pct / 100.0)
            s.sprites.append(Sprite(self.screen_w / 2, self.screen_h / 2,
                                    self.screen_w, self.screen_h,
                                    texture_key="bg", color=(d, d, d, 1.0)))

        # falling objects. Caught objects vanish at the catch line; MISSED ones
        # keep FALLING THROUGH the catcher for 250 ms while fading out and
        # rotating to 2× their tilt — lazer's DrawableCatchHitObject miss
        # (FadeOut(250).RotateTo(Rotation*2, 250, Easing.Out)). Blinking out at
        # the plate was the single biggest "not osu" tell on drops.
        # PERF: only objects with time_ms in [t-250, t+preempt] can pass the
        # visibility test below — bisect that window out of the (sorted)
        # object list instead of scanning the whole map every frame. The
        # per-object test is kept verbatim as the authoritative filter.
        if self._objs_sorted:
            _lo = bisect_left(self._obj_times, t_ms - 250)
            _hi = bisect_right(self._obj_times, t_ms + self.preempt)
        else:
            _lo, _hi = 0, len(self._objs)
        for obj, caught in zip(self._objs[_lo:_hi], self._caught[_lo:_hi]):
            end = obj.time_ms if caught else obj.time_ms + 250
            if not (obj.time_ms - self.preempt <= t_ms <= end):
                continue
            y = self._fruit_y(obj.time_ms, t_ms)
            sprites = self._object_sprites(obj, self._sx(obj.x), y, t_ms)
            if not caught and t_ms > obj.time_ms:
                mu = (t_ms - obj.time_ms) / 250.0            # 0..1 through the miss
                # Hidden + legacy skin: the fruit already faded to alpha 0 on
                # approach (stable keeps that SAME sprite for the fall-through)
                # — without this, an invisible fruit POPPED back into view as
                # it fell past the catcher. _hd_alpha is 0 past the plane, so
                # missed objects stay hidden. Skinless (argon) path untouched:
                # certified bit-identical.
                hd = (self._hd_alpha(obj.time_ms, t_ms)
                      if (self.hidden and self.skin is not None) else 1.0)
                for sp in sprites:
                    r, g, b, al = sp.color
                    sp.color = (r, g, b, al * max(0.0, 1.0 - mu) * hd)
                    sp.rotation = sp.rotation * (1.0 + mu)   # tilt → 2× over 250ms
            elif self.hidden:
                a = self._hd_alpha(obj.time_ms, t_ms)
                if a < 1.0:
                    for sp in sprites:
                        r, g, b, al = sp.color
                        sp.color = (r, g, b, al * a)
            s.sprites.extend(sprites)

        # catcher (+ dash trail + caught-fruit pile riding on the plate)
        cx, dashing = catcher_x_at(self.frames, t_ms)
        s.catcher_x = float(cx)          # HUD key counter (L/R from x delta)
        s.dashing = bool(dashing)        # HUD key counter (dash key state)
        scx = self._sx(cx)
        # screen-space catcher geometry for the HUD's catcher-tracking combo
        # (SceneState is a snapshot — the HUD never sees the sim itself)
        s.catcher_px = float(scx)
        s.plane_y_px = float(self.plane_y)
        s.pf_unit_px = float(self.unit_px)
        # Red-tint strength 0..1 with lazer's 180ms OutQuint fade in/out, so the
        # catcher/trail ramp white<->red instead of snapping to pure red.
        hyper_amt = self._hyper_amount(t_ms)
        hyper = hyper_amt > 0.0
        # Always build the trail when enabled — _dash_trail keys each ghost off
        # whether the catcher was dashing at that PAST instant, so trailing
        # ghosts persist and fade smoothly after a dash ends (and through the
        # hyperdash after-image window) instead of strobing with the live flag.
        if self.cfg.catcher_dash_trail:
            s.sprites.extend(self._dash_trail(t_ms, hyper_amt))
        s.sprites.extend(self._catcher_sprites(scx, dashing or hyper, hyper_amt, t_ms))
        s.sprites.extend(self._plate_stack(scx, t_ms))
        s.sprites.extend(self._catch_explosions(t_ms))

        # letterbox + dim during breaks (drawn last so bars sit on top)
        if self.cfg.letterbox_breaks and in_break:
            bar = self.screen_h * 0.11
            s.sprites.append(Sprite(self.screen_w / 2, bar / 2, self.screen_w, bar,
                                    texture_key=None, color=(0, 0, 0, 0.92)))
            s.sprites.append(Sprite(self.screen_w / 2, self.screen_h - bar / 2,
                                    self.screen_w, bar, texture_key=None, color=(0, 0, 0, 0.92)))

        # R3D intro splash -- topmost intro element, over the idle scene
        if self.logo_start_ms is not None:
            s.sprites.extend(self._logo_sprites(t_ms))

        cp = self.state_at(t_ms)
        s.combo = cp.combo
        s.score = int(cp.score * self.score_scale)
        s.hp = cp.hp
        s.accuracy = max(0.0, min(1.0, cp.accuracy))
        s.pp = cp.pp
        s.counts = cp.counts
        return s

    def compute_pp_curve(self, osu_path, mods) -> None:
        """Fill each checkpoint's running pp via rosu-pp (no-op if unavailable)."""
        try:
            import rosu_pp_py as rosu
        except Exception:
            return
        cps = self._checkpoints
        if not cps:
            return
        try:
            rbm = rosu.Beatmap(path=str(osu_path))
        except Exception:
            return
        # Catch is frequently played on converted osu!standard maps. Without
        # converting, rosu computes the *osu!standard* pp from the catch hit
        # counts -- roughly 2x wrong. Convert to the Catch ruleset so the pp
        # matches the game. Native-catch maps are left as-is.
        try:
            if rbm.mode != rosu.GameMode.Catch:
                rbm.convert(rosu.GameMode.Catch, int(mods))
        except Exception:
            pass
        n = len(cps)
        step = max(1, n // 80)
        samples = {}
        for i in range(0, n, step):
            cp = cps[i]
            c3, c1, c5, tmiss, miss = cp.counts
            try:
                samples[i] = rosu.Performance(
                    mods=int(mods), passed_objects=i + 1, n300=c3, n100=c1,
                    n50=c5, n_katu=tmiss, misses=miss, combo=cp.max_combo,
                ).calculate(rbm).pp
            except Exception:
                samples[i] = samples.get(i - step, 0.0)
        # final point = the authoritative full-play pp, taken straight from
        # the replay's own counts + max combo (not the geometric sim, which
        # only approximates per-type counts and combo).
        m = self.meta
        if m is not None:
            try:
                samples[n - 1] = rosu.Performance(
                    mods=int(mods), n300=m.count_300, n100=m.count_100,
                    n50=m.count_50, n_katu=m.count_katu, misses=m.count_miss,
                    combo=m.max_combo).calculate(rbm).pp
            except Exception:
                pass
        keys = sorted(samples)
        for i, cp in enumerate(cps):  # linear-interp pp between sampled checkpoints
            if i in samples:
                cp.pp = samples[i]
                continue
            lo = max(k for k in keys if k <= i)
            hi = min((k for k in keys if k >= i), default=lo)
            cp.pp = samples[lo] if hi == lo else (
                samples[lo] + (samples[hi] - samples[lo]) * (i - lo) / (hi - lo))

    # --- sprite emission ------------------------------------------------------

    def _hd_alpha(self, obj_time: int, t_ms: int) -> float:
        """osu!catch Hidden fade (CatchModHidden). Time-based on TimePreempt:
        fade starts at StartTime - 0.6*preempt and completes (invisible) at
        StartTime - 0.44*preempt — so a fruit is fully visible for the first
        ~40% of its fall, fades over 16% of the preempt, then is invisible for
        the last 44% before reaching the catcher."""
        p = self.preempt
        if p <= 0:
            return 1.0
        t_rem = obj_time - t_ms          # ms until the catch line
        a = (t_rem - 0.44 * p) / (0.16 * p)
        return 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)

    # lazer's default combo colours (SkinConfiguration.DefaultComboColours) —
    # the no-skin fallback palette, so skinless renders show lazer's orange /
    # green / blue / red fruits instead of a neon hue wheel.
    _LAZER_COMBO = ((1.0, 0.753, 0.0), (0.0, 0.792, 0.0),
                    (0.070, 0.486, 1.0), (0.949, 0.094, 0.224))

    def _combo_tint(self, combo_index: int) -> tuple[float, float, float]:
        # Precedence — stable AND lazer at DEFAULT settings ("beatmap skin/
        # colours" enabled in both games): the MAP's own [Colours] wins when
        # the .osu ships one; else the user skin's Combo1..N; else the
        # default-skin palette; else lazer's default combo colours.
        # The old order let a user skin's ini beat the map, so an all-red
        # skin (e.g. 3e9449 "red theme") painted EVERY fruit red — users read
        # that as the hyperdash cue leaking onto normal fruits ("impossible-
        # looking" renders). Skinless path order is unchanged (map → lazer
        # palette), keeping the certified argon output bit-identical.
        sk = self.skin
        cc = self.bm.combo_colors
        if cc:
            r, g, b = cc[combo_index % len(cc)]
            return (r / 255.0, g / 255.0, b / 255.0)
        if sk is not None:
            return sk.combo_color(combo_index)
        return self._LAZER_COMBO[combo_index % len(self._LAZER_COMBO)]

    def _object_sprites(self, obj, x, y, t_ms) -> list[Sprite]:
        # A custom uploaded skin's fruit sprites WIN when it ships them;
        # _skinned_object falls back to _argon_object per object-kind for any
        # element the skin lacks. The skinless base is osu!lazer's ARGON look:
        # glowing, combo-coloured wavy rings (stacked additive CircularBlobs)
        # over a white centre pip — NOT pulp clusters or flat discs; hyper
        # objects add Argon's red blob. (Argon is the DEFAULT only.)
        if self.skin is not None:
            return self._skinned_object(obj, x, y, t_ms)
        return self._argon_object(obj, x, y, t_ms)

    def _argon_variant(self, obj) -> int:
        """Pick one of the baked seed variants so successive objects vary like
        lazer's per-object random blob seeds (deterministic per object)."""
        return self._obj_index.get(id(obj), 0) % ARGON_VARIANTS

    def _banana_flare_alpha(self, obj_time: int, t_ms: int) -> float:
        """ArgonBananaPiece lens-flare fade: fully visible for the first 30% of
        the approach, then fades to 0 by 80% (OutQuint), invisible after."""
        p = self.preempt
        if p <= 0:
            return 0.0
        frac = (t_ms - (obj_time - p)) / p     # 0 at spawn .. 1 at the catch line
        if frac <= 0.3:
            return 1.0
        if frac >= 0.8:
            return 0.0
        u = (frac - 0.3) / 0.5
        return (1.0 - u) ** 5                   # ValueAt(1->0, OutQuint) == (1-u)^5

    def _argon_object(self, obj, x, y, t_ms) -> list[Sprite]:
        fs = self.fruit_screen
        if obj.kind is ObjType.TINY_DROPLET:
            size = fs * 0.34
        elif obj.kind is ObjType.DROPLET:
            size = fs * 0.58
        else:
            size = fs * 1.05
        d = size * ARGON_CANVAS        # blob canvas spans ARGON_CANVAS object-boxes
        v = self._argon_variant(obj)
        out: list[Sprite] = []

        # droplets / tiny droplets (ArgonDropletPiece): white pip + a smaller
        # glowing blob (additive, combo-tinted); hyper adds the red blob.
        if obj.kind in (ObjType.DROPLET, ObjType.TINY_DROPLET):
            tint = self._combo_tint(obj.combo_index)
            out.append(Sprite(x, y, d, d, texture_key="argon_pip", color=(1, 1, 1, 1)))
            out.append(Sprite(x, y, d, d, texture_key=f"argon_droplet_{v}",
                              color=(*tint, 1.0), additive=True))
            if obj.hyperdash and self.cfg.show_hyperdash:
                out.append(Sprite(x, y, d, d, texture_key=f"argon_drophyper_{v}",
                                  color=(1.0, 0.0, 0.0, 1.0), additive=True))
            return out

        # bananas (ArgonBananaPiece : ArgonFruitPiece): the fruit blob stack in
        # the banana colour + white pip, plus a horizontal lens-flare overlay
        # that fades out over the approach. Gentle deterministic tumble.
        if obj.kind is ObjType.BANANA:
            tint = (_hue((t_ms * 0.0009 + obj.x * 0.01) % 1.0)
                    if self.cfg.banana_rainbow else (1.0, 0.83, 0.15))
            seed = (int(obj.time_ms) * 2654435761) ^ (int(obj.x * 53) & 0xFFFF)
            spin_dir = 1.0 if (seed >> 4) & 1 else -1.0
            rot = ((seed % 628) / 100.0
                   + (t_ms - obj.time_ms + self.preempt) * 0.0008 * spin_dir)
            out.append(Sprite(x, y, d, d, texture_key="argon_pip",
                              color=(1, 1, 1, 1), rotation=rot))
            out.append(Sprite(x, y, d, d, texture_key=f"argon_fruit_{v}",
                              color=(*tint, 1.0), rotation=rot, additive=True))
            fa = self._banana_flare_alpha(obj.time_ms, t_ms)
            if fa > 0.0:
                out.append(Sprite(x, y, size * 2.2, size * 1.1,
                                  texture_key="argon_banana_flare",
                                  color=(1, 1, 1, fa), additive=True))
            return out

        # fruit (ArgonFruitPiece): white pip UNDER a stack of 3 additive,
        # combo-tinted wavy blobs; hyper adds Argon's red blob (shares seed).
        # lazer's small deterministic ±20° tilt (DrawableFruit ScalingContainer).
        tint = self._combo_tint(obj.combo_index)   # hyper keeps its combo colour
        rot = ((int(obj.time_ms) % 1000) / 1000.0 - 0.5) * 0.698   # ±20°
        hyper = obj.hyperdash and self.cfg.show_hyperdash
        out.append(Sprite(x, y, d, d, texture_key="argon_pip",
                          color=(1, 1, 1, 1), rotation=rot))
        out.append(Sprite(x, y, d, d, texture_key=f"argon_fruit_{v}",
                          color=(*tint, 1.0), rotation=rot, additive=True))
        if hyper:
            out.append(Sprite(x, y, d, d, texture_key=f"argon_hyper_{v}",
                              color=(1.0, 0.0, 0.0, 1.0), rotation=rot, additive=True))
        return out

    # Skinned-object quad multipliers (× fruit_screen). SIZE PARITY RULE for
    # FRUIT/BANANA: a legacy sprite FILLS its quad while the Argon pieces only
    # light a fraction of theirs — those two match the ARGON path's measured
    # VISIBLE diameters. Measured at CS3.3/720p (fruit_screen=118.9px, typical
    # legacy art fills ~0.94 fruit / ~0.77 droplet of its canvas):
    #   argon fruit ≈114px visible → skin quad 1.05 → ≈112-117px  (match)
    #   argon banana ring ≈115px  → skin quad 1.05 → ≈117px       (match)
    # DROPLETS are NOT argon-calibrated: lazer-argon's droplet blob is its own
    # quarter-fruit look, and matching it (quad 0.32/0.19) rendered legacy
    # droplets ~40% under real osu!. Legacy droplet size derives from lazer's
    # LEGACY path instead:
    #   droplet = 0.5 × the fruit scale  (half the 128 osu-px fruit diameter)
    #           × 0.8 legacy sprite draw scale (LegacyDropletPiece.Scale=0.8f)
    #           ⇒ true VISIBLE size 0.5·0.8 = 0.40 × fruit_screen
    #   quad   = visible / art fill = 0.40 / 0.77          = 0.52  (was 0.32)
    #   tiny   = 0.8 × droplet (stable's ratio) = 0.52·0.8 = 0.416 (was 0.19)
    # Cross-check vs the classic default skin under lazer (fruit-drop.png,
    # 128px canvas, art fill 0.555, drawn at 0.8·128·scale): visible = 0.444
    # × fruit_screen — our 0.52·0.77 = 0.40 lands within 10% of that truth,
    # between the old 0.55 (slightly big) and the argon-matched 0.32 (small).
    # Everything scales with CS via fruit_screen. Hyperfruit is NOT bigger —
    # see the hyper echo in _skinned_object (was ×1.32, an invented bump).
    _SKIN_FRUIT = 1.05
    _SKIN_DROPLET = 0.52
    _SKIN_TINY = 0.416
    _SKIN_BANANA = 1.05

    def _skinned_object(self, obj, x, y, t_ms) -> list[Sprite]:
        # Per-element skin honoring: use the skin's sprite for this object kind
        # when it ships one, else fall back to the Argon look for that kind.
        sk = self.skin
        if obj.kind in (ObjType.DROPLET, ObjType.TINY_DROPLET):
            if not sk.has("fruit-drop"):
                return self._argon_object(obj, x, y, t_ms)
            size = self.fruit_screen * (self._SKIN_DROPLET
                                        if obj.kind is ObjType.DROPLET
                                        else self._SKIN_TINY)
            # Droplets are the ONE object kind that SPINS while falling
            # (fruits hold a frozen tilt — see the FRUIT branch below).
            # lazer DrawableDroplet.Update:
            #   Rotation = lerp(start, start + 720°,
            #                   (now - spawn) / (TimePreempt + 2000))
            #   start    = RandomSingle(1) * 20°   (per-object seed)
            # DrawableTinyDroplet inherits the same spin. TimePreempt/2000
            # tick on the beatmap clock, so map them to the replay's real
            # timeline like self.preempt already is (2000 → 2000/rate).
            rot = 0.0
            if self.cfg.fruit_rotation:
                start = _obj_rand01(obj.time_ms, obj.x, 1) * 0.349   # 0..20°
                dur = self.preempt + 2000.0 / self.bm.rate           # real ms
                rot = start + 12.566 * (t_ms - (obj.time_ms - self.preempt)) / dur
            return self._base_overlay("fruit-drop", x, y, size,
                                      self._combo_tint(obj.combo_index), rot)
        if obj.kind is ObjType.BANANA:
            if not sk.has("fruit-bananas"):
                return self._argon_object(obj, x, y, t_ms)
            size = self.fruit_screen * self._SKIN_BANANA
            if self.cfg.banana_rainbow:
                tint = _hue((t_ms * 0.0009 + obj.x * 0.01) % 1.0)
            else:
                tint = (1.0, 0.85, 0.15)
            return self._base_overlay("fruit-bananas", x, y, size, tint)
        # FRUIT
        if not sk.has(sk.fruit_key(obj.combo_index)):
            return self._argon_object(obj, x, y, t_ms)
        hyper = obj.hyperdash and self.cfg.show_hyperdash
        # osu!lazer: a HYPERFRUIT is the SAME size as a normal fruit — the
        # hyper cue is a red additive echo of the same sprite at 1.2× BEHIND
        # it (LegacyCatchHitObjectPiece.hyperSprite: Scale 1.2, Alpha 0.7,
        # Colour red, Depth 1) — NOT a bigger fruit and NOT a red-tinted core.
        size = self.fruit_screen * self._SKIN_FRUIT
        tint = self._combo_tint(obj.combo_index)
        # FROZEN random tilt — fruits do NOT spin while falling. lazer
        # DrawableFruit.UpdateInitialTransforms: ScalingContainer.Rotation =
        # (RandomSingle(1) - 0.5) * 40 — ONE random angle in ±20°, rolled at
        # spawn from the object's seed and held all the way down. Only
        # droplets rotate (branch above). The old code spun fruits
        # continuously (rate × time) — an invented behaviour.
        rot = 0.0
        if self.cfg.fruit_rotation:
            rot = (_obj_rand01(obj.time_ms, obj.x, 1) - 0.5) * 0.698   # ±20°
        out: list[Sprite] = []
        if hyper:
            # Straight-alpha (not additive): our GL batch draws ALL additive
            # sprites in a second pass ON TOP, which would wash the fruit core
            # red — painter's order in the normal pass keeps the echo behind
            # the opaque fruit exactly like lazer's Depth 1. Echo colour =
            # skin.ini HyperDashFruit → HyperDash → red (LegacySkin's
            # CatchSkinColour.HyperDashFruit lookup chain).
            out.append(Sprite(x, y, size * 1.2, size * 1.2,
                              texture_key=sk.fruit_key(obj.combo_index),
                              color=(*self.hyper_fruit_rgb, 0.7), rotation=rot))
        out.extend(self._base_overlay(sk.fruit_key(obj.combo_index), x, y, size, tint, rot))
        return out

    def _base_overlay(self, base_key, x, y, size, tint, rot=0.0) -> list[Sprite]:
        out = [Sprite(x, y, size, size, texture_key=base_key, color=(*tint, 1.0), rotation=rot)]
        ov = f"{base_key}-overlay"
        if self.skin.has(ov):
            out.append(Sprite(x, y, size, size, texture_key=ov, color=(1, 1, 1, 1), rotation=rot))
        return out

    def _catcher_ghost(self, scx, rgb, alpha, scale=1.0, dy=0.0) -> list[Sprite]:
        """One ADDITIVE afterimage of the FULL catcher body (skin sprite, or the
        Argon bar+bumpers) at screen-x scx — the unit lazer's CatcherTrail draws
        (CatcherTrail.body = SkinnableCatcher, Blending = Additive)."""
        ck = getattr(self.skin, "catcher_key", None) if self.skin is not None else None
        if ck is not None and self.skin.has(ck):
            hb = self.catcher_w * self.skin.catcher_aspect
            return [Sprite(scx, self.plane_y + hb * 0.46 + dy,
                           self.catcher_w * scale, hb * scale,
                           texture_key=ck,
                           color=(*rgb, alpha), additive=True)]
        from .lazer_skin import argon_catcher_metrics
        g = argon_catcher_metrics(self.catcher_w, self.unit_px, self.plane_y)
        cy = g["cy"] + dy
        bx = (g["bar_w"] * 0.5 + g["bump_w"] * 0.5) * scale
        return [
            Sprite(scx, cy, g["bar_w"] * scale, g["bar_h"] * scale,
                   texture_key="argon_bar_cap", color=(*rgb, alpha), additive=True),
            Sprite(scx - bx, cy, g["bump_w"] * scale, g["bump_h"] * scale,
                   texture_key="argon_bar_cap", color=(*rgb, alpha), additive=True),
            Sprite(scx + bx, cy, g["bump_w"] * scale, g["bump_h"] * scale,
                   texture_key="argon_bar_cap", color=(*rgb, alpha), additive=True),
        ]

    def _dash_trail(self, t_ms, hyper_amt) -> list[Sprite]:
        """osu!lazer CatcherTrailDisplay. A dense ADDITIVE stack of full-catcher
        afterimages over the last 800 ms while dashing/hyperdashing — one every
        16 ms (CatcherArea trail_generation_interval), alpha 0.4·(1-age/800)^5
        (CatcherTrail FadeTo(0.4)→FadeOut(800, OutQuint)), lerped white->red by
        hyper_amt — PLUS the hyperdash after-image burst at each hyper onset
        (grows 0.95→1.2×, drifts up 10 px, fades over 1200 ms). Replaces the old
        3-ghost/80 ms non-additive trail that was ~50× too sparse to see."""
        out: list[Sprite] = []
        # white → skin.ini HyperDash colour by hyper_amt (lazer's hyper dash
        # trail colour; red default = the old hardcoded (1, 1-amt, 1-amt)).
        hr, hg, hb = self.hyper_rgb
        trail_rgb = (1.0 + (hr - 1.0) * hyper_amt,
                     1.0 + (hg - 1.0) * hyper_amt,
                     1.0 + (hb - 1.0) * hyper_amt)
        for age in range(16, 801, 16):
            alpha = 0.4 * (1.0 - age / 800.0) ** 5
            if alpha < 0.004:
                break                       # monotonic → no later age is brighter
            px, was_dashing = catcher_x_at(self.frames, t_ms - age)
            # Key each ghost off whether the catcher was dashing at THAT past
            # instant — NOT the live frame's dash bit. The replay's dash flag
            # flickers on/off frame-to-frame near dash edges; gating the whole
            # trail on the live flag made all 30 ghosts strobe together (the
            # "jittery" bug). Per-instant keying = ghosts fade out smoothly.
            if was_dashing:
                out.extend(self._catcher_ghost(self._sx(px), trail_rgb, alpha))
        # Hyperdash after-image: one red ghost per hyper onset (Easing.In pop).
        for h_start, _h_end in self._hyper_windows:
            age = t_ms - h_start
            if 0.0 <= age <= 1200.0:
                u = age / 1200.0
                e = u * u                   # Easing.In ≈ quadratic
                alpha = 1.0 - u
                if alpha < 0.004:
                    continue
                px, _ = catcher_x_at(self.frames, h_start)
                out.extend(self._catcher_ghost(
                    self._sx(px), self.hyper_after_rgb, alpha,
                    scale=0.95 + 0.25 * e, dy=-10.0 * self.unit_px * e))
        return out

    def _procedural_object(self, obj, x, y) -> Sprite:
        if obj.kind is ObjType.DROPLET:
            return Sprite(x, y, self.fruit_screen * 0.4, self.fruit_screen * 0.4, texture_key="droplet")
        if obj.kind is ObjType.TINY_DROPLET:
            return Sprite(x, y, self.fruit_screen * 0.22, self.fruit_screen * 0.22, texture_key="droplet")
        if obj.kind is ObjType.BANANA:
            return Sprite(x, y, self.fruit_screen * 1.1, self.fruit_screen * 1.1, texture_key="banana")
        size = self.fruit_screen * (1.3 if obj.hyperdash else 1.0)
        return Sprite(x, y, size, size, texture_key=FRUIT_TEX[obj.combo_index % 4])

    def _catcher_sprites(self, x, dashing, hyper_amt=0.0, t_ms=None) -> list[Sprite]:
        # A custom skin's catcher takes priority — same layout (CS/mod-driven
        # catcher_w), the skin supplies the sprite (fruit-catcher-idle, or
        # fruit-ryuuta for old-style skins — skin.catcher_key). The procedural
        # Argon catcher is only the base/fallback when the skin ships no catcher.
        ck = getattr(self.skin, "catcher_key", None) if self.skin is not None else None
        if (ck is not None and self.skin.has(ck)
                and not self.force_argon_catcher):
            # lazer tints the catcher ONLY when hyperdashing, in the skin's
            # [CatchTheBeat] HyperDash colour (Catcher falls back to
            # DEFAULT_HYPER_DASH_COLOUR = Color4.Red); plain dashing leaves the
            # body white — the trail is the dash cue, not a body tint.
            hr, hg, hb = self.hyper_rgb
            tint = (1.0 + (hr - 1.0) * hyper_amt,
                    1.0 + (hg - 1.0) * hyper_amt,
                    1.0 + (hb - 1.0) * hyper_amt, 1.0)
            w = self.catcher_w
            h = w * self.skin.catcher_aspect
            return [Sprite(x, self.plane_y + h * 0.46, w, h,
                           texture_key=ck, color=tint)]
        # osu!lazer ArgonCatcher: a white rounded catch bar (0.8 of the catcher
        # width) + a bumper at each end of the catch range + faint side lines
        # out to the screen edges. Footprint/placement unchanged (full width =
        # catcher_w, bar top on plane_y). Hyperdash turns it red + glowing.
        from .lazer_skin import argon_catcher_metrics
        g = argon_catcher_metrics(self.catcher_w, self.unit_px, self.plane_y)
        cy = g["cy"]
        # lazer's ArgonCatcher is white; hyperdash turns it full red (no glow —
        # the red after-image trail is the hyper cue, drawn in _dash_trail).
        col = (1.0, 1.0 - hyper_amt, 1.0 - hyper_amt, 1.0)
        out: list[Sprite] = []
        # main catch bar
        out.append(Sprite(x, cy, g["bar_w"], g["bar_h"],
                          texture_key="argon_bar_cap", color=col))
        # bumpers at the ends of the catch range (flanking the 0.8 bar)
        bx = g["bar_w"] * 0.5 + g["bump_w"] * 0.5
        out.append(Sprite(x - bx, cy, g["bump_w"], g["bump_h"],
                          texture_key="argon_bar_cap", color=col))
        out.append(Sprite(x + bx, cy, g["bump_w"], g["bump_h"],
                          texture_key="argon_bar_cap", color=col))
        # faint long lines out to the screen edges (alpha 0.25)
        left_outer = x - g["full_w"] * 0.5
        right_outer = x + g["full_w"] * 0.5
        line_c = (1.0, 1.0 - hyper_amt, 1.0 - hyper_amt, 0.25)
        if left_outer > 1.0:
            out.append(Sprite(left_outer * 0.5, cy, left_outer, g["line_h"],
                              texture_key=None, color=line_c))
        if right_outer < self.screen_w - 1.0:
            rw = self.screen_w - right_outer
            out.append(Sprite(right_outer + rw * 0.5, cy, rw, g["line_h"],
                              texture_key=None, color=line_c))
        return out

    def _plate_stack(self, scx, t_ms) -> list[Sprite]:
        """Caught fruit riding the catcher plate — lazer-faithful (Catcher.cs).
        Each fruit lands at the x it was caught (computePositionInStack offset,
        precomputed in the sim), rides the plate, PERSISTS through the whole
        combo, then at the combo's last object either EXPLODES outward (that
        object was hit: MoveX +offset·6 linear over 1s, Y bounce -50 OutSine /
        +100 InSine, FadeOut 750) or DROPS off (missed: +75 InSine, FadeOut 750).
        Skin fruit sprite at 0.5× object; Argon fallback when skinless."""
        import math
        out: list[Sprite] = []
        # HIDDEN + legacy skin: stable re-parents the SAME approach sprite
        # onto the plate — under HD it reached the plane at alpha 0, so the
        # caught pile is INVISIBLE in the real game ("notes don't show up on
        # the platter when HD is on"). Skinless path deliberately untouched:
        # lazer's CaughtObject.RestoreState does not copy alpha, so the
        # (certified bit-identical) argon path keeping its pile matches lazer.
        if self.hidden and self.skin is not None:
            return out
        up = self.unit_px
        # Caught fruit pile ABOVE the catcher — lazer anchors the caught container
        # to the catcher's TopCentre, so the pile rests ON the catch line and
        # builds UPWARD (in front of the bar). Was `+0.1` = BELOW the line, which
        # sank the base fruits behind the bar (Red's flag).
        if self.skin is not None:
            # Legacy catcher art carries its visible dish AT/just below the
            # catch plane (the sprite hangs below plane_y). The 0.34 lift —
            # tuned for the argon BAR — parked fruit bottoms ~a half-fruit
            # above the dish ("caught fruits float above the platter"). 0.10
            # rests them on/in the dish; the argon branch keeps the certified
            # 0.34 bit-identical.
            base_y = self.plane_y - self.fruit_screen * 0.10
        else:
            base_y = self.plane_y - self.fruit_screen * 0.34
        size = self.fruit_screen * 0.5
        for ct, ox, oy, ci, hy, clear_t, anim in self._plate:
            if clear_t is None or t_ms < ct:
                continue
            v = int(ct // 7) % ARGON_VARIANTS
            if t_ms < clear_t:
                # ON PLATE — rides the catcher at its stacked position.
                out.extend(self._plate_fruit(scx + ox * up, base_y + oy * up,
                                             size, ci, v, 1.0))
                continue
            age = t_ms - clear_t
            if age > 750.0:                                  # gone (FadeOut 750)
                continue
            alpha = 1.0 - age / 750.0
            # World-space: leaves the catcher from its last on-plate position.
            x0 = self._sx(catcher_x_at(self.frames, int(clear_t))[0]) + ox * up
            y0 = base_y + oy * up
            if anim == "drop":                               # miss: fall + fade
                u = min(age, 750.0) / 750.0
                sx = x0
                sy = y0 + 75.0 * up * (1.0 - math.cos(u * math.pi / 2.0))   # InSine
            else:                                            # hit: burst outward
                sx = x0 + (ox * 6.0) * up * (min(age, 1000.0) / 1000.0)     # linear
                if age <= 250.0:
                    sy = y0 - 50.0 * up * math.sin((age / 250.0) * math.pi / 2.0)   # OutSine
                else:
                    u = min(1.0, (age - 250.0) / 500.0)
                    sy = (y0 - 50.0 * up) + 100.0 * up * (1.0 - math.cos(u * math.pi / 2.0))  # InSine
            out.extend(self._plate_fruit(sx, sy, size, ci, v, alpha,
                                         banana=(anim == "banana")))
        return out

    def _plate_fruit(self, x, y, size, ci, v, alpha,
                     banana: bool = False) -> list[Sprite]:
        """One caught fruit — the skin's fruit sprite (base+overlay) if present,
        else the Argon blob+pip — drawn at `alpha`. `banana` swaps in the
        banana sprite/tint for a caught banana's immediate burst (bananas only
        enter the plate list on skinned renders — see _simulate)."""
        sk = self.skin
        if banana:
            tint = (1.0, 0.85, 0.15)
            if sk is not None and sk.has("fruit-bananas"):
                sprites = self._base_overlay("fruit-bananas", x, y, size, tint)
                for sp in sprites:
                    r, g, b, al = sp.color
                    sp.color = (r, g, b, al * alpha)
                return sprites
            d = size * ARGON_CANVAS   # skin ships no banana art -> argon banana
            return [Sprite(x, y, d, d, texture_key="argon_pip",
                           color=(1, 1, 1, alpha)),
                    Sprite(x, y, d, d, texture_key=f"argon_fruit_{v}",
                           color=(*tint, alpha), additive=True)]
        tint = self._combo_tint(ci)
        if sk is not None and sk.has(sk.fruit_key(ci)):
            sprites = self._base_overlay(sk.fruit_key(ci), x, y, size, tint)
            for sp in sprites:
                r, g, b, al = sp.color
                sp.color = (r, g, b, al * alpha)
            return sprites
        d = size * ARGON_CANVAS
        return [Sprite(x, y, d, d, texture_key="argon_pip", color=(1, 1, 1, alpha)),
                Sprite(x, y, d, d, texture_key=f"argon_fruit_{v}",
                       color=(*tint, alpha), additive=True)]

    def _catch_explosions(self, t_ms) -> list[Sprite]:
        """osu!lazer ArgonHitExplosion: every caught FRUIT fires a tall,
        combo-coloured vertical glow that scales up to (1.1, 20*s) over 200ms
        (OutQuint) then retracts to (1.1, 1) over 600ms (In), plus a large faint
        glow (radius 50, colour 20% toward white). The whole thing fades out
        over 400ms. s = clamp(combo/200, 0.35, 1.125). Droplets don't explode."""
        # SKIN HONORING: this is an ARGON element. A legacy skin (one that ships
        # its own fruit / catcher art) has no such effect — stable just stacks the
        # caught fruit on the plate — so firing it over a custom skin paints a
        # glow the skin doesn't have. Skip it entirely for skinned renders;
        # skinless stays fully Argon.
        sk = self.skin
        if sk is not None and (sk.has(sk.fruit_key(0))
                               or getattr(sk, "catcher_key", None) is not None):
            return []
        cts = getattr(self, "_catch_times", None)
        if cts is None:
            cts = self._catch_times = [c[0] for c in self._catches]
        import bisect
        lo = bisect.bisect_left(cts, t_ms - 400)
        hi = bisect.bisect_right(cts, t_ms)
        out: list[Sprite] = []
        base = self.fruit_screen * (20.0 / 128.0)     # explosion Size 20 vs OBJECT 128
        for i in range(lo, hi):
            ct, cx, ci, hy, cmb = self._catches[i]
            age = t_ms - ct
            if age < 0:
                continue
            fade = 1.0 - age / 400.0
            if fade <= 0.0:
                continue
            sx = self._sx(cx)
            tint = (1.0, 0.28, 0.28) if hy else self._combo_tint(ci)
            # tall glow: height scale 1 -> 20*s (OutQuint 200ms) -> 1 (In 600ms)
            s = min(max(cmb / 200.0, 0.35), 1.125)
            if age <= 200.0:
                u = age / 200.0
                hf = 1.0 + (20.0 * s - 1.0) * (1.0 - (1.0 - u) ** 5)
            else:
                u = min(1.0, (age - 200.0) / 600.0)
                hf = 20.0 * s + (1.0 - 20.0 * s) * (u * u)
            beam_w = base * 1.4
            beam_h = max(base, base * hf)
            # anchored at the catch plane, growing upward
            out.append(Sprite(sx, self.plane_y - beam_h * 0.5, beam_w, beam_h,
                              texture_key="catch_beam",
                              color=(tint[0], tint[1], tint[2], 0.55 * fade),
                              additive=True))
            # large faint glow, colour interpolated 20% toward white
            gtint = tuple(c + (1.0 - c) * 0.2 for c in tint)
            gsize = base * 5.0
            out.append(Sprite(sx, self.plane_y, gsize, gsize, texture_key="catch_glow",
                              color=(gtint[0], gtint[1], gtint[2], 0.26 * fade),
                              additive=True))
        return out

    def _logo_sprites(self, t_ms: int) -> list[Sprite]:
        """The R3D 'R' tile intro splash (show_logo), fading out exactly as the
        first fruit begins its approach -- ported from the std renderer's
        _draw_logo so the splash is identical across modes."""
        from .effects import logo_alpha, logo_scale, LOGO_UI_SIZE
        la = logo_alpha(t_ms, self.logo_start_ms, self.first_spawn_ms)
        if la is None:
            return []
        k_ui = self.screen_h / 1080.0
        d = LOGO_UI_SIZE * k_ui * logo_scale(t_ms, self.logo_start_ms)
        cx = self.screen_w / 2.0
        cy = self.screen_h * 0.44
        return [
            Sprite(cx, cy, d * 1.9, d * 1.9, texture_key="catch_glow",
                   color=(0.95, 0.28, 0.30, 0.45 * la), additive=True),
            Sprite(cx, cy, d, d, texture_key="logo_tile",
                   color=(1.0, 1.0, 1.0, la)),
        ]
