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

from osu_catch_renderer.skin.assets import ARGON_CANVAS, ARGON_VARIANTS
from osu_catch_renderer.render.dim import build_dim_envelope
from osu_catch_renderer.beatmap.models import (
    CatchBeatmap,
    CatchFrame,
    ObjType,
    RenderConfig,
    SceneState,
    Sprite,
    ar_to_preempt_ms,
    cs_to_catcher_half_width,
)
from osu_catch_renderer.beatmap.replay import catcher_x_at

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
        self.kiai_ranges = getattr(beatmap.timing, "kiai", []) if beatmap.timing is not None else []
        self.frames = frames
        # catcher facing timeline (lazer VisualDirection) — built lazily from
        # self.frames on first _facing_at call (frames can still be shifted by
        # _calibrate_offset during setup, which invalidates this).
        self._facing_changes: list[tuple[int, float]] | None = None
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
        # Approach window in MAP time — the axis the whole sim runs on (replay
        # frames and object times both live in beatmap ms; DT/HT only speed up
        # PLAYBACK, which render.py handles via map_step = frame_ms·rate).
        # AR preempt is defined in gameplay-clock ms and the gameplay clock IS
        # the map timeline, so the window is the full ar_to_preempt_ms — the
        # old ÷rate double-applied the mod: DT fruits spawned 1.5× too late
        # (approach 2.25× too fast in real time) and HT fruits lingered 1.33×
        # too long ("why is it like AR 3" — the on-screen density equals the
        # map-time window, which must equal the game's preempt exactly).
        # Verified against replay ground truth: DT (Circulation, 194) and HT
        # (Dance of The Violins, 266) both judge frame-exact on this axis.
        self.preempt = ar_to_preempt_ms(beatmap.ar)
        # Hidden (HD, mod bit 8): fruits fade out as they near the catcher.
        self.hidden = bool((getattr(meta, "mods", 0) or 0) & 8)
        self.half = cs_to_catcher_half_width(beatmap.cs)
        # R3D intro splash (show_logo): render.py sets logo_start_ms when the
        # flag is on; the splash fades out exactly as the first fruit begins
        # its approach (first object time - preempt), matching the std splash.
        self.logo_start_ms: float | None = None
        self.first_spawn_ms = (min((o.time_ms for o in self._objs), default=0)
                               - self.preempt)
        # Background dim envelope (std's DimEnvelope, ported in dim.py): the
        # dim GLIDES intro→game as the first approach begins, brightens into
        # breaks and re-dims at the resume anchor — smoothstep over the same
        # 900 ms std uses, replacing the old per-phase SNAP at the first note
        # / break edges. Built over the FULL map (bm.objects): a death only
        # ends the render early, it doesn't move the map's phase boundaries.
        _starts = [o.time_ms for o in beatmap.objects]
        self._dim_env = build_dim_envelope(
            cfg.bg_dim_intro / 100.0, cfg.bg_dim_game / 100.0,
            cfg.bg_dim_breaks / 100.0, _starts, self.preempt, beatmap.breaks)
        # Letterbox weight 0..1 with the SAME glides (a 0/0/1 envelope): the
        # break bars fade in/out in lockstep with the dim instead of snapping
        # while the background glides (--letterbox-breaks composition).
        self._break_env = build_dim_envelope(
            0.0, 0.0, 1.0, _starts, self.preempt, beatmap.breaks)

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
        # legacy hit-lighting events (skinned renders): one per CAUGHT
        # palpable object incl. bananas — (time, plate_offset_osu, kind,
        # combo_index, combo_at_judgement). See _legacy_hit_lighting.
        self._light_events: list = []
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
            self._facing_changes = None   # facing timeline shifts with frames
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
                    return 0
                idxs = [i for i, o in enumerate(objs) if o.kind is kind]
                if not idxs:
                    return 0
                diff = target_caught - sum(self._caught[i] for i in idxs)
                if diff > 0:   # need more caught: flip closest missed (smallest +margin)
                    for i in sorted((i for i in idxs if not self._caught[i]),
                                    key=lambda i: margin[i])[:diff]:
                        self._caught[i] = True
                elif diff < 0:  # need fewer: flip most-marginal caught (margin nearest 0)
                    for i in sorted((i for i in idxs if self._caught[i]),
                                    key=lambda i: margin[i])[diff:]:
                        self._caught[i] = False
                return abs(diff)
            _flips = (_reconcile(ObjType.FRUIT, m.count_300)
                      + _reconcile(ObjType.DROPLET, m.count_100)
                      + _reconcile(ObjType.TINY_DROPLET, m.count_50))
            # Guard: reconcile is meant for a HANDFUL of RNG-boundary objects
            # (tiny-droplet offsets we can't reproduce bit-exact). If geometry
            # disagrees with the .osr on a large fraction of objects, the replay
            # does NOT match the score — a truncated/desynced/corrupt .osr from
            # the API (e.g. a near-empty replay whose static catcher geometrically
            # catches ~nothing). Forcing the counts would fabricate the render
            # (a motionless catcher "catching" the whole map). Fail loudly so the
            # pipeline opens a bug report instead of shipping a fake video.
            _n = len(objs)
            if _n and _flips >= 50 and _flips > 0.20 * _n:
                raise RuntimeError(
                    "catch replay desynced from score: geometry and the .osr "
                    f"disagree on {_flips}/{_n} objects ({_flips / _n:.0%}) — "
                    "truncated or corrupt replay; refusing to render a "
                    "fabricated result")
            # --- combo-aware reconcile (honesty pass, std-renderer pattern):
            # counts now match the header, but the END-OF-MAP combo the HUD
            # shows is the run AFTER THE LAST MISS — with a wrongly-PLACED
            # miss it is wrong even with perfect counts. If our miss placement
            # cannot reproduce the header's max_combo, move misses between
            # same-kind objects (cheapest geometric margins first) until it
            # does. Runs BEFORE pass 2, so score/hp/plate/animation stay one
            # coherent stream — no display-only patching.
            self._reconcile_combo_runs(objs, margin, m)

        # --- pass 2: lazer-standardised ScoreV3 from reconciled catches ------
        # EXACT CatchScoreProcessor model (ppy/osu master 2026-07,
        # osu.Game.Rulesets.Catch/Scoring/CatchScoreProcessor.cs):
        #   fruitTinyScale = maxTiny / (maxTiny + maxFruit)  — large droplets
        #     are *purposefully* not counted (CatchScoreProcessor.Reset)
        #   comboPortion    = 1e6 − 400k·fruitTinyScale
        #   dropletsPortion = 400k·fruitTinyScale
        #   total = comboPortion·comboProgress + dropletsPortion·dropletsHit
        #           + bonusPortion, then × mod multiplier (ScoreProcessor
        #           .updateScore: TotalScore = round(TSWM × scoreMultiplier))
        #   combo change (GetComboScoreChange): Fruit +300·w, Droplet
        #     (LargeTickHit) +100·w, w = clamp(log₄(comboAfter), 0.5, log₄200)
        #   bonus (GetBaseScoreForResult): caught banana LargeBonus = +200
        # The previous sim here used the OSU processor's shape (500k/500k
        # split, √combo, acc⁵, +50 bananas) and ran 5%+ off on real plays.
        # The curve is additionally END-PINNED by render.py to the exact
        # converted header total (score_fidelity), so trajectory AND final
        # value both match what lazer/the osu! website shows.
        def _combo_w(c: int) -> float:
            if c <= 1:
                return 0.5
            return min(max(0.5, math.log(c, self._COMBO_BASE)), log_cap)

        _COMBO_CHANGE = {ObjType.FRUIT: 300.0, ObjType.DROPLET: 100.0}
        _combo_kinds = (ObjType.FRUIT, ObjType.DROPLET)
        max_fruit = sum(1 for o in objs if o.kind is ObjType.FRUIT)
        max_tiny = sum(1 for o in objs if o.kind is ObjType.TINY_DROPLET)
        _ft_div = max_tiny + max_fruit
        _fruit_tiny_scale = (max_tiny / _ft_div) if _ft_div else 0.0
        _combo_budget = 1_000_000.0 - 400_000.0 * _fruit_tiny_scale
        _droplets_budget = 400_000.0 * _fruit_tiny_scale
        max_combo_portion = 0.0
        _pc = 0
        for obj in objs:
            if obj.kind in _combo_kinds:
                _pc += 1
                max_combo_portion += _COMBO_CHANGE[obj.kind] * _combo_w(_pc)
        _mod_mult = mods_score_multiplier(getattr(self.meta, "mods", 0) or 0)

        combo = max_combo = 0
        hp = 1.0
        c300 = c100 = c50 = ckatu = cmiss = ctiny_miss = 0
        combo_portion = 0.0
        bonus = 0.0   # banana LargeBonus: +200 per caught banana (lazer ScoreV3)
        pending_hyper: int | None = None
        pending_target: float | None = None
        import random as _random
        _pile_ci = -1                          # current combo group for the pile
        _pile_placed: list = []                # (offset_x, offset_y) already placed
        _pile_adj = 128.0 * self.obj_scale * (10.0 / 64.0)   # jitter radius, osu units
        for obj, caught in zip(objs, self._caught):
            if obj.kind in (ObjType.FRUIT, ObjType.DROPLET):
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
                    combo_portion += _COMBO_CHANGE[obj.kind] * _combo_w(combo)
                    hp = min(1.0, hp + 0.025)
                    if obj.kind is ObjType.FRUIT:
                        c300 += 1
                        self._catches.append((obj.time_ms, obj.x, obj.combo_index, obj.hyperdash, combo))
                        _cxl, _ = catcher_x_at(self.frames, obj.time_ms)
                        self._light_events.append(
                            (obj.time_ms, obj.x - _cxl, "fruit",
                             obj.combo_index, combo, obj.x))
                        # lazer computePositionInStack: land at where it was caught
                        # (offset from plate centre), then jitter ONLY to de-overlap
                        # against fruit already on the plate this combo.
                        if obj.combo_index != _pile_ci:
                            _pile_ci = obj.combo_index
                            _pile_placed = []
                        cxh, _ = catcher_x_at(self.frames, obj.time_ms)
                        # lazer Catcher.computePositionInStack (ported exactly):
                        # land at the FULL offset where it was caught
                        # (palpableObject.X - catcher.X), then de-overlap-jitter
                        # below only when it collides with fruit already piled.
                        px = (obj.x - cxh)
                        py = 0.0
                        chk = _pile_adj * _pile_adj
                        rng = _random.Random(int(obj.time_ms))
                        _g = 0
                        while _g < 64 and any((px - fx) ** 2 + (py - fy) ** 2 < chk
                                              for fx, fy in _pile_placed):
                            # lazer: X += RNG(-adjustedRadius, adjustedRadius);
                            # Y -= RNG(0, 5).  (_pile_adj == lazer adjustedRadius:
                            # DisplaySize.X * 10/64 == 128*scale * 10/64.)
                            px += rng.uniform(-_pile_adj, _pile_adj)
                            py -= rng.uniform(0.0, 5.0)
                            _g += 1
                        # Keep the pile within the catch range (== the
                        # catcher body extent). The de-overlap jitter above
                        # can push a fruit past where it was ever catchable,
                        # which renders it OFF the plate (Red 2026-07-25). lazer
                        # keeps caught fruit on the catcher; clamp X to +/-half.
                        px = max(-self.half, min(self.half, px))
                        _pile_placed.append((px, py))
                        self._plate.append([obj.time_ms, px, py, obj.combo_index,
                                            obj.hyperdash, None, None])
                    else:
                        c100 += 1
                        _cxl, _ = catcher_x_at(self.frames, obj.time_ms)
                        self._light_events.append(
                            (obj.time_ms, obj.x - _cxl, "droplet",
                             obj.combo_index, combo, obj.x))
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
                else:
                    ctiny_miss += 1
            elif obj.kind is ObjType.BANANA and caught:
                # lazer standardised scoring counts each caught banana as a
                # LargeBonus (+200 — CatchScoreProcessor.GetBaseScoreForResult
                # overrides the default 50) in the BONUS portion of the total —
                # on banana-only maps ("Endless Spinner") this is the ENTIRE
                # score beyond the empty combo portion.
                bonus += 200.0
                _cxl, _ = catcher_x_at(self.frames, obj.time_ms)
                self._light_events.append(
                    (obj.time_ms, obj.x - _cxl, "banana",
                     obj.combo_index, combo, obj.x))
            if (obj.kind is ObjType.BANANA and caught
                    and self.skin is not None):
                # BANANAS ARE CAUGHT TOO ("bananas ignored by the platter"):
                # pass 1 already marks them caught geometrically, which makes
                # them vanish at the plane — but they never touched the plate,
                # so a shower looked like it rained straight through. lazer's
                # Catcher explodes a caught banana off the plate IMMEDIATELY
                # (bananas never persist in the pile; no combo/hp change,
                # LargeBonus +200 modelled in the branch above). clear_time is
                # pre-set to the catch time so the group pass below leaves it
                # alone, and anim="banana" routes the burst to the banana
                # sprite. Skin-gated: the certified argon path stays
                # bit-identical (flagged as a known gap for the next argon
                # certification pass).
                cxb, _ = catcher_x_at(self.frames, obj.time_ms)
                self._plate.append([obj.time_ms, (obj.x - cxb), 0.0,
                                    obj.combo_index, obj.hyperdash,
                                    obj.time_ms, "banana"])
            # lazer CatchScoreProcessor.ComputeTotalScore:
            #   comboPortion·comboProgress + dropletsPortion·dropletsHit
            #   + bonusPortion, × mod multiplier. comboProgress falls back
            #   to 1 when the map has no combo-giving objects (banana-only),
            #   exactly like ScoreProcessor.updateScore.
            _combo_progress = ((combo_portion / max_combo_portion)
                               if max_combo_portion > 0 else 1.0)
            _droplets_hit = (c50 / max_tiny) if max_tiny else 0.0
            score = (_combo_budget * _combo_progress
                     + _droplets_budget * _droplets_hit
                     + bonus) * _mod_mult
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
                # --- running-accuracy reconcile to the .osr header ------------
                # The per-type CAUGHT counts were already reconciled to the
                # header (pass-1 _reconcile). But the geometric sim can PARSE
                # more objects of a type than the header accounts for — most
                # often TINY DROPLETS, whose bit-exact RNG spacing we can't
                # reproduce (Stark's SS: 778 parsed tiny vs 724 in the header).
                # Reconcile caps caught at the header count, so the surplus
                # (778-724 = 54) land as PHANTOM misses that never happened in
                # the real play — dragging the running/checkpoint accuracy
                # below the header (2286/2340 = 97.7%) even on a genuine SS,
                # so the HUD showed "SS · 97%".
                #
                # The header is authoritative (osu! itself recorded it), so the
                # displayed accuracy must reconcile to it. Every checkpoint's
                # running miss total is discounted by the phantom share (spread
                # proportionally across the accumulated misses, since we can't
                # know each phantom's exact timing), so the running acc
                # converges to — and ENDS EXACTLY at — the header accuracy.
                #
                #   phantom = sim's final judged-miss total - header misses
                #   at checkpoint i:  denom_i = caught_i + miss_i
                #                                 - phantom*(miss_i / miss_final)
                #                     acc_i   = caught_i / denom_i
                #
                # miss_final == header misses => phantom == 0 => IDENTITY: an
                # honest sub-SS play (sim already matches the header) is left
                # untouched and still shows its true sub-SS acc. Only the
                # phantom surplus is discounted, and only ever downward in
                # count, so the >20%-desync guard (pass 1) still fails a truly
                # mismatched replay before we ever get here.
                header_miss = m.count_katu + m.count_miss
                cps = self._checkpoints
                if cps:
                    sim_miss_final = sum(cps[-1].counts[3:5])   # tiny + fruit/large misses
                    phantom = sim_miss_final - header_miss
                    if phantom > 0:
                        for cp in cps:
                            c300i, c100i, c50i, tmissi, missi = cp.counts
                            caughti = c300i + c100i + c50i
                            missacc = tmissi + missi
                            adj = (phantom * missacc / sim_miss_final
                                   if sim_miss_final > 0 else 0.0)
                            denom = caughti + missacc - adj
                            cp.accuracy = (caughti / denom) if denom > 0 else 1.0
                            cp.accuracy = max(0.0, min(1.0, cp.accuracy))
                        # pin the last checkpoint to the exact header value so
                        # the counter/grade land on the authoritative number
                        cps[-1].accuracy = self.real_accuracy

    # --- input overlay timeline ----------------------------------------------

    def _build_inputs(self) -> None:
        """Per-REPLAY-FRAME input segments for the key overlay (lazer
        CatchReplayFrame.FromLegacy semantics): over a frame interval
        [aᵢ, aᵢ₊₁) the held actions are MoveLeft/MoveRight = sign of the x
        delta TO the next frame (lazer attaches the movement to the EARLIER
        frame), and Dash = the frame's own button bit. Prefix onset counts
        give tap-accurate press counters at replay resolution — deriving
        held/counts from lerped VIDEO-frame positions aliased rapid taps into
        holds (a 60 ms tap can live entirely inside one 33 ms video frame's
        interval and net dx≈0) and froze the counters during tap-spam."""
        fs = self.frames
        starts: list[int] = []
        held: list[tuple[bool, bool, bool]] = []
        counts: list[tuple[int, int, int]] = []
        nl = nr = nd = 0
        prev = (False, False, False)
        for a, b in zip(fs, fs[1:]):
            st = (b.x < a.x, b.x > a.x, bool(a.dashing))
            starts.append(a.time_ms)
            held.append(st)
            nl += st[0] and not prev[0]
            nr += st[1] and not prev[1]
            nd += st[2] and not prev[2]
            counts.append((nl, nr, nd))
            prev = st
        if fs:   # tail: stationary, last frame's dash state
            st = (False, False, bool(fs[-1].dashing))
            starts.append(fs[-1].time_ms)
            held.append(st)
            nd += st[2] and not prev[2]
            counts.append((nl, nr, nd))
        self._in_starts, self._in_held, self._in_counts = starts, held, counts

    def input_state(self, t0: float, t1: float):
        """((L, R, D) held anywhere within the video-frame interval (t0, t1],
        (nL, nR, nD) presses begun up to t1) — both at replay-frame
        resolution. `t0/t1` are MAP-time ms (the replay's own axis), so
        DT/HT rate mods are inherently accounted for by the caller's
        map_step-sized interval."""
        if getattr(self, "_in_starts", None) is None:
            self._build_inputs()
        starts = self._in_starts
        if not starts:
            return (False, False, False), (0, 0, 0)
        i1 = bisect_right(starts, t1) - 1
        if i1 < 0:
            return (False, False, False), (0, 0, 0)
        i0 = max(0, bisect_right(starts, t0) - 1)
        L = R = D = False
        for i in range(i0, i1 + 1):
            # skip the part of segment i0 that ended before t0 ONLY when a
            # later segment begins inside the window (no smear back in time)
            if i > i0 or i1 == i0 or starts[min(i + 1, len(starts) - 1)] > t0:
                sl, sr, sd = self._in_held[i]
                L, R, D = L or sl, R or sr, D or sd
        return (L, R, D), self._in_counts[i1]

    def _reconcile_combo_runs(self, objs, margin, m) -> None:
        """Move misses so the combo-run structure reproduces the header's
        max_combo (the .osr ground truth) — the safety net behind the sim.

        The geometric sim + count reconcile can place a miss on the wrong
        object (borderline geometry, stable HR offsets on maps whose nested
        generation is not bit-exact, …). Combo runs are fully determined by
        WHERE the misses sit among the combo objects (fruits + large
        droplets), so: if max(run) != meta.max_combo, search single-miss
        moves (miss → caught same-kind object elsewhere; per-type counts
        preserved) and apply the target-hitting move with the lowest
        geometric cost (sum of |margin| of the two flipped calls — prefer
        un-missing the near-catch and missing the near-miss). No move can
        hit the target → leave the sim as-is (results screen still shows the
        header's own numbers). Runs before pass 2: the moved miss genuinely
        falls through / the moved catch lands on the plate, so the visual
        stream, HUD combo, score and HP all stay coherent."""
        target = int(getattr(m, "max_combo", 0) or 0)
        if target <= 0:
            return
        kinds = (ObjType.FRUIT, ObjType.DROPLET)
        idxs = [i for i, o in enumerate(objs) if o.kind in kinds]
        if not idxs:
            return
        caught = self._caught
        pos_of = {i: p for p, i in enumerate(idxs)}
        n = len(idxs)

        def max_run(miss_positions) -> int:
            prev, mx = -1, 0
            for s in miss_positions:
                mx = max(mx, s - prev - 1)
                prev = s
            return max(mx, n - 1 - prev)

        miss_pos = sorted(pos_of[i] for i in idxs if not caught[i])
        if max_run(miss_pos) == target:
            return
        # cost bound: the search is m×n — worth it only where it matters (a
        # low-miss play whose ONE misplaced miss glares). Heavy-miss plays are
        # statistically close already and the search would be seconds-slow.
        if len(miss_pos) > 80:
            return
        best = None   # (cost, src_obj_index, dst_obj_index)
        for src in [i for i in idxs if not caught[i]]:
            sp = pos_of[src]
            others = [p for p in miss_pos if p != sp]
            kind = objs[src].kind
            for dst in idxs:
                if not caught[dst] or objs[dst].kind is not kind:
                    continue
                cand = others + [pos_of[dst]]
                cand.sort()
                if max_run(cand) != target:
                    continue
                cost = abs(margin[src]) + abs(margin[dst])
                if best is None or cost < best[0]:
                    best = (cost, src, dst)
        if best is None:
            return
        _, src, dst = best
        caught[src] = True
        caught[dst] = False

    def state_at(self, t_ms: int) -> _Checkpoint:
        if not self._checkpoints:
            return _Checkpoint(t_ms, 0, 0, 1.0)
        i = bisect_right(self._cp_times, t_ms) - 1
        if i < 0:
            return _Checkpoint(t_ms, 0, 0, 1.0)
        return self._checkpoints[i]

    def catch_events(self):
        """(objects, caught) — the simulated objects (death-truncated on a
        fail) with their aligned caught verdicts, post count-reconcile.
        The hitsound mixer's input: only CAUGHT objects make a sound."""
        return self._objs, self._caught

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
        # preset bg dim via the DimEnvelope (intro/game/breaks levels with std's
        # smoothstep glides): % dim (higher=darker) → brightness. Computed
        # unconditionally so the storyboard (DimmableStoryboard) can share the
        # SAME dim even when there is no beatmap bg image; when there IS a bg the
        # value is the exact one the bg sprite has always used, so nothing
        # changes for existing renders.
        d = max(0.0, 1.0 - self._dim_env.level(t_ms))
        s.sb_brightness = d
        # dimmed beatmap background (drawn first, behind everything)
        if self.has_bg:
            s.sprites.append(Sprite(self.screen_w / 2, self.screen_h / 2,
                                    self.screen_w, self.screen_h,
                                    texture_key="bg", color=(d, d, d, 1.0)))
        # Split point for the storyboard underlay: it draws right after the
        # beatmap background image (index 1 when present, else 0 = behind the
        # playfield), with the playfield drawn on top.
        s.bg_split = len(s.sprites)

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
        # Z-ORDER (Top-250 player report, verified against lazer): the catch
        # playfield draws EARLIER hit objects IN FRONT — an earlier plain
        # fruit occludes a later hyperfruit's red echo/ring. Painter's order
        # means later-drawn = on top, so iterate the visible window in
        # REVERSE time order (latest first, earliest last). Both argon and
        # skinned paths; each object's own layer order is unchanged. (The GL
        # batch still lifts ADDITIVE sprites to a second pass — additive
        # blending commutes, so ordering there is visually irrelevant.)
        for obj, caught in zip(reversed(self._objs[_lo:_hi]),
                               reversed(self._caught[_lo:_hi])):
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
        # replay-frame-accurate key overlay state over THIS video frame's
        # map-time interval (render_core sets video_step_ms = frame_ms·rate;
        # the fallback derives it from cfg — identical for single renders).
        step = getattr(self, "video_step_ms", None)
        if step is None:
            step = 1000.0 / self.cfg.fps * (getattr(self.bm, "rate", 1.0) or 1.0)
        s.keys_held, s.key_counts = self.input_state(t_ms - step, t_ms)
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
        # Caught fruit render BEHIND the catcher body -- lazer Catcher.cs adds
        # caughtObjectContainer BEFORE `body = SkinnableCatcher()`, so the body
        # draws on top (Red 2026-07-25: "fruits appear before the plate instead
        # of behind"). This reverses the 2026-07-20 in-front tweak, which was
        # not lazer-accurate. Hit explosions stay in front (lazer puts
        # hitExplosionContainer after body).
        s.sprites.extend(self._plate_stack(scx, t_ms))
        s.sprites.extend(self._catcher_sprites(scx, dashing or hyper, hyper_amt, t_ms))
        s.sprites.extend(self._catch_explosions(t_ms, scx))

        # letterbox during breaks (drawn last so bars sit on top). Bar alpha
        # rides the 0→1 break envelope — the SAME smoothstep glides as the bg
        # dim — so the bars fade in at the break start and out across the
        # resume anchor instead of snapping while the background glides.
        if self.cfg.letterbox_breaks:
            lb = self._break_env.level(t_ms)
            if lb > 0.004:
                bar = self.screen_h * 0.11
                a = 0.92 * lb
                s.sprites.append(Sprite(self.screen_w / 2, bar / 2,
                                        self.screen_w, bar,
                                        texture_key=None, color=(0, 0, 0, a)))
                s.sprites.append(Sprite(self.screen_w / 2, self.screen_h - bar / 2,
                                        self.screen_w, bar,
                                        texture_key=None, color=(0, 0, 0, a)))

        # R3D intro splash -- topmost intro element, over the idle scene
        if self.logo_start_ms is not None:
            s.sprites.extend(self._logo_sprites(t_ms))

        cp = self.state_at(t_ms)
        s.combo = cp.combo
        s.score = int(round(cp.score * self.score_scale))
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
        # rosu's catch passed_objects domain is FRUITS + DROPLETS only (its
        # difficulty objects; it saturates at n_fruits+n_droplets, which is
        # also the map's max combo). Our checkpoints index EVERY sim object —
        # tiny droplets and bananas too — so the old `passed_objects=i + 1`
        # overshot by their count and hit the full-map ceiling mid-play; from
        # there rosu clamps to the FULL map and the result only moves when
        # `misses`/`combo` change, so the counter sat bit-frozen between
        # chain misses (the ~4:40 freeze report). Map each checkpoint to the
        # number of combo objects actually passed instead.
        _pc = 0
        passed = []
        for o in self._objs:
            if o.kind in (ObjType.FRUIT, ObjType.DROPLET):
                _pc += 1
            passed.append(_pc)
        samples = {}
        for i in range(0, n, step):
            cp = cps[i]
            c3, c1, c5, tmiss, miss = cp.counts
            try:
                samples[i] = rosu.Performance(
                    mods=int(mods), passed_objects=max(1, passed[i]),
                    n300=c3, n100=c1,
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
        # --pp: pin the live pp counter's ENDPOINT to the EXACT official pp
        # passed via --pp (RenderConfig.pp_override). Scale the whole
        # checkpoint curve by a constant (official / rosu_final) so the
        # counter keeps its rosu/score-progress SHAPE and only its endpoint
        # eases onto the passed value -- mirrors the taiko _final_pp
        # endpoint-anchor. The results card is pinned separately
        # (CatchLazerResults). No-op when --pp is absent.
        ov = getattr(self.cfg, "pp_override", None)
        if ov is not None:
            rosu_final = cps[-1].pp if cps else 0.0
            if rosu_final and rosu_final > 0:
                scale = float(ov) / rosu_final
                for cp in cps:
                    cp.pp *= scale
            else:  # rosu gave 0 / no final -> just show the flat official value
                for cp in cps:
                    cp.pp = float(ov)

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
        # Precedence (owner ground truth 2026-07-22, VK_CTB1.3 + "LegenD.":
        # stable shows WHITE fruit AND white ticks even though the map ships
        # red [Colours] — the USER's chosen skin's own Combo1..N wins):
        #   1. the user skin's skin.ini Combo1..N (combo_colors_custom — ONLY
        #      a skin the user actually picked; the bundled `_default-source`
        #      Night05 ini never qualifies, see CatchSkin.__init__),
        #   2. the map's own [Colours],
        #   3. any other skin-chain palette (default-skin ini / stock combos),
        #   4. lazer's default combo colours.
        # This is one shared source for fruits AND droplets/tiny droplets —
        # they must never disagree (the "red slider ticks under white fruits"
        # bug: droplets took the map red while the fruits' white overlay art
        # made them READ as skin-white). Skinless path order is unchanged
        # (map → lazer palette): certified argon output stays bit-identical.
        sk = self.skin
        # "Combo colors" setting (--combo-colors): "skin" makes the user's chosen
        # skin win OUTRIGHT (its own Combo1..N, else the default/stock skin
        # combos), ignoring the map's [Colours] entirely — this is what the site's
        # "Skin" choice means. "beatmap" (default) keeps the certified 2026-07-22
        # precedence below, byte-identical to before.
        if getattr(self.cfg, "combo_colors", "beatmap") == "skin":
            if sk is not None:
                return sk.combo_color(combo_index)
            return self._LAZER_COMBO[combo_index % len(self._LAZER_COMBO)]
        if sk is not None and getattr(sk, "combo_colors_custom", False):
            return sk.combo_color(combo_index)
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

    def _banana_scale(self, obj, t_ms) -> float:
        """lazer DrawableBanana.UpdateInitialTransforms scale animation (a
        'roughly matches osu-stable' port in lazer itself): the banana spawns
        at Scale·(0.6 + 1.6·RandomSingle(3)) and shrinks linearly to Scale·0.6
        over TimePreempt — bananas visibly pop in big and settle at 0.6× a
        fruit. Multiplies the normal object size (which already carries
        HitObject.Scale via fruit_screen)."""
        start = 0.6 + 1.6 * _obj_rand01(obj.time_ms, obj.x, 3)
        p = self.preempt
        if p <= 0:
            return 0.6
        u = (t_ms - (obj.time_ms - p)) / p
        u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        return start + (0.6 - start) * u

    def _banana_angle(self, obj, t_ms) -> float:
        """lazer DrawableBanana.Update rotation (source-verified): LERP from
        startAngle = 180°·(RandomSingle(1)·2−1) to endAngle with series 2,
        over TimePreempt; freely extrapolates for uncaught bananas."""
        import math
        a0 = (self._rand_single(obj, 1) * 2.0 - 1.0) * math.pi
        a1 = (self._rand_single(obj, 2) * 2.0 - 1.0) * math.pi
        p = self.preempt
        if p <= 0:
            return a1
        u = (t_ms - (obj.time_ms - p)) / p
        return a0 + (a1 - a0) * u

    def _rand_single(self, obj, series: int) -> float:
        """Stand-in for lazer's StatelessRNG.NextSingle(RandomSeed, series)."""
        return _obj_rand01(obj.time_ms, obj.x, series)

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
            # lazer DrawableBanana size-over-lifetime (skin-independent — the
            # transform sits on the drawable): spawn at 0.6+1.6·rand, settle
            # at 0.6× a fruit over the preempt. The flare rides the piece.
            bs = self._banana_scale(obj, t_ms)
            d *= bs
            out.append(Sprite(x, y, d, d, texture_key="argon_pip",
                              color=(1, 1, 1, 1), rotation=rot))
            out.append(Sprite(x, y, d, d, texture_key=f"argon_fruit_{v}",
                              color=(*tint, 1.0), rotation=rot, additive=True))
            fa = self._banana_flare_alpha(obj.time_ms, t_ms)
            if fa > 0.0:
                out.append(Sprite(x, y, size * bs * 2.2, size * bs * 1.1,
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
    # tiny droplet = 0.5x large droplet (osu!lazer DrawableTinyDroplet
    # halves ScaleContainer.Scale). Was 0.8x (0.416) which read as same size
    # as big ticks — Red approved lazer-accurate 0.5x 2026-08-08.
    _SKIN_TINY = 0.26
    _SKIN_BANANA = 1.05

    def _skinned_object(self, obj, x, y, t_ms) -> list[Sprite]:
        # Per-element skin honoring: use the skin's sprite for this object kind
        # when it ships one, else fall back to the Argon look for that kind.
        sk = self.skin
        if obj.kind in (ObjType.DROPLET, ObjType.TINY_DROPLET):
            # Legacy path whenever ANY droplet art resolves (base OR overlay):
            # stable resolves base and overlay per-FILE and independently, so a
            # skin shipping only one of the pair must still render legacy art
            # (the classic-default base backs a lone overlay) — never argon.
            if not (sk.has("fruit-drop") or sk.has("fruit-drop-overlay")):
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
            # tick on the GAMEPLAY clock, whose ms unit IS the map-time axis
            # this sim runs on (DT/HT compress real time at playback) — so the
            # duration is preempt + 2000 with NO rate scaling (the old ÷rate
            # double-applied the mod, same bug as the preempt window).
            rot = 0.0
            if self.cfg.fruit_rotation:
                start = _obj_rand01(obj.time_ms, obj.x, 1) * 0.349   # 0..20°
                dur = self.preempt + 2000.0                          # map ms
                rot = start + 12.566 * (t_ms - (obj.time_ms - self.preempt)) / dur
            return self._base_overlay("fruit-drop", x, y, size,
                                      self._combo_tint(obj.combo_index), rot)
        if obj.kind is ObjType.BANANA:
            # Per-FILE fallthrough guard (Sofia render bug): legacy path when
            # ANY banana art resolves — a skin shipping ONLY the -overlay
            # (VK_CTB1.3) must compose it over the classic-default base from
            # the fallback chain, never drop to the argon hue blobs.
            if not (sk.has("fruit-bananas") or sk.has("fruit-bananas-overlay")):
                return self._argon_object(obj, x, y, t_ms)
            size = self.fruit_screen * self._SKIN_BANANA
            # STABLE/lazer-legacy banana tint (owner in-game reference: classic
            # YELLOW bananas, never the rainbow): lazer DrawableBanana pins one
            # of three yellow variants per banana (getBananaColour, seeded per
            # object) — (255,240,0) / (255,192,0) / (214,221,28). The rainbow
            # hue-cycle stays an ARGON-path flourish only (_argon_object still
            # honours cfg.banana_rainbow); tinting legacy ART with it is what
            # painted Flameneon's "grey/rainbow circles".
            _BANANA_TINTS = ((1.0, 0.941, 0.0), (1.0, 0.753, 0.0),
                             (0.839, 0.867, 0.110))
            r = _obj_rand01(obj.time_ms, obj.x, 0)   # NextInt(3, seed): series 0
            tint = _BANANA_TINTS[min(2, int(r * 3.0))]
            # lazer DrawableBanana.Update (source-verified): rotation LERPs
            # from one random angle to another — 180·(RandomSingle(1)·2-1) →
            # 180·(RandomSingle(2)·2-1) — over TimePreempt (NOT the droplet
            # +720° spin), extrapolating freely past the plane like lazer.
            rot = 0.0
            if self.cfg.fruit_rotation:
                rot = self._banana_angle(obj, t_ms)
            # size-over-lifetime: spawn big, settle at 0.6× (lazer/stable)
            size *= self._banana_scale(obj, t_ms)
            return self._base_overlay("fruit-bananas", x, y, size, tint, rot)
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
        # Base and overlay resolve INDEPENDENTLY (stable per-file fallback):
        # either may be missing — draw whichever exists. An overlay-only skin
        # (base unresolvable even via the default chain) still shows its ring.
        out: list[Sprite] = []
        if self.skin.has(base_key):
            out.append(Sprite(x, y, size, size, texture_key=base_key,
                              color=(*tint, 1.0), rotation=rot))
        ov = f"{base_key}-overlay"
        if self.skin.has(ov):
            out.append(Sprite(x, y, size, size, texture_key=ov, color=(1, 1, 1, 1), rotation=rot))
        return out

    def _facing_at(self, t_ms) -> float:
        """Catcher facing at t_ms: +1.0 = right, -1.0 = left — osu!lazer
        CatcherArea.SetCatcherPosition: `if (lastPosition < newPosition)
        VisualDirection = Direction.Right; else if (lastPosition > newPosition)
        VisualDirection = Direction.Left;` — the facing flips ONLY while x
        strictly changes; a stationary catcher KEEPS its last direction, and
        the initial facing is Right (Catcher.VisualDirection's default).
        Catcher.Update turns it into `body.Scale = new Vector2((int)
        VisualDirection, 1)` — a horizontal mirror of the catcher body about
        its own centreline. Our sprite shader scales the unit quad by u_size
        BEFORE rotation, so a NEGATIVE sprite width is exactly that mirror.

        Since catcher_x_at lerps linearly between replay frames, the facing
        over a frame span (a, b] is simply sign(b.x - a.x) when nonzero —
        precomputed once into a sparse (time, facing) change list."""
        changes = self._facing_changes
        if changes is None:
            changes = [(-(1 << 60), 1.0)]         # Direction.Right default
            cur = 1.0
            fs = self.frames
            for a, b in zip(fs, fs[1:]):
                if b.x != a.x:
                    d = 1.0 if b.x > a.x else -1.0
                    if d != cur:
                        cur = d
                        changes.append((a.time_ms, d))
            self._facing_changes = changes
        import bisect
        i = bisect.bisect_right(changes, (t_ms, 2.0)) - 1
        return changes[i][1]

    def _catcher_ghost(self, scx, rgb, alpha, scale=1.0, dy=0.0,
                       facing=1.0) -> list[Sprite]:
        """One ADDITIVE afterimage of the FULL catcher body (skin sprite, or the
        Argon bar+bumpers) at screen-x scx — the unit lazer's CatcherTrail draws
        (CatcherTrail.body = SkinnableCatcher, Blending = Additive).
        `facing` mirrors the ghost: lazer's CatcherTrailEntry snapshots
        Catcher.BodyScale (= Scale * body.Scale, X sign = facing) at spawn
        time, and CatcherTrailDisplay keeps that sign for the ghost's life
        (updateCatcherTrailsScale preserves Math.Sign(oldEntry.Scale.X))."""
        ck = getattr(self.skin, "catcher_key", None) if self.skin is not None else None
        if ck is not None and self.skin.has(ck):
            hb = self.catcher_w * self.skin.catcher_aspect
            return [Sprite(scx, self.plane_y + hb * 0.46 + dy,
                           self.catcher_w * scale * facing, hb * scale,
                           texture_key=ck,
                           color=(*rgb, alpha), additive=True)]
        from osu_catch_renderer.skin.lazer_skin import argon_catcher_metrics
        g = argon_catcher_metrics(self.catcher_w, self.unit_px, self.plane_y)
        cy = g["cy"] + dy
        # facing flips the whole body in lazer; the Argon catcher is built
        # mirror-symmetric (see _catcher_sprites) so this is an identity —
        # wired anyway so the argon ghost stays exactly lazer's flipped body.
        bx = (g["bar_w"] * 0.5 + g["bump_w"] * 0.5) * scale * facing
        return [
            Sprite(scx, cy, g["bar_w"] * scale * facing, g["bar_h"] * scale,
                   texture_key="argon_bar_cap", color=(*rgb, alpha), additive=True),
            Sprite(scx - bx, cy, g["bump_w"] * scale * facing, g["bump_h"] * scale,
                   texture_key="argon_bar_cap", color=(*rgb, alpha), additive=True),
            Sprite(scx + bx, cy, g["bump_w"] * scale * facing, g["bump_h"] * scale,
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
                # ghost keeps the facing it had when it was SPAWNED (lazer
                # stores Catcher.BodyScale in the CatcherTrailEntry) — sample
                # facing at the same past instant as the position.
                out.extend(self._catcher_ghost(self._sx(px), trail_rgb, alpha,
                                               facing=self._facing_at(t_ms - age)))
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
                    scale=0.95 + 0.25 * e, dy=-10.0 * self.unit_px * e,
                    facing=self._facing_at(h_start)))
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
        #
        # FACING (community report: "the catcher never turns"): lazer mirrors
        # the catcher body to face its direction of travel — Catcher.Update:
        # body.Scale = new Vector2((int)VisualDirection, 1). Only the BODY
        # flips: the caught-fruit pile (caughtObjectContainer.Scale =
        # new Vector2(1 / Scale.X), unsigned) and the hit explosions stay
        # unmirrored on lazer master (the old FlipCatcherPlate skin option is
        # gone) — so _plate_stack/_catch_explosions are deliberately untouched.
        facing = self._facing_at(t_ms) if t_ms is not None else 1.0
        ck = getattr(self.skin, "catcher_key", None) if self.skin is not None else None
        # osu!catch Catcher.CurrentState = Kiai while catching fruit in a kiai
        # section (Catcher.cs OnNewResult). Legacy skins swap to
        # fruit-catcher-kiai; the Argon default catcher has no kiai state.
        if (t_ms is not None and self.skin is not None
                and self.skin.has("fruit-catcher-kiai")
                and any(a <= t_ms < b for a, b in self.kiai_ranges)):
            ck = "fruit-catcher-kiai"
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
            return [Sprite(x, self.plane_y + h * 0.46, w * facing, h,
                           texture_key=ck, color=tint)]
        # osu!lazer ArgonCatcher: a white rounded catch bar (0.8 of the catcher
        # width) + a bumper at each end of the catch range + faint side lines
        # out to the screen edges. Footprint/placement unchanged (full width =
        # catcher_w, bar top on plane_y). Hyperdash turns it red + glowing.
        from osu_catch_renderer.skin.lazer_skin import argon_catcher_metrics
        g = argon_catcher_metrics(self.catcher_w, self.unit_px, self.plane_y)
        cy = g["cy"]
        # lazer's ArgonCatcher is white; hyperdash turns it full red (no glow —
        # the red after-image trail is the hyper cue, drawn in _dash_trail).
        col = (1.0, 1.0 - hyper_amt, 1.0 - hyper_amt, 1.0)
        out: list[Sprite] = []
        # facing applies to the Argon body too (lazer flips the whole
        # SkinnableCatcher regardless of skin) — but ArgonCatcher.cs builds a
        # mirror-symmetric body (central circle bar, identical bumper +
        # 20×-width side line on EACH side), so the flip is a visual identity
        # here. Wired through regardless so this stays exactly lazer's body.
        # main catch bar
        out.append(Sprite(x, cy, g["bar_w"] * facing, g["bar_h"],
                          texture_key="argon_bar_cap", color=col))
        # bumpers at the ends of the catch range (flanking the 0.8 bar)
        bx = (g["bar_w"] * 0.5 + g["bump_w"] * 0.5) * facing
        out.append(Sprite(x - bx, cy, g["bump_w"] * facing, g["bump_h"],
                          texture_key="argon_bar_cap", color=col))
        out.append(Sprite(x + bx, cy, g["bump_w"] * facing, g["bump_h"],
                          texture_key="argon_bar_cap", color=col))
        # faint long lines out to the screen edges (alpha 0.25) — NOT flipped:
        # lazer's are symmetric 20×-catcher-width boxes on both sides (flip =
        # no-op); ours reach the screen edges by construction, and mirroring
        # their screen-space extents would break that.
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
            if sk is not None and (sk.has("fruit-bananas")
                                   or sk.has("fruit-bananas-overlay")):
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

    def _legacy_hit_lighting(self, t_ms, scx) -> list[Sprite]:
        """STABLE hit lighting on legacy-skinned renders — exact port of lazer
        LegacyHitExplosion (Skinning/Legacy/LegacyHitExplosion.cs), which is
        itself the stable behaviour: every CAUGHT palpable object (fruit,
        droplet, banana — Catcher.OnNewResult fires addLighting for any hit
        when HitLighting is on; stable defaults it ON) flashes the CLASSIC
        default skin's scoreboard-explosion sprites at the caught plate
        offset, tinted the object's accent colour, additive, riding the
        catcher:
          · beam  (explosion1 = scoreboard-explosion-2, rotated -90 = a
            VERTICAL streak): non-droplets only; x-scale 1 → 16·s over 160 ms
            Easing.Out, y 0.9 → 1.1; FadeOutFromOne(300);
            s = clamp(comboAtJudgement/200, 0.35, 1.125)
          · puff  (explosion2 = scoreboard-explosion-1): scale (0.9, 1) →
            (0.9, 1.3) over 500 ms Easing.Out; FadeOutFromOne(700)
        Container Scale 0.5; offset clamped inside the catch range. Textures
        come from the DEFAULT skin dir only (lazer: DefaultClassicSkin —
        never the user skin), loaded as catch_light_beam / catch_light_puff."""
        sk = self.skin
        if sk is None or not sk.has("catch_light_puff"):
            return []
        out: list[Sprite] = []
        up = self.unit_px
        # texture logical sizes → osu units via the ×0.5 container scale
        puff = sk.textures["catch_light_puff"]      # 39×105 logical (w×h swap below)
        beam = sk.textures.get("catch_light_beam")
        ph, pw = puff.shape[:2]
        _BANANA_TINTS = ((1.0, 0.941, 0.0), (1.0, 0.753, 0.0),
                         (0.839, 0.867, 0.110))
        # PERF: bisect the 700ms live window out of the (time-sorted) events
        times = getattr(self, "_light_times", None)
        if times is None:
            self._light_events.sort(key=lambda e: e[0])
            times = self._light_times = [e[0] for e in self._light_events]
        lo = bisect_left(times, t_ms - 700)
        hi = bisect_right(times, t_ms)
        for ct, off, kind, ci, cmb, ox in self._light_events[lo:hi]:
            age = t_ms - ct
            if not (0.0 <= age <= 700.0):
                continue
            if kind == "banana":
                # same tint the falling banana carried (same seed inputs)
                r = _obj_rand01(ct, ox, 0)
                tint = _BANANA_TINTS[min(2, int(r * 3.0))]
            else:
                tint = self._combo_tint(ci)
            offc = max(-self.half, min(self.half, off))
            x = scx + offc * up
            # puff: rotated -90 → screen w = tex_h·yscale, h = tex_w·0.9
            u5 = min(1.0, age / 500.0)
            ys = 1.0 + 0.3 * (1.0 - (1.0 - u5) ** 2)        # Easing.Out(quad)
            a2 = max(0.0, 1.0 - age / 700.0)
            p_w = ph * ys * 0.5 * up
            p_h = pw * 0.9 * 0.5 * up
            out.append(Sprite(x, self.plane_y - p_h * 0.5, p_w, p_h,
                              texture_key="catch_light_puff",
                              color=(*tint, a2), additive=True))
            # beam: fruits + bananas only, combo-scaled vertical streak
            if kind != "droplet" and beam is not None and age <= 300.0:
                bh, bw = beam.shape[:2]
                s = min(1.125, max(0.35, cmb / 200.0))
                u16 = min(1.0, age / 160.0)
                xs = 1.0 + (16.0 * s - 1.0) * (1.0 - (1.0 - u16) ** 2)
                a1 = max(0.0, 1.0 - age / 300.0)
                b_len = bw * xs * 0.5 * up                  # vertical extent
                b_th = bh * 1.0 * 0.5 * up                  # thickness (~0.9→1.1)
                out.append(Sprite(x, self.plane_y - b_len * 0.5, b_th, b_len,
                                  texture_key="catch_light_beam",
                                  color=(*tint, a1), additive=True))
        return out

    def _catch_explosions(self, t_ms, scx=None) -> list[Sprite]:
        """osu!lazer ArgonHitExplosion: every caught FRUIT fires a tall,
        combo-coloured vertical glow that scales up to (1.1, 20*s) over 200ms
        (OutQuint) then retracts to (1.1, 1) over 600ms (In), plus a large faint
        glow (radius 50, colour 20% toward white). The whole thing fades out
        over 400ms. s = clamp(combo/200, 0.35, 1.125). Droplets don't explode."""
        # SKIN HONORING: a legacy skin gets STABLE's hit lighting (the classic
        # scoreboard-explosion flash — see _legacy_hit_lighting), NOT the
        # Argon glow; skinless renders keep the full Argon explosion.
        sk = self.skin
        if sk is not None and (sk.has(sk.fruit_key(0))
                               or getattr(sk, "catcher_key", None) is not None):
            return self._legacy_hit_lighting(t_ms, scx if scx is not None
                                             else self._sx(256.0))
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
        from osu_catch_renderer.render.effects import logo_alpha, logo_scale, LOGO_UI_SIZE
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
