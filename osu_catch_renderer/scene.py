"""Simulate the catch playthrough and build per-frame draw lists.

The catcher path comes straight from the replay (we render what the player
did), so "caught" is decided geometrically: was the catcher within its
catch-range of the object's x at the object's time. That drives combo, a
score estimate, and an HP estimate. The HUD's *final* numbers come from the
replay's authoritative counts; the live values are our running simulation.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

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


class CatchSim:
    def __init__(self, beatmap: CatchBeatmap, frames: list[CatchFrame], cfg: RenderConfig,
                 skin=None, has_bg: bool = False, meta=None,
                 end_ms: int | None = None):
        self.bm = beatmap
        self.frames = frames
        self.cfg = cfg
        self.skin = skin
        self.meta = meta
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
        self._catches: list[tuple[int, int, bool]] = []   # (time, combo_index, hyper)
        self._hyper_windows: list[tuple[int, int]] = []    # catcher glows red in these
        self._simulate()

    # --- simulation -----------------------------------------------------------

    # lazer combo-portion log accumulation constants (CatchScoreProcessor)
    _COMBO_BASE = 4
    _COMBO_CAP = 200

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

        # --- pass 2: combo / score / hp / acc / counts from reconciled catches ---
        combo = max_combo = 0
        hp = 1.0
        c300 = c100 = c50 = ckatu = cmiss = ctiny_miss = 0
        combo_portion = 0.0
        pending_hyper: int | None = None
        for obj, caught in zip(objs, self._caught):
            if obj.kind in (ObjType.FRUIT, ObjType.DROPLET):
                if pending_hyper is not None:
                    self._hyper_windows.append((pending_hyper, obj.time_ms))
                    pending_hyper = None
                if caught:
                    combo += 1
                    max_combo = max(max_combo, combo)
                    combo_portion += 300.0 * min(max(0.5,
                        math.log(combo, self._COMBO_BASE)), log_cap)
                    hp = min(1.0, hp + 0.025)
                    if obj.kind is ObjType.FRUIT:
                        c300 += 1
                        self._catches.append((obj.time_ms, obj.combo_index, obj.hyperdash))
                    else:
                        c100 += 1
                    if obj.hyperdash:
                        pending_hyper = obj.time_ms
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
                    combo_portion += 30.0
                else:
                    ctiny_miss += 1
            caught_acc = c300 + c100 + c50
            total_acc = caught_acc + cmiss + ckatu + ctiny_miss
            acc = (caught_acc / total_acc) if total_acc else 1.0
            self._checkpoints.append(
                _Checkpoint(obj.time_ms, combo, int(combo_portion), hp, acc, max_combo,
                            counts=(c300, c100, c50, ctiny_miss, cmiss + ckatu)))

        # osu!catch legacy counts: 300=caught fruit, 100=caught large droplet,
        # 50=caught tiny droplet, katu=MISSED tiny, miss=missed fruit + large.
        self.final_counts = (c300, c100, c50, ctiny_miss, cmiss + ckatu)
        self.final_accuracy = acc if self._checkpoints else 1.0
        # anchor the displayed score curve to the replay's real final score
        if self.meta is not None and combo_portion > 0 and self.meta.score > 0:
            self.score_scale = self.meta.score / combo_portion
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
        times = [c.time for c in self._checkpoints]
        i = bisect_right(times, t_ms) - 1
        if i < 0:
            return _Checkpoint(t_ms, 0, 0, 1.0)
        return self._checkpoints[i]

    # --- geometry -------------------------------------------------------------

    def _sx(self, osu_x: float) -> float:
        return self.x_off + osu_x * self.unit_px

    def _fruit_y(self, obj_time: int, t: int) -> float:
        f = (t - (obj_time - self.preempt)) / self.preempt
        return -self.fruit_screen + (self.plane_y + self.fruit_screen) * f

    # --- frame ----------------------------------------------------------------

    def build_scene(self, t_ms: int) -> SceneState:
        s = SceneState(time_ms=t_ms)
        # dimmed beatmap background (drawn first, behind everything)
        if self.has_bg:
            # preset bg dim per phase: % dim (higher=darker) -> brightness mult
            in_break = any(a <= t_ms <= b for a, b in self.bm.breaks)
            first_t = self.bm.objects[0].time_ms if self.bm.objects else 0
            dim_pct = (self.cfg.bg_dim_breaks if in_break
                       else self.cfg.bg_dim_intro if t_ms < first_t
                       else self.cfg.bg_dim_game)
            d = max(0.0, 1.0 - dim_pct / 100.0)
            s.sprites.append(Sprite(self.screen_w / 2, self.screen_h / 2,
                                    self.screen_w, self.screen_h,
                                    texture_key="bg", color=(d, d, d, 1.0)))

        # falling objects within their approach window (and not yet caught/past)
        for obj, caught in zip(self._objs, self._caught):
            if obj.time_ms - self.preempt <= t_ms <= obj.time_ms:
                y = self._fruit_y(obj.time_ms, t_ms)
                sprites = self._object_sprites(obj, self._sx(obj.x), y, t_ms)
                if self.hidden:
                    a = self._hd_alpha(obj.time_ms, t_ms)
                    if a < 1.0:
                        for sp in sprites:
                            r, g, b, al = sp.color
                            sp.color = (r, g, b, al * a)
                s.sprites.extend(sprites)

        # catcher (+ dash trail + caught-fruit pile riding on the plate)
        cx, dashing = catcher_x_at(self.frames, t_ms)
        scx = self._sx(cx)
        hyper = self.cfg.show_hyperdash and any(a <= t_ms <= b for a, b in self._hyper_windows)
        if self.cfg.catcher_dash_trail and (dashing or hyper):
            s.sprites.extend(self._dash_trail(t_ms, hyper))
        s.sprites.extend(self._catcher_sprites(scx, dashing or hyper, hyper))
        s.sprites.extend(self._plate_stack(scx, t_ms))

        # letterbox + dim during breaks (drawn last so bars sit on top)
        if self.cfg.letterbox_breaks and any(a <= t_ms <= b for a, b in self.bm.breaks):
            bar = self.screen_h * 0.11
            s.sprites.append(Sprite(self.screen_w / 2, bar / 2, self.screen_w, bar,
                                    texture_key=None, color=(0, 0, 0, 0.92)))
            s.sprites.append(Sprite(self.screen_w / 2, self.screen_h - bar / 2,
                                    self.screen_w, bar, texture_key=None, color=(0, 0, 0, 0.92)))

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
        # final point = the authoritative full-play pp
        try:
            c3, c1, c5, tmiss, miss = cps[-1].counts
            samples[n - 1] = rosu.Performance(
                mods=int(mods), n300=c3, n100=c1, n50=c5, n_katu=tmiss,
                misses=miss, combo=cps[-1].max_combo).calculate(rbm).pp
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

    def _object_sprites(self, obj, x, y, t_ms) -> list[Sprite]:
        if self.skin is not None:
            return self._skinned_object(obj, x, y, t_ms)
        return [self._procedural_object(obj, x, y)]

    def _skinned_object(self, obj, x, y, t_ms) -> list[Sprite]:
        sk = self.skin
        if obj.kind in (ObjType.DROPLET, ObjType.TINY_DROPLET):
            # lazer: large droplet (slider tick) ~ half a fruit, tiny ~ quarter
            size = self.fruit_screen * (0.55 if obj.kind is ObjType.DROPLET else 0.30)
            return self._base_overlay("fruit-drop", x, y, size, sk.combo_color(obj.combo_index))
        if obj.kind is ObjType.BANANA:
            size = self.fruit_screen * 1.05
            if self.cfg.banana_rainbow:
                tint = _hue((t_ms * 0.0009 + obj.x * 0.01) % 1.0)
            else:
                tint = (1.0, 0.85, 0.15)
            return self._base_overlay("fruit-bananas", x, y, size, tint)
        # FRUIT
        hyper = obj.hyperdash and self.cfg.show_hyperdash
        size = self.fruit_screen * (1.32 if hyper else 1.05)
        tint = (1.0, 0.35, 0.35) if hyper else sk.combo_color(obj.combo_index)
        # gentle spin while falling, deterministic per fruit
        rot = 0.0
        if self.cfg.fruit_rotation:
            spin_dir = 1.0 if (int(obj.x) % 2 == 0) else -1.0
            rot = spin_dir * (t_ms - obj.time_ms + self.preempt) * 0.0011
        return self._base_overlay(sk.fruit_key(obj.combo_index), x, y, size, tint, rot)

    def _base_overlay(self, base_key, x, y, size, tint, rot=0.0) -> list[Sprite]:
        out = [Sprite(x, y, size, size, texture_key=base_key, color=(*tint, 1.0), rotation=rot)]
        ov = f"{base_key}-overlay"
        if self.skin.has(ov):
            out.append(Sprite(x, y, size, size, texture_key=ov, color=(1, 1, 1, 1), rotation=rot))
        return out

    def _dash_trail(self, t_ms, hyper) -> list[Sprite]:
        """Faded catcher afterimages at recent positions while dashing."""
        out: list[Sprite] = []
        w = self.catcher_w
        h = w * (self.skin.catcher_aspect if self.skin else 0.32)
        base = (1.0, 0.4, 0.4) if hyper else (0.8, 0.85, 1.0)
        key = "fruit-catcher-idle" if (self.skin and self.skin.has("fruit-catcher-idle")) else "catcher"
        for k, dt in enumerate((26, 52, 80)):
            px, _ = catcher_x_at(self.frames, t_ms - dt)
            alpha = 0.32 * (1.0 - k / 3.0)
            out.append(Sprite(self._sx(px), self.plane_y + h * 0.46, w, h,
                              texture_key=key, color=(*base, alpha)))
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

    def _catcher_sprites(self, x, dashing, hyper=False) -> list[Sprite]:
        if hyper:
            tint = (1.0, 0.45, 0.55, 1.0)
        elif dashing:
            tint = (1.0, 0.8, 0.9, 1.0)
        else:
            tint = (1, 1, 1, 1)
        # A custom skin's catcher takes priority — same layout (CS/mod-driven
        # catcher_w), the skin supplies the sprite. The procedural lazer plate is
        # only the base/fallback when the skin ships no catcher.
        if self.skin is not None and self.skin.has("fruit-catcher-idle"):
            w = self.catcher_w
            h = w * self.skin.catcher_aspect
            return [Sprite(x, self.plane_y + h * 0.46, w, h,
                           texture_key="fruit-catcher-idle", color=tint)]
        from .lazer_skin import CATCHER_ASPECT
        sprite_w = self.catcher_w * 1.02          # +pad baked into the texture
        sprite_h = sprite_w * (CATCHER_ASPECT + 2 * (6 / 568)) / (1 + 2 * (6 / 568))
        return [Sprite(x, self.plane_y + sprite_h * 0.44, sprite_w, sprite_h,
                       texture_key="lazer_catcher", color=tint)]

    def _plate_stack(self, scx, t_ms) -> list[Sprite]:
        """Caught fruit piled on the plate, riding with the catcher and fading."""
        if self.skin is None:
            return []
        STACK_MS = 850
        plate_half = self.half * self.x_scale
        out: list[Sprite] = []
        recent = [c for c in self._catches if 0 <= t_ms - c[0] <= STACK_MS][-12:]
        for ct, ci, hy in recent:
            alpha = 1.0 - (t_ms - ct) / STACK_MS
            ox = (((ct * 131) % 100) / 100 - 0.5) * plate_half * 1.4
            oy = -(((ct * 73) % 4)) * self.fruit_screen * 0.10
            size = self.fruit_screen * 0.55
            tint = (1.0, 0.35, 0.35) if hy else self.skin.combo_color(ci)
            key = self.skin.fruit_key(ci)
            out.append(Sprite(scx + ox, self.plane_y + self.fruit_screen * 0.15 + oy,
                              size, size, texture_key=key, color=(*tint, alpha)))
        return out
