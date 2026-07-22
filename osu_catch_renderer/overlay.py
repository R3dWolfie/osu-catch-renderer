"""osu!catch VERSUS OVERLAY — N players' replays on ONE catch field.

Colour-codes players EXACTLY like the std `showdown!mrgd` merge engine
(merge_mod.py): the per-player element (there the cursor, here the CATCHER +
its dash trail) is GRAYSCALED so it keeps its shape/gradients but drops its hue,
then MULTIPLY-tinted by the player's colour — so it recolours cleanly to any hue
instead of muddying a pre-coloured skin. The fruits (falling AND on the plate)
stay FULL SKIN. Player colours come from the same `_colors(n)` HSV palette, with
R3D forced to red.

The catcher texture is grayscaled once at upload (render_core, key
`fruit-catcher-idle__ovl`); the Argon bar-cap is already white so it tints as-is.
"""
from __future__ import annotations

import colorsys
from bisect import bisect_left, bisect_right

from .models import SceneState, Sprite
from .replay import catcher_x_at

# element textures that get grayscale+tint per player (catcher + its trail).
# Everything else (fruits, plate, explosions) stays full skin.
_CATCHER_GRAY = {"fruit-catcher-idle": "fruit-catcher-idle__ovl"}
_CATCHER_TINTABLE = {"argon_bar_cap"}          # already white → tints directly
_R3D_RED = (0.96, 0.26, 0.26)                  # R3D is always red (std rule)


def player_colors(names: list) -> list:
    """Std `merge_mod._colors(n)` palette (hsv h=(i/n+0.11), s=0.82, v=1.0),
    with R3D forced red + identical picks de-collided by a small hue nudge."""
    n = len(names)
    cols = [colorsys.hsv_to_rgb((i / max(1, n) + 0.11) % 1.0, 0.82, 1.0)
            for i in range(n)]
    for i, nm in enumerate(names):
        if str(nm).strip().upper() == "R3D":
            cols[i] = _R3D_RED
    # de-collide: nudge any colour that lands too close to an earlier one
    for i in range(1, n):
        for j in range(i):
            if sum((cols[i][k] - cols[j][k]) ** 2 for k in range(3)) < 0.02:
                h, s, v = colorsys.rgb_to_hsv(*cols[i])
                cols[i] = colorsys.hsv_to_rgb((h + 0.13) % 1.0, 0.82, 1.0)
    return cols


class CatchOverlaySim:
    """Drop-in for `CatchSim` in the render loop: same `build_scene(t_ms)` +
    `logo_start_ms`/`compute_pp_curve` surface, composites N players."""

    def __init__(self, sims: list, names: list, gray_keys=None,
                 catcher_keys=None):
        assert sims, "overlay needs at least one sim"
        self.sims = sims
        self.names = names
        self.colors = player_colors(names)
        # texture keys that render_core baked a grayscale "{key}__ovl" copy of
        # (the BASE skin's fruit sprites) so the caught fruits recolour cleanly.
        self.gray_keys = set(gray_keys or ())
        # per-player catcher texture key — each player's OWN skin's catcher,
        # grayscaled (base/unlinked players fall back to the base catcher). The
        # base playfield + fruits come from the invoker's skin; only the catcher
        # is per-player. None → everyone uses the base catcher gray.
        self.catcher_keys = list(catcher_keys) if catcher_keys else \
            ["fruit-catcher-idle__ovl"] * len(sims)
        self.base = sims[0]
        n = len(self.base._objs)
        self._merged = [any(i < len(s._caught) and bool(s._caught[i])
                            for s in sims) for i in range(n)]
        self.logo_start_ms = None

    def compute_pp_curve(self, *a, **k):
        return None

    def _colorize_rig(self, sprites, col, opacity, catcher_key):
        """Recolour a player's rig to their colour, drop it to `opacity`, and
        draw it ADDITIVELY so overlapping players' colours ADD UP and glow. The
        CATCHER swaps to this player's OWN grayscale catcher (`catcher_key`); the
        caught fruits swap to the BASE skin's grayscale fruit; both multiply-tint.
        The white Argon bar/trail multiplies directly; everything else just takes
        the opacity."""
        for sp in sprites:
            cr, cg, cb, ca = sp.color
            if sp.texture_key == "fruit-catcher-idle":     # player's OWN catcher
                sp.texture_key = catcher_key
                sp.color = (col[0] * cr, col[1] * cg, col[2] * cb, ca * opacity)
            elif sp.texture_key in self.gray_keys:          # base caught fruits
                sp.texture_key = sp.texture_key + "__ovl"
                sp.color = (col[0] * cr, col[1] * cg, col[2] * cb, ca * opacity)
            elif sp.texture_key in _CATCHER_TINTABLE:       # white Argon bar/trail
                sp.color = (col[0] * cr, col[1] * cg, col[2] * cb, ca * opacity)
            else:
                sp.color = (cr, cg, cb, ca * opacity)
            sp.additive = True      # additive mix → overlaps glow
        return sprites

    def build_scene(self, t_ms: int) -> SceneState:
        base = self.base
        s = SceneState(time_ms=t_ms)

        if base.has_bg:
            in_break = any(a <= t_ms <= b for a, b in base.bm.breaks)
            first_t = base.bm.objects[0].time_ms if base.bm.objects else 0
            dim_pct = (base.cfg.bg_dim_breaks if in_break
                       else base.cfg.bg_dim_intro if t_ms < first_t
                       else base.cfg.bg_dim_game)
            d = max(0.0, 1.0 - dim_pct / 100.0)
            s.sprites.append(Sprite(base.screen_w / 2, base.screen_h / 2,
                                    base.screen_w, base.screen_h,
                                    texture_key="bg", color=(d, d, d, 1.0)))

        # SHARED falling fruits — FULL SKIN
        # PERF: only objects with time_ms in [t-250, t+preempt] can pass the
        # visibility test below — bisect that window out of the (sorted)
        # object list instead of scanning the whole map every frame (same
        # windowing CatchSim.build_scene already does; the per-object test
        # stays the authoritative filter, and iteration order is unchanged).
        if base._objs_sorted:
            _lo = bisect_left(base._obj_times, t_ms - 250)
            _hi = bisect_right(base._obj_times, t_ms + base.preempt)
        else:
            _lo, _hi = 0, len(base._objs)
        for obj, caught in zip(base._objs[_lo:_hi], self._merged[_lo:_hi]):
            end = obj.time_ms if caught else obj.time_ms + 250
            if not (obj.time_ms - base.preempt <= t_ms <= end):
                continue
            y = base._fruit_y(obj.time_ms, t_ms)
            sprites = base._object_sprites(obj, base._sx(obj.x), y, t_ms)
            if not caught and t_ms > obj.time_ms:
                mu = (t_ms - obj.time_ms) / 250.0
                for sp in sprites:
                    r, g, b, al = sp.color
                    sp.color = (r, g, b, al * max(0.0, 1.0 - mu))
                    sp.rotation = sp.rotation * (1.0 + mu)
            elif base.hidden:
                a = base._hd_alpha(obj.time_ms, t_ms)
                if a < 1.0:
                    for sp in sprites:
                        r, g, b, al = sp.color
                        sp.color = (r, g, b, al * a)
            s.sprites.extend(sprites)

        # PER-PLAYER: the whole rig (catcher + trail + caught-fruit plate +
        # explosions) is recoloured to the player's colour AND drawn at 1/N
        # opacity, so overlapping players BLEND (translucent) instead of forming
        # one solid blob — no vertical stagger. Catcher + caught fruits both
        # recolour (grayscale the skin texture, multiply the colour).
        n = len(self.sims)
        opacity = min(1.0, 2.0 / max(1, n))     # 2× of 1/N (Red), clamped
        for i, (sim, col) in enumerate(zip(self.sims, self.colors)):
            cx, dashing = catcher_x_at(sim.frames, t_ms)
            scx = sim._sx(cx)
            hyper_amt = sim._hyper_amount(t_ms)
            rig = []
            if sim.cfg.catcher_dash_trail:
                rig.extend(sim._dash_trail(t_ms, hyper_amt))
            rig.extend(sim._catcher_sprites(scx, dashing or hyper_amt > 0.0,
                                            hyper_amt, t_ms))
            rig.extend(sim._plate_stack(scx, t_ms))
            rig.extend(sim._catch_explosions(t_ms))
            ckey = (self.catcher_keys[i] if i < len(self.catcher_keys)
                    else "fruit-catcher-idle__ovl")
            s.sprites.extend(self._colorize_rig(rig, col, opacity, ckey))

        if base.cfg.letterbox_breaks and any(a <= t_ms <= b for a, b in base.bm.breaks):
            bar = base.screen_h * 0.11
            s.sprites.append(Sprite(base.screen_w / 2, bar / 2, base.screen_w,
                                    bar, texture_key=None, color=(0, 0, 0, 0.92)))
            s.sprites.append(Sprite(base.screen_w / 2, base.screen_h - bar / 2,
                                    base.screen_w, bar, texture_key=None,
                                    color=(0, 0, 0, 0.92)))

        if self.logo_start_ms is not None:
            base.logo_start_ms = self.logo_start_ms
            s.sprites.extend(base._logo_sprites(t_ms))

        cp = base.state_at(t_ms)
        s.combo = cp.combo
        s.score = int(cp.score * base.score_scale)
        s.hp = cp.hp
        s.accuracy = max(0.0, min(1.0, cp.accuracy))
        s.counts = cp.counts
        cx0, d0 = catcher_x_at(base.frames, t_ms)
        s.catcher_x = float(cx0)
        s.dashing = bool(d0)
        # Feed the real std versus-merge leaderboard (osu-std MergeBoard): pass
        # (name, colour, sim, end_ms); the HUD wraps each sim in a HudData-shaped
        # adapter and drives ONE persistent MergeBoard. Score = the catch sim's
        # standardised ScoreV3 (state_at().score) — the same score the std merge
        # board shows.
        s.overlay_board = [
            (self.names[i], self.colors[i], self.sims[i],
             (self.sims[i].frames[-1].time_ms if self.sims[i].frames else None))
            for i in range(len(self.sims))]
        return s
