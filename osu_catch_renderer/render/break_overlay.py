"""osu!lazer's BreakOverlay, ported 1:1 onto catch's CPU/PIL HUD compositing.

Source of truth — ppy/osu master (read 2026-07-23, keep in sync):
  osu.Game/Screens/Play/BreakOverlay.cs        fade/slide timings, progress
                                               bar semantics, layout Y=±15
  osu.Game/Screens/Play/BreakTracker.cs        which breaks count (HasEffect),
                                               Period = (Start, End - FADE)
  osu.Game/Screens/Play/Break/BreakInfo.cs     "CURRENT PROGRESS" + info lines
  osu.Game/Screens/Play/Break/BreakInfoLine.cs label Yellow/value YellowLight,
                                               2px split margins, acc format
  osu.Game/Screens/Play/Break/RemainingTimeCounter.cs  ceil(ms/1000) seconds
  osu.Game/Screens/Play/Break/BreakArrows.cs   chevron pair geometry/offsets
  osu.Game/Screens/Play/Break/GlowIcon.cs      sharp icon + BlueLighter glow
  osu.Game/Screens/Play/Break/BlurredIcon.cs   blur-only + additive + a=0.7
  osu.Game/Beatmaps/Timing/BreakPeriod.cs      MIN_BREAK_DURATION = 650

The exact lazer timeline (BreakOverlay.updateDisplay, absolute from b.Start;
the tracker's Period trims BREAK_FADE_DURATION=325ms off the END, so with
D = period duration = break duration - 325):
  t'=0..325   fadeContainer.FadeIn(325) [linear]; arrows slide in (OutQuint,
              325ms); counter X -50->0 / info X +50->0 (OutQuint, 325ms);
              progress-bar CONTAINER width 0 -> 0.3 rel (OutQuint, 325ms)
  t'=0..D+325 counter counts (D+325 = full break duration) -> 0, linear;
              display = ceil(count/1000)
  every frame bar width DampContinuously(current, target, halfTime=40ms);
              target = max(0, (Period.End - now - 325) / D)  [reaches 0
              already 325ms BEFORE the fade-out starts]
  t'=D        fadeContainer.FadeOut(325); arrows slide back out (OutQuint,
              325ms); bar container width snaps to 0 — gone at t'=D+325,
              exactly the break's end.

ARROWS — deliberately NOT blinking: lazer master's BreakArrows only slide
in/out and hold (Show/Hide MoveToX, Easing.OutQuint); no flash/blink
transform exists anywhere in the file or its history since the 2018 redesign.
The "one bright one dim" pair per side is the sharp GlowIcon (60px chevron,
sigma-10 BlueLighter glow) in front of the big BlurredIcon (130px, sigma-20,
blur-only, additive, alpha 0.7). The cursor-parallax micro-motion
(ParallaxContainer, cursor-driven) has no analogue in a fixed render and is
dropped. If the owner wants an actual blink it would be OUR deviation — not
ported here.

VALUES — live, not snapshotted: BreakOverlay.LoadComplete BindTo()s the
ScoreProcessor's Accuracy/Rank bindables, so lazer updates them DURING the
break too (in practice they're constant — no judgements land mid-break).
We sample the sim's running accuracy each frame accordingly.

TYPOGRAPHY — lazer draws Torus (info) + Venera numerals (counter). This
engine's HUD stack stands in exactly like the rest of the HUD does: the
bundled OFL Nunito (argon_hud's Torus stand-in) for the text lines, the
Argon counter glyphs (argon_counter) for the countdown digits. DELIBERATE
on skinned renders too: stable has no break-overlay equivalent and the owner
wants THIS lazer look on both paths, so the overlay is identical lazer
styling regardless of skin (only the rest of the HUD switches).

Coordinates are lazer's 768-tall UI space (argon_hud.LAZER_UI_HEIGHT)
scaled by lk = screen_h/768; the arrows' X offsets are RELATIVE TO WIDTH
(GlowIcon RelativePositionAxes = Axes.X), matching upstream. All times are
MAP-time ms (lazer runs these transforms on the rate-adjusted
FrameStableClock, which is the map timeline — DT/HT inherently correct).
"""
from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from osu_catch_renderer.argon.argon_counter import argon_digit_advance, argon_glyph_rgba
from osu_catch_renderer.argon.argon_hud import ARGON_GLYPH_CAP_SCALE, LAZER_UI_HEIGHT, NUNITO_PATH

# --- lazer constants (files cited in the module docstring) -------------------
MIN_BREAK_DURATION = 650.0        # BreakPeriod.MIN_BREAK_DURATION (HasEffect)
BREAK_FADE_MS = MIN_BREAK_DURATION / 2.0   # BreakOverlay.BREAK_FADE_DURATION
REMAINING_MAX_W = 0.3             # remaining_time_container_max_size
VERTICAL_MARGIN = 15.0            # BreakOverlay.vertical_margin (lazer px)
BAR_H = 8.0                       # remainingTimeBox Height
DAMP_HALF_MS = 40.0               # Interpolation.DampContinuously halfTime
SLIDE_X = 50.0                    # counter/info MoveToX slide distance

GLOW_ICON_SIZE = 60.0             # BreakArrows glow_icon_*
GLOW_ICON_SIGMA = 10.0
GLOW_ICON_FINAL = 0.22            # X offsets, RELATIVE TO WIDTH
GLOW_ICON_OFFSCREEN = 0.6
BLUR_ICON_SIZE = 130.0            # BreakArrows blurred_icon_*
BLUR_ICON_SIGMA = 20.0
BLUR_ICON_FINAL = 0.38
BLUR_ICON_OFFSCREEN = 0.7
BLUR_ICON_ALPHA = 0.7

BLUE_LIGHTER = (0xDD, 0xFF, 0xFF)   # OsuColour.BlueLighter (glow colour)
YELLOW = (0xFF, 0xCC, 0x22)         # OsuColour.Yellow (info labels)
YELLOW_LIGHT = (0xFF, 0xDD, 0x55)   # OsuColour.YellowLight (info values)
SHADOW_GRAY = (51, 51, 51)          # OsuColour.Gray(0.2f)
SHADOW_ALPHA = 0.8                  # .Opacity(0.8f)
SHADOW_RADIUS = 260.0               # EdgeEffect shadow radius (lazer px)
SHADOW_CORE_W, SHADOW_CORE_H = 80.0, 4.0   # the CircularContainer core

COUNTER_SIZE = 33.0               # RemainingTimeCounter OsuFont.Numeric 33
TITLE_SIZE = 15.0                 # "CURRENT PROGRESS" bold 15
LINE_SIZE = 17.0                  # BreakInfoLine label/value size
LINE_MARGIN = 2.0                 # BreakInfoLine margin each side of centre
FLOW_SPACING = 5.0                # BreakInfo FillFlow Spacing(5)

# BeatmapsStrings rank display text (osu-resources Localisation/Web) — the
# GradeDisplay renders ScoreRank.GetLocalisableDescription(): X="SS",
# XH="Silver SS", SH="Silver S". lazer's HD/FL AdjustRank turns X/S silver.
_HD, _FL = 1 << 3, 1 << 10


def _out_quint(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 5


def grade_display(accuracy: float, mods: int) -> str:
    """The break overlay's Grade line text: the engine's own catch grade
    (hud._catch_grade — single source for cutoffs) mapped to lazer's rank
    display strings, with the HD/FL silver adjustment lazer applies."""
    from osu_catch_renderer.hud.hud import _catch_grade   # lazy: hud imports this module
    g = _catch_grade(max(0.0, min(1.0, accuracy)) * 100.0, 0)
    if mods & (_HD | _FL):
        if g == "SS":
            return "Silver SS"
        if g == "S":
            return "Silver S"
    return g


class LazerBreakOverlay:
    """Stateful per-render overlay: bakes the static art once, then draws
    per frame during effective breaks (duration >= MIN_BREAK_DURATION).
    Frames must arrive in monotonic map-time order (they do — same contract
    as the ArgonHud rolling counters); the damped bar width is replayed
    statefully like lazer's always-running Update()."""

    def __init__(self, w: int, h: int, breaks, mods: int = 0):
        self.w, self.h = int(w), int(h)
        self.mods = int(mods or 0)
        self.lk = self.h / LAZER_UI_HEIGHT
        # BreakTracker.Breaks: only HasEffect breaks, Period end trimmed by
        # BREAK_FADE_DURATION. We keep (start, D) with D = period duration;
        # the overlay is on screen over [start, start + D + 325].
        self.periods = sorted(
            (float(s), float(e - s) - BREAK_FADE_MS)
            for s, e in (breaks or ())
            if (e - s) >= MIN_BREAK_DURATION)
        self._starts = [p[0] for p in self.periods]
        # DampContinuously state (remainingTimeBox.Width, RELATIVE 0..1)
        self._bar_w = 0.0
        self._last_t: float | None = None
        self._text_cache: dict = {}
        self._font_cache: dict = {}
        if not self.periods:
            return                     # no effective breaks -> never draws
        lk = self.lk
        self._shadow = self._bake_shadow()
        # arrows: right-pointing bakes, mirrored for the left-pointing pair
        self._glow_r = self._bake_glow_icon()
        self._glow_l = self._glow_r.transpose(Image.FLIP_LEFT_RIGHT)
        blur_r = self._bake_blurred_icon()
        self._blur_r = blur_r
        self._blur_l = blur_r[:, ::-1].copy()
        # countdown digits: Argon counter glyphs (lit cells, no wireframe
        # backing — that's a score-counter decoration, not break art)
        dh = max(4, int(round(COUNTER_SIZE * lk)))
        dw = max(2, int(round(argon_digit_advance() * dh)))
        self._digits = {}
        for ch in "0123456789":
            cell = Image.fromarray(argon_glyph_rgba(ch), "RGBA")
            self._digits[ch] = cell.resize((dw, dh), Image.LANCZOS)
        self._digit_w, self._digit_h = dw, dh

    # --- bakes ---------------------------------------------------------------

    def _bake_shadow(self) -> Image.Image:
        """The fadeContainer's first child: an invisible 80x4 pill whose
        EdgeEffect SHADOW (radius 260, gray(0.2) @ 0.8) is the big soft dark
        blob behind the centre block. Approximated as a quadratic falloff
        over the radius from the pill edge (o!f edge-effect profile)."""
        lk = self.lk
        R = SHADOW_RADIUS * lk
        cw, ch = SHADOW_CORE_W * lk, SHADOW_CORE_H * lk
        W = int(math.ceil(cw + 2 * R))
        H = int(math.ceil(ch + 2 * R))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = ch / 2.0                      # pill corner radius
        qx = np.abs(xx - W / 2.0) - (cw / 2.0 - r)
        qy = np.abs(yy - H / 2.0) - (ch / 2.0 - r)
        d = (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
             + np.minimum(np.maximum(qx, qy), 0.0) - r)
        fall = np.clip(1.0 - d / R, 0.0, 1.0) ** 2
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = SHADOW_GRAY
        rgba[..., 3] = np.round(fall * SHADOW_ALPHA * 255.0).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    def _chevron_mask(self, size_px: int) -> Image.Image:
        """FontAwesome Solid.ChevronRight silhouette: a bold '>' polyline
        (glyph aspect ~0.63 in a square SpriteIcon cell), round caps/joint."""
        s = size_px
        m = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(m)
        w = max(2, int(round(s * 0.17)))
        pts = [(0.36 * s, 0.14 * s), (0.67 * s, 0.50 * s),
               (0.36 * s, 0.86 * s)]
        d.line(pts, fill=255, width=w, joint="curve")
        for px, py in (pts[0], pts[2]):
            d.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2],
                      fill=255)
        return m

    def _bake_glow_icon(self) -> Image.Image:
        """GlowIcon: sharp white chevron over its BlueLighter gaussian glow
        (GlowingDrawable: blurred silhouette tinted GlowColour, original on
        top)."""
        lk = self.lk
        s = max(4, int(round(GLOW_ICON_SIZE * lk)))
        sigma = GLOW_ICON_SIGMA * lk
        pad = int(math.ceil(3 * sigma)) + 1
        cv = Image.new("L", (s + 2 * pad, s + 2 * pad), 0)
        cv.paste(self._chevron_mask(s), (pad, pad))
        glow_a = cv.filter(ImageFilter.GaussianBlur(sigma))
        glow = Image.new("RGBA", cv.size, BLUE_LIGHTER + (0,))
        glow.putalpha(glow_a)
        sharp = Image.new("RGBA", cv.size, (255, 255, 255, 0))
        sharp.putalpha(cv)
        return Image.alpha_composite(glow, sharp)

    def _bake_blurred_icon(self) -> np.ndarray:
        """BlurredIcon: blur-only (DrawOriginal=false), additive, alpha 0.7.
        Kept as a premultiplied float RGB field ready for additive blending."""
        lk = self.lk
        s = max(4, int(round(BLUR_ICON_SIZE * lk)))
        sigma = BLUR_ICON_SIGMA * lk
        pad = int(math.ceil(3 * sigma)) + 1
        cv = Image.new("L", (s + 2 * pad, s + 2 * pad), 0)
        cv.paste(self._chevron_mask(s), (pad, pad))
        a = (np.asarray(cv.filter(ImageFilter.GaussianBlur(sigma)),
                        np.float32) / 255.0) * BLUR_ICON_ALPHA
        col = np.array(BLUE_LIGHTER, np.float32)
        return a[..., None] * col[None, None, :]

    def _font(self, size_l: float, weight: int) -> ImageFont.FreeTypeFont:
        key = (size_l, weight)
        f = self._font_cache.get(key)
        if f is None:
            px = max(6, int(round(size_l * ARGON_GLYPH_CAP_SCALE * self.lk)))
            try:
                f = ImageFont.truetype(NUNITO_PATH, px)
                try:
                    f.set_variation_by_axes([weight])
                except Exception:  # noqa: BLE001 — non-variable build
                    pass
            except OSError:
                from osu_catch_renderer.hud.fonts import font as _fallback
                f = _fallback(px)
            self._font_cache[key] = f
        return f

    def _text(self, text: str, size_l: float, weight: int,
              color) -> Image.Image:
        """A text run as a cached RGBA sprite, canvas height = the font's
        full line box (ascent+descent) so pastes centre like o!f's line
        anchoring."""
        key = (text, size_l, weight, color)
        im = self._text_cache.get(key)
        if im is None:
            f = self._font(size_l, weight)
            x0, _, x1, _ = f.getbbox(text)
            try:
                asc, desc = f.getmetrics()
            except AttributeError:
                asc, desc = f.getbbox("Ag")[3], 0
            im = Image.new("RGBA", (max(1, x1 - x0), max(1, asc + desc)),
                           color + (0,))
            ImageDraw.Draw(im).text((-x0, 0), text, font=f,
                                    fill=color + (255,))
            self._text_cache[key] = im
        return im

    # --- per-frame -----------------------------------------------------------

    def _paste(self, img: Image.Image, sprite: Image.Image,
               cx: float, cy: float, alpha: float) -> None:
        """Alpha-over paste of an RGBA sprite CENTRED at (cx, cy), scaled by
        the overlay alpha. PIL clips partial/off-frame boxes itself."""
        if alpha <= 0.004:
            return
        x = int(round(cx - sprite.width / 2.0))
        y = int(round(cy - sprite.height / 2.0))
        a = sprite.getchannel("A")
        if alpha < 0.999:
            a = a.point(lambda v: int(v * alpha))
        img.paste(sprite, (x, y), a)

    def _add(self, img: Image.Image, field: np.ndarray,
             cx: float, cy: float, alpha: float) -> None:
        """Additive blend of a premultiplied float-RGB field centred at
        (cx, cy) — BlendingParameters.Additive for the BlurredIcons."""
        if alpha <= 0.004:
            return
        fh, fw = field.shape[:2]
        x0 = int(round(cx - fw / 2.0))
        y0 = int(round(cy - fh / 2.0))
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x0 + fw, img.width), min(y0 + fh, img.height)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        sub = field[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] * alpha
        box = (ix0, iy0, ix1, iy1)
        base = np.asarray(img.crop(box), np.float32)
        out = np.clip(base + sub, 0.0, 255.0).astype(np.uint8)
        img.paste(Image.fromarray(out, "RGB"), box[:2])

    def _bar_pill(self, w_px: int, h_px: int, alpha: float) -> Image.Image:
        """remainingTimeBox: a white fully-rounded Circle, h = min(8, w)."""
        from osu_catch_renderer.argon.argon_hud import bake_pill_alpha
        a = bake_pill_alpha(w_px, h_px) * (alpha * 255.0)
        rgba = np.zeros((h_px, w_px, 4), np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = np.round(a).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    def draw(self, img: Image.Image, t_ms: float, accuracy: float) -> None:
        """Compose the overlay for map time t_ms onto the PIL frame. Called
        every frame (the bar damp runs continuously, like lazer's Update);
        cheap no-op outside break windows."""
        if not self.periods:
            return
        t = float(t_ms)
        dt = 16.7 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        # active period: overlay lives over [start, start + D + FADE]
        idx = bisect_right(self._starts, t) - 1
        cur = None
        if idx >= 0:
            s0, D = self.periods[idx]
            if t <= s0 + D + BREAK_FADE_MS:
                cur = (s0, D)

        # remainingTimeBox.Width — DampContinuously toward
        # max(0, (Period.End - now - FADE) / D), EVERY frame, in/out of breaks
        if cur is None:
            target = 0.0
        else:
            s0, D = cur
            target = max(0.0, (s0 + D - t - BREAK_FADE_MS) / D) if D > 0 else 0.0
        self._bar_w = target + (self._bar_w - target) * (0.5 ** (dt / DAMP_HALF_MS))

        if cur is None:
            return
        s0, D = cur
        tp = t - s0                       # time since break start
        # fadeContainer alpha: linear FadeIn/FadeOut over BREAK_FADE_MS
        if tp >= D:
            alpha = max(0.0, 1.0 - (tp - D) / BREAK_FADE_MS)
        else:
            alpha = min(1.0, tp / BREAK_FADE_MS)
        if alpha <= 0.004:
            return

        lk = self.lk
        cx, cy = self.w / 2.0, self.h / 2.0
        p_in = _out_quint(tp / BREAK_FADE_MS)

        # 1) shadow blob (first fadeContainer child)
        self._paste(img, self._shadow, cx, cy, alpha)

        # 2) progress bar: container width 0 -> 0.3 (OutQuint, 325ms), snap
        #    to 0 at t'=D; pill width rides the damped fraction
        wc = REMAINING_MAX_W * p_in if tp < D else 0.0
        bw = int(round(wc * self.w * max(0.0, min(1.0, self._bar_w))))
        if bw >= 2:
            bh = max(1, int(round(min(BAR_H * lk, bw))))
            self._paste(img, self._bar_pill(bw, bh, alpha), cx, cy, 1.0)

        # 3) remaining-time counter: ceil(count/1000); count runs linearly
        #    from the FULL break duration to 0 at the break's end
        count = max(0.0, (D + BREAK_FADE_MS) - tp)
        text = str(int(math.ceil(count / 1000.0)))
        dx = -SLIDE_X * lk * (1.0 - p_in)          # MoveToX(-50 -> 0)
        run_w = self._digit_w * len(text)
        left = cx + dx - run_w / 2.0
        bottom = cy - VERTICAL_MARGIN * lk
        for i, ch in enumerate(text):
            self._paste(img, self._digits[ch],
                        left + (i + 0.5) * self._digit_w,
                        bottom - self._digit_h / 2.0, alpha)

        # 4) BreakInfo (slides +50 -> 0): title, then Accuracy / Grade lines
        #    split 2px either side of centre; values LIVE like lazer's
        #    bindables (constant mid-break in practice)
        dxi = SLIDE_X * lk * (1.0 - p_in)
        cxi = cx + dxi
        y0 = cy + VERTICAL_MARGIN * lk
        title = self._text("CURRENT PROGRESS", TITLE_SIZE, 700,
                           (255, 255, 255))
        self._paste(img, title, cxi, y0 + TITLE_SIZE * lk / 2.0, alpha)
        acc = max(0.0, min(1.0, float(accuracy)))
        acc_txt = f"{math.floor(acc * 10000.0) / 100.0:.2f}%"  # FormatAccuracy
        rows = [("Accuracy", acc_txt),
                ("Grade", grade_display(acc, self.mods))]
        ly = y0 + (TITLE_SIZE + FLOW_SPACING) * lk
        for label, value in rows:
            mid = ly + LINE_SIZE * lk / 2.0
            lab = self._text(label, LINE_SIZE, 400, YELLOW)
            val = self._text(value, LINE_SIZE, 700, YELLOW_LIGHT)
            self._paste(img, lab, cxi - LINE_MARGIN * lk - lab.width / 2.0,
                        mid, alpha)
            self._paste(img, val, cxi + LINE_MARGIN * lk + val.width / 2.0,
                        mid, alpha)
            ly += LINE_SIZE * lk

        # 5) arrows, topmost: slide in over the fade (OutQuint), hold, slide
        #    back out from t'=D. X offsets are fractions of the WIDTH.
        if tp >= D:
            po = _out_quint((tp - D) / BREAK_FADE_MS)
            g_off = GLOW_ICON_FINAL + (GLOW_ICON_OFFSCREEN - GLOW_ICON_FINAL) * po
            b_off = BLUR_ICON_FINAL + (BLUR_ICON_OFFSCREEN - BLUR_ICON_FINAL) * po
        else:
            g_off = GLOW_ICON_OFFSCREEN + (GLOW_ICON_FINAL - GLOW_ICON_OFFSCREEN) * p_in
            b_off = BLUR_ICON_OFFSCREEN + (BLUR_ICON_FINAL - BLUR_ICON_OFFSCREEN) * p_in
        # origins CentreRight/CentreLeft: the offset is the icon's inner
        # LAYOUT edge (AutoSize box = the 60/130px icon; the glow/blur
        # overhang is draw-only, like o!f's inflated draw quad) -> shift
        # each sprite centre outward by half the LAYOUT size, not the
        # padded canvas
        g_half = GLOW_ICON_SIZE * lk / 2.0
        b_half = BLUR_ICON_SIZE * lk / 2.0
        self._add(img, self._blur_r, cx - b_off * self.w - b_half, cy, alpha)
        self._add(img, self._blur_l, cx + b_off * self.w + b_half, cy, alpha)
        self._paste(img, self._glow_r, cx - g_off * self.w - g_half, cy, alpha)
        self._paste(img, self._glow_l, cx + g_off * self.w + g_half, cy, alpha)
