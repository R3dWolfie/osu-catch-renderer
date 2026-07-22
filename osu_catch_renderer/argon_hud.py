"""osu!lazer **Argon** HUD counters — the osu!STANDARD renderer's Argon
score / accuracy / combo components, ported VERBATIM onto catch's CPU/PIL
compositing pipeline (owner directive 2026-07: "use the STD hud, it's
essentially the same — only the platter and fruit differ per mode").

Source of truth (constants, layout, animation — keep in sync):
  osu_std_renderer/render/hud.py       ArgonScoreCounter/_argon_score_block,
                                       ArgonAccuracyCounter/_argon_accuracy,
                                       ArgonComboCounter/_argon_combo,
                                       _argon_seg_run, _argon_progress,
                                       argon_combo_scale_at, rolled(),
                                       layout_run(), the ARGON_* constants
  osu_std_renderer/render/textures.py  bake_wedge, bake_pill, bake_glyphs,
                                       ARGON_GLYPH_CAP_SCALE (the segment
                                       font itself is already ported in
                                       catch's argon_counter.py)

Differences from STD (deliberate, catch has no equivalent data):
  * score/acc/combo values come from catch's CatchSim scene (kept wiring);
    the 250 ms OutQuad RollingCounter tween is replayed STATEFULLY here
    (frames render in monotonic time order) instead of from an event list.
  * the PP counter is catch's own house element (STD gates it behind a
    settings flag too) — drawn under the accuracy block in the same font.
Everything is procedural (segment glyphs, wedge, pills) or the bundled OFL
Nunito — a missing font falls back to catch's fonts.font resolver, so a
stripped checkout still renders.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageFont

from .argon_counter import (argon_digit_advance, argon_glyph_rgba,
                            argon_wireframe_dot_rgba, argon_wireframe_rgba)

# --- constants (STD render/hud.py — values straight from ppy/osu Argon*) -----
LAZER_UI_HEIGHT = 768.0        # lazer's HUD space (DrawSizePreserving 768)
BLUE0 = (0x99 / 255.0, 0xDD / 255.0, 0xFF / 255.0)      # OsuColour.Blue0

ARGON_ROLL_MS = 250.0          # Argon counters' RollingDuration
ARGON_DIGIT_H = 30.0           # argon-counter glyph height (240 px × 0.125)
ARGON_WIRE_ALPHA = 0.25        # WireframeOpacity default
ARGON_LABEL_H = 12.0           # OsuFont.Torus 12 bold labels
ARGON_LABEL_GAP = 12.0         # NumberContainer.Y with a label
ARGON_SCORE_RIGHT_X = 250.0    # score right edge (components_x_offset+200)
ARGON_SCORE_TOP_Y = 50.0       # wedge2.y(20) + 30
ARGON_SCORE_DIGITS = 6         # GameplayScoreCounter RequiredDisplayDigits
ARGON_ACC_POS = (-20.0, 20.0)  # TopRight
ARGON_COMBO_POS = (36.0, -66.0)          # BottomLeft (ruleset layout)
ARGON_COMBO_SCALE = 1.3
ARGON_POP_UP = 1.1             # scale factor on increment
ARGON_POP_DOWN = 0.8           # on any decrease
ARGON_POP_MIN, ARGON_POP_MAX = 0.6, 1.4
ARGON_POP_MS = 500.0
ARGON_MISS_FLASH_MS = 2000.0

ARGON_PROGRESS_BAR_H = 10.0
ARGON_PROGRESS_WIDTH = 0.9     # Scale = (0.9, 1)
ARGON_PROGRESS_BOTTOM = 10.0   # Position (0, -padding)
ARGON_INFO_H = 14.0
GRAPH_BUCKETS = 100            # display_granularity 200, halved (quad budget)
GRAPH_TIERS = 5

# ArgonKeyCounter(Display) — STD render/hud.py values (straight from
# ppy/osu ArgonKeyCounter.cs / ArgonKeyCounterDisplay.cs, scale_factor 1.5)
ARGON_KEY_W, ARGON_KEY_H = 52.5, 45.0    # 35×30 × 1.5
ARGON_KEY_SPACING = 2.0
ARGON_KEY_LINE_H = 4.5                   # line_height 3 × 1.5
ARGON_KEY_PRESS_OFFSET = 4.0             # indicator drop on press (lazer px)
ARGON_KEY_NAME_H = 15.0                  # 10 × 1.5
ARGON_KEY_COUNT_H = 21.0                 # 14 × 1.5
ARGON_KEYS_POS = (-60.0, -66.0)          # BottomRight (parity with STD)
# lazer's catch key counters use the generic KeyCounterActionTrigger label
# "B{(int)action + 1}" (CatchAction: MoveLeft=0, MoveRight=1, Dash=2)
CATCH_KEY_LABELS = ("B1", "B2", "B3")

# ArgonHealthDisplay placement (the bar itself is catch's argon_health.py)
HP_POS = (50.0, 20.0)
HP_WIDTH = 300.0
HP_LINE_Y = 30.0               # healthLine: health.y + MAIN_PATH_RADIUS
HP_LINE_SIZE = (45.0, 3.0)

# ArgonWedgePiece (STD render/textures.py)
WEDGE_W = 380.0
WEDGE_H = 72.0
WEDGE_SHEAR = 0.8              # osu!framework Shear = (0.8, 0)
WEDGE_R = 10.0                 # CornerRadius (pre-shear space)
WEDGE_COLOR = (0x66, 0xCC, 0xFF)   # AccentColour #66CCFF

_AA_PX = 1.5                   # texture-space anti-alias band
_TRACKING = 0.05               # procedural-glyph letter-spacing

# Nunito label font (the lazer-Torus stand-in; see STD textures.py) — the
# requested height is multiplied by DejaVu/Nunito cap-fill so the VISIBLE
# text size matches STD's exactly.
DEJAVU_GLYPH_CAP_FILL = 0.7154
ARGON_GLYPH_CAP_FILL = 0.7025
ARGON_GLYPH_CAP_SCALE = DEJAVU_GLYPH_CAP_FILL / ARGON_GLYPH_CAP_FILL
ARGON_FONT_WEIGHT = 500        # Nunito Medium (variable-font `wght` axis)
NUNITO_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "assets", "fonts", "Nunito[wght].ttf"))
LABEL_BAKE_H = 128             # native bake height (STD DIGIT_HEIGHT)
# STD bakes its label bank over this exact charset; the union vertical
# extent is shared per bank, so the charset must match for identical aspects.
HUD_CHARSET = "0123456789.%x,:-!ABCDEFGHIJKLMNOPQRSTUVWXYZp×/"

GRADE_COLORS = {               # STD hud.py GRADE_COLORS (procedural badge)
    "SS": (0.94, 0.86, 0.47),
    "S": (0.94, 0.86, 0.47),
    "A": (0.43, 0.86, 0.51),
    "B": (0.43, 0.71, 0.86),
    "C": (0.78, 0.51, 0.86),
    "D": (0.86, 0.43, 0.43),
    "F": (1.0, 0.353, 0.353),
}


# --- easing / tween (osu!framework Easing.*) ---------------------------------

def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def ease_out_quad(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) * (1.0 - p)


def ease_out_quint(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 5


def ease_out_quart(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 4


def rolled(prev: float, target: float, age_ms: float,
           roll_ms: float = ARGON_ROLL_MS, ease=ease_out_quad) -> float:
    """RollingCounter<T> value tween: prev → target over roll_ms after a
    change, eased (Argon counters: 250 ms Easing.Out(Quad))."""
    if age_ms >= roll_ms or roll_ms <= 0:
        return target
    if age_ms <= 0:
        return prev
    return prev + (target - prev) * ease(age_ms / roll_ms)


def _fmt_time(seconds: float) -> str:
    """SongProgressInfo.formatTime: m:ss, leading '-' when negative."""
    neg = seconds < 0
    s = abs(seconds)
    m = int(s // 60)
    return f"{'-' if neg else ''}{m}:{int(s % 60):02d}"


def layout_run(text: str, aspects: dict, height: float,
               tracking: float = _TRACKING,
               mono_advance: float | None = None):
    """STD hud.layout_run: [(char, center_x_from_run_left, draw_width)],
    total_width. mono_advance (aspect units): digits occupy a fixed cell."""
    gap = tracking * height
    x = 0.0
    out = []
    for ch in text:
        if ch == " ":
            x += 0.45 * height
            continue
        w = aspects.get(ch, 0.6) * height
        cell = w
        if mono_advance is not None and ch.isdigit():
            cell = mono_advance * height
        out.append((ch, x + cell / 2.0, w))
        x += cell + gap
    total = max(x - gap, 0.0) if out or text else 0.0
    return out, total


def density_buckets(starts, ends) -> list:
    """STD hud._density_buckets — ArgonSongProgressGraph.Objects: object
    density over GRAPH_BUCKETS buckets (each object spans start..end)."""
    if not starts:
        return []
    first, last = min(starts), max(ends)
    interval = (last - first + 1) / GRAPH_BUCKETS
    vals = [0] * GRAPH_BUCKETS
    for s, e in zip(starts, ends):
        i0 = int((s - first) / interval)
        i1 = int((e - first) / interval)
        for i in range(max(i0, 0), min(i1, GRAPH_BUCKETS - 1) + 1):
            vals[i] += 1
    return vals


# --- procedural bakes (STD render/textures.py ports) --------------------------

def bake_wedge(scale: float = 1.0) -> np.ndarray:
    """ArgonWedgePiece: a rounded rect sheared by (0.8, 0) with a vertical
    gradient of #66CCFF from alpha 0 (top) to 0.25 (bottom). Baked in the
    SHEARED frame (canvas width = W + 0.8·H) so it pastes axis-aligned;
    masking (corner radius) applies pre-shear, as upstream."""
    w = int(round((WEDGE_W + WEDGE_SHEAR * WEDGE_H) * scale))
    h = int(round(WEDGE_H * scale))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64) / scale
    # unshear: canvas x = u - 0.8·y + 0.8·H  (u = pre-shear x)
    u = xx + WEDGE_SHEAR * yy - WEDGE_SHEAR * WEDGE_H
    r = WEDGE_R
    qx = np.abs(u - WEDGE_W / 2.0) - (WEDGE_W / 2.0 - r)
    qy = np.abs(yy - WEDGE_H / 2.0) - (WEDGE_H / 2.0 - r)
    d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
        + np.minimum(np.maximum(qx, qy), 0.0) - r
    shape = np.clip(-d / (_AA_PX / scale), 0.0, 1.0)
    grad = 0.25 * np.clip(yy / WEDGE_H, 0.0, 1.0)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = WEDGE_COLOR[0]
    rgba[..., 1] = WEDGE_COLOR[1]
    rgba[..., 2] = WEDGE_COLOR[2]
    rgba[..., 3] = np.round(shape * grad * 255.0).astype(np.uint8)
    return rgba


def bake_pill_alpha(width: int, height: int) -> np.ndarray:
    """STD bake_pill (alpha field only): fully-rounded bar, radius = h/2 —
    the RoundedBar / BoxElement / healthLine stand-in."""
    width = max(1, int(width))
    height = max(1, int(height))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    r = height / 2.0 - 1.0
    qx = np.abs(xx - (width - 1) / 2.0) - (width / 2.0 - 1.0 - r)
    qy = np.abs(yy - (height - 1) / 2.0) - (height / 2.0 - 1.0 - r)
    d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
        + np.minimum(np.maximum(qx, qy), 0.0) - r
    return np.clip(-d / _AA_PX, 0.0, 1.0)


def _load_label_font(px: int):
    """The bundled OFL Nunito pinned to wght 500 (STD _load_argon_font);
    falls back to catch's resolver so a stripped checkout still renders."""
    try:
        f = ImageFont.truetype(NUNITO_PATH, max(int(px), 6))
    except OSError:
        from .fonts import font as _catch_font
        return _catch_font(max(int(px), 6))
    try:
        f.set_variation_by_axes([ARGON_FONT_WEIGHT])
    except Exception:  # noqa: BLE001 — non-variable build → default face
        pass
    return f


def bake_label_glyphs(chars: str = HUD_CHARSET,
                      height: int = LABEL_BAKE_H) -> dict:
    """STD textures.bake_glyphs (Nunito loader): each char as an alpha
    bitmap, all sharing one vertical extent (union of glyph bboxes) so
    runs baseline-align when centred."""
    font = _load_label_font(int(height * 0.95))
    boxes = {}
    for ch in chars:
        try:
            boxes[ch] = font.getbbox(ch)
        except AttributeError:      # ancient PIL bitmap font
            w, h = font.getsize(ch)  # type: ignore[attr-defined]
            boxes[ch] = (0, 0, w, h)
    top = min(b[1] for b in boxes.values())
    bottom = max(b[3] for b in boxes.values())
    pad = 4
    out = {}
    from PIL import ImageDraw
    for ch in chars:
        x0, _, x1, _ = boxes[ch]
        w = (x1 - x0) + 2 * pad
        h = (bottom - top) + 2 * pad
        img = Image.new("L", (max(w, 1), max(h, 1)), 0)
        ImageDraw.Draw(img).text((pad - x0, pad - top), ch, font=font,
                                 fill=255)
        out[ch] = img
    return out


# --- stateful animation replays ------------------------------------------------

class _Rolling:
    """RollingCounter tween replayed statefully (frames are monotonic)."""
    __slots__ = ("prev", "target", "t0")

    def __init__(self, v: float = 0.0):
        self.prev = self.target = float(v)
        self.t0 = -1e18

    def at(self, t: float, value: float) -> float:
        v = float(value)
        if v != self.target:
            self.prev = rolled(self.prev, self.target, t - self.t0)
            self.target = v
            self.t0 = t
        return rolled(self.prev, self.target, t - self.t0)


class _ComboAnim:
    """ArgonComboCounter's NumberContainer pop (×1.1 up / ×0.8 down, clamp
    [0.6, 1.4], ease back OutQuint over 500 ms — 2000 ms on a combo-breaking
    miss) + the FlashColour(Red, 2000, OutQuint), replayed statefully."""
    __slots__ = ("last", "scale", "since", "dur", "break_t")

    def __init__(self):
        self.last = 0
        self.scale = 1.0
        self.since = None
        self.dur = ARGON_POP_MS
        self.break_t = None

    def observe(self, t: float, combo: int) -> None:
        if combo == self.last:
            return
        old, new = self.last, combo
        eased = self.value(t)
        was_miss = old > 1 and new == 0
        factor = ARGON_POP_UP if new > old else ARGON_POP_DOWN
        self.scale = min(max(eased * factor, ARGON_POP_MIN), ARGON_POP_MAX)
        self.since = t
        self.dur = ARGON_MISS_FLASH_MS if was_miss else ARGON_POP_MS
        # STD flashes only while the LAST combo change is a break
        self.break_t = t if was_miss else None
        self.last = combo

    def value(self, t: float) -> float:
        if self.since is None:
            return 1.0
        p = ease_out_quint((t - self.since) / self.dur)
        return self.scale + (1.0 - self.scale) * p

    def flash_color(self, t: float):
        if self.break_t is None:
            return (1.0, 1.0, 1.0)
        p = 1.0 - ease_out_quint((t - self.break_t) / ARGON_MISS_FLASH_MS)
        return (1.0, 1.0 - 0.75 * p, 1.0 - 0.75 * p)


class _KeyCounter:
    """KeyCounterController replayed statefully (frames arrive in monotonic
    time order): per-key held state, running press COUNT (incremented on
    each rising edge) and the last press/release times that drive the
    ArgonKeyCounter indicator/name tweens."""
    __slots__ = ("held", "counts", "press_t", "release_t")

    def __init__(self, n: int = 3):
        self.held = [False] * n
        self.counts = [0] * n
        self.press_t: list = [None] * n
        self.release_t: list = [None] * n

    def observe(self, t: float, held) -> None:
        for i, h in enumerate(held):
            h = bool(h)
            if h and not self.held[i]:
                self.counts[i] += 1
                self.press_t[i] = t
            elif self.held[i] and not h:
                self.release_t[i] = t
            self.held[i] = h


# --- the HUD ---------------------------------------------------------------------

class ArgonHud:
    """Draws STD's Argon score / accuracy / combo counters (+ wedges, the
    healthLine dash and the Argon song-progress strip) onto a PIL RGB image.
    All layout math is in LAZER px (768-high space) × ``lk`` device px, the
    exact expressions from STD's hud.py."""

    def __init__(self, w: int, h: int, first_ms: float, last_ms: float,
                 density=None, hud_scale: float = 1.0,
                 hud_opacity: float = 1.0):
        self.w, self.h = int(w), int(h)
        self.lk = h / LAZER_UI_HEIGHT      # device px per lazer px
        self.ui_w_l = w / self.lk          # screen width in lazer px
        self.es = float(hud_scale)
        self.op = float(hud_opacity)
        self.first_t = float(first_ms)
        self.last_t = max(float(last_ms), self.first_t + 1.0)

        # segment cells (argon_counter.py — REAL argon-counter-*.png sprites
        # when argon_assets/ ships them, procedural 7-segment bakes otherwise).
        # The fixed-width digit advance is DERIVED from the real glyph's aspect
        # (STD's argon_seg_advance = aspect['8']): ~1.0 square with the ripped
        # sprites (the approved wider look that matches STD), 0.55 with the
        # procedural fallback. Anchors/heights below read seg_advance
        # dynamically, so the run just widens — placement is unchanged.
        self.seg_advance = argon_digit_advance()
        self._alpha: dict = {}
        for ch in "0123456789":
            self._alpha[f"lit_{ch}"] = Image.fromarray(
                argon_glyph_rgba(ch)[..., 3], "L")
        for ch in ".%x":
            self._alpha[f"lit_{ch}"] = Image.fromarray(
                argon_glyph_rgba(ch)[..., 3], "L")
        self._alpha["wire"] = Image.fromarray(
            argon_wireframe_rgba()[..., 3], "L")
        self._alpha["wire_dot"] = Image.fromarray(
            argon_wireframe_dot_rgba()[..., 3], "L")

        # Nunito label bank (STD aglyph_*): aspects for layout_run
        self.label_aspect: dict = {}
        for ch, im in bake_label_glyphs().items():
            self._alpha[f"lab_{ch}"] = im
            self.label_aspect[ch] = im.width / im.height
        self.label_mono = max(self.label_aspect[c] for c in "0123456789")

        # wedge baked once at device scale (STD bakes ×1 and GPU-scales;
        # same geometry, crisper AA)
        self._wedge = Image.fromarray(bake_wedge(self.es * self.lk), "RGBA")
        if self.op < 1.0:
            a = np.asarray(self._wedge).copy()
            a[..., 3] = (a[..., 3] * self.op).astype(np.uint8)
            self._wedge = Image.fromarray(a, "RGBA")

        # tinted/scaled sprite cache
        self._cache: dict = {}

        # rolling counters + combo pop/flash state
        self._score = _Rolling(0.0)
        self._acc = _Rolling(1.0)
        self._combo = _Rolling(0.0)
        self._combo_anim = _ComboAnim()

        # key counter state (B1/B2/B3) + the cached indicator-pill alpha
        self._keys = _KeyCounter(len(CATCH_KEY_LABELS))
        self._key_pill_a = None

        # Argon progress strip statics (density graph is per-render constant)
        self._progress_static(density or [])

    # -- sprite cache ------------------------------------------------------------

    def _cell(self, name: str, w_px: float, h_px: float, color,
              alpha: float) -> Image.Image | None:
        w = max(1, int(round(w_px)))
        h = max(1, int(round(h_px)))
        r = int(round(_clamp01(color[0]) * 255))
        g = int(round(_clamp01(color[1]) * 255))
        b = int(round(_clamp01(color[2]) * 255))
        a8 = int(round(_clamp01(alpha) * 255))
        if a8 == 0:
            return None
        key = (name, w, h, r, g, b, a8)
        cache = self._cache
        hit = cache.get(key)
        if hit is not None:
            # LRU refresh (dicts iterate in insertion order): re-inserting on
            # hit keeps hot glyphs alive, so the eviction below can't dump the
            # per-frame digit cells the way the old wholesale clear() did.
            del cache[key]
            cache[key] = hit
            return hit
        base = self._alpha.get(name)
        if base is None:
            return None
        am = np.asarray(base.resize((w, h), Image.LANCZOS), dtype=np.uint16)
        out = np.zeros((h, w, 4), np.uint8)
        out[..., 0] = r
        out[..., 1] = g
        out[..., 2] = b
        out[..., 3] = (am * a8 // 255).astype(np.uint8)
        im = Image.fromarray(out, "RGBA")
        if len(cache) >= 4096:           # combo pops make continuous sizes
            cache.pop(next(iter(cache)))     # evict least-recently-used
        cache[key] = im
        return im

    def _paste_center(self, img, cell, cx_px: float, cy_px: float) -> None:
        img.paste(cell, (int(round(cx_px - cell.width / 2.0)),
                         int(round(cy_px - cell.height / 2.0))), cell)

    # -- Argon 7-segment runs (STD _argon_seg_run / _argon_seg_width) -------------

    def _seg_width(self, n: int, h_l: float) -> float:
        return self.seg_advance * h_l * n

    def _seg_run(self, img, text: str, right_x_l: float, top_y_l: float,
                 h_l: float, alpha: float, color=(1, 1, 1),
                 wire_n: int | None = None, scale: float = 1.0,
                 pivot=None) -> float:
        """Right-aligned Argon counter run: the unlit all-segments '8' (or
        dot) wireframe backing at WireframeOpacity 0.25 under each cell +
        the lit glyph in the SAME cell rect, so lit + ghost register by
        construction. wire_n adds leading unlit cells; scale/pivot drive
        the combo pop. Returns the full run width (lazer px)."""
        cw = self.seg_advance * h_l
        n_lit = len(text)
        n_wire = max(wire_n if wire_n is not None else n_lit, n_lit)
        cy = top_y_l + h_l / 2.0
        if pivot is None:
            pivot = (right_x_l - n_wire * cw, cy)
        wire_a = ARGON_WIRE_ALPHA * self.op
        lk = self.lk
        gy = pivot[1] + (cy - pivot[1]) * scale
        w_px = cw * scale * lk
        h_px = h_l * scale * lk
        for i in range(n_wire):
            cx = right_x_l - (n_wire - i) * cw + cw / 2.0
            gx = pivot[0] + (cx - pivot[0]) * scale
            lit_idx = i - (n_wire - n_lit)
            ch = text[lit_idx] if 0 <= lit_idx < n_lit else None
            wkey = "wire_dot" if ch == "." else "wire"
            wcell = self._cell(wkey, w_px, h_px, color, wire_a)
            if wcell is not None:
                self._paste_center(img, wcell, gx * lk, gy * lk)
            if ch is not None:
                lcell = self._cell(f"lit_{ch}", w_px, h_px, color, alpha)
                if lcell is not None:
                    self._paste_center(img, lcell, gx * lk, gy * lk)
        return n_wire * cw

    # -- Nunito label runs (STD _lrun / _lrun_width) --------------------------------

    def _lrun(self, img, text: str, x_l: float, y_l: float, h_l: float,
              color, alpha: float, mono: bool = False) -> float:
        hd = h_l * ARGON_GLYPH_CAP_SCALE
        entries, total = layout_run(
            text, self.label_aspect, hd,
            mono_advance=(self.label_mono if mono else None))
        lk = self.lk
        cy = (y_l + h_l / 2.0) * lk
        for ch, cx, wch in entries:
            cell = self._cell(f"lab_{ch}", wch * lk, hd * lk, color, alpha)
            if cell is not None:
                self._paste_center(img, cell, (x_l + cx) * lk, cy)
        return total

    def _lrun_width(self, text: str, h_l: float, mono: bool = False) -> float:
        hd = h_l * ARGON_GLYPH_CAP_SCALE
        _, total = layout_run(
            text, self.label_aspect, hd,
            mono_advance=(self.label_mono if mono else None))
        return total

    # -- components (STD draw order: wedges+score, accuracy, combo) -----------------

    def draw_wedges(self, img) -> None:
        """The two ArgonWedgePiece backdrops (ArgonSkin.cs: 380×72 at
        (-50,15)/(-46,20)); canvas includes the 0.8·H shear overhang left."""
        k = self.es * self.lk
        for px, py in ((-50.0, 15.0), (-46.0, 20.0)):
            img.paste(self._wedge,
                      (int(round((px - WEDGE_SHEAR * WEDGE_H) * k)),
                       int(round(py * k))), self._wedge)

    def draw_score(self, img, t: float, score: int) -> None:
        """ArgonScoreCounter: right edge x=250, y=50, ShowLabel false, 6
        wireframe digits minimum — the count GROWS with the score."""
        es = self.es
        val = int(round(self._score.at(t, float(score))))
        text = str(max(val, 0))
        self._seg_run(img, text, ARGON_SCORE_RIGHT_X * es,
                      ARGON_SCORE_TOP_Y * es, ARGON_DIGIT_H * es, self.op,
                      wire_n=max(ARGON_SCORE_DIGITS, len(text)))

    def draw_accuracy(self, img, t: float, acc: float,
                      grade: str | None = None) -> None:
        """ArgonAccuracyCounter: TopRight (-20, 20) — whole part + '.dd' at
        ×0.5 (bottom-aligned) + a FULL-height '%', 'ACCURACY' label in
        Blue0, wireframes ###/.##/# (+ STD's procedural grade badge left
        of the label when a grade is passed)."""
        es = self.es
        disp = round(self._acc.at(t, float(acc)) * 100.0, 2)
        whole = int(disp)
        frac = int(round((disp - whole) * 100))
        whole_txt, frac_txt = str(whole), f".{frac:02d}"
        h = ARGON_DIGIT_H * es
        hh = h * 0.5                       # fractionPart Scale = 0.5
        hpct = h                           # percentText: FULL digit height
        right = self.ui_w_l + ARGON_ACC_POS[0] * es
        top = ARGON_ACC_POS[1] * es
        num_top = top + ARGON_LABEL_GAP * es
        w_pct = self._seg_width(1, hpct)
        self._seg_run(img, "%", right, num_top, hpct, 0.95 * self.op)
        frac_right = right - w_pct - 2.0 * es
        w_frac = self._seg_width(len(frac_txt), hh)
        self._seg_run(img, frac_txt, frac_right, num_top + (h - hh), hh,
                      0.95 * self.op)
        whole_right = frac_right - w_frac - 1.0 * es
        w_whole = self._seg_run(img, whole_txt, whole_right, num_top, h,
                                0.95 * self.op, wire_n=3)
        label_left = whole_right - w_whole
        self._lrun(img, "ACCURACY", label_left, top, ARGON_LABEL_H * es,
                   BLUE0, 0.95 * self.op)
        if grade:
            gh = 24.0 * es
            color = GRADE_COLORS.get(grade, (0.8, 0.8, 0.85))
            gw = self._lrun_width(grade, gh)
            self._lrun(img, grade, label_left - 14.0 * es - gw,
                       num_top + h / 2.0 - gh / 2.0, gh, color,
                       0.95 * self.op)

    def draw_combo(self, img, t: float, combo: int) -> None:
        """ArgonComboCounter: BottomLeft (36, -66) ×1.3 — '<n>x' with the
        'COMBO' label, pop ×1.1/×0.8 clamp [0.6,1.4] eased OutQuint,
        FlashColour red over 2 s on a combo-breaking miss, ≥2 cells."""
        es = self.es
        self._combo_anim.observe(t, int(combo))
        displayed = int(round(self._combo.at(t, float(combo))))
        text = f"{displayed}x"
        sc = ARGON_COMBO_SCALE * es
        h = ARGON_DIGIT_H * sc
        label_h = ARGON_LABEL_H * sc
        block_h = (ARGON_LABEL_GAP + ARGON_DIGIT_H) * sc
        x = ARGON_COMBO_POS[0] * es
        bottom = LAZER_UI_HEIGHT + ARGON_COMBO_POS[1] * es
        top = bottom - block_h
        num_top = top + ARGON_LABEL_GAP * sc
        color = self._combo_anim.flash_color(t)
        scale = self._combo_anim.value(t)
        pivot = (x, num_top)               # NumberContainer scales TopLeft
        cw = self.seg_advance * h
        n_cells = max(2, len(text))        # DisplayXSymbol 2-cell floor
        right_x = x + n_cells * cw
        self._seg_run(img, text, right_x, num_top, h, 0.95 * self.op,
                      color=color, scale=scale, pivot=pivot, wire_n=n_cells)
        self._lrun(img, "COMBO", x, top, label_h, BLUE0, 0.95 * self.op)

    def draw_pp(self, img, pp: float) -> None:
        """Catch's house pp counter, restyled to the Argon block grammar:
        'PP' label + a 0.6-height segment run under the accuracy block."""
        es = self.es
        text = f"{max(pp, 0.0):.0f}"
        h = ARGON_DIGIT_H * es * 0.6
        right = self.ui_w_l + ARGON_ACC_POS[0] * es
        top = (ARGON_ACC_POS[1] + ARGON_LABEL_GAP + ARGON_DIGIT_H
               + 10.0) * es
        num_top = top + ARGON_LABEL_GAP * es
        self._seg_run(img, text, right, num_top, h, 0.95 * self.op)
        lw = self._lrun_width("PP", ARGON_LABEL_H * es)
        self._lrun(img, "PP", right - lw, top, ARGON_LABEL_H * es, BLUE0,
                   0.95 * self.op)

    def draw_key_counter(self, img, t: float, held, counts=None) -> None:
        """ArgonKeyCounterDisplay (STD _argon_key_overlay, ported 1:1):
        horizontal row, BottomRight at (-60, -66); per counter the
        indicator pill drops 4 px on press (60 ms OutQuint) and returns
        over 250 ms OutQuart, the key name flashes white while held (back
        to Blue0 over 200 ms OutQuart), the running count sits below.
        Catch keys: B1 = move left, B2 = move right, B3 = dash (lazer's
        generic KeyCounterActionTrigger labels)."""
        ks = self._keys
        ks.observe(t, held)
        # authoritative press counts from the sim's replay-frame timeline
        # (rapid taps within one video frame otherwise vanish from the count)
        if counts is not None:
            ks.counts = [int(c) for c in counts]
        es, lk = self.es, self.lk
        n = len(CATCH_KEY_LABELS)
        w_cell, h_cell = ARGON_KEY_W * es, ARGON_KEY_H * es
        total_w = n * w_cell + (n - 1) * ARGON_KEY_SPACING * es
        right = self.ui_w_l + ARGON_KEYS_POS[0] * es
        bottom = LAZER_UI_HEIGHT + ARGON_KEYS_POS[1] * es
        x0 = right - total_w
        top = bottom - h_cell
        ind_h = ARGON_KEY_LINE_H * es
        # indicator pill alpha field is size-constant — bake once
        w_px = max(1, int(round(w_cell * lk)))
        h_px = max(1, int(round(ind_h * lk)))
        if self._key_pill_a is None:
            self._key_pill_a = bake_pill_alpha(w_px, h_px)
        for ch in range(n):
            cx0 = x0 + ch * (w_cell + ARGON_KEY_SPACING * es)
            pressed = ks.held[ch]
            press_t = ks.press_t[ch]
            release_t = ks.release_t[ch]
            # indicator pill: y offset + alpha (STD expressions verbatim)
            if pressed and press_t is not None:
                age = t - press_t
                dy = ARGON_KEY_PRESS_OFFSET * ease_out_quint(age / 60.0)
                ind_alpha = 0.5 + 0.5 * _clamp01(age / 10.0)
                name_white = _clamp01(age / 10.0)
            else:
                age = t - release_t if release_t is not None else math.inf
                p = ease_out_quart(age / 250.0)
                dy = ARGON_KEY_PRESS_OFFSET * (1.0 - p)
                ind_alpha = 1.0 - 0.5 * p
                name_white = (1.0 - ease_out_quart(age / 200.0)
                              if release_t is not None else 0.0)
            a = self._key_pill_a * (ind_alpha * self.op)
            rgba = np.full((h_px, w_px, 4), 255, np.uint8)
            rgba[..., 3] = np.round(a * 255.0).astype(np.uint8)
            self._paste_center(img, Image.fromarray(rgba, "RGBA"),
                               (cx0 + w_cell / 2.0) * lk,
                               (top + dy * es + ind_h / 2.0) * lk)
            name_col = tuple(BLUE0[i] + (1.0 - BLUE0[i]) * name_white
                             for i in range(3))
            pad_top = ind_h + ARGON_KEY_PRESS_OFFSET * es
            self._lrun(img, CATCH_KEY_LABELS[ch], cx0 + 3.0 * es,
                       top + pad_top + 2.0 * es, ARGON_KEY_NAME_H * es,
                       name_col, 0.95 * self.op)
            self._lrun(img, f"{ks.counts[ch]:,}", cx0 + 3.0 * es,
                       bottom - ARGON_KEY_COUNT_H * es - 1.0 * es,
                       ARGON_KEY_COUNT_H * es, (1, 1, 1), 0.95 * self.op,
                       mono=True)

    def draw_health_line(self, img) -> None:
        """The BoxElement healthLine (CornerRadius .5): (0, 30), 45×3."""
        es, lk = self.es, self.lk
        lw, lh = HP_LINE_SIZE
        # PERF: the pill is size/opacity-constant per render — bake it once.
        pill = getattr(self, "_health_line_pill", None)
        if pill is None:
            w_px = max(1, int(round(lw * es * lk)))
            h_px = max(1, int(round(lh * es * lk)))
            a = bake_pill_alpha(w_px, h_px) * self.op
            rgba = np.full((h_px, w_px, 4), 255, np.uint8)
            rgba[..., 3] = np.round(a * 255.0).astype(np.uint8)
            pill = self._health_line_pill = Image.fromarray(rgba, "RGBA")
        self._paste_center(img, pill, (lw / 2.0) * es * lk,
                           HP_LINE_Y * es * lk)

    # -- ArgonSongProgress (STD _argon_progress) -------------------------------------

    def _progress_static(self, density) -> None:
        """Precompute the per-render constants of the 90 %-width bottom
        strip: geometry, the additive density-graph layer and the
        background pill (bg gray(0.2) α.3)."""
        es, lk = self.es, self.lk
        self._bar_h = ARGON_PROGRESS_BAR_H * es
        self._pw = self.ui_w_l * ARGON_PROGRESS_WIDTH
        self._px0 = (self.ui_w_l - self._pw) / 2.0
        self._pbottom = LAZER_UI_HEIGHT - ARGON_PROGRESS_BOTTOM * es
        # device-px strip rect [x0, y0, x1, y1)
        self._sx0 = int(round(self._px0 * lk))
        self._sx1 = int(round((self._px0 + self._pw) * lk))
        self._sy1 = int(round(self._pbottom * lk))
        self._sy0 = int(round((self._pbottom - self._bar_h) * lk))
        sw = max(1, self._sx1 - self._sx0)
        sh = max(1, self._sy1 - self._sy0)
        # additive graph layer (column alpha ∝ filled tiers, gray .2 α.45)
        add = np.zeros((sh, sw), np.float32)
        if density:
            vmax = max(density) or 1
            bw = sw / len(density)
            for i, v in enumerate(density):
                if v <= 0:
                    continue
                tiers = max(1, round(v / vmax * GRAPH_TIERS))
                gh = int(round(sh * tiers / GRAPH_TIERS))
                xa = int(round(i * bw + bw * 0.05))
                xb = int(round(i * bw + bw * 0.95))
                if xb > xa and gh > 0:
                    add[sh - gh:, xa:xb] = 0.45
        self._graph_add = (add * 0.2 * 255.0 * self.op)[..., None]
        self._pill_bg = bake_pill_alpha(sw, sh) * (0.3 * self.op)
        # PERF hoists for draw_progress: the broadcast view of the bg pill is
        # constant; the fill pill is cached per width (frac moves ~1px/frame,
        # so consecutive frames usually reuse the previous bake).
        self._pill_bg_b = self._pill_bg[..., None]
        self._fill_pill: tuple | None = None       # (fw, alpha-with-axis)

    def draw_progress(self, img, t: float) -> None:
        """The Argon progress strip: faint density graph (additive), the
        rounded bar (bg gray(0.2) α.3, fill gray(0.9) α.95) and the
        elapsed / remaining info row."""
        es, lk = self.es, self.lk
        sx0, sx1, sy0, sy1 = self._sx0, self._sx1, self._sy0, self._sy1
        region = np.asarray(
            img.crop((sx0, sy0, sx1, sy1)), dtype=np.float32)
        # density graph — additive
        region += self._graph_add
        # bar background pill — over
        a = self._pill_bg_b
        region = region * (1.0 - a) + (0.2 * 255.0) * a
        # fill pill — over
        frac = (t - self.first_t) / max(self.last_t - self.first_t, 1.0)
        frac = _clamp01(frac) if t >= self.first_t else 0.0
        if frac > 0.003:
            fw = max(1, int(round((sx1 - sx0) * frac)))
            if self._fill_pill is None or self._fill_pill[0] != fw:
                fa = (bake_pill_alpha(fw, sy1 - sy0)
                      * (0.95 * self.op))[..., None]
                self._fill_pill = (fw, fa)
            fa = self._fill_pill[1]
            region[:, :fw] = (region[:, :fw] * (1.0 - fa)
                              + (0.9 * 255.0) * fa)
        img.paste(Image.fromarray(
            np.clip(region, 0.0, 255.0).astype(np.uint8), "RGB"),
            (sx0, sy0))
        # info row: elapsed (left) / remaining (right), mono digits
        info_h = ARGON_INFO_H * es
        info_y = self._pbottom - self._bar_h * 2.0 - info_h - 2.0 * es
        cur = _fmt_time((t - self.first_t) / 1000.0)
        # clamp remaining to >= 0 — frames after the last object briefly
        # flashed "-0:01" (2026-07-22 polish)
        left = _fmt_time(max(self.last_t - t, 0.0) / 1000.0)
        self._lrun(img, cur, self._px0, info_y, info_h, (1, 1, 1),
                   0.9 * self.op, mono=True)
        wl = self._lrun_width(left, info_h, mono=True)
        self._lrun(img, left, self._px0 + self._pw - wl, info_y, info_h,
                   (1, 1, 1), 0.9 * self.op, mono=True)
