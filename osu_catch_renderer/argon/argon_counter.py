"""osu!lazer **Argon** counter font (score / combo / accuracy / pp).

Prefers the REAL osu!lazer `argon-counter-*.png` sprites when they are present
in a live ``argon_assets/`` dir next to this module (copied from the
quarantined ``argon_assets.RIPPED_CCBYNC.bak/``) — so the HUD digits are
pixel-exact to lazer. When that dir is absent it falls back to drawing the
squared numerals **procedurally** as rounded 7-segment bars — the same
licence-clean technique the osu!STANDARD renderer uses
(osu_std_renderer/render/textures.py: `bake_argon_segment`,
`bake_wireframe_cell`, `bake_wireframe_dot`, `_ARGON_SEG_MAP`), so a stripped
checkout still renders. Either path yields the SAME fixed 132x240 cell, so the
HUD / font layout is byte-for-byte unchanged.

Mirrors `ArgonCounterTextComponent`:
  * fixed-width 132x240 cells (aspect 0.55), displayed at the requested
    cell height,
  * the LIT glyph and the UNLIT "wireframes" (all-segments '8') backing come
    from the SAME segment geometry on the SAME cell canvas, so lit + ghost
    register BY CONSTRUCTION — the dim ⊠ placeholder look for unlit digits
    (WireframeOpacity 0.25),
  * '.'/'%'/'x' are drawn procedurally in the same segment weight; the '.'
    slot gets its own small dot wireframe (matching STD) instead of a full '8'.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Procedural 7-segment glyph baking — ported verbatim from the STD renderer
# (osu_std_renderer/render/textures.py) so catch's counter is pixel-coherent
# with STD's. Pure geometry: needs no font and no external assets.
# ---------------------------------------------------------------------------

_AA_PX = 1.5                      # texture-space anti-alias band (STD value)

ARGON_SEG_W = 132                # one fixed-width digit cell (aspect 0.55)
ARGON_SEG_H = 240
# standard 7-segment map: A top, B upper-right, C lower-right, D bottom,
# E lower-left, F upper-left, G middle.
_ARGON_SEG_MAP = {
    "0": "ABCDEF", "1": "BC", "2": "ABGED", "3": "ABGCD", "4": "FGBC",
    "5": "AFGCD", "6": "AFGEDC", "7": "ABC", "8": "ABCDEFG", "9": "ABCDFG",
}


def _argon_seg_fields(width: int, height: int, seg_frac: float,
                      gap_frac: float):
    """(xx, yy, seg_alpha dict) for one segment cell — each of the seven
    segments A..G as a rounded-bar alpha field (shared by the lit glyphs
    and the all-segments wireframe backing)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    t = seg_frac * width                   # segment thickness
    gap = gap_frac * height
    m = t * 0.75                           # cell inset
    x0, x1 = m, width - m
    y0, ym, y1 = m, height / 2.0, height - m
    half_v = (ym - y0) / 2.0

    def hseg(cy: float) -> np.ndarray:
        qx = np.abs(xx - width / 2.0) - (x1 - x0 - 2 * t - 2 * gap) / 2.0
        qy = np.abs(yy - cy) - t / 2.0
        d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
            + np.minimum(np.maximum(qx, qy), 0.0) - t * 0.18
        return np.clip(-d / _AA_PX, 0.0, 1.0)

    def vseg(cx: float, cy: float) -> np.ndarray:
        qx = np.abs(xx - cx) - t / 2.0
        qy = np.abs(yy - cy) - (half_v - t / 2.0 - gap)
        d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
            + np.minimum(np.maximum(qx, qy), 0.0) - t * 0.18
        return np.clip(-d / _AA_PX, 0.0, 1.0)

    seg = {
        "A": hseg(y0 + t / 2.0), "G": hseg(ym), "D": hseg(y1 - t / 2.0),
        "F": vseg(x0 + t / 2.0, (y0 + ym) / 2.0),
        "B": vseg(x1 - t / 2.0, (y0 + ym) / 2.0),
        "E": vseg(x0 + t / 2.0, (ym + y1) / 2.0),
        "C": vseg(x1 - t / 2.0, (ym + y1) / 2.0),
    }
    return xx, yy, seg


def _argon_seg_char_alpha(char: str, xx, yy, seg,
                          width: int, height: int, seg_frac: float):
    """Alpha field for one glyph — a subset of the seven segments for a
    digit, or a procedural '.'/'%'/'x' in the same segment weight."""
    t = seg_frac * width
    if char in _ARGON_SEG_MAP:
        a = np.zeros((height, width))
        for s in _ARGON_SEG_MAP[char]:
            a = np.maximum(a, seg[s])
        return a
    if char in (".", "dot"):
        r = t * 0.85
        cx, cy = width / 2.0, height - t * 0.75 - r
        d = np.hypot(xx - cx, yy - cy)
        return np.clip((r - d) / _AA_PX, 0.0, 1.0)
    if char in ("%", "percent"):
        # A clean percent: two OPEN rings (upper-left + lower-right) joined by
        # a bold diagonal slash. A wider ring band keeps a clear centre hole so
        # the rings don't collapse to dots when the counter is small.
        rr = height * 0.135                # ring radius
        band = t * 0.5                     # ring HALF-thickness → open centre
        cxu, cyu = width * 0.31, height * 0.205
        cxl, cyl = width * 0.69, height * 0.795
        du = np.abs(np.hypot(xx - cxu, yy - cyu) - rr)
        dl = np.abs(np.hypot(xx - cxl, yy - cyl) - rr)
        ring = np.maximum(np.clip((band - du) / _AA_PX, 0.0, 1.0),
                          np.clip((band - dl) / _AA_PX, 0.0, 1.0))
        # diagonal slash from bottom-left to top-right
        mm = t * 0.95
        dirx, diry = (width - 2 * mm), -(height - 2 * mm)
        L = math.hypot(dirx, diry)
        nx, ny = -diry / L, dirx / L
        px, py = xx - width / 2.0, yy - height / 2.0
        dperp = np.abs(px * nx + py * ny)
        dalong = np.abs(px * (dirx / L) + py * (diry / L))
        slash = (np.clip((t * 0.42 - dperp) / _AA_PX, 0.0, 1.0)
                 * np.clip((L / 2.0 - dalong) / _AA_PX, 0.0, 1.0))
        return np.maximum(ring, slash)
    if char in ("x",):
        th = t * 0.60
        arm = height * 0.24
        px, py = xx - width / 2.0, yy - height * 0.5
        s2 = 1.0 / math.sqrt(2.0)
        d1 = np.abs((px - py) * s2)
        d2 = np.abs((px + py) * s2)
        reach = np.clip((arm - np.hypot(px, py)) / _AA_PX, 0.0, 1.0)
        return np.maximum(np.clip((th - d1) / _AA_PX, 0.0, 1.0),
                          np.clip((th - d2) / _AA_PX, 0.0, 1.0)) * reach
    return np.zeros((height, width))


def bake_argon_segment(char: str, width: int = ARGON_SEG_W,
                       height: int = ARGON_SEG_H, seg_frac: float = 0.14,
                       gap_frac: float = 0.045) -> np.ndarray:
    """A single lit Argon-counter glyph (white, tintable): digit segments
    or the '.'/'%'/'x' symbols, on the shared cell canvas."""
    xx, yy, seg = _argon_seg_fields(width, height, seg_frac, gap_frac)
    a = _argon_seg_char_alpha(char, xx, yy, seg, width, height, seg_frac)
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba


def bake_wireframe_cell(width: int = ARGON_SEG_W, height: int = ARGON_SEG_H,
                        seg_frac: float = 0.14,
                        gap_frac: float = 0.045) -> np.ndarray:
    """The all-segments '8' wireframe backing (WireframeOpacity 0.25). Same
    segment geometry as the lit glyphs, so a lit digit registers exactly on
    top of its own wireframe cell."""
    xx, yy, seg = _argon_seg_fields(width, height, seg_frac, gap_frac)
    a = np.zeros((height, width))
    for s in "ABCDEFG":
        a = np.maximum(a, seg[s])
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(a * 255.0).astype(np.uint8)
    return rgba


def bake_wireframe_dot(width: int = ARGON_SEG_W, height: int = ARGON_SEG_H,
                       seg_frac: float = 0.14) -> np.ndarray:
    """The '.' wireframe backing (a dot) — matches the lit '.' glyph."""
    xx, yy, seg = _argon_seg_fields(width, height, seg_frac, 0.045)
    a = _argon_seg_char_alpha(".", xx, yy, seg, width, height, seg_frac)
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Real Argon-counter sprites (osu!lazer's own argon-counter-*.png) — when a
# live ``argon_assets/`` dir ships them they REPLACE the procedural bakes so
# the digits are pixel-exact to lazer; otherwise the procedural bakes above are
# used unchanged. Mirrors the STD renderer's textures.py (_argon_counter_cell /
# argon_seg_advance): lazer's argon-counter is a FIXED-WIDTH *square* font —
# every digit ships on a 240x240 canvas, '.' on a narrow 52x240 canvas. We
# normalise every glyph onto ONE common square cell (ARGON_SEG_H x ARGON_SEG_H)
# by scaling to the cell HEIGHT and centring horizontally, RGB forced white
# (tintable), alpha preserved. So digits fill the square (aspect ~1.0, the
# approved wider look that matches STD), while the narrow '.' dot stays small
# and centred rather than stretching. The digit ADVANCE derives from the real
# glyph's aspect (argon_digit_advance() = aspect of '8' ≈ 1.0); the procedural
# fallback keeps its 132x240 (0.55) aspect, so a stripped checkout is unchanged.
# ---------------------------------------------------------------------------

_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "argon_assets")

# glyph char -> the argon-counter-<stem>.png stem
_SPRITE_STEM = {**{str(d): str(d) for d in range(10)},
                ".": "dot", "%": "percentage", "x": "x"}


def _real_sprite_path(stem: str) -> str:
    return os.path.join(_ASSET_DIR, f"argon-counter-{stem}.png")


def _load_real_sprite(stem: str):
    """Load a real argon-counter sprite as RGBA, or None if absent/unreadable
    (→ the caller falls back to the procedural bake)."""
    p = _real_sprite_path(stem)
    if not os.path.isfile(p):
        return None
    try:
        return Image.open(p).convert("RGBA")
    except Exception:  # noqa: BLE001 — a bad asset just falls back to procedural
        return None


_ARGON_COUNTER_CELL = ARGON_SEG_H          # 240 — common SQUARE cell (glyph height)


def _sprite_to_cell(img: Image.Image) -> np.ndarray:
    """Normalise a real argon-counter sprite onto the common SQUARE cell
    (ARGON_SEG_H x ARGON_SEG_H) as a WHITE (tintable) RGBA array — a 1:1 mirror
    of the STD renderer's _argon_counter_cell so catch shares STD's true
    fixed-width square metric (the approved wider Argon look). Scaled by HEIGHT
    to the cell, centred horizontally, RGB forced white, only the alpha kept:
    digits fill the square (aspect ~1.0), the narrow '.' dot stays small and
    centred rather than stretching to fill."""
    cell = _ARGON_COUNTER_CELL
    if img.height != cell:
        nw = max(1, round(img.width * cell / img.height))
        img = img.resize((nw, cell), Image.LANCZOS)
    if img.width > cell:                         # defensive (real art is <= cell)
        nh = max(1, round(img.height * cell / img.width))
        img = img.resize((cell, nh), Image.LANCZOS)
    src = np.asarray(img)
    h, w = src.shape[:2]
    rgba = np.zeros((cell, cell, 4), dtype=np.uint8)
    y0, x0 = (cell - h) // 2, (cell - w) // 2
    rgba[y0:y0 + h, x0:x0 + w, 3] = src[..., 3]
    rgba[..., 0:3] = 255                         # white → tint does the colour
    return rgba


def argon_glyph_rgba(char: str) -> np.ndarray:
    """A lit Argon-counter glyph as WHITE tintable RGBA: the real lazer sprite
    normalised onto the square cell when argon_assets/ ships it (aspect ~1.0),
    else the procedural 132x240 bake (aspect 0.55)."""
    stem = _SPRITE_STEM.get(char)
    if stem is not None:
        img = _load_real_sprite(stem)
        if img is not None:
            return _sprite_to_cell(img)
    return bake_argon_segment(char)


def argon_wireframe_rgba() -> np.ndarray:
    """The all-segments '8' wireframe backing: real 'wireframes' sprite, else
    the procedural bake."""
    img = _load_real_sprite("wireframes")
    if img is not None:
        return _sprite_to_cell(img)
    return bake_wireframe_cell()


def argon_wireframe_dot_rgba() -> np.ndarray:
    """The '.' wireframe backing. lazer ships no dedicated dot-wireframe, so the
    real dot sprite doubles as its own dim backing (a lit dot registers on top
    of it by construction); the procedural dot otherwise."""
    img = _load_real_sprite("dot")
    if img is not None:
        return _sprite_to_cell(img)
    return bake_wireframe_dot()


def argon_digit_advance() -> float:
    """The fixed-width digit advance = the '8' glyph aspect (width/height):
    ~1.0 (real square sprites, matching STD's approved wider counter) or 0.55
    (procedural fallback). Mirrors STD's argon_seg_advance = aspect['8'] so the
    counter run derives its cell width from the actual glyph, undistorted."""
    g = argon_glyph_rgba("8")
    return g.shape[1] / g.shape[0]


# ---------------------------------------------------------------------------
# ArgonFont — same public interface as before (measure / render), but the
# glyphs are baked procedurally instead of loaded from PNGs. All cells are a
# fixed 132x240 (STD's monospace 7-segment cell), advanced edge-to-edge to
# match STD's `argon_seg_advance` (= cell width / height).
# ---------------------------------------------------------------------------

_LOOKUP = {".": "dot", "%": "percentage", "x": "x", "#": "wireframes"}
_NATIVE = float(ARGON_SEG_H)     # 240 — cell height the metrics are relative to
_SPACING_NATIVE = 0.0            # STD packs cells edge-to-edge (advance = width)
_WIREFRAME_OPACITY = 0.25


def _lookup(ch: str) -> str:
    if ch.isdigit():
        return ch
    return _LOOKUP.get(ch, ch)


def _to_image(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(rgba, "RGBA")


class ArgonFont:
    """Argon-counter font. Loads the REAL argon-counter-*.png sprites from the
    live ``argon_assets/`` dir when present (tintable, pixel-exact to lazer),
    else bakes the glyphs procedurally as a fallback. ``asset_dir`` is accepted
    for call-site compatibility but IGNORED (the dir is resolved next to this
    module)."""

    def __init__(self, asset_dir=None):
        self.glyphs: dict[str, Image.Image] = {}
        for ch in "0123456789":
            self.glyphs[ch] = _to_image(argon_glyph_rgba(ch))
        self.glyphs["dot"] = _to_image(argon_glyph_rgba("."))
        self.glyphs["percentage"] = _to_image(argon_glyph_rgba("%"))
        self.glyphs["x"] = _to_image(argon_glyph_rgba("x"))
        self.glyphs["wireframes"] = _to_image(argon_wireframe_rgba())
        self._wf = self.glyphs["wireframes"]
        self._wf_dot = _to_image(argon_wireframe_dot_rgba())

    def _glyph(self, ch: str):
        return self.glyphs.get(_lookup(ch))

    def _advance(self, ch: str, scale: float) -> float:
        g = self._glyph(ch)
        w = g.width if g is not None else _NATIVE
        return (w - _SPACING_NATIVE) * scale

    def measure(self, text: str, cell_h: float, min_slots: int | None = None) -> int:
        scale = cell_h / _NATIVE
        slots = self._slots(text, min_slots)
        if not slots:
            return 0
        w = sum(self._advance(c if c is not None else "8", scale) for c in slots[:-1])
        last = slots[-1] if slots[-1] is not None else "8"
        g = self._glyph(last)
        w += (g.width if g else _NATIVE) * scale
        return int(round(w))

    def _slots(self, text: str, min_slots: int | None):
        slots = list(text)
        if min_slots and len(slots) < min_slots:
            slots = [None] * (min_slots - len(slots)) + slots   # None => wireframe-only
        return slots

    def _tinted(self, glyph: Image.Image, scale: float, tint, alpha_mul: float) -> Image.Image:
        w = max(1, int(round(glyph.width * scale)))
        h = max(1, int(round(glyph.height * scale)))
        g = glyph.resize((w, h), Image.LANCZOS)
        a = np.asarray(g)[..., 3].astype(np.float32) * alpha_mul
        out = np.zeros((h, w, 4), np.uint8)
        out[..., 0] = int(tint[0] * 255)
        out[..., 1] = int(tint[1] * 255)
        out[..., 2] = int(tint[2] * 255)
        out[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA")

    def render(self, text: str, cell_h: float, tint=(1.0, 1.0, 1.0),
               wf_opacity: float = _WIREFRAME_OPACITY, min_slots: int | None = None) -> Image.Image:
        """Render `text` as an RGBA image of cell height `cell_h`."""
        scale = cell_h / _NATIVE
        slots = self._slots(text, min_slots)
        W = max(1, self.measure(text, cell_h, min_slots) + 4)
        H = max(1, int(round(_NATIVE * scale)))
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        x = 0.0
        for ch in slots:
            # wireframe backing behind every slot (digits + leading placeholders).
            # '.' slots get the small dot wireframe (matching STD) instead of the
            # full all-segments '8'.
            if wf_opacity > 0:
                wf_src = self._wf_dot if ch == "." else self._wf
                if wf_src is not None:
                    wf = self._tinted(wf_src, scale, tint, wf_opacity)
                    canvas.alpha_composite(wf, (int(round(x)), 0))
            if ch is not None:
                g = self._glyph(ch)
                if g is not None:
                    lit = self._tinted(g, scale, tint, 1.0)
                    canvas.alpha_composite(lit, (int(round(x)), 0))
            ref = ch if ch is not None else "8"
            x += self._advance(ref, scale)
        return canvas
