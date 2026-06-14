"""osu!lazer **Argon** counter font (score / combo / accuracy / pp).

Renders numbers with the real `argon-counter` texture glyphs lifted from
ppy/osu-resources (`Textures/Gameplay/Fonts/argon-counter-*.png`) — the squared
segmented numerals. Mirrors `ArgonCounterTextComponent`:
  * glyphs are 240px cells displayed at 0.125x; advance = texWidth - 16 native px
    (lazer's -2px display Spacing),
  * a dim "wireframe" (all-segments) glyph sits behind every slot at
    WireframeOpacity (0.25) — that's the ⊠ placeholder look for unlit digits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_LOOKUP = {".": "dot", "%": "percentage", "x": "x", "#": "wireframes"}
_NATIVE = 240.0
_SPACING_NATIVE = 16.0          # lazer Spacing -2 at 0.125 display => 16 native px
_WIREFRAME_OPACITY = 0.25


def _lookup(ch: str) -> str:
    if ch.isdigit():
        return ch
    return _LOOKUP.get(ch, ch)


class ArgonFont:
    def __init__(self, asset_dir: Path):
        self.dir = Path(asset_dir)
        self.glyphs: dict[str, Image.Image] = {}
        for name in [str(i) for i in range(10)] + ["dot", "percentage", "x", "wireframes"]:
            p = self.dir / f"argon-counter-{name}.png"
            if p.is_file():
                self.glyphs[name] = Image.open(p).convert("RGBA")
        self._wf = self.glyphs.get("wireframes")

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
            # wireframe backing behind every slot (digits + leading placeholders)
            if self._wf is not None and wf_opacity > 0:
                wf = self._tinted(self._wf, scale, tint, wf_opacity)
                canvas.alpha_composite(wf, (int(round(x)), 0))
            if ch is not None:
                g = self._glyph(ch)
                if g is not None:
                    lit = self._tinted(g, scale, tint, 1.0)
                    canvas.alpha_composite(lit, (int(round(x)), 0))
            ref = ch if ch is not None else "8"
            x += self._advance(ref, scale)
        return canvas
