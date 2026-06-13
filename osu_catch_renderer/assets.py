"""Procedural textures for Phase 1 (no skin yet).

Generates simple fruit/droplet/banana/catcher RGBA sprites with PIL so the
render is watchable before SkinPair wiring (Phase 2). Combo colours roughly
match osu!catch's default fruit palette.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# default-ish catch combo colours (pear/grape/pineapple/raspberry)
COMBO_COLORS = [
    (138, 196, 86),    # green
    (170, 110, 200),   # purple
    (240, 200, 80),    # yellow
    (225, 90, 110),    # red
]

_TEX = 128


def _circle(color, *, outline=(255, 255, 255), glow=False) -> np.ndarray:
    img = Image.new("RGBA", (_TEX, _TEX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 10
    d.ellipse([pad, pad, _TEX - pad, _TEX - pad], fill=color + (255,),
              outline=outline + (255,), width=6)
    # little highlight
    d.ellipse([_TEX * 0.30, _TEX * 0.22, _TEX * 0.52, _TEX * 0.44],
              fill=(255, 255, 255, 110))
    return np.array(img)


def _droplet() -> np.ndarray:
    img = Image.new("RGBA", (_TEX, _TEX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([_TEX * 0.3, _TEX * 0.3, _TEX * 0.7, _TEX * 0.7],
              fill=(150, 220, 255, 255), outline=(255, 255, 255, 255), width=4)
    return np.array(img)


def _banana() -> np.ndarray:
    img = Image.new("RGBA", (_TEX, _TEX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([12, 12, _TEX - 12, _TEX - 12], fill=(255, 220, 60, 255),
              outline=(255, 255, 255, 255), width=6)
    return np.array(img)


def _catcher() -> np.ndarray:
    # a soft trapezoid plate; drawn white so the sprite colour tints it
    w = h = _TEX
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(8, h - 10), (w - 8, h - 10), (w - 28, 18), (28, 18)],
              fill=(255, 255, 255, 255))
    return np.array(img)


def catch_glow_rgba(size: int = 128) -> np.ndarray:
    """Soft white radial glow (alpha falloff), tinted/additive at draw time."""
    import math
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = img.load()
    c = (size - 1) / 2.0
    for y in range(size):
        for x in range(size):
            d = math.hypot(x - c, y - c) / c
            a = max(0.0, 1.0 - d)
            a = a * a * a
            px[x, y] = (255, 255, 255, int(255 * a))
    return np.array(img)


def build_textures() -> dict[str, np.ndarray]:
    tex: dict[str, np.ndarray] = {}
    for i, c in enumerate(COMBO_COLORS):
        tex[f"fruit{i}"] = _circle(c)
    tex["droplet"] = _droplet()
    tex["banana"] = _banana()
    tex["catcher"] = _catcher()
    return tex
