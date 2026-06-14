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


def catch_beam_rgba(w: int = 48, h: int = 256) -> np.ndarray:
    """Soft vertical light shaft for the catch hit explosion: bright centre
    column with a gaussian-ish horizontal falloff (so the edges glow instead of
    being hard lines) and a tapered top so the beam tip fades out."""
    xs = np.abs(np.arange(w) - (w - 1) / 2.0) / ((w - 1) / 2.0)
    hf = np.clip(1.0 - xs, 0.0, 1.0) ** 1.7          # horizontal soft falloff
    ys = np.arange(h) / (h - 1)                        # 0 (top) .. 1 (bottom)
    vf = np.clip(ys, 0.0, 1.0) ** 0.5                  # bright at base, soft tip
    a = (vf[:, None] * hf[None, :] * 255.0).astype(np.uint8)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[..., 0] = 255; img[..., 1] = 255; img[..., 2] = 255
    img[..., 3] = a
    return img


def argon_fruit_ring_rgba(n: int = 160) -> np.ndarray:
    """lazer ArgonFruitPiece outline: an organic (slightly wavy) white ring,
    tinted per-combo + additive at draw. Approximates CircularBlob."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    dx = xx - c; dy = yy - c
    d = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    base_r = n * 0.40
    thick = n * 0.085
    r = base_r * (1.0 + 0.055 * np.sin(ang * 5 + 0.7)
                  + 0.03 * np.sin(ang * 3 + 2.1)
                  + 0.022 * np.sin(ang * 8 + 4.0))
    a = np.clip(1.0 - np.abs(d - r) / thick, 0.0, 1.0) ** 1.4
    img = np.zeros((n, n, 4), dtype=np.uint8)
    img[..., 0] = 255; img[..., 1] = 255; img[..., 2] = 255
    img[..., 3] = (a * 255).astype(np.uint8)
    return img


def argon_fruit_dot_rgba(n: int = 64) -> np.ndarray:
    """Small solid white centre dot (lazer ArgonFruitPiece core Circle)."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    d = np.hypot(xx - c, yy - c) / (n * 0.5)
    a = np.where(d < 0.6, 1.0, np.clip(1.0 - (d - 0.6) / 0.4, 0.0, 1.0) ** 2)
    img = np.zeros((n, n, 4), dtype=np.uint8)
    img[..., :3] = 255
    img[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return img


def build_textures() -> dict[str, np.ndarray]:
    tex: dict[str, np.ndarray] = {}
    for i, c in enumerate(COMBO_COLORS):
        tex[f"fruit{i}"] = _circle(c)
    tex["droplet"] = _droplet()
    tex["banana"] = _banana()
    tex["catcher"] = _catcher()
    return tex
