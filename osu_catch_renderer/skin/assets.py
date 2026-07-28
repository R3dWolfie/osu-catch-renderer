"""Procedural textures for the catch renderer.

The default (skinless) catch objects reproduce osu!lazer's modern **Argon**
catch skin: glowing, combo-coloured *wavy rings* (CircularBlobs, additive) with
a white centre pip — NOT the old Default-skin pulp clusters. The CircularBlob
primitive is ported pixel-exact from ppy/osu-framework's
`sh_CircularBlobUtils.h` shader (MIT); the fruit/droplet/banana layering is
ported from ppy/osu's `Skinning/Argon/Argon*Piece.cs` (MIT). Combo colours
match osu!catch's default fallback palette.
"""
from __future__ import annotations

import math
import os
import random as _rnd

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
    """Soft vertical light shaft for the Argon catch hit explosion: bright
    centre column with a gaussian-ish horizontal falloff (so the edges glow
    instead of being hard lines) and a tapered top so the beam tip fades out."""
    xs = np.abs(np.arange(w) - (w - 1) / 2.0) / ((w - 1) / 2.0)
    hf = np.clip(1.0 - xs, 0.0, 1.0) ** 1.7          # horizontal soft falloff
    ys = np.arange(h) / (h - 1)                        # 0 (top) .. 1 (bottom)
    vf = np.clip(ys, 0.0, 1.0) ** 0.5                  # bright at base, soft tip
    a = (vf[:, None] * hf[None, :] * 255.0).astype(np.uint8)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[..., 0] = 255; img[..., 1] = 255; img[..., 2] = 255
    img[..., 3] = a
    return img


# =============================================================================
# osu!lazer ARGON catch objects (ported from ppy/osu + ppy/osu-framework, MIT)
# =============================================================================
# Argon fruits are glowing, combo-coloured *wavy rings* built from stacked
# `CircularBlob`s (additive) over a white centre pip — NOT pulp clusters. The
# CircularBlob is a seeded wavy ANNULUS: its outer edge sits at the sprite
# radius, perturbed by low-frequency value noise; it is filled inward by
# `InnerRadius` (fraction of the radius, 0 = invisible .. 1 = solid disc). We
# reproduce the framework shader `blobAlphaAt` pixel-for-pixel in numpy, so the
# ring shape/softness matches the game exactly (only the per-seed noise phase
# differs — .NET's RNG isn't reproduced, but the look is identical).
#
# Layering (Skinning/Argon/ArgonFruitPiece.cs), all blobs AccentColour + additive:
#   blob1: Size 1.1, InnerRadius 0.5, Alpha 0.15   (wide faint outer halo; seed A)
#   blob2: Size 1.0, InnerRadius 0.2, Alpha 0.5    (mid ring; own seed)
#   blob3: Size 1.0, InnerRadius 0.05, Alpha 1.0   (thin bright edge ring; own seed)
#   HYPER: Size 1.15, InnerRadius 0.08, Alpha 1.0, RED, seed A
#   + a white Circle pip, Size 20 (of the 128px box), drawn UNDER the blobs.
# Droplet (ArgonDropletPiece): pip + layers scaled 0.7 -> blob(IR0.5,a0.15,x0.7)
#   + blob(IR0.4,a0.5,x0.49); hyper blob(IR0.5,a0.15,x0.7,RED).
# Banana (ArgonBananaPiece : ArgonFruitPiece): fruit blobs (banana colour) + a
#   horizontal white lens-flare overlay that fades out 30%->80% of the approach.

ARGON_CANVAS = 1.2          # bake canvas spans 1.2 * OBJECT box (fits 1.1/1.15x)
ARGON_N = 192               # bake resolution (px)
ARGON_VARIANTS = 5          # a handful of seed variants so fruits vary like lazer
_OBJECT_BOX = 128.0         # 2 * CatchHitObject.OBJECT_RADIUS(64)
_PIP_PX = 20.0              # white centre Circle size (of the 128px box)


def _fract(x):
    return x - np.floor(x)


def _shader_random(x, y):
    # random(vec2 st) = fract(sin(dot(st,(12.9898,78.233))) * 43758.5453123)
    return _fract(np.sin(x * 12.9898 + y * 78.233) * 43758.5453123)


def _shader_noise(x, y):
    # 2D value noise (thebookofshaders 11), verbatim from sh_CircularBlobUtils.h
    ix = np.floor(x); iy = np.floor(y)
    fx = x - ix; fy = y - iy
    a = _shader_random(ix, iy)
    b = _shader_random(ix + 1.0, iy)
    c = _shader_random(ix, iy + 1.0)
    d = _shader_random(ix + 1.0, iy + 1.0)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    return a + (b - a) * ux + (c - a) * uy * (1.0 - ux) + (d - b) * ux * uy


def _blob_alpha(m, inner_radius, noise_pos, frequency=1.5, amplitude=0.3):
    """Exact port of sh_CircularBlobUtils.blobAlphaAt over an m x m grid that
    represents the blob's DrawSize [0,1]^2. Returns float32 alpha in [0,1].
    (frequency/amplitude are the CircularBlob defaults, unchanged by Argon.)"""
    HALF_PI = 1.57079632679
    TWO_PI = 6.28318530718
    xs = (np.arange(m) + 0.5) / m
    px, py = np.meshgrid(xs, xs)                  # px=x (cols), py=y (rows)
    ang = np.arctan2(0.5 - py, 0.5 - px) - HALF_PI
    ang = np.where(ang < 0.0, ang + TWO_PI, ang)
    complexity = (frequency + amplitude) * 0.5 + 1.0
    point_count = int(np.ceil(5.0 * complexity))
    search_range = 0.1 * complexity
    path_radius = inner_radius * 0.25
    texel = 1.5 / m                               # matches shader's 1.5/DrawWidth
    nx0, ny0 = noise_pos
    start_angle = ang - search_range * 0.5
    shortest = np.ones((m, m), np.float64)
    for i in range(point_count):
        a = start_angle + search_range * (i / point_count)
        ca = np.cos(a - HALF_PI); sa = np.sin(a - HALF_PI)
        nv = _shader_noise(nx0 + ca * frequency, ny0 + sa * frequency)
        rad = 0.5 - path_radius - texel - nv * 0.5 * amplitude
        posx = 0.5 + ca * rad; posy = 0.5 + sa * rad
        shortest = np.minimum(shortest, np.hypot(px - posx, py - posy))
    x = shortest - path_radius
    t = np.clip((x - texel) / (0.0 - texel), 0.0, 1.0)   # smoothstep(texel,0,x)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _noise_pos(seed):
    r = _rnd.Random(seed)
    return (r.uniform(0.0, 1000.0), r.uniform(0.0, 1000.0))


def _paste_center(canvas, patch):
    n = canvas.shape[0]; m = patch.shape[0]
    if m <= n:
        o = (n - m) // 2
        canvas[o:o + m, o:o + m] += patch
    else:                                          # patch larger than canvas
        c = (m - n) // 2
        canvas += patch[c:c + n, c:c + n]


def _blob_layer(size_rel, inner_radius, seed):
    """One CircularBlob's alpha, centred in the ARGON_N canvas. size_rel is the
    blob's Size relative to the OBJECT box (canvas spans ARGON_CANVAS boxes)."""
    m = max(2, int(round(size_rel / ARGON_CANVAS * ARGON_N)))
    c = np.zeros((ARGON_N, ARGON_N), np.float32)
    _paste_center(c, _blob_alpha(m, inner_radius, _noise_pos(seed)))
    return c


def _to_white_rgba(alpha):
    img = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    img[..., :3] = 255
    img[..., 3] = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    return img


def _fruit_seeds(v):
    r = _rnd.Random(0x1CE + v * 977)
    return (r.randint(1, 2**31 - 1), r.randint(1, 2**31 - 1), r.randint(1, 2**31 - 1))


def catch_argon_fruit_rgba(v: int) -> np.ndarray:
    """Composited fruit blob stack (3 additive combo-tinted rings). Because all
    three share the AccentColour, additively summing their weighted alphas here
    is identical to three separate additive draws over an 8-bit target. Drawn
    additive + combo-tinted; the white pip is a separate sprite underneath."""
    s1, s2, s3 = _fruit_seeds(v)
    inten = (0.15 * _blob_layer(1.1, 0.5, s1)
             + 0.5 * _blob_layer(1.0, 0.2, s2)
             + 1.0 * _blob_layer(1.0, 0.05, s3))
    return _to_white_rgba(inten)


def catch_argon_hyper_rgba(v: int) -> np.ndarray:
    """Red hyper blob (ArgonFruitPiece.hyperBorderPiece): Size 1.15, IR 0.08,
    Alpha 1.0, shares the outer blob's seed. Drawn additive, tinted red."""
    s1, _, _ = _fruit_seeds(v)
    return _to_white_rgba(_blob_layer(1.15, 0.08, s1))


def _droplet_seeds(v):
    r = _rnd.Random(0xD09 + v * 613)
    return (r.randint(1, 2**31 - 1), r.randint(1, 2**31 - 1))


def catch_argon_droplet_rgba(v: int) -> np.ndarray:
    """ArgonDropletPiece body: layers container scaled 0.7 -> blob(IR0.5,a0.15)
    at 0.7 + blob(IR0.4,a0.5) at 0.49. Additive + combo-tinted."""
    s1, s2 = _droplet_seeds(v)
    inten = (0.15 * _blob_layer(0.7, 0.5, s1)
             + 0.5 * _blob_layer(0.7 * 0.7, 0.4, s2))
    return _to_white_rgba(inten)


def catch_argon_droplet_hyper_rgba(v: int) -> np.ndarray:
    """Droplet red hyper blob: IR0.5, a0.15, scale 0.7, shares blob-1 seed."""
    s1, _ = _droplet_seeds(v)
    return _to_white_rgba(0.15 * _blob_layer(0.7, 0.5, s1))


def catch_argon_pip_rgba(n: int = ARGON_N) -> np.ndarray:
    """The white centre Circle (Size 20 of the 128px box), AA'd. Straight alpha,
    drawn UNDER the additive blobs."""
    SS = 4
    d = _PIP_PX / (ARGON_CANVAS * _OBJECT_BOX) * n
    img = Image.new("L", (n * SS, n * SS), 0)
    c = n * SS / 2.0; r = d * SS / 2.0
    ImageDraw.Draw(img).ellipse([c - r, c - r, c + r, c + r], fill=255)
    a = np.asarray(img.resize((n, n), Image.LANCZOS), np.float32) / 255.0
    return _to_white_rgba(a)


def catch_argon_banana_flare_rgba(w: int = 384, h: int = 192) -> np.ndarray:
    """ArgonBananaPiece lens flare: a horizontal white streak (bright central
    lens + soft horizontal falloff) on a wide canvas. Additive white; the
    per-frame OutQuint fade is applied at draw time in scene.py."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / (w / 2.0)                     # -1..1 across width
    ny = (yy - cy) / (h / 2.0)
    # thin horizontal streak: tight gaussian in y, gentle falloff to the ends
    streak = (np.exp(-((ny / (5.0 / h)) ** 2))
              * np.clip(1.0 - np.abs(nx), 0.0, 1.0) ** 0.6 * 0.8)
    # bright central lens blob (the Circle 8x8 scaled (25,1))
    lens = np.exp(-((nx / 0.24) ** 2 + (ny / 0.45) ** 2)) * 0.55
    a = np.clip(streak + lens, 0.0, 1.0)
    return _to_white_rgba(a)


def build_argon_textures() -> dict[str, np.ndarray]:
    """All Argon catch-object textures, keyed for upload. Baked once per render
    (~2s). Fruit/hyper/droplet come in ARGON_VARIANTS seed variants so fruits
    look varied like lazer; the pip + banana flare are shared."""
    tex: dict[str, np.ndarray] = {}
    for v in range(ARGON_VARIANTS):
        tex[f"argon_fruit_{v}"] = catch_argon_fruit_rgba(v)
        tex[f"argon_hyper_{v}"] = catch_argon_hyper_rgba(v)
        tex[f"argon_droplet_{v}"] = catch_argon_droplet_rgba(v)
        tex[f"argon_drophyper_{v}"] = catch_argon_droplet_hyper_rgba(v)
    tex["argon_pip"] = catch_argon_pip_rgba()
    tex["argon_banana_flare"] = catch_argon_banana_flare_rgba()
    return tex


# --- R3D intro logo splash ---------------------------------------------------
# The glossy beveled 'R' tile shown during the intro (show_logo), fading out
# as the first fruit begins its approach. Ported from the std renderer
# (osu_std_renderer.render.textures.bake_logo_tile): load R3D's REAL logo
# asset (own IP, license-clean -- the SAME logo.png the std splash uses, so
# the splash is identical across modes) and fall back to a simple procedural
# red 'R' tile only if the asset is missing.
LOGO_TILE_RED = (216, 44, 54)


def bake_logo_tile(size: int = 256) -> np.ndarray:
    """RGBA tile for the intro splash. Prefers assets/logo.png (the real R3D
    logo); procedural fallback (rounded red tile + white R) only if missing."""
    try:
        lp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")
        im = Image.open(lp).convert("RGBA").resize((size, size), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8).copy()
    except Exception:
        pass
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    drw = ImageDraw.Draw(img)
    drw.rounded_rectangle([0, 0, size - 1, size - 1],
                          radius=int(size * 0.18), fill=LOGO_TILE_RED + (255,))
    try:
        from osu_catch_renderer.hud.fonts import font as _font
        f = _font(int(size * 0.66))
        box = f.getbbox("R")
        rw, rh = box[2] - box[0], box[3] - box[1]
        drw.text(((size - rw) / 2.0 - box[0], (size - rh) / 2.0 - box[1]),
                 "R", font=f, fill=(255, 255, 255, 255))
    except Exception:
        pass
    return np.asarray(img, dtype=np.uint8).copy()


def build_textures() -> dict[str, np.ndarray]:
    """Legacy procedural discs for the no-skin fallback paths. The Argon catch
    objects (glowing wavy combo rings + white pip) live under the `argon_*`
    keys and are uploaded separately via build_argon_textures() so they are
    present regardless of skin (the plate pile + hit explosions use them too)."""
    tex: dict[str, np.ndarray] = {}
    for i, c in enumerate(COMBO_COLORS):
        tex[f"fruit{i}"] = _circle(c)
    tex["droplet"] = _droplet()
    tex["banana"] = _banana()
    tex["catcher"] = _catcher()
    return tex
