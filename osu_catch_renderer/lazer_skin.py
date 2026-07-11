"""Procedural osu!lazer **Argon** catcher (no skin sprites).

Red's parity target is osu!lazer's built-in Argon look as the BASE. The Argon
catcher (ppy/osu `Skinning/Argon/ArgonCatcher.cs`, MIT) is minimal and WHITE:

  - a white rounded **bar** spanning `ALLOWED_CATCH_RANGE` (0.8) of the catcher
    width, height 10 (playfield units) — the catch zone;
  - a white rounded **bumper** at each end of the catch range, width (1-0.8)/2
    = 0.1 of the catcher width, height 4;
  - a faint (alpha 0.25) **long line**, height 1.8, extending from each end out
    toward the screen edge.

All pieces are Color4.White — no gradient, glow, or dish. The catcher is drawn
directly from primitive quads in scene.py; this module only bakes the shared
rounded-capsule texture and returns the pixel geometry (footprint unchanged:
full width = the renderer's catcher_w, placed with its top on the catch plane).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Catcher.ALLOWED_CATCH_RANGE — the catchable fraction of the catcher width.
ALLOWED_CATCH_RANGE = 0.8


def argon_bar_cap_rgba(w: int = 512, h: int = 64) -> np.ndarray:
    """A white horizontal capsule (rounded-rect, corner radius = h/2), AA'd.
    Stretched to the bar/bumper sizes at draw time; the end caps stay rounded
    (a thin near-horizontal bar's caps read as rounded regardless of stretch)."""
    SS = 4
    img = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(img).rounded_rectangle(
        [0, 0, w * SS - 1, h * SS - 1], radius=h * SS / 2.0, fill=255)
    a = np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255.0
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = 255
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def argon_catcher_metrics(catcher_w: float, unit_px: float,
                          plane_y: float) -> dict:
    """Pixel geometry of the Argon catcher pieces. Footprint is unchanged: the
    full catcher spans `catcher_w` with the top of the bar on `plane_y` (where
    the old plate's top lip sat). Heights come from Argon's absolute px (10 / 4
    / 1.8) scaled by the playfield unit; widths are the Argon fractions of the
    catcher width (bar 0.8, bumper 0.1 each -> together exactly 1.0)."""
    bar_h = 10.0 * unit_px
    bump_h = 4.0 * unit_px
    line_h = max(1.0, 1.8 * unit_px)
    bar_w = ALLOWED_CATCH_RANGE * catcher_w
    bump_w = (1.0 - ALLOWED_CATCH_RANGE) / 2.0 * catcher_w
    cy = plane_y + bar_h * 0.5           # bar top on plane_y (old top-lip line)
    return {
        "full_w": catcher_w, "bar_w": bar_w, "bar_h": bar_h,
        "bump_w": bump_w, "bump_h": bump_h, "line_h": line_h, "cy": cy,
    }
