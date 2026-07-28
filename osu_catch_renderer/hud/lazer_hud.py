"""osu!lazer HUD layout, read from the skin itself.

A skin exported from lazer's skin editor carries `MainHUDComponents.json`, which
lists every HUD component with its concrete class, Anchor/Origin, Position,
Scale and Rotation. That is the user's ACTUAL in-game HUD arrangement — so the
renderer can reproduce it exactly instead of hardcoding one layout.

Coordinates follow osu!framework: the HUD lives in a fixed-height UI space
(768 lazer px tall, see LAZER_UI_HEIGHT); a component's Anchor picks a point on
the parent, `Position` offsets from it (in lazer px), and the component is then
placed so its Origin lands on that point.

Anchor/Origin are [Flags]: x0=8 left, x1=16 centre, x2=32 right,
y0=1 top, y1=2 centre, y2=4 bottom. So 9=TopLeft, 33=TopRight, 12=BottomLeft,
36=BottomRight, 34=CentreRight, 18=Centre.
"""
from __future__ import annotations

import json
from pathlib import Path

LAZER_UI_HEIGHT = 768.0

# ruleset section keys in the JSON ("global" always applies)
CATCH_SECTIONS = ("global", "fruits")


def anchor_frac(a: int) -> tuple[float, float]:
    """Anchor/Origin flags -> (fx, fy) fractions of the box, 0..1."""
    fx = 0.5 if a & 16 else (1.0 if a & 32 else 0.0)
    fy = 0.5 if a & 2 else (1.0 if a & 4 else 0.0)
    return fx, fy


def short_type(t: str) -> str:
    """'osu.Game.Skinning.LegacyScoreCounter, osu.Game, ...' -> 'LegacyScoreCounter'."""
    return t.split(",", 1)[0].strip().rsplit(".", 1)[-1]


class Component:
    __slots__ = ("type", "pos", "rotation", "scale", "anchor", "origin",
                 "settings", "width", "height", "section")

    def __init__(self, d: dict, section: str):
        self.section = section
        self.type = short_type(d.get("Type", ""))
        p = d.get("Position") or {}
        self.pos = (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
        self.rotation = float(d.get("Rotation") or 0.0)
        s = d.get("Scale") or {}
        # lazer stores a negative scale for a flipped element; magnitude is the
        # size, the sign only mirrors — we take the magnitude and ignore mirroring
        # (none of the supported components rely on being mirrored).
        self.scale = (abs(float(s.get("x", 1.0))) or 1.0,
                      abs(float(s.get("y", 1.0))) or 1.0)
        self.anchor = int(d.get("Anchor") or 0)
        self.origin = int(d.get("Origin") or 0)
        self.settings = d.get("Settings") or {}
        self.width = d.get("Width")
        self.height = d.get("Height")

    def place(self, screen_w: int, screen_h: int, w: float, h: float) -> tuple[int, int]:
        """Top-left pixel at which to paste a `w`x`h` drawable on screen."""
        k = screen_h / LAZER_UI_HEIGHT
        ax, ay = anchor_frac(self.anchor)
        ox, oy = anchor_frac(self.origin)
        px = ax * screen_w + self.pos[0] * k
        py = ay * screen_h + self.pos[1] * k
        return int(round(px - ox * w)), int(round(py - oy * h))

    def k(self, screen_h: int) -> float:
        """lazer px -> screen px, including this component's own Scale."""
        return (screen_h / LAZER_UI_HEIGHT) * self.scale[1]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<{self.type} sec={self.section} anchor={self.anchor} "
                f"origin={self.origin} pos={self.pos} scale={self.scale[0]:.3f} "
                f"rot={self.rotation:.0f}>")


def load_layout(skin_dir) -> dict[str, Component]:
    """Parse the skin's `MainHUDComponents.json`.

    Returns {short_type: Component} for the sections that apply to catch, with
    a ruleset-specific entry overriding the global one of the same type. Missing
    file / malformed JSON returns {} so the caller falls back to the default
    layout — a broken layout file must never kill a render.
    """
    if not skin_dir:
        return {}
    p = Path(skin_dir) / "MainHUDComponents.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        info = data.get("DrawableInfo") or {}
    except (OSError, ValueError):
        return {}
    out: dict[str, Component] = {}
    for section in CATCH_SECTIONS:
        for entry in (info.get(section) or []):
            try:
                c = Component(entry, section)
            except (TypeError, ValueError):
                continue
            if c.type:
                out[c.type] = c          # later section (ruleset) wins
    return out
