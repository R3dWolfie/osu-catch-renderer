"""Load osu!catch skin elements (Night05 by default) following VRender's
uskin -> default -> skip philosophy.

Catch fruit/drop/banana sprites ship greyscale and are tinted by the combo
colour at draw time (the GL shader multiplies texture * sprite.color); their
`-overlay` companions are drawn untinted on top. The catcher is full-colour
and drawn as-is. @2x variants are preferred when present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# fruit shape cycles by combo index, matching lazer's VisualRepresentation
FRUIT_SHAPES = ["fruit-pear", "fruit-grapes", "fruit-apple", "fruit-orange"]

# elements we load (base name -> whether it also has an -overlay)
_ELEMENTS = {
    "fruit-pear": True, "fruit-grapes": True, "fruit-apple": True, "fruit-orange": True,
    "fruit-drop": True, "fruit-bananas": True,
    "fruit-catcher-idle": False, "fruit-catcher-kiai": False, "fruit-catcher-fail": False,
    # legacy single-sprite catcher (pippi) — the catcher for old skins.
    # lazer CatchLegacySkinTransformer: skin version < 2.3 uses fruit-ryuuta
    # (LegacyCatcherOld) even when fruit-catcher-idle exists; newer skins use
    # fruit-catcher-idle and only fall back to fruit-ryuuta when idle is absent.
    "fruit-ryuuta": False,
}

_DEFAULT_COMBOS = [
    (1.0, 0.4, 1.0), (0.4, 0.66, 1.0), (0.4, 0.5, 1.0), (0.62, 0.4, 1.0),
]


class CatchSkin:
    def __init__(self, skin_dir: Path, default_dir: Path | None = None):
        skin_dir = self._resolve_root(skin_dir) if skin_dir else skin_dir
        if default_dir:
            default_dir = self._resolve_root(default_dir)
        self.dirs = [d for d in (skin_dir, default_dir) if d and d.is_dir()]
        # the USER's chosen skin dir (resolved) — combo_colors_custom is
        # scoped to it: only a skin the user actually picked may beat the
        # beatmap's [Colours] (lazer rule — scene._combo_tint). The bundled
        # service default ("_default-source" / Night05) ships Combo1..4 but
        # is nobody's choice, so it must never override the map's palette.
        self._user_skin_dir = (
            skin_dir if (skin_dir and Path(skin_dir).is_dir()
                         and Path(skin_dir).name != "_default-source")
            else None)
        self.textures: dict[str, np.ndarray] = {}
        self._load_elements()
        self._load_hit_lighting(default_dir)
        self.combo_colors = self._load_combos()
        self._load_hyper_colors()
        # Which sprite IS the catcher for this skin (lazer version rule above);
        # None = no legacy catcher → scene falls back to the Argon bar.
        self.catcher_key = self._resolve_catcher_key()
        self.catcher_aspect = self._aspect(self.catcher_key or "fruit-catcher-idle",
                                           324 / 305)

    @staticmethod
    def _resolve_root(d: Path) -> Path:
        """Find the actual skin root. The bundled default extracts its sprites
        into a `_default-source` subdir (danser appends it manually), and some
        .osk archives wrap everything in one folder — so if the given dir has
        no skin.ini / catch sprites, descend one level to a subdir that does."""
        d = Path(d)

        def has_skin(p: Path) -> bool:
            return (p / "skin.ini").is_file() or any(p.glob("fruit-*.png"))

        if not d.is_dir() or has_skin(d):
            return d
        subs = sorted(c for c in d.iterdir() if c.is_dir())
        for sub in subs:
            if has_skin(sub):
                return sub
        return d

    # --- public ---------------------------------------------------------------

    def fruit_key(self, combo_index: int) -> str:
        return FRUIT_SHAPES[combo_index % len(FRUIT_SHAPES)]

    def combo_color(self, combo_index: int) -> tuple[float, float, float]:
        if not self.combo_colors:
            return (1.0, 1.0, 1.0)
        return self.combo_colors[combo_index % len(self.combo_colors)]

    def has(self, key: str) -> bool:
        return key in self.textures

    # --- catcher resolution ---------------------------------------------------

    @staticmethod
    def _dir_has(d: Path, basename: str) -> bool:
        return any((d / f"{basename}{s}.png").is_file() for s in ("@2x", ""))

    @staticmethod
    def _skin_version(d: Path) -> float:
        """skin.ini [General] Version of ONE skin dir. Stable's rule: missing
        ini / missing key = 1.0 (old); 'latest' = new."""
        ini = d / "skin.ini"
        if not ini.is_file():
            return 1.0
        try:
            for line in ini.read_text(errors="replace").splitlines():
                s = line.strip()
                if s.lower().startswith("version") and ":" in s:
                    val = s.split(":", 1)[1].strip()
                    if val.lower() == "latest":
                        return 99.0
                    try:
                        return float(val)
                    except ValueError:
                        return 1.0
        except Exception:  # noqa: BLE001 — unreadable ini = old skin
            return 1.0
        return 1.0

    def _resolve_catcher_key(self) -> str | None:
        """The catcher sprite key, per lazer's CatchLegacySkinTransformer:
        the FIRST dir (user skin, then default) that ships any catcher wins as
        a unit (a user skin's ryuuta must not be beaten by the default's
        idle); within it, version < 2.3 prefers fruit-ryuuta (LegacyCatcherOld)
        when present, else fruit-catcher-idle, else fruit-ryuuta."""
        for d in self.dirs:
            has_idle = self._dir_has(d, "fruit-catcher-idle")
            has_ryuuta = self._dir_has(d, "fruit-ryuuta")
            if not (has_idle or has_ryuuta):
                continue
            if has_ryuuta and (not has_idle or self._skin_version(d) < 2.3):
                return "fruit-ryuuta"
            return "fruit-catcher-idle"
        return None

    # --- loading --------------------------------------------------------------

    def _resolve(self, basename: str) -> Path | None:
        for d in self.dirs:
            for stem in (f"{basename}@2x", basename):
                p = d / f"{stem}.png"
                if p.is_file():
                    return p
        return None

    # fruit/drop/banana bases ship dark-grey in many skins; brighten them so
    # the combo-colour tint reads vibrant (osu draws fruit bright, not muddy).
    _BOOST = {"fruit-pear", "fruit-grapes", "fruit-apple", "fruit-orange",
              "fruit-drop", "fruit-bananas"}

    def _load_elements(self) -> None:
        for base, has_overlay in _ELEMENTS.items():
            keys = [base] + ([f"{base}-overlay"] if has_overlay else [])
            for key in keys:
                p = self._resolve(key)
                if p is None:
                    continue
                tex = _rgba(p)
                if key in self._BOOST:
                    tex = _brighten(tex, 1.9)
                self.textures[key] = tex

    def _load_hit_lighting(self, default_dir: Path | None) -> None:
        """Catch hit-lighting textures — lazer LegacyHitExplosion takes them
        from the CLASSIC DEFAULT skin ONLY (`skins.DefaultClassicSkin
        .GetTexture(...)`; the user skin is deliberately never consulted), so
        resolve them strictly from the default-skin dir (which now carries the
        classic scoreboard-explosion art). Keys note lazer's intentional
        sprite swap: the BEAM (explosion1) is scoreboard-explosion-2 and the
        plate PUFF (explosion2) is scoreboard-explosion-1."""
        if not (default_dir and Path(default_dir).is_dir()):
            return
        d = Path(default_dir)
        for key, base in (("catch_light_beam", "scoreboard-explosion-2"),
                          ("catch_light_puff", "scoreboard-explosion-1")):
            for stem in (f"{base}@2x", base):
                p = d / f"{stem}.png"
                if p.is_file():
                    tex = _rgba(p)
                    if "@2x" in p.name:   # store at LOGICAL size for sizing
                        im = Image.fromarray(tex).resize(
                            (max(1, tex.shape[1] // 2),
                             max(1, tex.shape[0] // 2)), Image.LANCZOS)
                        tex = np.array(im)
                    self.textures[key] = tex
                    break

    def _load_combos(self) -> list[tuple[float, float, float]]:
        # combo_colors_custom: True only when the combos came from the
        # USER's chosen skin's own skin.ini (see __init__ / _combo_tint).
        self.combo_colors_custom = False
        for d in self.dirs:
            ini = d / "skin.ini"
            if not ini.is_file():
                continue
            combos: dict[int, tuple[float, float, float]] = {}
            for line in ini.read_text(errors="replace").splitlines():
                s = line.strip()
                low = s.lower()
                if low.startswith("combo") and ":" in s and low[5:6].isdigit():
                    idx_str, val = s.split(":", 1)
                    try:
                        idx = int(idx_str.strip()[5:])
                        r, g, b = (int(x) for x in val.split(",")[:3])
                        combos[idx] = (r / 255, g / 255, b / 255)
                    except ValueError:
                        continue
            if combos:
                self.combo_colors_custom = (d == self._user_skin_dir)
                return [combos[k] for k in sorted(combos)]
        return list(_DEFAULT_COMBOS)

    def _load_hyper_colors(self) -> None:
        """skin.ini [CatchTheBeat] hyperdash cue colours, as 0..1 RGB:
          HyperDash           — catcher tint / trail colour  (default red)
          HyperDashFruit      — the hyperfruit echo          (falls back to HyperDash)
          HyperDashAfterImage — the hyper-onset after-image  (falls back to HyperDash)
        Scoped to the USER's chosen skin dir only (like combo_colors_custom):
        the bundled service default (`_default-source` / Night05) ships its own
        HyperDash line, but it is nobody's choice — a user skin with NO
        HyperDash must keep osu!'s stock red, not inherit the bundle's."""
        vals: dict[str, tuple[float, float, float]] = {}
        d = self._user_skin_dir
        ini = (d / "skin.ini") if d else None
        if ini is not None and ini.is_file():
            section = ""
            for line in ini.read_text(errors="replace").splitlines():
                s = line.strip()
                if s.startswith("[") and s.endswith("]"):
                    section = s[1:-1].strip().lower()
                    continue
                if section != "catchthebeat" or ":" not in s:
                    continue
                key, val = s.split(":", 1)
                key = key.strip().lower()
                if key not in ("hyperdash", "hyperdashfruit",
                               "hyperdashafterimage"):
                    continue
                try:  # R,G,B — a 4th (alpha) component is ignored like stable
                    r, g, b = (min(255, max(0, int(x)))
                               for x in val.split(",")[:3])
                except ValueError:
                    continue
                vals[key] = (r / 255, g / 255, b / 255)
        self.hyper_color = vals.get("hyperdash", (1.0, 0.0, 0.0))
        self.hyper_fruit_color = vals.get("hyperdashfruit", self.hyper_color)
        self.hyper_afterimage_color = vals.get("hyperdashafterimage",
                                               self.hyper_color)

    def _aspect(self, key: str, default: float) -> float:
        t = self.textures.get(key)
        if t is None:
            return default
        h, w = t.shape[:2]
        return h / w if w else default


def _rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def _brighten(rgba: np.ndarray, factor: float) -> np.ndarray:
    out = rgba.astype("f4")
    out[..., :3] = np.clip(out[..., :3] * factor, 0, 255)
    return out.astype("u1")
