"""Danser-style HUD composited from real skin sprites.

Score / accuracy / combo use the skin's number glyphs (score-*, combo-*),
HP uses scorebar-bg + scorebar-colour, grade uses ranking-*, mods use
selection-mod-*. A thin progress bar and the player/title line are drawn with
PIL. Everything composites on the CPU after GL readback.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# osu mod bit -> selection-mod-<name>. Nightcore (512) supersedes DT icon.
_MODS = [
    (2, "easy"), (8, "hidden"), (16, "hardrock"), (512, "nightcore"),
    (64, "doubletime"), (256, "halftime"), (1024, "flashlight"),
    (1, "nofail"), (16384, "perfect"), (32, "suddendeath"),
    (128, "relax"), (4096, "spunout"), (2048, "autoplay"),
]


class DanserHud:
    def __init__(self, skin_dir: Path, resolution, meta, beatmap,
                 first_ms: int, last_ms: int, cfg=None):
        self.cfg = cfg
        # Resolve to the real skin root the same way CatchSkin does, so the HUD
        # finds score/combo/ranking/mod sprites for ANY skin whose assets live
        # in a subdir (the bundled default's `_default-source`, single-folder
        # .osk archives, etc.) — not just skins with skin.ini at the top level.
        from .skin import CatchSkin
        self.dir = CatchSkin._resolve_root(Path(skin_dir)) if skin_dir else Path(skin_dir)
        self.w, self.h = resolution
        self.meta = meta
        self.bm = beatmap
        self.first_ms = first_ms
        self.last_ms = max(last_ms, first_ms + 1)
        H = self.h
        self.score_glyphs = self._glyphs("score", int(H * 0.050))
        self.acc_glyphs = self._glyphs("score", int(H * 0.032))
        # legacy catch combo uses the combo font, falling back to the score font
        # (stable's default ComboPrefix == ScorePrefix == "score").
        self.combo_glyphs = (self._glyphs("combo", int(H * 0.072))
                             or self._glyphs("score", int(H * 0.072)))
        self.scorebar_bg = self._load("scorebar-bg", int(H * 0.05))
        self.scorebar_col = self._load("scorebar-colour-0", int(H * 0.05))
        self.grades = {g: self._load(f"ranking-{g}", int(H * 0.060))
                       for g in ("X", "XH", "S", "SH", "A", "B", "C", "D")}
        self.mod_imgs = self._mods(meta.mods, int(H * 0.052))
        # danser-style client badge: mark osu!stable replays (game_version < 30M)
        if 0 < getattr(meta, "game_version", 0) < 30000000:
            self.mod_imgs.append(self._text_badge("Stable", int(H * 0.052)))
        self.font = _font(int(H * 0.024))
        self.font_small = _font(int(H * 0.020))
        self.font_combo = _font(int(H * 0.072))   # PIL fallback for combo glyphs
        from .argon_health import ArgonHealth
        self.argon_hp = ArgonHealth(self.w, self.h)
        self._hp_last_t = None

    # --- public ---------------------------------------------------------------

    def overlay(self, rgb: np.ndarray, scene) -> np.ndarray:
        img = Image.fromarray(rgb, "RGB")
        pad = int(self.w * 0.012)
        # the Argon health bar hugs the very top; keep the score stack below it
        top_y = int(0.014 * self.h + 40.0 * (self.h / 1080.0) + 0.012 * self.h)
        cfg = self.cfg
        def _on(n):
            return cfg is None or getattr(cfg, n, True)

        # score (top-right, no leading zeros). Images are built either way so the
        # layout (y / mods_y / pp rows) stays consistent; only the paste is gated.
        score_img = self._number(f"{scene.score}", self.score_glyphs, overlap=1)
        if _on("show_score"):
            self._paste(img, score_img, self.w - pad - score_img.width, top_y)
        y = top_y + score_img.height + int(self.h * 0.008)

        # accuracy (under score, right-aligned) + grade to its left
        acc_img = self._number(f"{scene.accuracy * 100:.2f}%", self.acc_glyphs, overlap=1)
        if _on("show_score"):
            self._paste(img, acc_img, self.w - pad - acc_img.width, y)
        grade = self.grades.get(self._grade(scene.accuracy))
        if grade is not None and _on("show_grade"):
            self._paste(img, grade, self.w - pad - acc_img.width - grade.width - 8,
                        y - (grade.height - acc_img.height) // 2)

        # mods row (under accuracy)
        mods_y = y + acc_img.height + int(self.h * 0.01)
        if _on("show_mods"):
            mx = self.w - pad
            for m in self.mod_imgs:
                mx -= m.width + 4
                self._paste(img, m, mx, mods_y)

        # pp counter + hit counter (top-right, under the mods row)
        ry = mods_y + int(self.h * 0.052) + int(self.h * 0.015)
        cfg = self.cfg
        if cfg is not None and getattr(cfg, "show_pp_counter", False) and scene.pp > 0:
            d = ImageDraw.Draw(img)
            txt = f"{scene.pp:.0f}pp"
            pf = _font(int(self.h * 0.032))
            bb = d.textbbox((0, 0), txt, font=pf)
            d.text((self.w - pad - (bb[2] - bb[0]), ry), txt, font=pf, fill=(245, 235, 255))
            ry += int(self.h * 0.045)
        if cfg is not None and getattr(cfg, "show_hit_counter", False):
            c300, c100, c50, _tmiss, miss = scene.counts
            d = ImageDraw.Draw(img)
            hf = _font(int(self.h * 0.024))
            for label, val, col in (("300", c300, (110, 200, 255)),
                                    ("100", c100, (130, 255, 160)),
                                    ("50", c50, (255, 225, 120)),
                                    ("Miss", miss, (255, 110, 110))):
                txt = f"{val}x {label}"
                bb = d.textbbox((0, 0), txt, font=hf)
                d.text((self.w - pad - (bb[2] - bb[0]), ry), txt, font=hf, fill=col)
                ry += int(self.h * 0.03)

        # combo (large white number, centre-screen — lazer base look)
        if scene.combo > 0 and _on("show_combo"):
            d = ImageDraw.Draw(img)
            txt = str(scene.combo)
            f = _font(int(self.h * 0.117))
            bbox = d.textbbox((0, 0), txt, font=f)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx, cy = self.w * 0.5, self.h * 0.44
            d.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), txt,
                   font=f, fill=(247, 247, 248))

        # HP bar (osu!lazer Argon) — top, full width, ported from lazer source
        if _on("show_hp_bar"):
            t = scene.time_ms
            dt = 16.0 if self._hp_last_t is None else max(0.0, min(100.0, t - self._hp_last_t))
            self._hp_last_t = t
            arr = np.asarray(img).copy()
            self.argon_hp.update_draw(arr, scene.hp, dt)
            img = Image.fromarray(arr)

        # progress bar (bottom edge)
        self._draw_progress(img, scene.time_ms)

        # player + title (top-left, under scorebar)
        d = ImageDraw.Draw(img)
        ty = int(self.h * 0.075)
        d.text((pad, ty), self.meta.player_name, font=self.font, fill=(245, 245, 250))
        title = f"{self.bm.artist} - {self.bm.title} [{self.bm.version}]".strip(" -")
        title = _ellipsize(d, title, self.font_small, self.w - pad * 2)
        d.text((pad, ty + int(self.h * 0.03)), title, font=self.font_small, fill=(190, 190, 205))

        # watermark (bottom-right) — free renders are forced to the site URL
        wm = getattr(self.cfg, "watermark", "") if self.cfg else ""
        if wm:
            wmf = _font(int(self.h * 0.022))
            wb = d.textbbox((0, 0), wm, font=wmf)
            d.text((self.w - pad - (wb[2] - wb[0]), self.h - pad - (wb[3] - wb[1]) - int(self.h * 0.006)),
                   wm, font=wmf, fill=(238, 238, 245))
        return np.asarray(img)

    # --- compositing helpers --------------------------------------------------

    def _draw_scorebar(self, img, hp: float):
        if self.scorebar_bg is None:
            # fallback bar
            d = ImageDraw.Draw(img)
            bx, by, bw, bh = int(self.w * 0.012), int(self.h * 0.018), int(self.w * 0.33), 12
            d.rectangle([bx, by, bx + bw, by + bh], fill=(35, 35, 48))
            d.rectangle([bx, by, bx + int(bw * hp), by + bh], fill=(120, 220, 150))
            return
        x0, y0 = int(self.w * 0.008), int(self.h * 0.012)
        self._paste(img, self.scorebar_bg, x0, y0)
        if self.scorebar_col is not None:
            # the colour bar fills from the left, clipped to current HP
            off_x = int(self.scorebar_bg.width * 0.05)
            off_y = (self.scorebar_bg.height - self.scorebar_col.height) // 2
            cw = max(1, int(self.scorebar_col.width * hp))
            clip = self.scorebar_col.crop((0, 0, cw, self.scorebar_col.height))
            self._paste(img, clip, x0 + off_x, y0 + off_y)

    def _draw_progress(self, img, t_ms: int):
        frac = (t_ms - self.first_ms) / (self.last_ms - self.first_ms)
        frac = max(0.0, min(1.0, frac))
        d = ImageDraw.Draw(img)
        y = self.h - 6
        d.rectangle([0, y, self.w, self.h], fill=(30, 30, 40))
        d.rectangle([0, y, int(self.w * frac), self.h], fill=(150, 200, 255))

    def _number(self, text: str, glyphs: dict, overlap: int) -> Image.Image:
        gh = max((g.height for g in glyphs.values()), default=int(self.h * 0.03))
        imgs = []
        for ch in text:
            key = {".": "dot", ",": "comma", "%": "percent", "x": "x"}.get(ch, ch)
            g = glyphs.get(key)
            if g is None and not ch.isspace():
                # font fallback for a char the skin lacks (or whose glyph we
                # rejected, e.g. a junk score-percent) so e.g. "%" still shows.
                f = _font(gh)
                d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
                bb = d0.textbbox((0, 0), ch, font=f)
                cw, chh = bb[2] - bb[0], bb[3] - bb[1]
                g = Image.new("RGBA", (max(1, cw + 2), gh), (0, 0, 0, 0))
                ImageDraw.Draw(g).text((1 - bb[0], (gh - chh) // 2 - bb[1]), ch,
                                       font=f, fill=(255, 255, 255, 255))
            if g is not None:
                imgs.append(g)
        if not imgs:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        h = max(g.height for g in imgs)
        w = sum(g.width for g in imgs) - overlap * (len(imgs) - 1)
        out = Image.new("RGBA", (max(1, w), h), (0, 0, 0, 0))
        x = 0
        for g in imgs:
            out.alpha_composite(g, (x, (h - g.height) // 2))
            x += g.width - overlap
        return out

    @staticmethod
    def _paste(img, sprite, x, y):
        if sprite is None:
            return
        img.paste(sprite, (int(x), int(y)), sprite)

    # --- loading --------------------------------------------------------------

    def _resolve(self, base: str) -> Path | None:
        for stem in (f"{base}@2x", base):
            p = self.dir / f"{stem}.png"
            if p.is_file():
                return p
        return None

    def _load(self, base: str, target_h: int) -> Image.Image | None:
        p = self._resolve(base)
        if p is None:
            return None
        im = Image.open(p).convert("RGBA")
        return self._scale_h(im, target_h)

    @staticmethod
    def _scale_h(im: Image.Image, target_h: int) -> Image.Image:
        if im.height == target_h or im.height == 0:
            return im
        w = max(1, int(im.width * target_h / im.height))
        return im.resize((w, target_h), Image.LANCZOS)

    def _glyphs(self, prefix: str, target_h: int) -> dict:
        chars = {str(i): str(i) for i in range(10)}
        chars.update({"comma": "comma", "dot": "dot", "percent": "percent", "x": "x"})
        out = {}
        for key, suffix in chars.items():
            im = self._load(f"{prefix}-{suffix}", target_h)
            # Some skins ship a decorative banner as score-percent (seen 1016x20).
            # A real digit/symbol glyph is roughly square; reject absurd aspect so
            # it can't balloon the number image (overlaps HUD + covers the field).
            if im is not None and im.width <= im.height * 3:
                out[key] = im
        return out

    def _text_badge(self, text: str, h: int) -> Image.Image:
        """A danser-style pill badge (e.g. 'Stable') sized like a mod icon."""
        f = _font(int(h * 0.5))
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bb = tmp.textbbox((0, 0), text, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = int(h * 0.28)
        W, H = tw + pad * 2, h
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H // 4,
                            fill=(38, 40, 54, 235), outline=(255, 255, 255, 210),
                            width=max(1, H // 20))
        d.text(((W - tw) // 2 - bb[0], (H - th) // 2 - bb[1]), text, font=f,
               fill=(255, 255, 255, 255))
        return img

    def _mods(self, mods: int, target_h: int) -> list:
        out = []
        seen = set()
        for bit, name in _MODS:
            if mods & bit and name not in seen:
                if name == "doubletime" and (mods & 512):
                    continue  # nightcore icon already covers it
                im = self._load(f"selection-mod-{name}", target_h)
                if im is not None:
                    out.append(im)
                    seen.add(name)
        return out

    @staticmethod
    def _grade(acc: float) -> str:
        # same cutoffs as _catch_grade (lazer/wiki); "X" = SS ranking sprite
        if acc >= 1.0:
            return "X"
        if acc >= 0.98:
            return "S"
        if acc >= 0.94:
            return "A"
        if acc >= 0.90:
            return "B"
        if acc >= 0.85:
            return "C"
        return "D"


def _catch_grade(acc_pct: float, misses: int) -> str:
    """osu!catch grade thresholds (lazer CatchScoreProcessor + osu! wiki;
    same cutoffs on stable). SS requires a perfect 100%."""
    if acc_pct >= 100.0:
        return "SS"
    if acc_pct >= 98.0:
        return "S"
    if acc_pct >= 94.0:
        return "A"
    if acc_pct >= 90.0:
        return "B"
    if acc_pct >= 85.0:
        return "C"
    return "D"


# match osu_renderer (mania) results-card grade colours exactly
_GRADE_COLOURS = {
    "SS": (240, 220, 120), "X": (240, 220, 120), "XH": (240, 220, 120),
    "S": (240, 220, 120), "SH": (240, 220, 120),
    "A": (110, 220, 130), "B": (110, 180, 220),
    "C": (200, 130, 220), "D": (220, 110, 110),
}


def _ellipsize(draw, text: str, font, max_w: int) -> str:
    """Trim text with a trailing … so it fits within max_w pixels."""
    if draw.textbbox((0, 0), text, font=font)[2] <= max_w:
        return text
    while text and draw.textbbox((0, 0), text + "…", font=font)[2] > max_w:
        text = text[:-1]
    return (text + "…") if text else ""


def draw_results(rgb, meta, bm, opacity: float):
    """Post-game results card, matching osu_renderer's _draw_results_overlay
    (same vertical stack, sizes, grade colours, font) for cross-mode
    consistency — minus mania's UR/histogram, which catch has no concept of.
    Uses the replay's authoritative counts.
    """
    import numpy as np
    a = max(0.0, min(1.0, opacity))
    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    img = Image.alpha_composite(img, Image.new("RGBA", img.size, (0, 0, 0, int(0.7 * a * 255))))
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx = W // 2
    y = int(H * 0.10)
    A = int(a * 255)

    def line(size, text, color, gap):
        nonlocal y
        font = _font(size)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text((cx - tw // 2 - bbox[0], y - bbox[1]), text, font=font, fill=(*color, A))
        y += th + gap

    caught = meta.count_300 + meta.count_100 + meta.count_50
    total = caught + meta.count_katu + meta.count_miss
    acc = (100.0 * caught / total) if total else 100.0
    grade = _catch_grade(acc, meta.count_miss + meta.count_katu)
    gc = _GRADE_COLOURS.get(grade, (200, 200, 220))
    line(220, grade, gc, int(H * 0.02))
    line(96, f"{meta.score:,}", (255, 255, 255), 10)
    line(56, f"{acc:.2f}%", (235, 235, 245), 10)
    line(40, f"Max combo {meta.max_combo}x", (200, 200, 220), 24)

    # catch judgment row: Fruit / Drop / Droplet / Miss, colour-coded
    cells = [("Fruit", meta.count_300, (255, 230, 120)),
             ("Drop", meta.count_100, (140, 220, 140)),
             ("Droplet", meta.count_50, (180, 180, 180)),
             ("Miss", meta.count_miss + meta.count_katu, (240, 80, 80))]
    f36 = _font(36)
    rendered = [(f"{lab}: {cnt}", col) for lab, cnt, col in cells]
    widths = [d.textbbox((0, 0), t, font=f36)[2] for t, _ in rendered]
    gap = 28
    x = cx - (sum(widths) + gap * (len(rendered) - 1)) // 2
    for (t, col), w in zip(rendered, widths):
        d.text((x, y), t, font=f36, fill=(*col, A))
        x += w + gap

    title = f"{bm.artist} - {bm.title} [{bm.version}]".strip(" -")
    rf = _font(26)
    full = _ellipsize(d, f"{meta.player_name}  ·  {title}", rf, int(W * 0.92))
    fb = d.textbbox((0, 0), full, font=rf)
    d.text((cx - (fb[2] - fb[0]) // 2, y + 70), full, font=rf, fill=(180, 180, 200, A))

    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


from .fonts import font as _font  # skin-aware, host-robust font resolver
