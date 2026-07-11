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
        self.dir = CatchSkin._resolve_root(Path(skin_dir)) if skin_dir else None
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
        # Per-element skin honoring (STD-style): if the uploaded skin ships a
        # number font, draw score / accuracy / combo from ITS glyphs instead of
        # the Argon counters (like STD's per-element approach). A skinless
        # render never loads glyphs, so it stays 100% Argon. The fruit/catcher
        # already switch to the skin in scene.py; this covers the HUD numbers.
        self._use_skin_hud = bool(self.score_glyphs)
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
        # Argon health bar at STD's exact placement (ArgonSkin: HP_POS
        # (50,20) lazer px, fixed Width=300 — the fractions below make the
        # internal k == lk, so every lazer constant lands 1:1 like STD's).
        from .argon_health import ArgonHealth
        from .argon_hud import (HP_POS, HP_WIDTH, LAZER_UI_HEIGHT, ArgonHud,
                                density_buckets)
        lk = self.h / LAZER_UI_HEIGHT
        self.argon_hp = ArgonHealth(self.w, self.h,
                                    width_frac=HP_WIDTH * lk / self.w,
                                    left_frac=HP_POS[0] * lk / self.w,
                                    top_frac=HP_POS[1] / LAZER_UI_HEIGHT)
        self._hp_last_t = None
        # key-counter input derivation state: previous catcher x (osu px).
        # L/R held = the SIGN of the catcher's x movement since the last
        # frame (replays store positions, not keys); dash comes straight
        # from the replay frame's dash bit via the scene.
        self._kc_prev_x = None
        # STD's Argon counters (score / accuracy / combo + wedges + song
        # progress) — argon_hud.py is the 1:1 port of the STD renderer's
        # Argon HUD components (owner directive: identical HUD across modes)
        starts = ([o.time_ms for o in getattr(beatmap, "objects", [])]
                  or [first_ms])
        self.argon = ArgonHud(self.w, self.h, first_ms, self.last_ms,
                              density=density_buckets(starts, starts))

    # --- public ---------------------------------------------------------------

    def overlay(self, rgb: np.ndarray, scene) -> np.ndarray:
        # Faithful osu!lazer **Argon** HUD (ArgonSkin.cs default layout),
        # drawn by argon_hud.ArgonHud — the STD renderer's Argon score /
        # accuracy / combo components ported 1:1 (owner directive 2026-07:
        # catch's HUD must look identical to STD's; only platter + fruit
        # differ per mode). Values (score/acc/combo/hp/pp) still come from
        # catch's own sim — only the visual rendering is STD's.
        img = Image.fromarray(rgb, "RGB")
        # An uploaded skin that ships its own number font drives a skin HUD
        # (score/combo/accuracy from the skin's glyphs); skinless stays Argon.
        if self._use_skin_hud:
            return self._overlay_skin(img, scene)
        ah = self.argon
        t = float(scene.time_ms)
        cfg = self.cfg

        def _on(n, default=True):
            return cfg is None or getattr(cfg, n, default)

        # ---- SCORE WEDGES (ArgonWedgePiece backdrops, UNDER bar + score) ----
        if _on("show_score") or _on("show_hp_bar"):
            ah.draw_wedges(img)

        # ---- HEALTH BAR (top-left; catch's argon_health at STD's spot) ----
        if _on("show_hp_bar"):
            dt = (16.0 if self._hp_last_t is None
                  else max(0.0, min(100.0, t - self._hp_last_t)))
            self._hp_last_t = t
            arr = np.asarray(img).copy()
            self.argon_hp.update_draw(arr, scene.hp, dt)
            img = Image.fromarray(arr)
            # the 45×3 BoxElement healthLine at (0, 30) lazer px (STD detail)
            ah.draw_health_line(img)

        if _on("show_score"):
            # ---- SCORE (top-left, right edge x=250 lazer px, 6 wireframes) ----
            ah.draw_score(img, t, scene.score)
            # ---- ACCURACY (top-right) + STD's procedural grade badge ----
            grade = None
            if (_on("show_grade") and t >= self.first_ms
                    and sum(scene.counts) > 0):
                grade = _catch_grade(scene.accuracy * 100.0, scene.counts[4])
            ah.draw_accuracy(img, t, scene.accuracy, grade=grade)
            # ---- PP (top-right, under accuracy; catch house element) ----
            if (cfg is not None and getattr(cfg, "show_pp_counter", False)
                    and scene.pp > 0):
                ah.draw_pp(img, scene.pp)

        # ---- COMBO (bottom-left ×1.3, pop + red break flash, '<n>x') ----
        if _on("show_combo"):
            ah.draw_combo(img, t, scene.combo)

        # ---- ArgonSongProgress strip (bottom, 90% width) ----
        if _on("show_progress"):
            ah.draw_progress(img, t)

        # ---- Argon KEY COUNTER (bottom-right; B1=left B2=right B3=dash) ----
        # Held states are derived, not stored: replays carry the catcher's
        # absolute x + the dash bit, so L/R = the direction the catcher
        # moved since the previous rendered frame (small deadzone kills
        # interpolation jitter; real walk speed is ~8 osu px/frame at 60fps).
        if _on("show_key_counter"):
            x = float(getattr(scene, "catcher_x", 0.0))
            dashing = bool(getattr(scene, "dashing", False))
            dx = 0.0 if self._kc_prev_x is None else x - self._kc_prev_x
            self._kc_prev_x = x
            dead = 0.05                       # osu px per frame
            ah.draw_key_counter(img, t, (dx < -dead, dx > dead, dashing))

        # watermark (bottom-right) — free renders are forced to the site URL
        self._draw_watermark(img)
        return np.asarray(img)

    # --- skin HUD (per-element skin honoring) ---------------------------------

    def _overlay_skin(self, img, scene) -> np.ndarray:
        """Legacy-skin HUD: draw score / accuracy / combo from the SKIN's own
        number glyphs (STD-style per-element honoring), HP from the skin's
        scorebar. Only reached when the skin ships a score font — skinless
        renders never get here (they stay all-Argon). The catcher/fruit come
        from the skin via scene.py; progress is a thin bar."""
        cfg = self.cfg
        W, H = self.w, self.h

        def _on(n, default=True):
            return cfg is None or getattr(cfg, n, default)

        pad = int(W * 0.010)
        top = int(H * 0.020)
        # HP: skin scorebar (or its built-in PIL fallback when scorebar_bg is None)
        if _on("show_hp_bar"):
            self._draw_scorebar(img, max(0.0, min(1.0, scene.hp)))
        # SCORE (top-right) + ACCURACY under it + grade badge
        if _on("show_score"):
            num = self._number(str(max(int(scene.score), 0)), self.score_glyphs, 0)
            self._paste(img, num, W - pad - num.width, top)
            acc_txt = f"{max(0.0, min(1.0, scene.accuracy)) * 100:.2f}%"
            accg = self.acc_glyphs or self.score_glyphs
            accimg = self._number(acc_txt, accg, 0)
            acc_y = top + num.height + int(H * 0.006)
            self._paste(img, accimg, W - pad - accimg.width, acc_y)
            if (_on("show_grade") and scene.time_ms >= self.first_ms
                    and sum(scene.counts) > 0):
                g = _catch_grade(scene.accuracy * 100.0, scene.counts[4])
                gkey = {"SS": "X"}.get(g, g)      # SS uses the ranking-X sprite
                gim = self.grades.get(gkey)
                if gim is not None:
                    self._paste(img, gim,
                                W - pad - max(num.width, accimg.width)
                                - int(W * 0.006) - gim.width, top)
        # COMBO (bottom-left)
        if _on("show_combo"):
            cg = self.combo_glyphs or self.score_glyphs
            cimg = self._number(f"{max(int(scene.combo), 0)}x", cg, 0)
            self._paste(img, cimg, pad, H - pad - cimg.height)
        # PROGRESS (thin bar, bottom)
        if _on("show_progress"):
            self._draw_progress(img, int(scene.time_ms))
        # watermark (bottom-right)
        self._draw_watermark(img)
        return np.asarray(img)

    def _draw_watermark(self, img) -> None:
        wm = getattr(self.cfg, "watermark", "") if self.cfg else ""
        if not wm:
            return
        pad = int(self.w * 0.012)
        d = ImageDraw.Draw(img)
        wmf = _font(int(self.h * 0.022))
        wb = d.textbbox((0, 0), wm, font=wmf)
        d.text((self.w - pad - (wb[2] - wb[0]),
                self.h - pad - (wb[3] - wb[1]) - int(self.h * 0.006)),
               wm, font=wmf, fill=(238, 238, 245))

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
        if self.dir is None:
            return None
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


# the per-render lazer results screen (baked once, drawn per outro frame).
# Keyed on id(meta) so a long-lived process can render several replays;
# False = the bake failed once → stay on the legacy card for this render.
_LAZER_RESULTS: dict = {}


def draw_results(rgb, meta, bm, opacity: float, board=None, age_ms=None,
                 osu_path=None, sim=None):
    """The osu!catch RESULTS SCREEN — the osu!(lazer) ranking screen, the
    faithful port of the std renderer's render/lazer_results.py (owner spec
    2026-07: results-screen parity with std). The screen itself lives in
    lazer_results.CatchLazerResults: black background, the AccuracyCircle
    (arc + graded ring + rank badges + glowing grade letter), the rounded
    featured panel with catch's Fruit/Drop/Droplet/Miss judgments, and the
    flank leaderboard cards laid out around it exactly like std.

    `age_ms` (ms since the results started) drives the ported two-stage
    animation (stage-1 reveal, then the stage-2 stats panels unfolding from
    the right); None (legacy callers) renders the settled screen. `sim` (the
    CatchSim) feeds the stage-2 COMBO panel its checkpoint series; None →
    the panel falls back to the rosu strain curve. Fully fail-soft: any
    problem falls back to the legacy text card — LOUDLY."""
    try:
        key = id(meta)
        scr = _LAZER_RESULTS.get(key)
        if scr is None:
            from .lazer_results import CatchLazerResults
            _LAZER_RESULTS.clear()          # one render at a time
            scr = CatchLazerResults((rgb.shape[1], rgb.shape[0]), meta, bm,
                                    board=board, osu_path=osu_path, sim=sim)
            _LAZER_RESULTS[key] = scr
        if scr is False:                    # earlier bake failed → legacy
            return _draw_results_legacy(rgb, meta, bm, opacity, board=board)
        return scr.render_frame(rgb, opacity, age_ms)
    except Exception as e:  # noqa: BLE001 — results must never kill a render
        import sys
        import traceback
        print("[catch-renderer] !!! LAZER RESULTS SCREEN FAILED — falling "
              f"back to the legacy results card: {e}", file=sys.stderr)
        traceback.print_exc()
        _LAZER_RESULTS[id(meta)] = False
        return _draw_results_legacy(rgb, meta, bm, opacity, board=board)


def _draw_results_legacy(rgb, meta, bm, opacity: float, board=None):
    """LEGACY results card (pre-2026-07 lazer-parity port) — the centred text
    stack over the dimmed gameplay frame. Kept intact as the fail-soft
    fallback for draw_results and for an emergency revert.

    Post-game results card, matching osu_renderer's _draw_results_overlay
    (same vertical stack, sizes, grade colours, font) for cross-mode
    consistency — minus mania's UR/histogram, which catch has no concept of.
    Uses the replay's authoritative counts.

    `board` (an lb_cards.BakedBoard | None): the per-map render leaderboard —
    compact ranked flank cards of the OTHER renders of this map, drawn around
    the centred stack (parity with the std renderer). None (leaderboard off /
    solo render / no other renders) → the plain results card, UNCHANGED.
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

    # When the map leaderboard is shown, the flank cards hug a centre column of
    # `CENTER_CLEAR_FRAC * W`; render the wide judgment row as a 2×2 grid and the
    # caption as the player name alone so the featured stack stays inside that
    # column and never collides with the flanks. Boardless → the original single
    # judgment row + full player·title caption (existing renders unchanged).
    compact = board is not None and getattr(board, "compact", False)
    if compact:
        from .lb_cards import CENTER_CLEAR_FRAC
        col_w = int(W * CENTER_CLEAR_FRAC * 0.98)
    else:
        col_w = int(W * 0.92)

    # catch judgment row: Fruit / Drop / Droplet / Miss, colour-coded
    cells = [("Fruit", meta.count_300, (255, 230, 120)),
             ("Drop", meta.count_100, (140, 220, 140)),
             ("Droplet", meta.count_50, (180, 180, 180)),
             ("Miss", meta.count_miss + meta.count_katu, (240, 80, 80))]
    f36 = _font(36)
    rendered = [(f"{lab}: {cnt}", col) for lab, cnt, col in cells]
    widths = [d.textbbox((0, 0), t, font=f36)[2] for t, _ in rendered]
    gap = 28
    row_h = d.textbbox((0, 0), "Ay", font=f36)[3] + 14
    if compact:
        # 2×2 grid, two centred cells per row
        for r in range(2):
            pair = rendered[2 * r:2 * r + 2]
            pw = widths[2 * r:2 * r + 2]
            x = cx - (sum(pw) + gap * (len(pair) - 1)) // 2
            for (t, col), w in zip(pair, pw):
                d.text((x, y), t, font=f36, fill=(*col, A))
                x += w + gap
            y += row_h
        title_y = y + 12
    else:
        x = cx - (sum(widths) + gap * (len(rendered) - 1)) // 2
        for (t, col), w in zip(rendered, widths):
            d.text((x, y), t, font=f36, fill=(*col, A))
            x += w + gap
        title_y = y + 70

    rf = _font(26)
    if compact:
        # the map title already rides the leaderboard banner up top; keep the
        # caption to the player so it fits the centre column.
        caption = meta.player_name
    else:
        title = f"{bm.artist} - {bm.title} [{bm.version}]".strip(" -")
        caption = f"{meta.player_name}  ·  {title}"
    full = _ellipsize(d, caption, rf, col_w)
    fb = d.textbbox((0, 0), full, font=rf)
    d.text((cx - (fb[2] - fb[0]) // 2, title_y), full, font=rf, fill=(180, 180, 200, A))

    out = Image.alpha_composite(img, layer)
    # per-map render leaderboard: flank cards around the centred stack (parity
    # with the std renderer). No-op when there's no board → renders unchanged.
    # Fail-soft: a compositing error must never crash the render mid-loop; the
    # frame just falls back to the plain results card.
    if board is not None:
        try:
            from .lb_cards import draw_board
            draw_board(out, board, opacity)
        except Exception:  # noqa: BLE001 — a board never breaks a render
            pass
    return np.asarray(out.convert("RGB"))


from .fonts import font as _font  # skin-aware, host-robust font resolver
