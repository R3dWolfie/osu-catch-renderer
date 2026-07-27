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

# osu mod bit -> selection-mod-<name>. Nightcore (512) supersedes DT; Perfect (16384) supersedes SD.
_MODS = [
    (2, "easy"), (8, "hidden"), (16, "hardrock"), (512, "nightcore"),
    (64, "doubletime"), (256, "halftime"), (1024, "flashlight"),
    (1, "nofail"), (16384, "perfect"), (32, "suddendeath"),
    (128, "relax"), (4096, "spunout"), (2048, "autoplay"),
]


class _CatchHud:
    """HudData-shaped adapter over a CatchSim for the std versus-merge
    `MergeBoard`: exposes `score_at(t)` (the sim's standardised ScoreV3) and
    `acc_at(t)` — the only two the board reads."""
    __slots__ = ("_sim",)

    def __init__(self, sim):
        self._sim = sim

    def score_at(self, t):
        cp = self._sim.state_at(int(min(float(t), 1e9)))
        return int(cp.score * getattr(self._sim, "score_scale", 1.0))

    def acc_at(self, t):
        cp = self._sim.state_at(int(min(float(t), 1e9)))
        return max(0.0, min(1.0, cp.accuracy))


class DanserHud:
    def __init__(self, skin_dir: Path, resolution, meta, beatmap,
                 first_ms: int, last_ms: int, cfg=None, default_skin_dir=None):
        self.cfg = cfg
        # Resolve to the real skin root the same way CatchSkin does, so the HUD
        # finds score/combo/ranking/mod sprites for ANY skin whose assets live
        # in a subdir (the bundled default's `_default-source`, single-folder
        # .osk archives, etc.) — not just skins with skin.ini at the top level.
        from .skin import CatchSkin
        self.dir = CatchSkin._resolve_root(Path(skin_dir)) if skin_dir else None
        # DEFAULT-SKIN FALLBACK (osu behaviour: an element the skin doesn't ship
        # comes from the default skin). CatchSkin already did this for the
        # fruit/catcher (render.py passes cfg.default_skin_dir); the HUD did NOT,
        # so a skin with a score font but no scorebar / inputoverlay silently fell
        # back to ARGON pieces instead of the real legacy ones. Search order is
        # user skin → default skin.
        self.default_dir = (CatchSkin._resolve_root(Path(default_skin_dir))
                            if default_skin_dir else None)
        self.w, self.h = resolution
        self.meta = meta
        self.bm = beatmap
        self.first_ms = first_ms
        self.last_ms = max(last_ms, first_ms + 1)
        H = self.h
        # Honor skin.ini [Fonts] ScorePrefix / ComboPrefix — a skin can put its
        # number font in a subfolder (e.g. "Fonts/score/score") instead of the
        # top-level "score"/"combo". Without this the HUD looks for score-0.png
        # at the root, finds nothing, and falls back to the Argon counters even
        # though the skin ships a full font (the "missing skinned fonts" bug).
        score_prefix, combo_prefix = self._font_prefixes()
        # USER SKIN ONLY (default_ok=False): this doubles as the skin-HUD probe
        # below, and the default skin ships score-* — letting it satisfy this
        # would flip every skinless render out of the Argon HUD.
        self.score_glyphs = self._glyphs(score_prefix, int(H * 0.050),
                                         default_ok=False)
        self.acc_glyphs = self._glyphs(score_prefix, int(H * 0.032))
        # legacy catch combo uses the combo font, falling back to the score font
        # (stable's default ComboPrefix == ScorePrefix == "score").
        self.combo_glyphs = (self._glyphs(combo_prefix, int(H * 0.072))
                             or self._glyphs(score_prefix, int(H * 0.072)))
        # Per-element skin honoring (STD-style): if the uploaded skin ships a
        # number font, draw score / accuracy / combo from ITS glyphs instead of
        # the Argon counters (like STD's per-element approach). A skinless
        # render never loads glyphs, so it stays 100% Argon. The fruit/catcher
        # already switch to the skin in scene.py; this covers the HUD numbers.
        # LAYOUT decision — lazer renders a LEGACY layout (score top-right, HP
        # top-left, combo, key overlay) for any legacy skin, and the Argon layout
        # only when there's no legacy skin at all. It is NOT keyed on the score
        # font: `_use_skin_hud = bool(score_glyphs)` was an all-or-nothing switch,
        # so a skin shipping scorebar + inputoverlay but no score font was thrown
        # to the FULL Argon HUD and showed none of its own elements. Every element
        # below now resolves independently (skin asset -> lazer's own default).
        self._use_skin_hud = self.dir is not None and (
            bool(self.score_glyphs)
            or (self.dir / "skin.ini").is_file()
            or self._resolve("scorebar-bg", default_ok=False) is not None
            or self._resolve("inputoverlay-key", default_ok=False) is not None
            or self._resolve("fruit-catcher-idle", default_ok=False) is not None
            or self._resolve("fruit-ryuuta", default_ok=False) is not None
        )
        # Load the scorebar pieces at NATIVE size. They must share ONE scale
        # factor: normalising each to the same HEIGHT (the old behaviour) ignores
        # their different native aspect ratios, so the frame and the fill came out
        # at wildly different widths and didn't line up at all.
        # USER SKIN ONLY. lazer falls back to its OWN default health display
        # (Argon) when a skin ships no scorebar — NOT to another skin's. Our
        # bundled `_default-source` IS Night05, so allowing the default here
        # painted Night05's purple starry scorebar onto every skin. Verified
        # against Red's lazer captures: Night05/VOEZ/TOMAT-OS show their own
        # scorebar; a skin without one shows lazer's Argon bar.
        self.scorebar_bg = self._load_native("scorebar-bg", default_ok=False)
        # stable animates scorebar-colour-{n}; a non-animated skin ships plain
        # scorebar-colour. Only looking for "-0" left such skins with NO fill.
        self.scorebar_col = (
            self._load_native("scorebar-colour-0", default_ok=False)
            or self._load_native("scorebar-colour", default_ok=False))
        # Does the USER's skin ship its own scorebar frame? If so we composite
        # their two sprites the legacy way. If not, the DEFAULT skin's frame is
        # used only as a colour source — its art sits inside a largely
        # transparent, faintly-alpha'd canvas, so it can't be registered against
        # the fill reliably; we draw a clean track instead (see _draw_scorebar).
        #
        # LIVE grade badge — owner decision 2026-07-22: STRICTLY the user
        # skin's own ranking art, never the default skin's (a skin without
        # ranking icons shows NO live badge and the score/acc/pie block lays
        # out without the gap). Stable draws ranking-{g}-SMALL next to the
        # accuracy — prefer that art at NATIVE logical size × H/768 (the same
        # rule as every legacy HUD sprite; the old load squashed the BIG
        # results-screen ranking-* into a fixed 0.060H box). A skin shipping
        # only the big art gets it scaled to the stable small-badge height.
        # Scaled ONCE here — per-frame resizes were part of the jitter.
        self.grades = {}
        _k768 = H / 768.0
        for g in ("X", "XH", "S", "SH", "A", "B", "C", "D"):
            gim = self._load_native(f"ranking-{g}-small", default_ok=False)
            if gim is not None:
                gw = max(1, int(round(gim.width * _k768)))
                gh = max(1, int(round(gim.height * _k768)))
                if (gw, gh) != gim.size:
                    gim = gim.resize((gw, gh), Image.LANCZOS)
            else:
                big = self._load_native(f"ranking-{g}", default_ok=False)
                if big is not None and big.height > 0:
                    gh = max(1, int(round(40 * _k768)))   # stable -small height
                    gim = big.resize(
                        (max(1, int(big.width * gh / big.height)), gh),
                        Image.LANCZOS)
            self.grades[g] = gim
        self.mod_imgs = self._mods(meta.mods, int(H * 0.052))
        # danser-style client badge: mark osu!stable replays (game_version < 30M).
        # Kept as the LAST mod_imgs entry (leftmost in the right-to-left stack)
        # and ALSO remembered on its own: the skinned layout separates the pill
        # from the icon stack with a clean gap (see _draw_mod_icons) — the
        # 16-unit stable overlap between the rectangular pill and a circular
        # selection-mod icon left only an unreadable sliver of the icon poking
        # out of the pill's right edge (the "mini-chip" of the 2026-07-22
        # top-right jumble report).
        self.client_badge = None
        if 0 < getattr(meta, "game_version", 0) < 30000000:
            self.client_badge = self._text_badge("Stable", int(H * 0.052))
            self.mod_imgs.append(self.client_badge)
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
        # LEGACY key overlay (osu default when the skin ships no key element).
        # catch has THREE inputs — left / right / dash — not std's B1/B2/B3,
        # which is what the borrowed Argon counter was mislabelling them as.
        # USER SKIN ONLY: the *default* skin's inputoverlay-key is a near-black
        # button that's only legible over its background bar; without that bar it
        # vanishes. So honour a skin that ships its own key art, else fall back to
        # a clean drawn key box below (guaranteed readable over any gameplay).
        self.key_bg = self._load("inputoverlay-background", int(H * 0.052),
                                 default_ok=False)
        # NATIVE LOGICAL size (@2x halved), like every legacy sprite: stable/
        # lazer draw inputoverlay-key at texture size in the 768-tall HUD space
        # (LegacyKeyCounter: fixed 48x46 cell, sprite centred ON it, free to
        # overlap the neighbours). Skins pad the button art with transparency
        # to centre it on the overlay bar — squeezing the whole padded canvas
        # into the key cell shrank the visible button (~58% coverage on padded
        # sprites), which read as a blown-up gap between the stacked keys.
        self.key_img = self._load_native("inputoverlay-key", default_ok=False)
        # The skin's OWN lazer HUD arrangement (MainHUDComponents.json), when it
        # ships one. Empty dict -> default legacy placement below.
        from .lazer_hud import load_layout
        self.layout = load_layout(self.dir)
        self._kc_counts = [0, 0, 0]
        self._roll = {}     # rolling-counter state: key -> (from, to, start_t)
        self._kc_held_prev = (False, False, False)
        # legacy key-press animation (lazer LegacyKeyCounter, see
        # _draw_key_overlay): per key (scale_at_edge, target, edge_ms) —
        # each edge retweens FROM the scale it interrupted — plus the
        # first-activation time that drives the name->count crossfade.
        self._kc_anim = [(1.0, 1.0, float("-inf"))] * 3
        self._kc_first_press_t = [None, None, None]
        # STD's Argon counters (score / accuracy / combo + wedges + song
        # progress) — argon_hud.py is the 1:1 port of the STD renderer's
        # Argon HUD components (owner directive: identical HUD across modes)
        starts = ([o.time_ms for o in getattr(beatmap, "objects", [])]
                  or [first_ms])
        self.argon = ArgonHud(self.w, self.h, first_ms, self.last_ms,
                              density=density_buckets(starts, starts))
        # lazer's BreakOverlay (countdown + progress bar + CURRENT PROGRESS
        # info + slide-in chevrons) — break_overlay.py, a 1:1 port of
        # osu.Game/Screens/Play/BreakOverlay.cs. Drawn on BOTH HUD paths:
        # stable has no equivalent panel and the owner wants the lazer look
        # on skinned renders too, so it's deliberately lazer-styled always
        # (independent of --letterbox-breaks, which lazer doesn't gate on).
        from .break_overlay import LazerBreakOverlay
        self.break_overlay = LazerBreakOverlay(
            self.w, self.h, getattr(beatmap, "breaks", None) or [],
            mods=int(getattr(meta, "mods", 0) or 0))

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

        # VERSUS OVERLAY (skinless): leaderboard + shared progress, no single HUD.
        board = getattr(scene, "overlay_board", None)
        if board:
            if _on("show_progress"):
                ah.draw_progress(img, t)
            self._draw_overlay_board(img, board, t)
            self._draw_watermark(img)
            return np.asarray(img)
        # ---- SCORE WEDGES (ArgonWedgePiece backdrops, UNDER bar + score) ----
        if _on("show_score") or _on("show_hp_bar"):
            ah.draw_wedges(img)

        # ---- HEALTH BAR (Argon health display; skinless renders only) ----
        if _on("show_hp_bar"):
            dt = (16.0 if self._hp_last_t is None
                  else max(0.0, min(100.0, t - self._hp_last_t)))
            self._hp_last_t = t
            # PERF: the bar only ever touches its own clip box — round-trip
            # just that crop through numpy instead of the whole frame (the
            # composites clip identically inside it; pixels are unchanged).
            xa, ya, xb, yb = self.argon_hp.clip_box(self.w, self.h)
            if xb > xa and yb > ya:
                sub = np.array(img.crop((xa, ya, xb, yb)))
                self.argon_hp.update_draw(sub, scene.hp, dt, origin=(xa, ya))
                img.paste(Image.fromarray(sub, "RGB"), (xa, ya))
            else:                       # fully off-screen: advance state only
                self.argon_hp.update_draw(
                    np.zeros((1, 1, 3), np.uint8), scene.hp, dt,
                    origin=(-10**9, -10**9))
            ah.draw_health_line(img)

        if _on("show_score"):
            ah.draw_score(img, t, scene.score)
            grade = None
            if (_on("show_grade") and t >= self.first_ms
                    and sum(scene.counts) > 0):
                grade = _catch_grade(scene.accuracy * 100.0, scene.counts[4])
            ah.draw_accuracy(img, t, scene.accuracy, grade=grade)
            if (cfg is not None and getattr(cfg, "show_pp_counter", False)
                    and scene.pp > 0):
                ah.draw_pp(img, scene.pp)

        if _on("show_combo"):
            ah.draw_combo(img, t, scene.combo)

        if _on("show_progress"):
            ah.draw_progress(img, t)

        if _on("show_key_counter"):
            held, counts = self._input_from_scene(scene)
            ah.draw_key_counter(img, t, held, counts=counts)

        if _on("show_mods"):
            # below the argon accuracy + pp block (pp bottom ≈ 102 lazer
            # units; see argon_hud.draw_pp) — right-aligned with them.
            from .argon_hud import ARGON_ACC_POS
            right = int((ah.ui_w_l + ARGON_ACC_POS[0] * ah.es) * ah.lk)
            top = int(115.0 * ah.es * ah.lk)
            self._draw_mod_icons(img, t, right, top)

        # HIT COUNTER — STD's house element (osu-std render/hud.py
        # _hit_counter: live judgment rows under the whole top-right
        # stack, drawn on BOTH of std's component paths). Below the mod
        # icon row when it's up — std clears its pill stack the same way —
        # else at the mod row's own slot under the acc + pp block.
        if cfg is not None and getattr(cfg, "show_hit_counter", False):
            top_l = 115.0 * ah.es
            if _on("show_mods") and self.mod_imgs:
                top_l += (max(im.height for im in self.mod_imgs) / ah.lk
                          + 8.0 * ah.es)
            ah.draw_hit_counter(img, scene.counts, top_l)

        # lazer z-order: BreakOverlay is a LATER overlay-component child than
        # HUDOverlay (Player.createOverlayComponents), so it composites above
        # every HUD element. Watermark stays topmost (house element).
        self.break_overlay.draw(img, t, scene.accuracy)

        self._draw_watermark(img)
        return np.asarray(img)

    def _input_from_scene(self, scene):
        """(held, counts) for the key overlay. The sim now supplies REPLAY-
        FRAME-accurate state (scene.keys_held: any press/movement within this
        video frame's map-time interval; scene.key_counts: cumulative press
        onsets at replay resolution) — rapid taps register on every video
        frame they span instead of aliasing into holds, dash comes from the
        replay's real button bit, and DT/HT are inherent (the interval is
        map_step-sized). Fallback (overlay sims / older scenes): the legacy
        per-video-frame dx derivation, counts=None → HUD edge-counting."""
        kh = getattr(scene, "keys_held", None)
        if kh is not None:
            return tuple(kh), getattr(scene, "key_counts", None)
        x = float(getattr(scene, "catcher_x", 0.0))
        dashing = bool(getattr(scene, "dashing", False))
        dx = 0.0 if self._kc_prev_x is None else x - self._kc_prev_x
        self._kc_prev_x = x
        return (dx < -0.05, dx > 0.05, dashing), None


    def _overlay_skin(self, img, scene) -> np.ndarray:
        """LEGACY-skin HUD. lazer renders the legacy layout for any legacy skin,
        with EVERY element resolving independently: the skin's asset when it ships
        one, otherwise lazer's own default. Skinless renders never reach here."""
        cfg = self.cfg
        W, H = self.w, self.h
        ah = self.argon
        t = float(scene.time_ms)

        def _on(n, default=True):
            return cfg is None or getattr(cfg, n, default)

        pad = int(W * 0.010)
        top = int(H * 0.020)
        # VERSUS OVERLAY: a per-player leaderboard replaces the single-player HUD
        # (a lone score/acc/combo would be redundant + player-0-only). Keep the
        # shared song progress; the board carries everyone's score/acc/counts.
        board = getattr(scene, "overlay_board", None)
        if board:
            if _on("show_progress"):
                self._draw_song_progress_pie(
                    img, int(scene.time_ms),
                    getattr(self, "_acc_box", (W - 10, 10, 0, 0)))
                self._draw_argon_song_progress(img, int(scene.time_ms))
            self._draw_overlay_board(img, board, t)
            self._draw_watermark(img)
            return np.asarray(img)
        # ---- HEALTH BAR (top-left; catch's argon_health at STD's spot) ----
        if _on("show_hp_bar"):
            # LEGACY layout -> LegacyHealthDisplay (skin's scorebar sprites when it
            # ships them, else the classic-default look). NOT the Argon bar: a
            # legacy skin without a scorebar falls back to the classic DEFAULT
            # skin, not to Argon.
            self._draw_legacy_health(img, scene.hp, t)
        if _on("show_score") and not self.score_glyphs:
            # skin ships no number font -> lazer's own default counters.
            # Grade badge only when the USER skin ships ranking art (owner
            # rule 2026-07-22: skinned renders never invent a badge).
            ah.draw_score(img, t, scene.score)
            grade0 = None
            if (_on("show_grade") and t >= self.first_ms
                    and sum(scene.counts) > 0
                    and any(g is not None for g in self.grades.values())):
                grade0 = _catch_grade(scene.accuracy * 100.0, scene.counts[4])
            ah.draw_accuracy(img, t, scene.accuracy, grade=grade0)
        elif _on("show_score"):
            # LegacyScoreCounter: TopRight, Scale .96, Margin horizontal 10.
            disp = int(round(self._rolled("score", float(max(int(scene.score), 0)),
                                          t, 1000.0)))
            k = self.h / 768.0
            sc_c = self._comp("LegacyScoreCounter")
            s_scale = (sc_c.scale[1] if sc_c else 0.96)
            sh = max(1, int(self.h * 0.052 * (s_scale / 0.96)))
            # PERF: single-slot memo — whenever the rolled score is unchanged
            # from the previous frame (settled counter: any pause > the 1s
            # roll), reuse the assembled+resized digits. Same inputs -> same
            # bytes; a changed value rebuilds exactly as before.
            _sk = (disp, sh)
            _sm = getattr(self, "_score_num_memo", None)
            if _sm is not None and _sm[0] == _sk:
                num = _sm[1]
            else:
                num = self._number(f"{disp:06d}",
                                   self.score_glyphs, self._score_overlap())
                num = num.resize((max(1, int(num.width * sh / num.height)), sh),
                                 Image.LANCZOS)
                self._score_num_memo = (_sk, num)
            sx, sy = self._place("LegacyScoreCounter", num.width, num.height,
                                 (W - int(10 * k) - num.width, int(10 * k)))
            self._paste(img, num, sx, sy)
            # LegacyAccuracyCounter: TopRight, Scale .576; its TOP edge sits on
            # the score's BOTTOM edge (LegacySkin positions it at runtime).
            disp_acc = self._rolled("acc", max(0.0, min(1.0, scene.accuracy)),
                                    t, 375.0)
            acc_txt = self._acc_text(disp_acc)
            accg = self.acc_glyphs or self.score_glyphs
            ac_c = self._comp("LegacyAccuracyCounter")
            a_scale = (ac_c.scale[1] if ac_c else 0.576)
            ah_px = max(1, int(self.h * 0.052 * (a_scale / 0.96)))
            # PERF: single-slot memo, same pattern as the score digits — the
            # accuracy text settles 375ms after every change, so most frames
            # redraw an identical string. Same inputs -> same bytes.
            _ak = (acc_txt, ah_px)
            _am = getattr(self, "_acc_num_memo", None)
            if _am is not None and _am[0] == _ak:
                accimg = _am[1]
            else:
                accimg = self._number(acc_txt, accg, self._score_overlap())
                accimg = accimg.resize(
                    (max(1, int(accimg.width * ah_px / accimg.height)), ah_px),
                    Image.LANCZOS)
                self._acc_num_memo = (_ak, accimg)
            axd = W - int(17 * k) - accimg.width
            ayd = sy + num.height
            ax, ay = self._place("LegacyAccuracyCounter", accimg.width,
                                 accimg.height, (axd, ayd))
            self._paste(img, accimg, ax, ay)
            # FIXED anchor for the pie + grade badge (the "jittery top-right"
            # bug): the accuracy counter is right-aligned, so its LEFT edge
            # moves every frame while a 375ms roll ticks the digits through
            # different glyph widths — and the pie/badge were anchored to it.
            # Anchor them to the widest possible accuracy text ("100.00%")
            # instead, computed once per glyph scale: rock-solid positions,
            # and the badge can never collide with the accuracy digits.
            ref_w = getattr(self, "_acc_ref_w", None)
            if ref_w is None or getattr(self, "_acc_ref_h", None) != ah_px:
                refimg = self._number("100.00%", accg, self._score_overlap())
                ref_w = (max(1, int(refimg.width * ah_px / refimg.height))
                         if refimg.height else accimg.width)
                self._acc_ref_w, self._acc_ref_h = ref_w, ah_px
            # A skin whose lazer layout repositions the accuracy keeps the
            # live edge (custom layouts place the block wherever they like).
            ax_fix = ax if ac_c is not None else W - int(17 * k) - ref_w
            self._acc_box = (ax_fix, ay, ref_w, accimg.height)
            # ROW 2 GEOMETRY — all FIXED for the whole render (anchored on the
            # 100.00% reference width, so nothing moves as digits roll): the
            # song-progress pie immediately LEFT of the acc block with a small
            # fixed gap, the grade badge (user-skin ranking art only)
            # immediately LEFT of the pie with the same gap, both vertically
            # centred on the acc row. Replaces the old 18+33+8-unit badge
            # offset, which hardcoded the pie geometry a second time and read
            # as the badge floating with a big gap (2026-07-22 report).
            k768 = self.h / 768.0
            gap8 = max(2, int(8 * k768))
            pie_box = max(8, int(33 * k768))
            pie_x = ax_fix - gap8 - pie_box
            pie_y = int(ay + accimg.height / 2 - pie_box / 2)
            self._pie_xy = (pie_x, pie_y)
            if (_on("show_grade") and scene.time_ms >= self.first_ms
                    and sum(scene.counts) > 0):
                g = _catch_grade(scene.accuracy * 100.0, scene.counts[4])
                gim = self.grades.get({"SS": "X"}.get(g, g))
                if gim is not None:
                    # stable: the small grade sits on the ACCURACY ROW, left
                    # of the progress pie — art at native size (loaded
                    # pre-scaled: no squash, no per-frame resize).
                    bx = pie_x - gap8 - gim.width
                    by = int(ay + accimg.height / 2 - gim.height / 2)
                    self._paste(img, gim, bx, by)
        # COMBO — osu!CATCH does NOT use LegacyDefaultComboCounter: the catch
        # legacy transformer only supplies KeyCounter/SpectatorList/Leaderboard.
        # Catch's combo is LegacyCatchComboCounter, drawn ABOVE THE CATCHER and
        # tracking Catcher.X, fading out ~1s after the last increment — which is
        # why it appears at different x positions (and half-faded) in captures.
        if _on("show_combo"):
            self._draw_catch_combo(img, scene, t)
        # SONG PROGRESS — authentic: LegacySongProgress is a 33x33 circular PIE
        # and legacy has NO time text (SongProgressInfo is Default/Argon only).
        # A skin whose layout also asks for ArgonSongProgress (Red's, rotated -90)
        # additionally gets that bar.
        if _on("show_progress"):
            self._draw_song_progress_pie(img, int(scene.time_ms),
                                         getattr(self, "_acc_box",
                                                 (W - 10, 10, 0, 0)),
                                         xy=getattr(self, "_pie_xy", None))
            self._draw_argon_song_progress(img, int(scene.time_ms))
        # KEY COUNTER — the LEGACY inputoverlay (skin's own, else the default
        # skin's), labelled with catch's real inputs: left / right / dash.
        # Held = sign of catcher movement + the replay's dash bit.
        if _on("show_key_counter"):
            held, counts = self._input_from_scene(scene)
            self._draw_key_overlay(img, t, held, counts)
        # MOD ICONS — stable stacks them top-right under the accuracy row,
        # right-aligned with the accuracy block. ROW 3 clears the WHOLE of
        # row 2 (acc digits, pie AND grade badge): the clearance counts the
        # tallest loaded ranking art even while no badge is up yet, so the
        # row never shifts when the badge appears mid-play — fixed anchors,
        # zero overlap (the old top ignored the pie/badge heights and let
        # tall ranking art touch the mod row).
        k768 = H / 768.0
        ab = getattr(self, "_acc_box", None)
        if ab is not None and ab[3] > 0:
            bh_max = getattr(self, "_grade_h_max", None)
            if bh_max is None:
                bh_max = max((g.height for g in self.grades.values()
                              if g is not None), default=0)
                self._grade_h_max = bh_max
            row2_mid = ab[1] + ab[3] / 2.0
            half = max(ab[3] / 2.0, max(8, int(33 * k768)) / 2.0,
                       bh_max / 2.0)
            m_top = int(row2_mid + half + 12 * k768)
        else:
            m_top = int(H * 0.140)
        if _on("show_mods"):
            self._draw_mod_icons(img, t, W - int(17 * k768), m_top,
                                 pill_gap=max(2, int(8 * k768)))
        # HOUSE COUNTERS — pp + live hit tallies, top-right stack. std draws
        # these settings-surface elements identically on BOTH of its component
        # paths (osu-std render/hud.py _draw_timed: _mod_pills →
        # _hit_counter → _pp_counter, stacked under the mod row), and mania's
        # skinned path stacks its pp under the mod pills the same way — so
        # catch's skinned layout stacks them below the mod icon row (whose
        # top row-3 computed above), right-aligned with the accuracy block.
        # House (Argon) glyphs, NOT skin digits: no sibling engine draws its
        # counters from the skin font (std = procedural glyph bank on every
        # path, mania = PIL house text). Values are the sim's live scene
        # (scene.pp interpolated rosu checkpoints / scene.counts tallies).
        if cfg is not None and (getattr(cfg, "show_pp_counter", False)
                                or getattr(cfg, "show_hit_counter", False)):
            from .argon_hud import ARGON_DIGIT_H, ARGON_LABEL_GAP
            gap8 = max(2, int(8 * k768))
            stack_y = m_top
            if _on("show_mods") and self.mod_imgs:
                stack_y += max(im.height for im in self.mod_imgs) + gap8
            right_l = (W - int(17 * k768)) / ah.lk
            if getattr(cfg, "show_pp_counter", False):
                if scene.pp > 0:
                    ah.draw_pp(img, scene.pp, right_l=right_l,
                               top_l=stack_y / ah.lk)
                # slot RESERVED even while pp is 0 (intro, or rosu missing)
                # so the hit rows below never shift mid-play — fixed
                # anchors, the same rule as the row-2/row-3 geometry above.
                stack_y += int((ARGON_LABEL_GAP + ARGON_DIGIT_H * 0.6)
                               * ah.es * ah.lk) + gap8
            if getattr(cfg, "show_hit_counter", False):
                ah.draw_hit_counter(img, scene.counts, stack_y / ah.lk,
                                    right_l=right_l)
        # lazer BreakOverlay — SAME lazer-styled overlay as the argon path
        # (deliberate: stable has no break panel; owner wants THIS look on
        # skinned renders too). Above the HUD, per lazer's overlay z-order.
        self.break_overlay.draw(img, t, scene.accuracy)

        # watermark (bottom-right)
        self._draw_watermark(img)
        return np.asarray(img)

    def _draw_overlay_board(self, img, board, t) -> None:
        """Versus Overlay leaderboard — the REAL std versus-merge `MergeBoard`
        (osu-std, the glassy 'Rail' board), fed the catch sims' standardised
        ScoreV3 via a HudData-shaped adapter. One persistent MergeBoard instance
        (cached) so its row-slide animation runs; composited top-left each frame.
        `board` = [(name, colour, sim, end_ms), …]."""
        mb = getattr(self, "_merge_board", None)
        if mb is None:
            import os
            import sys
            std_pkg = os.environ.get("R3D_STD_PKG", "/home/foof/r3drender/osu-std")
            if std_pkg not in sys.path:
                sys.path.insert(0, std_pkg)
            try:
                from osu_std_renderer.merge import MergeBoard
                entries = [(name, col, _CatchHud(sim), end_ms)
                           for (name, col, sim, end_ms) in board]
                skin_dirs = [d for d in (self.dir, self.default_dir) if d]
                mb = self._merge_board = MergeBoard(entries, self.h,
                                                    skin_dirs=skin_dirs)
            except Exception:      # noqa: BLE001 — board never breaks a render
                self._merge_board = False
                return
        if mb is False:
            return
        try:
            _W, _H, arr = mb.render(float(t))
            bimg = Image.fromarray(arr, "RGBA")
            img.paste(bimg, (0, 0), bimg)
        except Exception:          # noqa: BLE001
            return

    def _draw_mod_icons(self, img, t: float, right_x: int, top_y: int,
                        pill_gap: int | None = None) -> None:
        """In-play mod icons — STABLE semantics (these are stable replays):
        the skin's selection-mod-<name> sprites (per-file user → default-skin
        resolution, prepared in __init__ as self.mod_imgs + the danser-style
        'Stable' badge), stacked top-right under the accuracy block RIGHT-TO-
        LEFT with a slight overlap, and kept faintly visible through the whole
        play (full alpha in the intro, settling to 0.65 as gameplay starts —
        lazer fades them out entirely, stable does not; owner picked stable).
        mod_imgs was previously BUILT AND NEVER DRAWN — --show-mods was a
        silent no-op on catch, argon and skinned alike.

        `pill_gap` (skinned layout only; None = exact legacy behaviour for the
        argon caller): the 'Stable' pill at the end of the stack steps a clean
        `pill_gap` px LEFT of the icons instead of overlapping the last one —
        the 16-unit stable overlap between the opaque pill and a circular mod
        icon read as a mystery second circle with an unreadable dark sliver
        ('mini-chip') poking out of the pill (2026-07-22 top-right jumble)."""
        if not self.mod_imgs:
            return
        a = (1.0 if t < self.first_ms
             else 1.0 - 0.35 * min(1.0, (t - self.first_ms) / 400.0))
        overlap = int(16 * (self.h / 768.0))
        x_right = right_x
        for im in self.mod_imgs:
            sp = im
            if a < 1.0:
                key = ("_mod_faded", id(im))
                cache = getattr(self, "_mod_fade_cache", None)
                if cache is None:
                    cache = self._mod_fade_cache = {}
                sp = cache.get(key)
                if sp is None:
                    sp = im.copy()
                    sp.putalpha(sp.getchannel("A").point(
                        lambda v: int(v * 0.65)))
                    cache[key] = sp
                if a > 0.66:   # still fading in the first 400ms: exact alpha
                    sp = im.copy()
                    sp.putalpha(sp.getchannel("A").point(
                        lambda v, _a=a: int(v * _a)))
            if (pill_gap is not None and x_right != right_x
                    and im is getattr(self, "client_badge", None)):
                # the pill after a non-empty icon stack: clean gap, no overlap
                x_right -= overlap + pill_gap
            self._paste(img, sp, x_right - sp.width, top_y)
            x_right -= sp.width - overlap
        return

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
        W = self.w
        if self.scorebar_bg is not None:
            # USER skin's own scorebar: composite their frame + fill at one shared
            # scale (never normalise each to the same height — different native
            # aspects would put the frame and fill at unrelated widths).
            k = (W * 0.42) / max(1, self.scorebar_bg.width)
            bg = self.scorebar_bg.resize(
                (max(1, int(self.scorebar_bg.width * k)),
                 max(1, int(self.scorebar_bg.height * k))), Image.LANCZOS)
            self._paste(img, bg, x0, y0)
            if self.scorebar_col is not None:
                col = self.scorebar_col.resize(
                    (max(1, int(self.scorebar_col.width * k)),
                     max(1, int(self.scorebar_col.height * k))), Image.LANCZOS)
                off_x = max(0, (bg.width - col.width) // 2)
                off_y = max(0, (bg.height - col.height) // 2)
                cw = max(1, int(col.width * hp))
                self._paste(img, col.crop((0, 0, cw, col.height)),
                            x0 + off_x, y0 + off_y)
            return
        # Fallback: clean track + the default skin's colour art as the fill.
        bw, bh = int(W * 0.42), max(8, int(self.h * 0.022))
        d = ImageDraw.Draw(img)
        r = bh // 2
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=r, fill=(24, 24, 32))
        cw = max(1, int(bw * hp))
        if self.scorebar_col is not None:
            fill = self.scorebar_col.resize((bw, bh), Image.LANCZOS)
            self._paste(img, fill.crop((0, 0, cw, bh)), x0, y0)
        else:
            d.rounded_rectangle([x0, y0, x0 + cw, y0 + bh], radius=r,
                                fill=(120, 220, 150))
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=r,
                            outline=(210, 210, 225), width=max(1, bh // 8))

    # lazer applies a per-component Margin IN ADDITION to Anchor/Origin/Position.
    # It lives in the component's C# ctor, not in MainHUDComponents.json, so it
    # has to be reapplied here or right-anchored elements sit flush to the edge.
    _MARGINS = {                       # type -> (horizontal, vertical) lazer units
        "LegacyScoreCounter": (10.0, 0.0),
        "LegacyAccuracyCounter": (17.0, 9.0),
        "LegacyDefaultComboCounter": (10.0, 10.0),
    }

    def _place(self, type_name: str, w: float, h: float,
               default_xy: tuple[int, int]) -> tuple[int, int]:
        """Top-left px for a w x h element: the skin's own lazer layout entry if
        it has one, else lazer's default legacy placement."""
        c = self.layout.get(type_name)
        if c is None:
            return default_xy
        x, y = c.place(self.w, self.h, w, h)
        mh, mv = self._MARGINS.get(type_name, (0.0, 0.0))
        if mh or mv:
            k = self.h / 768.0
            if c.anchor & 32:          # right-anchored -> inset leftwards
                x -= int(mh * k)
            elif c.anchor & 8:         # left-anchored -> inset rightwards
                x += int(mh * k)
            if c.anchor & 4:           # bottom-anchored -> inset upwards
                y -= int(mv * k)
            elif c.anchor & 1:         # top-anchored -> inset downwards
                y += int(mv * k)
        return x, y

    def _comp(self, type_name: str):
        return self.layout.get(type_name)

    @staticmethod
    def _legacy_fill_colour(hp: float):
        """LegacyHealthDisplay.getFillColour — NOTE the genuine discontinuity at
        hp=0.2 (source returns pure black just under it, 40% grey at it). Kept."""
        if hp < 0.2:
            u = max(0.0, min(1.0, (0.2 - hp) / 0.2))
            return (int(0 + (255 - 0) * u), 0, 0)          # black -> red
        if hp < 0.5:
            u = max(0.0, min(1.0, (0.5 - hp) / 0.5))
            v = int(255 * (1.0 - u))
            return (v, v, v)                               # white -> black
        return (255, 255, 255)

    def _hp_scaled(self, name: str, im, k: float):
        """`im` scaled by k, cached — the scorebar art is static, so the
        (possibly full-screen) LANCZOS resize happens once per render, not per
        frame."""
        cache = getattr(self, "_hp_scaled_cache", None)
        if cache is None:
            cache = self._hp_scaled_cache = {}
        key = (name, round(k, 5))
        out = cache.get(key)
        if out is None:
            out = im.resize((max(1, int(round(im.width * k))),
                             max(1, int(round(im.height * k)))), Image.LANCZOS)
            cache[key] = out
        return out

    def _hp_ki(self) -> dict:
        """Marker art, loaded ONCE (a per-frame _resolve miss re-scans the whole
        skin dir case-insensitively — ~55ms/frame on the NAS mount)."""
        cache = getattr(self, "_hp_ki_cache", None)
        if cache is None:
            cache = self._hp_ki_cache = {
                n: self._load_native(f"scorebar-{n}", default_ok=False)
                for n in ("marker", "ki", "kidanger", "kidanger2")}
        return cache

    def _hp_marker(self, hp: float):
        """The skin's own HP-marker art: scorebar-marker (new style), else
        scorebar-ki / -kidanger (<50%) / -kidanger2 (<20%) (old style, lazer's
        LegacyOldStyleMarker cutoffs). None when the skin ships no marker art
        at all. USER SKIN ONLY, like the rest of the scorebar."""
        cache = self._hp_ki()
        if cache["marker"] is not None:
            return cache["marker"]
        if hp < 0.2:
            return cache["kidanger2"] or cache["kidanger"] or cache["ki"]
        if hp < 0.5:
            return cache["kidanger"] or cache["ki"]
        return cache["ki"]

    def _draw_legacy_health(self, img, hp: float, t: float = 0.0) -> None:
        """LegacyHealthDisplay.

        SKIN SPRITES AT NATIVE LOGICAL SIZE: lazer's legacy HUD lives in a
        DrawSizePreservingFillContainer targeting 1024x768, so one logical
        texture px (@2x art halved) == screen_h/768 px, and scorebar-bg sits
        UNSCALED at the component's top-left. Skins rely on that to paint
        full-canvas HUD frames — e.g. a 1378x786 scorebar-bg whose art covers
        the whole 768p screen (corner shards, right-edge key-overlay arrows).
        The old behaviour stretched the bg to a 695-display-unit "bar", which
        crushed such full-canvas art into the top-left quadrant and smeared
        its decorations over the playfield.

        Fill: scorebar-colour(-0) at (3,10)*1.6 old style / (7.5,7.8)*1.6 new
        style (new style == the skin ships scorebar-marker; lazer's
        LegacyHealthDisplay offsets), CROPPED from the right by hp, never
        scaled. Marker centred on the fill's end — old style on the fill's top
        edge, new style on its centreline — using the skin's own marker art
        via _hp_marker; a 1x1 blank ki (how skins hide the marker) therefore
        shows nothing, and only a skin with a scorebar but no ki/marker art at
        all falls back to the drawn glowing ball.

        A skin with NO scorebar keeps the classic-default drawn look
        (geometry in display units: bg 695x44; fill 645x10 at (12.0, 12.48);
        glowing-ball marker centred at (12 + fillWidth, 17.48)).
        """
        hp = max(0.0, min(1.0, hp))
        k = self.h / 768.0
        c = self._comp("LegacyHealthDisplay")
        if c is not None:
            k *= c.scale[1]
        col = self._legacy_fill_colour(hp)
        marker_sprite = None

        if self.scorebar_bg is not None:
            # the skin's own scorebar, legacy-native geometry
            bg = self._hp_scaled("bg", self.scorebar_bg, k)
            x0, y0 = self._place("LegacyHealthDisplay", bg.width, bg.height,
                                 (0, 0))
            self._paste(img, bg, int(x0), int(y0))
            new_style = self._hp_ki()["marker"] is not None
            fpx, fpy = (7.5, 7.8) if new_style else (3.0, 10.0)
            fx, fy = x0 + fpx * 1.6 * k, y0 + fpy * 1.6 * k
            fw, fh = 0.0, 0.0
            if self.scorebar_col is not None:
                cimg = self._hp_scaled("colour", self.scorebar_col, k)
                cw = int(cimg.width * hp)
                if cw > 0:
                    self._paste(img, cimg.crop((0, 0, cw, cimg.height)),
                                int(fx), int(fy))
                fw, fh = cimg.width * hp, float(cimg.height)
            mx = fx + fw
            my = fy + (fh / 2.0 if new_style else 0.0)
            marker_sprite = self._hp_marker(hp)
        else:
            # classic-default look, drawn to the display-unit geometry
            bg_w, bg_h = 695.0 * k, 44.0 * k
            x0, y0 = self._place("LegacyHealthDisplay", bg_w, bg_h, (0, 0))
            fx, fy = x0 + 12.0 * k, y0 + 12.48 * k
            fw_max, fh = 645.0 * k, 10.0 * k
            fw = fw_max * hp
            d = ImageDraw.Draw(img)
            r = bg_h / 2.0
            d.rounded_rectangle([x0, y0 + bg_h * 0.18, x0 + bg_w, y0 + bg_h * 0.82],
                                radius=r * 0.55, fill=(26, 28, 36))
            if fw > 1:
                fr = fh / 2.0
                d.rounded_rectangle([fx, fy, fx + fw, fy + fh], radius=fr, fill=col)
            mx, my = fx + fw, y0 + 17.48 * k
        # marker BULGE: on any HP GAIN the marker snaps to 1.2x then eases to
        # 0.8x over 150ms and RESTS at 0.8 (lazer's actual behaviour, kept).
        prev = getattr(self, "_hp_prev", None)
        if prev is not None and hp > prev + 0.0005:
            self._hp_bulge_t = t
        self._hp_prev = hp
        bt = getattr(self, "_hp_bulge_t", None)
        if bt is None:
            mscale = 1.0
        else:
            age = t - bt
            mscale = (1.2 if age <= 0 else
                      (1.2 + (0.8 - 1.2) * (age / 150.0) if age < 150.0 else 0.8))
        if self.scorebar_bg is not None:
            if marker_sprite is not None:
                # the skin's own marker art (possibly a deliberate 1x1 blank)
                mw = max(1, int(round(marker_sprite.width * k * mscale)))
                mh = max(1, int(round(marker_sprite.height * k * mscale)))
                if (mw, mh) == marker_sprite.size:
                    m = marker_sprite
                else:
                    # PERF: mscale RESTS at 0.8 after the first HP gain, so
                    # this resize ran every frame forever. Cache per target
                    # size (the bulge sweeps only a handful of sizes; art
                    # images live for the whole render, so id() keys are
                    # stable). Same resize inputs -> same bytes.
                    mcache = getattr(self, "_hp_marker_cache", None)
                    if mcache is None:
                        mcache = self._hp_marker_cache = {}
                    mkey = (id(marker_sprite), mw, mh)
                    m = mcache.get(mkey)
                    if m is None:
                        m = marker_sprite.resize((mw, mh), Image.LANCZOS)
                        mcache[mkey] = m
                self._paste(img, m, int(mx - mw / 2.0), int(my - mh / 2.0))
                return
            # skin scorebar without any marker art -> the drawn ball below
        # marker: glowing ball centred on the fill's right edge.
        # PERF: the ball only ever touches its own ~(2*1.9*mr)px box —
        # round-trip just that crop through RGBA instead of the whole frame
        # (a full-frame convert+composite+convert EVERY frame was the single
        # biggest per-frame HUD cost on skinned renders). alpha_composite is
        # per-pixel, and every pixel outside the box was untouched by the old
        # full-frame composite, so the output bytes are identical.
        mr = 12.0 * k * mscale
        ext = mr * 1.9 + 2.0
        bx0, by0 = max(0, int(mx - ext)), max(0, int(my - ext))
        bx1 = min(img.width, int(mx + ext) + 2)
        by1 = min(img.height, int(my + ext) + 2)
        if bx1 <= bx0 or by1 <= by0:
            return
        sub = img.crop((bx0, by0, bx1, by1)).convert("RGBA")
        layer = Image.new("RGBA", sub.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for rr, a in ((mr * 1.9, 45), (mr * 1.35, 90)):
            ld.ellipse([mx - rr - bx0, my - rr - by0,
                        mx + rr - bx0, my + rr - by0], fill=(*col, a))
        ld.ellipse([mx - mr * 0.62 - bx0, my - mr * 0.62 - by0,
                    mx + mr * 0.62 - bx0, my + mr * 0.62 - by0],
                   fill=(*col, 255))
        img.paste(Image.alpha_composite(sub, layer).convert("RGB"), (bx0, by0))

    def _draw_song_progress_pie(self, img, t_ms: int, acc_box,
                                xy: tuple[int, int] | None = None) -> None:
        """LegacySongProgress — a 33x33 CIRCULAR PIE (not a bar, and with NO time
        text; SongProgressInfo is Default/Argon only). 2px white ring, pie at 0.92
        of the box, 4px centre dot. Gameplay = white @60% counting up; before the
        first object it mirrors and counts DOWN in yellow-green.

        `xy`: precomputed default top-left (the skinned single-player layout's
        fixed row-2 anchor). None (overlay-board path) keeps the legacy
        derivation from acc_box. A skin's own lazer layout still wins."""
        span = max(1, self.last_ms - self.first_ms)
        frac = max(0.0, min(1.0, (t_ms - self.first_ms) / span))
        k = self.h / 768.0
        c = self._comp("LegacySongProgress")
        if c is not None:
            k *= c.scale[1]
        box = max(8, int(33 * k))
        if xy is not None:
            dx, dy = int(xy[0]), int(xy[1])
        else:
            # default: vertically centred on the accuracy counter, 18 units
            # left of it
            ax, ay, aw, ah = acc_box
            dx = int(ax - 18 * (self.h / 768.0) - box)
            dy = int(ay + ah / 2 - box / 2)
        x, y = self._place("LegacySongProgress", box, box, (dx, dy))
        d = ImageDraw.Draw(img)
        intro = t_ms < self.first_ms
        col = (199, 255, 47) if intro else (255, 255, 255)
        prog = (1.0 - frac) if intro else frac
        # pie (0.92 of the box), drawn from 12 o'clock
        pad = int(box * 0.04)
        pb = [x + pad, y + pad, x + box - pad, y + box - pad]
        if prog > 0.001:
            end = -90.0 + 360.0 * max(0.0, min(1.0, prog))
            # ~60% alpha over the frame. PERF: the pie only touches its own
            # box — round-trip just that crop through RGBA instead of the
            # whole 1080p frame (two full-frame converts + a full-frame
            # alpha_composite, every frame). Per-pixel identical: pixels
            # outside the box were untouched by the full-frame composite.
            bx0, by0 = max(0, int(x) - 2), max(0, int(y) - 2)
            bx1 = min(img.width, int(x) + box + 3)
            by1 = min(img.height, int(y) + box + 3)
            if bx1 > bx0 and by1 > by0:
                sub = img.crop((bx0, by0, bx1, by1)).convert("RGBA")
                layer = Image.new("RGBA", sub.size, (0, 0, 0, 0))
                ImageDraw.Draw(layer).pieslice(
                    [pb[0] - bx0, pb[1] - by0, pb[2] - bx0, pb[3] - by0],
                    -90.0, end, fill=(*col, 153))
                img.paste(Image.alpha_composite(sub, layer).convert("RGB"),
                          (bx0, by0))
            d = ImageDraw.Draw(img)
        # 2px white ring + 4px centre dot
        d.ellipse([x, y, x + box, y + box], outline=(255, 255, 255),
                  width=max(1, int(2 * k)))
        r = max(1, int(2 * k))
        cx, cy = x + box / 2, y + box / 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))

    def _draw_argon_song_progress(self, img, t_ms: int) -> None:
        """ArgonSongProgress as placed by the skin (Red's is Rotation -90 -> a
        VERTICAL bar down the left edge, show_graph/show_time false). Only drawn
        when the skin's layout actually asks for it."""
        c = self._comp("ArgonSongProgress")
        if c is None:
            return
        span = max(1, self.last_ms - self.first_ms)
        frac = max(0.0, min(1.0, (t_ms - self.first_ms) / span))
        k = self.h / 768.0
        vertical = abs(abs(c.rotation) - 90.0) < 1.0
        # relative width -> fraction of the HUD rect's long axis
        base = (self.w if not vertical else self.w) if c.width else 400.0
        length = max(8, int(abs(float(c.width or 1.0)) * base * c.scale[0]))
        thick = max(3, int(10 * k * c.scale[1]))
        acc = c.settings.get("accent_colour") or "#FFFFFF"
        try:
            col = tuple(int(acc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            col = (255, 255, 255)
        d = ImageDraw.Draw(img)
        if vertical:
            x, y = self._place("ArgonSongProgress", thick, length,
                               (int(self.w * 0.013), int(self.h * 0.08)))
            y = max(0, min(y, self.h - length))
            d.rectangle([x, y, x + thick, y + length], fill=(26, 26, 34))
            fh = int(length * frac)
            if fh > 0:
                d.rectangle([x, y, x + thick, y + fh], fill=col)
        else:
            x, y = self._place("ArgonSongProgress", length, thick,
                               (0, self.h - thick))
            d.rectangle([x, y, x + length, y + thick], fill=(26, 26, 34))
            fw = int(length * frac)
            if fw > 0:
                d.rectangle([x, y, x + fw, y + thick], fill=col)

    def _draw_catch_combo(self, img, scene, t: float) -> None:
        """LegacyCatchComboCounter — centred above the catcher, tracking its X.

        lazer: counter centre sits 175 units above the top of the CatcherArea
        (512 x 106.75) at Catcher.X. On increment it fades in, holds ~1000 ms and
        fades out over 300 ms; the glyph pops 1.5 -> 0.8 (250 ms OutQuad) then
        1.0 -> 1.1 (60 ms) -> 1.0 (30 ms). Combo 0 fades out over 400 ms.
        """
        # ALPHA is SOLID while the combo is alive — fade IN once when the
        # combo starts (0 -> positive) and OUT only when it breaks
        # (positive -> 0). catch increments the combo on EVERY caught fruit
        # (many per second), so the previous "reset the fade-in on each
        # increment + fade out after a 1000ms hold" made the counter flash /
        # blink between catches (community report 2026-07-24). The size POP
        # still fires per increment (lazer LegacyCatchComboCounter), decoupled
        # from alpha via its own timer.
        combo = max(int(getattr(scene, "combo", 0)), 0)
        prev = getattr(self, "_combo_prev", None)
        if prev != combo:
            if combo > (prev or 0):
                self._combo_pop_t = t                    # per-increment pop
                if not prev:
                    self._combo_appear_t = t             # start-of-combo fade-in
                self._combo_last_pos = combo
                self._combo_break_t = None
            elif combo == 0 and prev:
                self._combo_break_t = t                  # begin fade-out
            self._combo_prev = combo

        appear_t = getattr(self, "_combo_appear_t", None)
        break_t = getattr(self, "_combo_break_t", None)
        last_pos = getattr(self, "_combo_last_pos", 0)

        if combo > 0:
            draw_combo = combo
            a = (t - appear_t) if appear_t is not None else 1e9
            if a < 0:
                return
            alpha = min(1.0, a / 60.0)                    # 60ms fade-in, then solid
        elif break_t is not None and last_pos > 0:
            draw_combo = last_pos                         # fade the last number out
            fo = t - break_t
            if fo < 0 or fo >= 400.0:
                return
            alpha = 1.0 - fo / 400.0                      # lazer: combo 0 -> 400ms
        else:
            return

        # pop: 1.5 -> 0.8 over 250ms (OutQuad), then 1.0 -> 1.1 -> 1.0, driven by
        # the last increment (settles to 1.0; during fade-out it's already 1.0).
        pop_t = getattr(self, "_combo_pop_t", None)
        page = (t - pop_t) if pop_t is not None else 1e9
        if page < 0:
            page = 1e9
        if page <= 250.0:
            u = page / 250.0
            scale = 1.5 + (0.8 - 1.5) * (1.0 - (1.0 - u) ** 2)
        elif page <= 310.0:
            scale = 1.0 + 0.1 * ((page - 250.0) / 60.0)
        elif page <= 340.0:
            scale = 1.1 - 0.1 * ((page - 310.0) / 30.0)
        else:
            scale = 1.0
        cg = self.combo_glyphs or self.score_glyphs
        if not cg:
            return
        base_h = self.h * 0.075 * 0.8      # LegacyCatchComboCounter Scale = 0.8
        th = max(1, int(base_h * scale))
        # PERF: single-slot memo of the assembled+resized digits. After the
        # 340ms pop settles, scale (and so th) is constant until the combo
        # changes — the same image was rebuilt every frame. The fade below
        # works on a copy so the cached image stays pristine. Same inputs ->
        # same bytes as the per-frame rebuild.
        _ck = (draw_combo, th)
        _cm = getattr(self, "_combo_num_memo", None)
        if _cm is not None and _cm[0] == _ck:
            txt = _cm[1]
            if txt is None:
                return
            tw = txt.width
        else:
            txt = self._number(str(draw_combo), cg, self._combo_overlap())
            if txt.width < 1 or txt.height < 1:
                self._combo_num_memo = (_ck, None)
                return
            tw = max(1, int(txt.width * th / txt.height))
            txt = txt.resize((tw, th), Image.LANCZOS)
            self._combo_num_memo = (_ck, txt)
        if alpha < 1.0:
            txt = txt.copy()
            txt.putalpha(txt.getchannel("A").point(lambda v: int(v * alpha)))
        # catcher position -> screen: track the catcher like stable/lazer's
        # LegacyCatchComboCounter. SceneState carries the geometry
        # (catcher_px / plane_y_px / pf_unit_px).
        # PLACEMENT — verified against lazer source 2026-07-22
        # (osu.Game.Rulesets.Catch/UI/CatcherArea.cs): comboDisplay has
        # Anchor=TopLeft (of the CatcherArea, whose top edge IS the catch
        # plane: CatchPlayfield anchors the area BottomLeft/TopLeft),
        # Origin=Centre, Margin{Bottom=350}. osu!framework folds margins into
        # OriginPosition (LayoutSize/2 - (margin.Left, margin.Top)), so a
        # bottom margin of 350 lifts the layout CENTRE 175 units: the counter
        # is centred exactly 175 playfield units ABOVE the catch plane. X:
        # CatcherArea.UpdateAfterChildren sets comboDisplay.X = Catcher.X
        # every frame, unclamped — it FOLLOWS the catcher (we keep a
        # screen-edge clamp below purely so digits never crop at the walls).
        # An earlier eyeballed 120 sat the counter visibly LOWER than
        # stable/lazer (player report 2026-07-22: real client shows it "a bit
        # above the middle") — 175, the source value, restores parity.
        cx = getattr(scene, "catcher_px", None)
        py = getattr(scene, "plane_y_px", None)
        up = getattr(scene, "pf_unit_px", None)
        if cx is not None and py is not None and up:
            cx = float(cx)
            cy = float(py) - 175.0 * float(up)
            # keep the number fully on screen at the playfield edges
            cx = max(tw / 2.0 + 2.0, min(self.w - tw / 2.0 - 2.0, cx))
            cy = max(th / 2.0 + 2.0, min(cy, self.h - th / 2.0 - 2.0))
        else:                  # legacy snapshot without geometry
            cx, cy = self.w / 2.0, self.h * 0.45
        self._paste(img, txt, int(cx - tw / 2), int(cy - th / 2))

    def _draw_key_overlay(self, img, t, held, counts=None) -> None:
        """LEGACY key overlay. Catch's three inputs are LEFT / RIGHT / DASH — the
        borrowed Argon counter labelled them B1/B2/B3, which is std/mania naming
        and simply wrong here. Press counts are edge-counted from the held states
        (the replay stores catcher positions + a dash bit, not key events).

        `inputoverlay-background` is deliberately not drawn, and the key art is
        taken from the USER's skin only: the default skin's key sprite is a
        near-black button that's legible only over that background bar, and the
        bar's art sits in a faintly-alpha'd canvas that can't be trimmed to its
        opaque bounds — rotated for a vertical stack it renders as a smear. A
        drawn key box is used instead, readable over any gameplay.

        COUNT FONT: when number glyphs are available (skin's combo/score font,
        like _draw_catch_combo uses), the press counts are rendered from THOSE
        glyphs so the key counter matches the score/combo font — osu!stable
        draws the skin's number font on the input overlay. Skin fonts are
        usually LIGHT art, so over the drawn box the box flips to a dark
        neutral (a white glyph on the white lazer box would be invisible);
        a skin whose font is dark keeps the white box. A skin that ships its
        own inputoverlay-key gets the glyphs straight over its sprite, exactly
        like stable. No glyphs at all (Argon/skinless) -> the unchanged white
        box + dark lazer glyph.

        PRESS ANIMATION (ppy/osu master, osu.Game/Skinning/LegacyKeyCounter.cs):
        on press the keyContainer — the key sprite AND its text — scales about
        its own centre to 0.75 over 160 ms Easing.Out (OutQuad), and back to
        1.0 the same way on release; an edge mid-tween continues FROM the
        current scale, exactly like an osu!framework transform replacing a
        running one (tap-spam springs partway, never snaps). The sprite tint
        flips INSTANTLY to the flow-index ActiveColour while held and back to
        white on release (`keySprite.Colour = ActiveColour` has no transition
        in source); the TEXT colour itself never changes on press — it is the
        constant InputOverlayText colour — so the drawn-box paths keep their
        existing held colour flip as the box-language stand-in for that sprite
        tint. The first press additionally crossfades name -> count over the
        same 160 ms Easing.Out (initialNameText.FadeOut / overlayKeyText.
        FadeIn). `t` is MAP-time ms: lazer's HUD animates on the gameplay
        clock, so DT/HT rate-scaling of the tween is inherent."""
        labels = ("B1", "B2", "B3")
        # Row mapping (unchanged, catch's three real inputs): B1 = move left,
        # B2 = move right, B3 = dash. `counts` = the sim's replay-resolution
        # press onsets (authoritative — tap-spam races up like stable's
        # overlay); None -> legacy per-video-frame edge counting fallback.
        if counts is not None:
            self._kc_counts = [int(counts[0]), int(counts[1]), int(counts[2])]
        else:
            for i in range(3):
                if held[i] and not self._kc_held_prev[i]:
                    self._kc_counts[i] += 1
        t = float(t)
        for i in range(3):
            h, hp = bool(held[i]), self._kc_held_prev[i]
            if h != hp:
                # edge -> new ScaleTo from the interrupted tween's current value
                self._kc_anim[i] = (self._kc_scale(i, t),
                                    0.75 if h else 1.0, t)
                if h and self._kc_first_press_t[i] is None:
                    self._kc_first_press_t[i] = t
        self._kc_held_prev = (bool(held[0]), bool(held[1]), bool(held[2]))

        W, H = self.w, self.h
        d = ImageDraw.Draw(img)
        # osu!stable stacks the key overlay VERTICALLY on the right edge.
        ks = int(H * 0.058)
        gap = int(ks * 0.14)
        cf = _font(int(ks * 0.42))
        lf = _font(int(ks * 0.34))
        # skin number font for the counts (combo first, like the catch combo).
        glyphs = self.combo_glyphs or self.score_glyphs
        dark_box = False
        if glyphs:
            # alpha-weighted mean luminance of the digit art (cached): light
            # glyphs need the dark box, dark glyphs stay on the white one.
            lum = getattr(self, "_kc_glyph_lum", None)
            if lum is None:
                asum = lsum = 0.0
                amax = 0.0
                for g in glyphs.values():
                    a = np.asarray(g, dtype=np.float32)
                    if a.ndim != 3 or a.shape[2] < 4:
                        continue
                    al = a[..., 3] / 255.0
                    ll = (0.299 * a[..., 0] + 0.587 * a[..., 1]
                          + 0.114 * a[..., 2]) / 255.0
                    asum += float(al.sum())
                    lsum += float((ll * al).sum())
                    amax = max(amax, float(a[..., 3].max()))
                lum = lsum / asum if asum > 0.0 else 1.0
                self._kc_glyph_lum = lum
                # peak alpha of the font: some skins ship a deliberately
                # ghost-transparent number font (seen at 10% max alpha). The
                # score/combo honour that, but a near-invisible digit inside
                # the small key box reads as a bug -> the counter normalises
                # such a font to readable opacity (shape/colour untouched).
                self._kc_glyph_amax = amax
            dark_box = self.key_img is None and lum >= 0.5
        stack_w, stack_h = ks, ks * 3 + gap * 2
        kk = H / 768.0
        # catch: Anchor CentreRight, Origin TopRight, Position (0,-40)*1.6=(0,-64)
        dflt = (W - stack_w, int(H * 0.5 - 64 * kk))
        x0, y0 = self._place("LegacyKeyCounterDisplay", stack_w, stack_h, dflt)
        for i, lab in enumerate(labels):
            ky = y0 + i * (ks + gap)
            # LegacyKeyCounterDisplay active colours are per FLOW INDEX:
            # index<2 -> #ffde00 (yellow), index>=2 -> #f8009e (magenta).
            # catch has 3 keys, so the DASH key (B3) flashes magenta.
            act = (0xFF, 0xDE, 0x00) if i < 2 else (0xF8, 0x00, 0x9E)
            # keyContainer scale at map-time t (see the docstring). `anim`
            # gates every scaled/faded draw below: a key at rest takes the
            # EXACT pre-animation code paths, so no-input stretches stay
            # byte-identical frame to frame.
            s = self._kc_scale(i, t)
            anim = bool(held[i]) or s != 1.0
            if self.key_img is not None:
                # Draw at NATIVE LOGICAL size * (H/768), centred on the cell —
                # what stable/lazer do (see the key_img load comment). CACHED
                # by (w, h, ActiveColour): the 160 ms tween quantises to a
                # couple dozen pixel sizes per tint at most. Oversized art is
                # clamped so a rogue sprite can never cover the playfield.
                kw = self.key_img.width * kk
                kh = self.key_img.height * kk
                cap = ks * 2.0
                if max(kw, kh) > cap:
                    f = cap / max(kw, kh)
                    kw, kh = kw * f, kh * f
                tinted = bool(held[i])   # keySprite.Colour: instant in lazer
                # The cache key MUST carry the ActiveColour, not just a bool:
                # the movement keys (i<2) tint #ffde00 and the dash key (i>=2)
                # #f8009e, but at the SAME cell size. Keying on `tinted` alone
                # let the first tinted key drawn at a given size (always a
                # yellow movement key, and the cache persists across frames)
                # satisfy the dash key's lookup too — so the dash key rendered
                # yellow instead of magenta (the reported bug). Include `act`
                # so each flow colour gets its own entry.
                ck = (max(1, int(kw * s)), max(1, int(kh * s)),
                      act if tinted else None)
                cache = getattr(self, "_kc_key_cache", None)
                if cache is None:
                    cache = self._kc_key_cache = {}
                key = cache.get(ck)
                if key is None:
                    key = self.key_img.resize(ck[:2], Image.LANCZOS)
                    if tinted:           # multiply by ActiveColour
                        r, g, b, a = key.split()
                        key = Image.merge("RGBA", (
                            r.point(lambda v: v * act[0] // 255),
                            g.point(lambda v: v * act[1] // 255),
                            b.point(lambda v: v * act[2] // 255), a))
                    cache[ck] = key
                self._paste(img, key, x0 + (ks - key.width) // 2,
                            ky + (ks - key.height) // 2)
            elif dark_box:
                # dark neutral box so the skin's LIGHT glyphs read; held state
                # keeps the accent as a full-strength outline + a muted fill
                # (full accent fill would drown a white glyph).
                fill = (tuple(int(0.35 * c + 0.65 * n) for c, n in
                              zip(act, (34, 34, 42))) if held[i]
                        else (34, 34, 42))
                outline = act if held[i] else (92, 92, 104)
                self._kc_box(d, x0, ky, ks, s, fill, outline)
            else:
                # lazer's default key counter: WHITE rounded box, dark glyph,
                # YELLOW while held (verified in Red's captures).
                fill = act if held[i] else (248, 248, 252)
                self._kc_box(d, x0, ky, ks, s, fill, (228, 228, 236))
            # lazer shows the COUNT once the key has been pressed, and the KEY
            # NAME (B1/B2/B3) while the count is still 0 — image #43 shows
            # exactly "5", "4", "B3". `fade` is the one-time name->count
            # crossfade (160 ms Easing.Out from the FIRST activation); it
            # always completes before the release tween settles, so the rest
            # path below never sees a partial fade.
            n = self._kc_counts[i]
            fp = self._kc_first_press_t[i]
            fade = ((1.0 if fp is None else self._kc_ease_out((t - fp) / 160.0))
                    if n > 0 else 0.0)
            if not anim:
                # at rest — the EXACT pre-animation draw (byte-stable at idle)
                if n > 0 and glyphs:
                    num = self._kc_num(n, ks, glyphs)
                    if num.width > 0 and num.height > 0:
                        self._paste(img, num, x0 + (ks - num.width) // 2,
                                    ky + (ks - num.height) // 2)
                    continue
                txt = str(n) if n > 0 else lab
                f = cf if n > 0 else lf
                bb = d.textbbox((0, 0), txt, font=f)
                tx = x0 + (ks - (bb[2] - bb[0])) // 2 - bb[0]
                ty = ky + (ks - (bb[3] - bb[1])) // 2 - bb[1]
                d.text((tx, ty), txt, font=f,
                       fill=(232, 232, 238) if dark_box else (28, 28, 34))
                continue
            # animating: text scales WITH the container (name and count both
            # live inside lazer's keyContainer) and honours the crossfade.
            cx, cy = x0 + ks / 2.0, ky + ks / 2.0
            col = (232, 232, 238) if dark_box else (28, 28, 34)
            if fade < 1.0:               # key name (fading out after press 1)
                self._kc_text(img, lab, _font(max(6, int(ks * 0.34 * s))),
                              col, 1.0 - fade, cx, cy)
            if n > 0 and fade > 0.0:     # the count (fading in on press 1)
                if glyphs:
                    num = self._kc_num(n, ks, glyphs)
                    if num.width > 0 and num.height > 0:
                        num = num.resize(
                            (max(1, int(round(num.width * s))),
                             max(1, int(round(num.height * s)))),
                            Image.LANCZOS)
                        if fade < 1.0:
                            num.putalpha(num.getchannel("A").point(
                                lambda v: int(v * fade)))
                        self._paste(img, num, int(cx) - num.width // 2,
                                    int(cy) - num.height // 2)
                else:
                    self._kc_text(img, str(n),
                                  _font(max(6, int(ks * 0.42 * s))),
                                  col, fade, cx, cy)

    @staticmethod
    def _kc_ease_out(u: float) -> float:
        """osu!framework Easing.Out (= OutQuad), clamped to [0, 1]."""
        u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        return u * (2.0 - u)

    def _kc_scale(self, i: int, t: float) -> float:
        """lazer LegacyKeyCounter keyContainer scale at map-time t:
        ScaleTo(0.75 press / 1.0 release, 160 ms, Easing.Out), each edge
        retweening FROM the value it interrupted (set in _draw_key_overlay).
        Settled tweens return their target EXACTLY (1.0 at rest), which the
        byte-stability of idle frames relies on."""
        frm, tgt, t0 = self._kc_anim[i]
        u = (t - t0) / 160.0
        if u >= 1.0:
            return tgt
        return frm + (tgt - frm) * self._kc_ease_out(u)

    @staticmethod
    def _kc_box(d, x0, ky, ks, s, fill, outline) -> None:
        """The drawn key box at animation scale `s`, centred on its cell — the
        stand-in for lazer's keyContainer scale on the box paths. `s == 1.0`
        is the EXACT pre-animation call (byte-stable at idle)."""
        if s == 1.0:
            d.rounded_rectangle([x0, ky, x0 + ks, ky + ks], radius=ks // 5,
                                fill=fill, outline=outline,
                                width=max(1, ks // 22))
            return
        half = ks * s / 2.0
        cx, cy = x0 + ks / 2.0, ky + ks / 2.0
        bs = max(1, int(round(ks * s)))
        d.rounded_rectangle([cx - half, cy - half, cx + half, cy + half],
                            radius=bs // 5, fill=fill, outline=outline,
                            width=max(1, bs // 22))

    def _kc_num(self, n: int, ks: int, glyphs) -> Image.Image:
        """The composed count digits for the key overlay — skin number font,
        same glyph assembly as the combo/score. CACHED by (count, box size):
        the composed digits are pixel-identical every frame until the count
        changes, and the per-frame assemble + LANCZOS (x3 keys, every frame)
        was measurably dragging full skinned renders. PERF."""
        _cache = getattr(self, "_kc_num_cache", None)
        if _cache is None:
            _cache = self._kc_num_cache = {}
        num = _cache.get((n, ks))
        if num is None:
            num = self._number(str(n), glyphs, self._combo_overlap())
            if num.width > 0 and num.height > 0:
                th = max(1, int(ks * 0.46))
                tw = max(1, int(num.width * th / num.height))
                wmax = max(1, int(ks * 0.84))
                if tw > wmax:        # long counts: fit the box width
                    th = max(1, int(th * wmax / tw))
                    tw = wmax
                num = num.resize((tw, th), Image.LANCZOS)
                amax = getattr(self, "_kc_glyph_amax", 255.0)
                if 0.0 < amax < 200.0:  # ghost font -> readable digits
                    k = 230.0 / amax
                    num.putalpha(num.getchannel("A").point(
                        lambda v: min(255, int(v * k))))
            _cache[(n, ks)] = num
        return num

    def _kc_text(self, img, txt, f, col, alpha, cx, cy) -> None:
        """Animated key-overlay text: rendered on a transparent tile so it can
        fade (the frame is RGB — ImageDraw cannot alpha-blend onto it) and
        pasted centred on the key cell."""
        d0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bb = d0.textbbox((0, 0), txt, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        if tw <= 0 or th <= 0:
            return
        a = max(0, min(255, int(round(alpha * 255.0))))
        if a == 0:
            return
        tile = Image.new("RGBA", (tw + 2, th + 2), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((1 - bb[0], 1 - bb[1]), txt, font=f,
                                  fill=col + (a,))
        self._paste(img, tile, int(cx) - tile.width // 2,
                    int(cy) - tile.height // 2)

    @staticmethod
    def _mmss(ms: int) -> str:
        ms = max(0, int(ms))
        return f"{ms // 60000:01d}:{(ms // 1000) % 60:02d}"

    def _draw_progress_legacy(self, img, t_ms: int):
        """Song progress: a VERTICAL bar down the LEFT edge (Red's lazer layout),
        filling from the TOP downward, plus the progress COUNTER (elapsed/total).
        The skin HUD previously drew only a bare horizontal strip and no counter."""
        span = max(1, self.last_ms - self.first_ms)
        frac = max(0.0, min(1.0, (t_ms - self.first_ms) / span))
        W, H = self.w, self.h
        d = ImageDraw.Draw(img)
        bw = max(4, int(W * 0.0045))
        x = int(W * 0.013)
        y0, y1 = int(H * 0.085), int(H * 0.915)
        r = bw // 2
        d.rounded_rectangle([x, y0, x + bw, y1], radius=r, fill=(26, 26, 34))
        fh = int((y1 - y0) * frac)
        if fh > 0:
            d.rounded_rectangle([x, y0, x + bw, y0 + fh], radius=r,
                                fill=(244, 244, 250))
        # counter, ABOVE the bar — below it would collide with the combo counter
        # that sits bottom-left.
        pf = _font(int(H * 0.021))
        txt = f"{self._mmss(t_ms - self.first_ms)} / {self._mmss(span)}"
        bb = d.textbbox((0, 0), txt, font=pf)
        tx = x - bb[0]
        ty = y0 - (bb[3] - bb[1]) - int(H * 0.014) - bb[1]
        d.text((tx + 1, ty + 1), txt, font=pf, fill=(0, 0, 0))
        d.text((tx, ty), txt, font=pf, fill=(236, 236, 244))

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

    def _resolve(self, base: str, *, default_ok: bool = True) -> Path | None:
        """Locate a skin asset: user skin first, then the DEFAULT skin (osu's
        own fallback). `default_ok=False` restricts the search to the user's
        skin — used for the score-font probe that decides skin-HUD vs Argon,
        which must NOT be satisfied by the default skin (a skinless render has
        to stay 100% Argon)."""
        roots = [self.dir]
        if default_ok and self.default_dir is not None:
            roots.append(self.default_dir)
        for root in roots:
            if root is None:
                continue
            for stem in (f"{base}@2x", base):
                p = root / f"{stem}.png"
                if p.is_file():
                    return p
                # Case-insensitive fallback: osu skins are case-blind (Windows),
                # but our filesystem isn't — e.g. skin.ini ComboPrefix "combo" vs
                # files named "Combo-0.png". Scan the parent dir case-blind.
                parent, want = p.parent, p.name.lower()
                if parent.is_dir():
                    for f in parent.iterdir():
                        if f.name.lower() == want:
                            return f
        return None

    def _load_native(self, base: str, *, default_ok: bool = True):
        """Load an asset at its LOGICAL size (an `@2x` asset is double-resolution
        art, so its logical size is half the pixel size)."""
        p = self._resolve(base, default_ok=default_ok)
        if p is None:
            return None
        im = Image.open(p).convert("RGBA")
        if "@2x" in p.name:
            im = im.resize((max(1, im.width // 2), max(1, im.height // 2)),
                           Image.LANCZOS)
        return im

    def _load(self, base: str, target_h: int,
              *, default_ok: bool = True) -> Image.Image | None:
        p = self._resolve(base, default_ok=default_ok)
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

    def _ini_font_int(self, key: str) -> int:
        """An int from skin.ini [Fonts] (ScoreOverlap / ComboOverlap); 0 if absent."""
        cache = getattr(self, "_ini_font_cache", None)
        if cache is None:
            cache = self._ini_font_cache = {}
        if key in cache:
            return cache[key]
        out = 0
        try:
            ini = (self.dir / "skin.ini") if self.dir else None
            if ini and ini.is_file():
                in_fonts = False
                for raw in ini.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw.strip()
                    if line.startswith("["):
                        in_fonts = line.lower() == "[fonts]"
                        continue
                    if in_fonts and ":" in line:
                        k, _, val = line.partition(":")
                        if k.strip().lower() == key:
                            out = int(float(val.strip()))
        except (OSError, ValueError):
            out = 0
        cache[key] = out
        return out

    @staticmethod
    def _ease_out_quad(p: float) -> float:
        p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
        return 1.0 - (1.0 - p) * (1.0 - p)

    def _roll_at(self, frm, to, t0, t, dur):
        p = 1.0 if dur <= 0 else (t - t0) / dur
        return frm + (to - frm) * self._ease_out_quad(p)

    def _rolled(self, key: str, target: float, t: float, dur: float) -> float:
        """Displayed value of a rolling counter. Lazer: fixed-duration Easing.Out
        (score 1000ms, accuracy 375ms) — when the target changes the roll restarts
        from the CURRENTLY displayed value, not from 0."""
        st = self._roll.get(key)
        if st is None:
            self._roll[key] = (target, target, t - dur)   # start settled
            return target
        frm, to, t0 = st
        if to != target:
            frm = self._roll_at(frm, to, t0, t, dur)       # continue from here
            to, t0 = target, t
            self._roll[key] = (frm, to, t0)
        return self._roll_at(frm, to, t0, t, dur)

    @staticmethod
    def _acc_text(acc: float) -> str:
        """PercentageCounter FLOORS to 2dp (89.99999% -> 89.99%, never rounds up)."""
        import math as _m
        a = max(0.0, min(1.0, acc)) * 100.0
        return f"{_m.floor(a * 100) / 100:.2f}%"

    def _score_overlap(self) -> int:
        return self._ini_font_int("scoreoverlap")

    def _combo_overlap(self) -> int:
        """skin.ini [Fonts] ComboOverlap — px the combo digits overlap. Red's
        skin sets 8; we previously always used 0, leaving a visible gap."""
        v = getattr(self, "_combo_overlap_cache", None)
        if v is not None:
            return v
        out = 0
        try:
            ini = self.dir / "skin.ini" if self.dir else None
            if ini and ini.is_file():
                in_fonts = False
                for raw in ini.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw.strip()
                    if line.startswith("["):
                        in_fonts = line.lower() == "[fonts]"
                        continue
                    if in_fonts and ":" in line:
                        k, _, val = line.partition(":")
                        if k.strip().lower() == "combooverlap":
                            out = int(float(val.strip()))
            
        except (OSError, ValueError):
            out = 0
        self._combo_overlap_cache = out
        return out

    def _font_prefixes(self) -> tuple[str, str]:
        """(ScorePrefix, ComboPrefix) from skin.ini [Fonts]; defaults score/combo.
        Backslashes → forward slashes (osu path convention). Reads only the
        shallowest skin.ini (what the HUD's resolved skin root points at)."""
        score, combo = "score", "score"   # lazer: ComboPrefix defaults to "score"
        if self.dir is None:
            return score, combo
        ini = self.dir / "skin.ini"
        if not ini.is_file():
            return score, combo
        try:
            in_fonts = False
            for raw in ini.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if line.startswith("["):
                    in_fonts = line.lower() == "[fonts]"
                    continue
                if in_fonts and ":" in line:
                    k, _, v = line.partition(":")
                    k, v = k.strip().lower(), v.strip().replace("\\", "/")
                    if k == "scoreprefix" and v:
                        score = v
                    elif k == "comboprefix" and v:
                        combo = v
        except OSError:
            pass
        return score, combo

    def _glyphs(self, prefix: str, target_h: int,
                *, default_ok: bool = True) -> dict:
        chars = {str(i): str(i) for i in range(10)}
        chars.update({"comma": "comma", "dot": "dot", "percent": "percent", "x": "x"})
        out = {}
        for key, suffix in chars.items():
            im = self._load(f"{prefix}-{suffix}", target_h, default_ok=default_ok)
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
                if name == "suddendeath" and (mods & 16384):
                    continue  # perfect icon already covers it (PF = SD|Perfect)
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

    # catch judgment row: Fruit / Drop / Droplet / Miss, colour-coded (droplet
    # = lb_cards.RESULT_COLORS light blue, not gray — 2026-07-22 polish)
    # Miss = count_miss ONLY (tiny-droplet misses are not misses in either
    # client); missed tinies stay visible via the Droplet caught/total cell.
    cells = [("Fruit", str(meta.count_300), (255, 230, 120)),
             ("Drop", str(meta.count_100), (140, 220, 140)),
             ("Droplet",
              (f"{meta.count_50}/{meta.count_50 + meta.count_katu}"
               if meta.count_katu else str(meta.count_50)), (153, 219, 255)),
             ("Miss", str(meta.count_miss), (240, 80, 80))]
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
