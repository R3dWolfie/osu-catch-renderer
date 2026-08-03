"""CLI entrypoint, parity with osu_renderer.cli.

    python -m osu_catch_renderer REPLAY.osr BEATMAP_DIR -o out.mp4 \
        [--resolution 1920x1080] [--fps 60] [--encoder auto]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from osu_catch_renderer.beatmap.models import RenderConfig
from osu_catch_renderer.render.render import render_catch


def _resolution(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="osu_catch_renderer")
    ap.add_argument("osr", type=Path, help="replay .osr file")
    ap.add_argument("beatmap_dir", type=Path, help="dir with .osu + audio + bg")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--resolution", type=_resolution, default=(1920, 1080))
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--encoder", default="auto",
                    help="auto | h264_vaapi | h264_nvenc | libx264")
    ap.add_argument("--encoder-device", default=None, help="e.g. /dev/dri/renderD128")
    ap.add_argument("--skin", type=Path, default=None, help="extracted skin dir (e.g. Night05)")
    ap.add_argument("--default-skin", type=Path, default=None, help="fallback skin dir")
    ap.add_argument("--combo-colors", choices=("beatmap", "skin"), default="beatmap",
                    help="combo-colour source: beatmap [Colours] (default) or the skin's")
    ap.add_argument("--overlay-osr", action="append", type=Path, default=[],
                    help="extra replay(s) for a versus OVERLAY (repeatable)")
    ap.add_argument("--player-skin", "--catcher-skin", dest="player_skin",
                    action="append", type=str, default=[], metavar="DIR",
                    help="per-player skin dir for the OVERLAY platter (catcher) "
                         "art — repeatable, one per player, aligned [primary, "
                         "then each --overlay-osr in order]. '' (empty) or '-' "
                         "= no preset -> that player keeps the base --skin "
                         "catcher. The art is grayscaled + hue-tinted per "
                         "player exactly like the base catcher; only the art "
                         "under the tint changes. (--catcher-skin is a "
                         "compatible alias.)")
    BA = argparse.BooleanOptionalAction
    ap.add_argument("--skip-intro", action=BA, default=True, help="start at first object")
    ap.add_argument("--results", action=BA, default=True, help="results-screen outro")
    ap.add_argument("--countdown", action=BA, default=False, help="3-2-1 before first object")
    ap.add_argument("--letterbox-breaks", action=BA, default=True)
    ap.add_argument("--hyperdash", action=BA, default=True)
    ap.add_argument("--fruit-rotation", action=BA, default=True)
    ap.add_argument("--dash-trail", action=BA, default=True)
    ap.add_argument("--banana-rainbow", action=BA, default=True)
    ap.add_argument("--pp-counter", action=BA, default=True)
    ap.add_argument("--pp", type=float, default=None,
                    help="EXACT final pp to show (osu's OFFICIAL pp). The "
                         "results card + the live counter's ENDPOINT are "
                         "pinned to this; the live curve keeps its rosu "
                         "shape. Omit to keep the rosu estimate.")
    ap.add_argument("--sr", type=float, default=None,
                    help="EXACT star rating to show (osu's OFFICIAL SR). The "
                         "results card's star-rating pill is pinned to this. "
                         "Omit to keep the rosu SR estimate.")
    ap.add_argument("--hit-counter", action=BA, default=True)
    ap.add_argument("--key-counter", action=BA, default=True,
                    help="Argon key counter bottom-right (B1/B2/B3 = "
                         "move-left / move-right / dash press counts)")
    ap.add_argument("--watermark", default="")
    ap.add_argument("--music-volume", type=int, default=100)
    ap.add_argument("--general-volume", type=int, default=100)
    ap.add_argument("--audio-offset", type=int, default=0, help="ms; -earlier")
    ap.add_argument("--beatmap-hitsounds", action=BA, default=True,
                    help="use the beatmap's custom hitsound samples; OFF = "
                         "resolve from the skin chain only (the site's "
                         "'Use the beatmap's hitsounds' toggle)")
    ap.add_argument("--hitsounds", default="on", metavar="MODE",
                    help="caught-object hitsounds under the music (stable "
                         "behaviour). 'off'/'none' disable; any other value "
                         "(incl. NAS-spec words like perfect/score/acc) = on "
                         "(default)")
    ap.add_argument("--hitsound-volume", type=int, default=100,
                    help="hitsound track volume 0-100 (default 100)")
    ap.add_argument("--nightcore-hitsounds", action=BA, default=False,
                    help="beat overlay (metronome): clap each beat + finish "
                         "each downbeat across the whole song (mod-independent)")
    ap.add_argument("--bg-dim-intro", type=int, default=0)
    ap.add_argument("--bg-dim-game", type=int, default=70)
    ap.add_argument("--bg-dim-breaks", type=int, default=0)
    ap.add_argument("--bg-blur", type=int, default=0)
    ap.add_argument("--results-seconds", type=float, default=None)
    ap.add_argument("--show-combo", action=BA, default=True)
    ap.add_argument("--show-score", action=BA, default=True)
    ap.add_argument("--show-hp", action=BA, default=True)
    ap.add_argument("--show-grade", action=BA, default=True)
    ap.add_argument("--show-mods", action=BA, default=True)
    ap.add_argument("--logo", action=BA, default=False,
                    help="show_logo: the R3D 'R' tile splash during the intro, "
                         "fading out as gameplay starts (parity with std)")
    ap.add_argument("--leaderboard", action=BA, default=True,
                    help="per-map render leaderboard on the results screen "
                         "(featured play flanked by other renders of the same "
                         "map, from the local render DB); default on")
    ap.add_argument("--leaderboard-source", choices=("r3d", "osu"),
                    default="r3d",
                    help="flank-card source: 'r3d' = the local render DB "
                         "(default), 'osu' = the map's osu! GLOBAL top scores "
                         "from --leaderboard-json (silently falls back to r3d "
                         "when that file is missing/empty/invalid)")
    ap.add_argument("--leaderboard-json", type=Path, default=None,
                    help="path to the bot-written osu! global scores JSON "
                         "(only read when --leaderboard-source osu)")
    ap.add_argument("--featured-avatar-png", type=Path, default=None,
                    help="PNG of the FEATURED player's REAL osu! avatar for "
                         "the results CENTRE card (service passes the player's "
                         "osu! pfp). Absent -> the procedural username chip. "
                         "The old render-DB Discord lookup is gone, so the "
                         "card can never show the owner's pfp.")
    ap.add_argument("--storyboard", action=BA, default=False,
                    help="parse the .osu/.osb and render the map's storyboard "
                         "(in-house engine, parity with std/taiko). DEFAULT "
                         "OFF; when off the render is byte-identical to today's.")
    args = ap.parse_args(argv)

    # --player-skin count must align to the player list [primary, *overlay-osr].
    # A mismatch NEVER kills a render: pad with "" (= base skin) / truncate,
    # loudly, so a bot-side alignment bug degrades to base-skin platters.
    player_skins = list(args.player_skin or [])
    if player_skins:
        expected = 1 + len(args.overlay_osr or [])
        if len(player_skins) != expected:
            print(f"[catch-renderer] warning: {len(player_skins)} "
                  f"--player-skin entries for {expected} players "
                  f"(primary + {len(args.overlay_osr or [])} overlay); "
                  "padding/truncating with base-skin entries",
                  file=sys.stderr)
            player_skins = (player_skins + [""] * expected)[:expected]

    cfg = RenderConfig(
        resolution=args.resolution,
        fps=args.fps,
        encoder=args.encoder,
        encoder_device=args.encoder_device,
        skin_dir=args.skin,
        default_skin_dir=args.default_skin,
        skip_intro=args.skip_intro,
        show_results=args.results,
        show_countdown=args.countdown,
        letterbox_breaks=args.letterbox_breaks,
        show_hyperdash=args.hyperdash,
        fruit_rotation=args.fruit_rotation,
        catcher_dash_trail=args.dash_trail,
        banana_rainbow=args.banana_rainbow,
        show_pp_counter=args.pp_counter,
        pp_override=args.pp,
        sr_override=args.sr,
        show_hit_counter=args.hit_counter,
        show_key_counter=args.key_counter,
        watermark=args.watermark,
        music_volume=args.music_volume,
        general_volume=args.general_volume,
        audio_offset_ms=args.audio_offset,
        # off/none (any case) disable; unknown values are gracefully ON — the
        # NAS spec's "hitsounds" field carries mania-flavoured words for some
        # engines, and only "none" maps to a meaning here.
        hitsounds=str(args.hitsounds).strip().lower()
        not in ("off", "none", "0", "false", "no"),
        hitsound_volume=max(0, min(100, args.hitsound_volume)),
        beatmap_hitsounds=args.beatmap_hitsounds,
        combo_colors=args.combo_colors,
        nightcore_hitsounds=args.nightcore_hitsounds,
        bg_dim_intro=args.bg_dim_intro,
        bg_dim_game=args.bg_dim_game,
        bg_dim_breaks=args.bg_dim_breaks,
        bg_blur=args.bg_blur,
        show_combo=args.show_combo,
        show_score=args.show_score,
        show_hp_bar=args.show_hp,
        show_grade=args.show_grade,
        show_mods=args.show_mods,
        show_logo=args.logo,
        show_leaderboard=args.leaderboard,
        leaderboard_source=args.leaderboard_source,
        leaderboard_json=args.leaderboard_json,
        load_storyboard=args.storyboard,
    )
    if args.results_seconds is not None:
        cfg.results_ms = int(args.results_seconds * 1000)

    def progress(pct: int) -> None:
        print(f"\rrendering… {pct:3d}%", end="", file=sys.stderr, flush=True)

    # Hand the FEATURED player's osu! avatar PNG to the results screen (the
    # centre card). Set once here, before rendering; a missing/unreadable file
    # leaves the featured card on the procedural chip. Replaces the old
    # render-DB Discord lookup (which could show the SITE OWNER's pfp).
    if args.featured_avatar_png is not None:
        try:
            from osu_catch_renderer.hud import lazer_results as _lr
            _lr.set_featured_avatar_png(args.featured_avatar_png)
        except Exception:  # noqa: BLE001 -- avatar wiring never breaks a render
            pass

    out = render_catch(args.osr, args.beatmap_dir, args.output, cfg,
                       progress_callback=progress,
                       overlay_osr=args.overlay_osr or None,
                       catcher_skins=player_skins or None)
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
