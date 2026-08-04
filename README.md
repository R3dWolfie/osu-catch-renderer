# osu-catch-renderer

Renders osu!catch (mode 2 / "fruits") replays to MP4.

Part of the [R3D Renderer](https://renderer.r3dwolfie.com) — a self-hosted osu! replay→MP4 service (Discord bot + website, package `mania_ordr`) that dispatches each render to a per-mode engine repo. **This is the catch engine.** The core invokes it as a fresh subprocess per render, so an edit-in-place deploys on the next render with no service restart. The prod branch is `catch-v2`.

## What it renders / fidelity

Pipeline: parse `.osu` → generate catch objects → parse `.osr` → per-frame scene (falling fruit + catcher driven by the replay) → GL sprite draw → ffmpeg encode.

- **Catch gameplay**: falling fruit/droplets/bananas, replay-driven catcher, hyperdash, fruit rotation, dash trail, banana rainbow (each individually toggleable).
- **Lazer-standardised scoring (ScoreV3)**: shows one score scale everywhere — osu!lazer's standardised total (max ~1,000,000 × mod multiplier). `beatmap/score_fidelity.py` converts the legacy `.osr` header under every source interpretation (stable ScoreV1, website classic-export, lazer local export) and picks the candidate closest to the in-house `CatchSim` simulation. Scoring/geometry logic is a behavioural Python port of osu!'s `osu.Game.Rulesets.Catch` (`Catcher.cs`, `CatchScoreProcessor.cs`, `ApplyPositionOffsets`, etc.; ppy/osu, MIT).
- **Argon HUD/skin**: the default look. Bundled Argon combo/counter art lives in `argon_assets/`; the Argon HUD, health bar, and score counter are drawn in-house (`argon/`). Optional external skin dirs can be supplied.
- **Versus overlay**: multiple replays composited on one field via `--overlay-osr`, with per-player catcher skins (`--player-skin` / `--catcher-skin`), hue-tinted per player.
- **Storyboard**: in-house engine (parity with the std/taiko engines), default **off**; when off the output is byte-identical to a no-storyboard render.
- **pp / SR**: an optional PP counter; values can be passed in (`--pp`, `--sr`) or computed via `rosu_pp_py` when available.

No osu! game assets are bundled — only original or procedurally-generated art, plus the Nunito variable font (SIL OFL 1.1, under `assets/fonts/`) used for skinless "Argon league" text.

## Usage

Invoked as a module (`__main__` → `osu_catch_renderer.cli:main`):

```
python -m osu_catch_renderer REPLAY.osr BEATMAP_DIR -o out.mp4 \
    [--resolution 1920x1080] [--fps 60] [--encoder auto]
```

Positional arguments:

| Arg | Meaning |
| --- | --- |
| `osr` | replay `.osr` file |
| `beatmap_dir` | directory containing the `.osu` + audio + background |

Key options (all verified in `cli.py`):

- `-o, --output` (required) — output path
- `--resolution WxH` (default `1920x1080`), `--fps` (default `60`)
- `--encoder` — `auto` | `h264_vaapi` | `h264_nvenc` | `libx264`; `--encoder-device` (e.g. `/dev/dri/renderD128`)
- `--skin DIR`, `--default-skin DIR` — extracted skin dir + fallback
- `--combo-colors {beatmap,skin}` (default `beatmap`)
- `--overlay-osr OSR` (repeatable) — extra replay(s) for a versus overlay
- `--player-skin` / `--catcher-skin DIR` (repeatable) — per-player catcher art, aligned `[primary, *overlay-osr]`
- `--storyboard` / `--no-storyboard` (default off)
- `--pp FLOAT`, `--sr FLOAT`, and boolean toggles like `--hyperdash`, `--fruit-rotation`, `--dash-trail`, `--banana-rainbow`, `--pp-counter`, `--hit-counter`, `--key-counter`, `--leaderboard`, `--results`, `--letterbox-breaks`
- audio: `--music-volume`, `--general-volume`, `--hitsound-volume`, `--audio-offset MS`, `--hitsounds MODE`, `--beatmap-hitsounds`, `--nightcore-hitsounds`
- background: `--bg-dim-intro`, `--bg-dim-game` (default 70), `--bg-dim-breaks`, `--bg-blur`
- HUD element toggles: `--show-combo`, `--show-score`, `--show-hp`, `--show-grade`, `--show-mods`, `--logo`, `--watermark TEXT`

Run `python -m osu_catch_renderer --help` for the complete, authoritative list.

## Requirements

- Python 3.10+ (uses `X | Y` type unions and `BooleanOptionalAction`)
- **`ffmpeg`** on `PATH` — used both to decode audio/hitsound samples and as the video encode sink (raw rgb24 piped to a subprocess)
- Third-party Python packages (imported in-tree): `numpy`, `Pillow` (PIL), `osrparse`, `moderngl` (GL sprite pipeline), and `rosu_pp_py` (lazily imported, only for pp computation)

There is no `requirements.txt` or `pyproject.toml` in the repo — dependencies are supplied by the surrounding R3D Renderer environment / shared venv (verify).

## Layout

```
osu_catch_renderer/
  __main__.py, cli.py      # entry point + argparse CLI
  beatmap/                 # .osu/.osr parsing, catch object gen, hitsounds,
                           #   storyboard parse, legacy_random, score_fidelity
  render/                  # orchestrator (render.py), GL (gl.py), scene,
                           #   overlay, effects, flashlight, storyboard engine,
                           #   loudnorm cache, dim, break overlay
  skin/                    # skin loading + lazer skin + asset resolution
  hud/                     # HUD, lazer HUD/results, leaderboard, fonts
  argon/                   # Argon HUD, health bar, score counter
  argon_assets/            # bundled Argon counter art (PNG)
  assets/                  # logo, Nunito font (OFL), default nightcore samples
test/                      # storyboard / lead-in tests
COPYRIGHT, LICENSE
```

## License

**GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).** Copyright (C) 2026 Cool Adults. See `LICENSE` for the full text and `COPYRIGHT` for third-party attribution (osu! / ppy ports under MIT; danser-go studied as a behavioural reference under GPL-3.0; bundled Nunito font under SIL OFL 1.1).

osu! is a rhythm game by peppy / ppy Pty Ltd. This is an independent, unofficial renderer and is not affiliated with or endorsed by ppy.
