"""Data model for the catch renderer.

osu! playfield x is 0..512 (the std coordinate space; catch maps x directly).
Time is in absolute milliseconds from the start of the audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ObjType(Enum):
    FRUIT = "fruit"          # a caught-or-missed big fruit (circle / slider head/tail/repeat)
    DROPLET = "droplet"      # slider-path droplet (counts toward score/combo)
    TINY_DROPLET = "tiny"    # slider-path tiny droplet (accuracy only, no combo break)
    BANANA = "banana"        # spinner banana shower


@dataclass(frozen=True)
class CatchObject:
    """One catchable thing falling toward the catcher plane.

    `x` is the osu! x of the *center* in 0..512. `combo_index` selects the
    fruit colour/sprite. `hyperdash` marks a fruit that requires a hyperdash
    to reach the next one (`hyper_target_x` is where the catcher must be by
    the next object so we can colour the dash).
    """
    time_ms: int
    x: float
    kind: ObjType
    combo_index: int = 0
    new_combo: bool = False
    hyperdash: bool = False
    hyper_target_x: float | None = None


@dataclass(frozen=True)
class CatchFrame:
    """One replay frame: catcher center x in 0..512, and whether the dash
    key is held this frame."""
    time_ms: int
    x: float
    dashing: bool


@dataclass
class CatchBeatmap:
    objects: list[CatchObject]
    cs: float = 5.0
    ar: float = 9.0
    od: float = 7.0
    hp: float = 5.0
    audio_filename: str | None = None
    background: str | None = None
    breaks: list = field(default_factory=list)   # [(start_ms, end_ms)] break periods
    title: str = ""
    artist: str = ""
    version: str = ""
    creator: str = ""   # mapper (the results screen's "mapped by …")
    rate: float = 1.0   # playback rate (DT/NC=1.5, HT=0.75); times already scaled

    @property
    def length_ms(self) -> int:
        return max((o.time_ms for o in self.objects), default=0)


@dataclass(frozen=True)
class ReplayMeta:
    mode: int
    beatmap_md5: str
    player_name: str
    mods: int
    score: int
    max_combo: int
    count_300: int
    count_100: int
    count_50: int
    count_katu: int
    count_miss: int
    accuracy: float
    grade: str
    game_version: int = 0   # osr client version; <30000000 = osu!stable
    # Map-time (ms) at which the player failed (HP hit 0), from the .osr
    # life-bar graph. None = the play passed / no fail detected. Used to
    # end the render at death instead of rendering the unplayed remainder.
    death_ms: int | None = None
    # When the play happened (osrparse's parsed .osr timestamp, a datetime);
    # None when unavailable. The results screen's "Played on …" footer.
    timestamp: object = None


@dataclass
class RenderConfig:
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 60
    encoder: str = "auto"               # auto | h264_vaapi | h264_nvenc | libx264
    encoder_device: str | None = None   # e.g. /dev/dri/renderD128
    skin_dir: Path | None = None        # user skin (extracted .osk dir, e.g. Night05)
    default_skin_dir: Path | None = None  # fallback skin for missing elements
    # visual tuning
    lead_in_ms: int = 1500              # blank/approach before first object
    tail_ms: int = 1500                 # hold after last object
    catcher_plane: float = 0.86         # fraction of screen height where the plate sits
    playfield_margin: float = 0.0       # osu-x margin handled in geometry
    # --- settings-menu options ---
    skip_intro: bool = True             # start at first object; else render full intro from 0
    show_countdown: bool = False        # "3-2-1" before the first object
    show_results: bool = True           # results/ranking screen outro
    results_ms: int = 4500              # how long to hold the results screen
    letterbox_breaks: bool = True       # dim + letterbox bars during map breaks
    show_hyperdash: bool = True         # red hyper-fruit + dash glow
    fruit_rotation: bool = True         # fruits spin as they fall
    catcher_dash_trail: bool = True     # afterimage trail on dash
    banana_rainbow: bool = True         # rainbow banana showers
    show_pp_counter: bool = True        # running pp counter (needs rosu-pp-py)
    show_hit_counter: bool = True       # 300/100/50/miss tallies
    watermark: str = ""                # bottom-right branding (free renders forced to site URL)
    # audio (0..100 from preset; 100 = unchanged)
    music_volume: int = 100
    general_volume: int = 100
    audio_offset_ms: int = 0            # +later / -earlier, relative to gameplay
    # background (% dim 0..100; higher = darker. blur in px)
    bg_dim_intro: int = 0
    bg_dim_game: int = 70
    bg_dim_breaks: int = 0
    bg_blur: int = 0
    # HUD element toggles
    show_combo: bool = True
    show_score: bool = True
    show_hp_bar: bool = True
    show_grade: bool = True
    show_mods: bool = True
    # intro R3D "R" splash (parity with std show_logo; off by default so
    # existing renders are unchanged)
    show_logo: bool = False
    # results-screen map leaderboard (parity with the std renderer): the featured
    # play flanked by compact ranked cards of the OTHER renders of this map.
    # Default source = the local render DB; "osu" reads the bot-written osu!
    # global scores JSON (falls back to the DB when absent). Default-on but a
    # no-op when the map has no other renders, so existing renders are unchanged.
    show_leaderboard: bool = True
    leaderboard_source: str = "r3d"      # r3d | osu
    leaderboard_json: Path | None = None


@dataclass
class Sprite:
    """A single textured/coloured quad to draw this frame (back-to-front)."""
    x: float                 # screen px, center
    y: float                 # screen px, center
    w: float
    h: float
    texture_key: str | None = None      # atlas key; None = solid colour quad
    color: tuple[float, float, float, float] = (1, 1, 1, 1)
    rotation: float = 0.0
    additive: bool = False           # additive blend (glow / catch explosion)


@dataclass
class SceneState:
    """Everything to draw for one frame, plus HUD numbers."""
    sprites: list[Sprite] = field(default_factory=list)
    combo: int = 0
    score: int = 0
    accuracy: float = 1.0
    hp: float = 1.0
    time_ms: int = 0
    pp: float = 0.0
    counts: tuple = (0, 0, 0, 0, 0)   # (fruit, large-drop, tiny, miss-tiny, miss)


# osu!catch geometry constants -------------------------------------------------

PLAYFIELD_WIDTH = 512.0   # osu x units


def cs_to_catcher_half_width(cs: float) -> float:
    """Catcher catch-range half-width in osu x units.

    osu!catch: catcherWidth scales with CS. Base catcher is 106.75 px at the
    playfield scale; the catchable plate is ~0.8 of that. We approximate with
    the documented scaling: size factor = 1 - 0.7*(CS-5)/5, clamped.
    """
    scale = 1.0 - 0.7 * (cs - 5.0) / 5.0
    scale = max(0.25, min(1.75, scale))
    base_half = 106.75 / 2.0
    return base_half * scale * 0.8


def ar_to_preempt_ms(ar: float) -> float:
    """Time a falling object is on screen before reaching the plate (ms).

    Standard osu AR->preempt curve (same as std): AR5=1200ms, faster above.
    """
    if ar < 5.0:
        return 1200.0 + 600.0 * (5.0 - ar) / 5.0
    if ar > 5.0:
        return 1200.0 - 750.0 * (ar - 5.0) / 5.0
    return 1200.0
