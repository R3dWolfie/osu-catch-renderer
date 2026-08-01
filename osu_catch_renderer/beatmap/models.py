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
class HitSample:
    """The hitsound spec one catchable carries out of beatmap parsing —
    the lazer `HitObject.Samples` equivalent, catch-shaped:

      * kind "hit"    — a fruit (circle / juice-stream head/repeat/tail):
        hitnormal always + whistle/finish/clap per `bits` (osu bits 2/4/8);
        lazer JuiceStream: fruit edges carry GetNodeSamples(nodeIndex).
      * kind "tick"   — a large droplet: lazer JuiceStream renames the
        stream's samples to "slidertick" (Samples.Select(s.With(@"slidertick"))).
      * kind "banana" — Banana.default_banana_samples (BananaHitSampleInfo:
        "Gameplay/metronomelow" / "Gameplay/catch-banana", volume 100).

    TINY droplets carry NO HitSample (None) — lazer's TinyDroplet is created
    with no Samples, so a caught tiny is silent.

    Zero set/index/volume mean "resolve from the active timing point at play
    time" (stable semantics — hitsounds.py does that resolution)."""
    bits: int = 0            # whistle/finish/clap addition bits (2/4/8)
    normal_set: int = 0      # 0=timing point's; 1/2/3=normal/soft/drum
    addition_set: int = 0    # 0=inherit the (resolved) normal set
    index: int = 0           # custom sample index; 0=timing point's
    volume: int = 0          # 0..100; 0=timing point's
    filename: str = ""       # per-object custom file (beatmap dir), plays alone
    kind: str = "hit"        # "hit" | "tick" | "banana"


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
    # hitsound spec (None = silent when caught — tiny droplets); see HitSample
    sample: HitSample | None = None


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
    combo_colors: list = field(default_factory=list)   # [Colours] Combo1..N (RGB)
    # hitsound timing data (beatmap._Timing — sample_info(t) resolves the
    # active sampleSet/sampleIndex/volume) + the [General] SampleSet default.
    timing: object = None
    sample_set_default: int = 1   # 1=normal 2=soft 3=drum

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
    show_key_counter: bool = True       # Argon key counter (B1/B2/B3), bottom-right
    # EXACT final pp to display (osu's OFFICIAL pp, supplied by the service
    # via --pp). None -> keep the rosu estimate. When set, the results-card
    # PP AND the live counter's ENDPOINT are pinned to this value (the live
    # curve keeps its rosu/score-progress SHAPE; only the endpoint is
    # anchored). Mirrors the taiko renderer's pp_override -- see
    # scene.compute_pp_curve + lazer_results.CatchLazerResults.
    pp_override: float | None = None
    # EXACT star rating to display (osu's OFFICIAL SR, supplied by the service
    # via --sr). None -> keep the rosu SR estimate. When set, the results
    # card's star-rating pill shows this value exactly. Static display value
    # (no live SR counter). Mirrors pp_override -- see
    # lazer_results.CatchLazerResults.
    sr_override: float | None = None
    watermark: str = ""                # bottom-right branding (free renders forced to site URL)
    # audio (0..100 from preset; 100 = unchanged)
    music_volume: int = 100
    general_volume: int = 100
    audio_offset_ms: int = 0            # +later / -earlier, relative to gameplay
    # caught-object hitsounds mixed under the music (stable behaviour: every
    # CAUGHT fruit/droplet plays its hit samples, misses play nothing).
    # Default ON to match the game — the bot passes nothing for catch singles.
    hitsounds: bool = True
    hitsound_volume: int = 100          # 0..100 gain on the hitsound track
    # False = ignore the beatmap's custom samples (and per-object filename
    # overrides); resolution then starts at the skin chain. Mirrors the
    # site preset "Use the beatmap's hitsounds".
    beatmap_hitsounds: bool = True
    # Beat overlay (metronome): clap each beat + finish each downbeat across the
    # whole song, mixed into the hitsound track. Off by default; the site's
    # "Beat overlay (metronome)" toggle. Mod-independent (a general metronome).
    nightcore_hitsounds: bool = False
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
    # texture UV offset/scale — the storyboard mirrors flipped sprites via a UV
    # flip instead of a negative GL size. Identity defaults ((0,0)/(1,1)) make
    # `in_uv * uv_scale + uv_off == in_uv`, so every existing (gameplay/HUD)
    # sprite is sampled bit-identically to before.
    uv_off: tuple[float, float] = (0.0, 0.0)
    uv_scale: tuple[float, float] = (1.0, 1.0)


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
    # catcher input state for the HUD's Argon key counter (set by
    # CatchSim.build_scene from the replay frames)
    catcher_x: float = 0.0            # catcher centre x, osu units 0..512
    dashing: bool = False             # dash key held this frame (replay)
    # screen-space catcher geometry for the HUD's catcher-tracking combo
    # counter (LegacyCatchComboCounter). None on paths that don't fill them
    # (the HUD then falls back to its old fixed placement).
    catcher_px: float | None = None   # catcher centre x, SCREEN px
    plane_y_px: float | None = None   # catch plane y, SCREEN px
    pf_unit_px: float | None = None   # screen px per osu playfield unit
    # Key-overlay input state at REPLAY-FRAME resolution (CatchSim.input_state):
    # keys_held = (left, right, dash) held anywhere within this video frame's
    # map-time interval; key_counts = cumulative press onsets. None on paths
    # that don't fill them — the HUD then falls back to its per-video-frame
    # dx derivation (which aliases rapid taps; see scene._build_inputs).
    keys_held: tuple | None = None
    key_counts: tuple | None = None
    # Storyboard compositing hooks (only read when --storyboard is on):
    # `bg_split` = index in `sprites` where the beatmap background ends and the
    # playfield begins, so the SB underlay can be drawn between them; the
    # playfield draws on top. `sb_brightness` = 1 - background dim envelope, so
    # the storyboard is tinted by the SAME dim as the bg (DimmableStoryboard).
    bg_split: int = 0
    sb_brightness: float = 1.0


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
