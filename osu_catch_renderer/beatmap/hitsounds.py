"""Caught-object hitsounds — offline pre-mix under the music.

In the game every CAUGHT object plays its hit samples; misses play nothing.
lazer ground truth (osu.Game.Rulesets.Catch + osu.Game @ master, read
2026-07-22):

  * Playback is gated on the judgement: DrawableHitObject.UpdateState —
    ``if (!force && newState == ArmedState.Hit) PlaySamples();`` — a catch
    ruleset object reaches ArmedState.Hit only when caught, so missed
    fruits/droplets are SILENT.
  * JuiceStream.CreateNestedHitObjects: large droplets play the stream's
    samples renamed "slidertick" (``Samples.Select(s => s.With(@"slidertick"))``),
    head/repeat/tail fruits play ``GetNodeSamples(nodeIndex++)`` (edgeSounds/
    edgeSets), and TinyDroplet is created with NO samples — caught tinies are
    silent.
  * Banana: ``default_banana_samples`` = BananaHitSampleInfo, lookup
    "Gameplay/metronomelow" / "Gameplay/catch-banana", volume 100 — never the
    spinner's own hitsounds.
  * Rate mods: lazer's ModRateAdjust.ApplyToSample pitches samples with the
    rate (AdjustableProperty.Frequency), but STABLE plays hit samples at
    natural pitch (only the music is tempo-shifted). Like every other R3D
    engine we match stable: event TIMES compress by the clock rate
    (wall = (t_map - start_ms)/rate), the sample PCM is untouched.

Sample resolution is stable's (ported from the std renderer's proven
record/hitsounds.py SampleBank + the mania v2 hitsounds.py shape):
BEATMAP dir by the custom sample index (index 1 = bare ``<set>-<sound>``,
N>=2 = ``<set>-<sound>N``; missing file falls through; a ZERO-BYTE file is
deliberate silence — the classic blanking convention) → SKIN chain (user skin
dir, then default skin dir; .wav/.ogg/.mp3) → deterministic SYNTH default so
a render never goes silent. Timing-point sampleSet/sampleIndex/volume resolve
at the catch time; per-object hitSample fields override when non-zero; volume
floors at 0.08 (stable).

The mix is one float32 stereo 44.1 kHz WAV on the VIDEO time axis, handed to
the encode ffmpeg as a second audio input. render._spawn_ffmpeg loudnorms the
SONG ALONE and amixes this track on top afterwards — the mania v2 LOUDNORM
FIX (2026-07-12, #17): loudnorm on the song+hits mix let its gain duck the
song ~4 dB under every hit peak.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_EXTS = (".wav", ".ogg", ".mp3")   # stable GetSample extension order
# Bundled osu! DEFAULT nightcore drums (ppy/osu-resources Legacy skin) — final
# fallback for the NC-mod overlay when the skin OMITS a nightcore sample.
_DEFAULT_NC_DIR = Path(__file__).resolve().parent.parent / "assets" / "default_nightcore"
SET_NAMES = {1: "normal", 2: "soft", 3: "drum"}
ADDITIONS = ((2, "hitwhistle"), (4, "hitfinish"), (8, "hitclap"))
# global hit gain over per-event volume — the mania v2 DEFAULT_HIT_GAIN
# (music sits at loudnorm -18 LUFS; hits ride on top at this ceiling (2026-07-31: -8 LU rebalance vs music)).
DEFAULT_HIT_GAIN = 0.22
VOLUME_FLOOR = 0.08       # stable floors sample volume at 8%


# --- decode -------------------------------------------------------------------

def _decode_pcm(path: Path) -> np.ndarray | None:
    """Decode a sample file -> float32 stereo 44.1k (N,2) via ffmpeg (the
    venv has no soundfile — same no-BASS subprocess shape as the std
    renderer's decode_to_pcm). Zero-byte files mean SILENCE; undecodable
    files return None (fall through to the next source)."""
    try:
        if path.stat().st_size == 0:
            return np.zeros((1, CHANNELS), dtype=np.float32)
    except OSError:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-f", "f32le", "-acodec", "pcm_f32le",
           "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=30)
    except Exception:  # noqa: BLE001 — a bad sample never kills a render
        return None
    if proc.returncode != 0:
        return None       # undecodable -> fall through to the next source
    # a VALID decode of zero frames (e.g. the 44-byte header-only WAVs skins
    # ship to blank a sound) is deliberate SILENCE, same as a zero-byte file
    # — it must STOP the chain, not fall through to the synth default.
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    n = (len(pcm) // CHANNELS) * CHANNELS
    if n == 0:
        return np.zeros((1, CHANNELS), dtype=np.float32)
    return pcm[:n].reshape(-1, CHANNELS).copy()


# --- synthesized defaults (compact port of the std renderer's two banks) ------

def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE


def _noise(dur: float, seed_name: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(seed_name.encode()))
    n = rng.standard_normal(int(dur * SAMPLE_RATE)).astype(np.float32)
    k = np.ones(8, dtype=np.float32) / 8.0
    return np.convolve(n, k, mode="same")


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.repeat(x.astype(np.float32)[:, None], 2, axis=1)


def synth_style_for(has_custom_skin: bool) -> str:
    """Which synth bank fills missing samples — the std renderer's league
    rule: skinless renders synthesize the ARGON (lazer-ish) family, a custom
    skin's gaps synthesize the LEGACY (classic-osu) character."""
    return "legacy" if has_custom_skin else "argon"


def synth_sample(name: str, style: str = "argon") -> np.ndarray:
    """Deterministic placeholder for `name` ("<set>-<sound>" or "banana") —
    used only when neither the beatmap nor any skin dir provides the file."""
    if name == "banana":
        # lazer "Gameplay/metronomelow": a short low metronome tick
        t = _t(0.06)
        x = (0.55 * np.sin(2 * math.pi * 820.0 * t) * np.exp(-90.0 * t)
             + 0.22 * _noise(0.06, "banana") * np.exp(-250.0 * t))
        return _stereo(x)
    if style == "legacy":
        return _synth_legacy(name)
    return _synth_argon(name)


def _synth_argon(name: str) -> np.ndarray:
    """The ARGON (lazer-family) bank — std renderer _synth_argon, trimmed to
    the sounds catch can emit."""
    base = name.split("-", 1)[-1]
    if base == "hitnormal":
        t = _t(0.05)
        x = (0.75 * np.sin(2 * math.pi * 500.0 * t) * np.exp(-70.0 * t)
             + 0.3 * _noise(0.05, name) * np.exp(-220.0 * t))
    elif base == "hitwhistle":
        t = _t(0.12)
        x = 0.55 * np.sin(2 * math.pi * 1250.0 * t) * np.exp(-25.0 * t)
    elif base == "hitfinish":
        t = _t(0.4)
        x = (0.4 * _noise(0.4, name)
             + 0.3 * np.sin(2 * math.pi * 880.0 * t)
             + 0.2 * np.sin(2 * math.pi * 1320.0 * t)) * np.exp(-8.0 * t)
    elif base == "hitclap":
        t = _t(0.06)
        x = 0.8 * _noise(0.06, name) * np.exp(-80.0 * t)
    elif base == "slidertick":
        t = _t(0.02)
        x = 0.5 * np.sin(2 * math.pi * 3000.0 * t) * np.exp(-300.0 * t)
    else:   # unknown name — quiet click, never silence
        t = _t(0.03)
        x = 0.4 * np.sin(2 * math.pi * 1000.0 * t) * np.exp(-150.0 * t)
    return _stereo(x)


def _lowpass(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    kern = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kern, mode="same").astype(np.float32)


def _highpass(x: np.ndarray, k: int) -> np.ndarray:
    return (x - _lowpass(x, k)).astype(np.float32)


def _lg_noise(dur: float, name: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(f"lg:{name}".encode()))
    return rng.standard_normal(int(dur * SAMPLE_RATE)).astype(np.float32)


def _synth_legacy(name: str) -> np.ndarray:
    """The LEGACY bank — std renderer _synth_legacy (classic osu default
    sample character), trimmed to catch's sounds; soft duller, drum punchier."""
    set_name, _dash, base = name.partition("-")
    soft = set_name == "soft"
    drum = set_name == "drum"
    if base == "hitnormal":
        dur = 0.07
        t = _t(dur)
        n = _lg_noise(dur, name)
        if soft:
            x = (0.38 * np.sin(2 * math.pi * 330.0 * t) * np.exp(-65.0 * t)
                 + 0.30 * _lowpass(n, 24) * np.exp(-120.0 * t))
        elif drum:
            x = (0.70 * np.sin(2 * math.pi * 175.0 * t) * np.exp(-50.0 * t)
                 + 0.35 * _lowpass(_highpass(n, 96), 6) * np.exp(-160.0 * t))
        else:
            x = (0.55 * np.sin(2 * math.pi * 440.0 * t) * np.exp(-85.0 * t)
                 + 0.45 * _lowpass(_highpass(n, 48), 4) * np.exp(-170.0 * t))
    elif base == "hitwhistle":
        dur = 0.13 if drum else 0.22
        t = _t(dur)
        f_hi, f_lo = (2093.0, 1568.0)          # C7 -> G6 two-tone
        if soft:
            f_hi, f_lo = f_hi * 0.75, f_lo * 0.75
        freq = np.where(t < dur * 0.45, f_hi, f_lo).astype(np.float32)
        phase = 2.0 * math.pi * np.cumsum(freq) / SAMPLE_RATE
        amp = 0.30 if soft else 0.42
        x = amp * np.sin(phase) * np.exp(-9.0 * t)
        x += 0.3 * amp * np.sin(2.0 * phase) * np.exp(-14.0 * t)
    elif base == "hitfinish":
        dur = 0.45 if soft else (0.5 if drum else 0.7)
        t = _t(dur)
        n = _highpass(_lg_noise(dur, name), 10 if soft else 5)
        x = (0.35 if soft else 0.5) * n * np.exp(-4.5 * t)
        for i, f in enumerate((3135.0, 4699.0, 6271.0)):
            x += 0.07 * np.sin(2 * math.pi * f * t) * np.exp(-(6.0 + i) * t)
        if drum:
            x += 0.40 * np.sin(2 * math.pi * 100.0 * t) * np.exp(-18.0 * t)
    elif base == "hitclap":
        dur = 0.12
        t = _t(dur)
        n = _lowpass(_highpass(_lg_noise(dur, name), 64), 12 if soft else 6)
        env = np.exp(-60.0 * t)
        for tap_ms, amp in ((0.0, 0.5), (11.0, 0.7), (22.0, 1.0)):
            i0 = int(tap_ms / 1000.0 * SAMPLE_RATE)
            tap = np.zeros_like(env)
            tt = t[: len(t) - i0]
            tap[i0:] = np.exp(-220.0 * tt)
            env = np.maximum(env, amp * tap)
        x = (0.55 if soft else 0.8) * n * env
        if drum:
            x += 0.30 * np.sin(2 * math.pi * 160.0 * t) * np.exp(-70.0 * t)
    elif base == "slidertick":
        dur = 0.02
        t = _t(dur)
        f = 1200.0 if drum else (1500.0 if soft else 1900.0)
        x = (0.35 if soft else 0.45) * np.sin(2 * math.pi * f * t) \
            * np.exp(-320.0 * t)
    else:   # unknown name — quiet classic click, never silence
        t = _t(0.03)
        x = 0.35 * np.sin(2 * math.pi * 800.0 * t) * np.exp(-140.0 * t)
    return _stereo(x.astype(np.float32))


# --- sample bank --------------------------------------------------------------

class SampleBank:
    """(set, sound, index) -> PCM through BEATMAP(custom index) -> SKIN chain
    -> SYNTH default, with caching + per-source bookkeeping (std renderer's
    SampleBank shape). `skin_dirs` = [user skin, default skin] resolved dirs."""

    def __init__(self, beatmap_dir: Path | None, skin_dirs=(),
                 synth_style: str = "argon"):
        self.synth_style = synth_style
        self.skin_dirs = [Path(d) for d in skin_dirs if d]
        self._beatmap_files: dict[str, Path] = {}
        if beatmap_dir is not None:
            d = Path(beatmap_dir)
            if d.is_dir():
                for p in sorted(d.rglob("*")):
                    if p.is_file() and p.suffix.lower() in SAMPLE_EXTS:
                        self._beatmap_files.setdefault(p.name.lower(), p)
        self._pcm_cache: dict[Path, np.ndarray | None] = {}
        self._cache: dict[tuple, tuple[np.ndarray, str]] = {}
        self.source_counts: dict[str, int] = {"beatmap": 0, "skin": 0,
                                              "synth": 0}

    def _load(self, path: Path) -> np.ndarray | None:
        if path in self._pcm_cache:
            return self._pcm_cache[path]
        pcm = _decode_pcm(path)
        self._pcm_cache[path] = pcm
        return pcm

    def file(self, filename: str) -> np.ndarray | None:
        """A hitSample FILENAME override — the exact file in the beatmap dir
        (plays alone; caller falls back to the resolved default when
        missing/undecodable — the mania v2 rule)."""
        p = self._beatmap_files.get(filename.lower())
        if p is None:
            return None
        pcm = self._load(p)
        if pcm is not None:
            self.source_counts["beatmap"] += 1
        return pcm

    def get(self, set_id: int, sound: str, index: int = 0
            ) -> tuple[np.ndarray, str]:
        key = (set_id, sound, index)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        name = (f"{SET_NAMES[set_id]}-{sound}" if set_id in SET_NAMES
                else sound)
        pcm, src = self._resolve(name, index)
        self._cache[key] = (pcm, src)
        self.source_counts[src] += 1   # one per UNIQUE (set,sound,index)
        return pcm, src

    def _resolve(self, name: str, index: int) -> tuple[np.ndarray, str]:
        # 1. beatmap folder by custom index (index 0 = skin's defaults;
        #    index 1 = bare name, N>=2 = suffixed — stable's rule)
        if index > 0:
            base = name if index == 1 else f"{name}{index}"
            for ext in SAMPLE_EXTS:
                p = self._beatmap_files.get(f"{base}{ext}".lower())
                if p is not None:
                    pcm = self._load(p)
                    if pcm is not None:
                        return pcm, "beatmap"
        # 2. skin chain: user skin dir, then default skin dir
        for d in self.skin_dirs:
            for ext in SAMPLE_EXTS:
                p = d / f"{name}{ext}"
                if p.is_file():
                    pcm = self._load(p)
                    if pcm is not None:
                        return pcm, "skin"
        # 3. deterministic synthesized default
        return synth_sample(name, style=self.synth_style), "synth"

    def banana(self) -> tuple[np.ndarray, str]:
        """lazer BananaHitSampleInfo lookup: "catch-banana" then
        "metronomelow" through the skin chain, else the synth tick."""
        key = ("banana",)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = None
        for cand in ("catch-banana", "metronomelow"):
            for d in self.skin_dirs:
                for ext in SAMPLE_EXTS:
                    p = d / f"{cand}{ext}"
                    if p.is_file():
                        pcm = self._load(p)
                        if pcm is not None:
                            out = (pcm, "skin")
                            break
                if out:
                    break
            if out:
                break
        if out is None:
            out = (synth_sample("banana"), "synth")
        self._cache[key] = out
        self.source_counts[out[1]] += 1
        return out

    def nc_sample(self, base: str) -> np.ndarray | None:
        """ModNightcore sample (nightcore-kick/-clap/-hat/-finish): the SKIN
        chain first, then the bundled osu! DEFAULT as the FINAL fallback. A skin
        that ships a SILENT nightcore file plays (near-)silence (skin wins); a
        skin that OMITS it falls back to the default (osu! default-skin parity).
        No synth."""
        for d in list(self.skin_dirs) + [_DEFAULT_NC_DIR]:
            if not d or not Path(d).is_dir():
                continue
            for ext in SAMPLE_EXTS:
                p = Path(d) / f"{base}{ext}"
                if p.is_file():
                    pcm = self._load(p)
                    if pcm is not None:
                        return pcm
        return None


# --- beat overlay (metronome) -------------------------------------------------

_METRONOME_GAIN = 0.14        # beat-overlay click sits under the caught hits


def _layer_metronome_catch(track, bm, bank, start_ms, rate, n) -> int:
    """Beat-overlay metronome (site 'Beat overlay (metronome)' toggle): a clap
    on every beat + a finish on every downbeat, across the whole song, mixed
    into the caught-object hitsound track. Beats come from the beatmap's
    uninherited (red) timing points (assumed 4/4, matching the mania v2
    overlay). Placed in VIDEO time ((t_map - start_ms)/rate) so DT/NC/HT stay
    beat-aligned; the clap/finish PCM keeps natural pitch (stable behaviour).
    Mod-INDEPENDENT — a general metronome, not gated on the NC mod. Returns
    beats laid."""
    timing = getattr(bm, "timing", None)
    pts = getattr(timing, "points", None) if timing else None
    if not pts:
        return 0
    reds = [(t, v) for (t, v, uninh) in pts if uninh and v > 0]
    if not reds:
        return 0
    default_set = getattr(bm, "sample_set_default", 1) or 1
    rate = rate or 1.0
    horizon = start_ms + (n / SAMPLE_RATE * 1000.0) * rate
    laid = 0
    for i, (ptime, beat) in enumerate(reds):
        beat = max(60.0, float(beat))          # cap <60ms (>1000 BPM) sanity
        seg_end = reds[i + 1][0] if i + 1 < len(reds) else horizon
        seg_end = min(seg_end, horizon)
        k = 0
        t = float(ptime)
        while t < seg_end:
            downbeat = (k % 4 == 0)
            if hasattr(timing, "sample_info"):
                tp_set, tp_idx, _tp_vol = timing.sample_info(t)
            else:
                tp_set, tp_idx = 0, 0
            base = tp_set if tp_set in SET_NAMES else default_set
            pcm, _src = bank.get(base, "hitfinish" if downbeat else "hitclap",
                                 tp_idx)
            start = int(((t - start_ms) / rate) / 1000.0 * SAMPLE_RATE)
            if 0 <= start < n and len(pcm):
                end = min(n, start + len(pcm))
                track[start:end] += pcm[:end - start] * _METRONOME_GAIN
                laid += 1
            k += 1
            t = ptime + k * beat
    return laid


# --- ModNightcore beat overlay (NC-mod-gated, distinct from the metronome) -----

_NC_MOD_GAIN = 0.20        # nightcore-kick/clap/hat/finish drums


def _layer_nightcore_mod_catch(track, bm, bank, start_ms, rate, n,
                               *, play_hats: bool = True) -> int:
    """osu! ModNightcore beat overlay — the drum pattern osu! plays on each
    beat AUTOMATICALLY while the Nightcore mod is active. NOT the general
    metronome (_layer_metronome_catch) above; both can lay. Half-beat grid
    (BeatSyncedContainer Divisor=2): per 4/4 bar, kick on beats 1 & 3, clap on
    2 & 4, hat on the off-beats, plus a finish cymbal at the start of every 4th
    bar — from the SKIN's nightcore-kick/-clap/-hat/-finish samples (silent
    sample → silence; missing sample → skipped, no synth). Beats come from the
    beatmap's uninherited (red) timing points (assumed 4/4 — catch stores no
    signature); placed in VIDEO time ((t-start)/rate) so they ride the sped-up
    track. Hats gate on SliderTickRate%2==0. Returns samples laid.

    Port of osu.Game/Rulesets/Mods/ModNightcore.NightcoreBeatContainer."""
    timing = getattr(bm, "timing", None)
    pts = getattr(timing, "points", None) if timing else None
    if not pts:
        return 0
    reds = [(t, v) for (t, v, uninh) in pts if uninh and v > 0]
    if not reds:
        return 0
    samples = {name: bank.nc_sample(f"nightcore-{name}")
               for name in ("kick", "clap", "hat", "finish")}
    if not any(v is not None for v in samples.values()):
        return 0
    rate = rate or 1.0
    horizon = start_ms + (n / SAMPLE_RATE * 1000.0) * rate
    seg_len = 4 * 8                            # 4/4: beatsPerBar(4) * 2 * 4 bars
    laid = 0
    for i, (ptime, beat) in enumerate(reds):
        beat = max(60.0, float(beat))          # cap <60ms (>1000 BPM) sanity
        half = beat / 2.0
        seg_end = reds[i + 1][0] if i + 1 < len(reds) else horizon
        seg_end = min(seg_end, horizon)
        k = 0
        t = float(ptime)
        while t < seg_end:
            bseg = k % seg_len
            r = bseg % 4
            names = []
            if r == 0:
                names.append("kick")
            elif r == 2:
                names.append("clap")
            elif play_hats:
                names.append("hat")
            if bseg == 0:
                names.append("finish")
            start = int(((t - start_ms) / rate) / 1000.0 * SAMPLE_RATE)
            if 0 <= start < n:
                for name in names:
                    pcm = samples.get(name)
                    if pcm is not None and len(pcm):
                        end = min(n, start + len(pcm))
                        track[start:end] += pcm[:end - start] * _NC_MOD_GAIN
                        laid += 1
            k += 1
            t = ptime + k * half
    return laid


# --- track build --------------------------------------------------------------

def build_hitsound_track(objs, caught, bm, *, beatmap_dir: Path | None,
                         skin_dirs=(), out_wav: Path, start_ms: float,
                         rate: float = 1.0, duration_ms: float,
                         synth_style: str = "argon",
                         gain: float = DEFAULT_HIT_GAIN,
                         nightcore: bool = False,
                         nc_mod: bool = False,
                         hitsounds_on: bool = True) -> Path | None:
    """Mix every CAUGHT object's resolved samples at its catch time into one
    stereo WAV on the VIDEO time axis (wall = (t_map - start_ms)/rate — the
    same compression the video applies; sample PCM keeps natural pitch,
    stable behaviour). `objs`/`caught` are the sim's aligned object/verdict
    lists (post count-reconcile, post death-truncation). Returns the WAV
    path, or None when nothing was placed (encode then keeps the song-only
    chain)."""
    timing = getattr(bm, "timing", None)
    default_set = getattr(bm, "sample_set_default", 1) or 1
    bank = SampleBank(beatmap_dir, skin_dirs, synth_style)

    n = int(duration_ms / 1000.0 * SAMPLE_RATE) + SAMPLE_RATE // 10
    track = np.zeros((n, CHANNELS), dtype=np.float32)
    rate = rate or 1.0

    debug = []
    debug_on = bool(os.environ.get("R3D_CATCH_HITS_DEBUG"))

    def _peak(pcm) -> float:
        return float(np.max(np.abs(pcm))) if len(pcm) else 0.0
    placed = 0

    def _mix(wall_ms: float, pcm: np.ndarray, vol: float) -> None:
        nonlocal placed
        start = int(wall_ms / 1000.0 * SAMPLE_RATE)
        if start >= n or start < 0 or len(pcm) == 0:
            return
        end = min(n, start + len(pcm))
        track[start:end] += pcm[:end - start] * vol
        placed += 1

    _obj_iter = zip(objs, caught) if hitsounds_on else iter(())
    for obj, is_caught in _obj_iter:
        s = getattr(obj, "sample", None)
        if not is_caught or s is None:
            continue   # misses (and tiny droplets) are silent — lazer/stable
        t_map = float(obj.time_ms)
        wall = (t_map - start_ms) / rate
        if timing is not None and hasattr(timing, "sample_info"):
            tp_set, tp_idx, tp_vol = timing.sample_info(t_map)
        else:
            tp_set, tp_idx, tp_vol = 0, 0, 100
        vol = (s.volume or tp_vol or 100) / 100.0
        vol = max(VOLUME_FLOOR, min(1.0, vol)) * gain
        idx = s.index or tp_idx
        base = (s.normal_set if s.normal_set in SET_NAMES
                else (tp_set if tp_set in SET_NAMES else default_set))
        add = s.addition_set if s.addition_set in SET_NAMES else base

        if s.kind == "banana":
            pcm, src = bank.banana()
            _mix(wall, pcm, 1.0 * gain)   # BananaHitSampleInfo volume 100
            if debug_on:
                debug.append({"t_map": t_map, "wall_ms": wall,
                              "sound": "banana", "src": src,
                              "peak": _peak(pcm) * gain})
            continue
        if s.kind == "tick":
            pcm, src = bank.get(base, "slidertick", idx)
            _mix(wall, pcm, vol)
            if debug_on:
                debug.append({"t_map": t_map, "wall_ms": wall,
                              "sound": "slidertick", "set": base,
                              "index": idx, "vol": vol, "src": src,
                              "peak": _peak(pcm) * vol})
            continue
        # kind "hit": custom filename plays ALONE; missing file falls through
        if s.filename:
            pcm = bank.file(s.filename)
            if pcm is not None:
                _mix(wall, pcm, vol)
                if debug_on:
                    debug.append({"t_map": t_map, "wall_ms": wall,
                                  "sound": f"file:{s.filename}", "vol": vol,
                                  "src": "beatmap", "peak": _peak(pcm) * vol})
                continue
        sounds = [("hitnormal", base)]          # LayeredHitSounds default: always
        for bit, sname in ADDITIONS:
            if s.bits & bit:
                sounds.append((sname, add))
        for sname, sset in sounds:
            pcm, src = bank.get(sset, sname, idx)
            _mix(wall, pcm, vol)
            if debug_on:
                debug.append({"t_map": t_map, "wall_ms": wall, "sound": sname,
                              "set": sset, "index": idx, "vol": vol,
                              "src": src, "peak": _peak(pcm) * vol})

    # The general metronome is SUPPRESSED while NC is active (the NC drum
    # overlay plays instead — osu! never plays both on one render).
    nc_beats = 0
    if nightcore and not nc_mod:
        nc_beats = _layer_metronome_catch(track, bm, bank, start_ms, rate, n)

    # ModNightcore beat overlay — AUTOMATIC when the NC mod is active,
    # independent of the `nightcore` metronome toggle above (both may lay).
    nc_mod_beats = 0
    if nc_mod:
        play_hats = (int(round(getattr(bm, "tick_rate", 1.0) or 1.0)) % 2 == 0)
        nc_mod_beats = _layer_nightcore_mod_catch(track, bm, bank, start_ms,
                                                  rate, n, play_hats=play_hats)

    if placed == 0 and nc_beats == 0 and nc_mod_beats == 0:
        return None
    np.clip(track, -1.0, 1.0, out=track)
    out_wav = Path(out_wav)
    _write_wav_f32(out_wav, track)
    if debug_on:
        Path(str(out_wav) + ".events.json").write_text(json.dumps(debug))
    sc = bank.source_counts
    print(f"[catch-renderer] hitsounds: {placed} samples placed "
          f"(sources beatmap={sc['beatmap']} skin={sc['skin']} "
          f"synth={sc['synth']}) + {nc_beats} metronome beats "
          f"+ {nc_mod_beats} NC-mod beats -> {out_wav.name}",
          file=sys.stderr, flush=True)
    return out_wav


def _write_wav_f32(path: Path, buf: np.ndarray) -> Path:
    """float32 WAV writer (no soundfile in this venv — the std renderer's
    struct-built RIFF, ffmpeg reads it natively)."""
    import struct
    data = buf.astype("<f4").tobytes()
    with open(path, "wb") as fh:
        byte_rate = SAMPLE_RATE * CHANNELS * 4
        fh.write(b"RIFF")
        fh.write(struct.pack("<I", 36 + len(data)))
        fh.write(b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 3, CHANNELS, SAMPLE_RATE,
                             byte_rate, CHANNELS * 4, 32))
        fh.write(b"data")
        fh.write(struct.pack("<I", len(data)))
        fh.write(data)
    return path
