"""Shared loudnorm PCM cache — cross-engine, box-local (catch engine).

The single-pass loudnorm pass (``LOUDNORM``) is deterministic in
(source bytes, playback rate, pitch mode, param string) yet reruns the full
ffmpeg normalise on every render of the same track (~2-3 s for a typical song).
Memoise its f32le PCM output on the fast local SSD so a repeat render — or
another in-house engine rendering the same track — skips the pass.

This mirrors, byte-for-byte, the shared cache the sibling engines use
(osu-mania-renderer-v2 render/loudnorm_cache.py, osu-std record/audio.py):
identical directory, key recipe, artifact format (``{key}.f32le`` = raw
little-endian float32, 48 kHz stereo) and kill-switch, so a track normalised by
one mode is reused by another. Any divergence in the four contract points
below (dir, key, format, kill-switch) silently breaks that interop — keep them
in lock-step with the siblings.

Best-effort throughout: a missing / truncated / unreadable cache entry, or ANY
build failure, returns ``None`` so the caller falls back to the inline
(fused-loudnorm) encode path and the render still succeeds.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# The loudnorm param string. MUST stay byte-identical to the sibling engines
# (mania encode.py LOUDNORM / osu-std _LOUDNORM_FILTER) and to the literal used
# in render._audio_filter / _hitsound_filter_complex, or the shared cache key
# diverges and cross-engine reuse silently stops.
LOUDNORM = "loudnorm=I=-18:TP=-1.5:LRA=11"

# --- contract shared with the sibling engines --------------------------------
DEFAULT_CACHE_DIR = "/data/r3d/loudnorm-cache"
LOUDNORM_CACHE_SR = 48000          # raw artifact geometry (Hz)
LOUDNORM_CACHE_CH = 2              # stereo
CACHE_EXT = "f32le"                # raw little-endian float32, 48 kHz stereo
_STRIDE = LOUDNORM_CACHE_CH * 4    # bytes per PCM frame (float32 * channels)
_CHUNK = 1 << 20                   # 1 MiB source-hash read chunk


# --- loudness by ONE fixed gain (default OFF) ---------------------------------
# R3D_CATCH_FIXED_GAIN=1, or the node-wide R3D_FIXED_GAIN=1 (every engine reads
# that one; the engine's own switch wins when both are set).
# The one-pass loudnorm filter is the whole cost of this cache's build: 17.5 s
# for a 482 s song that takes 0.5 s to decode (it works at 192 kHz, on one
# thread). With the switch the build instead measures the integrated loudness
# (ffmpeg `ebur128`) at the point loudnorm stood, and applies ONE gain for the
# whole track to the same -18 LUFS, held back where it would put a sample above
# the same -1.5 dB: under a second. NOT the same sound: one-pass loudnorm moves
# its gain as the track goes (it lifts quiet passages); a fixed gain leaves the
# track's own dynamics alone. Hence a switch.
# The artifact is the same kind (48 kHz stereo f32le) under its own key, and
# the recipe below is, number for number, the std engine's
# (osu_std_renderer/record/audio.py), so these entries are shared too.
TARGET_LUFS = -18.0
PEAK_CEILING_DB = -1.5
EBUR128 = "ebur128=framelog=quiet"
FIXED_GAIN_PARAM = f"fixedgain:I={TARGET_LUFS:g}:P={PEAK_CEILING_DB:g}"
_NOTHING_LUFS = -70.0      # what ebur128 reports when nothing passed its gate


def fixed_gain_on() -> bool:
    def flag(name: str) -> "bool | None":
        v = os.environ.get(name)
        return None if v is None else \
            v.strip().lower() not in ("", "0", "false", "no", "off")
    own = flag("R3D_CATCH_FIXED_GAIN")
    return bool(flag("R3D_FIXED_GAIN")) if own is None else own


PIN_192K = "aformat=sample_rates=192000"


def fixed_gain_chain(rate_filters: "list[str]") -> str:
    """The `-af` chain of the measuring build: the rate filters, then the
    measurement where loudnorm stood.

    loudnorm only takes 192 kHz, and with it in the chain ffmpeg resamples to
    192 kHz BEFORE an `atempo` (DT/HT), so the stock time-stretch runs at
    192 kHz. Stretched at the file's own rate the song is a different (equally
    valid) stretch: another length by a few ms and not sample-aligned with
    stock's. Pinning the same rate at the same place keeps the stretch exactly
    stock's, so the only thing the switch changes is the loudness. Measured:
    with the pin the 1.5x and 0.75x songs match stock's length to the sample
    and correlate 0.999+ at lag 0; without it 0.2-0.35. Chains without atempo
    (NoMod, NC) come out the same either way, so they skip the pin and its
    cost. Same rule as the std engine, so the entries stay shared."""
    parts = list(rate_filters)
    if any("atempo" in f for f in parts):
        parts.append(PIN_192K)
    return ",".join(parts + [EBUR128])


def parse_integrated_lufs(stderr_text: str) -> "float | None":
    """The integrated loudness out of ffmpeg's `ebur128` summary; None when
    there is no summary to read. Silence reads -70.0."""
    m = re.findall(r"^\s*I:\s+(-?\d+(?:\.\d+)?) LUFS", stderr_text, re.M)
    return float(m[-1]) if m else None


def fixed_gain_db(integrated_lufs: "float | None", peak: float) -> float:
    """dB for the whole track: up or down to TARGET_LUFS, never so far up that
    the loudest sample (`peak`, linear) passes PEAK_CEILING_DB. Silence is left
    as it is."""
    if integrated_lufs is None or integrated_lufs <= _NOTHING_LUFS:
        return 0.0
    gain = TARGET_LUFS - integrated_lufs
    if peak > 0.0:
        gain = min(gain, PEAK_CEILING_DB - 20.0 * math.log10(peak))
    return gain


def cache_disabled() -> bool:
    """Kill-switch, matching the sibling engines: ``R3D_NO_LOUDNORM_CACHE``.

    Default ON (unset / 0 / false / no / off = enabled); any other value
    disables the whole path — one env var kills the cache across every engine."""
    return os.environ.get("R3D_NO_LOUDNORM_CACHE", "").strip().lower() \
        not in ("", "0", "false", "no", "off")


def cache_dir() -> Path:
    return Path(os.environ.get("R3D_LOUDNORM_CACHE_DIR", DEFAULT_CACHE_DIR))


def compute_key(source: Path, rate: float, pitch: bool,
                param: str = LOUDNORM) -> str:
    """Stable hash of everything determining the loudnorm OUTPUT — sha256 of the
    SOURCE audio bytes + playback rate + pitch mode + the exact loudnorm param
    string. Byte-for-byte the sibling engines recipe (double sha256, ``rate``
    via ``repr(float)``) so the ``{key}.f32le`` artifacts are shared.

    Raises OSError if the source cannot be read (caller then runs uncached)."""
    h = hashlib.sha256()
    with open(source, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    material = "\n".join((
        f"src={h.hexdigest()}",
        f"rate={float(rate)!r}",
        f"pitch={1 if pitch else 0}",
        f"param={param}",
    )).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _rate_filters(rate: float, pitch: bool) -> list[str]:
    """The rate/pitch ffmpeg filters, IDENTICAL to the sibling engines for
    NoMod/DT/HT (so the artifact is byte-shared). NC uses the same resample
    chain as mania v2; the std engine differs, so a cross-engine NC hit is
    perceptually-equal but not byte-shared (documented shared-cache caveat)."""
    if rate == 1.0:
        return []
    if pitch:  # NC — resample-based pitch-up (speed AND pitch rise together)
        return ["aresample=44100", f"asetrate=44100*{rate}", "aresample=44100"]
    return [f"atempo={rate}"]  # DT / HT — pitch-preserving tempo shift


def _valid(path: Path) -> bool:
    """Usable iff a whole number of PCM frames and non-empty — treats an empty
    or truncated file as a miss (matches the siblings load-side stride check)."""
    try:
        sz = path.stat().st_size
    except OSError:
        return False
    return sz > 0 and (sz % _STRIDE) == 0


def _build(source: Path, rate: float, pitch: bool, target: Path) -> bool:
    """Run the loudnorm pre-pass -> atomically publish raw f32le PCM to
    ``target``. Returns True on success. The ffmpeg command matches the sibling
    engines decode (``-af <rate>,loudnorm -f f32le -ar 48000 -ac 2``) so the
    bytes are cross-engine-compatible for NoMod/DT/HT."""
    af = ",".join(_rate_filters(rate, pitch) + [LOUDNORM])
    tmp: str | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(target.parent), prefix=".tmp-", suffix="." + CACHE_EXT)
        os.close(fd)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(source),
            "-vn", "-af", af,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(LOUDNORM_CACHE_SR), "-ac", str(LOUDNORM_CACHE_CH),
            "-y", tmp,
        ]
        r = subprocess.run(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0 or not _valid(Path(tmp)):
            raise RuntimeError(
                (r.stderr or b"").decode(errors="ignore")[-400:] or "bad pcm")
        os.replace(tmp, target)  # atomic on the same filesystem
        return True
    except Exception:  # noqa: BLE001 — cache build is best-effort
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def _build_fixed(source: Path, rate: float, pitch: bool, target: Path) -> bool:
    """The fixed-gain build: decode with the loudness measured in the same
    pass, apply the one gain, publish atomically. False when ffmpeg failed or
    printed no summary that can be read (the caller then builds the stock
    artifact instead)."""
    af = fixed_gain_chain(_rate_filters(rate, pitch))
    tmp: str | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(target.parent), prefix=".tmp-", suffix="." + CACHE_EXT)
        os.close(fd)
        # `info` is the level ebur128 prints its summary at
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
             "-i", str(source), "-vn", "-af", af,
             "-f", "f32le", "-acodec", "pcm_f32le",
             "-ar", str(LOUDNORM_CACHE_SR), "-ac", str(LOUDNORM_CACHE_CH),
             "-y", tmp],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
        lufs = parse_integrated_lufs((r.stderr or b"").decode(errors="replace"))
        if r.returncode != 0 or not _valid(Path(tmp)) or lufs is None:
            raise RuntimeError("no measurement")
        pcm = np.fromfile(tmp, dtype="<f4")
        gain = fixed_gain_db(lufs, float(np.abs(pcm).max()) if pcm.size else 0.0)
        pcm *= np.float32(10.0 ** (gain / 20.0))
        pcm.tofile(tmp)
        os.replace(tmp, target)  # atomic on the same filesystem
        print("[catch] song loudness: "
              + ("silent, left as it is" if lufs <= _NOTHING_LUFS else
                 f"{lufs:.1f} LUFS, one gain of {gain:+.1f} dB"),
              file=sys.stderr, flush=True)
        return True
    except Exception:  # noqa: BLE001 — cache build is best-effort
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def get_or_build_normalized(source, *, rate: float, pitch: bool,
                            fixed_gain: "bool | None" = None):
    """Return a cached, loudness-normalised raw-f32le PCM ``Path`` for
    ``source`` at the given rate/pitch — building it (and populating the shared
    cache) on a miss. Returns ``None`` when the cache is disabled or on ANY
    failure, so the caller falls back to the inline (fused) loudnorm path.
    Never raises. ``fixed_gain`` None = follow the switch (see fixed_gain_on)."""
    try:
        if source is None or cache_disabled():
            return None
        source = Path(source)
        if fixed_gain_on() if fixed_gain is None else fixed_gain:
            # one measured gain in place of the loudnorm pass; if that cannot
            # be built here, the stock artifact below
            target = cache_dir() / (
                f"{compute_key(source, rate, pitch, FIXED_GAIN_PARAM)}.{CACHE_EXT}")
            if _valid(target) or _build_fixed(source, rate, pitch, target):
                return target
        target = cache_dir() / f"{compute_key(source, rate, pitch)}.{CACHE_EXT}"
        if _valid(target):
            return target  # HIT
        if _build(source, rate, pitch, target):
            return target  # MISS -> built
        return None
    except Exception:  # noqa: BLE001 — must never break a render
        return None
