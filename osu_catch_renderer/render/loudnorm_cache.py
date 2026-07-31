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
import os
import subprocess
import tempfile
from pathlib import Path

# The loudnorm param string. MUST stay byte-identical to the sibling engines
# (mania encode.py LOUDNORM / osu-std _LOUDNORM_FILTER) and to the literal used
# in render._audio_filter / _hitsound_filter_complex, or the shared cache key
# diverges and cross-engine reuse silently stops.
LOUDNORM = "loudnorm=I=-10:TP=-1.5:LRA=11"

# --- contract shared with the sibling engines --------------------------------
DEFAULT_CACHE_DIR = "/data/r3d/loudnorm-cache"
LOUDNORM_CACHE_SR = 48000          # raw artifact geometry (Hz)
LOUDNORM_CACHE_CH = 2              # stereo
CACHE_EXT = "f32le"                # raw little-endian float32, 48 kHz stereo
_STRIDE = LOUDNORM_CACHE_CH * 4    # bytes per PCM frame (float32 * channels)
_CHUNK = 1 << 20                   # 1 MiB source-hash read chunk


def cache_disabled() -> bool:
    """Kill-switch, matching the sibling engines: ``R3D_NO_LOUDNORM_CACHE``.

    Default ON (unset / 0 / false / no / off = enabled); any other value
    disables the whole path — one env var kills the cache across every engine."""
    return os.environ.get("R3D_NO_LOUDNORM_CACHE", "").strip().lower() \
        not in ("", "0", "false", "no", "off")


def cache_dir() -> Path:
    return Path(os.environ.get("R3D_LOUDNORM_CACHE_DIR", DEFAULT_CACHE_DIR))


def compute_key(source: Path, rate: float, pitch: bool) -> str:
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
        f"param={LOUDNORM}",
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


def get_or_build_normalized(source, *, rate: float, pitch: bool):
    """Return a cached, loudness-normalised raw-f32le PCM ``Path`` for
    ``source`` at the given rate/pitch — building it (and populating the shared
    cache) on a miss. Returns ``None`` when the cache is disabled or on ANY
    failure, so the caller falls back to the inline (fused) loudnorm path.
    Never raises."""
    try:
        if source is None or cache_disabled():
            return None
        source = Path(source)
        target = cache_dir() / f"{compute_key(source, rate, pitch)}.{CACHE_EXT}"
        if _valid(target):
            return target  # HIT
        if _build(source, rate, pitch, target):
            return target  # MISS -> built
        return None
    except Exception:  # noqa: BLE001 — must never break a render
        return None
