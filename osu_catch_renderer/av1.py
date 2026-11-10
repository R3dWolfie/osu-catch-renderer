"""Hardware AV1 master-video encoding for the catch renderer.

Shared by BOTH render paths (render/render.py and _metal/render/render.py),
like preview_hw.py, so the two engines cannot drift on the AV1 recipe or the
hardware probe. Implements Leon's AV1 migration design
(CrossComms .../renderer-core/av1-migration-2026-10-23/migration-design.md):

  * AV1 is the MASTER + site-only-preview codec. The Discord embed/compact
    previews STAY H.264 (that lives in render.py and is untouched) — Red
    established that Discord cannot play inline AV1, so the small Discord
    sidecar remains the explicit compatibility exception.
  * FOUR hardware backends with BACKEND-SPECIFIC controls — NOT a mechanical
    `h264_* -> av1_*` rename (NVENC/VAAPI/AMF/QSV have different knobs):
      - av1_vaapi : `-rc_mode CQP -global_quality N` (NOT `-qp`, which the AMD
                    VAAPI driver IGNORES — Leon's test-kit finding: -qp 40/90/160
                    produced identical files, only -global_quality changed output).
      - av1_nvenc : `-rc vbr -cq N -b:v 0` with a maxrate cap (CQ-VBR, mirrors
                    the engine's H.264 NVENC #87 size policy).
      - av1_qsv   : `-global_quality N` (ICQ), nv12.
      - av1_amf   : `-rc qvbr -qvbr_quality_level N` (AMD AMF, Windows).
  * Output is 8-bit 4:2:0 (nv12 for vaapi/qsv, yuv420p for nvenc/amf) in AV1
    Main profile, `-movflags +faststart`, a ~2 s GOP.
  * A REAL hardware probe: a 320x240 AV1 test-encode on the SELECTED device
    with the PRODUCTION backend recipe and a strict timeout, whose output is
    confirmed to be ACTUAL AV1 (ffprobe `codec_name=av1`). Compiled-in encoders
    and GPU marketing names are NOT sufficient (design).
  * FAIL-CLOSED: an AV1 master cannot fall back to H.264 / CPU / libsvtav1 /
    libaom / librav1e. When no hardware AV1 encoder is usable the selection
    raises AV1Unavailable and the render fails / requeues; it NEVER silently
    ships an H.264 master.

Only hardware encoders appear here on purpose. libsvtav1 / libaom / librav1e
are deliberately excluded so no code path can accidentally use a CPU AV1
encoder for a production master.

PROVISIONAL QUALITY DEFAULTS. The per-backend quality numbers below are
starting points (the VAAPI default follows the test-kit's quality-good
global_quality=90), overridable by env, NOT Leon-approved production recipes.
Final recipes come from Red's measurements on real hardware (see tests-for-red).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The four candidate HARDWARE AV1 backends. No CPU encoder is listed: a
# production AV1 master must be hardware-encoded (fail-closed, no CPU fallback).
AV1_ENCODERS: tuple[str, ...] = ("av1_vaapi", "av1_nvenc", "av1_amf", "av1_qsv")

# Auto-select-a-hardware-AV1-backend selectors (`--encoder av1`, R3D_ENCODER=av1).
# Bare "auto" is left as the engine's existing H.264 auto on purpose: flipping
# the fleet default to AV1 is a gated rollout decision (coordinator capability
# enforcement + signed bundle + canaries), carried per-job as required_video_codec
# =av1, NOT an engine default. AV1 is therefore opt-in via an explicit av1_*
# encoder or this "av1" auto selector.
AV1_AUTO: tuple[str, ...] = ("av1", "auto-av1", "av1_auto")

# Candidate order for the "av1" auto selector: NVENC first (R3D renders on
# NVIDIA by policy), then VAAPI (AMD/Intel on Linux — the 7900 XTX), then QSV
# and AMF. Each is still gated behind the REAL probe below.
_AUTO_ORDER: tuple[str, ...] = ("av1_nvenc", "av1_vaapi", "av1_qsv", "av1_amf")

_DEFAULT_VAAPI_DEVICE = "/dev/dri/renderD128"


class AV1Unavailable(RuntimeError):
    """No usable hardware AV1 encoder for the requested selection.

    Raised instead of falling back to H.264/CPU, so the caller fails the AV1
    master cleanly (and the node requeues on another eligible node)."""


def is_explicit(encoder: str) -> bool:
    """True for an explicit single backend (av1_vaapi / av1_nvenc / ...)."""
    return encoder in AV1_ENCODERS


def is_auto(encoder: str) -> bool:
    """True for the `av1` auto-select-a-backend selector."""
    return encoder in AV1_AUTO


def is_request(encoder: str) -> bool:
    """True for ANY AV1 request (explicit backend or the `av1` auto selector)."""
    return is_explicit(encoder) or is_auto(encoder)


def needs_vaapi_upload(encoder: str) -> bool:
    """True for a backend that needs the `format=nv12,hwupload` VAAPI filter and
    a `-vaapi_device` (av1_vaapi). NVENC/AMF take system yuv420p; QSV takes
    system nv12 and uploads internally — none of them use the VAAPI filtergraph."""
    return encoder == "av1_vaapi"


def _env(name: str, default: str) -> str:
    v = (os.environ.get(name) or "").strip()
    return v or default


def _target_bps(w: int, h: int, fps: float) -> int:
    """Resolution-scaled bitrate ladder (same formula as render.py's
    nvenc_target_bps — duplicated here to keep this module import-free of the
    engine). Used only as the NVENC CQ-VBR maxrate CAP."""
    ref = 1280.0 * 720.0 * 30.0
    target = 4_000_000.0 * ((float(w) * float(h) * float(fps)) / ref) ** 0.70
    return int(min(16_000_000.0, max(2_500_000.0, target)))


def master_video_args(encoder: str, *, w: int, h: int, fps: float,
                      video_bitrate: "int | None" = None) -> list[str]:
    """Video-codec arguments for the AV1 master branch.

    Returns the `-vf`/`-c:v`/rate-control tokens (and, for VAAPI, the
    `format=nv12,hwupload` filter) but NOT `-movflags +faststart` (the caller
    appends that, as it does for the H.264 encoders) and NOT any audio.

    Backend-specific by design — see the module docstring. 8-bit 4:2:0, AV1
    Main profile, a ~2 s GOP (`R3D_AV1_GOP` overrides, default 2*fps)."""
    gop = _env("R3D_AV1_GOP", str(max(1, int(round(float(fps) * 2)))))

    if encoder == "av1_vaapi":
        # VAAPI: frames uploaded to a GPU surface as nv12 (8-bit 4:2:0). CQP
        # quality via -global_quality (NOT -qp — ignored by the AMD driver).
        base = ["-vf", "format=nv12,hwupload", "-c:v", "av1_vaapi"]
        tail = ["-profile:v", "main", "-g", gop]
        if video_bitrate:
            vb = int(video_bitrate)
            return base + ["-rc_mode", "VBR", "-b:v", str(vb),
                           "-maxrate", str(int(vb * 1.5))] + tail
        return base + ["-rc_mode", "CQP",
                       "-global_quality", _env("R3D_AV1_VAAPI_GQ", "90")] + tail

    if encoder == "av1_nvenc":
        # NVENC AV1 (Ada / RTX 40+). System yuv420p; CQ-VBR with a maxrate cap,
        # mirroring the engine's H.264 NVENC #87 size policy. -cq is 0..63.
        v = ["-c:v", "av1_nvenc", "-pix_fmt", "yuv420p",
             "-preset", _env("R3D_AV1_NVENC_PRESET", "p4")]
        if video_bitrate:
            vb = int(video_bitrate)
            return v + ["-rc", "vbr", "-b:v", str(vb),
                        "-maxrate", str(int(vb * 1.5)), "-bufsize", str(vb * 2),
                        "-g", gop]
        cap = _target_bps(w, h, fps)
        return v + ["-rc", "vbr", "-cq", _env("R3D_AV1_NVENC_CQ", "32"),
                    "-b:v", "0", "-maxrate", str(cap), "-bufsize", str(cap * 2),
                    "-g", gop]

    if encoder == "av1_qsv":
        # Intel QSV AV1 (Arc). System nv12 (uploaded internally by the QSV
        # encoder). ICQ-style quality via -global_quality. UNTESTED — needs
        # Intel Arc validation (design: enable the profile only after that).
        v = ["-c:v", "av1_qsv", "-pix_fmt", "nv12"]
        if video_bitrate:
            vb = int(video_bitrate)
            return v + ["-b:v", str(vb), "-maxrate", str(int(vb * 1.5)), "-g", gop]
        return v + ["-global_quality", _env("R3D_AV1_QSV_GQ", "90"), "-g", gop]

    if encoder == "av1_amf":
        # AMD AMF AV1 (Windows RDNA3). System yuv420p. QVBR quality level
        # (0..51). UNTESTED — needs Windows AMD validation.
        v = ["-c:v", "av1_amf", "-pix_fmt", "yuv420p", "-usage", "high_quality"]
        if video_bitrate:
            vb = int(video_bitrate)
            return v + ["-b:v", str(vb), "-maxrate", str(int(vb * 1.5)), "-g", gop]
        return v + ["-rc", "qvbr",
                    "-qvbr_quality_level", _env("R3D_AV1_AMF_QVBR", "30"), "-g", gop]

    raise AV1Unavailable(f"not a known AV1 encoder: {encoder!r}")


# ---- real hardware probe -----------------------------------------------------
# Results are cached per (encoder, device, ffmpeg identity) FOR THE PROCESS only.
# Cross-process / cross-render capability caching keyed on GPU + driver + ffmpeg
# + bundle + profile is the contributor/coordinator layer's job (out of scope of
# the engine); the engine re-probes in a fresh process, which is cheap and never
# stale.
_probe_cache: "dict[tuple, tuple[bool, str]]" = {}


def _ffmpeg_identity(ffmpeg: str) -> tuple:
    try:
        st = os.stat(os.path.realpath(ffmpeg))
        return (os.path.realpath(ffmpeg), st.st_size, st.st_mtime_ns)
    except OSError:
        return (ffmpeg, 0, 0)


def _compiled_in(ffmpeg: str, encoder: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        return False
    return any(len(parts) >= 2 and parts[1] == encoder
               for parts in (line.split() for line in out.splitlines()))


def _ffprobe_codec(path: str, ffprobe: str) -> str:
    """codec_name of the first video stream, or '' on any trouble."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "json", path],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return ""
        streams = json.loads(out.stdout or "{}").get("streams") or []
        return (streams[0].get("codec_name") or "") if streams else ""
    except Exception:  # noqa: BLE001
        return ""


def probe(encoder: str, device: "str | None" = None, *,
          ffmpeg: "str | None" = None, timeout: float = 20.0,
          use_cache: bool = True) -> tuple[bool, str]:
    """Real hardware AV1 probe for ONE backend on the SELECTED device.

    Test-encodes a 320x240 clip with the PRODUCTION backend recipe
    (master_video_args) to a real MP4, then confirms with ffprobe that the
    output is ACTUALLY AV1 (codec_name=av1). Returns (usable, reason); never
    raises. A compiled-in encoder or a matching GPU name is NOT enough — the
    device must really produce an AV1 bitstream.

    This is the gate the design requires before a node may claim AV1 work."""
    if encoder not in AV1_ENCODERS:
        return False, f"{encoder!r} is not a hardware AV1 encoder"
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found on PATH"
    if needs_vaapi_upload(encoder):
        device = device or _DEFAULT_VAAPI_DEVICE
    key = (encoder, device, _ffmpeg_identity(ffmpeg))
    if use_cache and key in _probe_cache:
        return _probe_cache[key]

    result: tuple[bool, str]
    if not _compiled_in(ffmpeg, encoder):
        result = (False, f"{encoder} not compiled into ffmpeg")
    elif needs_vaapi_upload(encoder) and not Path(device).exists():
        result = (False, f"VAAPI device {device} does not exist")
    else:
        ffprobe = shutil.which("ffprobe") or (
            ffmpeg[:-6] + "ffprobe" if ffmpeg.endswith("ffmpeg") else "ffprobe")
        out_fd, out_path = tempfile.mkstemp(prefix="r3d_av1_probe_", suffix=".mp4")
        os.close(out_fd)
        try:
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
            if needs_vaapi_upload(encoder):
                cmd += ["-vaapi_device", device]
            cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x240",
                    "-r", "5", "-i", "pipe:0", "-frames:v", "5", "-an"]
            cmd += master_video_args(encoder, w=320, h=240, fps=5)
            cmd += ["-movflags", "+faststart", out_path]
            try:
                proc = subprocess.run(cmd, input=bytes(320 * 240 * 3 * 5),
                                      capture_output=True, timeout=timeout,
                                      check=False)
            except subprocess.TimeoutExpired:
                result = (False, "AV1 encoder startup probe timed out")
            except OSError as exc:
                result = (False, str(exc))
            else:
                if proc.returncode != 0:
                    err = proc.stderr.decode(errors="replace").strip().splitlines()
                    result = (False, err[0][:500] if err
                              else f"ffmpeg exit code {proc.returncode}")
                else:
                    codec = _ffprobe_codec(out_path, ffprobe)
                    if codec == "av1":
                        result = (True, "")
                    elif not codec:
                        # encode succeeded but the output is unreadable/empty:
                        # treat as unusable rather than assume AV1.
                        result = (False, "probe output has no readable video stream")
                    else:
                        result = (False, f"produced {codec!r}, not av1")
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    if use_cache:
        _probe_cache[key] = result
    return result


def resolve_encoder(encoder: str, device: "str | None" = None, *,
                    ffmpeg: "str | None" = None) -> tuple[str, "str | None"]:
    """Resolve an AV1 request to a PROVEN hardware backend + device.

    For an explicit av1_* encoder: confirm it really produces AV1 on the
    selected device, else raise AV1Unavailable (fail-closed).

    For the `av1` auto selector: probe the candidate backends in order and
    return the first that really produces AV1, else raise AV1Unavailable listing
    why each failed.

    Never returns a non-AV1 encoder and never falls back to H.264/CPU."""
    if is_explicit(encoder):
        dev = device
        if needs_vaapi_upload(encoder):
            dev = dev or _DEFAULT_VAAPI_DEVICE
        ok, reason = probe(encoder, dev, ffmpeg=ffmpeg)
        if ok:
            print(f"av1: {encoder} confirmed (hardware AV1 on "
                  f"{dev or 'default device'})", file=sys.stderr, flush=True)
            return encoder, dev
        raise AV1Unavailable(
            f"AV1 master requested ({encoder}) but hardware AV1 encode is "
            f"unavailable on {dev or 'the selected device'}: {reason}. "
            f"Fail-closed: no H.264/CPU fallback for an AV1 master.")

    if is_auto(encoder):
        failures: list[str] = []
        for candidate in _AUTO_ORDER:
            dev = device
            if needs_vaapi_upload(candidate):
                dev = dev or _DEFAULT_VAAPI_DEVICE
                if not Path(dev).exists():
                    failures.append(f"{candidate}: VAAPI device {dev} absent")
                    continue
            ok, reason = probe(candidate, dev, ffmpeg=ffmpeg)
            if ok:
                print(f"av1: auto selected {candidate} (hardware AV1 on "
                      f"{dev or 'default device'})", file=sys.stderr, flush=True)
                return candidate, dev
            failures.append(f"{candidate}: {reason}")
            print(f"av1: {candidate} unusable: {reason}",
                  file=sys.stderr, flush=True)
        raise AV1Unavailable(
            "AV1 master requested (auto) but no usable hardware AV1 encoder: "
            + "; ".join(failures)
            + ". Fail-closed: no H.264/CPU fallback for an AV1 master.")

    raise AV1Unavailable(f"not an AV1 encoder request: {encoder!r}")
