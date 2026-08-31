"""osu!catch fail (death) post-pass — a faithful port of osu!lazer's real
fail animation (``FailAnimationContainer``).

On a FAILED replay the sim is already truncated at ``death_ms`` (no fruit spawn
past death, catcher frozen). Instead of ending on a hard cut, this module plays
osu!'s fail animation over the final ``FAIL_FADE_MS`` of gameplay ENDING at
``death_ms`` (lazer plays it FORWARD from the fail instant; we can't render past
death, so we build the same wind-down UP TO death and hold the dead frame for
the frozen tail + results).

osu!lazer ``FailAnimationContainer`` (osu.Game/Screens/Play), matched here:

    private const float duration = 2500;                       # FAIL_FADE_MS
    Content.ScaleTo(0.85f, duration, Easing.OutQuart);         # scale-down
    Content.RotateTo(1,    duration, Easing.OutQuart);         # +1 deg tilt
    Content.FadeColour(Color4.Gray, duration);                 # darken->gray (linear)
    redFlashLayer  = Color4.Red.Opacity(0.6f), additive        # red wash
    this.TransformBindableTo(trackFreq, 0, duration);          # freq 1->0 (audio)
    failLowPassFilter.CutoffTo(300, duration, Easing.OutCubic);# LP muffle -> 300 Hz
    volumeAdjustment = 0.5                                      # track volume x0.5

VISUAL (this module, ``apply_death``): the composited frame (playfield AND HUD)
is affine-transformed — rotate +1 deg, scale to 0.85, and a small downward
FALL (our addition; lazer masks + darkens rather than translating, but Red
wants the classic "playfield drops away") — all eased OutQuart to their target
AT death, over black; then colour-graded (darken toward gray, linear, + an
additive red wash) so the field dies red-tinted and dark. Applied AFTER
flashlight + HUD so the whole frame dies together. Held at the floor for the
frozen tail; the results screen then fades in over the dead frame.

AUDIO (``apply_fail_audio``): the muxed audio's final ``FAIL_FADE_MS`` before
death is time-warped so playback speed decays 1->0 (a resample, so pitch drops
WITH speed — the classic record "grinding to a halt"), swept through a low-pass
to 300 Hz and faded to silence; everything after death is silenced (the song
has stopped). Done as an isolated decode -> numpy warp -> remux post-pass so it
touches only failed renders.

Everything here is invoked ONLY on failed plays (the caller gates the whole
path on ``failed``), so passing renders are byte-identical.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── osu!lazer FailAnimationContainer constants ──────────────────────────────
# video-time span of the fail animation (lazer ``duration = 2500``). The caller
# multiplies this by the playback rate for the MAP-time window, so DT/HT still
# read ~2.5 s of VIDEO.
FAIL_FADE_MS = 2500.0

ROT_DEG = 1.0        # Content.RotateTo(1)  — +1 degree tilt
SCALE_TARGET = 0.85  # Content.ScaleTo(0.85f)
FALL_FRAC = 0.045    # our addition: content drops this fraction of frame height
GRAY_MIX = 0.5       # FadeColour(Gray): multiply pixels toward 0.5 (linear)
RED_ADD = 0.30       # additive red wash (lazer redFlashLayer, Red.Opacity(0.6))


def _out_quart(p: float) -> float:
    """Easing.OutQuart on [0,1] (clamped): 1-(1-p)^4 (decelerating)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1.0 - p
    return 1.0 - q * q * q * q


def death_progress(t_ms: float, death_ms: float, fade_ms: float) -> float:
    """Linear death position in [0,1] at frame map-time ``t_ms``.

    0 before the window opens (``death_ms - fade_ms``), ramps LINEARLY to 1 at
    ``death_ms``, then HELD at 1 for every later (tail / frozen) frame. The
    per-transform osu easings (OutQuart etc.) are applied in ``apply_death``,
    matching lazer where each transform carries its own easing over ``duration``."""
    if fade_ms <= 0:
        return 1.0 if t_ms >= death_ms else 0.0
    p = (t_ms - (death_ms - fade_ms)) / fade_ms
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return p


def apply_death(rgb: "np.ndarray", p: float) -> "np.ndarray":
    """Apply osu!'s fail transform to a composited frame ``rgb`` (H,W,3 uint8)
    at linear death position ``p`` in [0,1].

    ``p <= 0`` returns ``rgb`` unchanged (identity for pre-window frames);
    otherwise a NEW array. Colour-grade first (darken toward gray + additive
    red wash, linear in ``p``), then affine (rotate +1 deg, scale->0.85, fall),
    eased OutQuart to their target at death, filling revealed area with black."""
    if p <= 0.0:
        return rgb
    if p > 1.0:
        p = 1.0

    # ── colour grade (numpy): FadeColour(Gray) darken + additive red wash ──
    f = rgb.astype(np.float32)
    f *= (1.0 - GRAY_MIX * p)                 # -> 0.5x brightness at death (gray)
    f[..., 0] += (RED_ADD * 255.0) * p        # additive red (lazer red layer)
    np.clip(f, 0.0, 255.0, out=f)
    graded = f.astype(np.uint8)

    # ── affine (PIL): rotate + scale-down + fall about centre, black fill ──
    from PIL import Image

    h, w = graded.shape[0], graded.shape[1]
    eq = _out_quart(p)
    theta = np.deg2rad(ROT_DEG * eq)
    s = 1.0 - (1.0 - SCALE_TARGET) * eq       # -> 0.85 at death
    ty = FALL_FRAC * h * eq                    # downward fall (px)
    cx, cy = w * 0.5, h * 0.5
    k = 1.0 / s
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    # PIL AFFINE maps OUTPUT->INPUT: xi=a*xo+b*yo+c, yi=d*xo+e*yo+f. This is the
    # inverse of forward: po = R(theta)*(s*(pi-c)) + c + (0,ty).
    a = k * ct
    b = k * st
    c = cx - k * (ct * cx + st * (cy + ty))
    d = -k * st
    e = k * ct
    fcoef = cy - k * (-st * cx + ct * (cy + ty))
    # RGBA pipeline: `graded` may be HxWx4 (fromarray infers "RGBA"). PIL's
    # BILINEAR resampling on RGBA is ALPHA-WEIGHTED (premultiplied) — with
    # the canvas's GL-garbage alpha that skews the RGB result badly
    # (measured maxdiff 247/255 vs the 3ch path). Forcing alpha opaque
    # makes the premultiply a no-op, restoring the exact per-channel math
    # of the RGB path; `graded` is a fresh array, so the mutation is safe.
    # The opaque fill keeps the border blend identical too (a zero-alpha
    # fill would zero the interpolated border RGB under premultiply).
    if graded.shape[2] == 4:
        graded[..., 3] = 255
        fill = (0, 0, 0, 255)
    else:
        fill = (0, 0, 0)
    img = Image.fromarray(graded).transform(
        (w, h), Image.AFFINE, (a, b, c, d, e, fcoef),
        resample=Image.BILINEAR, fillcolor=fill)
    return np.asarray(img)


# ── audio: the record "grinds to a halt" ────────────────────────────────────
def _lp_sweep(x: "np.ndarray", sr: int, f_hi: float, f_lo: float) -> "np.ndarray":
    """One-pole low-pass whose cutoff sweeps ``f_hi`` -> ``f_lo`` (Easing.OutCubic)
    across ``x`` (N,2 float). Mirrors lazer's ``CutoffTo(300, duration, OutCubic)``."""
    n = len(x)
    if n == 0:
        return x
    t = np.arange(n, dtype=np.float64) / max(1, n - 1)
    fc = f_hi + (f_lo - f_hi) * (1.0 - (1.0 - t) ** 3)   # OutCubic
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * np.pi * fc)
    alpha = dt / (rc + dt)                                # per-sample smoothing
    y = np.empty_like(x)
    prev = x[0].astype(np.float64).copy()
    xa = x.astype(np.float64)
    for i in range(n):
        prev += alpha[i] * (xa[i] - prev)
        y[i] = prev
    return y


def apply_fail_audio(output_path: "Path | str", death_video_s: float,
                     window_s: float, *, sr: int = 48000) -> bool:
    """Grind the muxed audio to a halt at death and silence the tail.

    Decodes ``output_path``'s audio, time-warps the ``window_s`` before
    ``death_video_s`` so playback speed decays 1->0 (pitch drops with speed —
    a resampled record slowing to a stop), sweeps a low-pass to 300 Hz, fades
    to silence, silences everything after death, and remuxes (video copied).

    Fully fail-soft: returns True on success, False (leaving the file
    untouched) on any problem. ONLY call on failed renders."""
    output_path = Path(output_path)
    try:
        dec = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(output_path),
             "-map", "0:a:0", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True)
        if dec.returncode != 0 or not dec.stdout:
            return False                       # no audio stream / decode failed
        A = np.frombuffer(dec.stdout, dtype=np.float32).reshape(-1, 2).copy()
        n = len(A)
        if n == 0:
            return False
        d = int(round(death_video_s * sr))
        d = max(0, min(d, n))
        w = int(round(window_s * sr))
        w = min(w, d)
        if w >= 64:
            seg = A[d - w:d].astype(np.float64)
            tau = np.arange(w, dtype=np.float64)
            # speed(tau)=1-tau/w -> source offset = tau - tau^2/(2w) in [0,w/2].
            # Continuous with the un-warped audio at the window start (offset 0,
            # speed 1); reads the real last ~w/2 s stretched over w s, slowing.
            src = tau - (tau * tau) / (2.0 * w)
            i0 = np.floor(src).astype(np.int64)
            np.clip(i0, 0, w - 1, out=i0)
            frac = (src - i0)[:, None]
            i1 = np.minimum(i0 + 1, w - 1)
            warped = seg[i0] * (1.0 - frac) + seg[i1] * frac
            warped = _lp_sweep(warped, sr, 18000.0, 300.0)   # muffle to 300 Hz
            warped *= (1.0 - tau / w)[:, None]               # fade to silence
            A[d - w:d] = warped.astype(np.float32)
        A[d:] = 0.0                            # song has stopped: silence the tail

        tmp = output_path.with_suffix(".failaudio.mp4")
        mux = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-nostdin",
             "-i", str(output_path),
             "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "pipe:0",
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
             str(tmp)],
            input=A.astype("<f4").tobytes(), capture_output=True)
        if mux.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 8000:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        os.replace(tmp, output_path)
        return True
    except Exception as e:  # noqa: BLE001 — audio grind never breaks a render
        print(f"[catch] fail-audio grind skipped: {e}", file=sys.stderr,
              flush=True)
        return False
