"""osu!catch fail (death) post-pass — the closing "death beat" of a FAILED play.

On a FAILED replay the sim is already truncated at ``death_ms`` (no fruit spawn
past death, catcher frozen). Instead of the video ending on a hard cut, this
module ramps a death effect over the final ~1 second of gameplay ENDING at
``death_ms``: the composited frame progressively (a) desaturates toward luma,
(b) darkens, and (c) takes a dark-red tint — ending on a darkened, frozen
frame (the osu! fail feel). Frames at/after ``death_ms`` (the ``tail_ms`` hold)
sit at the fully-dead floor, so the results screen — if enabled — then fades in
over that darkened frame (gameplay dies, THEN results).

Applied AFTER flashlight + HUD compositing (unlike the flashlight, which is
pre-HUD) so the whole frame — playfield and the score / combo / accuracy HUD —
dies uniformly. The intensity is a pure function of the frame's map time vs.
``death_ms``; it is ONLY ever invoked on failed plays, so passing renders are
byte-identical (the caller gates the whole path on ``failed``).

The ramp is defined in map time; the caller scales the window by the playback
rate (DT/HT) so it always reads as ~1 s of VIDEO. Eased with a smoothstep
(shared shape with dim.py) so death eases in rather than a linear-harsh wipe.
"""
from __future__ import annotations

import numpy as np

# video-time span of the death ramp (~1 s). The caller multiplies this by the
# playback rate so DT/HT stay ~1 s on screen.
FAIL_FADE_MS = 1000.0

# full-death (p=1) look, applied AFTER a near-full desaturation:
_DESAT = 0.85     # how far toward grayscale at full death (1.0 = pure gray)
_END_R = 0.55     # per-channel brightness floor -> a dark red wash
_END_G = 0.16
_END_B = 0.16


def _smoothstep(p: float) -> float:
    """3p² - 2p³ on [0,1] (clamped) — the same ease dim.py uses."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return p * p * (3.0 - 2.0 * p)


def death_progress(t_ms: float, death_ms: float, fade_ms: float) -> float:
    """Eased death intensity in [0,1] at frame map-time ``t_ms``.

    0 before the ramp opens (``death_ms - fade_ms``), smoothsteps up to 1 at
    ``death_ms``, then HELD at 1 for every later (tail / frozen) frame."""
    if fade_ms <= 0:
        return 1.0 if t_ms >= death_ms else 0.0
    return _smoothstep((t_ms - (death_ms - fade_ms)) / fade_ms)


def apply_death(rgb: "np.ndarray", p: float) -> "np.ndarray":
    """Blend the death shade into ``rgb`` (H,W,3 uint8) at eased intensity
    ``p`` in [0,1].

    ``p <= 0`` returns ``rgb`` unchanged (identity for pre-window frames);
    otherwise a NEW array. Order: desaturate toward luma, then apply the dark
    red channel wash — both scaled by ``p`` so the effect grows smoothly to
    its floor at death."""
    if p <= 0.0:
        return rgb
    if p > 1.0:
        p = 1.0
    f = rgb.astype(np.float32)
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    d = _DESAT * p
    f = f * (1.0 - d) + lum[..., None] * d
    f[..., 0] *= 1.0 - (1.0 - _END_R) * p
    f[..., 1] *= 1.0 - (1.0 - _END_G) * p
    f[..., 2] *= 1.0 - (1.0 - _END_B) * p
    return np.clip(f, 0.0, 255.0).astype(np.uint8)
