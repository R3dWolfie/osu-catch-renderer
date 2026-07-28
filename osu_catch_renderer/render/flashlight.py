"""osu!catch Flashlight (FL) — the darkening post-pass.

Ported 1:1 from lazer master (values cited so the fidelity is auditable):

  * ``CatchModFlashlight.DefaultFlashlightSize => 203.125f`` — the lit RADIUS in
    playfield units (the 512-wide osu!catch coordinate space; ``CatchPlayfield.
    WIDTH == 512``). Multiplied by the user ``SizeMultiplier`` (default 1.0,
    range 0.5..1.5) and the combo scale below.
  * ``CatchModFlashlight.GetComboScaleFor`` — combo SHRINKS the circle (catch's
    own overrides differ from std's 0.625/0.8125):
        combo >= 200 -> 0.770
        combo >= 100 -> 0.885
        else         -> 1.000
  * ``ModFlashlight.GetSize`` — during breaks the size instead grows: ``size *=
    2.5f`` (isBreakTime); otherwise the combo scale applies. Whenever the target
    changes it eases over ``FLASHLIGHT_FADE_DURATION == 800`` ms (a linear
    ``TransformTo``, Easing.None), which we reproduce with a linear ramp.
  * ``CatchFlashlight.FlashlightSmoothness = 1.4f`` — the edge feathers from the
    radius R out to R*1.4 (the ``sh_CircularFlashlight`` shader's
    ``smoothstep(flashlightRadius, flashlightRadius * flashlightSmoothness,
    dist)``; ``flashlightRadius = length(size) = size.y``).
  * ``CatchFlashlight.Update`` sets ``FlashlightPosition`` to the CATCHER's draw
    position every frame — the circle tracks the catcher plate, not a cursor.
  * ``FlashlightDim`` is left at its default 0 for catch, so inside the circle
    the playfield is fully clear (no interior dim). HD is independent of FL —
    it fades the fruit on approach and does not touch the flashlight.

The lit region (dist < R) is fully transparent (gameplay shows); the annulus
R..R*1.4 feathers to black; everything past R*1.4 is solid black. Composited as
a black overlay whose alpha is ``smoothstep`` gives, per pixel,
``out = gameplay * (1 - smoothstep(R, R*1.4, dist))`` — exactly lazer's
CircularFlashlight over a black box with dim 0.

Applied as a post-pass over the composited playfield frame, BEFORE the HUD is
drawn on top, so score / accuracy / combo / break overlays stay lit (in lazer
the Flashlight sits in the playfield/overlays layer and the HUD is above it).
"""
from __future__ import annotations

import numpy as np

# ── lazer constants (cited above) ─────────────────────────────────────────────
FL_BIT = 1 << 10                      # 1024 — osu! Flashlight mod bit
DEFAULT_FLASHLIGHT_SIZE = 203.125     # CatchModFlashlight.DefaultFlashlightSize (playfield units)
SIZE_MULTIPLIER = 1.0                 # CatchModFlashlight.SizeMultiplier default (0.5..1.5, step .1)
SMOOTHNESS = 1.4                      # CatchFlashlight.FlashlightSmoothness
FADE_MS = 800.0                       # ModFlashlight.FLASHLIGHT_FADE_DURATION
BREAK_MULT = 2.5                      # ModFlashlight.GetSize break-time size *= 2.5f


def has_flashlight(mods) -> bool:
    """True when the replay's mods enable Flashlight (bit 1<<10)."""
    return bool(int(mods or 0) & FL_BIT)


def combo_scale(combo: int) -> float:
    """CatchModFlashlight.GetComboScaleFor — the circle shrinks with combo."""
    if combo >= 200:
        return 0.770
    if combo >= 100:
        return 0.885
    return 1.0


class CatchFlashlight:
    """Stateful per-render flashlight post-pass (frames arrive in map-time order,
    so the 800 ms size transition can be integrated frame to frame).

    ``break_env`` (optional) is the sim's DimEnvelope for [Events] breaks; when
    its weight at the frame time is >0.5 we treat it as break time and grow the
    circle 2.5x, matching lazer's ``isBreakTime`` branch. Absent env → never a
    break (fail-soft), which only omits the break widening, never the effect."""

    def __init__(self, break_env=None, size_multiplier: float = SIZE_MULTIPLIER):
        self._break_env = break_env
        self.size_multiplier = float(size_multiplier)
        # linear-transition state, all in playfield units
        self._cur = None            # current animated size
        self._from = None           # transition start value
        self._to = None             # transition target value
        self._start = 0.0           # transition start time (ms)

    # -- size (playfield units) with lazer's 800ms linear TransformTo ----------
    def _target(self, combo: int, break_active: bool) -> float:
        base = DEFAULT_FLASHLIGHT_SIZE * self.size_multiplier
        if break_active:
            return base * BREAK_MULT
        return base * combo_scale(combo)

    def _size(self, t_ms: float, combo: int, break_active: bool) -> float:
        tgt = self._target(combo, break_active)
        if self._cur is None:                      # first frame: snap, no ramp
            self._cur = self._from = self._to = tgt
            self._start = t_ms
            return self._cur
        if tgt != self._to:                        # new target → start an 800ms ramp
            self._from = self._cur
            self._to = tgt
            self._start = t_ms
        if FADE_MS <= 0:
            frac = 1.0
        else:
            frac = min(1.0, max(0.0, (t_ms - self._start) / FADE_MS))
        self._cur = self._from + (self._to - self._from) * frac
        return self._cur

    def radius_px(self, scene, break_active: bool | None = None) -> float:
        """Lit radius R in screen pixels for this frame (exposed for tests)."""
        if break_active is None:
            break_active = self._is_break(getattr(scene, "time_ms", 0))
        combo = int(getattr(scene, "combo", 0) or 0)
        unit = getattr(scene, "pf_unit_px", None) or 0.0
        return self._size(float(getattr(scene, "time_ms", 0)), combo, break_active) * unit

    def _is_break(self, t_ms) -> bool:
        if self._break_env is None:
            return False
        try:
            return self._break_env.level(t_ms) > 0.5
        except Exception:                          # noqa: BLE001 — break widening is optional
            return False

    # -- the post-pass ---------------------------------------------------------
    def apply(self, rgb: "np.ndarray", scene) -> "np.ndarray":
        """Darken ``rgb`` (H,W,3 uint8) outside the lit circle centred on the
        catcher plate. Returns a NEW array (GL readback is read-only). If the
        frame lacks catcher geometry, returns ``rgb`` unchanged."""
        cx = getattr(scene, "catcher_px", None)
        cy = getattr(scene, "plane_y_px", None)
        unit = getattr(scene, "pf_unit_px", None)
        if cx is None or cy is None or not unit or unit <= 0:
            return rgb

        break_active = self._is_break(getattr(scene, "time_ms", 0))
        combo = int(getattr(scene, "combo", 0) or 0)
        R = self._size(float(getattr(scene, "time_ms", 0)), combo, break_active) * unit
        if R <= 0:
            return rgb
        Ro = R * SMOOTHNESS                          # outer feather radius (fully black beyond)

        h, w = rgb.shape[:2]
        out = np.zeros_like(rgb)                     # outside the Ro box is exactly black
        # Bounding box: any pixel with max(|dx|,|dy|) > Ro has dist > Ro -> black,
        # so only the [c ± Ro] square can be non-black. This bounds the per-frame
        # cost to ~(2*Ro)^2 pixels instead of the whole frame.
        x0 = max(0, int(np.floor(cx - Ro)))
        x1 = min(w, int(np.ceil(cx + Ro)) + 1)
        y0 = max(0, int(np.floor(cy - Ro)))
        y1 = min(h, int(np.ceil(cy + Ro)) + 1)
        if x1 <= x0 or y1 <= y0:                     # circle fully off-frame → all black
            return out

        ys = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(cy)
        xs = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(cx)
        dist = np.sqrt(xs * xs + ys * ys)            # (bh, bw)
        # smoothstep(R, Ro, dist): 0 inside R, 1 beyond Ro. Keep = 1 - that.
        denom = max(Ro - R, 1e-6)
        t = np.clip((dist - R) / denom, 0.0, 1.0)
        black = t * t * (3.0 - 2.0 * t)              # shader smoothstep
        keep = (1.0 - black)[..., None]              # (bh, bw, 1)
        patch = rgb[y0:y1, x0:x1].astype(np.float32) * keep
        out[y0:y1, x0:x1] = np.rint(patch).astype(np.uint8)
        return out
