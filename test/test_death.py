"""Death (fail) post-pass unit tests — the osu! fail-animation math.

Verifies the pure pieces the render loop wires up ONLY on a failed play: the
linear death ramp (death_progress) and the affine + colour-grade transform
(apply_death). The render loop gates the whole path on ``failed``, so passing
renders never call these (byte-identical passes).
"""
import numpy as np

from osu_catch_renderer.render.death import (FAIL_FADE_MS, apply_death,
                                             death_progress)


def test_progress_window_and_hold():
    d, fade = 28000.0, 2500.0
    # before the ramp opens -> 0
    assert death_progress(d - fade - 1, d, fade) == 0.0
    assert death_progress(d - fade, d, fade) == 0.0
    # exactly at death -> full
    assert death_progress(d, d, fade) == 1.0
    # held at the floor for the frozen tail
    assert death_progress(d + 5000, d, fade) == 1.0


def test_progress_is_linear():
    d, fade = 28000.0, 2500.0
    # the WHEN ramp is now linear; per-transform easings live in apply_death.
    assert abs(death_progress(d - fade / 2, d, fade) - 0.5) < 1e-6
    assert abs(death_progress(d - fade * 0.75, d, fade) - 0.25) < 1e-6


def test_progress_zero_fade_is_step():
    assert death_progress(27999, 28000, 0.0) == 0.0
    assert death_progress(28000, 28000, 0.0) == 1.0


def test_apply_identity_before_window():
    img = np.full((8, 8, 3), 200, np.uint8)
    out = apply_death(img, 0.0)
    assert out is img                      # p<=0 returns the SAME array


def test_apply_darkens_and_reddens_and_falls():
    # a bright frame at full death: darker overall, red the brightest channel
    # (red wash), and a black border/corner from the scale-down + fall.
    img = np.full((64, 64, 3), 220, np.uint8)
    out = apply_death(img, 1.0)
    assert out is not img
    f = out.astype(np.float32)
    r, g, b = f[..., 0].mean(), f[..., 1].mean(), f[..., 2].mean()
    assert r > g and r > b                 # red-tinted (additive red wash)
    assert f.mean() < 220                  # darkened from the source
    assert out[0, 0].sum() == 0            # top-left corner fell to black


def test_apply_grows_with_progress():
    img = np.full((64, 64, 3), 220, np.uint8)
    early = apply_death(img, 0.2).astype(np.float32).mean()
    late = apply_death(img, 0.9).astype(np.float32).mean()
    assert late < early                    # the field darkens/shrinks toward death


def test_fade_default_is_lazer_duration():
    assert FAIL_FADE_MS == 2500.0
