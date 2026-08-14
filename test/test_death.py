"""Death (fail) post-pass unit tests — the closing 'death beat' math.

Verifies the pure pieces the render loop wires up ONLY on a failed play:
the eased ramp (death_progress) and the shade (apply_death). The render loop
itself gates the whole path on ``failed``, so passing renders never call these.
"""
import numpy as np

from osu_catch_renderer.render.death import (FAIL_FADE_MS, apply_death,
                                             death_progress)


def test_progress_window_and_hold():
    d, fade = 28000.0, 1000.0
    # before the ramp opens -> 0
    assert death_progress(d - fade - 1, d, fade) == 0.0
    assert death_progress(d - fade, d, fade) == 0.0
    # exactly at death -> full
    assert death_progress(d, d, fade) == 1.0
    # held at the floor for the frozen tail
    assert death_progress(d + 5000, d, fade) == 1.0


def test_progress_eased_not_linear():
    d, fade = 28000.0, 1000.0
    mid = death_progress(d - fade / 2, d, fade)   # smoothstep(0.5) == 0.5
    quarter = death_progress(d - fade * 0.75, d, fade)  # smoothstep(0.25) < 0.25
    assert abs(mid - 0.5) < 1e-6
    assert quarter < 0.25            # eased in, not a linear-harsh wipe
    assert 0.0 < quarter


def test_progress_zero_fade_is_step():
    assert death_progress(27999, 28000, 0.0) == 0.0
    assert death_progress(28000, 28000, 0.0) == 1.0


def test_apply_identity_before_window():
    img = np.full((4, 4, 3), 200, np.uint8)
    out = apply_death(img, 0.0)
    assert out is img                      # p<=0 returns the SAME array


def test_apply_darkens_desaturates_and_reddens():
    # a saturated blue frame: full death must darken, kill blue, and end
    # redder than green/blue (dark red wash).
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 2] = 240                      # pure blue
    out = apply_death(img, 1.0).astype(np.float32)
    r, g, b = out[..., 0].mean(), out[..., 1].mean(), out[..., 2].mean()
    assert b < 240                         # darkened from the original blue
    assert r >= g and r >= b               # red is the brightest channel (tint)
    assert out.max() < 200                 # overall darker than the source


def test_fade_default_is_one_second():
    assert FAIL_FADE_MS == 1000.0
