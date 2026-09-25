"""Regression tests for lazer Catch Mirror auto-detection (scene.CatchSim._maybe_apply_mirror).

Mirror is a lazer-only mod that flips the playfield horizontally. lazer's legacy .osr
export writes mods:0 in the stable bitfield (the real mod set lives in an appended
LegacyReplaySoloScoreInfo block osrparse does not decode), so without detection every
fruit is placed on the wrong side and the honesty guard false-rejects a legit replay.

_maybe_apply_mirror uses no instance state, so we drive it directly with a dummy self
(None) and lightweight beatmap/meta stubs. It is NON-MUTATING (Aussie release-gate fix
2026-09-06): it RETURNS the per-sim beatmap to use (a copy with flipped objects when it
fires, else the input unchanged) and must never touch the caller's objects list — see
test_mixed_mirror_and_stable_overlay_share_no_geometry for the exact bug it guards.
"""
from types import SimpleNamespace

from osu_catch_renderer.beatmap.models import (
    CatchObject, CatchFrame, ObjType, CatchBeatmap, RenderConfig, ReplayMeta,
)
from osu_catch_renderer.render.scene import CatchSim

LAZER = 30000019          # a lazer legacy-export version (>= 30000000)
STABLE = 20210520         # a stable client version


def _meta(game_version, n_fruit):
    """A full-clear ReplayMeta for a map of n_fruit fruit — enough for CatchSim's
    count-reconcile to build cleanly; the geometry assertions don't depend on the
    score/accuracy values, only on game_version driving the Mirror gate."""
    return ReplayMeta(
        mode=2, beatmap_md5="x", player_name="p", mods=0, score=0,
        max_combo=n_fruit, count_300=n_fruit, count_100=0, count_50=0,
        count_katu=0, count_miss=0, accuracy=1.0, grade="S",
        game_version=game_version, death_ms=None)


class _BM:
    """Minimal stand-in for CatchBeatmap: the detector only reads .cs and .objects."""
    def __init__(self, cs, objects):
        self.cs = cs
        self.objects = objects


def _objects(n=120):
    """n fruit spread across the playfield, one flagged hyperdash with a target."""
    objs = []
    for i in range(n):
        x = float((i * 37) % 500 + 6)          # 6..505, well spread
        hyper = (i == 10)
        objs.append(CatchObject(
            time_ms=i * 100, x=x, kind=ObjType.FRUIT,
            hyperdash=hyper,
            hyper_target_x=(400.0 if hyper else None),
        ))
    return objs


def _frames(objs, mirror):
    """One frame per object placing the catcher exactly on the (optionally mirrored)
    fruit, so identity vs 512-x alignment is unambiguous. Extra tail frame so every
    object is inside the frame span."""
    fr = [CatchFrame(time_ms=o.time_ms, x=(512.0 - o.x) if mirror else o.x, dashing=False)
          for o in objs]
    fr.append(CatchFrame(time_ms=objs[-1].time_ms + 100, x=fr[-1].x, dashing=False))
    return fr


def test_lazer_mirror_detected_and_flipped():
    objs = _objects()
    orig = [(o.x, o.hyper_target_x) for o in objs]
    bm = _BM(cs=4.0, objects=list(objs))
    frames = _frames(objs, mirror=True)          # catcher tracks the mirrored layout
    out = CatchSim._maybe_apply_mirror(None, bm, frames, SimpleNamespace(game_version=LAZER))
    # the RETURNED beatmap carries the flipped layout: x -> 512 - x
    for o, (ox, _) in zip(out.objects, orig):
        assert o.x == 512.0 - ox
    # and the INPUT beatmap's own list was never mutated (non-mutating contract)
    for o, (ox, _) in zip(bm.objects, orig):
        assert o.x == ox


def test_mirrored_hyper_target_is_flipped():
    objs = _objects()
    bm = _BM(cs=4.0, objects=list(objs))
    frames = _frames(objs, mirror=True)
    out = CatchSim._maybe_apply_mirror(None, bm, frames, SimpleNamespace(game_version=LAZER))
    hyper = [o for o in out.objects if o.hyperdash][0]
    # hyper_target_x is an ABSOLUTE coord and must mirror too (Aussie review fix)
    assert hyper.hyper_target_x == 512.0 - 400.0


def test_lazer_nomod_unchanged():
    objs = _objects()
    orig = [o.x for o in objs]
    bm = _BM(cs=4.0, objects=list(objs))
    frames = _frames(objs, mirror=False)         # catcher tracks the real layout
    out = CatchSim._maybe_apply_mirror(None, bm, frames, SimpleNamespace(game_version=LAZER))
    # not mirrored -> the input beatmap is returned unchanged
    assert out is bm
    assert [o.x for o in out.objects] == orig


def test_stable_replay_never_flipped():
    """Even geometry that would look 'mirrored' must not be touched for a stable
    replay: Catch Mirror cannot legitimately exist there."""
    objs = _objects()
    orig = [o.x for o in objs]
    bm = _BM(cs=4.0, objects=list(objs))
    frames = _frames(objs, mirror=True)          # would trip the heuristic if allowed
    out = CatchSim._maybe_apply_mirror(None, bm, frames, SimpleNamespace(game_version=STABLE))
    assert out is bm                             # gate returns the input, no copy
    assert [o.x for o in out.objects] == orig


def test_mixed_mirror_and_stable_overlay_share_no_geometry():
    """Aussie's release-gate repro (Catch PR #3 shared-beatmap ownership bug).

    render_core hands the SAME CatchBeatmap instance to the primary sim and to
    every versus-overlay sim. A lazer Mirror primary must operate on its own
    per-sim copy: flipping the shared beatmap in place made a following
    stable/Nomod overlay (whose stable gate correctly refuses Mirror detection)
    build against the already-flipped geometry.

    Expected after the fix:
        shared beatmap unchanged: True
        mirror sim is mirrored:   True
        stable sim is original:   True
    """
    objs = _objects()
    orig_x = [o.x for o in objs]
    shared = CatchBeatmap(objects=list(objs), cs=4.0, ar=9.0)
    cfg = RenderConfig()

    # 1) PRIMARY: lazer Mirror — frames track the mirrored layout so detection fires.
    mirror_sim = CatchSim(shared, _frames(objs, mirror=True), cfg, skin=None,
                          meta=_meta(LAZER, len(objs)))
    # 2) OVERLAY: stable Nomod on the real layout — the stable gate refuses Mirror.
    stable_sim = CatchSim(shared, _frames(objs, mirror=False), cfg, skin=None,
                          meta=_meta(STABLE, len(objs)))

    shared_unchanged = [o.x for o in shared.objects] == orig_x
    mirror_mirrored = [o.x for o in mirror_sim.bm.objects] == [512.0 - x for x in orig_x]
    stable_original = [o.x for o in stable_sim.bm.objects] == orig_x

    assert shared_unchanged, "shared beatmap was mutated by the Mirror sim"
    assert mirror_mirrored, "Mirror sim did not build against flipped geometry"
    assert stable_original, "stable/Nomod sim inherited the primary's flipped geometry"
    # the two sims must not share one objects list either
    assert mirror_sim.bm.objects is not stable_sim.bm.objects
    assert mirror_sim.bm.objects is not shared.objects
