"""Exercise fail classification through real replay parsing and simulation.

The render is stopped immediately after CatchSim construction, before any GPU,
audio or encoder work. Fixtures are generated here and contain no player data.
"""
import importlib
import lzma
import struct

import pytest


@pytest.fixture(params=["osu_catch_renderer", "osu_catch_renderer._metal"])
def backend(request):
    return (
        importlib.import_module(request.param + ".beatmap.models"),
        importlib.import_module(request.param + ".beatmap.replay"),
        importlib.import_module(request.param + ".render.render"),
    )


def _string(value):
    data = value.encode("utf-8")
    length = len(data)
    prefix = bytearray([0x0b])
    while length >= 0x80:
        prefix.append((length & 0x7f) | 0x80)
        length >>= 7
    return bytes(prefix) + bytes([length]) + data


def _replay(tmp_path, parser, frames, *, counts, life, mods=0, version=20260101):
    previous = 0
    events = []
    for time_ms, x in frames:
        events.append(f"{time_ms - previous}|{x}|0|0,")
        previous = time_ms
    events.append("-12345|0|0|1,")
    blob = lzma.compress("".join(events).encode("ascii"), format=lzma.FORMAT_ALONE)
    data = bytes([2]) + struct.pack("<i", version)
    data += _string("synthetic-map") + _string("fixture") + _string("synthetic-replay")
    data += struct.pack("<6H", *counts)
    data += struct.pack("<iHBi", 0, counts[0] + counts[1], 0, mods)
    data += _string(life) + struct.pack("<qi", 0, len(blob)) + blob + struct.pack("<q", 0)
    path = tmp_path / "fixture.osr"
    path.write_bytes(data)
    return parser.parse_replay(path)


def _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path):
    real_sim = renderer.CatchSim

    class SimulationReady(Exception):
        def __init__(self, sim):
            self.sim = sim

    def stop_after_sim(*args, **kwargs):
        # Keep the real geometry, count reconciliation and desync guard.
        raise SimulationReady(real_sim(*args, **kwargs))

    monkeypatch.setattr(renderer, "CatchSim", stop_after_sim)
    with pytest.raises(SimulationReady) as done:
        renderer.render_core(bm, frames, meta, tmp_path / "unused.mp4", models.RenderConfig())
    return done.value.sim


def _fruit_map(models, count):
    return models.CatchBeatmap(objects=[
        models.CatchObject(i * 1000, 256.0, models.ObjType.FRUIT)
        for i in range(1, count + 1)
    ])


@pytest.mark.parametrize("version,life", [
    (20260101, "0|1,30000|0,31000|0.5,100000|0.7,"),
    (30000019, ""),
])
def test_completed_low_score_continues_after_zero_or_early_input_stop(
        backend, monkeypatch, tmp_path, version, life):
    models, parser, renderer = backend
    bm = _fruit_map(models, 100)
    # Exactly 60 catches / 40 misses; input stops at 60s in the lazer case.
    # The stationary tail at x=0 legitimately misses every remaining fruit.
    raw = [(i * 1000, 256 if i <= 60 else 0) for i in range(1, 101)]
    if not life:
        raw = raw[:60] + [(60001, 0)]
    frames, meta = _replay(tmp_path, parser, raw, counts=(60, 0, 0, 0, 0, 40),
                           life=life, mods=8 | 128, version=version)
    sim = _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)
    assert sim._end_ms is None
    assert len(sim._objs) == 100
    assert sim._objs[-1].time_ms == 100000
    assert sum(sim._caught) == 60


@pytest.mark.parametrize("mods", [0, 1 << 5, (1 << 14) | (1 << 5)])
def test_genuine_early_failure_stops_at_last_judged_object(
        backend, monkeypatch, tmp_path, mods):
    models, parser, renderer = backend
    bm = _fruit_map(models, 100)
    frames, meta = _replay(tmp_path, parser, [(1000, 256), (30000, 256)],
                           counts=(29, 0, 0, 0, 0, 1), life="0|1,30000|0,", mods=mods)
    sim = _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)
    assert sim._end_ms == 30000
    assert len(sim._objs) == 30
    assert sim._objs[-1].time_ms == 30000


@pytest.mark.parametrize("deficit", [1, 4, 5])
@pytest.mark.parametrize("lifebar", [True, False])
def test_fd_tolerance_does_not_turn_sparse_early_failure_into_clear(
        backend, monkeypatch, tmp_path, deficit, lifebar):
    models, parser, renderer = backend
    bm = _fruit_map(models, 10)
    reached = 10 - deficit
    frames, meta = _replay(tmp_path, parser, [(1000, 256), (reached * 1000, 256)],
                           counts=(reached - 1, 0, 0, 0, 0, 1),
                           life=f"0|1,{reached * 1000}|0," if lifebar else "",
                           version=20260101 if lifebar else 30000019)
    sim = _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)
    assert sim._end_ms == reached * 1000
    assert len(sim._objs) == reached


def test_no_fail_exempts_zero_life(backend, monkeypatch, tmp_path):
    models, parser, renderer = backend
    bm = _fruit_map(models, 10)
    frames, meta = _replay(tmp_path, parser, [(1000, 256), (10000, 256)],
                           counts=(10, 0, 0, 0, 0, 0), life="0|1,3000|0,", mods=1)
    assert meta.death_ms is None
    sim = _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)
    assert len(sim._objs) == 10
    assert sim._end_ms is None


@pytest.mark.parametrize("death_ms,want_end,want_objects", [
    (9799, 9799, 9),
    (9800, None, 10),
])
def test_final_note_grace_remains_bounded_to_200ms(
        backend, monkeypatch, tmp_path, death_ms, want_end, want_objects):
    models, parser, renderer = backend
    bm = _fruit_map(models, 10)
    frames, meta = _replay(tmp_path, parser, [(1000, 256), (death_ms, 256)],
                           counts=(8, 0, 0, 0, 0, 1), life=f"0|1,{death_ms}|0,")
    sim = _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)
    assert sim._end_ms == want_end
    assert len(sim._objs) == want_objects


def test_zero_fd_cannot_prove_a_clear(backend, monkeypatch, tmp_path):
    models, parser, renderer = backend
    bm = models.CatchBeatmap(objects=[
        models.CatchObject(i * 1000, 256, models.ObjType.BANANA)
        for i in range(1, 11)
    ])
    frames, meta = _replay(tmp_path, parser, [(1000, 256), (3000, 256)],
                           counts=(0, 0, 0, 0, 0, 0), life="")
    sim = _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)
    assert sim._end_ms == 3000
    assert len(sim._objs) == 3


def test_full_header_does_not_disable_real_desync_guard(backend, monkeypatch, tmp_path):
    models, parser, renderer = backend
    bm = _fruit_map(models, 100)
    frames, meta = _replay(tmp_path, parser, [(1000, 0), (100000, 0)],
                           counts=(100, 0, 0, 0, 0, 0), life="0|1,30000|0,31000|1,")
    with pytest.raises(RuntimeError, match=r"desynced.*100/100 objects"):
        _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("mods", [0, 64, 256])
def test_native_and_convert_slider_clear_keeps_bonus_tail_in_map_time(
        backend, monkeypatch, tmp_path, mode, mods):
    models, parser, renderer = backend
    beatmap_parser = importlib.import_module(renderer.__package__.rsplit(".", 1)[0]
                                            + ".beatmap.beatmap")
    path = tmp_path / "generated.osu"
    circles = "\n".join(f"256,192,{i * 1000},1,0,0:0:0:0:" for i in range(1, 11))
    path.write_text(f"""osu file format v14
[General]
Mode:{mode}
[Difficulty]
CircleSize:5
ApproachRate:9
OverallDifficulty:7
HPDrainRate:5
SliderMultiplier:1
SliderTickRate:1
[TimingPoints]
0,500,4,1,0,100,1,0
[HitObjects]
{circles}
256,192,11000,2,0,L|256:392,1,200
256,192,13000,8,0,15000
""")
    bm = beatmap_parser.parse_beatmap(path, mods=mods, position_offsets=False)
    # Ten circles + slider head/tail = 12 fruit, one large tick, 14 tinies,
    # and 33 bonus bananas. The final tail fruit follows every tiny.
    assert len(bm.objects) == 60
    assert max(o.time_ms for o in bm.objects if o.kind is models.ObjType.TINY_DROPLET) < 12000
    frames, meta = _replay(tmp_path, parser, [(1000, 256), (8000, 256)],
                           counts=(12, 1, 14, 0, 0, 0), life="", mods=mods,
                           version=30000019)
    sim = _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)
    assert sim._end_ms is None
    assert len(sim._objs) == 60
    assert sim._objs[-1].time_ms == 15000
    assert meta.death_ms == 8000


@pytest.mark.parametrize("backend", ["osu_catch_renderer"], indirect=True)
def test_completed_lazer_mirror_keeps_full_per_sim_geometry(backend, monkeypatch, tmp_path):
    models, parser, renderer = backend
    objects = [models.CatchObject(i * 1000, float((i * 37) % 500 + 6), models.ObjType.FRUIT)
               for i in range(1, 121)]
    bm = models.CatchBeatmap(objects=objects)
    frames, meta = _replay(tmp_path, parser, [(o.time_ms, 512 - o.x) for o in objects],
                           counts=(120, 0, 0, 0, 0, 0), life="0|1,30000|0,31000|1,",
                           version=30000019)
    sim = _simulate(monkeypatch, renderer, models, bm, frames, meta, tmp_path)
    assert sim._end_ms is None
    assert len(sim._objs) == 120
    assert sum(sim._caught) == 120
    assert [o.x for o in bm.objects] == [o.x for o in objects]
    assert [o.x for o in sim._objs] == [512 - o.x for o in objects]
