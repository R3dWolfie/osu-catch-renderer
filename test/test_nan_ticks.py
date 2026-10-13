"""Legacy NaN control points suppress large ticks, not fruit or tiny droplets."""
import importlib
from collections import Counter

import pytest


@pytest.fixture(params=['osu_catch_renderer', 'osu_catch_renderer._metal'])
def backend(request):
    return importlib.import_module(request.param + '.beatmap.beatmap')


def _objects(backend, timing):
    fields = '0,192,0,2,0,L|320:192,2,320'.split(',')
    return backend._slider_objects(fields, 0, 0, 0, False,
                                   timing=backend._parse_timing(timing),
                                   slider_mult=1.6, tick_rate=1, hr=False)


def test_nan_suppresses_ticks_but_keeps_repeats_and_tinies(backend):
    normal = _objects(backend, '0,500,4,1,0,100,1,0')
    disabled = _objects(backend, '0,500,4,1,0,100,1,0\n0,NaN,4,1,0,100,0,0')
    n = Counter(o.kind.name for o in normal)
    d = Counter(o.kind.name for o in disabled)
    assert n['DROPLET'] == 2
    assert d['DROPLET'] == 0
    assert n['FRUIT'] == d['FRUIT'] == 3
    assert d['TINY_DROPLET'] > n['TINY_DROPLET']
    assert [(o.time_ms, o.x) for o in normal if o.kind.name == 'FRUIT'] == [
        (o.time_ms, o.x) for o in disabled if o.kind.name == 'FRUIT']


@pytest.mark.parametrize('reset', ['500,4,1,0,100,1,0', '-100,4,1,0,100,0,0'])
def test_finite_control_point_reenables_ticks(backend, reset):
    t = backend._parse_timing('0,500,4,1,0,100,1,0\n0,NaN,4,1,0,100,0,0\n1000,' + reset)
    assert not t.generate_ticks(999)
    assert t.generate_ticks(1000)


def test_nan_resets_prior_velocity_to_one(backend):
    t = backend._parse_timing('0,500,4,1,0,100,1,0\n0,-50,4,1,0,100,0,0\n1000,NaN,4,1,0,100,0,0')
    assert t.sv_mult(999) == 2
    assert t.sv_mult(1000) == 1
    assert not t.generate_ticks(1000)


def test_green_overrides_red_at_same_time_in_either_order(backend):
    for points in ('0,500,4,1,0,100,1,0\n0,NaN,4,1,0,100,0,0',
                   '0,NaN,4,1,0,100,0,0\n0,500,4,1,0,100,1,0'):
        assert not backend._parse_timing(points).generate_ticks(0)


def test_default_and_normal_timing_still_generate_ticks(backend):
    assert backend._parse_timing('').generate_ticks(0)
    assert backend._parse_timing('0,500,4,1,0,100,1,0').generate_ticks(-1)
    assert backend._parse_timing('0,500,4,1,0,100,1,0').generate_ticks(0)
