"""Regression tests for the osu!stable replay lead-in fix (see
osu_catch_renderer/replay._recover_leadin_offset).

osrparse silently drops the up-to-two leading (256,-500) placeholder frames
WITHOUT accumulating their deltas, discarding the audio lead-in / intro-skip
that osu!'s LegacyScoreDecoder folds into the running clock. Accumulating the
survivors from 0 then starts the catcher clock the whole lead-in too early ->
fruit dropped through a stationary catcher / mass miss.

Runnable two ways:  pytest test/test_leadin.py   OR   python test/test_leadin.py
"""
from __future__ import annotations

import lzma
import struct
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_catch_renderer.replay import _recover_leadin_offset, parse_replay

HERE = Path(__file__).resolve().parent
STABLE_LEADIN = HERE / "stable_leadin.osr"   # real corpus replay, 2342ms intro-skip
LAZER = HERE / "lazer.osr"
_STABLE_LEADIN_MS = 2342


def _uleb_string(s: str) -> bytes:
    b = s.encode("utf-8")
    out = bytearray([0x0b])
    n = len(b)
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | 0x80 if n else byte)
        if not n:
            break
    return bytes(out) + b


def _make_osr(frames: str, mode: int = 2) -> bytes:
    blob = lzma.compress(frames.encode("ascii"), format=lzma.FORMAT_ALONE)
    out = bytearray()
    out.append(mode)
    out += struct.pack("<i", 20260101)         # stable game version
    out += _uleb_string("beatmapmd5")
    out += _uleb_string("player")
    out += _uleb_string("replaymd5")
    out += struct.pack("<6h", 0, 0, 0, 0, 0, 0)
    out += struct.pack("<i", 0)                # score
    out += struct.pack("<h", 0)                # max combo
    out += struct.pack("<b", 0)                # perfect
    out += struct.pack("<i", 0)                # mods
    out += _uleb_string("")                    # life bar
    out += struct.pack("<q", 0)                # timestamp
    out += struct.pack("<i", len(blob))
    out += blob
    out += struct.pack("<q", 0)                # online score id
    return bytes(out)


def test_recover_leadin_synthetic(tmp_path):
    # two (256,-500) placeholders summing to 2342 -> that is the lead-in.
    p = tmp_path / "s.osr"
    p.write_bytes(_make_osr("0|256|-500|0,2342|256|-500|0,14|100|100|0,20|100|100|0,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 2342


def test_recover_leadin_lazer_synthetic(tmp_path):
    # no (256,-500) prefix (lazer) -> nothing to recover.
    p = tmp_path / "l.osr"
    p.write_bytes(_make_osr("0|100|100|0,16|100|100|0,17|100|100|0,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 0


def test_recover_leadin_single_placeholder(tmp_path):
    # only the i=0 frame is a placeholder; i=1 is already real -> lead-in 0.
    p = tmp_path / "one.osr"
    p.write_bytes(_make_osr("0|256|-500|0,50|10|10|0,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 0


def test_recover_leadin_failsoft(tmp_path):
    # unparseable garbage must fail soft to 0 (pre-fix behaviour), never raise.
    p = tmp_path / "junk.osr"
    p.write_bytes(b"not an osr file at all")
    assert _recover_leadin_offset(p) == 0


def test_lazer_zero_offset():
    # a real lazer replay carries no placeholder frames -> byte-identical.
    assert _recover_leadin_offset(LAZER) == 0


def test_real_stable_leadin_recovered():
    assert _recover_leadin_offset(STABLE_LEADIN) == _STABLE_LEADIN_MS


def test_parse_seeds_clock_matches_osu():
    """End-to-end on a real 2342ms-intro-skip stable replay: the parsed catcher
    clock must start at the osu!-correct time (seed + first survivor delta),
    which the OLD from-0 logic got wrong by the whole lead-in."""
    from osrparse import Replay
    seed = _recover_leadin_offset(STABLE_LEADIN)
    r = Replay.from_path(STABLE_LEADIN)
    deltas = [int(e.time_delta) for e in (r.replay_data or [])
              if int(e.time_delta) != -12345]
    expected_first = max(0, seed + deltas[0])            # osu! truth
    old_first = max(0, 0 if deltas[0] < -5000 else deltas[0])  # pre-fix logic

    frames, _ = parse_replay(STABLE_LEADIN)
    assert frames == sorted(frames, key=lambda f: f.time_ms)
    assert all(f.time_ms >= 0 for f in frames)
    assert min(f.time_ms for f in frames) == expected_first
    # the whole point: the fix moved the clock, old logic was wrong here.
    assert old_first != expected_first
    assert expected_first >= _STABLE_LEADIN_MS


if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    test_recover_leadin_synthetic(tmp)
    test_recover_leadin_lazer_synthetic(tmp)
    test_recover_leadin_single_placeholder(tmp)
    test_recover_leadin_failsoft(tmp)
    test_lazer_zero_offset()
    test_real_stable_leadin_recovered()
    test_parse_seeds_clock_matches_osu()
    print("all catch lead-in tests PASSED")
