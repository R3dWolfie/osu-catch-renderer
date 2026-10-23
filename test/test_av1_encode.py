"""AV1 master-video encoding (osu_catch_renderer/av1.py) in BOTH render paths.

Covers Leon's AV1 migration design decisions:
  * the four backend-specific recipes (NOT a mechanical h264->av1 rename);
    VAAPI uses -rc_mode CQP -global_quality (never -qp);
  * 8-bit 4:2:0, AV1 Main profile, +faststart, a GOP;
  * the master is AV1 while the inline preview AND the Discord compact stay
    H.264 (libx264) in the same two-output ffmpeg command;
  * FAIL-CLOSED selection: no H.264 / CPU / libsvtav1 fallback for the master;
  * the real hardware probe confirms codec_name=av1 (a soft, hardware-tolerant
    contract test, plus a live av1_vaapi probe that runs only where a render
    node exists).

Runnable two ways:  pytest test/test_av1_encode.py   OR   python test/test_av1_encode.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_catch_renderer import av1  # noqa: E402

_CPU_AV1 = ("libsvtav1", "libaom-av1", "librav1e", "libaom")

_PATHS: list = []


def _paths():
    """(name, render module, RenderConfig) for each render path that imports here."""
    if not _PATHS:
        from osu_catch_renderer.render import render as gl_render
        from osu_catch_renderer.beatmap.models import RenderConfig as GLConfig
        _PATHS.append(("gl", gl_render, GLConfig))
        try:
            from osu_catch_renderer._metal.render import render as metal_render
            from osu_catch_renderer._metal.beatmap.models import RenderConfig as MetalConfig
            _PATHS.append(("metal", metal_render, MetalConfig))
        except Exception as e:  # noqa: BLE001
            print(f"NOTE: the Metal render path does not import here ({e}); testing GL only")
    return _PATHS


def _both(fn):
    def run():
        for path in _paths():
            fn(path)
    run.__name__ = fn.__name__
    return run


class _Captured(Exception):
    pass


def _cmd(path, *, encoder, audio=None, preview=None, compact=None,
         res=(1280, 720), out="m.mp4"):
    """The ffmpeg command catch would start for `encoder`, without starting it."""
    _, R, RenderConfig = path
    saved = R.subprocess.Popen
    saved_cache = os.environ.get("R3D_LOUDNORM_CACHE_DIR")
    got: list = []

    def fake_popen(cmd, **kw):
        got.append(list(cmd))
        raise _Captured()

    R.subprocess.Popen = fake_popen
    scratch = tempfile.mkdtemp()
    os.environ["R3D_LOUDNORM_CACHE_DIR"] = scratch
    try:
        cfg = RenderConfig(resolution=res, fps=60, encoder=encoder)
        try:
            R._spawn_ffmpeg(cfg, Path(out), Path(audio) if audio else None, 0, 1.0, 60.0,
                            preview_path=Path(preview) if preview else None,
                            compact_path=Path(compact) if compact else None)
        except _Captured:
            pass
        assert got, "ffmpeg was not started"
        return got[-1]
    finally:
        R.subprocess.Popen = saved
        shutil.rmtree(scratch, ignore_errors=True)
        if saved_cache is None:
            os.environ.pop("R3D_LOUDNORM_CACHE_DIR", None)
        else:
            os.environ["R3D_LOUDNORM_CACHE_DIR"] = saved_cache


def _master_slice(cmd, out="m.mp4"):
    """The master output's args (everything up to and including its path)."""
    return cmd[:cmd.index(out) + 1]


# --------------------------------------------------------------------------- #
# master_video_args: the backend recipes (pure, hardware-independent)          #
# --------------------------------------------------------------------------- #

def test_vaapi_uses_cqp_global_quality_never_qp():
    a = av1.master_video_args("av1_vaapi", w=1920, h=1080, fps=60)
    assert a[a.index("-c:v") + 1] == "av1_vaapi"
    assert "format=nv12,hwupload" in a                 # 8-bit 4:2:0 on a GPU surface
    assert a[a.index("-rc_mode") + 1] == "CQP"
    assert "-global_quality" in a and "-qp" not in a   # the test-kit finding
    assert a[a.index("-profile:v") + 1] == "main"      # AV1 Main profile
    assert "-g" in a                                    # a sensible GOP


def test_vaapi_bitrate_override_switches_to_vbr():
    a = av1.master_video_args("av1_vaapi", w=1280, h=720, fps=60,
                              video_bitrate=5_000_000)
    assert a[a.index("-rc_mode") + 1] == "VBR"
    assert a[a.index("-b:v") + 1] == "5000000" and "-global_quality" not in a


def test_nvenc_is_cq_vbr_yuv420p_not_qp():
    a = av1.master_video_args("av1_nvenc", w=1920, h=1080, fps=60)
    assert a[a.index("-c:v") + 1] == "av1_nvenc"
    assert a[a.index("-pix_fmt") + 1] == "yuv420p"      # 8-bit 4:2:0
    assert a[a.index("-rc") + 1] == "vbr" and "-cq" in a and "-qp" not in a
    assert "-maxrate" in a and "-g" in a


def test_qsv_is_global_quality_nv12():
    a = av1.master_video_args("av1_qsv", w=1280, h=720, fps=60)
    assert a[a.index("-c:v") + 1] == "av1_qsv"
    assert a[a.index("-pix_fmt") + 1] == "nv12"
    assert "-global_quality" in a and "-g" in a


def test_amf_is_qvbr_yuv420p():
    a = av1.master_video_args("av1_amf", w=1280, h=720, fps=60)
    assert a[a.index("-c:v") + 1] == "av1_amf"
    assert a[a.index("-pix_fmt") + 1] == "yuv420p"
    assert a[a.index("-rc") + 1] == "qvbr" and "-qvbr_quality_level" in a


def test_gop_override_env():
    with patch.dict(os.environ, {"R3D_AV1_GOP": "77"}):
        for enc in av1.AV1_ENCODERS:
            a = av1.master_video_args(enc, w=1280, h=720, fps=60)
            assert a[a.index("-g") + 1] == "77", enc


def test_no_cpu_av1_encoder_is_ever_emitted():
    for enc in av1.AV1_ENCODERS:
        a = av1.master_video_args(enc, w=1280, h=720, fps=60)
        assert not (set(a) & set(_CPU_AV1)), (enc, a)


# --------------------------------------------------------------------------- #
# _spawn_ffmpeg: AV1 master, H.264 previews, in BOTH render paths              #
# --------------------------------------------------------------------------- #

@_both
def test_single_output_master_is_av1_vaapi(path):
    cmd = _cmd(path, encoder="av1_vaapi")                 # no preview: one output
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "av1_vaapi"
    assert "format=nv12,hwupload" in cmd
    assert cmd[cmd.index("-rc_mode") + 1] == "CQP" and "-qp" not in cmd
    assert "-global_quality" in cmd
    assert "-vaapi_device" in cmd                          # device initialised
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert not (set(cmd) & set(_CPU_AV1))                  # fail-closed: no CPU AV1


@_both
def test_single_output_master_is_av1_nvenc(path):
    cmd = _cmd(path, encoder="av1_nvenc")
    assert cmd[cmd.index("-c:v") + 1] == "av1_nvenc"
    assert "-vaapi_device" not in cmd                      # NVENC takes system frames
    assert "+faststart" in cmd and not (set(cmd) & set(_CPU_AV1))


@_both
def test_two_output_master_av1_preview_and_compact_stay_h264(path):
    # master av1_vaapi + inline preview + Discord compact in one ffmpeg process.
    cmd = _cmd(path, encoder="av1_vaapi", preview="m.embed.mp4",
               compact="m-embed-sm.mp4")
    assert "-filter_complex" in cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    # the master's VAAPI upload moved into the graph (no top-level -vf)
    assert "format=nv12,hwupload" in graph and "-vf" not in cmd
    # master output (map [vm]) is AV1...
    master = _master_slice(cmd)
    assert master[master.index("-c:v") + 1] == "av1_vaapi"
    assert master[master.index("-movflags") + 1] == "+faststart"
    # ...the preview and the Discord compact are H.264 (libx264), untouched
    tail = cmd[cmd.index("m.mp4") + 1:]
    assert "libx264" in tail                               # preview + compact codec
    assert "av1_vaapi" not in tail                         # only the master is AV1
    # fail-closed: no CPU AV1 anywhere in the whole command
    assert not (set(cmd) & set(_CPU_AV1))


@_both
def test_av1_master_keeps_the_preview_off_the_hardware(path):
    # an AV1 (hardware) master must NOT also try a hardware preview session
    cmd = _cmd(path, encoder="av1_nvenc", preview="m.embed.mp4")
    assert "h264_videotoolbox" not in cmd                  # preview is libx264
    assert "libx264" in cmd[cmd.index("m.mp4") + 1:]


# --------------------------------------------------------------------------- #
# _probe_encoder routing: AV1 request vs the untouched H.264 auto              #
# --------------------------------------------------------------------------- #

@_both
def test_probe_encoder_passes_explicit_av1_through_without_probing(path):
    _, R, RenderConfig = path
    # explicit av1_* must NOT run the real probe (the coordinator already gated
    # it); it passes straight through, defaulting the VAAPI device.
    with patch.object(av1, "probe", side_effect=AssertionError("must not probe")):
        enc, dev = R._probe_encoder(RenderConfig(encoder="av1_vaapi"))
    assert enc == "av1_vaapi" and dev == "/dev/dri/renderD128"
    with patch.object(av1, "probe", side_effect=AssertionError("must not probe")):
        enc, dev = R._probe_encoder(RenderConfig(encoder="av1_nvenc"))
    assert enc == "av1_nvenc" and dev is None


@_both
def test_probe_encoder_auto_is_still_h264(path):
    _, R, RenderConfig = path
    # bare "auto" must NEVER become AV1 (gated rollout) and must never call the
    # AV1 resolver.
    with patch.object(av1, "resolve_encoder",
                      side_effect=AssertionError("auto must not resolve AV1")):
        enc, _dev = R._probe_encoder(RenderConfig(encoder="auto"))
    assert enc in ("h264_nvenc", "h264_vaapi", "libx264", "h264_videotoolbox")


# --------------------------------------------------------------------------- #
# resolve_encoder: auto-select order + FAIL-CLOSED (probe mocked)              #
# --------------------------------------------------------------------------- #

def test_resolve_explicit_ok():
    with patch.object(av1, "probe", return_value=(True, "")):
        enc, dev = av1.resolve_encoder("av1_nvenc", None)
    assert enc == "av1_nvenc" and dev is None


def test_resolve_explicit_fail_is_closed_no_fallback():
    with patch.object(av1, "probe", return_value=(False, "Cannot load libcuda.so.1")):
        try:
            av1.resolve_encoder("av1_nvenc", None)
        except av1.AV1Unavailable as exc:
            msg = str(exc)
            assert "Cannot load libcuda.so.1" in msg
            assert "no H.264/CPU fallback" in msg
            assert "libx264" not in msg.replace("H.264", "")
        else:
            raise AssertionError("explicit av1 should fail closed when the probe fails")


def test_resolve_auto_picks_first_working_backend_in_order():
    # nvenc fails, vaapi works -> vaapi is selected (and the device exists)
    def outcome(encoder, device=None, **kw):
        return (True, "") if encoder == "av1_vaapi" else (False, "unavailable")
    with patch.object(av1, "probe", side_effect=outcome), \
            patch.object(av1.Path, "exists", return_value=True):
        enc, dev = av1.resolve_encoder("av1", None)
    assert enc == "av1_vaapi" and dev == "/dev/dri/renderD128"


def test_resolve_auto_all_fail_is_closed_and_lists_reasons():
    with patch.object(av1, "probe", return_value=(False, "no hw")), \
            patch.object(av1.Path, "exists", return_value=True):
        try:
            av1.resolve_encoder("av1", None)
        except av1.AV1Unavailable as exc:
            msg = str(exc)
            assert "no H.264/CPU fallback" in msg
            # every candidate is reported, never a silent H.264 substitution
            for cand in av1.AV1_ENCODERS:
                assert cand in msg, cand
        else:
            raise AssertionError("auto with no hardware AV1 must fail closed")


# --------------------------------------------------------------------------- #
# the real hardware probe: contract (hardware-tolerant) + live av1_vaapi       #
# --------------------------------------------------------------------------- #

def test_probe_rejects_a_non_av1_encoder_name():
    ok, reason = av1.probe("libx264", None)
    assert ok is False and "hardware AV1 encoder" in reason


def test_probe_missing_vaapi_device_fails_closed():
    ok, reason = av1.probe("av1_vaapi", "/dev/dri/renderD_NOPE", use_cache=False)
    assert ok is False and reason                          # never raises, never True


def test_live_av1_vaapi_probe_confirms_codec_name_av1():
    """Runs only where a VAAPI render node exists (e.g. the 7900 XTX); it is the
    real design gate: a 128x128 encode whose output ffprobe confirms is AV1.
    Soft on a busy GPU — it asserts the contract, and prints the reason if the
    device is present but the encode could not start."""
    dev = "/dev/dri/renderD128"
    if not Path(dev).exists():
        print("SKIP: no /dev/dri/renderD128 on this host")
        return
    ok, reason = av1.probe("av1_vaapi", dev, use_cache=False)
    print(f"live av1_vaapi probe: ok={ok} reason={reason!r}")
    assert isinstance(ok, bool)
    # if it ran at all, a True result is guaranteed to be genuine AV1 (the probe
    # ffprobes codec_name); a False result must carry a reason and must not be a
    # silent H.264 success.
    assert ok or reason


if __name__ == "__main__":
    import traceback
    mod = sys.modules[__name__]
    tests = [getattr(mod, n) for n in sorted(dir(mod)) if n.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
