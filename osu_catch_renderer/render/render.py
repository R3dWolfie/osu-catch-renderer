"""Phase 1 orchestrator: parse -> simulate -> per-frame GL draw + HUD -> ffmpeg.

Owns a small ffmpeg subprocess (raw rgba on stdin — RGBA zero-copy
pipeline 2026-08-28; the alpha byte is GL garbage and ffmpeg ignores it)
so it stays decoupled
from osu_renderer's encode FIFO machinery. HUD text is composited on the CPU
with PIL after GL readback — cheap and avoids a GL text pass for Phase 1.
"""
from __future__ import annotations

import hashlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
import pathlib

import numpy as np

from osu_catch_renderer.render._round8 import to8_clipped as _to8c
from PIL import Image, ImageDraw, ImageFont

from osu_catch_renderer.skin.assets import build_textures
from osu_catch_renderer.beatmap.beatmap import parse_beatmap
from osu_catch_renderer.render.death import (FAIL_FADE_MS, apply_death,
                                             apply_fail_audio,
                                             death_progress)
from osu_catch_renderer.render.flashlight import CatchFlashlight, has_flashlight
from osu_catch_renderer.render.gl import SpriteRenderer
from osu_catch_renderer import preview_hw as _phw
from osu_catch_renderer.render import loudnorm_cache
from osu_catch_renderer.beatmap.models import RenderConfig, ar_to_preempt_ms, ObjType
from osu_catch_renderer.beatmap.replay import parse_replay
from osu_catch_renderer.render.scene import CatchSim, mods_score_multiplier


class CatchRenderError(RuntimeError):
    pass


class _FrameWriter:
    """ffmpeg stdin writer thread — ported from the std renderer's proven
    FfmpegPipe (osu_std_renderer/record/encode.py), minus the process
    ownership (this renderer already owns its ffmpeg Popen).

    Frames are handed to the thread over a small bounded queue: the
    serialisation (`tobytes` — a negative-stride flip copy) and the blocking
    pipe write happen OFF the render thread, overlapping the next frame's
    draw. Order is FIFO so the byte stream ffmpeg sees is unchanged. The
    queue bounds memory (~4 frames) and provides natural backpressure when
    ffmpeg is the bottleneck; writer errors surface on the next push()
    instead of deadlocking the producer.

    R3D_FRAME_MD5=1 hashes every raw frame writer-side (blake2b) and prints
    one digest at close — bit-identical output proof across perf changes
    (same env/mechanism as the std renderer)."""

    # 12 frames of elasticity (~75 MB at 1080p): ffmpeg's ingest is fast on
    # average but stalls in bursts (loudnorm's 3 s blocks + muxer interleave);
    # a 4-deep queue let every stall block the render thread.
    _QUEUE_FRAMES = 12

    def __init__(self, proc, *, best_effort: bool = False,
                 queue_frames: "int | None" = None):
        self._stdin = proc.stdin
        # best_effort (the embed sidecar): a broken/stalled embed encoder must
        # NEVER fail or stall the master render — push() degrades to a no-op and
        # the node re-encodes as before. The master writer keeps the strict
        # raise-on-error contract.
        self._best_effort = best_effort
        self._q: "queue.Queue" = queue.Queue(
            maxsize=queue_frames or self._QUEUE_FRAMES)
        self._werr: BaseException | None = None
        self._hash = None
        self._hash_frames = 0
        # Only the master writer emits the frame-stream hash — the embed sidecar
        # sees the SAME frames, so a second digest would just be noise (and the
        # tee's whole point is that the master stream is unchanged).
        if os.environ.get("R3D_FRAME_MD5") and not best_effort:
            self._hash = hashlib.blake2b(digest_size=16)
        self._thread = threading.Thread(target=self._writer,
                                        name="ffmpeg-writer", daemon=True)
        self._thread.start()

    def _writer(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                return
            if self._werr is not None:
                continue          # drain (never write after an error)
            try:
                # PERF: a C-contiguous frame is written straight from its
                # buffer (memoryview) — no 6 MB tobytes copy per frame, and
                # no GIL-held memcpy stealing time from the render thread.
                # Non-contiguous frames (flipud views) keep the copy path.
                # Bytes on the pipe are identical either way.
                if isinstance(frame, np.ndarray) and frame.flags.c_contiguous:
                    data = memoryview(frame).cast("B")
                else:
                    data = frame.tobytes()
                if self._hash is not None:
                    self._hash.update(data)
                    self._hash_frames += 1
                self._stdin.write(data)
            except BaseException as e:  # noqa: BLE001 — surfaced on push()
                self._werr = e

    def push(self, frame_rgb) -> None:
        """Queue one frame. The master writer re-raises the writer thread's
        error, so a dead ffmpeg surfaces here just like the old synchronous
        write did (BrokenPipeError included). A best-effort (embed sidecar)
        writer NEVER raises and NEVER blocks the render indefinitely: on a
        writer error it drops silently, and on a full queue it waits only
        briefly before disabling itself (a stalled embed encoder must not pace
        the master render)."""
        if self._werr is not None:
            if self._best_effort:
                return
            raise self._werr
        if self._best_effort:
            # COPY before queueing: the embed sidecar lags the master and its
            # queue is far deeper than the render's readback ring, so it must NOT
            # retain a reference to a pooled/zero-copy readback buffer. That
            # buffer is reused within a few frames; the writer thread would then
            # serialize whatever OVERWROTE it, producing an out-of-order,
            # HUD-flickering companion (the scrambled Discord embed). A private
            # copy decouples the queued frame from ring reuse. Decimated => cheap.
            try:
                frame_rgb = frame_rgb.copy()
            except AttributeError:
                frame_rgb = bytes(frame_rgb)
            try:
                self._q.put(frame_rgb, timeout=5.0)
            except queue.Full:
                self._werr = RuntimeError("embed sidecar stalled; disabled")
                print("[catch] inline embed sidecar stalled -> disabled "
                      "(node will re-encode)", file=sys.stderr, flush=True)
            return
        self._q.put(frame_rgb)

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()
        if self._hash is not None:
            print(f"frame-stream-hash: {self._hash.hexdigest()} "
                  f"({self._hash_frames} frames)", file=sys.stderr, flush=True)


# HARNESS: R3D_CATCH_DUMP=dir,i0,i1,... saves the RAW composited gameplay frame
# (post-flashlight, post-HUD, pre-encode) so two runs can be diffed exactly.
_dump_env = os.environ.get("R3D_CATCH_DUMP")
_DUMP = None
if _dump_env:
    _dp = _dump_env.split(",")
    _DUMP = (_dp[0], {int(x) for x in _dp[1:]})


class _CompositeWorker:
    """HUD/results compositing pipeline stage (render thread -> here ->
    _FrameWriter). The render thread hands work items over a small bounded
    queue; this thread runs the flashlight pass, hud.overlay and the outro's
    draw_results in STRICT FIFO ORDER and pushes finished frames to the
    writer. hud.overlay / draw_results are deterministic functions of the
    scene snapshot + their own sequential state, and that state now lives on
    THIS thread only, so frame content and order are byte-identical to the
    old inline calls — the compositing simply overlaps the GL draw/readback
    of later frames (PIL/numpy release the GIL for their big ops).

    Queue depth bounds raw-readback lifetime: SpriteRenderer's host staging
    pool must exceed queue + in-process + 1 so a queued raw frame is never
    overwritten before this thread consumes it (see gl._HOST_POOL).

    Items: ("g", raw, scene)  gameplay frame -> flashlight + HUD -> writer
           ("r", fn, None)    outro frame    -> fn(last_gameplay) -> writer
           ("f", None, None)  frozen final gameplay frame re-push.

    On a FAILED play (``death_ms`` set) a death shade is applied to each
    gameplay frame AFTER the HUD, ramping over the ~1 s (``death_fade_ms`` of
    map time) ending at ``death_ms`` and held at its floor for the frozen tail.
    Passing plays pass ``death_ms=None`` and the death path never runs."""

    _QUEUE = 3

    def __init__(self, hud, writer, fl, perf=None, *, death_ms=None,
                 death_fade_ms=0.0, embed_writer=None, embed_stride=1):
        self._hud = hud
        self._writer = writer
        # Optional best-effort second sink: the 720p Discord-embed sidecar, fed
        # the SAME composited frames as the master so it is ready at render-done
        # (no post-render re-encode). None => unchanged single-output behaviour.
        # embed_stride decimates the sidecar to ~30 fps (every Nth frame) so the
        # second pipe carries far fewer frames and libx264 encodes a fraction of
        # the work -> the tee's cost on the master render stays small.
        self._embed_writer = embed_writer
        self._embed_stride = max(1, int(embed_stride))
        self._embed_i = 0
        self._fl = fl
        self._perf = perf
        self._death_ms = death_ms
        self._death_fade_ms = float(death_fade_ms)
        self.last_gameplay = None
        self._q: "queue.Queue" = queue.Queue(maxsize=self._QUEUE)
        self._werr: BaseException | None = None
        self._thread = threading.Thread(target=self._run,
                                        name="hud-composite", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        pc = time.perf_counter
        perf = self._perf
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._werr is not None:
                continue                      # drain (never emit after error)
            kind, a, b = item[0], item[1], item[2]
            # 4th element (gameplay only): the render thread already drew the
            # flashlight as a GL quad this frame, so do NOT run the CPU pass --
            # CatchFlashlight._size() is STATEFUL (an 800 ms ramp) and a second
            # call per frame corrupts it.
            fl_on_gpu = item[3] if len(item) > 3 else False
            try:
                if kind == "g":               # gameplay: flashlight + HUD
                    raw, scene = a, b
                    if self._fl is not None and not fl_on_gpu:
                        raw = self._fl.apply(raw, scene)
                    t0 = pc() if perf is not None else 0.0
                    out = self._hud.overlay(raw, scene)
                    if perf is not None:
                        perf["hud"] += pc() - t0
                    # FAIL death beat: on a failed play, ramp a desaturate +
                    # darken + red-tint shade over the whole composited frame
                    # (playfield AND HUD) across the final ~1 s ending at
                    # death, holding the floor for the frozen tail. Gated on
                    # death_ms, so passing renders never touch this.
                    if self._death_ms is not None:
                        p = death_progress(getattr(scene, "time_ms", 0),
                                           self._death_ms, self._death_fade_ms)
                        if p > 0.0:
                            out = apply_death(out, p)
                    self.last_gameplay = out
                    if _DUMP is not None:
                        _i = getattr(self, "_dump_n", 0)
                        self._dump_n = _i + 1
                        if _i in _DUMP[1]:
                            import numpy as _dnp
                            _dnp.save(f"{_DUMP[0]}/f{_i:06d}.npy", out)
                    self._emit(out)
                elif kind == "r":             # outro: results screen
                    if self.last_gameplay is None:
                        raise CatchRenderError(
                            "outro before any gameplay frame")
                    t0 = pc() if perf is not None else 0.0
                    out = a(self.last_gameplay)
                    if perf is not None:
                        perf["results"] += pc() - t0
                    self._emit(out)
                else:                         # "f": frozen gameplay frame
                    if self.last_gameplay is None:
                        raise CatchRenderError(
                            "outro before any gameplay frame")
                    self._emit(self.last_gameplay)
            except BaseException as e:  # noqa: BLE001 — surfaced on push()
                self._werr = e

    def _emit(self, frame) -> None:
        """Hand one finished frame to the master writer (strict) and, when the
        inline-embed sidecar is enabled, every Nth frame to its best-effort
        writer (decimated to ~30 fps). The master push happens FIRST and is
        byte-for-byte the old single-output behaviour; the embed push is
        decimated + can never raise (best-effort)."""
        self._writer.push(frame)
        if self._embed_writer is not None:
            if self._embed_i % self._embed_stride == 0:
                self._embed_writer.push(frame)
            self._embed_i += 1

    def push(self, item) -> None:
        """Queue one work item; re-raises this thread's error so a failed
        HUD/results pass (or a dead ffmpeg below it) fails the render loudly
        exactly like the old inline call did."""
        if self._werr is not None:
            raise self._werr
        self._q.put(item)

    def close(self) -> None:
        """Drain + join. Re-raises a pending worker error (except
        BrokenPipeError, which the caller surfaces via ffmpeg's exit code,
        matching the old inline flow)."""
        self._q.put(None)
        self._thread.join()
        if self._werr is not None and not isinstance(self._werr,
                                                     BrokenPipeError):
            raise self._werr


def render_catch(
    osr_path: Path,
    beatmap_dir: Path,
    output_path: Path,
    cfg: RenderConfig | None = None,
    *,
    progress_callback=None,
    overlay_osr=None,
    catcher_skins=None,
) -> Path:
    """Render `osr_path` over the beatmap in `beatmap_dir`. `overlay_osr` (a list
    of extra .osr paths) turns it into a versus OVERLAY: all replays race the
    same fruit stream on one field, catchers colour-coded per player.
    `catcher_skins` (aligned to [primary] + overlay_osr) gives each player their
    OWN skin's catcher; entries that are falsy/'-' fall back to the base skin."""
    cfg = cfg or RenderConfig()
    frames, meta = parse_replay(osr_path)
    overlay_extra = None
    if overlay_osr:
        overlay_extra = []
        for extra in overlay_osr:
            efr, emt = parse_replay(Path(extra))
            overlay_extra.append((efr, emt, getattr(emt, "player_name", "P")))
    osu_path = _find_osu(beatmap_dir, meta.beatmap_md5)
    bm = parse_beatmap(osu_path, mods=meta.mods)
    if not bm.objects:
        raise CatchRenderError(f"no hit objects parsed from {osu_path.name}")
    audio = bm.audio_filename and (beatmap_dir / bm.audio_filename)
    audio = audio if (audio and audio.is_file()) else None
    bg = bm.background and (beatmap_dir / bm.background)
    bg = bg if (bg and bg.is_file()) else None
    # replay md5 → so the results-screen leaderboard can exclude THIS render's
    # own DB row from the flanks (mirrors the std renderer).
    try:
        replay_md5 = hashlib.md5(Path(osr_path).read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001 — hashing never blocks a render
        replay_md5 = ""
    return render_core(bm, frames, meta, output_path, cfg, audio=audio, bg=bg,
                       progress_callback=progress_callback, osu_path=osu_path,
                       replay_md5=replay_md5, overlay_extra=overlay_extra,
                       catcher_skins=catcher_skins)


def _build_storyboard(cfg, renderer, osu_path, w, h):
    """Construct the storyboard renderer when --storyboard is on, else None.

    Gated on cfg.load_storyboard (DEFAULT OFF) — while off this returns None
    and the frame loop takes its exact single-draw path, so live renders are
    byte-identical. Auto-discovers the map's .osb next to the .osu. Fully
    fail-soft: any parse/build problem logs LOUDLY and renders without the
    storyboard rather than crashing."""
    if not getattr(cfg, "load_storyboard", False) or osu_path is None:
        return None
    try:
        from osu_catch_renderer.beatmap.storyboard import parse_storyboard
        from osu_catch_renderer.render.storyboard_engine import StoryboardEngine
        from osu_catch_renderer.render.storyboard_render import StoryboardRenderer
        sb_data = parse_storyboard(osu_path)
        engine = StoryboardEngine(sb_data)
        if not engine.sprites:
            print("[catch-renderer] storyboard: no drawable sprites — "
                  "rendering without storyboard", file=sys.stderr, flush=True)
            return None
        sbr = StoryboardRenderer(renderer, engine, Path(osu_path).parent,
                                 w, h, widescreen=sb_data.widescreen)
        c = sb_data.counts()
        print(f"[catch-renderer] storyboard: {len(engine.sprites)} drawable "
              f"sprites ({c['sprites']} sprite, {c['animations']} animation, "
              f"{c['videos']} video, {c['samples']} sample event(s) NOT played "
              f"— storyboard audio deferred), widescreen={sb_data.widescreen}",
              file=sys.stderr, flush=True)
        return sbr
    except Exception as e:  # noqa: BLE001 — a storyboard must never break a render
        import traceback
        print(f"[catch-renderer] WARNING: storyboard load failed ({e!r}) — "
              "rendering without storyboard", file=sys.stderr)
        traceback.print_exc()
        return None


def render_core(
    bm,
    frames,
    meta,
    output_path: Path,
    cfg: RenderConfig,
    *,
    audio: Path | None = None,
    bg: Path | None = None,
    progress_callback=None,
    osu_path: Path | None = None,
    replay_md5: str = "",
    overlay_extra=None,
    catcher_skins=None,
) -> Path:
    """Render from already-parsed beatmap/frames/meta. Shared by the osr path
    and tests.

    `overlay_extra` (list of (frames, meta, name)) turns this into a VERSUS
    OVERLAY: the primary player (frames/meta) plus these extra players' replays
    race the SAME fruit stream on one field. Each catcher is grayscaled +
    colour-coded per player. `catcher_skins` (aligned to [primary]+overlay_extra)
    gives each player their OWN skin's catcher; falsy/'-' → base skin catcher.
    The base playfield/fruits/HUD come from the base skin (`cfg.skin_dir`).
    Single renders leave overlay_extra None."""
    from osu_catch_renderer.hud.fonts import set_skin_font
    # prefer a font bundled in the skin, else a robust system font (must run
    # before the HUD builds its glyph/text fonts below).
    set_skin_font(cfg.skin_dir)

    skin = None
    if cfg.skin_dir is not None:
        from osu_catch_renderer.skin.skin import CatchSkin
        skin = CatchSkin(cfg.skin_dir, cfg.default_skin_dir)
    # Failed play: end the render at death instead of playing the unreached
    # remainder with a frozen catcher (which reads as phantom misses). Only
    # treat it as a fail if death lands meaningfully before the last object
    # — a life dip to 0 on the final note still effectively finished the map.
    last_obj = bm.objects[-1].time_ms
    death_ms = getattr(meta, "death_ms", None)
    _from_lifebar = getattr(meta, "death_from_lifebar", False)
    if death_ms is None:
        failed = False
    elif _from_lifebar:
        # Reliable stable HP-0: a life dip on the final note still finished the map.
        failed = death_ms < last_obj - 200
    else:
        # Lazer frame-timing fallback (no life bar): "death" is just where the
        # replay's INPUT stopped, which on a PASS lands before the last object
        # when the ending needs no catcher movement (bananas / held-still). The
        # header proves a clear DETERMINISTICALLY: osu!catch judges every fruit +
        # big droplet (caught OR missed), so if count_300+count_100+count_miss
        # covers (essentially) all generated FD, the player reached the end and
        # it is NOT a fail. Only a genuine early death leaves FD objects unjudged.
        # Replaces the old <0.85*last heuristic (which could truncate a clear that
        # ends >15% early -> render a fabricated FC) and matches the guard in
        # versus_telemetry. (Bugs 2026-08-16 ManuAoK lazer S; 2026-08-26 Veeti
        # fabricated FC on the snap he missed.)
        _total_fd = sum(1 for o in bm.objects
                        if o.kind in (ObjType.FRUIT, ObjType.DROPLET))
        _hdr_fd = int(meta.count_300) + int(meta.count_100) + int(meta.count_miss)
        _cleared = _hdr_fd >= _total_fd - 4
        failed = (death_ms < last_obj - 200) and not _cleared
    sim_end_ms = int(death_ms) if failed else None
    sim = CatchSim(bm, frames, cfg, skin=skin, has_bg=bg is not None,
                   meta=meta, end_ms=sim_end_ms)
    # the PRIMARY player's sim — hitsounds come from ITS caught objects even
    # in a versus overlay (sim is rebound to CatchOverlaySim below).
    base_sim = sim
    _overlay_gray_keys = set()
    _player_catcher_bakes = []      # [(texture_key, rgba)] grayscaled + uploaded below
    if overlay_extra:
        from osu_catch_renderer.render.overlay import CatchOverlaySim
        from osu_catch_renderer.skin.skin import CatchSkin
        extra_sims = [CatchSim(bm, fr, cfg, skin=skin, has_bg=bg is not None,
                               meta=mt, end_ms=None)
                      for (fr, mt, _n) in overlay_extra]
        # BASE skin fruit sprites (incl. the base catcher) get a grayscale "__ovl"
        # copy so the caught fruits + any base/unlinked catcher recolour cleanly.
        _overlay_gray_keys = {k for k in (skin.textures if skin else ())
                              if isinstance(k, str) and k.startswith("fruit")}
        # PER-PLAYER catcher (platter): each player's OWN skin's catcher art,
        # resolved by the SAME rules the base skin uses (CatchSkin: idle vs
        # ryuuta by skin version, @2x preference, per-file fallback to the
        # DEFAULT skin when the player's skin ships no catcher), grayscaled to
        # its own key so the per-player hue tint recolours THEIR art. Players
        # without a preset ('' / '-' / missing dir / unresolvable skin) fall
        # back to the base catcher gray. Distinct dirs that resolve to the SAME
        # catcher file (shared skin, or both falling through to the default)
        # share ONE texture; art identical to the base skin's reuses ITS gray.
        # `catcher_skins` aligns to [primary] + overlay_extra.
        _base_ck = ((getattr(skin, "catcher_key", None) if skin else None)
                    or "fruit-catcher-idle")
        _base_gray = f"{_base_ck}__ovl"
        _base_aspect = skin.catcher_aspect if skin is not None else 324 / 305
        _base_src = (str(skin._resolve(skin.catcher_key))
                     if skin is not None and skin.catcher_key else None)
        catcher_keys = []
        catcher_aspects = []            # player art h/w ÷ base art h/w
        _src_seen: dict[str, tuple[str, float]] = {}
        for i in range(1 + len(overlay_extra)):
            sd = (catcher_skins[i] if catcher_skins and i < len(catcher_skins)
                  else None)
            sd = str(sd) if sd and str(sd) not in ("", "-") else None
            entry = None                # (texture_key, aspect_ratio_vs_base)
            if sd and Path(sd).is_dir():
                try:
                    _csk = CatchSkin(Path(sd), cfg.default_skin_dir)
                    _ck = getattr(_csk, "catcher_key", None)
                    ctex = _csk.textures.get(_ck) if _ck else None
                    _src = str(_csk._resolve(_ck)) if ctex is not None else None
                    if _src is not None:
                        if _src == _base_src:           # same art as the base
                            entry = (_base_gray, 1.0)
                        elif _src in _src_seen:         # shared player skin
                            entry = _src_seen[_src]
                        else:
                            key = f"fruit-catcher-idle__ovl_p{i}"
                            _player_catcher_bakes.append((key, ctex))
                            entry = (key, _csk.catcher_aspect / _base_aspect)
                            _src_seen[_src] = entry
                except Exception:      # noqa: BLE001 — bad skin → base catcher
                    entry = None
            if entry is None:
                entry = (_base_gray, 1.0)
            catcher_keys.append(entry[0])
            catcher_aspects.append(entry[1])
        sim = CatchOverlaySim(
            [sim] + extra_sims,
            [getattr(meta, "player_name", "P1")]
            + [n for (_f, _m, n) in overlay_extra],
            gray_keys=_overlay_gray_keys, catcher_keys=catcher_keys,
            catcher_aspects=catcher_aspects)

    # ── SCORE FIDELITY: one lazer-standardised scale everywhere ─────────────
    # The .osr header total means different things per source (stable ScoreV1,
    # osu-web legacy export of a lazer play, lazer classic display, lazer
    # standardised). score_fidelity converts the header under every
    # interpretation with lazer's own math and picks the one consistent with
    # our sim; the sim's curve is then END-PINNED (std honesty pattern) so the
    # in-video counter ENDS EXACTLY on that number, and meta.score is swapped
    # so the results screen + leaderboard card show the same value. The
    # authoritative total is exported via a `<output>.mp4.score.json` sidecar
    # for the bot (renders.score_v3 → website card). Fail-soft: any problem
    # leaves the sim un-pinned and the header score displayed as before.
    score_fid: dict | None = None
    if osu_path is not None:
        try:
            from osu_catch_renderer.beatmap.score_fidelity import (compute_candidates,
                                         resolve_authoritative)

            def _pin(one_sim, one_meta):
                final = (one_sim._checkpoints[-1].score
                         if one_sim._checkpoints else 0)
                fid = compute_candidates(
                    one_meta, bm.objects, osu_path,
                    mods_score_multiplier(getattr(one_meta, "mods", 0) or 0))
                val, src = resolve_authoritative(fid, final)
                if final > 0 and val > 0:
                    one_sim.score_scale = val / final
                fid.pop("legacy_attrs", None)
                fid.pop("osu_facts", None)
                fid.update({"score_v3": int(val), "source": src,
                            "sim_final": int(final),
                            "player": getattr(one_meta, "player_name", "")})
                return fid

            score_fid = _pin(base_sim, meta)
            score_fid["players"] = [dict(score_fid)]
            if overlay_extra:
                for _es, (_f, _mt, _n) in zip(extra_sims, overlay_extra):
                    _pf = _pin(_es, _mt)
                    _pf["player"] = _pf["player"] or _n
                    score_fid["players"].append(_pf)
            import dataclasses as _dc
            import sys as _sfsys
            print(f"[catch] score fidelity: header={meta.score:,} -> "
                  f"standardised {score_fid['score_v3']:,} "
                  f"(source={score_fid['source']}, "
                  f"sim_final={score_fid['sim_final']:,})",
                  file=_sfsys.stderr, flush=True)
            meta = _dc.replace(meta, score=int(score_fid["score_v3"]))
        except Exception as _sf_e:  # noqa: BLE001 — never break a render
            import sys as _sfsys
            print(f"[catch] score fidelity FAILED (header score kept): "
                  f"{_sf_e}", file=_sfsys.stderr, flush=True)
            score_fid = None
    if cfg.show_pp_counter and osu_path is not None:
        sim.compute_pp_curve(osu_path, meta.mods)
    preempt = ar_to_preempt_ms(bm.ar)
    first = bm.objects[0].time_ms
    last = min(last_obj, int(death_ms)) if failed else last_obj
    # skip_intro: start at the first object's approach; else render the full
    # intro from the song start.
    if cfg.skip_intro:
        start_ms = int(first - preempt - cfg.lead_in_ms)
    else:
        start_ms = min(0, int(first - preempt - cfg.lead_in_ms))
    # intro R3D splash window opens at the render's first frame (no seizure
    # card in catch, so it begins immediately -- std offsets by the seizure
    # duration). The sim fades it out at the first fruit's approach.
    sim.logo_start_ms = start_ms if cfg.show_logo else None
    gameplay_end_ms = int(last + cfg.tail_ms)
    # results outro (matches osu_renderer: 800ms gap, then the card) — on by default
    RESULTS_GAP_MS, FADE_MS = 800, 400
    if cfg.show_results:
        results_start_ms = gameplay_end_ms + RESULTS_GAP_MS
        total_end_ms = results_start_ms + cfg.results_ms
    else:
        results_start_ms = total_end_ms = gameplay_end_ms
    # DT/HT playback: the simulation lives on the map-time axis, but a DT play
    # should *look* 1.5x faster. So gameplay frames advance map-time by
    # frame_ms*rate per output frame (fewer frames at the same fps => sped up),
    # and the audio is atempo'd by the same rate. The results outro stays
    # real-time for cross-mode consistency with the mania renderer.
    rate = getattr(bm, "rate", 1.0) or 1.0
    frame_ms = 1000.0 / cfg.fps
    map_step = frame_ms * rate
    # key-overlay input aggregation window = exactly one output frame's span
    # of map time (rate-aware, so DT/HT taps neither smear nor vanish).
    sim.video_step_ms = map_step
    gameplay_frames = max(1, int((gameplay_end_ms - start_ms) / map_step))
    outro_frames = max(0, int((total_end_ms - gameplay_end_ms) / frame_ms)) if cfg.show_results else 0
    n_frames = gameplay_frames + outro_frames

    w, h = cfg.resolution
    renderer = SpriteRenderer(w, h)
    if skin is not None:
        for key, rgba in skin.textures.items():
            renderer.upload_texture(key, rgba)
    else:
        for key, rgba in build_textures().items():
            renderer.upload_texture(key, rgba)
    # OVERLAY: grayscale every base skin fruit sprite (std _whiten_skin_cursor
    # method) so the caught fruits recolour cleanly by a colour multiply — keeps
    # the shape, drops the hue. Plus each player's OWN catcher (its own key).
    def _gray(rgba):
        r = np.asarray(rgba).astype(np.float32)
        lum = 0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2]
        return _to8c(np.clip(np.stack([lum, lum, lum, r[..., 3]], axis=-1),
                             0, 255))
    for key in _overlay_gray_keys:
        renderer.upload_texture(f"{key}__ovl", _gray(skin.textures[key]))
    for key, ctex in _player_catcher_bakes:
        renderer.upload_texture(key, _gray(ctex))
    # osu!lazer ARGON catch objects (glowing wavy combo rings + white pip) and
    # the Argon catcher bar — uploaded regardless of skin: the skinless object
    # path, the caught-fruit plate pile, and the hit explosions all use them.
    from osu_catch_renderer.skin.assets import (build_argon_textures, catch_glow_rgba, catch_beam_rgba,
                         bake_logo_tile)
    for key, rgba in build_argon_textures().items():
        renderer.upload_texture(key, rgba)
    from osu_catch_renderer.skin.lazer_skin import argon_bar_cap_rgba
    renderer.upload_texture("argon_bar_cap", argon_bar_cap_rgba())
    renderer.upload_texture("catch_glow", catch_glow_rgba())
    renderer.upload_texture("catch_beam", catch_beam_rgba())
    renderer.upload_texture("logo_tile", bake_logo_tile())
    if bg is not None:
        _bg_tex = _bg_cover(bg, w, h, cfg.bg_blur)
        if _bg_tex is not None:
            renderer.upload_texture("bg", _bg_tex)

    # Storyboard renderer (phase 4/5): constructed only when --storyboard is on
    # (see _build_storyboard). While None, the frame loop takes the exact
    # single-draw path it always has, so live renders are byte-identical.
    storyboard = _build_storyboard(cfg, renderer, osu_path, w, h)

    total_dur_s = n_frames / cfg.fps
    # Caught-object hitsounds (stable behaviour; default ON): pre-mix every
    # caught object's samples into a wall-time WAV; the encode amixes it on
    # top of the loudnormed song (see hitsounds.py for the lazer semantics
    # + the mania v2 loudnorm-duck fix this mirrors). Fully fail-soft — any
    # problem leaves the song-only chain (renders unchanged).
    hits_wav = None
    # ModNightcore beat overlay is AUTOMATIC when the NC mod (bit 512) is on.
    is_nc = bool(int(getattr(meta, "mods", 0) or 0) & 512)   # Nightcore bit
    if audio is not None and (getattr(cfg, "hitsounds", True)
                              or getattr(cfg, "nightcore_hitsounds", False)
                              or is_nc):
        try:
            from osu_catch_renderer.beatmap.hitsounds import build_hitsound_track, synth_style_for
            objs, caught_flags = base_sim.catch_events()
            skin_dirs = skin.dirs if skin is not None else []
            has_custom = (skin is not None
                          and getattr(skin, "_user_skin_dir", None) is not None)
            bdir = osu_path.parent if osu_path is not None else audio.parent
            # "Use the beatmap's hitsounds" OFF: hand the bank no beatmap
            # dir at all — custom samples + filename overrides vanish and
            # every event resolves via skin chain -> synth.
            if not getattr(cfg, "beatmap_hitsounds", True):
                bdir = None
            hits_wav = build_hitsound_track(
                objs, caught_flags, bm,
                beatmap_dir=bdir, skin_dirs=skin_dirs,
                out_wav=output_path.with_suffix(".hits.wav"),
                start_ms=start_ms, rate=rate,
                duration_ms=total_dur_s * 1000.0,
                synth_style=synth_style_for(has_custom),
                nightcore=getattr(cfg, "nightcore_hitsounds", False),
                nc_mod=is_nc,
                hitsounds_on=getattr(cfg, "hitsounds", True),
                # beat overlays stop at gameplay end, not into results (taiko ac73af2)
                gameplay_end_ms=float(gameplay_end_ms))
        except Exception as e:  # noqa: BLE001 — hitsounds never break a render
            print(f"[catch-renderer] hitsounds skipped: {e}", file=sys.stderr)
            hits_wav = None
    # INLINE PREVIEW (R3D_PREVIEW_INLINE=1, default OFF): have the SAME ffmpeg
    # that encodes the master also write the lean 720p30 preview embed as a
    # second output, so it is finished the moment the render is. Without it the
    # node re-encodes the finished master afterwards (fleet medians 10-270 s)
    # before anything can be published. Single renders only.
    preview_path = None
    if os.environ.get("R3D_PREVIEW_INLINE") == "1" and not overlay_extra:
        preview_path = output_path.parent / (output_path.stem + ".embed.mp4")
        print(f"[catch] inline preview -> {preview_path.name}",
              file=sys.stderr, flush=True)
    # A failed play gets its audio rewritten after the render
    # (apply_fail_audio), so that file is not append-only: no streaming, no
    # marker, and the client runs its normal post-render pass.
    # INLINE DISCORD COPY (R3D_COMPACT_INLINE=1, default OFF; needs the inline
    # preview): a THIRD output of the same ffmpeg, encoded to the compact plan
    # the node/bot use for `-embed-sm.mp4`, so nothing is left to encode after
    # the render. Not on failed plays: their audio is rewritten afterwards and
    # the copy would not carry it.
    compact_path = None
    if (preview_path is not None and not overlay_extra and not failed
            and _compact_wanted(total_dur_s, cfg.resolution[0], cfg.resolution[1],
                                cfg.fps, 0.35)):
        compact_path = output_path.parent / (output_path.stem + ".embed-sm.mp4")
        print(f"[catch] inline discord copy -> {compact_path.name}",
              file=sys.stderr, flush=True)
    stream_master = _stream_master_requested(overlay_extra) and not failed
    if stream_master:
        _write_stream_marker(output_path, compact=compact_path is not None)
        print("[catch] streamable master (no faststart, loudnorm in-engine)",
              file=sys.stderr, flush=True)
    proc = _spawn_ffmpeg(cfg, output_path, audio, start_ms, rate, total_dur_s,
                         hitsound_wav=hits_wav, is_nc=is_nc,
                         preview_path=preview_path,
                         stream_master=stream_master,
                         compact_path=compact_path)
    # Argon is the DEFAULT skin: skinless renders stay all-Argon (parity with
    # the STD renderer). DanserHud now handles skin_dir=None; plain _Hud only if
    # DanserHud fails to build.
    try:
        from osu_catch_renderer.hud.hud import DanserHud
        hud = DanserHud(cfg.skin_dir, cfg.resolution, meta, bm, first, last, cfg=cfg,
                        default_skin_dir=cfg.default_skin_dir)
    except Exception:
        hud = _Hud(w, h, meta, bm)

    from osu_catch_renderer.hud.hud import draw_results
    # results-screen map leaderboard (parity with std): build + bake ONCE, up
    # front, so the outro just composites the pre-baked cards each frame. Fully
    # fail-soft — any problem leaves the plain results card (renders unchanged).
    baked_board = None
    if cfg.show_results and getattr(cfg, "show_leaderboard", True):
        try:
            from osu_catch_renderer.hud.lb_cards import build_catch_board
            baked_board = build_catch_board(cfg, meta, bm, replay_md5)
        except Exception as e:  # noqa: BLE001 — a board must never break a render
            # aliased: a bare `import sys` here would make `sys` a LOCAL of the
            # whole render_core, breaking every other `sys.stderr` in the function
            # (incl. the inline-embed logs) with UnboundLocalError.
            import sys as _lbsys
            print(f"[catch-renderer] leaderboard skipped: {e}", file=_lbsys.stderr)
            baked_board = None
    # Async pipeline (ported from the std renderer's proven design):
    #   * GPU readback goes through a 3-deep PBO ring (read_rgb_async returns
    #     None while the ring fills; frames pop out ~2 frames late, in strict
    #     submission order; read_drain() flushes the tail).
    #   * HUD compositing is deferred until a frame's pixels pop out of the
    #     ring: the scene snapshot is queued alongside, and hud.overlay is a
    #     deterministic function of it — called once per frame, in frame
    #     order, exactly as the synchronous path did.
    #   * Flashlight + HUD compositing + the outro's results screen run on a
    #     dedicated composite thread in strict FIFO order (_CompositeWorker),
    #     overlapping the GL draw/readback of later frames.
    #   * The ffmpeg pipe write happens on a writer thread behind a small
    #     bounded queue (_FrameWriter), fed only by the composite thread.
    # Frame count, order and bytes are identical to the synchronous path.
    _t_render0 = time.monotonic()
    _PERF = os.environ.get("R3D_CATCH_PERF")
    _pt = {"scene": 0.0, "draw": 0.0, "read": 0.0, "hud": 0.0,
           "results": 0.0, "enq": 0.0}
    _pc = time.perf_counter
    writer = _FrameWriter(proc)
    # INLINE EMBED SIDECAR (R3D_EMBED_INLINE=1, default OFF): produce the 720p
    # Discord embed DURING the render from the same composited frames, so it is
    # ready at render-done and the node skips its post-render re-encode pass.
    # Single renders only (overlays are multi-catcher). Fully best-effort — any
    # spawn/encode failure leaves no sidecar and the node re-encodes as before.
    # The master output is untouched (byte-identical frame stream).
    embed_path = None
    embed_proc = None
    embed_writer = None
    embed_stride = 1
    if os.environ.get("R3D_EMBED_INLINE") == "1" and not overlay_extra:
        try:
            # decimate the sidecar to the plan's fps (60 keeps every frame; 30
            # takes every Nth) so the companion encoder never does more temporal
            # work than the target companion needs.
            _plan_fps = _embed_compact_plan(total_dur_s)[3]
            embed_stride = max(1, round(float(cfg.fps) / float(_plan_fps)))
            embed_path = output_path.parent / (output_path.stem + ".embed-sm.mp4")
            embed_proc = _spawn_embed_ffmpeg(
                cfg, embed_path, audio, start_ms, rate, total_dur_s,
                hitsound_wav=hits_wav, is_nc=is_nc,
                in_fps=float(cfg.fps) / embed_stride)
            embed_writer = _FrameWriter(embed_proc, best_effort=True,
                                        queue_frames=24)
            print(f"[catch] inline embed sidecar -> {embed_path.name}",
                  file=sys.stderr, flush=True)
        except Exception as _ee:  # noqa: BLE001 — sidecar must never break a render
            print(f"[catch] inline embed spawn failed ({_ee}); node re-encodes",
                  file=sys.stderr, flush=True)
            embed_path = embed_proc = embed_writer = None
    pending = deque()          # scene snapshots awaiting their pixels

    # osu!catch Flashlight (FL, mod bit 1<<10): a soft-edged black vignette
    # centred on the catcher plate that shrinks with combo (see flashlight.py for
    # the ported lazer values). Post-pass over the composited playfield BEFORE the
    # HUD draws, so score/acc/combo/break overlays stay lit — lazer keeps the
    # Flashlight in the playfield layer with the HUD above it. Single renders only
    # (a versus overlay has many catchers); strictly gated on the FL bit, so
    # non-FL replays render byte-identically.
    fl = None
    # R3D_CATCH_GPU_FL: draw the flashlight as ONE multiply-blend quad inside the
    # main GL pass instead of a CPU numpy post-pass on the composite thread.
    # Z-order is preserved: the HUD is still composited on the CPU after readback,
    # so it stays ABOVE the flashlight exactly as lazer has it.
    # DEFAULT ON (ship as intended): GPU flashlight quad. Escape hatch:
    # R3D_CATCH_GPU_FL=0 forces the old CPU post-pass.
    _CGPU_FL = os.environ.get("R3D_CATCH_GPU_FL", "1") != "0"
    if has_flashlight(getattr(meta, "mods", 0)) and not overlay_extra:
        fl = CatchFlashlight(break_env=getattr(sim, "_break_env", None))

    # compositing pipeline stage: flashlight + HUD + results run on their own
    # thread (strict FIFO), overlapping the GL draw/readback of later frames.
    # FAIL death beat (catch only): scale the ~2.5 s osu fail ramp (FAIL_FADE_MS)
    # by playback rate so a DT/HT fail still reads ~2.5 s of VIDEO. Only when `failed`.
    _death_arg = float(death_ms) if failed else None
    _death_fade = FAIL_FADE_MS * rate if failed else 0.0
    comp = _CompositeWorker(hud, writer, fl, perf=_pt if _PERF else None,
                            death_ms=_death_arg, death_fade_ms=_death_fade,
                            embed_writer=embed_writer, embed_stride=embed_stride)

    # Loop-INVARIANT flashlight-on-GPU decision (reconciled onto embed-tee):
    # a constant, not a per-frame local, because _emit runs ~2 frames behind the
    # PBO ring; the worker safely defaults off for a 3-tuple push.
    _FL_ON_GPU = (fl is not None) and _CGPU_FL

    def _emit_gameplay(raw):
        scene = pending.popleft()
        _t0 = _pc()
        comp.push(("g", raw, scene, _FL_ON_GPU))
        _pt["enq"] += _pc() - _t0

    try:
        try:
            for i in range(n_frames):
                if i < gameplay_frames:
                    t = int(start_ms + i * map_step)
                    _t0 = _pc()
                    scene = sim.build_scene(t)
                    _t1 = _pc(); _pt["scene"] += _t1 - _t0
                    renderer.begin()
                    if storyboard is None:
                        # exact single-draw path (byte-identical to pre-SB)
                        renderer.draw(scene.sprites)
                    else:
                        # interleave the two storyboard z-slices around the
                        # playfield: bg image -> SB underlay (Background/Fail/
                        # Pass/Foreground) -> playfield sprites -> SB overlay
                        # (Overlay layer). catch's flashlight/HUD/results are
                        # CPU-composited after readback, so the whole GL pass
                        # sits under them — the SB Overlay lands over gameplay,
                        # under the HUD, as in lazer. Both slices share the bg
                        # dim (scene.sb_brightness).
                        n = scene.bg_split
                        b = scene.sb_brightness
                        if n:
                            renderer.draw(scene.sprites[:n])
                        storyboard.draw_underlay(t, b)
                        renderer.draw(scene.sprites[n:])
                        storyboard.draw_overlay(t, b)
                    _t2 = _pc(); _pt["draw"] += _t2 - _t1
                    pending.append(scene)
                    # flashlight BEFORE readback and before the CPU HUD, so the
                    # HUD lands on top. gl_params advances the 800 ms ramp ONCE.
                    if _FL_ON_GPU:
                        _flp = fl.gl_params(scene)
                        if _flp is not None:
                            renderer.draw_flashlight(*_flp)
                    raw = renderer.read_rgb_async()
                    _pt["read"] += _pc() - _t2
                    if raw is not None:
                        _emit_gameplay(raw)
                else:
                    # gameplay -> outro boundary: flush the PBO ring first so
                    # last_gameplay is the true final gameplay frame and
                    # ordering is preserved across the boundary.
                    for raw in renderer.read_drain():
                        _emit_gameplay(raw)
                    # outro: frozen final gameplay frame, then the results card
                    # fades in (consistent with the mania renderer). Real-time.
                    # (No .copy(): draw_results/render_frame never mutate their
                    # input — they fromarray-copy — and the writer only reads,
                    # so the frozen frame can be pushed by reference. PERF.)
                    t = int(gameplay_end_ms + (i - gameplay_frames) * frame_ms)
                    if cfg.show_results and t >= results_start_ms:
                        op = min(1.0, (t - results_start_ms) / FADE_MS)
                        age = float(t - results_start_ms)

                        # age_ms drives the lazer results screen's two-stage
                        # animation (arc sweep / grade punch / score roll /
                        # card slide-in, then the stage-2 stats panels
                        # unfolding from the right); osu_path lets it compute
                        # stars + pp (rosu); sim feeds the stage-2 COMBO panel
                        # its checkpoint series. Runs on the composite thread
                        # over ITS frozen final gameplay frame (identical to
                        # the old inline call: same args, same FIFO position).
                        def _results_frame(lg, op=op, age=age):
                            return draw_results(
                                lg, meta, bm, op, board=baked_board,
                                age_ms=age, osu_path=osu_path, sim=sim,
                                pp_override=cfg.pp_override,
                                sr_override=cfg.sr_override)

                        comp.push(("r", _results_frame, None))
                    else:
                        comp.push(("f", None, None))
                if progress_callback and i % cfg.fps == 0:
                    progress_callback(int(i / n_frames * 100))
            # map end with no outro configured: flush the ring tail.
            for raw in renderer.read_drain():
                _emit_gameplay(raw)
        except BrokenPipeError:
            pass               # ffmpeg died — surfaced via ret below
    finally:
        # composite errors are re-raised AFTER the ffmpeg/GL cleanup below —
        # raising here would leak the ffmpeg child + GL context.
        _comp_err = None
        try:
            comp.close()
        except BaseException as e:  # noqa: BLE001 — deferred, never swallowed
            _comp_err = e
        writer.close()
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        ret = proc.wait()
        # MASTER render time — measured at master completion, BEFORE the embed
        # finalize, so the reported render speed reflects the render alone.
        _render_wall = time.monotonic() - _t_render0
        # inline embed sidecar: drain + finalize AFTER the master (a stalled
        # embed can never delay the master's completion). Best-effort — any
        # failure just leaves no sidecar and the node re-encodes as before.
        if embed_writer is not None or embed_proc is not None:
            _embed_t0 = time.monotonic()
            if embed_writer is not None:
                try:
                    embed_writer.close()
                except BaseException:  # noqa: BLE001 — never fails a render
                    pass
            _eret = -1
            if embed_proc is not None:
                try:
                    if embed_proc.stdin:
                        try:
                            embed_proc.stdin.close()
                        except BrokenPipeError:
                            pass
                    _eret = embed_proc.wait(timeout=180)
                except Exception:  # noqa: BLE001
                    try:
                        embed_proc.kill()
                        embed_proc.wait(timeout=10)
                    except Exception:  # noqa: BLE001
                        pass
            _embed_ok = (_eret == 0 and embed_path is not None
                         and embed_path.exists()
                         and embed_path.stat().st_size > 8_000)
            _embed_tail = time.monotonic() - _embed_t0
            print("[catch] inline embed sidecar "
                  f"{'READY' if _embed_ok else 'FAILED'} "
                  f"({embed_path.name if embed_path else '?'}; "
                  f"finalize tail {_embed_tail:.1f}s)",
                  file=sys.stderr, flush=True)
        renderer.release()
        # drop the temp hitsound WAV (R3D_CATCH_KEEP_HITS=1 keeps it for
        # alignment debugging/verification)
        if hits_wav is not None and not os.environ.get("R3D_CATCH_KEEP_HITS"):
            try:
                Path(hits_wav).unlink(missing_ok=True)
            except OSError:
                pass
        import sys as _rsys
        # render-only wall (excludes the embed finalize tail) so the bot-parsed
        # "done:" speed reflects the render, not the sidecar.
        _wall = _render_wall
        if _PERF:
            import sys as _psys
            print("PERF " + " ".join(f"{k}={v:.2f}s" for k, v in _pt.items()),
                  file=_psys.stderr, flush=True)
        print(f"done: {n_frames} frames in {_wall:.1f}s "
              f"({(n_frames / _wall) if _wall else 0.0:.1f} fps) ret={ret}",
              file=_rsys.stderr, flush=True)
        if storyboard is not None:
            try:
                st = storyboard.stats()
                print(f"storyboard cache: {st['uploads']} uploads, "
                      f"{st['evictions']} evictions, {st['peak_mb']:.0f} MB "
                      f"peak, {st['resident']} resident",
                      file=_rsys.stderr, flush=True)
            except Exception:  # noqa: BLE001 — stats print never breaks a render
                pass
        if _comp_err is not None:
            raise _comp_err

    if ret != 0:
        tail = ""
        errlog = getattr(proc, "_catch_errlog", None)
        if errlog and Path(errlog).exists():
            tail = Path(errlog).read_text(errors="replace")[-800:]
        # a hardware preview that failed must not fail the NEXT render too
        if _phw.note_preview_failure(list(getattr(proc, "args", []) or []),
                                     tail.encode("utf-8", "replace")):
            tail += ("\n[the preview's hardware encoder is now off for 24 h on "
                     "this node; the next render uses the CPU preview]")
        raise CatchRenderError(f"ffmpeg exited {ret}\n{tail}")
    if not output_path.exists() or output_path.stat().st_size < 8_000:
        raise CatchRenderError("output too small / missing — render likely failed")
    # FAIL audio grind-to-halt (catch only): on a failed play, ramp the muxed
    # audio's final ~FAIL_FADE_MS before death to a slowing, pitch-dropping,
    # low-passed stop (osu!'s track freq 1->0), then silence the frozen tail.
    # Isolated decode->warp->remux post-pass, fully fail-soft; gated on `failed`
    # so passing renders keep byte-identical audio.
    if failed:
        import sys as _fa_sys
        _death_video_s = (int(death_ms) - start_ms) / rate / 1000.0
        if apply_fail_audio(output_path, _death_video_s, FAIL_FADE_MS / 1000.0):
            print(f"[catch] fail-audio grind applied (death @ {_death_video_s:.2f}s "
                  f"video, window {FAIL_FADE_MS/1000.0:.2f}s)",
                  file=_fa_sys.stderr, flush=True)
    # score-fidelity sidecar: `<output>.score.json` next to the mp4 — the bot
    # (cli/r3d_render.py) reads it into the completion marker so the website
    # card stores/displays the SAME standardised total the counter ended on.
    if score_fid is not None:
        try:
            import json as _json
            sidecar = Path(str(output_path) + ".score.json")
            # Gameplay-start anchor for the YT versus HUD (all-mode sync):
            # video-seconds into THIS panel where map-time 0 lands (frame 0 is
            # map-time start_ms), plus the rate-mods speed.
            _map0_video_s = round((0 - start_ms) / (rate * 1000.0), 6)
            sidecar.write_text(_json.dumps(
                {"schema": 1, "mode": 2,
                 "map0_video_s": _map0_video_s, "rate": float(rate),
                 **score_fid}, default=str))
        except Exception as _sc_e:  # noqa: BLE001 — sidecar is best-effort
            print(f"[catch] score sidecar write failed: {_sc_e}",
                  file=sys.stderr, flush=True)

    # dash sidecar: `<output>.dash.json` — per-player dash timeline for the YT
    # versus overlay, which parses replays with osrparse directly and so reads
    # `dashing=False` for every frame on any replay whose Left1 bit shares its
    # byte with another button (e.g. Smoke=16 -> ButtonState 17; osrparse's
    # exact `==1` compare fails). The renderer already recovers dash (raw Left1
    # bit mask, or velocity reconstruction — see replay.py), so we export the
    # authoritative per-player dash so the overlay consumes it instead of its
    # own broken osrparse count. Opt-in via R3D_CATCH_DASH_SIDECAR (default off,
    # so every existing render is byte-identical); fully fail-soft.
    if os.environ.get("R3D_CATCH_DASH_SIDECAR"):
        try:
            import json as _json

            def _dash_runs(fr):
                """(runs, edges, dash_frames): dash intervals [start_ms,end_ms]
                in MAP time; edges = rising-edge (dash-press) count = len(runs);
                dash_frames = frames with dash held."""
                runs = []
                start = None
                held = 0
                for f in fr:
                    if f.dashing:
                        held += 1
                        if start is None:
                            start = f.time_ms
                    elif start is not None:
                        runs.append([start, prev])
                        start = None
                    prev = f.time_ms
                if start is not None:
                    runs.append([start, prev])
                return runs, len(runs), held

            # (name, frames, dash_derived) per player, primary first. sim.frames
            # is the timeline-shifted stream the engine actually rendered.
            _dash_players = [(getattr(meta, "player_name", ""),
                              base_sim.frames, getattr(meta, "dash_derived", False))]
            if overlay_extra:
                for _es, (_f, _mt, _n) in zip(extra_sims, overlay_extra):
                    _dash_players.append(
                        (getattr(_mt, "player_name", "") or _n, _es.frames,
                         getattr(_mt, "dash_derived", False)))
            _players_out = []
            for _nm, _fr, _drv in _dash_players:
                _runs, _edges, _held = _dash_runs(_fr)
                _players_out.append({
                    "player": _nm,
                    "source": "velocity_derived" if _drv else "legacy_bit",
                    "dash_edges": _edges,
                    "dash_frames": _held,
                    "total_frames": len(_fr),
                    "runs": _runs,
                })
            _map0_video_s = round((0 - start_ms) / (rate * 1000.0), 6)
            dash_sidecar = Path(str(output_path) + ".dash.json")
            dash_sidecar.write_text(_json.dumps(
                {"schema": 1, "mode": 2,
                 "map0_video_s": _map0_video_s, "rate": float(rate),
                 "players": _players_out}))
        except Exception as _dc_e:  # noqa: BLE001 — sidecar is best-effort
            print(f"[catch] dash sidecar write failed: {_dc_e}",
                  file=sys.stderr, flush=True)
    if progress_callback:
        progress_callback(100)
    return output_path


# --- ffmpeg -------------------------------------------------------------------

def _probe_encoder(cfg: RenderConfig) -> tuple[str, str | None]:
    if cfg.encoder != "auto":
        # vaapi always needs a device for the hwupload filter; default it.
        if cfg.encoder == "h264_vaapi":
            return cfg.encoder, cfg.encoder_device or "/dev/dri/renderD128"
        return cfg.encoder, cfg.encoder_device
    # nvenc FIRST: R3D renders on NVIDIA (2070S / 1070). The old vaapi-first
    # auto-probe silently won over the far-faster nvenc whenever R3D_ENCODER
    # was unset — a landmine if the worker env ever drops.
    # a listed hardware encoder is tried with one frame before it is trusted
    # (_encoder_starts); one that does not start is skipped
    if _ffmpeg_has("h264_nvenc") and _encoder_starts("h264_nvenc", None):
        return "h264_nvenc", None
    dev = cfg.encoder_device or "/dev/dri/renderD128"
    if (Path(dev).exists() and _ffmpeg_has("h264_vaapi")
            and _encoder_starts("h264_vaapi", dev)):
        return "h264_vaapi", dev
    return "libx264", None


def _ffmpeg_has(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        return False
    return name in out


_ENCODER_PROBE_TIMEOUT_S = 5.0   # how long a hardware encoder may take to start


def _encoder_starts(encoder: str, device: "str | None") -> bool:
    """Does `encoder` really start here? One black 128x128 frame through it to
    a null output. `ffmpeg -encoders` only says the encoder was COMPILED IN:
    h264_nvenc is listed on a machine with no NVIDIA card, h264_vaapi on one
    whose driver cannot encode, and until this check such an encoder was
    chosen and the render died on its first frame. A probe that hangs is
    killed at the timeout. R3D_ENCODER_PROBE=0 turns the check off (the
    listing alone decides, as before).

    The check and its command are the mania engine's (osu-mania-renderer #31,
    TheAussie), made synchronous."""
    if os.environ.get("R3D_ENCODER_PROBE", "1").strip().lower() \
            in ("0", "false", "no", "off"):
        return True
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if encoder == "h264_vaapi":
        cmd += ["-vaapi_device", device or "/dev/dri/renderD128"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "128x128", "-r", "1",
            "-i", "pipe:0", "-frames:v", "1", "-an", "-vf",
            "format=nv12,hwupload" if encoder == "h264_vaapi" else "format=yuv420p",
            "-c:v", encoder, "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, input=bytes(128 * 128 * 3), capture_output=True,
                           timeout=_ENCODER_PROBE_TIMEOUT_S, check=False)
        if r.returncode == 0:
            return True
        why = r.stderr.decode(errors="replace").strip().splitlines()
        why = why[0][:300] if why else f"ffmpeg exit code {r.returncode}"
    except subprocess.TimeoutExpired:
        why = "encoder startup probe timed out"
    except OSError as e:
        why = str(e)
    print(f"encoder: {encoder} is listed by ffmpeg but does not start here "
          f"({why}); trying the next one", file=sys.stderr, flush=True)
    return False


# ---- libx264 knobs behind env hooks (same names in all four engines) --------
# libx264 is the master encoder wherever a node has no hardware encoder (every
# Mac). The four engines ask it for different things (std crf 16 / faster,
# taiko crf 20 / veryfast, catch crf 23 / veryfast, mania 2500k / medium), so
# these make the choice settable per run without a code edit:
#   R3D_X264_PRESET   R3D_X264_CRF   R3D_X264_THREADS   R3D_X264_PARAMS
# THE DEFAULTS REPRODUCE THIS ENGINE'S CURRENT COMMAND EXACTLY (veryfast, crf 23, cores - 2 threads):
# with none of them set the ffmpeg argv is unchanged, argument for argument.
_X264_PRESET = os.environ.get("R3D_X264_PRESET", "").strip()
_X264_CRF = os.environ.get("R3D_X264_CRF", "").strip()
_X264_THREADS = os.environ.get("R3D_X264_THREADS", "").strip()
_X264_PARAMS = os.environ.get("R3D_X264_PARAMS", "").strip()


def nvenc_target_bps(w: int, h: int, fps: float) -> int:
    """Resolution-scaled NVENC bitrate ladder (R3D cross-engine policy, 2026-07).

    Replaces the flat per-engine bitrate: scale a 4 Mbps 720p30 reference
    by pixel rate with a perceptual exponent (0.70 -- deliberately NOT
    linear), clamped to [2.5, 16] Mbps.  Anchors: 720p30=4.0M,
    720p60=6.5M, 1080p30=7.1M, 1080p60=11.5M, 1440p60/1080p120+=16M cap.
    Callers pair the target with maxrate=1.5x / bufsize=2x for NVENC VBR.
    Same formula in all four engines (catch/taiko/std/mania v2).
    """
    ref = 1280.0 * 720.0 * 30.0
    target = 4_000_000.0 * ((float(w) * float(h) * float(fps)) / ref) ** 0.70
    return int(min(16_000_000.0, max(2_500_000.0, target)))


# STREAMABLE MASTER (R3D_STREAM_MASTER=1, default OFF). The contributor client
# can upload the master WHILE it renders if (a) the file is written front to
# back -- so no +faststart, which rewrites the whole file at close; the site
# moves the moov atom itself -- and (b) nothing rewrites it afterwards, so the
# loudness pass the client would run on the finished file (loudnorm on the
# MIXED output) is applied here instead. A marker file tells the client this
# engine honoured the flag; without it the client keeps its post-render pass.
STREAM_LOUDNORM = "loudnorm=I=-18:TP=-1.5:LRA=11"


def _stream_master_requested(overlay_extra) -> bool:
    return os.environ.get("R3D_STREAM_MASTER") == "1" and not overlay_extra


def _write_stream_marker(output_path: Path, compact: bool = False) -> None:
    """`<out stem>.stream.json`, written BEFORE ffmpeg starts. `compact` says
    the inline Discord copy is being written too, also without +faststart."""
    import json as _json
    marker = output_path.parent / (output_path.stem + ".stream.json")
    marker.write_text(_json.dumps(
        {"schema": 1, "faststart": False, "loudnorm": STREAM_LOUDNORM,
         "compact": bool(compact)}))



def _compact_wanted(total_dur_s, w, h, fps, default_factor) -> bool:
    """Whether to write the inline Discord copy for this render.

    R3D_COMPACT_INLINE=1 asks for it. The copy costs a second software encode
    for the whole render, and it is only used when the master is too big to be
    the Discord file itself, so R3D_COMPACT_IF_OVER_BYTES=<n> limits it to
    renders whose master is EXPECTED to exceed n bytes: duration x bitrate,
    with the bitrate taken from R3D_COMPACT_EXPECT_BPS (what this node's
    masters of this kind have actually averaged, supplied by the client) or,
    lacking that, the encoder ladder times `default_factor`. Without a limit,
    or without a duration, the copy is always written."""
    if os.environ.get("R3D_COMPACT_INLINE") != "1":
        return False
    try:
        limit = int(os.environ.get("R3D_COMPACT_IF_OVER_BYTES", "0") or 0)
    except ValueError:
        limit = 0
    if limit <= 0 or not total_dur_s or total_dur_s <= 0:
        return True
    try:
        bps = float(os.environ.get("R3D_COMPACT_EXPECT_BPS", "0") or 0)
    except ValueError:
        bps = 0.0
    if bps <= 0:
        bps = nvenc_target_bps(int(w), int(h), float(fps)) * default_factor
    return total_dur_s * bps / 8.0 > 0.9 * limit


def _preview_video_bps(total_dur_s: "float | None") -> int:
    """Video bitrate of the lean preview embed. Mirrors the contributor
    client's makeEmbedVariant (and the bot's _transcode_embed_unbounded):
    ~1.4 Mbps, lowered on long maps so the file stays <= ~24 MiB, floor 500k."""
    vbps = 1_400_000
    if total_dur_s and total_dur_s > 0:
        vbps = int(24 * 1024 * 1024 * 8 / total_dur_s) - 128_000
        vbps = max(500_000, min(1_400_000, vbps))
    return vbps


def _preview_sink_args(preview_path) -> list:
    """Output arguments for the inline preview.

    Default: one faststart mp4 at ``preview_path`` (unchanged).

    LIVE PREVIEW (R3D_PREVIEW_LIVE=1, default OFF): the same encode is written
    as 2 s self-contained fMP4 segments plus a growing playlist in
    ``<out stem>.live/`` (init.mp4, seg_00000.m4s ..., live.m3u8), so the
    contributor client can upload the preview WHILE the render runs and the
    site can play it before the render is done. No ``.embed.mp4`` is written in
    this mode; the client stitches one from the segments (a stream copy). A
    segment is renamed into place only when it is complete, and is listed in
    the playlist only after that."""
    if os.environ.get("R3D_PREVIEW_LIVE") != "1":
        return ["-movflags", "+faststart", str(preview_path)]
    live_dir = str(preview_path)[:-len(".embed.mp4")] + ".live"
    os.makedirs(live_dir, exist_ok=True)
    for _old in os.listdir(live_dir):       # a retry must not show stale segments
        try:
            os.remove(os.path.join(live_dir, _old))
        except OSError:
            pass
    return ["-f", "hls", "-hls_time", "2", "-hls_segment_type", "fmp4",
            "-hls_playlist_type", "event",
            "-hls_flags", "independent_segments+temp_file",
            "-hls_fmp4_init_filename", "init.mp4",
            "-hls_segment_filename", os.path.join(live_dir, "seg_%05d.m4s"),
            os.path.join(live_dir, "live.m3u8")]


def _spawn_ffmpeg(cfg: RenderConfig, output_path: Path, audio: Path | None,
                  start_ms: int, rate: float = 1.0, total_dur_s: float | None = None,
                  hitsound_wav: Path | None = None, is_nc: bool = False,
                  preview_path: "Path | None" = None,
                  stream_master: bool = False,
                  compact_path: "Path | None" = None):
    w, h = cfg.resolution
    enc, dev = _probe_encoder(cfg)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if enc == "h264_vaapi" and dev:
        cmd += ["-vaapi_device", dev]
    # stream_master: the master gets the final loudness pass here and no
    # +faststart (see STREAM_LOUDNORM above). Default False = command unchanged.
    _mfast = [] if stream_master else ["-movflags", "+faststart"]
    # RGBA ZERO-COPY PIPELINE: the frame producer hands the GL readback
    # buffer straight down the pipe (no 24<->32-bit repack). rgba input
    # yields BIT-IDENTICAL yuv420p to rgb24 (verified with framemd5);
    # the alpha byte is ignored by the encoder.
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(cfg.fps),
            "-i", "pipe:0"]
    # Loudnorm PCM cache (shared cross-engine; kill-switch R3D_NO_LOUDNORM_CACHE).
    # `prenorm` is a raw f32le@48k-stereo file with the rate/pitch change AND
    # loudnorm ALREADY baked in (full song, keyed on source+rate+pitch+params,
    # no per-render trim). When present, the song input is this file and the
    # filtergraph SKIPS the rate/pitch filters + loudnorm, keeping only the
    # per-render align/volume/apad(+hitsound mix). The post-loudnorm 48k resample
    # baked into the artifact reframes away loudnorm's look-ahead flush frame, so
    # a cold miss (build-then-read) and a warm hit (read) are byte-identical
    # through amix. `None` (kill-switch / cache miss build failure) falls back to
    # the unchanged inline fused-loudnorm path below.
    prenorm = None
    if audio is not None:
        prenorm = loudnorm_cache.get_or_build_normalized(
            audio, rate=rate, pitch=is_nc)
        if prenorm is not None:
            cmd += ["-f", "f32le",
                    "-ar", str(loudnorm_cache.LOUDNORM_CACHE_SR),
                    "-ac", str(loudnorm_cache.LOUDNORM_CACHE_CH),
                    "-i", str(prenorm)]
        else:
            cmd += ["-i", str(audio)]
        if hitsound_wav is not None:
            cmd += ["-i", str(hitsound_wav)]

    # video codec + pixel path (collected in `vc`; appended below)
    vc: list = []
    if enc == "h264_vaapi":
        if cfg.video_bitrate:
            vc += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi",
                    "-b:v", str(cfg.video_bitrate)]
        else:
            # CQ23 quality target (#87 R3D size policy): quality-based
            # encode is ~3-4x smaller than the flat 8M on osu gameplay,
            # which shrinks the node->coordinator upload (the real
            # "Finalizing" cost). VAAPI CQP is driver-dependent -> Aussie
            # to validate on AMD before the fleet bundle.
            vc += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi",
                    "-rc_mode", "CQP", "-qp", "23"]
    elif enc == "h264_nvenc":
        if cfg.video_bitrate:
            # Explicit override -> honor the bitrate target exactly.
            _tgt = int(cfg.video_bitrate)
            vc += ["-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
                    "-b:v", str(_tgt), "-maxrate", str(int(_tgt * 1.5)),
                    "-bufsize", str(_tgt * 2)]
        else:
            # CQ23 quality target (#87 R3D size policy): -cq 23 with the
            # resolution ladder as a maxrate CAP. On osu gameplay this
            # lands ~3-4x under the flat bitrate -> much smaller masters
            # -> faster node->coordinator upload (the "Finalizing" cost).
            _cap = nvenc_target_bps(w, h, cfg.fps)
            vc += ["-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
                    "-rc", "vbr", "-cq", "23", "-b:v", "0",
                    "-maxrate", str(_cap), "-bufsize", str(_cap * 2)]
    else:
        # CPU-encode thread cap (R3D host-governance, 2026-09): leave >=2
        # logical cores free for the machine's owner. Uncapped, libx264 spawns
        # threads on EVERY core at normal priority and can freeze a
        # contributor's desktop (the rel/Stella "semi-crash"). Harmless on
        # dedicated render boxes: libx264 is only the no-HW-encoder fallback.
        # Same cap in all four engines (catch/taiko/std/mania v2).
        _thr = ["-threads", _X264_THREADS
                or str(max(2, (os.cpu_count() or 4) - 2))]
        if _X264_PARAMS:
            _thr += ["-x264-params", _X264_PARAMS]
        _preset = _X264_PRESET or "veryfast"
        if cfg.video_bitrate:
            _vb = int(cfg.video_bitrate)
            vc += ["-c:v", "libx264", "-preset", _preset, "-pix_fmt", "yuv420p",
                    "-b:v", str(_vb), "-maxrate", str(int(_vb * 1.5)),
                    "-bufsize", str(_vb * 2)] + _thr
        else:
            vc += ["-c:v", "libx264", "-preset", _preset, "-pix_fmt", "yuv420p",
                    "-crf", _X264_CRF or "23"] + _thr

    acodec = ["-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-shortest"]
    if preview_path is None:
        cmd += vc
        if audio is not None:
            # `prenorm` -> canonical builders (rate/pitch + loudnorm are baked into
            # the cached f32le input); else the original inline fused-loudnorm path.
            pre = prenorm is not None
            if hitsound_wav is not None:
                # song + hitsound track: -filter_complex (the -af path can't mix a
                # second input). The song chain is IDENTICAL to _audio_filter minus
                # apad; hits amix AFTER the song's loudnorm (mania v2 fix #17).
                fc = _hitsound_filter_complex(
                    start_ms, rate, total_dur_s,
                    music_volume=cfg.music_volume,
                    general_volume=cfg.general_volume,
                    audio_offset_ms=cfg.audio_offset_ms,
                    hitsound_volume=getattr(cfg, "hitsound_volume", 100),
                    is_nc=is_nc, pre_normalized=pre)
                if stream_master:
                    fc += f";[aout]{STREAM_LOUDNORM}[aoutn]"
                cmd += ["-filter_complex", fc, "-map", "0:v",
                        "-map", "[aoutn]" if stream_master else "[aout]"]
            else:
                af = _audio_filter(start_ms, rate, total_dur_s,
                                   music_volume=cfg.music_volume,
                                   general_volume=cfg.general_volume,
                                   audio_offset_ms=cfg.audio_offset_ms, is_nc=is_nc,
                                   pre_normalized=pre)
                if stream_master:
                    af = (af + "," if af else "") + STREAM_LOUDNORM
                if af:
                    cmd += ["-af", af]
            cmd += acodec

        # web-streamable: move the moov atom to the front so browsers/iOS can
        # play before the whole file downloads (loudnorm re-adds this, but be
        # robust if that post-step is skipped/fails).
        cmd += _mfast + [str(output_path)]
    else:
        # TWO OUTPUTS FROM ONE PROCESS. The frame pipe is read once; `split`
        # hands the SAME frames to the master encoder (unchanged settings) and
        # to a 720p30 libx264 preview. The audio graph is the master's own, then
        # `asplit`; the preview branch gets the loudness pass the contributor
        # client would otherwise apply before cutting its embed, so the preview
        # needs no post-processing at all.
        graph = []
        pfps = min(30, int(round(float(cfg.fps))))
        # hardware preview (osu_catch_renderer/preview_hw.py; a Mac's media
        # engine, where the probe passes): the first frame repeated in front,
        # cut off again after encoding. "" leaves the graph as it was.
        # Only when the master is on a software encoder: then the preview's is
        # the one hardware session this process holds. A master that is itself
        # on a hardware encoder keeps the preview on x264 (a second session can
        # be refused, and one failed output kills the render).
        preview_hw = str(enc).startswith("lib") and _phw.preview_on_media_engine()
        if preview_hw and audio is not None and not (total_dur_s and total_dur_s > 0):
            preview_hw = False   # no known length to end the audio at: stay on x264
        _lead = ("," + _phw.vt_lead_in_filter(pfps)) if preview_hw else ""
        vm_tail = "null"
        if enc == "h264_vaapi":
            # the master's "-vf format=nv12,hwupload" moves into the graph
            vm_tail = "format=nv12,hwupload"
            vc = [x for i, x in enumerate(vc)
                  if not (x == "-vf" or (i and vc[i - 1] == "-vf"))]
        if compact_path is not None:
            # third branch: the Discord copy, to the compact plan (never
            # upscaled past the master, never above the master's frame rate)
            c_h, c_max, c_abps, c_fps = _embed_compact_plan(total_dur_s)
            c_h = min(c_h, int(h))
            c_fps = min(c_fps, int(round(float(cfg.fps))))
            graph.append(f"[0:v]split=3[vm0][vp0][vc0];[vm0]{vm_tail}[vm];"
                         f"[vp0]fps={pfps},scale=-2:720{_lead}[vp];"
                         f"[vc0]fps={c_fps},scale=-2:{c_h}:flags=bilinear[vc]")
        else:
            graph.append(f"[0:v]split=2[vm0][vp0];[vm0]{vm_tail}[vm];"
                         f"[vp0]fps={pfps},scale=-2:720{_lead}[vp]")
        if audio is not None:
            pre = prenorm is not None
            if hitsound_wav is not None:
                graph.append(_hitsound_filter_complex(
                    start_ms, rate, total_dur_s,
                    music_volume=cfg.music_volume,
                    general_volume=cfg.general_volume,
                    audio_offset_ms=cfg.audio_offset_ms,
                    hitsound_volume=getattr(cfg, "hitsound_volume", 100),
                    is_nc=is_nc, pre_normalized=pre))
            else:
                af = _audio_filter(start_ms, rate, total_dur_s,
                                   music_volume=cfg.music_volume,
                                   general_volume=cfg.general_volume,
                                   audio_offset_ms=cfg.audio_offset_ms, is_nc=is_nc,
                                   pre_normalized=pre)
                graph.append(f"[1:a]{af or 'anull'}[aout]")
            # PIN THE SHARED BRANCH TO 48 kHz BEFORE THE SPLIT. loudnorm runs at
            # 192 kHz internally and, left alone, wins format negotiation back
            # THROUGH asplit: the master's own chain (amix/apad on the cached
            # 48 kHz pre-normalised song) then runs at 192 kHz and is resampled
            # back down for the encoder, so the master's audio bytes change even
            # though nothing about the master was touched. Measured on the
            # loudnorm-cache path: master audio md5 differed with the preview
            # on. With the pin the 192 kHz conversion stays on the preview branch.
            # 48000 is what the master is encoded at (-ar 48000 below).
            _ac = "[ac]" if compact_path is not None else ""
            _an = 3 if compact_path is not None else 2
            if stream_master:
                # ONE loudness pass on the shared branch: the master, the
                # preview (and the Discord copy) carry the same normalised audio.
                graph.append(f"[aout]{STREAM_LOUDNORM},"
                             f"aformat=sample_rates=48000,asplit={_an}[am][ap]{_ac}")
            elif compact_path is not None:
                # the Discord copy is cut from the FINAL (normalised) audio,
                # like the one the client encodes from the finished master
                graph.append("[aout]aformat=sample_rates=48000,asplit=2[am][ap0];"
                             "[ap0]loudnorm=I=-18:TP=-1.5:LRA=11,asplit=2[ap][ac]")
            else:
                graph.append("[aout]aformat=sample_rates=48000,asplit=2[am][ap0];"
                             "[ap0]loudnorm=I=-18:TP=-1.5:LRA=11[ap]")
        if _lead and audio is not None:
            # the usual end-of-stream rule (`-shortest` / `-t`) measures the
            # preview's video BEFORE the lead-in is cut off, so it cannot be
            # used on this output (see where it is left out below). The video's
            # length is known here: end the preview's audio there instead.
            graph[-1] = graph[-1].replace("[ap]", "[ap_full]")
            graph.append(f"[ap_full]atrim=end={float(total_dur_s):.6f}[ap]")
        cmd += ["-filter_complex", ";".join(graph)]
        # output 1: the master, exactly as without the preview
        cmd += ["-map", "[vm]"] + (["-map", "[am]"] if audio is not None else [])
        cmd += vc + (acodec if audio is not None else [])
        cmd += _mfast + [str(output_path)]
        # output 2: the preview. libx264 unless a hardware session was proved
        # to open here (`preview_hw`): a second NVENC/VAAPI session can fail to
        # open (session limits), and one failed output kills the whole process
        # and with it the render. On a Mac the master is on x264, so the
        # preview's is the only hardware session this process holds.
        vbps = _preview_video_bps(total_dur_s)
        cmd += ["-map", "[vp]"] + (["-map", "[ap]"] if audio is not None else [])
        if preview_hw:
            cmd += _phw.hw_video_args(vbps, pfps)
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-b:v", str(vbps), "-maxrate", str(int(vbps * 1.25)),
                    "-bufsize", str(vbps * 2), "-g", "30",
                    "-threads", str(max(2, min(4, (os.cpu_count() or 4) - 2)))]
        if audio is not None:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000"]
            if not _lead:
                cmd += ["-shortest"]
        cmd += _preview_sink_args(preview_path)
        if compact_path is not None:
            # output 3: the Discord copy. Same recipe as the node's own compact
            # encode (libx264 veryfast crf 21 + VBV at the plan's maxrate).
            cmd += ["-map", "[vc]"] + (["-map", "[ac]"] if audio is not None else [])
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-crf", "21", "-maxrate", str(c_max),
                    "-bufsize", str(max(1, c_max // 2)), "-g", str(c_fps),
                    "-threads", str(max(2, min(6, (os.cpu_count() or 4) // 2)))]
            if audio is not None:
                cmd += ["-c:a", "aac", "-ar", "48000", "-b:a", str(c_abps), "-shortest"]
            cmd += _mfast + [str(compact_path)]
    import tempfile
    errf = tempfile.NamedTemporaryFile(
        prefix="catch_ffmpeg_", suffix=".log", delete=False, mode="w+",
    )
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf, bufsize=0)
    # PERF: grow the stdin pipe from the 64 KB default (a 6 MB 1080p frame =
    # ~95 kernel wakeups) up to pipe-max-size (1 MB unprivileged). Fewer
    # syscalls + smoother handoff; bytes on the pipe are unchanged.
    try:
        import fcntl
        F_SETPIPE_SZ = 1031
        fcntl.fcntl(proc.stdin.fileno(), F_SETPIPE_SZ, 1 << 20)
    except (OSError, ImportError, AttributeError):
        # fcntl + F_SETPIPE_SZ are Linux-only; on Windows contributors `import
        # fcntl` raises ModuleNotFoundError (an ImportError, NOT OSError) which
        # used to escape and crash EVERY catch render (exit 1). Skip the pipe-size
        # optimization there -- the render is correct with the default pipe.
        pass
    proc._catch_errlog = errf.name  # type: ignore[attr-defined]
    return proc


def _embed_compact_plan(total_dur_s: "float | None") -> "tuple[int, int, int, int]":
    """(scale_h, maxrate_bps, audio_bps, fps) for the Discord companion — mirrors
    embed_attach._compact_plan at the config default 80 MiB budget so the sidecar
    is the SAME to-spec asset the node/bot would produce and can be adopted
    verbatim (embed_attach.compact_meets_target). Short plays keep 1080p60; longer
    step to 720p60; only a genuinely too-small budget drops to 720p30.

    Budget is 56 MiB (NOT the bot's 80 MiB): a NODE companion must stay under the
    64 MiB node->coordinator sidecar upload limit, matching the client
    embed_variants.go cap (0.4.10). Raise both only once the coordinator upload
    limit is raised."""
    budget_bits = 56 * 1024 * 1024 * 8
    dur = float(total_dur_s or 0.0)
    if dur <= 1:
        return 1080, 8_000_000, 192_000, 60
    total_rate = int(budget_bits / dur) or 1
    pref = 192_000 if dur <= 240 else (128_000 if dur <= 600 else 96_000)
    audio = min(pref, max(32_000, total_rate // 4))
    maxrate = max(32_000, min(8_000_000, total_rate - audio))
    if maxrate >= 3_000_000:
        return 1080, maxrate, audio, 60
    if maxrate >= 500_000:
        return 720, maxrate, audio, 60
    return 720, maxrate, audio, 30


def _spawn_embed_ffmpeg(cfg: RenderConfig, embed_path: Path, audio: "Path | None",
                        start_ms: int, rate: float = 1.0,
                        total_dur_s: "float | None" = None,
                        hitsound_wav: "Path | None" = None, is_nc: bool = False,
                        in_fps: "float | None" = None):
    """Concurrent Discord-companion sidecar produced DURING the render from the
    SAME composited RGBA frames (see _CompositeWorker). Recipe mirrors
    embed_attach.transcode_compact at the to-spec plan (1080p60 short -> 720p60 ->
    720p30) so the node can ADOPT it verbatim (compact_meets_target) instead of
    re-encoding post-render — the sidecar is ready the instant the render
    finishes. libx264 crf21 + VBV to the plan maxrate, 48 kHz AAC. Audio graph
    identical to the master's (shared loudnorm PCM cache — a warm hit, no double
    loudnorm). Best-effort at the writer: if this ffmpeg dies the sidecar is
    simply absent and the node re-encodes."""
    w, h = cfg.resolution
    scale_h, maxrate, audio_bps, _plan_fps = _embed_compact_plan(total_dur_s)
    if scale_h > h:
        scale_h = h  # never upscale beyond the master's height
    # Frames arrive pre-decimated to `in_fps` (composite-thread stride), so the
    # input rate IS in_fps -> playback speed stays correct and no fps filter is
    # needed. Fewer input frames == less pipe traffic + less libx264 work.
    # nice -n 15: the sidecar runs at LOW priority so it consumes only spare
    # cores and never steals from the render's (CPU/GIL-bound) frame generation.
    # If the box is saturated it simply finishes a beat after render-done —
    # still far cheaper than the node's from-scratch post-render re-encode.
    cmd = ["nice", "-n", "15",
           "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
           "-r", str(in_fps or cfg.fps), "-i", "pipe:0"]
    prenorm = None
    if audio is not None:
        prenorm = loudnorm_cache.get_or_build_normalized(
            audio, rate=rate, pitch=is_nc)
        if prenorm is not None:
            cmd += ["-f", "f32le",
                    "-ar", str(loudnorm_cache.LOUDNORM_CACHE_SR),
                    "-ac", str(loudnorm_cache.LOUDNORM_CACHE_CH),
                    "-i", str(prenorm)]
        else:
            cmd += ["-i", str(audio)]
        if hitsound_wav is not None:
            cmd += ["-i", str(hitsound_wav)]
    # video: decimate to 30 fps + scale to 720p (the crawler-safe embed size).
    if audio is not None and hitsound_wav is not None:
        # hitsound path needs -filter_complex for the audio mix; append the
        # video scale to the same graph so both are mapped explicitly.
        pre = prenorm is not None
        fc = _hitsound_filter_complex(
            start_ms, rate, total_dur_s,
            music_volume=cfg.music_volume, general_volume=cfg.general_volume,
            audio_offset_ms=cfg.audio_offset_ms,
            hitsound_volume=getattr(cfg, "hitsound_volume", 100),
            is_nc=is_nc, pre_normalized=pre)
        fc += f";[0:v]scale=-2:{scale_h}:flags=bilinear[vs]"
        cmd += ["-filter_complex", fc, "-map", "[vs]", "-map", "[aout]"]
    else:
        cmd += ["-vf", f"scale=-2:{scale_h}:flags=bilinear"]
        if audio is not None:
            pre = prenorm is not None
            af = _audio_filter(
                start_ms, rate, total_dur_s,
                music_volume=cfg.music_volume, general_volume=cfg.general_volume,
                audio_offset_ms=cfg.audio_offset_ms, is_nc=is_nc,
                pre_normalized=pre)
            if af:
                cmd += ["-af", af]
    # CPU encode: veryfast + VBV; cap threads so a contributor desktop stays
    # usable while the GPU handles the master (same governance as libx264 above).
    # Thread cap sweet spot (idle-box sweep 2026-09-27): ~half the cores, capped
    # at 6. libx264 veryfast 1080p60 keeps pace with the render at 4-6 threads
    # (finalize tail ~0.5s either way), and a LOWER cap stops the companion
    # fighting the CPU/GIL-bound frame generation for cores -> +5% render instead
    # of +22% at the old cpu-2 cap. Override with R3D_EMBED_THREADS.
    _emb_thr = os.environ.get("R3D_EMBED_THREADS", "")
    _nthr = int(_emb_thr) if _emb_thr.isdigit() and int(_emb_thr) > 0 \
        else max(2, min(6, (os.cpu_count() or 4) // 2))
    _thr = ["-threads", str(_nthr)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-crf", "21", "-maxrate", str(maxrate),
            "-bufsize", str(max(1, maxrate // 2)),
            "-g", str(int(_plan_fps))] + _thr
    if audio is not None:
        cmd += ["-c:a", "aac", "-ar", "48000",
                "-b:a", str(audio_bps), "-shortest"]
    cmd += ["-movflags", "+faststart", str(embed_path)]
    import tempfile
    errf = tempfile.NamedTemporaryFile(
        prefix="catch_embed_", suffix=".log", delete=False, mode="w+")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf, bufsize=0)
    try:
        import fcntl
        fcntl.fcntl(proc.stdin.fileno(), 1031, 1 << 20)
    except (OSError, ImportError, AttributeError):
        pass
    proc._catch_errlog = errf.name  # type: ignore[attr-defined]
    return proc


def _audio_filter(start_ms: int, rate: float = 1.0, total_dur_s: float | None = None,
                  music_volume: int = 100, general_volume: int = 100,
                  audio_offset_ms: int = 0, is_nc: bool = False,
                  pre_normalized: bool = False) -> str:
    """Speed the song to the mod rate (DT/HT), then align so video t=0 is
    `start_ms` into the rate-adjusted song. Applies preset volume + offset.

    `pre_normalized` = the song input is the shared loudnorm PCM cache artifact
    (raw f32le@48k with the rate/pitch change AND loudnorm ALREADY baked in), so
    the rate filters and the inline loudnorm are OMITTED here; only the
    per-render align/volume/apad remain. When False the chain is byte-for-byte
    the original fused pipeline (kill-switch / cache-miss fallback)."""
    parts = []
    if not pre_normalized and abs(rate - 1.0) > 1e-3:
        if is_nc:
            # Nightcore = a PURE RESAMPLE: speed AND pitch up together by the
            # rate, exactly like osu (and the mania v2 / taiko renderers).
            # Reinterpreting the samples at SR*rate then resampling back to SR is
            # artifact-free. The old atempo was speed-ONLY (pitch preserved), so
            # catch NC never pitched up (wrong). Normalise to 44100 first so a
            # 48 kHz master still speeds by exactly `rate` — asetrate is absolute.
            parts.append("aresample=44100")
            parts.append(f"asetrate={int(round(44100 * rate))}")
            parts.append("aresample=44100")
        else:
            parts.append(f"atempo={rate:.4f}")  # DT/HT: pitch-preserving speed
    # start_ms is in MAP time; after atempo the song plays at map/rate, so the
    # real offset where video t=0 lands is start_ms/rate. audio_offset shifts the
    # song vs gameplay (negative = audio earlier).
    real_start = (start_ms - audio_offset_ms) / rate
    if real_start > 0:
        parts.append(f"atrim=start={real_start / 1000:.3f}")
        parts.append("asetpts=PTS-STARTPTS")
    elif real_start < 0:
        parts.append(f"adelay={int(-real_start)}:all=1")
    # Pad with silence so the audio spans the full video (incl. the results
    # outro past the song's end). Bound the pad to the exact video duration —
    # an UNBOUNDED apad races the (slow) raw-video pipe and overflows the
    # filtergraph buffer (ffmpeg reports it as ENOSPC and dies).
    # Loudness-normalise to a consistent EBU R128 baseline (single-pass) so
    # hot beatmap masters stop blasting: I=-18 LUFS, true-peak -1.5 dBTP.
    # The volume trim below is applied AFTER, relative to this baseline.
    # (Skipped when pre_normalized: loudnorm is already baked into the cache.)
    if not pre_normalized:
        parts.append("loudnorm=I=-18:TP=-1.5:LRA=11")
    vol = (general_volume / 100.0) * (music_volume / 100.0)
    if abs(vol - 1.0) > 1e-3:
        parts.append(f"volume={max(0.0, vol):.3f}")
    if total_dur_s and total_dur_s > 0:
        parts.append(f"apad=whole_dur={total_dur_s:.3f}")
    else:
        parts.append("apad")
    return ",".join(parts)


def _hitsound_filter_complex(start_ms: int, rate: float,
                             total_dur_s: float | None,
                             music_volume: int = 100,
                             general_volume: int = 100,
                             audio_offset_ms: int = 0,
                             hitsound_volume: int = 100,
                             is_nc: bool = False,
                             pre_normalized: bool = False) -> str:
    """The hitsound-enabled audio graph. The SONG chain reproduces
    _audio_filter exactly (atempo -> align -> loudnorm -> volume) so the
    music bed is bit-identical to a hitsound-less render; the pre-mixed hits
    WAV (input 2, already on the video time axis at natural pitch) is amixed
    ON TOP of the normalised song — never through loudnorm, whose gain would
    duck the song ~4 dB under every hit (mania v2 LOUDNORM FIX 2026-07-12,
    #17) — then a clamp-only true-peak limiter catches summed peaks and apad
    spans the results outro. Hits take general x hitsound volume (stable's
    master x effect), not music volume.

    `pre_normalized` = the song input is the shared loudnorm PCM cache artifact
    (rate/pitch + loudnorm already baked in), so the rate filters and the inline
    loudnorm are OMITTED from the song chain; only align/volume remain before the
    amix. When False the chain is byte-for-byte the original fused pipeline."""
    song = []
    if not pre_normalized and abs(rate - 1.0) > 1e-3:
        if is_nc:
            # NC = pure resample (speed + pitch); see _audio_filter. Keeps the
            # song chain identical to the hitsound-less render.
            song.append("aresample=44100")
            song.append(f"asetrate={int(round(44100 * rate))}")
            song.append("aresample=44100")
        else:
            song.append(f"atempo={rate:.4f}")
    real_start = (start_ms - audio_offset_ms) / rate
    if real_start > 0:
        song.append(f"atrim=start={real_start / 1000:.3f}")
        song.append("asetpts=PTS-STARTPTS")
    elif real_start < 0:
        song.append(f"adelay={int(-real_start)}:all=1")
    if not pre_normalized:
        song.append("loudnorm=I=-18:TP=-1.5:LRA=11")
    vol = (general_volume / 100.0) * (music_volume / 100.0)
    if abs(vol - 1.0) > 1e-3:
        song.append(f"volume={max(0.0, vol):.3f}")
    # A pre-normalised song with no per-render align/volume has an EMPTY chain;
    # feed [1:a] straight through anull so the [song] label is still valid.
    song_str = ",".join(song) if song else "anull"
    hvol = (general_volume / 100.0) * (hitsound_volume / 100.0)
    hits = ([f"volume={max(0.0, hvol):.3f}"] if abs(hvol - 1.0) > 1e-3
            else ["anull"])
    tail = ["amix=inputs=2:duration=longest:normalize=0:weights=1 1",
            "alimiter=limit=0.95:level=disabled:attack=1:release=20"]
    if total_dur_s and total_dur_s > 0:
        tail.append(f"apad=whole_dur={total_dur_s:.3f}")
    else:
        tail.append("apad")
    return (f"[1:a]{song_str}[song];"
            f"[2:a]{','.join(hits)}[hits];"
            f"[song][hits]{','.join(tail)}[aout]")


log = logging.getLogger(__name__)


def _bg_cover(path: Path, w: int, h: int, blur: int = 0) -> "np.ndarray | None":
    """Load the beatmap background and cover-crop it to WxH (no distortion).
    Returns None if the (user-supplied) background can't be decoded, so the
    caller skips the bg upload -- same as a map with no background."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:  # noqa: BLE001 -- a corrupt user bg must not crash the render
        log.warning("background image failed to decode, skipping: %s (%s)", path, e)
        return None
    scale = max(w / im.width, h / im.height)
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    im = im.crop((left, top, left + w, top + h))
    if blur and blur > 0:
        from PIL import ImageFilter
        im = im.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    return np.array(im)


def _find_osu(beatmap_dir: Path, md5: str) -> Path:
    osus = sorted(beatmap_dir.glob("*.osu"))
    if not osus:
        raise CatchRenderError(f"no .osu in {beatmap_dir}")
    if md5:
        for p in osus:
            if hashlib.md5(p.read_bytes()).hexdigest() == md5:
                return p
    # DMCA/mirror-down recovery: the bot's manual-.osz upload path writes a
    # ".r3d_forced_osu" marker naming the difficulty it matched when the
    # replay's exact md5 is not in the archive (a pack shipping a different
    # version, or an unsubmitted map). Honour it before the mode/first
    # fallback so we render THAT diff -- and resolve ITS audio/bg -- instead
    # of the first same-mode one (which desyncs or renders silently).
    _forced_marker = beatmap_dir / ".r3d_forced_osu"
    if _forced_marker.is_file():
        try:
            _forced = beatmap_dir / pathlib.Path(
                _forced_marker.read_text(encoding="utf-8").strip()
            ).name
        except OSError:
            _forced = None
        if _forced is not None and _forced.is_file() \
                and _forced.suffix.lower() == ".osu":
            return _forced
    # fall back to a Mode:2 beatmap, else the first
    for p in osus:
        head = p.read_text(encoding="utf-8", errors="replace")[:4000]
        if "Mode: 2" in head or "Mode:2" in head:
            return p
    return osus[0]


# --- HUD ----------------------------------------------------------------------

class _Hud:
    def __init__(self, w, h, meta, bm):
        self.w, self.h = w, h
        self.meta = meta
        self.bm = bm
        big = max(20, int(h * 0.07))
        med = max(16, int(h * 0.035))
        small = max(12, int(h * 0.025))
        self.f_combo = _font(big)
        self.f_score = _font(med)
        self.f_small = _font(small)

    def overlay(self, rgb: np.ndarray, scene) -> np.ndarray:
        # RGBA zero-copy canvas (fallback HUD): wrap the writable 4ch frame
        # in place like DanserHud does; 3ch legacy input keeps the old copy.
        from osu_catch_renderer.hud.hud import _img_from_rgb, _img_out
        img = _img_from_rgb(rgb)
        d = ImageDraw.Draw(img)
        # combo bottom-left
        if scene.combo > 0:
            d.text((int(self.w * 0.02), int(self.h * 0.86)), f"{scene.combo}x",
                   font=self.f_combo, fill=(255, 255, 255))
        # score top-right
        d.text((int(self.w * 0.98), int(self.h * 0.03)), f"{scene.score:,}",
               font=self.f_score, fill=(255, 255, 255), anchor="ra")
        # player + title top-left
        d.text((int(self.w * 0.02), int(self.h * 0.03)), self.meta.player_name,
               font=self.f_small, fill=(230, 230, 240))
        title = f"{self.bm.artist} - {self.bm.title} [{self.bm.version}]".strip(" -")
        d.text((int(self.w * 0.02), int(self.h * 0.065)), title,
               font=self.f_small, fill=(180, 180, 200))
        # hp bar top center
        bx, by, bw, bh = int(self.w * 0.30), int(self.h * 0.02), int(self.w * 0.40), 10
        d.rectangle([bx, by, bx + bw, by + bh], fill=(40, 40, 50))
        d.rectangle([bx, by, bx + int(bw * scene.hp), by + bh], fill=(120, 220, 140))
        return _img_out(img)


from osu_catch_renderer.hud.fonts import font as _font  # skin-aware, host-robust font resolver
