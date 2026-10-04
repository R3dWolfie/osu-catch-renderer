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
from PIL import Image, ImageDraw, ImageFont

from osu_catch_renderer._metal.skin.assets import build_textures
from osu_catch_renderer._metal.beatmap.beatmap import parse_beatmap
from osu_catch_renderer._metal.render.death import (FAIL_FADE_MS, apply_death,
                                             apply_fail_audio,
                                             death_progress)
from osu_catch_renderer._metal.render.flashlight import CatchFlashlight, has_flashlight
from osu_catch_renderer._metal.render.gl import SpriteRenderer
from osu_catch_renderer._metal.render import loudnorm_cache
from osu_catch_renderer._metal.beatmap.models import RenderConfig, ar_to_preempt_ms, ObjType
from osu_catch_renderer._metal.beatmap.replay import parse_replay
from osu_catch_renderer._metal.render.scene import CatchSim, mods_score_multiplier


class CatchRenderError(RuntimeError):
    pass


# PARALLEL OUTRO (R3D_OUTRO_WORKERS=N, default 1 = shipped behaviour).
# The results screen is 318 frames x ~12.2 ms of PURE function of
# (frozen_frame, opacity, age_ms) -- render_frame's only self-mutation is a
# memo cache, and hud.draw_results now keys its screen per THREAD, so frames
# are independent. Crucially the GL render thread is IDLE during the outro
# (no frames left to draw), so unlike the gameplay phase there IS spare
# capacity here. Batched so at most N frames are in flight (N x 8.29 MB).
try:
    _OUTRO_WORKERS = max(1, int(os.environ.get("R3D_OUTRO_WORKERS", "1") or 1))
except ValueError:
    _OUTRO_WORKERS = 1
try:
    _GL_STUB = int(os.environ.get("R3D_GL_STUB", "0") or 0)
except ValueError:
    _GL_STUB = 0
# x264 preset. The encoder is ~7.2 of 10 cores (measured), so this is the
# largest single Front-1 lever -- but it CHANGES OUTPUT BYTES, so it is a
# deliverable decision, not a tuning knob. Default keeps the shipped value.
_X264_PRESET = os.environ.get("R3D_X264_PRESET", "veryfast").strip() or "veryfast"
_ABLATE_HUD = os.environ.get("R3D_ABLATE_HUD") == "1"

# R3D_TIMELINE=<path>: append one row per stage per frame, so the ACTUAL
# sequencing and overlap across the three threads can be reconstructed instead
# of inferred from per-stage sums. Sums mislead in a contended pipeline -- a
# stage that "costs" 2 ms may be 2 ms of waiting for another thread.
# Row: thread,seq,stage,t_start_ns,t_end_ns
_TIMELINE_PATH = os.environ.get("R3D_TIMELINE", "").strip()


class _Timeline:
    """Lock-free-ish per-thread append. Each thread owns its own list, so the
    only shared mutation is registering the list once."""

    __slots__ = ("_per_thread", "_lock", "t0")

    def __init__(self):
        import threading
        self._per_thread = {}
        self._lock = threading.Lock()
        self.t0 = time.perf_counter_ns()

    def buf(self):
        import threading
        k = threading.get_ident()
        b = self._per_thread.get(k)
        if b is None:
            b = []
            with self._lock:
                self._per_thread[k] = b
        return b

    def dump(self, path):
        names = {}
        import threading
        for t in threading.enumerate():
            names[t.ident] = t.name
        n = 0
        with open(path, "w") as f:
            f.write("thread\tseq\tstage\tstart_us\tend_us\n")
            for ident, rows in self._per_thread.items():
                nm = names.get(ident, f"t{ident}")
                for seq, stage, a, b in rows:
                    f.write(f"{nm}\t{seq}\t{stage}\t"
                            f"{(a - self.t0) / 1000.0:.1f}\t"
                            f"{(b - self.t0) / 1000.0:.1f}\n")
                    n += 1
        return n


_TL = _Timeline() if _TIMELINE_PATH else None

_pc_w = time.perf_counter


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
    (same env/mechanism as the std renderer).

    R3D_DUMP_RGBA=<dir> additionally dumps the PRE-ENCODE RGBA buffers to
    disk. This is the same byte stream ffmpeg ingests (post-HUD-composite,
    pre-codec), so a cross-node comparison made from these files isolates the
    RENDERER from the encoder — a Mac node (VideoToolbox/libx264) and a Linux
    node (NVENC/VAAPI) can never match at the .mp4 level, and the fleet is
    already mixed-vendor, so the meaningful question is the magnitude of the
    pixel difference, not bit-identity. Writes:
      index.txt   `<n> <blake2b-128> <w>x<h>` for EVERY frame -- cheap, and it
                  localises WHICH frames diverge
      frame_%06d.rgba  raw buffers for a sampled subset (see the knobs below)
      meta.txt    geometry + the ffmpeg line that turns a .rgba into a PNG
    Knobs: R3D_DUMP_RGBA_INDEX=0 (skip the per-frame index -- it is the
    expensive part), R3D_DUMP_RGBA_EVERY=<n> (stride, default 30) and
    R3D_DUMP_RGBA_MAX=<n> (cap on dumped buffers, default 60 -- 1080p RGBA is
    ~8.3 MB/frame, so an uncapped dump of a full render is tens of GB). The
    cap that stopped a dump is reported at close; it never truncates silently.
    Debug-only and off by default: the per-frame hash costs writer-thread CPU."""

    # 12 frames of elasticity (~75 MB at 1080p): ffmpeg's ingest is fast on
    # average but stalls in bursts (loudnorm's 3 s blocks + muxer interleave);
    # a 4-deep queue let every stall block the render thread.
    # R3D_WRITER_QUEUE overrides this. 12 was sized for 8.29 MB RGBA frames
    # (~100 MB of elasticity). With the GPU yuv converter the frames are
    # 3.11 MB, so the same memory buys 32 -- and the reason the queue exists
    # (ffmpeg ingest stalling in bursts) is exactly what deeper buffering
    # absorbs. Measured: the pipeline uses only 1.6 of 10 cores at null sink,
    # so the loss to the encoder is a serial handoff, not CPU capacity.
    _QUEUE_FRAMES = int(os.environ.get("R3D_WRITER_QUEUE", "12") or 12)

    def __init__(self, proc, perf=None):
        self._stdin = proc.stdin
        self._perf = perf          # R3D_CATCH_PERF: stage (e), the pipe write
        self._q: "queue.Queue" = queue.Queue(maxsize=self._QUEUE_FRAMES)
        self._werr: BaseException | None = None
        self._yuv_submit = None    # set_yuv_pipe(): GPU rgba -> yuv420p
        self._yuv_acquire = None
        self._hash = None
        self._hash_frames = 0
        if os.environ.get("R3D_FRAME_MD5"):
            self._hash = hashlib.blake2b(digest_size=16)
        # R3D_DUMP_RGBA=<dir> — pre-encode buffer dump (see class docstring).
        self._dump_dir = None
        self._dump_dir_str = None
        self._dump_n = 0            # frames seen
        self._dump_written = 0      # buffers actually written
        self._dump_capped = False
        self._dump_index = None
        self._dump_err = None
        _dd = os.environ.get("R3D_DUMP_RGBA", "").strip()
        if _dd:
            try:
                self._dump_dir = Path(_dd)
                self._dump_dir.mkdir(parents=True, exist_ok=True)
                self._dump_dir_str = str(self._dump_dir)
                self._dump_every = max(1, int(os.environ.get("R3D_DUMP_RGBA_EVERY", "30")))
                self._dump_max = max(0, int(os.environ.get("R3D_DUMP_RGBA_MAX", "60")))
                # The full-coverage index hashes EVERY frame, which is ~8.3 MB
                # of memory traffic per 1080p frame and roughly halves render
                # throughput. Worth it (it localises which frames diverge), but
                # R3D_DUMP_RGBA_INDEX=0 turns it off for a buffers-only dump.
                self._dump_index = (
                    open(self._dump_dir / "index.txt", "w")
                    if os.environ.get("R3D_DUMP_RGBA_INDEX", "1") != "0" else None)
            except Exception as e:  # noqa: BLE001 — a debug dump never kills a render
                print(f"[catch] R3D_DUMP_RGBA disabled ({e})", file=sys.stderr, flush=True)
                self._dump_dir = None
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
                if isinstance(frame, (bytes, bytearray, memoryview)):
                    data = frame          # already planar YUV from the GPU
                elif isinstance(frame, np.ndarray) and frame.flags.c_contiguous:
                    data = memoryview(frame).cast("B")
                else:
                    data = frame.tobytes()
                if self._hash is not None:
                    self._hash.update(data)
                    self._hash_frames += 1
                _tl_w0 = time.perf_counter_ns() if _TL is not None else 0
                if self._perf is not None:
                    _tp = _pc_w()
                    self._stdin.write(data)
                    self._perf["pipe"] += _pc_w() - _tp
                    self._perf["pipe_n"] += 1
                else:
                    self._stdin.write(data)
                if _TL is not None:
                    self._tl_n = getattr(self, "_tl_n", 0) + 1
                    _TL.buf().append((self._tl_n, "w:pipe", _tl_w0,
                                      time.perf_counter_ns()))
                if self._dump_dir is not None and self._yuv_submit is None:
                    # After the pipe write on purpose: the dump must not sit
                    # between the producer and ffmpeg, and the bytes on the
                    # pipe are already gone by the time we touch disk.
                    # (With the YUV converter on, push() already dumped the
                    # RGBA -- what lands here is planar YUV, not the frame.)
                    self._dump(frame, data)
            except BaseException as e:  # noqa: BLE001 — surfaced on push()
                self._werr = e

    def _dump(self, frame, data) -> None:
        """Best-effort pre-encode RGBA dump; never raises into the writer."""
        try:
            n = self._dump_n
            self._dump_n += 1
            h, w = (frame.shape[0], frame.shape[1]) if isinstance(frame, np.ndarray) \
                else (0, 0)
            if self._dump_index is not None:
                self._dump_index.write(
                    f"{n} {hashlib.blake2b(data, digest_size=16).hexdigest()} {w}x{h}\n")
            if n % self._dump_every:
                return
            if self._dump_written >= self._dump_max:
                self._dump_capped = True
                return
            with open(self._dump_dir / f"frame_{n:06d}.rgba", "wb") as f:
                f.write(data)
            self._dump_written += 1
            if self._dump_written == 1 and w:
                (self._dump_dir / "meta.txt").write_text(
                    f"width={w}\nheight={h}\npix_fmt=rgba\n"
                    f"stride_every={self._dump_every}\nmax_buffers={self._dump_max}\n"
                    f"# .rgba -> png:\n"
                    f"# ffmpeg -f rawvideo -pix_fmt rgba -s {w}x{h} "
                    f"-i frame_000000.rgba -frames:v 1 frame_000000.png\n")
        except Exception as e:  # noqa: BLE001
            if self._dump_err is None:
                self._dump_err = e
                print(f"[catch] R3D_DUMP_RGBA write failed, dump stopped ({e})",
                      file=sys.stderr, flush=True)
            self._dump_dir = None

    def set_yuv_pipe(self, submit, acquire) -> None:
        """Hand the writer a PIPELINED GPU rgba->yuv420p converter.
        Conversion happens on the caller's thread (the composite thread, which
        binds) and is submitted WITHOUT waiting for the GPU -- we queue frame N
        and collect frame N-depth, so the thread never stalls on GPU latency.
        The earlier synchronous version cost ~1.0 ms/frame right here."""
        self._yuv_submit, self._yuv_acquire = submit, acquire

    def push(self, frame_rgb) -> None:
        """Queue one frame. Re-raises the writer thread's error, so a dead
        ffmpeg surfaces here just like the old synchronous write did
        (BrokenPipeError included)."""
        if self._werr is not None:
            raise self._werr
        if (self._yuv_submit is None
                or isinstance(frame_rgb, (bytes, bytearray, memoryview))):
            # already planar yuv420p (the GPU outro converts on the die, so the
            # RGBA never crosses) -- straight to the writer
            self._q.put(frame_rgb)
            return
        # The RGBA dump, when on, must still see RGBA -- taken here, before the
        # conversion, rather than in the writer thread.
        if self._dump_dir is not None:
            self._dump(frame_rgb,
                       memoryview(frame_rgb).cast("B")
                       if getattr(frame_rgb, "flags", None) is not None
                       and frame_rgb.flags.c_contiguous else frame_rgb.tobytes())
        # Ring full -> drain one before submitting. FIFO either way: submits go
        # in frame order and acquires come out in the same order.
        while not self._yuv_submit(frame_rgb):
            out = self._yuv_acquire(True)
            if out is None:
                raise CatchRenderError("GPU yuv ring deadlocked")
            self._q.put(out)
        out = self._yuv_acquire(False)
        if out is not None:
            self._q.put(out)

    def close(self) -> None:
        # Drain the GPU yuv pipeline first -- it holds `depth` frames that have
        # been submitted but not collected, and every frame must be emitted.
        if self._yuv_acquire is not None:
            while True:
                out = self._yuv_acquire(True)
                if out is None:
                    break
                self._q.put(out)
        self._q.put(None)
        self._thread.join()
        if self._hash is not None:
            print(f"frame-stream-hash: {self._hash.hexdigest()} "
                  f"({self._hash_frames} frames)", file=sys.stderr, flush=True)
        if self._dump_dir_str is not None:
            if self._dump_index is not None:
                try:
                    self._dump_index.close()
                except Exception:  # noqa: BLE001
                    pass
            # Report the cap explicitly -- a truncated dump that LOOKS complete
            # would silently weaken the cross-node comparison it exists for.
            cap = (f", CAPPED at R3D_DUMP_RGBA_MAX={self._dump_max} "
                   f"(raise it or widen R3D_DUMP_RGBA_EVERY for more)"
                   if self._dump_capped else "")
            idx = (f" + {self._dump_n} index rows" if self._dump_index is not None
                   else " (index off)")
            print(f"rgba-dump: {self._dump_written} buffers{idx} -> "
                  f"{self._dump_dir_str}{cap}", file=sys.stderr, flush=True)


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
           ("d", frame, None) outro frame already composed on the GPU by the
                              render thread -> straight to the writer
           ("f", None, None)  frozen final gameplay frame re-push.

    On a FAILED play (``death_ms`` set) a death shade is applied to each
    gameplay frame AFTER the HUD, ramping over the ~1 s (``death_fade_ms`` of
    map time) ending at ``death_ms`` and held at its floor for the frozen tail.
    Passing plays pass ``death_ms=None`` and the death path never runs."""

    _QUEUE = 3

    def __init__(self, hud, writer, fl, perf=None, *, death_ms=None,
                 death_fade_ms=0.0, huds=None):
        self._hud = hud
        self._writer = writer
        self._fl = fl
        self._perf = perf
        self._death_ms = death_ms
        self._death_fade_ms = float(death_fade_ms)
        self.last_gameplay = None
        self._werr: BaseException | None = None
        # PARALLEL COMPOSITE (R3D_COMPOSITE_WORKERS=N, default 1 = old path).
        # The composite is the critical path at 4.84 ms/frame on one thread, and
        # on a free-threaded build the real Argon path scales 6.72x across
        # threads (measured). Two workers already drop it below the render
        # thread's 3.96 ms, which is the whole available win -- more workers
        # spend cores for no fps.
        #
        # Each worker owns its OWN DanserHud so the per-frame animation state
        # (_hp_last_t, _kc_*, _roll, ArgonHealth's damped _hp/_glow) is never
        # shared or raced. NOTE: with N>1 each HUD sees only every Nth frame, so
        # damped animations advance on a coarser clock -- output is NOT
        # byte-identical to N=1. Correct fix is a sequential scalar pre-pass
        # feeding stateless workers; this path exists to measure the ceiling.
        self._huds = list(huds) if huds else [hud]
        n = len(self._huds)
        self._q: "queue.Queue" = queue.Queue(maxsize=max(self._QUEUE, n * 2))
        self._seq_in = 0
        self._next_out = 0
        self._pending: dict = {}
        self._order_lock = threading.Lock()
        self._outro_pool = None
        if _OUTRO_WORKERS > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._outro_pool = ThreadPoolExecutor(
                max_workers=_OUTRO_WORKERS, thread_name_prefix="outro")
        self._threads = [
            threading.Thread(target=self._run, args=(i,),
                             name=f"hud-composite-{i}", daemon=True)
            for i in range(n)
        ]
        for t in self._threads:
            t.start()
        self._thread = self._threads[0]      # back-compat for close()

    def _emit_ordered(self, seq, frame):
        """Release finished frames to the writer in submission order."""
        with self._order_lock:
            self._pending[seq] = frame
            while self._next_out in self._pending:
                self._writer.push(self._pending.pop(self._next_out))
                self._next_out += 1

    def _flush_outro(self, batch, perf, pc):
        """Compute a batch of outro frames in parallel and emit in order."""
        if not batch:
            return
        _tl0 = time.perf_counter_ns() if _TL is not None else 0
        _tl_n = len(batch)
        _tl_sq = batch[0][0]
        t0 = pc() if perf is not None else 0.0
        if len(batch) == 1 or self._outro_pool is None:
            for sq, fn in batch:
                self._emit_ordered(sq, fn(self.last_gameplay))
        else:
            lg = self.last_gameplay
            futs = [(sq, self._outro_pool.submit(fn, lg)) for sq, fn in batch]
            for sq, fu in futs:
                self._emit_ordered(sq, fu.result())
        if perf is not None:
            perf["results"] += pc() - t0
        if _TL is not None:
            _TL.buf().append((_tl_sq, f"c:outro_x{_tl_n}", _tl0,
                              time.perf_counter_ns()))
        del batch[:]

    def _run(self, widx=0) -> None:
        pc = time.perf_counter
        perf = self._perf
        hud = self._huds[widx]
        _obatch: list = []
        while True:
            got = self._q.get()
            if got is None:
                self._flush_outro(_obatch, perf, pc)
                # wake the remaining workers, then exit
                for _ in range(len(self._threads) - 1):
                    self._q.put(None)
                return
            seq, item = got
            if self._werr is not None:
                continue                      # drain (never emit after error)
            kind, a, b = item
            try:
                if kind == "g":               # gameplay: flashlight + HUD
                    _tl_c0 = time.perf_counter_ns() if _TL is not None else 0
                    self._flush_outro(_obatch, perf, pc)
                    raw, scene = a, b
                    _tf = pc() if perf is not None else 0.0
                    if self._fl is not None:
                        raw = self._fl.apply(raw, scene)
                    t0 = pc() if perf is not None else 0.0
                    if perf is not None:
                        perf["fl"] += t0 - _tf
                    # R3D_ABLATE_HUD=1 -- MEASUREMENT ONLY. Skips the whole HUD
                    # so the frames are WRONG; it exists to price the HUD's
                    # share of the wall clock without guessing from per-stage
                    # timings, which lie in a contended pipeline.
                    out = raw if _ABLATE_HUD else hud.overlay(raw, scene)
                    if perf is not None:
                        perf["hud"] += pc() - t0
                    # FAIL death beat: on a failed play, ramp a desaturate +
                    # darken + red-tint shade over the whole composited frame
                    # (playfield AND HUD) across the final ~1 s ending at
                    # death, holding the floor for the frozen tail. Gated on
                    # death_ms, so passing renders never touch this.
                    _td = pc() if perf is not None else 0.0
                    if self._death_ms is not None:
                        p = death_progress(getattr(scene, "time_ms", 0),
                                           self._death_ms, self._death_fade_ms)
                        if p > 0.0:
                            out = apply_death(out, p)
                            if perf is not None:
                                perf["death_n"] += 1
                    if perf is not None:
                        _te = pc(); perf["death"] += _te - _td
                    self.last_gameplay = out
                    self._emit_ordered(seq, out)
                    if perf is not None:
                        perf["push"] += pc() - _te
                    if _TL is not None:
                        _TL.buf().append((seq, "c:composite", _tl_c0,
                                          time.perf_counter_ns()))
                elif kind == "r":             # outro: results screen
                    if self.last_gameplay is None:
                        raise CatchRenderError(
                            "outro before any gameplay frame")
                    _obatch.append((seq, a))
                    if len(_obatch) >= _OUTRO_WORKERS:
                        self._flush_outro(_obatch, perf, pc)
                elif kind == "d":             # already composed (GPU outro)
                    self._flush_outro(_obatch, perf, pc)
                    self._dcount = getattr(self, "_dcount", 0) + 1
                    self._emit_ordered(seq, a)
                else:                         # "f": frozen gameplay frame
                    _tl_f0 = time.perf_counter_ns() if _TL is not None else 0
                    self._flush_outro(_obatch, perf, pc)
                    if self.last_gameplay is None:
                        raise CatchRenderError(
                            "outro before any gameplay frame")
                    self._emit_ordered(seq, self.last_gameplay)
            except BaseException as e:  # noqa: BLE001 — surfaced on push()
                self._werr = e

    def push(self, item) -> None:
        """Queue one work item; re-raises this thread's error so a failed
        HUD/results pass (or a dead ffmpeg below it) fails the render loudly
        exactly like the old inline call did."""
        if self._werr is not None:
            raise self._werr
        self._q.put((self._seq_in, item))
        self._seq_in += 1

    def close(self) -> None:
        """Drain + join. Re-raises a pending worker error (except
        BrokenPipeError, which the caller surfaces via ffmpeg's exit code,
        matching the old inline flow)."""
        self._q.put(None)
        for t in self._threads:
            t.join()
        if self._pending:
            raise CatchRenderError(
                f"composite reorder buffer left {len(self._pending)} frames "
                f"unflushed (next_out={self._next_out})")
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
        from osu_catch_renderer._metal.beatmap.storyboard import parse_storyboard
        from osu_catch_renderer._metal.render.storyboard_engine import StoryboardEngine
        from osu_catch_renderer._metal.render.storyboard_render import StoryboardRenderer
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


def _su_mark(label):
    """R3D_CATCH_STARTUP=1: startup timeline, to see what the ~1.2s before the
    render loop is actually spent on."""
    import time as _t, sys as _s
    if not os.environ.get("R3D_CATCH_STARTUP"):
        return
    now = _t.perf_counter()
    prev = getattr(_su_mark, "_prev", None) or getattr(_su_mark, "_t0", now)
    if not hasattr(_su_mark, "_t0"):
        _su_mark._t0 = now; prev = now
    print("STARTUP %-34s %7.1f ms  (cum %7.1f ms)"
          % (label, 1e3*(now-prev), 1e3*(now-_su_mark._t0)), file=_s.stderr, flush=True)
    _su_mark._prev = now


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
    from osu_catch_renderer._metal.hud.fonts import set_skin_font
    # prefer a font bundled in the skin, else a robust system font (must run
    # before the HUD builds its glyph/text fonts below).
    set_skin_font(cfg.skin_dir)

    skin = None
    if cfg.skin_dir is not None:
        from osu_catch_renderer._metal.skin.skin import CatchSkin
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
    _su_mark("parse+setup -> CatchSim")
    sim = CatchSim(bm, frames, cfg, skin=skin, has_bg=bg is not None,
                   meta=meta, end_ms=sim_end_ms)
    # the PRIMARY player's sim — hitsounds come from ITS caught objects even
    # in a versus overlay (sim is rebound to CatchOverlaySim below).
    base_sim = sim
    _overlay_gray_keys = set()
    _player_catcher_bakes = []      # [(texture_key, rgba)] grayscaled + uploaded below
    if overlay_extra:
        from osu_catch_renderer._metal.render.overlay import CatchOverlaySim
        from osu_catch_renderer._metal.skin.skin import CatchSkin
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
            from osu_catch_renderer._metal.beatmap.score_fidelity import (compute_candidates,
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
    # STARTUP OVERLAP (R3D_PARALLEL_INIT=1): GL init costs ~495 ms and the
    # DanserHud glyph/atlas bake + leaderboard bake cost ~420 ms, and they are
    # completely independent -- GL never touches them. Run them concurrently
    # and the startup drops from ~975 ms to ~550 ms. The HUD build is pure
    # PIL/numpy so it is safe off the main thread; the GL context stays on the
    # main thread (contexts are thread-affine and the render loop runs here).
    _init_out = {}
    _init_th = None
    if os.environ.get("R3D_PARALLEL_INIT") == "1":
        def _bg_init(_meta=None):
            try:
                from osu_catch_renderer._metal.hud.hud import DanserHud
                _init_out["hud"] = DanserHud(
                    cfg.skin_dir, cfg.resolution, meta, bm, first, last,
                    cfg=cfg, default_skin_dir=cfg.default_skin_dir)
            except Exception:  # noqa: BLE001 - same fallback as the inline path
                _init_out["hud"] = None
            if cfg.show_results and getattr(cfg, "show_leaderboard", True):
                try:
                    from osu_catch_renderer._metal.hud.lb_cards import build_catch_board
                    _init_out["board"] = build_catch_board(cfg, meta, bm, replay_md5)
                except Exception as e:  # noqa: BLE001
                    print(f"[catch-renderer] leaderboard skipped: {e}",
                          file=sys.stderr)
                    _init_out["board"] = None
        _init_th = threading.Thread(target=_bg_init, name="catch-init", daemon=True)
        _init_th.start()

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
    _su_mark("sim build")
    # R3D_METAL=1 swaps the GL backend for the Metal one (Apple Silicon only).
    # Not byte-identical to GL -- different rasteriser, and no mipmaps yet.
    # Measured 10.9-19.6x on the draw+readback half of the frame.
    # NOTE `sys` is a LOCAL in render_core (an `import sys` sits in an except
    # branch), so touching the module-level name here raises UnboundLocalError.
    # Fourth time this trap has bitten in this file -- use an alias.
    import sys as _msys
    if os.environ.get("R3D_METAL") == "1" and _msys.platform == "darwin":
        from osu_catch_renderer._metal.render.metal.renderer import MetalSpriteRenderer
        renderer = MetalSpriteRenderer(w, h)
        # GPU health bar: DEFAULT OFF. The shader is correct (61-70 dB vs numpy,
        # max|diff|=2) but running it MID-HUD needs a round trip -- two full-frame
        # copies plus a waitUntilCompleted stall -- which costs MORE than the
        # 1.013 ms of bar work it removes. Measured: hud 2.608 -> 3.658 ms,
        # 323 -> 255 fps. It only pays PRE-READBACK inside the sprite pass, which
        # needs the HP damping advanced on the render thread (stateless pre-pass).
        if os.environ.get("R3D_METAL_HP", "0") == "1":
            from osu_catch_renderer._metal.hud import hud as _hudm
            from osu_catch_renderer._metal.argon.argon_health import ArgonHealth as _AHc
            _probe = _AHc(w, h)
            renderer.r.hp_init([_probe.main.d_perp, _probe.main.t_near,
                                _probe.glow.d_perp, _probe.glow.t_near])
            _hudm.set_metal_hp(renderer)
        if os.environ.get("R3D_METAL_DEATH", "1") == "1":
            from osu_catch_renderer._metal.render import death as _dth
            renderer.r.death_init()
            _dth.set_metal_backend(renderer)
        print(f"[catch-renderer] METAL backend: {renderer.device}",
              file=_msys.stderr)
    else:
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
        return np.clip(np.stack([lum, lum, lum, r[..., 3]], axis=-1), 0, 255).astype(np.uint8)
    for key in _overlay_gray_keys:
        renderer.upload_texture(f"{key}__ovl", _gray(skin.textures[key]))
    for key, ctex in _player_catcher_bakes:
        renderer.upload_texture(key, _gray(ctex))
    # osu!lazer ARGON catch objects (glowing wavy combo rings + white pip) and
    # the Argon catcher bar — uploaded regardless of skin: the skinless object
    # path, the caught-fruit plate pile, and the hit explosions all use them.
    from osu_catch_renderer._metal.skin.assets import (build_argon_textures, catch_glow_rgba, catch_beam_rgba,
                         bake_logo_tile)
    for key, rgba in build_argon_textures().items():
        renderer.upload_texture(key, rgba)
    from osu_catch_renderer._metal.skin.lazer_skin import argon_bar_cap_rgba
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
    _su_mark("SpriteRenderer (GL)")
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
            from osu_catch_renderer._metal.beatmap.hitsounds import build_hitsound_track, synth_style_for
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
    _su_mark("storyboard+hitsounds")
    # INLINE PREVIEW (R3D_PREVIEW_INLINE=1, default OFF): same contract as the GL
    # path (render/render.py) -- the ffmpeg that encodes the master also writes
    # the lean 720p30 preview embed as a second output, so it is finished the
    # moment the render is. Single renders only.
    preview_path = None
    if os.environ.get("R3D_PREVIEW_INLINE") == "1" and not overlay_extra:
        preview_path = output_path.parent / (output_path.stem + ".embed.mp4")
        print(f"[catch] inline preview -> {preview_path.name}",
              file=sys.stderr, flush=True)
    proc = _spawn_ffmpeg(cfg, output_path, audio, start_ms, rate, total_dur_s,
                         hitsound_wav=hits_wav, is_nc=is_nc,
                         preview_path=preview_path)
    # Argon is the DEFAULT skin: skinless renders stay all-Argon (parity with
    # the STD renderer). DanserHud now handles skin_dir=None; plain _Hud only if
    # DanserHud fails to build.
    _su_mark("ffmpeg spawn")
    if _init_th is not None:
        _init_th.join()                       # overlapped with GL init above
        hud = _init_out.get("hud") or _Hud(w, h, meta, bm)
    else:
        try:
            from osu_catch_renderer._metal.hud.hud import DanserHud
            hud = DanserHud(cfg.skin_dir, cfg.resolution, meta, bm, first, last,
                            cfg=cfg, default_skin_dir=cfg.default_skin_dir)
        except Exception:
            hud = _Hud(w, h, meta, bm)

    _su_mark("DanserHud build (glyphs/atlas)")
    from osu_catch_renderer._metal.hud.hud import draw_results
    # results-screen map leaderboard (parity with std): build + bake ONCE, up
    # front, so the outro just composites the pre-baked cards each frame. Fully
    # fail-soft — any problem leaves the plain results card (renders unchanged).
    baked_board = None
    if _init_th is not None:
        baked_board = _init_out.get("board")
    elif cfg.show_results and getattr(cfg, "show_leaderboard", True):
        try:
            from osu_catch_renderer._metal.hud.lb_cards import build_catch_board
            baked_board = build_catch_board(cfg, meta, bm, replay_md5)
        except Exception as e:  # noqa: BLE001 — a board must never break a render
            # NO local `import sys` here. It made `sys` a local for the WHOLE
            # of render_core, so every bare `sys.stderr` earlier in the
            # function raised UnboundLocalError -- which swallowed the real
            # error (an ENOSPC from a full disk surfaced as a `sys` traceback).
            # The module-level import is in scope; use it.
            print(f"[catch-renderer] leaderboard skipped: {e}", file=sys.stderr)
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
    _su_mark("ffmpeg spawn + HUD + board")
    _t_render0 = time.monotonic()
    _PERF = os.environ.get("R3D_CATCH_PERF")
    # NOTE these accumulate on THREE different threads (render / composite /
    # writer) which run concurrently, so they do NOT sum to wall time. Read
    # them as per-thread busy totals: the thread whose total approaches wall
    # is the critical path. `enq`/`enq_outro` are the render thread BLOCKED
    # handing work to the composite thread, i.e. direct backpressure evidence.
    _pt = {"scene": 0.0, "draw": 0.0, "read": 0.0, "hud": 0.0,
           "results": 0.0, "enq": 0.0, "enq_outro": 0.0, "pipe": 0.0,
           "drain": 0.0, "finalize": 0.0, "pipe_n": 0, "n_gameplay": 0,
           "n_outro": 0,
           # full-accounting stages (R3D_CATCH_PERF): begin/pend split out so
           # the per-frame sum can be forced to equal loop_wall.
           "begin": 0.0, "pend": 0.0, "loop_wall": 0.0,
           # unmeasured composite-thread work (real maps block in enq
           # even with the HUD and the encoder both removed)
           "fl": 0.0, "death": 0.0, "death_n": 0, "push": 0.0}
    _pc = time.perf_counter
    writer = _FrameWriter(proc, perf=_pt if _PERF else None)
    # GPU RGBA -> yuv420p (R3D_METAL_YUV=1). Takes swscale out of ffmpeg and
    # drops the pipe from 8.29 MB/frame to 3.11 MB. It is a port of swscale's
    # own path (2-pixel horizontal chroma sum + 8-tap vertical bicubic), so the
    # output tracks today's: Y bit-exact, chroma 99.8% exact / 1 LSB worst.
    if (os.environ.get("R3D_METAL_YUV") == "1"
            and os.environ.get("R3D_METAL") == "1"
            and sys.platform == "darwin"
            and hasattr(renderer, "r") and hasattr(renderer.r, "yuv_init")):
        try:
            renderer.r.yuv_init()
            writer.set_yuv_pipe(renderer.r.yuv_submit, renderer.r.yuv_acquire)
            print(f"[catch-renderer] GPU rgba->yuv420p: ffmpeg runs no swscale "
                  f"(ring {renderer.r._yring}, pipeline depth "
                  f"{renderer.r._ydepth})", file=sys.stderr)
        except Exception as _ye:      # noqa: BLE001 - fall back to rgba
            print(f"[catch-renderer] GPU yuv unavailable ({_ye}), rgba path",
                  file=sys.stderr)
    # PREBAKE THE ACCURACY ARCS on a spare thread while gameplay renders. The
    # arc bake was 2.841 ms of a 4.46 ms outro frame -- 64% of the featured
    # panel -- because its cache key changes every frame during the sweep, so
    # the cache never hit. ~69 buckets at ~2.8 ms is ~0.2 s of work, and the
    # render and writer threads are idle 88% / 75% of the wall.
    # Identical pixels by construction: bake_accuracy_arc is a pure function of
    # (px, bucket). A miss just falls back to baking, so this can never be
    # wrong, only ineffective. R3D_NO_ARC_PREBAKE=1 disables it.
    if cfg.show_results and os.environ.get("R3D_NO_ARC_PREBAKE") != "1":
        def _prebake_arcs():
            try:
                from osu_catch_renderer._metal.hud.lazer_results import prebake_for
                _t0 = _pc()
                n = prebake_for(meta, cfg.resolution[1], cfg.fps)
                if _PERF:
                    print(f"[catch] arc prebake: {n} buckets in "
                          f"{1e3 * (_pc() - _t0):.0f} ms", file=sys.stderr)
            except Exception as _pe:      # noqa: BLE001 - purely an optimisation
                if _PERF:
                    print(f"[catch] arc prebake skipped ({_pe})",
                          file=sys.stderr)
        threading.Thread(target=_prebake_arcs, name="arc-prebake",
                         daemon=True).start()

    # GPU RESULTS SCREEN (R3D_METAL_RESULTS=1). The outro is the last stage
    # still composited entirely in PIL on the composite thread -- measured
    # 4.12 ms/frame against ~1.7 for a gameplay frame that does far more,
    # because gameplay is on the GPU and this is not. Every element is a paste
    # of a baked tile, so the results screen's OWN layout code records a draw
    # list (see DrawRecorder) and the sprite pass replays it; the geometry
    # therefore cannot drift from the CPU path.
    #
    # Only the op >= 1.0 frames go this way: their base is pure black, so they
    # need nothing from the composite thread. The fade-in frames composite over
    # the frozen final gameplay frame, which only the compositor holds.
    _rs_gpu = None
    _rs_inflight = 0
    _rs_pushed = 0
    _rs_cap = 0.0
    _rs_sub = 0.0
    _rs_subm = 0
    _RS_DEPTH = max(1, int(os.environ.get("R3D_RESULTS_DEPTH", "3") or 3))
    # DEFAULT OFF. Converting straight off the render target moves 3.11 MB
    # instead of 19.7, but the compute kernel reads `bufs[s]` while the render
    # pass wrote `texes[s]` -- Metal does not track that a texture and its
    # backing buffer alias, so the kernel can read before the pass lands. Same
    # composition via the copies path and via this one produce DIFFERENT frame
    # hashes, which is that hazard showing. Fix: read through an rgba8Uint
    # texture view so the dependency is tracked.
    _rs_direct = (os.environ.get("R3D_RESULTS_DIRECT", "0") == "1"
                  and os.environ.get("R3D_METAL_YUV") == "1")
    # GPU RESULTS SCREEN (R3D_METAL_RESULTS=1). Every element of the outro is a
    # paste of a baked tile, so the results screen's OWN layout code records a
    # draw list (DrawRecorder) and the sprite pass replays it -- the geometry
    # cannot drift from the CPU path. Render and yuv conversion go in ONE
    # command buffer with no per-frame wait.
    #
    # Only op >= 1.0 frames qualify: their base is pure black, so they need
    # nothing from the composite thread, which owns the frozen final frame.
    #
    # BAKED ON A SPARE THREAD. Constructing the instance costs ~184 ms, and
    # doing it eagerly here cost more than the per-frame saving: 0.70 ms/frame
    # GPU vs ~1.4 ms wall CPU saves ~68 ms over 97 frames, against 184 ms of
    # serial startup -- a net 6% LOSS. The outro does not start for ~1.2 s.
    _rs_box = {}
    _rs_th = None
    if (cfg.show_results and os.environ.get("R3D_METAL_RESULTS") == "1"
            and os.environ.get("R3D_METAL") == "1"
            and sys.platform == "darwin"
            and hasattr(renderer, "draw_results_gpu")):
        def _rs_build():
            try:
                from osu_catch_renderer._metal.hud.lazer_results import (
                    CatchLazerResults)
                _t0 = _pc()
                _rs_box["scr"] = CatchLazerResults(
                    cfg.resolution, meta, bm, board=baked_board,
                    osu_path=osu_path, sim=sim, pp_override=cfg.pp_override,
                    sr_override=cfg.sr_override)
                _rs_box["ms"] = 1e3 * (_pc() - _t0)
            except Exception as _rse:      # noqa: BLE001 - CPU path is default
                _rs_box["err"] = _rse
        _rs_th = threading.Thread(target=_rs_build, name="results-bake",
                                  daemon=True)
        _rs_th.start()
        renderer.results_gpu_init()

    def _rs_ready():
        """The instance, or None. Joins the bake thread on first use -- by then
        gameplay has been running for ~1.2 s and it is long done."""
        nonlocal _rs_gpu, _rs_th
        if _rs_gpu is not None:
            return _rs_gpu
        if _rs_th is None:
            return None
        _rs_th.join()
        _rs_th = None
        if "err" in _rs_box:
            print(f"[catch-renderer] GPU results unavailable "
                  f"({_rs_box['err']})", file=sys.stderr)
            return None
        _rs_gpu = _rs_box.get("scr")
        if _rs_gpu is not None:
            print(f"[catch-renderer] GPU results screen: sprite-pass outro "
                  f"(bake {_rs_box.get('ms', 0):.0f} ms, off the render "
                  f"thread)", file=sys.stderr)
        return _rs_gpu

    def _rs_flush():
        """Drain every in-flight GPU outro frame and push them in submission
        order. Called before ANY other push, which is what makes the ordering
        structural rather than a set of special cases."""
        nonlocal _rs_inflight
        n = 0
        while _rs_inflight > 0:
            fr = (renderer.results_collect(1, force=True) if _rs_direct
                  else renderer.r.acquire(force=True))
            if fr is None:
                break
            _rs_inflight -= 1
            comp.push(("d", fr, None))
            n += 1
        return n

    pending = deque()          # scene snapshots awaiting their pixels

    # osu!catch Flashlight (FL, mod bit 1<<10): a soft-edged black vignette
    # centred on the catcher plate that shrinks with combo (see flashlight.py for
    # the ported lazer values). Post-pass over the composited playfield BEFORE the
    # HUD draws, so score/acc/combo/break overlays stay lit — lazer keeps the
    # Flashlight in the playfield layer with the HUD above it. Single renders only
    # (a versus overlay has many catchers); strictly gated on the FL bit, so
    # non-FL replays render byte-identically.
    fl = None
    # R3D_FORCE_FL=1 -- MEASUREMENT ONLY. Flashlight is gated on the
    # replay's mods bit, so a non-FL replay never exercises the `fl` stage;
    # ours read 0.000 ms all session. This forces it on so the pass can be
    # priced on content we already have. The output is NOT what that replay
    # should look like.
    if (has_flashlight(getattr(meta, "mods", 0))
            or os.environ.get("R3D_FORCE_FL") == "1") and not overlay_extra:
        fl = CatchFlashlight(break_env=getattr(sim, "_break_env", None))

    # compositing pipeline stage: flashlight + HUD + results run on their own
    # thread (strict FIFO), overlapping the GL draw/readback of later frames.
    # FAIL death beat (catch only): scale the ~2.5 s osu fail ramp (FAIL_FADE_MS)
    # by playback rate so a DT/HT fail still reads ~2.5 s of VIDEO. Only when `failed`.
    _death_arg = float(death_ms) if failed else None
    _death_fade = FAIL_FADE_MS * rate if failed else 0.0
    _nw = os.environ.get("R3D_COMPOSITE_WORKERS", "1").strip()
    _nw = int(_nw) if _nw.isdigit() and int(_nw) >= 1 else 1
    _huds = [hud]
    if _nw > 1:
        # one HUD per worker: the animation state must never be shared
        from osu_catch_renderer._metal.hud.hud import DanserHud as _DH
        for _ in range(_nw - 1):
            try:
                _huds.append(_DH(cfg.skin_dir, cfg.resolution, meta, bm, first,
                                 last, cfg=cfg,
                                 default_skin_dir=cfg.default_skin_dir))
            except Exception:  # noqa: BLE001 - fall back to fewer workers
                break
        import sys as _cwsys      # `sys` is shadowed as a local in this scope
        print(f"[catch] composite workers: {len(_huds)}", file=_cwsys.stderr,
              flush=True)
    comp = _CompositeWorker(hud, writer, fl, perf=_pt if _PERF else None,
                            death_ms=_death_arg, death_fade_ms=_death_fade,
                            huds=_huds)

    def _emit_gameplay(raw):
        scene = pending.popleft()
        _t0 = _pc()
        comp.push(("g", raw, scene))
        _pt["enq"] += _pc() - _t0
        _pt["n_gameplay"] += 1

    # ---- in-pass GPU HUD (R3D_METAL_HUD=1) ----------------------------------
    # Draws wedges + the health bar INSIDE the sprite pass, before readback, so
    # there is no CPU<->GPU round trip. Requires advancing the damped HP state on
    # THIS thread (the stateless pre-pass) -- the composite thread must not touch
    # argon_hp once this is on, which hud.set_gpu_hud_owns_bar() enforces by
    # skipping both elements there.
    # GPU flashlight. The CPU pass measured 9.616 ms/frame and took the render
    # from 605.8 to 97.2 fps -- below the pre-Metal baseline -- so this is by
    # far the biggest single win available, and it only shows up on FL replays.
    _fl_gpu = None
    if (fl is not None and os.environ.get("R3D_METAL") == "1"
            and os.environ.get("R3D_METAL_FL", "1") == "1"
            and sys.platform == "darwin"
            and hasattr(renderer, "r") and hasattr(renderer.r, "flash_inline")):
        try:
            from osu_catch_renderer._metal.render.flashlight import fl_gpu_params
            _fl_gpu = {"fl": fl, "params": fl_gpu_params}
            comp._fl = None    # the composite thread must NOT also apply it.
            # Clearing the local `fl` is too late: _CompositeWorker already
            # captured it at construction, ~30 lines above.
            print("[catch-renderer] GPU flashlight: in-pass multiply",
                  file=sys.stderr)
        except Exception as _fe:      # noqa: BLE001 - fall back to the CPU pass
            print(f"[catch-renderer] GPU flashlight unavailable ({_fe})",
                  file=sys.stderr)
            _fl_gpu = None

    _hud_gpu = None
    if (os.environ.get("R3D_METAL_HUD") == "1"
            and hasattr(renderer, "hud_gpu_init")
            and getattr(hud, "argon_hp", None) is not None
            and getattr(hud, "argon", None) is not None):
        try:
            from osu_catch_renderer._metal.hud import hud as _hudmod
            renderer.hud_gpu_init(hud.argon_hp, hud.argon)
            _hudmod.set_gpu_hud_owns_bar(True)
            _owned = "wedges + health bar"
            _ah = hud.argon
            if all(hasattr(_ah, a) for a in ("_sx0", "_sx1", "_sy0", "_sy1",
                                             "_graph_add", "op")):
                renderer.progress_gpu_init(_ah)
                _hudmod.set_gpu_hud_owns_progress(True)
                _owned += " + progress strip"
            _hud_gpu = {"hp": hud.argon_hp, "last_t": None,
                        "params": _hudmod.hp_gpu_params,
                        "ah": _ah,
                        "progress": getattr(renderer, "_pg_ready", False)}
            print(f"[catch-renderer] in-pass GPU HUD: {_owned}",
                  file=_msys.stderr)
        except Exception as _e:      # noqa: BLE001 - fall back to the CPU HUD
            print(f"[catch-renderer] GPU HUD unavailable ({_e}); CPU HUD",
                  file=_msys.stderr)
            _hud_gpu = None

    _loop_t0 = _pc()
    try:
        try:
            for i in range(n_frames):
                if i < gameplay_frames:
                    t = int(start_ms + i * map_step)
                    _t0 = _pc()
                    scene = sim.build_scene(t)
                    _tl_a = time.perf_counter_ns() if _TL is not None else 0
                    _t1 = _pc(); _pt["scene"] += _t1 - _t0
                    renderer.begin()
                    _tb = _pc(); _pt["begin"] += _tb - _t1
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
                    if (_fl_gpu is not None
                            and os.environ.get("R3D_METAL_FL_UNDER_HUD", "1") == "1"):
                        # DEFAULT ON (R3D_METAL_FL_UNDER_HUD=0 restores the old order): GL catch applies
                        # the flashlight BEFORE the HUD composite, so the bar /
                        # wedges / progress strip stay lit. Drawing the vignette
                        # last (below) matched this fork's own Metal+CPU-FL
                        # intermediate, not shipped GL output.
                        _fp = _fl_gpu["params"](_fl_gpu["fl"], scene, w, h)
                        if _fp is not None:
                            renderer.r.flash_inline(_fp)
                    if _hud_gpu is not None and not getattr(
                            scene, "overlay_board", None):
                        # The versus overlay is a different HUD entirely --
                        # no wedges, no bar -- so skip the whole in-pass
                        # block on those frames rather than drawing
                        # furniture that mode never shows.
                        # STATELESS PRE-PASS: advance the damped HP state here,
                        # on the render thread in frame order, then draw wedges +
                        # the bar into the OPEN sprite pass. No round trip.
                        _dt = (16.0 if _hud_gpu["last_t"] is None
                               else max(0.0, min(100.0, t - _hud_gpu["last_t"])))
                        _hud_gpu["last_t"] = t
                        renderer.draw_hud_gpu(_hud_gpu["params"](
                            _hud_gpu["hp"], scene.hp, _dt, w, h))
                        if _hud_gpu["progress"]:
                            _ahp = _hud_gpu["ah"]
                            _fr = ((t - _ahp.first_t)
                                   / max(_ahp.last_t - _ahp.first_t, 1.0))
                            _fr = (min(1.0, max(0.0, _fr))
                                   if t >= _ahp.first_t else 0.0)
                            renderer.draw_progress_gpu(_fr)
                    if (_fl_gpu is not None
                            and os.environ.get("R3D_METAL_FL_UNDER_HUD", "1") != "1"):
                        # LAST thing in the pass, AFTER the in-pass HUD. The CPU
                        # pass ran on the READBACK frame, which already contained
                        # the in-pass wedges/bar/strip, so it dimmed those too.
                        # Drawing the vignette before them instead left them lit
                        # and differed by ~50k pixels by up to 255 -- the math was right,
                        # the ORDER was wrong.
                        _fp = _fl_gpu["params"](_fl_gpu["fl"], scene, w, h)
                        if _fp is not None:
                            renderer.r.flash_inline(_fp)
                    if _GL_STUB:
                        renderer.gl_hud_stub(_GL_STUB)
                    _t2 = _pc(); _pt["draw"] += _t2 - _tb
                    if _TL is not None:
                        _tl_b = time.perf_counter_ns()
                        _TL.buf().append((i, "r:scene_draw", _tl_a, _tl_b))
                    pending.append(scene)
                    _tp = _pc(); _pt["pend"] += _tp - _t2
                    raw = renderer.read_rgb_async()
                    _pt["read"] += _pc() - _tp
                    if _TL is not None:
                        _tl_c = time.perf_counter_ns()
                        _TL.buf().append((i, "r:read", _tl_b, _tl_c))
                    if raw is not None:
                        _emit_gameplay(raw)
                else:
                    # gameplay -> outro boundary: flush the PBO ring first so
                    # last_gameplay is the true final gameplay frame and
                    # ordering is preserved across the boundary.
                    _td = _pc()
                    # Only drain while gameplay frames are still in flight.
                    # This runs on EVERY outro iteration, and with the GPU
                    # outro sharing the ring it was pulling outro frames back
                    # out and trying to emit them as gameplay -- `pending` is
                    # empty by then, so it died on popleft() at outro frame 73
                    # and truncated the render to 565 of 811 frames.
                    if pending:
                        for raw in renderer.read_drain():
                            _emit_gameplay(raw)
                    _pt["drain"] += _pc() - _td
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

                        _t0 = _pc()
                        _did_gpu = False
                        _rs_scr = _rs_ready() if op >= 1.0 else None
                        if (_rs_scr is not None
                                and _rs_scr.gpu_supported(op, age)):
                            try:
                                _tc0 = _pc()
                                _fa, _ops = _rs_scr.capture_ops(w, h, op, age)
                                _rs_cap += _pc() - _tc0
                                _tc1 = _pc()
                                if _rs_direct:
                                    # submit render+convert as ONE command
                                    # buffer and DO NOT WAIT. Waiting was the
                                    # whole problem: two serialised GPU round
                                    # trips per frame lost to 3 CPU workers
                                    # even after the host copies were gone.
                                    renderer.results_submit(_ops, 0.0, None)
                                    _rs_sub += _pc() - _tc1
                                    _rs_inflight += 1
                                    _rs_subm += 1
                                    _did_gpu = True
                                    while _rs_inflight >= _RS_DEPTH:
                                        _yb = renderer.results_collect(
                                            _RS_DEPTH, force=True)
                                        if _yb is None:
                                            break
                                        _rs_inflight -= 1
                                        comp.push(("d", _yb, None))
                                        _rs_pushed += 1
                                    _pt["enq_outro"] += _pc() - _t0
                                    _pt["n_outro"] += 1
                                    if progress_callback and i % cfg.fps == 0:
                                        progress_callback(int(i / n_frames * 100))
                                    continue
                                renderer.draw_results_gpu(_ops, 0.0, None)
                                renderer.r.commit()
                                _rs_inflight += 1
                                _rs_subm += 1
                                _did_gpu = True
                                # BATCHED, not ad-hoc. Frames stay in flight
                                # only while the NEXT frame is also GPU-bound;
                                # the batch is flushed (drained and pushed in
                                # order) before anything else can push. Ad-hoc
                                # draining let CPU frames jump ahead of
                                # in-flight GPU frames and misordered 173 of
                                # 811 frames -- ordering has to be structural.
                                if _rs_inflight >= _RS_DEPTH:
                                    _rs_pushed += _rs_flush()
                            except Exception as _rge:   # noqa: BLE001
                                _rs_pushed += _rs_flush()
                                if _PERF:
                                    print(f"[catch] GPU outro fell back "
                                          f"({_rge})", file=sys.stderr)
                                _rs_gpu = None
                        if not _did_gpu:
                            _rs_pushed += _rs_flush()
                            comp.push(("r", _results_frame, None))
                        _pt["enq_outro"] += _pc() - _t0
                    else:
                        _t0 = _pc()
                        _rs_pushed += _rs_flush()
                        comp.push(("f", None, None))
                        _pt["enq_outro"] += _pc() - _t0
                    _pt["n_outro"] += 1
                if progress_callback and i % cfg.fps == 0:
                    progress_callback(int(i / n_frames * 100))
            # GPU outro tail
            _rs_pushed += _rs_flush()
            if _rs_subm:
                print(f"[catch] GPU outro: submitted={_rs_subm} "
                      f"pushed={_rs_pushed} stranded={_rs_inflight} | "
                      f"capture_ops={1e3*_rs_cap/max(_rs_subm,1):.3f} ms/fr  "
                      f"submit+collect={1e3*_rs_sub/max(_rs_subm,1):.3f} ms/fr",
                      file=sys.stderr)
            # map end with no outro configured: flush the ring tail.
            for raw in renderer.read_drain():
                _emit_gameplay(raw)
            _pt["loop_wall"] = _pc() - _loop_t0
        except BrokenPipeError:
            _pt["loop_wall"] = _pc() - _loop_t0
            pass               # ffmpeg died — surfaced via ret below
    finally:
        # composite errors are re-raised AFTER the ffmpeg/GL cleanup below —
        # raising here would leak the ffmpeg child + GL context.
        _comp_err = None
        _tf = _pc()
        _fin = {}
        try:
            comp.close()
        except BaseException as e:  # noqa: BLE001 — deferred, never swallowed
            _comp_err = e
        _fin["comp_close"] = _pc() - _tf
        _t = _pc(); writer.close(); _fin["writer_close"] = _pc() - _t
        _t = _pc()
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        _fin["stdin_close"] = _pc() - _t
        _t = _pc(); ret = proc.wait(); _fin["ffmpeg_exit"] = _pc() - _t
        _pt["finalize"] += _pc() - _tf   # composite+writer queue drain + ffmpeg exit
        if os.environ.get("R3D_CATCH_STARTUP") or _PERF:
            import sys as _dsys       # `sys` is shadowed as a local in this scope
            print("DRAIN " + " ".join(f"{k}={1e3*v:.0f}ms" for k, v in _fin.items()),
                  file=_dsys.stderr, flush=True)
        renderer.release()
        # drop the temp hitsound WAV (R3D_CATCH_KEEP_HITS=1 keeps it for
        # alignment debugging/verification)
        if hits_wav is not None and not os.environ.get("R3D_CATCH_KEEP_HITS"):
            try:
                Path(hits_wav).unlink(missing_ok=True)
            except OSError:
                pass
        import sys as _rsys
        _wall = time.monotonic() - _t_render0
        if _PERF:
            import sys as _psys
            _ng = max(_pt["n_gameplay"], 1); _no = max(_pt["n_outro"], 1)
            _np = max(_pt["pipe_n"], 1)
            print("PERF " + " ".join(
                f"{k}={v:.2f}s" if isinstance(v, float) else f"{k}={v}"
                for k, v in _pt.items()), file=_psys.stderr, flush=True)
            # per-thread busy vs wall — the only correct way to read a
            # 3-thread pipeline (the raw totals above overlap in time).
            _rthread = (_pt["scene"] + _pt["draw"] + _pt["read"]
                        + _pt["enq"] + _pt["enq_outro"] + _pt["drain"])
            _cthread = _pt["hud"] + _pt["results"]
            print("PERF-THREADS wall=%.2fs render=%.2fs(%.0f%%) "
                  "composite=%.2fs(%.0f%%) writer=%.2fs(%.0f%%)"
                  % (_wall, _rthread, 100*_rthread/_wall,
                     _cthread, 100*_cthread/_wall,
                     _pt["pipe"], 100*_pt["pipe"]/_wall),
                  file=_psys.stderr, flush=True)
            _ng = max(1, _pt.get("n_gameplay", 0))
            _no = max(1, _pt.get("n_outro", 0))
            _acc = sum(_pt.get(k, 0.0) for k in
                       ("scene","begin","draw","pend","read","enq",
                        "enq_outro","drain"))
            _lw = _pt.get("loop_wall", 0.0)
            print("PERF-ACCOUNT loop_wall=%.2fs accounted=%.2fs residual=%.2fs (%.1f%%) "
                  "| gameplay=%d outro=%d"
                  % (_lw, _acc, _lw - _acc,
                     100.0 * (_lw - _acc) / _lw if _lw else 0.0, _ng, _no))
            _st = [_pt.get(k, 0.0) / _ng * 1e3 for k in
                   ("scene","begin","draw","pend","read","enq")]
            print("PERF-STAGES/gameplay-frame scene=%.3f begin=%.3f draw=%.3f "
                  "pend=%.3f read=%.3f enq=%.3f  sum=%.3f ms"
                  % (*_st, sum(_st)))
            print("PERF-COMPOSITE/frame fl=%.3f hud=%.3f death=%.3f (n=%d) "
                  "push=%.3f  => %.3f ms"
                  % (_pt.get("fl",0)/_ng*1e3, _pt.get("hud",0)/_ng*1e3,
                     _pt.get("death",0)/_ng*1e3, _pt.get("death_n",0),
                     _pt.get("push",0)/_ng*1e3,
                     (_pt.get("fl",0)+_pt.get("hud",0)+_pt.get("death",0)
                      +_pt.get("push",0))/_ng*1e3))
            print("PERF-OUTRO/frame enq_outro=%.3f ms over %d frames = %.2fs total"
                  % (_pt.get("enq_outro", 0.0) / _no * 1e3, _no,
                     _pt.get("enq_outro", 0.0)))
            try:
                from osu_catch_renderer._metal.render.gl import read_perf_report, _READ_PERF
                if _READ_PERF:
                    print(read_perf_report())
            except Exception:
                pass
            print("PERF-MS/FRAME scene=%.3f draw=%.3f read=%.3f | "
                  "hud=%.3f(n=%d) results=%.3f(n=%d) | pipe=%.3f(n=%d) | "
                  "enq_block=%.3f enq_outro_block=%.3f"
                  % (1e3*_pt["scene"]/_ng, 1e3*_pt["draw"]/_ng, 1e3*_pt["read"]/_ng,
                     1e3*_pt["hud"]/_ng, _ng, 1e3*_pt["results"]/_no, _no,
                     1e3*_pt["pipe"]/_np, _np,
                     1e3*_pt["enq"]/_ng, 1e3*_pt["enq_outro"]/_no),
                  file=_psys.stderr, flush=True)
        if os.environ.get("R3D_HUD_PERF"):
            try:
                from osu_catch_renderer._metal.hud.hud import hud_perf_report
                hud_perf_report()
            except Exception:  # noqa: BLE001
                pass
        if _TL is not None:
            try:
                _n = _TL.dump(_TIMELINE_PATH)
                print(f"[catch] timeline: {_n} rows -> {_TIMELINE_PATH}",
                      file=sys.stderr)
            except Exception as _tle:      # noqa: BLE001 - never break a render
                print(f"[catch] timeline dump failed: {_tle}", file=sys.stderr)
        if os.environ.get("R3D_RESULTS_PERF"):
            try:
                from osu_catch_renderer._metal.hud.lazer_results import CatchLazerResults
                CatchLazerResults._rp_report()
            except Exception:  # noqa: BLE001
                pass
        if os.environ.get("R3D_DRAW_COUNT") == "1":
            import sys as _dcs
            from osu_catch_renderer._metal.render.gl import _DRAW_STATS as _ds
            _f = max(_ds[0], 1)
            print("DRAW-STATS frames=%d sprites=%.1f/frame (max %d) "
                  "texkeys=%.1f/frame (max %d) additive=%.1f/frame"
                  % (_ds[0], _ds[1]/_f, _ds[2], _ds[3]/_f, _ds[4], _ds[5]/_f)
                  + "  ORDER-PRESERVING runs=%.1f/frame (max %d)"
                  % (_ds[6]/_f, _ds[7]),
                  file=_dcs.stderr, flush=True)
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
        raise CatchRenderError(f"ffmpeg exited {ret}\n{tail}")
    if os.environ.get("R3D_NULL_SINK") == "1":
        # Profiling sink writes no file by design; the size check would raise on
        # every run and mask a REAL failure behind an expected one.
        return output_path
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
    if _ffmpeg_has("h264_nvenc"):
        return "h264_nvenc", None
    dev = cfg.encoder_device or "/dev/dri/renderD128"
    if Path(dev).exists() and _ffmpeg_has("h264_vaapi"):
        return "h264_vaapi", dev
    # macOS render nodes (Apple Silicon pool): VideoToolbox is the platform HW
    # encoder, the darwin analogue of nvenc/vaapi -- but unlike nvenc it does
    # NOT win the auto-probe, because measured on an M1 Max it loses on BOTH
    # axes at matched bitrate: SSIM 0.828 @ 92 fps vs libx264 veryfast's 0.893
    # @ 133 fps (1080p60, high-entropy bg, vs the pre-encode RGBA as truth).
    # Two reasons: the pipeline is render/composite-bound around 140 fps so the
    # CPU encode is effectively free, and VT's block is tuned for low-power
    # realtime capture rather than quality-per-bit (it also pays an rgba->nv12
    # swscale + per-frame session sync that x264 avoids).
    # So HW encode here is OPT-IN: set R3D_MAC_HW_ENCODE=1 (or pass
    # --encoder h264_videotoolbox) to take it. Worth revisiting when a Mac node
    # runs several renders concurrently -- VT is a separate fixed-function block
    # and does not contend for the cores the renderer needs.
    if (sys.platform == "darwin"
            and os.environ.get("R3D_MAC_HW_ENCODE", "") == "1"
            and _ffmpeg_has("h264_videotoolbox")):
        return "h264_videotoolbox", None
    return "libx264", None


def _ffmpeg_has(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        return False
    return name in out


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


def _preview_video_bps(total_dur_s: "float | None") -> int:
    """Video bitrate of the lean preview embed. Mirrors the contributor
    client's makeEmbedVariant (and the bot's _transcode_embed_unbounded):
    ~1.4 Mbps, lowered on long maps so the file stays <= ~24 MiB, floor 500k."""
    vbps = 1_400_000
    if total_dur_s and total_dur_s > 0:
        vbps = int(24 * 1024 * 1024 * 8 / total_dur_s) - 128_000
        vbps = max(500_000, min(1_400_000, vbps))
    return vbps


def _spawn_ffmpeg(cfg: RenderConfig, output_path: Path, audio: Path | None,
                  start_ms: int, rate: float = 1.0, total_dur_s: float | None = None,
                  hitsound_wav: Path | None = None, is_nc: bool = False,
                  preview_path: "Path | None" = None):
    w, h = cfg.resolution
    enc, dev = _probe_encoder(cfg)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if enc == "h264_vaapi" and dev:
        cmd += ["-vaapi_device", dev]
    # RGBA ZERO-COPY PIPELINE: the frame producer hands the GL readback
    # buffer straight down the pipe (no 24<->32-bit repack). rgba input
    # yields BIT-IDENTICAL yuv420p to rgb24 (verified with framemd5);
    # the alpha byte is ignored by the encoder.
    # With the GPU converter on (R3D_METAL_YUV=1) we hand ffmpeg planar
    # yuv420p, so swscale never runs and the pipe carries 3.11 MB instead of
    # 8.29 MB per frame. Measured: 1.26x on the encoder side alone.
    _in_pix = ("yuv420p" if (os.environ.get("R3D_METAL_YUV") == "1"
                             and os.environ.get("R3D_METAL") == "1"
                             and sys.platform == "darwin") else "rgba")
    cmd += ["-f", "rawvideo", "-pix_fmt", _in_pix, "-s", f"{w}x{h}", "-r", str(cfg.fps),
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

    # video codec + pixel path
    _v0 = len(cmd)          # master video args start here (inline preview)
    if enc == "h264_vaapi":
        _vb = str(cfg.video_bitrate) if cfg.video_bitrate else "8M"
        cmd += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-b:v", _vb]
    elif enc == "h264_nvenc":
        # Resolution-scaled bitrate ladder (was flat 8M) -- R3D cross-engine
        # NVENC policy; see nvenc_target_bps above.
        _tgt = cfg.video_bitrate or nvenc_target_bps(w, h, cfg.fps)
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
                "-b:v", str(_tgt), "-maxrate", str(int(_tgt * 1.5)),
                "-bufsize", str(_tgt * 2)]
    elif enc == "hevc_videotoolbox":
        # macOS HW HEVC — candidate for a smaller h265 MASTER (delivery stays
        # h264 for browser/Discord). Same ladder as the other HW encoders.
        # R3D_HEVC_10BIT=1 selects the Main10 / p010le path: 10-bit costs the
        # HW block nothing and removes 8-bit banding in the HUD's gradients,
        # which is the main thing a master wants to preserve.
        # `-tag:v hvc1` (not the default hev1) is required for QuickTime /
        # Apple-ecosystem playback of an MP4-contained HEVC master.
        _tgt = cfg.video_bitrate or nvenc_target_bps(w, h, cfg.fps)
        _ten = os.environ.get("R3D_HEVC_10BIT", "") == "1"
        cmd += ["-c:v", "hevc_videotoolbox",
                "-pix_fmt", "p010le" if _ten else "yuv420p"]
        if _ten:
            cmd += ["-profile:v", "main10"]
        cmd += ["-b:v", str(_tgt), "-maxrate", str(int(_tgt * 1.5)),
                "-bufsize", str(_tgt * 2), "-allow_sw", "1", "-tag:v", "hvc1"]
    elif enc == "libx265":
        # CPU HEVC. `veryfast` to match the libx264 baseline preset so the
        # comparison is preset-for-preset rather than codec-vs-preset.
        _pf = "yuv420p10le" if os.environ.get("R3D_HEVC_10BIT", "") == "1" else "yuv420p"
        cmd += ["-c:v", "libx265", "-preset", "veryfast", "-pix_fmt", _pf,
                "-tag:v", "hvc1"]
        if cfg.video_bitrate:
            _vb = int(cfg.video_bitrate)
            cmd += ["-b:v", str(_vb), "-maxrate", str(int(_vb * 1.5)),
                    "-bufsize", str(_vb * 2)]
        else:
            cmd += ["-crf", "23"]
        cmd += ["-x265-params", "log-level=error"]
    elif os.environ.get("R3D_MAC_PRORES") == "1":
        # TWO-TIER (measurement only, NOT a deliverable): ProRes 422 LT on the
        # M1 Pro/Max ProRes accelerator -- 606 fps at 1080p60 vs libx264
        # veryfast's 426, at near-zero CPU. yuv422p10le is ProRes' NATIVE
        # format; forcing yuv420p adds a swscale pass and drops it to 281 fps.
        # ~22 MB/s vs h264's 0.6 MB/s, so this is a fast intermediate that a
        # later pass transcodes, never the file a viewer receives.
        # yuv422p10le is NOT in prores_videotoolbox's format list on ffmpeg
        # 8.1 (`-h encoder=prores_videotoolbox`) -- asking for it fails the
        # filter graph with -22 and writes a zero-byte file. p210le is the
        # supported 4:2:2 10-bit format and measured fastest of the ones that
        # do work (120 frames: p210le 0.34s, bgra 0.49s, yuv420p 0.52s).
        # R3D_MAC_PRORES_PROFILE: 0=Proxy 1=422 LT 2=422 3=422 HQ.
        cmd += ["-c:v", "prores_videotoolbox", "-profile:v",
                os.environ.get("R3D_MAC_PRORES_PROFILE", "1")]
        if _in_pix != "yuv420p":
            # rgba in -> let swscale make 4:2:2 10-bit, the fastest of the
            # formats this encoder actually accepts. With the GPU converter on,
            # the input is ALREADY yuv420p (which ProRes takes natively), so
            # asking for p210le would put a swscale pass back in.
            cmd += ["-pix_fmt", "p210le"]
    elif enc == "h264_videotoolbox":
        # VideoToolbox is a HW encoder like NVENC and has no CRF mode, so it is
        # driven by the SAME resolution-scaled ladder (nvenc_target_bps) rather
        # than falling through to libx264's -crf -- that keeps a Mac node's
        # bitrate policy identical to the rest of the fleet. `-allow_sw 1` lets
        # VT drop to its own software path instead of failing the render if no
        # HW encode session is available.
        _tgt = cfg.video_bitrate or nvenc_target_bps(w, h, cfg.fps)
        cmd += ["-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p",
                "-b:v", str(_tgt), "-maxrate", str(int(_tgt * 1.5)),
                "-bufsize", str(_tgt * 2), "-allow_sw", "1"]
    else:
        if cfg.video_bitrate:
            _vb = int(cfg.video_bitrate)
            cmd += ["-c:v", "libx264", "-preset", _X264_PRESET, "-pix_fmt", "yuv420p",
                    "-b:v", str(_vb), "-maxrate", str(int(_vb * 1.5)),
                    "-bufsize", str(_vb * 2)]
        else:
            # crf 23 = the shipped GL path's value (R3D size policy #87). This
            # backend forked before that change and was still on crf 20.
            cmd += ["-c:v", "libx264", "-preset", _X264_PRESET, "-pix_fmt", "yuv420p", "-crf", "23"]
        # R3D_X264_PARAMS: extra -x264-params, ":"-joined. The preset ladder
        # is coarse -- veryfast to ultrafast is +32% end-to-end for 3.2x the
        # file -- so the useful points are between them: ultrafast with cabac
        # and 8x8dct put BACK costs ~13% of its speed and saves ~20% of its
        # bitrate (measured standalone on 300 real frames).
        _xp = os.environ.get("R3D_X264_PARAMS", "").strip()
        if _xp:
            cmd += ["-x264-params", _xp]
        # R3D_X264_THREADS=n caps the encoder's thread count. x264 defaults to
        # one thread per core PER RENDER, so N concurrent renders spawn N*cores
        # encoder threads and contend with the engine's own. Profiling knob.
        _xt = os.environ.get("R3D_X264_THREADS", "").strip()
        if _xt.isdigit():
            cmd += ["-threads", _xt]

    _a0 = len(cmd)          # master audio args start here (inline preview)
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
            cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]"]
        else:
            af = _audio_filter(start_ms, rate, total_dur_s,
                               music_volume=cfg.music_volume,
                               general_volume=cfg.general_volume,
                               audio_offset_ms=cfg.audio_offset_ms, is_nc=is_nc,
                               pre_normalized=pre)
            if af:
                cmd += ["-af", af]
        # -ar 48000 as on the GL path: without it the inline-loudnorm path
        # (192 kHz internally) encodes a 96 kHz AAC master.
        cmd += ["-c:a", "aac", "-ar", "48000", "-b:a", "192k"]
        # `-shortest` makes ffmpeg hold the audio output until it learns the
        # video length, which defers the ENTIRE audio filtergraph to after the
        # last video frame (~950 ms of dead time at 1080p; drops to 28 ms with
        # no audio at all). apad=whole_dur already pins the audio to exactly
        # total_dur_s, so it should be redundant -- R3D_NO_SHORTEST=1 tests that.
        # `-shortest` costs ~950 ms: it holds the whole audio filtergraph until
        # it learns the video length. But we ALREADY know that length
        # (total_dur_s = n_frames / fps), so an explicit output duration gets
        # the same trim without the dependency -- and lets ffmpeg write audio
        # progressively instead of flushing it all after the last frame.
        # DEFAULT IS PLATFORM-CONDITIONAL (R3D's session, 2026-09-25):
        # `-shortest` stalls the audio filtergraph ~950 ms on macOS/ffmpeg 8.1,
        # but on Linux/ffmpeg 6.1.1 it finalises in ~55 ms either way -- so the
        # `-t` variant buys nothing there AND costs byte-identity (+512 silence
        # samples). Suspected ffmpeg 8.x behaviour change rather than a platform
        # difference, so a Linux node that moves to ffmpeg 8.x may want
        # R3D_AUDIO_TAIL=explicit_t explicitly.
        _default_tail = "explicit_t" if sys.platform == "darwin" else "shortest"
        _mode = os.environ.get("R3D_AUDIO_TAIL", _default_tail)
        if _mode == "shortest" or not (total_dur_s and total_dur_s > 0):
            cmd += ["-shortest"]
        elif _mode == "explicit_t":
            _tv = os.environ.get("R3D_AUDIO_T")      # measurement override
            cmd += ["-t", _tv if _tv else f"{total_dur_s:.6f}"]
        # "none" = neither (measurement only; leaves an AAC-granularity tail)

    if preview_path is not None:
        # TWO OUTPUTS FROM ONE PROCESS (port of the GL path's inline preview).
        # Everything above built the master's args exactly as without the
        # preview; take them back off `cmd` and re-emit them behind a
        # filter_complex that `split`s the frame pipe (read ONCE) into the
        # master encoder and a 720p30 libx264 preview, and `asplit`s the
        # master's own audio graph, with the client's loudness pass on the
        # preview branch only.
        vc, aargs = cmd[_v0:_a0], cmd[_a0:]
        del cmd[_v0:]
        vm_tail = "null"
        if "-vf" in vc:                 # vaapi: "-vf format=nv12,hwupload"
            _i = vc.index("-vf")
            vm_tail = vc[_i + 1]
            vc = vc[:_i] + vc[_i + 2:]
        pfps = min(30, int(round(float(cfg.fps))))
        graph = [f"[0:v]split=2[vm0][vp0];[vm0]{vm_tail}[vm];"
                 f"[vp0]scale=-2:720,fps={pfps}[vp]"]
        atail: list = []
        if audio is not None:
            if aargs[:1] == ["-filter_complex"]:
                # song + hitsounds: [graph, -map 0:v, -map [aout]] then codec
                graph.append(aargs[1])
                atail = aargs[6:]
            elif aargs[:1] == ["-af"]:
                graph.append(f"[1:a]{aargs[1]}[aout]")
                atail = aargs[2:]
            else:
                graph.append("[1:a]anull[aout]")
                atail = aargs
            # PIN THE SHARED BRANCH BEFORE THE SPLIT when the master is a
            # 48 kHz stream (loudnorm-cache path, or an explicit -ar 48000 --
            # which the master now always has). The preview's loudnorm runs at
            # 192 kHz and otherwise wins format negotiation back THROUGH asplit,
            # so the master's own chain would run at 192 kHz and its audio bytes
            # change.
            _pin = ("aformat=sample_rates=48000,"
                    if (prenorm is not None or "48000" in atail) else "")
            graph.append(f"[aout]{_pin}asplit=2[am][ap0];"
                         "[ap0]loudnorm=I=-18:TP=-1.5:LRA=11[ap]")
        cmd += ["-filter_complex", ";".join(graph)]
        # output 1: the master, args exactly as without the preview
        cmd += ["-map", "[vm]"] + (["-map", "[am]"] if audio is not None else [])
        cmd += vc + atail
        if os.environ.get("R3D_NO_FASTSTART") != "1":
            cmd += ["-movflags", "+faststart"]
        cmd += [str(output_path)]
        # output 2: the preview. libx264 always (a second HW session can fail
        # to open and one failed output kills the render). Same tail rule as the
        # master (-t / -shortest) so both files have the master's duration.
        vbps = _preview_video_bps(total_dur_s)
        _ptail = [x for x in atail[atail.index("192k") + 1:]] if "192k" in atail else []
        cmd += ["-map", "[vp]"] + (["-map", "[ap]"] if audio is not None else [])
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-b:v", str(vbps), "-maxrate", str(int(vbps * 1.25)),
                "-bufsize", str(vbps * 2), "-g", "30",
                "-threads", str(max(2, min(4, (os.cpu_count() or 4) - 2)))]
        if audio is not None:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000"] + _ptail
        cmd += ["-movflags", "+faststart", str(preview_path)]

    # web-streamable: move the moov atom to the front so browsers/iOS can
    # play before the whole file downloads (loudnorm re-adds this, but be
    # robust if that post-step is skipped/fails).
    # +faststart makes ffmpeg REWRITE the finished mp4 to move the moov atom to
    # the front (needed for progressive playback in browsers / Discord embeds).
    # R3D_NO_FASTSTART=1 is a measurement knob only -- dropping it breaks inline
    # playback, so it must not become the default.
    if preview_path is None:
        if os.environ.get("R3D_NO_FASTSTART") != "1":
            cmd += ["-movflags", "+faststart"]
        cmd += [str(output_path)]
    import tempfile
    if os.environ.get("R3D_NULL_SINK") == "1":
        # `cat` drains stdin and discards: identical pipe/subprocess structure,
        # no swscale and no encoder. The delta vs a real run IS the encode cost.
        cmd = ["cat"]
    errf = tempfile.NamedTemporaryFile(
        prefix="catch_ffmpeg_", suffix=".log", delete=False, mode="w+",
    )
    # macOS: the F_SETPIPE_SZ growth below is Linux-only, so a Mac node is
    # stuck on the 64 KiB default pipe -- 126 kernel handoffs for one 1080p
    # RGBA frame. A unix socketpair CAN be grown (SO_SNDBUF, capped by
    # kern.ipc.maxsockbuf) and ffmpeg reads "pipe:0" from any fd, socket
    # included. Measured on an M1 Max feeding 1080p RGBA to the engine's own
    # ffmpeg line: pipe 256 fps -> socketpair 406 fps, against a 419 fps
    # file-fed ceiling. Saturates at 1 MiB; 4/8 MiB add nothing.
    # Opt-in while it is being characterised: R3D_MAC_SOCKET_PIPE=1.
    if sys.platform == "darwin" and os.environ.get("R3D_MAC_SOCKET_PIPE") == "1":
        import socket as _sock
        _par, _chi = _sock.socketpair(_sock.AF_UNIX, _sock.SOCK_STREAM)
        for _s, _opt in ((_par, _sock.SO_SNDBUF), (_chi, _sock.SO_RCVBUF)):
            try:
                _s.setsockopt(_sock.SOL_SOCKET, _opt, 1 << 20)
            except OSError:
                pass          # keep the default buffer; still correct
        proc = subprocess.Popen(cmd, stdin=_chi.fileno(), stderr=errf,
                                stdout=subprocess.DEVNULL, bufsize=0)
        _chi.close()

        class _SockStdin:
            """File-like shim over the socket. `sendall` is deliberate: a raw
            SocketIO.write may write PARTIALLY and return a short count, and
            _FrameWriter ignores write()'s return value -- that would silently
            truncate a frame. sendall loops or raises."""
            def __init__(self, sk): self._sk = sk
            def write(self, b): self._sk.sendall(b); return len(b)
            def flush(self): pass
            def fileno(self): return self._sk.fileno()
            def close(self):
                try: self._sk.shutdown(_sock.SHUT_WR)
                except OSError: pass
                self._sk.close()
        proc.stdin = _SockStdin(_par)
        proc._catch_errlog = errf.name  # type: ignore[attr-defined]
        return proc
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf,
                            stdout=subprocess.DEVNULL, bufsize=0)
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
        if os.environ.get("R3D_AUDIO_ATRIM") == "1":
            # audio-only trim: -t would cut the VIDEO too (measured: loses a
            # frame). atrim lives in the audio chain so video is untouched.
            parts.append(f"atrim=end={total_dur_s:.6f}")
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
        if os.environ.get("R3D_AUDIO_ATRIM") == "1":
            # audio-only trim: -t would cut the VIDEO too (measured: loses a
            # frame). atrim lives in the audio chain so video is untouched.
            tail.append(f"atrim=end={total_dur_s:.6f}")
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
        from osu_catch_renderer._metal.hud.hud import _img_from_rgb, _img_out
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


from osu_catch_renderer._metal.hud.fonts import font as _font  # skin-aware, host-robust font resolver
