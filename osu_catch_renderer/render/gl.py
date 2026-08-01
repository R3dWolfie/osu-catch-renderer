"""Minimal moderngl sprite batch for the catch renderer.

Owns a standalone EGL context and an offscreen RGBA framebuffer. Draws
textured/solid quads with straight-alpha blending in painter's order, then
reads back tightly-packed RGB24 for the ffmpeg pipe. Deliberately tiny and
self-contained so it can be discarded when the VRender branch takes over.
"""
from __future__ import annotations

import numpy as np

try:
    import moderngl
except Exception as e:  # noqa: BLE001
    raise RuntimeError("moderngl is required for the catch renderer") from e

from osu_catch_renderer.beatmap.models import Sprite

_VERT = """
#version 330
in vec2 in_pos;      // unit quad corner [-0.5,0.5]
in vec2 in_uv;
uniform vec2 u_screen;   // (w, h) in px
uniform vec2 u_center;   // sprite center in px (origin top-left)
uniform vec2 u_size;     // sprite w,h in px
uniform float u_rot;     // radians
uniform vec2 u_uv_off;   // texture UV offset (storyboard flip mirroring)
uniform vec2 u_uv_scale; // texture UV scale  (default (1,1); -1 mirrors an axis)
out vec2 v_uv;
void main() {
    vec2 p = in_pos * u_size;
    float c = cos(u_rot), s = sin(u_rot);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    vec2 px = u_center + p;
    // px -> clip, with y flipped (top-left origin)
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    // identity default ((0,0)/(1,1)) == `in_uv`, so non-storyboard sprites are
    // sampled bit-identically; a storyboard flip passes off=1,scale=-1 per axis.
    v_uv = in_uv * u_uv_scale + u_uv_off;
}
"""

_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec4 u_color;
out vec4 f_color;
void main() {
    vec4 t = texture(u_tex, v_uv);
    f_color = t * u_color;
}
"""


class SpriteRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Honor R3D_EGL_DEVICE_INDEX so renders pin to the right GPU (pool
        # isolation: e.g. 1070=index 1 for Pool B). EGL ignores
        # CUDA_VISIBLE_DEVICES, so the device must be selected explicitly.
        import os
        dev = os.environ.get("R3D_EGL_DEVICE_INDEX", "").strip()
        if dev.isdigit():
            self.ctx = moderngl.create_context(
                standalone=True, backend="egl", device_index=int(dev))
        else:
            self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        # unit quad centered at origin, uv 0..1
        # in_pos.y=-0.5 renders at screen-top -> texture-top (v=0); in_pos.y=+0.5
        # renders at screen-bottom -> texture-bottom (v=1). (Matters for
        # vertically-asymmetric sprites like the catcher.)
        quad = np.array([
            -0.5, -0.5, 0.0, 0.0,
             0.5, -0.5, 1.0, 0.0,
            -0.5,  0.5, 0.0, 1.0,
             0.5,  0.5, 1.0, 1.0,
        ], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")],
        )
        self.prog["u_screen"].value = (float(width), float(height))
        # PERF: resolve uniform handles once (prog[...] is a dict lookup +
        # wrapper build per call) and set the sampler unit a single time —
        # it is always texture unit 0. GL state/output is unchanged.
        self.prog["u_tex"].value = 0
        self._u_color = self.prog["u_color"]
        self._u_center = self.prog["u_center"]
        self._u_size = self.prog["u_size"]
        self._u_rot = self.prog["u_rot"]
        # UV flip uniforms (storyboard mirroring). Bound to identity so every
        # gameplay/HUD sprite (uv_off=(0,0), uv_scale=(1,1)) samples exactly
        # `in_uv` — bit-identical to before. _draw_one only re-binds them when a
        # sprite's uv differs from what's currently bound, so a flag-off render
        # never touches them after this init (zero per-sprite overhead too).
        self._u_uv_off = self.prog["u_uv_off"]
        self._u_uv_scale = self.prog["u_uv_scale"]
        self._u_uv_off.value = (0.0, 0.0)
        self._u_uv_scale.value = (1.0, 1.0)
        self._cur_uv_off = (0.0, 0.0)
        self._cur_uv_scale = (1.0, 1.0)

        rb = self.ctx.renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer(color_attachments=[rb])
        self._textures: dict[str, moderngl.Texture] = {}
        self._white = self._make_texture_rgba(np.full((1, 1, 4), 255, dtype="u1"))

        # async PBO ring state (read_rgb_async / read_drain) — ported from
        # the std renderer's proven pipeline (osu_std_renderer/render/gl.py)
        self._pbos: list["moderngl.Buffer"] | None = None
        self._pbo_head = 0
        self._pbo_tail = 0

    # --- texture management ---------------------------------------------------

    def upload_texture(self, key: str, rgba: np.ndarray,
                       clamp: bool = False, mipmaps: bool = True) -> None:
        """rgba: HxWx4 uint8 array (top-left origin). clamp=True sets
        clamp-to-edge wrapping (the storyboard samples with a flipped UV that
        can graze the edge texel; repeat wrap would wrap the far edge in).
        mipmaps default True — existing callers pass neither and are unchanged
        (mipmapped LINEAR, repeat wrap, exactly as before)."""
        if rgba.dtype != np.uint8:
            rgba = rgba.astype("u1")
        if rgba.shape[2] == 3:
            a = np.full(rgba.shape[:2] + (1,), 255, dtype="u1")
            rgba = np.concatenate([rgba, a], axis=2)
        tex = self._make_texture_rgba(rgba, mipmaps=mipmaps)
        if clamp:
            tex.repeat_x = False
            tex.repeat_y = False
        self._textures[key] = tex

    def has_texture(self, key: str) -> bool:
        return key in self._textures

    def release_texture(self, key: str) -> None:
        """Free a cached texture by key (storyboard LRU eviction). No-op if
        the key is absent."""
        tex = self._textures.pop(key, None)
        if tex is not None:
            try:
                tex.release()
            except Exception:  # noqa: BLE001 - context may be tearing down
                pass

    def _make_texture_rgba(self, rgba: np.ndarray,
                           mipmaps: bool = True) -> "moderngl.Texture":
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, rgba.tobytes())
        if mipmaps:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    # --- drawing --------------------------------------------------------------

    def begin(self, clear=(0.04, 0.04, 0.06)) -> None:
        self.fbo.use()
        self.ctx.clear(*clear)

    def draw(self, sprites: list[Sprite]) -> None:
        # Painter's order for the normal (straight-alpha) sprites, then an
        # additive pass on top for glow/explosion sprites (lazer draws hit
        # explosions & catcher trails additively).
        add = []
        for sp in sprites:
            if sp.additive:
                add.append(sp)
            else:
                self._draw_one(sp)
        if add:
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            for sp in add:
                self._draw_one(sp)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def _draw_one(self, sp: Sprite) -> None:
        tex = self._textures.get(sp.texture_key) if sp.texture_key else self._white
        if tex is None:
            tex = self._white
        tex.use(location=0)
        self._u_color.value = sp.color
        self._u_center.value = (sp.x, sp.y)
        self._u_size.value = (sp.w, sp.h)
        self._u_rot.value = sp.rotation
        # UV flip (storyboard mirroring). Every gameplay/HUD sprite keeps the
        # identity default, so these never re-bind on a flag-off render and the
        # sampled UV stays exactly `in_uv` — byte-identical to before.
        if sp.uv_off != self._cur_uv_off:
            self._u_uv_off.value = sp.uv_off
            self._cur_uv_off = sp.uv_off
        if sp.uv_scale != self._cur_uv_scale:
            self._u_uv_scale.value = sp.uv_scale
            self._cur_uv_scale = sp.uv_scale
        self.vao.render(moderngl.TRIANGLE_STRIP)

    _PBO_RING = 3
    _HOST_POOL = 8             # see _pbo_host note in read_rgb_async

    def read_rgb_async(self) -> "np.ndarray | None":
        """Queue an async readback of the current fbo into a small PBO
        ring and return the OLDEST completed frame (top-left origin), or
        None while the ring is still filling. Frames come back in strict
        submission order — the render loop pushes them straight to ffmpeg,
        so the byte stream is identical to the synchronous read_rgb path,
        just ~RING-1 frames late. read_drain() flushes the tail. (Ported
        from the std renderer's proven osu_std_renderer/render/gl.py.)"""
        if self._pbos is None:
            size = self.width * self.height * 3
            self._pbos = [self.ctx.buffer(reserve=size)
                          for _ in range(self._PBO_RING)]
            # PERF: pooled CPU staging buffers — read_into() replaces
            # buf.read()'s fresh 6 MB bytes object per frame. The pool is
            # DEEPER than the PBO ring because popped frames now sit in the
            # composite stage's bounded queue (depth 3) before being
            # consumed; pool > ring + composite queue + in-process + 1 keeps
            # a queued frame from ever being overwritten (strict FIFO).
            self._pbo_host = [bytearray(size) for _ in range(self._HOST_POOL)]
            self._host_i = 0
        buf = self._pbos[self._pbo_head % len(self._pbos)]
        self.fbo.read_into(buf, components=3, alignment=1)
        self._pbo_head += 1
        if self._pbo_head - self._pbo_tail < len(self._pbos):
            return None
        return self._pop_pbo()

    def _pop_pbo(self) -> np.ndarray:
        buf = self._pbos[self._pbo_tail % len(self._pbos)]
        self._pbo_tail += 1
        host = self._pbo_host[self._host_i % len(self._pbo_host)]
        self._host_i += 1
        buf.read_into(host)
        arr = np.frombuffer(host, dtype="u1").reshape(
            (self.height, self.width, 3))
        return arr[::-1]  # same orientation contract as read_rgb (flip view)

    def read_drain(self) -> list:
        """Return every frame still in flight, oldest first (map end or
        the gameplay->outro boundary)."""
        out = []
        while self._pbos is not None and self._pbo_tail < self._pbo_head:
            out.append(self._pop_pbo())
        return out

    def read_rgb(self) -> np.ndarray:
        """Return HxWx3 uint8, top-left origin (ready for ffmpeg rgb24)."""
        data = self.fbo.read(components=3, alignment=1)
        arr = np.frombuffer(data, dtype="u1").reshape((self.height, self.width, 3))
        # moderngl reads bottom-left origin; flip to top-left
        return np.flipud(arr)

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass
