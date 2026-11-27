# The licence-plate networks' arithmetic outside the IPU, in numpy: what
# majestic's src/detect/lpr.c does on the camera, for calibrating the models
# and checking them here. Python 3.11 (the export side), numpy, opencv, PIL.
#
# The pipeline: an NV12 picture (BT.709 full range, as the SSC37x scaler emits
# it) -> lpr_spotter (plates as oriented boxes) -> the plate's bounding box
# cut out at 130x70 and turned from BT.709 to BT.601 -> lpr_stn (20 thin-plate
# spline control points) -> the crop resampled to 128x64 -> lpr_rec (features
# and their attention projection) -> lpr_dec.bin's attention decoder -> text.
import struct

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

M601 = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
M709 = np.array([[0.2126, 0.7152, 0.0722], [-0.1146, -0.3854, 0.5], [0.5, -0.4542, -0.0458]])
M709_TO_601 = M601 @ np.linalg.inv(M709)
K, CLAMP = 20, 0.3
GLYPHS = list("0123456789ABCEHIKMOPTXY") + list("БГДЖЗИЛПФЦЧШЩЭЮЯ")


# --- pictures --------------------------------------------------------------

def rgb_to_nv12(rgb):
    h, w = rgb.shape[0] & ~1, rgb.shape[1] & ~1
    yuv = rgb[:h, :w].astype(np.float64) @ M709.T + [0, 128, 128]
    y = np.clip(np.rint(yuv[..., 0]), 0, 255).astype(np.uint8)
    uv = yuv[..., 1:].reshape(h // 2, 2, w // 2, 2, 2).mean(axis=(1, 3))
    return y, np.clip(np.rint(uv), 0, 255).astype(np.uint8)


def resize_bilinear(src, x0, y0, x1, y1, ow, oh):
    """[x0,x1) x [y0,y1) of `src`, stretched, half-pixel centres, clamped."""
    h, w = src.shape[:2]
    xs = np.clip(x0 + (np.arange(ow) + 0.5) * (x1 - x0) / ow - 0.5, 0, w - 1)
    ys = np.clip(y0 + (np.arange(oh) + 0.5) * (y1 - y0) / oh - 0.5, 0, h - 1)
    xa = np.floor(xs).astype(int); xb = np.minimum(xa + 1, w - 1); wx = (xs - xa)[None, :, None]
    ya = np.floor(ys).astype(int); yb = np.minimum(ya + 1, h - 1); wy = (ys - ya)[:, None, None]
    s = src.astype(np.float64)
    if s.ndim == 2:
        s = s[..., None]
    top = s[ya][:, xa] * (1 - wx) + s[ya][:, xb] * wx
    bot = s[yb][:, xa] * (1 - wx) + s[yb][:, xb] * wx
    return top * (1 - wy) + bot * wy


def crop601(y, uv, quad, pad=0.08):
    """The plate's bounding box, grown by `pad`, at 130x70 BT.601 YUV."""
    h, w = y.shape
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    cy = resize_bilinear(y, x0, y0, x1, y1, 130, 70)[..., 0]
    cuv = resize_bilinear(uv, x0 / 2, y0 / 2, x1 / 2, y1 / 2, 130, 70)
    yuv = np.concatenate([cy[..., None], cuv], -1) - [0, 128, 128]
    return np.clip(np.rint(yuv @ M709_TO_601.T + [0, 128, 128]), 0, 255).astype(np.uint8)


def half(y, uv):
    """2x2 averages, as the camera halves its 832x512 look for the spotter."""
    q = lambda p: ((p[0::2, 0::2].astype(int) + p[1::2, 0::2] + p[0::2, 1::2]  # noqa: E731
                    + p[1::2, 1::2] + 2) >> 2).astype(np.uint8)
    return q(y), q(uv)


def nv12_to_rgb(y, uv):
    """What the IPU's NV12 input makes of a picture: BT.709 full range RGB."""
    up = lambda p: np.repeat(np.repeat(p, 2, 0), 2, 1).astype(np.float64) - 128  # noqa: E731
    yuv = np.stack([y.astype(np.float64), up(uv[..., 0]), up(uv[..., 1])], -1)
    return np.clip(yuv @ np.linalg.inv(M709).T, 0, 255)


# --- the thin-plate spline -------------------------------------------------

def _ctrl():
    x = np.linspace(-1, 1, K // 2)
    return np.concatenate([np.stack([x, -np.ones_like(x)], 1), np.stack([x, np.ones_like(x)], 1)])


def tps_matrix(oh=64, ow=128):
    c = _ctrl()
    u = lambda d2: d2 * np.log(d2 + 1e-12)  # noqa: E731
    L = np.zeros((K + 3, K + 3))
    L[:K, :K] = u(((c[:, None] - c[None]) ** 2).sum(-1))
    T = np.concatenate([np.ones((K, 1)), c], 1)
    L[:K, K:], L[K:, :K] = T, T.T
    gy, gx = np.meshgrid(np.linspace(-1, 1, oh), np.linspace(-1, 1, ow), indexing="ij")
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    UT = np.concatenate([u(((p[:, None] - c[None]) ** 2).sum(-1)), np.ones((len(p), 1)), p], 1)
    return (UT @ np.linalg.inv(L))[:, :K].astype(np.float32)


def rectify(crop, G, raw):
    """The STN's 40 raw outputs to the 128x64 crop it asks for (uint8, the
    camera's grid sampling: outside reads as 128, truncated)."""
    grid = G @ (_ctrl() + np.tanh(raw.reshape(K, 2)) * CLAMP)
    H, W = crop.shape[:2]
    ix = ((grid[:, 0] + 1) * W - 1) * 0.5
    iy = ((grid[:, 1] + 1) * H - 1) * 0.5
    x0, y0 = np.floor(ix).astype(int), np.floor(iy).astype(int)
    fx, fy = (ix - x0).astype(np.float32), (iy - y0).astype(np.float32)
    n = (crop.astype(np.float32) - 128) * 0.0078125
    acc = np.zeros((len(grid), 3), np.float32)
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + dx, y0 + dy
        ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        v = np.zeros((len(grid), 3), np.float32)
        v[ok] = n[yy[ok], xx[ok]]
        acc += wt[:, None] * v
    return np.clip(acc * 128 + 128, 0, 255).astype(np.uint8).reshape(64, 128, 3)


# --- the decoder -----------------------------------------------------------

class Decoder:
    """lpr_dec.bin: the attention decoder, run as src/detect/lpr.c runs it."""

    def __init__(self, path):
        b = open(path, "rb").read()
        assert b[:4] == b"LPRD"
        _, V, H, F, self.L = struct.unpack("<5I", b[4:24])
        a = np.frombuffer(b, "<f4", offset=24)
        sizes = [(V, H), (H, H), (H,), (4 * H, H + F), (4 * H, H), (4 * H,), (V, H + F), (V,)]
        parts, o = [], 0
        for s in sizes:
            n = int(np.prod(s))
            parts.append(a[o:o + n].reshape(s))
            o += n
        assert o == a.size
        self.embed, self.attn_h, self.attn_v, self.w_ih, self.w_hh, self.b, self.out_w, self.out_b = parts
        self.H = H

    def __call__(self, feat, proj):
        """feat [256,256], proj [256,128] -> text."""
        H, sig = self.H, (lambda x: 1 / (1 + np.exp(-x)))
        h = np.zeros(H, np.float32)
        c = np.zeros(H, np.float32)
        tok, text = 2, ""
        for _ in range(self.L):
            s = np.tanh(proj + self.attn_h @ h) @ self.attn_v
            s = np.exp(s - s.max())
            ctx = (s / s.sum()) @ feat
            g = self.w_ih @ np.concatenate([self.embed[tok], ctx]) + self.w_hh @ h + self.b
            c = sig(g[H:2 * H]) * c + sig(g[:H]) * np.tanh(g[2 * H:3 * H])
            h = sig(g[3 * H:]) * np.tanh(c)
            tok = int((self.out_w @ np.concatenate([h, ctx]) + self.out_b).argmax())
            if tok == 1:
                break
            if tok >= 3:
                text += GLYPHS[tok - 3]
        return text


def spot_best(o8, o16, thr=0.4):
    """The spotter's most certain plate as a quad (TL, TR, BR, BL), or None.
    Maps are [8, h, w]: objectness, dx, dy, log w, log h, sin 2a, cos 2a."""
    best = None
    for m, st in ((o8, 8), (o16, 16)):
        i = int(np.argmax(m[0]))
        gy, gx = divmod(i, m.shape[2])
        p = 1 / (1 + np.exp(-m[0].flat[i]))
        if p >= thr and (best is None or p > best[0]):
            best = (p, (gx + 0.5 + m[1].flat[i]) * st, (gy + 0.5 + m[2].flat[i]) * st,
                    np.exp(m[3].flat[i]) * st, np.exp(m[4].flat[i]) * st,
                    np.arctan2(m[5].flat[i], m[6].flat[i]) / 2)
    if best is None:
        return None
    _, cx, cy, w, h, a = best
    c, s = np.cos(a), np.sin(a)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return np.array([[c * x - s * y + cx, s * x + c * y + cy] for x, y in pts], np.float32)


# --- synthetic plates ------------------------------------------------------
#
# The models calibrate on plates rendered here and pasted into COCO pictures,
# not on photographs of real ones: nothing here is anybody's licence plate.

RU = "ABEKMHOPCTYX"


def _plate_one_line(rng, font):
    L = lambda: RU[rng.integers(len(RU))]  # noqa: E731
    D = lambda: str(rng.integers(10))  # noqa: E731
    reg = D() + D() + (D() if rng.random() < 0.4 else "")
    parts = (L(), D() + D() + D(), L() + L())
    W, H, split = 520, 112, 392
    img = Image.new("RGB", (W, H), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, W - 3, H - 3], outline=(15, 15, 15), width=5)
    d.line([split, 4, split, H - 5], fill=(15, 15, 15), width=4)
    big, small = ImageFont.truetype(font, 92), ImageFont.truetype(font, 70)
    rf = ImageFont.truetype(font, 62 if len(reg) == 2 else 52)
    x = 22
    for p, f, dy in zip(parts, (small, big, small), (22, 2, 22)):
        d.text((x, dy), p, font=f, fill=(10, 10, 10))
        x += d.textlength(p, font=f) + 8
    d.text((split + (W - split - d.textlength(reg, font=rf)) / 2, 2), reg, font=rf, fill=(10, 10, 10))
    d.text((split + 14, 82), "RUS", font=ImageFont.truetype(font, 18), fill=(10, 10, 10))
    for k, col in enumerate([(250, 250, 250), (30, 60, 180), (200, 30, 30)]):
        d.rectangle([split + 64, 84 + k * 6, split + 94, 89 + k * 6], fill=col)
    return img, "".join(parts) + reg


def _plate_two_line(rng, font):
    D = lambda: str(rng.integers(10))  # noqa: E731
    L = lambda: RU[rng.integers(len(RU))]  # noqa: E731
    top, bot, reg = D() + D() + D() + D(), L() + L(), D() + D()
    W, H = 290, 170
    img = Image.new("RGB", (W, H), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, W - 3, H - 3], outline=(15, 15, 15), width=5)
    f1, f2 = ImageFont.truetype(font, 84), ImageFont.truetype(font, 70)
    d.text(((W - d.textlength(top, font=f1)) / 2, 0), top, font=f1, fill=(10, 10, 10))
    d.text((24, 86), bot, font=f2, fill=(10, 10, 10))
    d.text((170, 86), reg, font=f2, fill=(10, 10, 10))
    return img, top + bot + reg


def _degrade(img, rng):
    a = np.asarray(img).astype(np.float32) * rng.uniform(0.45, 1.05) + rng.uniform(-25, 25)
    a += rng.normal(0, rng.uniform(1, 8), a.shape)
    img = Image.fromarray(np.clip(a * rng.uniform(0.85, 1.1, 3), 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.8))) if rng.random() < 0.6 else img


def scene(rng, bg, font):
    """A picture with one synthetic plate on it: picture, quad, text."""
    img = bg.copy()
    H, W = img.shape[:2]
    plate, text = (_plate_two_line if rng.random() < 0.2 else _plate_one_line)(rng, font)
    plate = _degrade(plate, rng)
    pw = W * rng.uniform(0.10, 0.30)
    ph = pw * plate.size[1] / plate.size[0]
    cx, cy = rng.uniform(pw, W - pw), rng.uniform(H * 0.35 + ph, H - ph)
    a, k = np.deg2rad(rng.uniform(-10, 10)), rng.uniform(-0.08, 0.08)
    pts = np.array([[-pw / 2, -ph / 2 * (1 - k)], [pw / 2, -ph / 2 * (1 + k)],
                    [pw / 2, ph / 2 * (1 + k)], [-pw / 2, ph / 2 * (1 - k)]])
    quad = np.stack([pts[:, 0] * np.cos(a) - pts[:, 1] * np.sin(a) + cx,
                     pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a) + cy], 1).astype(np.float32)
    pw_, ph_ = plate.size
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw_, 0], [pw_, ph_], [0, ph_]]), quad)
    warped = cv2.warpPerspective(np.asarray(plate), M, (W, H), flags=cv2.INTER_AREA)
    m = cv2.warpPerspective(np.full((ph_, pw_), 255, np.uint8), M, (W, H)).astype(np.float32)[..., None] / 255
    img = np.clip(img * (1 - m) + warped * m, 0, 255).astype(np.uint8)
    return img, quad, text


def window(rgb, quad, rng, w=416, h=256):
    """A w x h (13:8) view of a vehicle-sized region around the plate, as the
    camera's look at a vehicle frames it (the plate 12-35 % of the width),
    and where the plate is in it."""
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    ww = max(x1 - x0, 8) / rng.uniform(0.12, 0.35)
    wh = ww * h / w
    cx = (x0 + x1) / 2 + rng.uniform(-0.25, 0.25) * ww
    cy = (y0 + y1) / 2 - rng.uniform(0.0, 0.3) * wh
    H, W = rgb.shape[:2]
    pad = np.full((H + 2 * int(wh), W + 2 * int(ww), 3), 114, np.uint8)
    pad[int(wh):int(wh) + H, int(ww):int(ww) + W] = rgb
    ox, oy = int(ww), int(wh)
    out = resize_bilinear(pad, cx - ww / 2 + ox, cy - wh / 2 + oy, cx + ww / 2 + ox, cy + wh / 2 + oy, w, h)
    at = ((quad - [cx - ww / 2, cy - wh / 2]) * [w / ww, h / wh]).astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), at
