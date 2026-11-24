# sim.py raw  <name> <H> <W> <image-dir> <out-dir>
# sim.py dets <name> <H> <W> <image-dir> <out-file> <shard> <shards>
#
# Runs /out/<name>.img in the vendor's offline simulator, whose outputs are
# bit-identical to the SSC378DE's. "raw" keeps each image's NV12 and dequantized
# maps (<id>.npz) for eval.py; "dets" writes "<image id> x1 y1 x2 y2 score
# class" in input pixels -- every class at or above 0.001 in each cell,
# per-class NMS at IoU 0.7, at most 300 an image -- for operating_points.py.
# Runs inside the toolchain image.
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrator_custom
from calibrator_custom import utils
from frame import rgb

CONF, NMS_IOU, MAX_DET, BINS = 0.001, 0.7, 300, 16


def iou(a, b):
    w = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i)


def decode(outs, h):
    boxes, scores, classes = [], [], []
    logit_min = np.log(CONF / (1 - CONF))
    for box, cls in zip(outs[0::2], outs[1::2]):
        nc, gh, gw = cls.shape
        stride = h / gh
        c_idx, cell = np.nonzero(cls.reshape(nc, -1) >= logit_min)
        if not len(cell):
            continue
        d = box.reshape(4, BINS, -1)[:, :, cell]
        d = np.exp(d - d.max(1, keepdims=True))
        dist = (d / d.sum(1, keepdims=True) * np.arange(BINS)[None, :, None]).sum(1) * stride
        cx, cy = (cell % gw + 0.5) * stride, (cell // gw + 0.5) * stride
        boxes.append(np.stack([cx - dist[0], cy - dist[1], cx + dist[2], cy + dist[3]], 1))
        scores.append(1 / (1 + np.exp(-cls.reshape(nc, -1)[c_idx, cell])))
        classes.append(c_idx)
    if not boxes:
        return []
    b, s, c = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
    order = np.argsort(-s, kind="stable")
    b, s, c = b[order], s[order], c[order]
    keep = []
    for i in range(len(s)):
        if len(keep) == MAX_DET:
            break
        k = np.array(keep, int)
        same = k[c[k] == c[i]] if len(k) else k
        if len(same) and (iou(b[i], b[same]) > NMS_IOU).any():
            continue
        keep.append(i)
    return [(*b[i], s[i], c[i]) for i in keep]


class Model:
    def __init__(self, path):
        self.m = calibrator_custom.offline_simulator(path)
        self.inp = self.m.get_input_details()[0]
        self.out = self.m.get_output_details()

    def __call__(self, picture):
        x = utils.convert_to_input_formats(picture[None], self.inp)
        self.m.set_input(0, x)
        self.m.invoke()
        outs = []
        for i, o in enumerate(self.out):
            v = self.m.get_output(i)[0][..., :o["shape"][-1]]
            outs.append(v.transpose(2, 0, 1).astype(np.float32) * o["quantization"][0])
        return x, outs


mode, name, h, w, src, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
model = Model(f"/out/{name}.img")
files = sorted(glob.glob(src + "/*.jpg"))
if mode == "raw":
    os.makedirs(out, exist_ok=True)
    for p in files:
        x, outs = model(rgb(p, w, h))
        np.savez(f"{out}/{os.path.basename(p)[:-4]}.npz", nv12=x[0, :, :, 0], *outs)
else:
    shard, shards = int(sys.argv[7]), int(sys.argv[8])
    files = files[shard::shards]
    with open(out, "w") as f:
        for p in files:
            for d in decode(model(rgb(p, w, h))[1], h):
                f.write("%d %.2f %.2f %.2f %.2f %.4f %d\n" % ((int(os.path.basename(p)[:-4]),) + tuple(d)))
        f.write(f"# frames {len(files)}\n")
