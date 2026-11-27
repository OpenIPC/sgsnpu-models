# lpr_cal.py <coco-dir> <font.ttf> <out> <set-dir> <n> <seed>
#
# Inputs for the three licence-plate models, made from synthetic plates
# (lpr.py) pasted into COCO pictures: <set-dir>/spot (416x256 RGB, what the
# spotter sees after the IPU's NV12 conversion), <set-dir>/stn (70x130 BT.601
# YUV crops) and <set-dir>/rec (128x64, rectified by the FP32 STN), uint8
# .npy, and <set-dir>/plates.txt. The same seed gives the same set.
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lpr  # noqa: E402

coco, font, out, dst, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
stn = ort.InferenceSession(f"{out}/onnx/lpr_stn.onnx", providers=["CPUExecutionProvider"])
G = lpr.tps_matrix()
pics = sorted(glob.glob(f"{coco}/*.jpg"))
rng = np.random.default_rng(seed)
for d in ("spot", "stn", "rec"):
    os.makedirs(f"{dst}/{d}", exist_ok=True)
texts = []
for k in range(n):
    bg = np.asarray(Image.open(pics[k % len(pics)]).convert("RGB"))
    img, quad, text = lpr.scene(rng, bg, font)
    texts.append(text)
    np.save(f"{dst}/spot/{k:05d}.npy", lpr.window(img, quad, rng))
    y, uv = lpr.rgb_to_nv12(img)
    crop = lpr.crop601(y, uv, quad)
    np.save(f"{dst}/stn/{k:05d}.npy", crop)
    raw = stn.run(None, {"image": ((crop.astype(np.float32) - 128) / 128).transpose(2, 0, 1)[None]})[0]
    np.save(f"{dst}/rec/{k:05d}.npy", lpr.rectify(crop, G, raw.reshape(-1)))
open(f"{dst}/plates.txt", "w").write("\n".join(texts) + "\n")
print(f"{n} synthetic plates in {dst}")
