# lpr_eval.py rects <out>: the rectified crops the int8 STN asks for, from its
#                          simulator outputs, for the recognizer's simulation
# lpr_eval.py report <out>: the int8 plate pipeline against FP32 on the eval
#                          set; writes <out>/lpr_eval.md, non-zero on a failure
#
# The int8 side is the vendor's offline simulator (bit-identical to the
# SSC378DE) through the same steps the camera takes; the FP32 side is
# onnxruntime on the same inputs. The gate, on synthetic plates the FP32
# models were never trained on: the spotter finds the plate where it was
# pasted (IoU >= 0.5) on at least FIND of them in FP32, and int8 finds it on
# at least PASS of those; int8 reads the same text on at least READ of the
# ones FP32 reads right (on the rest FP32 is unsure, and an unsure read
# flipping is not quantization error); and the recognizer's features keep a
# mean cosine similarity of at least COS.
import glob
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lpr  # noqa: E402

FIND, PASS, READ, COS = 0.60, 0.95, 0.90, 0.98
# The camera takes plates from 0.40; synthetic ones score lower in FP32 too
# (a font and finish the spotter was not trained on), so here any plate the
# spotter rates at all is compared.
SPOT_SCORE = 0.10
mode, out = sys.argv[1], sys.argv[2]
ev, raw = f"{out}/lpr/eval", f"{out}/raw/lpr"
G = lpr.tps_matrix()
sess = lambda n: ort.InferenceSession(f"{out}/onnx/{n}.onnx", providers=["CPUExecutionProvider"])  # noqa: E731
norm = lambda u8: ((u8.astype(np.float32) - 128) / 128).transpose(2, 0, 1)[None]  # noqa: E731
ids = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{ev}/stn/*.npy"))]

if mode == "rects":
    os.makedirs(f"{ev}/rec_q", exist_ok=True)
    for i in ids:
        crop = np.load(f"{ev}/stn/{i}.npy")
        np.save(f"{ev}/rec_q/{i}.npy", lpr.rectify(crop, G, np.load(f"{raw}/stn/{i}.npy")))
    sys.exit(0)


def box_iou(a, b):
    ax0, ay0 = a.min(0); ax1, ay1 = a.max(0)
    bx0, by0 = b.min(0); bx1, by1 = b.max(0)
    w, h = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - i)


spot, stn, rec = sess("lpr_spotter"), sess("lpr_stn"), sess("lpr_rec")
dec = lpr.Decoder(f"{out}/lpr_dec.bin")
truth = open(f"{ev}/plates.txt").read().split()
where = np.load(f"{ev}/quads.npy")
fp_found = both_found = same_text = fp_ok = q_ok = kept = 0
cos = []
for k, i in enumerate(ids):
    q = np.load(f"{raw}/spot/{i}.npz")
    x = lpr.nv12_to_rgb(*lpr.rgb_to_nv12(np.load(f"{ev}/spot/{i}.npy"))).astype(np.float32)
    o8, o16 = spot.run(None, {"image": x.transpose(2, 0, 1)[None]})
    a = lpr.spot_best(o8[0], o16[0], SPOT_SCORE)
    b = lpr.spot_best(q["obb_s8"], q["obb_s16"], SPOT_SCORE)
    hit = lambda d: d is not None and box_iou(d, where[k]) >= 0.5  # noqa: E731
    fp_found += hit(a)
    both_found += hit(a) and hit(b)

    crop = np.load(f"{ev}/stn/{i}.npy")
    r = stn.run(None, {"image": norm(crop)})[0].reshape(-1)
    feat, proj = rec.run(None, {"image": norm(lpr.rectify(crop, G, r))})
    t_fp = dec(feat[0, :, 0, :].T, proj[0, :, 0, :].T)
    z = np.load(f"{raw}/rec/{i}.npz")
    t_q = dec(z["feat"], z["proj"])
    same_text += t_fp == t_q
    fp_ok += t_fp == truth[k]
    q_ok += t_q == truth[k]
    kept += t_fp == truth[k] and t_q == t_fp
    f = feat[0, :, 0, :].T.ravel()
    g = z["feat"].ravel()
    cos.append(float(f @ g / (np.linalg.norm(f) * np.linalg.norm(g) + 1e-12)))
n = len(ids)
rows = ["| check | FP32 against int8 (offline simulator) |", "|---|---|",
        f"| spotter: the pasted plate found, FP32 | {fp_found}/{n} |",
        f"| spotter: int8 finds it too | {both_found}/{fp_found} |",
        f"| recognizer: the same text where FP32 reads it right | {kept}/{fp_ok} |",
        f"| recognizer: features, mean cosine similarity | {np.mean(cos):.4f} |",
        f"| recognizer: the same text, all | {same_text}/{n} |",
        f"| synthetic plates read right, FP32 / int8 | {fp_ok}/{n} / {q_ok}/{n} |"]
ok = (fp_found >= FIND * n and both_found >= PASS * fp_found and fp_ok
      and kept >= READ * fp_ok and np.mean(cos) >= COS)
open(f"{out}/lpr_eval.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
sys.exit(0 if ok else 1)
