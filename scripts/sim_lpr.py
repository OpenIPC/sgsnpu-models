# sim_lpr.py <model> <in-dir> <out-dir>: /out/<model>.img in the vendor's
# offline simulator (bit-identical to the SSC378DE) on every uint8 .npy in
# <in-dir>, dequantized. Runs inside the toolchain image.
#   lpr_spotter -> <id>.npz obb_s8 [8,32,52], obb_s16 [8,16,26]
#   lpr_stn     -> <id>.npy the 40 raw control-point outputs
#   lpr_rec     -> <id>.npz feat [256,256], proj [256,128]
import glob
import os
import sys

import numpy as np
import calibrator_custom
from calibrator_custom import utils

name, src, out = sys.argv[1:4]
m = calibrator_custom.offline_simulator("/out/%s.img" % name)
inp, outs = m.get_input_details()[0], m.get_output_details()
os.makedirs(out, exist_ok=True)
for p in sorted(glob.glob(src + "/*.npy")):
    m.set_input(0, utils.convert_to_input_formats(np.load(p)[None], inp))
    m.invoke()
    res = [m.get_output(i)[..., :o["shape"][-1]].astype(np.float32) * o["quantization"][0]
           for i, o in enumerate(outs)]
    base = os.path.join(out, os.path.basename(p)[:-4])
    if name == "lpr_stn":
        np.save(base + ".npy", res[0].reshape(-1))
    elif name == "lpr_rec":
        np.savez(base + ".npz", feat=res[0].reshape(256, 256), proj=res[1].reshape(256, 128))
    else:
        np.savez(base + ".npz", obb_s8=res[0][0].transpose(2, 0, 1), obb_s16=res[1][0].transpose(2, 0, 1))
