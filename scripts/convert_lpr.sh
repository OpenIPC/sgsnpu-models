#!/bin/bash
# convert_lpr.sh <model>: runs inside the toolchain image, SDK at /sdk, <out>
# at /out. <out>/onnx/<model>.onnx -> <out>/<model>.img, calibrated on
# <out>/lpr/cal (lpr_cal.py). Outputs stay int16 with a scale.
#   lpr_spotter  NV12 in, BT.709 full range as the scaler emits it; the graph
#                turns the RGB the IPU makes of it into the BT.601 YUV the
#                spotter was trained on (its first 1x1 convolution), so the
#                network sees 0..255 (mean 0, std 1)
#   lpr_stn      packed 3-channel BT.601 YUV crop, (x - 128) / 128
#   lpr_rec      packed 3-channel BT.601 YUV, rectified, (x - 128) / 128
set -euo pipefail
name=$1
case $name in
lpr_spotter) shape=1,3,256,416 fmt=YUV_NV12 mean=0:0:0 std=1:1:1 outs=obb_s8,obb_s16 deq=FALSE,FALSE cal=spot sub=0 div=1 ;;
lpr_stn)     shape=1,3,70,130  fmt=RGB mean=128:128:128 std=128:128:128 outs=ctrl_raw deq=FALSE cal=stn sub=128 div=128 ;;
lpr_rec)     shape=1,3,64,128  fmt=RGB mean=128:128:128 std=128:128:128 outs=feat,proj deq=FALSE,FALSE cal=rec sub=128 div=128 ;;
*) echo "convert_lpr.sh: unknown model $name" >&2; exit 2 ;;
esac
work=/out/work/$name && rm -rf "$work" && mkdir -p "$work" && cd "$work"
cat > pre.py <<P
import numpy as np
def image_preprocess(path, norm=True):
    im = np.load(path)
    if norm:
        im = (im.astype("float32") - $sub) / $div
    return np.expand_dims(im, 0)
P
cat > input.ini <<I
[INPUT_CONFIG]
inputs='image';
training_input_formats=RGB;
input_formats=$fmt;
quantizations=TRUE;
mean=$mean;
std_value=$std;
[OUTPUT_CONFIG]
outputs='$outs';
dequantizations=$deq;
[CONV_CONFIG]
I
cp /out/onnx/$name.onnx .
set +u
cd /sdk && . ./cfg_env.sh && cd "$work"
set -u
python3 -W ignore /sdk/Scripts/ConvertTool/SGS_converter.py onnx \
	--model_file $name.onnx --input_shapes $shape --input_config input.ini \
	--input_arrays image --output_arrays $outs \
	-i /out/lpr/cal/$cal -n pre.py --output_file $name.img > log.txt 2>&1 ||
	{ tail -40 log.txt; exit 1; }
cp $name.img /out/$name.img
