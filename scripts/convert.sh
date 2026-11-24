#!/bin/bash
# convert.sh <name> <H> <W>: runs inside the toolchain image, SDK at /sdk,
# <out> at /out. <out>/onnx/<name>.onnx -> <out>/<name>.img.
#
# The network is trained on RGB 0..1 and takes the camera's NV12 as it is:
# YUV_NV12 puts the conversion (BT.709 full range, what the SSC37x scaler
# emits) in the model. Outputs stay int16 with a scale and no offset
# (dequantizations=FALSE), which spans the maps' whole range, so a class map
# needs no widening to score above 0.5.
set -euo pipefail
name=$1 h=$2 w=$3
work=/out/work/$name && rm -rf "$work" && mkdir -p "$work" && cd "$work"
cat > pre.py <<P
import sys
sys.path.insert(0, "/src/scripts")
from frame import rgb
import numpy as np
def image_preprocess(path, norm=True):
    im = rgb(path, $w, $h)
    return np.expand_dims(im.astype("float32") / 255 if norm else im, 0)
P
cat > input.ini <<I
[INPUT_CONFIG]
inputs='images';
training_input_formats=RGB;
input_formats=YUV_NV12;
quantizations=TRUE;
mean=0:0:0;
std_value=255:255:255;
[OUTPUT_CONFIG]
outputs='p0b,p0c,p1b,p1c,p2b,p2c';
dequantizations=FALSE,FALSE,FALSE,FALSE,FALSE,FALSE;
[CONV_CONFIG]
I
# The converter writes beside its input.
cp /out/onnx/$name.onnx .
cd /sdk && . ./cfg_env.sh && cd "$work"
python3 -W ignore /sdk/Scripts/ConvertTool/SGS_converter.py onnx \
	--model_file $name.onnx --input_shapes 1,3,$h,$w --input_config input.ini \
	--input_arrays images --output_arrays p0b,p0c,p1b,p1c,p2b,p2c \
	-i /out/images/cal -n pre.py --output_file $name.img > log.txt 2>&1 ||
	{ tail -30 log.txt; exit 1; }
cp $name.img /out/$name.img
