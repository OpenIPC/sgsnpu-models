#!/bin/bash
# lpr.sh <out> <cache>: the three licence-plate models majestic reads plates
# with (lpr_spotter, lpr_stn, lpr_rec) and the decoder it runs on the CPU
# (lpr_dec.bin), from lpr-onnx in sources.txt; called by build.sh.
#
# They are calibrated on synthetic plates pasted into COCO128 pictures
# (lpr_cal.py), not on photographs of real ones, and checked against FP32 on a
# second synthetic set (lpr_eval.py -> lpr_eval.md, which fails the build).
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
out=$1 cache=$2
tar -xzf "$cache/lpr-onnx-v1.tar.gz" -C "$out/onnx"
mv -f "$out/onnx/lpr_dec.bin" "$out/lpr_dec.bin"
unzip -qo -j -d "$cache" "$cache/dejavu-fonts-ttf-2.37.zip" \
	'dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed-Bold.ttf'
font=$cache/DejaVuSansCondensed-Bold.ttf coco=$cache/coco128/images/train2017
rm -rf "$out/lpr" "$out/raw/lpr"
python3 "$src/scripts/lpr_cal.py" "$coco" "$font" "$out" "$out/lpr/cal" 300 1
python3 "$src/scripts/lpr_cal.py" "$coco" "$font" "$out" "$out/lpr/eval" 300 2
for m in lpr_stn lpr_spotter lpr_rec; do
	bash "$src/scripts/in-sdk.sh" "$out" bash /src/scripts/convert_lpr.sh "$m" < /dev/null
done
sim() { bash "$src/scripts/in-sdk.sh" "$out" python3 -W ignore /src/scripts/sim_lpr.py "$@" < /dev/null; }
sim lpr_spotter /out/lpr/eval/spot /out/raw/lpr/spot
sim lpr_stn /out/lpr/eval/stn /out/raw/lpr/stn
python3 "$src/scripts/lpr_eval.py" rects "$out"
sim lpr_rec /out/lpr/eval/rec_q /out/raw/lpr/rec
python3 "$src/scripts/lpr_eval.py" report "$out"
