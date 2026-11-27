#!/bin/bash
# build.sh <toolchain.zip> [out-dir]
#
# The whole build, the same here and in .github/workflows/release.yml:
#   1. download the pinned weights and images (sources.txt), check their hashes
#   2. export each model in models.txt to head-less ONNX (export.py, this
#      host's Python with requirements-export.txt installed)
#   3. convert them with the vendor toolchain in its container (convert.sh)
#   4. the licence-plate models, calibrated on synthetic plates and checked
#      against FP32 (lpr.sh)
#   5. check the int8 models against FP32 (eval.py) and write SHA256SUMS
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
zip=$(readlink -f "$1") out=${2:-$src/out}
mkdir -p "$out/onnx" "$out/images/cal" "$out/images/eval" && out=$(readlink -f "$out")
cache=$src/.cache

bash "$src/scripts/fetch.sh" "$src/sources.txt" "$cache"
bash "$src/scripts/sdk.sh" "$zip" "$cache/sdk"
[ -d "$cache/coco128" ] || unzip -qo -d "$cache" "$cache/coco128.zip"
grep -v '^#' "$src/images.txt" | while read -r set id _; do
	[ -n "$set" ] && cp "$cache/coco128/images/train2017/$id.jpg" "$out/images/$set/"
done
docker build -q -t sgsnpu-s1 "$src/docker"

grep -v '^#' "$src/models.txt" | while read -r name weights h w keep groups; do
	[ -n "$name" ] || continue
	python3 "$src/scripts/export.py" "$cache/$weights" "$h" "$w" "$keep" "$out/onnx/$name.onnx"
	bash "$src/scripts/in-sdk.sh" "$out" bash /src/scripts/convert.sh "$name" "$h" "$w" < /dev/null
	rm -rf "$out/raw/$name"
	bash "$src/scripts/in-sdk.sh" "$out" python3 -W ignore /src/scripts/sim.py raw "$name" \
		"$h" "$w" /out/images/eval "/out/raw/$name" < /dev/null
done

bash "$src/scripts/lpr.sh" "$out" "$cache"

python3 "$src/scripts/eval.py" "$out"
cd "$out" && sha256sum ./*.img ./lpr_dec.bin | sed 's# \./# #' > SHA256SUMS && cat SHA256SUMS
