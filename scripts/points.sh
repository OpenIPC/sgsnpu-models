#!/bin/bash
# points.sh <out-dir> <val2017 images> <instances_val2017.json> [jobs]
#
# The presets' operating points: each model's int8 detections on all of COCO
# val2017 from the offline simulator, scored by operating_points.py. Writes
# <out>/<name>.json, which majestic reads beside the model, and points.md.
#
# A model reuse_points.sh found unchanged since the last release (listed in
# <out>/points.reused) keeps that release's points and is not simulated; with
# every model reused, COCO is not read at all.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
out=$(readlink -f "$1") jobs=${4:-$(nproc)}
reused() { [ -f "$out/points.reused" ] && grep -qx "$1" "$out/points.reused"; }
mkdir -p "$out/dets"; : > "$out/points.md"
grep -v '^#' "$src/models.txt" | while read -r name _ h w _ groups; do
	[ -n "$name" ] || continue
	if reused "$name"; then
		{ echo "### $name"; echo; cat "$out/$name.points.md"; echo; } >> "$out/points.md"
		continue
	fi
	inst=$(readlink -f "$3")
	IMAGES=$(readlink -f "$2"); export IMAGES
	rm -f "$out/dets/$name".*
	seq 0 $((jobs - 1)) | xargs -P "$jobs" -I{} bash "$src/scripts/in-sdk.sh" "$out" \
		python3 -W ignore /src/scripts/sim.py dets "$name" "$h" "$w" /images \
		"/out/dets/$name.{}" {} "$jobs" > "$out/dets/$name.log" 2>&1
	cat "$out/dets/$name".[0-9]* > "$out/dets/$name.txt"
	# Every shard says how many frames it ran; a short total is a shard that
	# did not, which would skew every threshold.
	n=$(awk '$1 == "#" && $2 == "frames" { t += $3 } END { print t + 0 }' "$out/dets/$name.txt")
	[ "$n" -eq 5000 ] || { echo "$name: $n of 5000 frames simulated"; exit 1; }
	python3 "$src/scripts/operating_points.py" "$inst" "$out/dets/$name.txt" \
		"$out/$name.json" "$out/$name.points.md" "$w" "$h" "$groups"
	{ echo "### $name"; echo; cat "$out/$name.points.md"; echo; } >> "$out/points.md"
done
cd "$out" && shopt -s nullglob && sha256sum ./*.img ./*.bin ./*.json | sed 's# \./# #' > SHA256SUMS && cat SHA256SUMS
