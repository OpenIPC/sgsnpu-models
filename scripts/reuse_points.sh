#!/bin/bash
# reuse_points.sh <out-dir> <release-tag>
#
# The operating points of a detector built byte for byte as in <release-tag>
# are that release's: they are measured on the model's int8 output, which is
# the same. For each model in models.txt whose <out>/<name>.img has the hash
# that release's SHA256SUMS gives it, its <name>.json and its points.md
# section are copied into <out> and its name listed in <out>/points.reused;
# points.sh measures only the rest, and needs COCO only when there are any
# (<out>/points.todo). Without a tag, or a release, everything is measured.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
out=$(readlink -f "$1") tag=${2:-}
: > "$out/points.reused"
grep -v '^#' "$src/models.txt" | awk 'NF {print $1}' > "$out/points.todo"
[ -n "$tag" ] || exit 0
prev=$out/prev && rm -rf "$prev" && mkdir -p "$prev"
url=https://github.com/OpenIPC/sgsnpu-models/releases/download/$tag
curl -sfL --retry 3 -o "$prev/SHA256SUMS" "$url/SHA256SUMS" &&
	curl -sfL --retry 3 -o "$prev/points.md" "$url/points.md" || {
	echo "reuse_points.sh: no SHA256SUMS or points.md in $tag; measuring all"; exit 0; }
: > "$out/points.todo"
grep -v '^#' "$src/models.txt" | awk 'NF {print $1}' | while read -r name; do
	was=$(awk -v f="$name.img" '$2 == f {print $1}' "$prev/SHA256SUMS")
	now=$(sha256sum "$out/$name.img" | cut -d' ' -f1)
	if [ -n "$was" ] && [ "$was" = "$now" ] &&
	   curl -sfL --retry 3 -o "$out/$name.json" "$url/$name.json"; then
		# Its section of points.md, heading excluded: points.sh adds it.
		awk -v h="### $name" '$0 == h {on = 1; next} /^### / {on = 0}
			on {l[++n] = $0} END {
				a = 1; while (a <= n && l[a] == "") a++
				b = n; while (b >= a && l[b] == "") b--
				for (i = a; i <= b; i++) print l[i] }' \
			"$prev/points.md" > "$out/$name.points.md"
		[ -s "$out/$name.points.md" ] || { echo "$name: no points in $tag"; exit 1; }
		echo "$name" >> "$out/points.reused"
		echo "$name: the same model as $tag, its operating points reused"
	else
		echo "$name" >> "$out/points.todo"
		echo "$name: changed since $tag, measured again"
	fi
done
