#!/bin/bash
# reuse_points.sh <out-dir> <release-tag>
#
# The operating points of a detector that <release-tag> measured on the same
# int8 model, the same way, are that release's. A model is reused only when
#   - its built <out>/<name>.img has the hash <release-tag>'s SHA256SUMS
#     gives it,
#   - its models.txt line (what its channels mean) is the one at the tag,
#   - what measures and scores the points (POINTS_INPUTS) is as at the tag,
#   - and the release's <name>.json has the hash its SHA256SUMS gives it.
# Then its <name>.json and its points.md section are copied into <out> and
# its name listed in <out>/points.reused; any other model, or any doubt, is
# listed in <out>/points.todo for points.sh to measure, which needs COCO only
# when there are any. Without a tag, or a release, everything is measured.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
out=$(readlink -f "$1") tag=${2:-}
# What decides the numbers: the simulation of each frame, the scoring, and
# the images and labels they are scored on. (points.sh only runs them.)
POINTS_INPUTS="scripts/sim.py scripts/frame.py scripts/operating_points.py coco.txt"
models() { grep -v '^#' "$src/models.txt" | awk 'NF {print $1}'; }
: > "$out/points.reused"
models > "$out/points.todo"
[ -n "$tag" ] || exit 0
# The tag's tree, to see what changed since: the checkout is shallow.
git -C "$src" fetch -q --depth 1 origin "refs/tags/$tag:refs/tags/$tag" 2>/dev/null || true
# shellcheck disable=SC2086
if ! git -C "$src" diff --quiet "refs/tags/$tag" HEAD -- $POINTS_INPUTS 2>/dev/null; then
	echo "reuse_points.sh: how points are measured changed since $tag (or it is unknown); measuring all"
	exit 0
fi
prev=$out/prev
mkdir -p "$prev"
url=https://github.com/OpenIPC/sgsnpu-models/releases/download/$tag
curl -sfL --retry 3 -o "$prev/SHA256SUMS" "$url/SHA256SUMS" &&
	curl -sfL --retry 3 -o "$prev/points.md" "$url/points.md" || {
	echo "reuse_points.sh: no SHA256SUMS or points.md in $tag; measuring all"; exit 0; }
git -C "$src" show "refs/tags/$tag:models.txt" > "$prev/models.txt" 2>/dev/null || : > "$prev/models.txt"
sum() { awk -v f="$2" '$2 == f {print $1}' "$1"; }
line() { grep -v '^#' "$1" | awk -v n="$2" '$1 == n'; }
: > "$out/points.todo"
models | while read -r name; do
	why=
	if [ "$(sum "$prev/SHA256SUMS" "$name.img")" != "$(sha256sum "$out/$name.img" | cut -d' ' -f1)" ]; then
		why="the model is not the one in $tag"
	elif [ "$(line "$prev/models.txt" "$name")" != "$(line "$src/models.txt" "$name")" ]; then
		why="its models.txt line changed since $tag"
	elif ! curl -sfL --retry 3 -o "$out/$name.json" "$url/$name.json" ||
	     [ "$(sum "$prev/SHA256SUMS" "$name.json")" != "$(sha256sum "$out/$name.json" | cut -d' ' -f1)" ]; then
		why="$tag's $name.json is missing or not what its SHA256SUMS says"
	else
		# Its section of points.md, heading and surrounding blanks excluded:
		# points.sh adds them.
		awk -v h="### $name" '$0 == h {on = 1; next} /^### / {on = 0}
			on {l[++n] = $0} END {
				a = 1; while (a <= n && l[a] == "") a++
				b = n; while (b >= a && l[b] == "") b--
				for (i = a; i <= b; i++) print l[i] }' \
			"$prev/points.md" > "$out/$name.points.md"
		[ -s "$out/$name.points.md" ] || why="$tag's points.md has no section for it"
	fi
	if [ -z "$why" ]; then
		echo "$name" >> "$out/points.reused"
		echo "$name: as in $tag, its operating points reused"
	else
		: > "$out/$name.points.md"
		[ ! -e "$out/$name.json" ] || mv -f "$out/$name.json" "$prev/$name.json.rejected"
		echo "$name" >> "$out/points.todo"
		echo "$name: measured again: $why"
	fi
done
